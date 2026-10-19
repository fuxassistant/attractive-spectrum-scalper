"""Causal burst recognition, retrospective recurrence and price landmark study.

Research only: previous exposed periods, snapshot costs, no fitted trading policy.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .response_lab import NS, spectral_features, split_dates
from .full_history import ObservedCalendar
from .core.storage import sha
from .ingest import save_json

VERSION = 'event_landmarks_v1'
HORIZONS = (30, 120)


def detect_bursts(move, volume, supported):
    """5s bins, preceding 120 bins; frozen-current threshold crossing + rearm.

    Return contains no future labels. Four events (three intervals) minimum for
    regular recurrence. An unsupported bin resets history and the state machine.
    """
    move, volume = np.asarray(move), np.asarray(volume)
    valid = np.asarray(supported, bool)
    m = pd.Series(np.where(valid, abs(move), np.nan))
    v = pd.Series(np.where(valid, volume, np.nan))
    mt = np.maximum(3., m.shift().rolling(120, min_periods=120).quantile(.95).to_numpy()*2.5)
    vt = np.maximum(10., v.shift().rolling(120, min_periods=120).quantile(.95).to_numpy()*2.5)
    ready = valid & np.isfinite(mt) & np.isfinite(vt)
    elevated = ready & ((abs(move) > mt) | (volume > vt))
    rows = []; quiet = 0; armed = False; history = []
    for i in range(len(move)):
        if not ready[i]:
            quiet = 0; armed = False; history = []; continue
        history = [j for j in history if (i-j)*5 <= 180]
        if not elevated[i]:
            quiet += 1
            if quiet >= 2: armed = True
            continue
        # A fall in rolling threshold alone must not create a fresh burst.
        p = abs(move[i]) > mt[i] and abs(move[i-1]) <= mt[i]
        q = volume[i] > vt[i] and volume[i-1] <= vt[i]
        if armed and (p or q):
            gaps = np.diff(history+[i])*5
            regular = len(gaps) >= 3 and min(gaps) >= 15 and max(gaps) <= 90 and gaps.std()/gaps.mean() <= .5
            kind = ('fresh_impulse' if not history else 'second_in_cluster' if len(history)==1
                    else 'recurrent_regular' if regular else 'recurrent_irregular')
            rows.append(dict(bin=i, event_class=kind, prior_events_180=len(history),
                interval_cv=float(gaps.std()/gaps.mean()) if len(gaps)>1 else 0.,
                interval_cv_known=len(gaps)>1, last_gap_s=int(gaps[-1]) if len(gaps) else 0,
                activity='joint' if p and q else 'price_led' if p else 'volume_led',
                move_threshold=float(mt[i]), volume_threshold=float(vt[i])))
            history.append(i); armed = False
        quiet = 0
    return rows, ready, elevated, mt, vt


def recurrence_labels(event_bins, all_bins, support):
    """Future repeat only after 180s full support; no negative label at EOF/gap."""
    events = np.asarray(event_bins, int); support = np.asarray(support, bool)
    labels = []
    for i in all_bins:
        mature = i+36 < len(support) and bool(support[i:i+37].all())
        count = int(np.sum((events>i)&(events<=i+36))) if mature else 0
        labels.append(dict(recurrence_mature=mature, future_events_180=count,
                           retrospectively_no_repeat=bool(mature and count==0)))
    return labels


def spectral_diagnostic(d, end):
    z=d.iloc[end-512:end+1]
    ret=np.diff(z.mid.to_numpy())/.1
    q=np.diff((z.c_trade_buy-z.c_trade_sell).to_numpy())
    vol=np.diff((z.c_trade_buy+z.c_trade_sell+z.c_trade_unknown).to_numpy())
    book=np.diff((z.c_add_bid-z.c_remove_bid-z.c_add_ask+z.c_remove_ask).to_numpy())
    channels=dict(return_=ret, classified_flow=q, volume=vol, book_flow=book, abs_return=abs(ret))
    channels['return']=channels.pop('return_')
    # Standardize each channel's energy before combining. Use complete 5s bins.
    energies=np.stack([ret[-510:]**2, q[-510:]**2]).reshape(2,102,5).sum(axis=2)
    totals=energies.sum(axis=1,keepdims=True)
    fractions=np.divide(energies,totals,out=np.zeros_like(energies),where=totals>0)
    peak=int(fractions.sum(axis=0).argmax()); left=2+peak*5
    _, original=spectral_features(channels)
    results=dict(return_peak5_share=float(fractions[0].max()), flow_peak5_share=float(fractions[1].max()))
    # Zero removal and linear interpolation with neighboring 5s are sensitivities,
    # neither is a null distribution or evidence that the shock causes coupling.
    variants={'original':channels}
    for name,pad in [('remove5',0),('interpolate15',5)]:
        copy={k:v.copy() for k,v in channels.items()}; a=max(0,left-pad); b=min(512,left+5+pad)
        for k,v in copy.items():
            if name=='remove5': v[a:b]=0
            else: v[a:b]=np.linspace(v[max(0,a-1)],v[min(511,b)],b-a+2)[1:-1]
        variants[name]=copy
    for name,chs in variants.items():
        cross=original if name=='original' else spectral_features(chs)[1]
        for pair in ('classified_flow_to_return','volume_to_abs_return'):
            vals=[cross[f'{pair}_{a}_{b}_coherence'] for a,b in ((4,8),(8,16),(16,32))]
            known=all(cross[f'{pair}_{a}_{b}_known'] for a,b in ((4,8),(8,16),(16,32)))
            results[f'{pair}_{name}']=float(np.mean(vals)); results[f'{pair}_{name}_known']=known
    return results


def process_capture(job):
    receipt, reference, folder=job; folder=Path(folder)
    market=Path(receipt['market']); feat=Path(receipt['features']); key=receipt['key']
    for p,h in ((market/'seconds.parquet',receipt['market_payloads']['seconds.parquet']),
                (feat/'features.parquet',receipt['feature_sha256'])):
        if sha(p)!=h: raise ValueError('Input changed: '+str(p))
    d=pq.read_table(market/'seconds.parquet').to_pandas()
    f=pq.read_table(feat/'features.parquet',columns=['t_ns','entry_allowed']).to_pandas()
    t=d.t_ns.to_numpy(); av=d.available_ns.to_numpy(); ep=d.epoch.to_numpy(); mid=d.mid.to_numpy()
    publication=np.maximum.accumulate(av)
    if not np.array_equal(t,f.t_ns): raise ValueError('Clock mismatch')
    cal=ObservedCalendar(reference['path'],reference['sha256'])
    ok=d.valid.to_numpy()&~cal.closed(t)&np.isfinite(mid)
    edge=ok[1:]&ok[:-1]&(np.diff(t)==NS)&(np.diff(ep)==0)
    bad=np.r_[0,np.cumsum(~edge)]
    ends=np.flatnonzero((t%(5*NS)==0)&(np.arange(len(t))>=5))
    # Source seconds are dense; include and invalidate any missing bin clock.
    if len(ends)>1 and np.any(np.diff(t[ends])!=5*NS): raise ValueError('Non-dense bin clock')
    supported=(bad[ends]==bad[ends-5])
    move=(mid[ends]-mid[ends-5])/.1
    def delta(name):
        a=d['c_'+name].to_numpy(); return a[ends]-a[ends-5]
    buy,sell,unknown=(delta(n) for n in ('trade_buy','trade_sell','trade_unknown'))
    volume=buy+sell+unknown; flow=buy-sell
    supported &= (buy>=0)&(sell>=0)&(unknown>=0)
    events,ready,elevated,mt,vt=detect_bursts(move,volume,supported)
    # Baseline as of BEFORE current event; used for matched control selection.
    volstate=pd.Series(np.where(supported,move,np.nan)).shift().rolling(120).std().to_numpy()
    up=pd.Series(np.where(supported,unknown,np.nan)).shift().rolling(120).sum().to_numpy()
    vp=pd.Series(np.where(supported,volume,np.nan)).shift().rolling(120).sum().to_numpy()
    up=np.divide(up,vp,out=np.zeros_like(up),where=vp>0)
    recurrences=recurrence_labels([e['bin'] for e in events],[e['bin'] for e in events],supported)
    allowed=f.entry_allowed.to_numpy(); rows=[]; controls_used=set()
    dropped=dict(outside_entry_session=0, unsupported_history=0, unavailable_history=0)
    control_pool=np.flatnonzero(ready & ~elevated & (np.arange(len(ends))%12==0) & allowed[ends])

    def make_row(bi,meta,kind,match_id):
        end=int(ends[bi]); decision=int(publication[end]); start=end-5
        if not allowed[end]:
            if kind=='event': dropped['outside_entry_session']+=1
            return None
        if end<605 or bad[end]!=bad[end-605]:
            if kind=='event': dropped['unsupported_history']+=1
            return None
        if np.max(av[end-605:end+1])>decision:
            if kind=='event': dropped['unavailable_history']+=1
            return None
        sign=int(np.sign(move[bi]))
        if sign==0 and abs(flow[bi])>unknown[bi]: sign=int(np.sign(flow[bi]))
        total=volume[bi]
        row=dict(key=key,date=receipt['entry']['selection']['date'],symbol=receipt['entry']['selection']['symbol'],
                 row_kind=kind,match_id=match_id,bin=bi,end_index=end,t_ns=int(t[end]),decision_ns=decision,
                 event_start_price=float(mid[start]),event_end_price=float(mid[end]),direction=sign,
                 move_ticks=float(move[bi]),volume=float(total),classified_flow=float(flow[bi]),unknown_volume=float(unknown[bi]),
                 unknown_fraction=float(unknown[bi]/total) if total else 0.,
                 pressure_lower=float((flow[bi]-unknown[bi])/total) if total else 0.,
                 pressure_upper=float((flow[bi]+unknown[bi])/total) if total else 0.,
                 flow_direction_robust=bool(abs(flow[bi])>unknown[bi]),
                 baseline_volatility=float(volstate[bi]),baseline_unknown=float(up[bi]),
                 spread_before=float((d.ask.iloc[start]-d.bid.iloc[start])/.1),
                 spread_after=float((d.ask.iloc[end]-d.bid.iloc[end])/.1),**meta)
        for side in ('bid','ask'):
            before=float(d[f'depth_{side}_12'].iloc[start]); after=float(d[f'depth_{side}_12'].iloc[end])
            row[f'depth_{side}_change_fraction']=(after-before)/max(before,1)
            for name in ('add','remove','matched_fill_remove','post_trade_add'):
                col=d[f'c_{name}_{side}']; row[f'{name}_{side}']=float(col.iloc[end]-col.iloc[start])
        row['opposite_depth_change_fraction']=row['depth_ask_change_fraction' if sign>0 else 'depth_bid_change_fraction'] if sign else 0.
        row['same_way_flow_robust']=bool(sign and sign*flow[bi]>unknown[bi])
        row['opposing_flow_robust']=bool(sign and -sign*flow[bi]>unknown[bi])
        row['spread_change']=row['spread_after']-row['spread_before']
        j=int(np.searchsorted(t,decision,side='right'))
        for h in HORIZONS:
            k=j+h
            mature=(k<len(t) and allowed[j] and bad[k]==bad[end] and t[k]-t[j]==h*NS
                    and d.age_ns.iloc[j]<=NS and d.age_ns.iloc[k]<=NS)
            row[f'mature_{h}']=bool(mature)
            row[f'label_available_ns_{h}']=int(max(av[j:k+1])) if mature else 0
            row[f'y_{h}']=float((mid[k]-mid[j])/.1) if mature else 0.
            long=float(d.bid.iloc[k]-d.ask.iloc[j]-.30) if mature else 0.
            short=float(d.bid.iloc[j]-d.ask.iloc[k]-.30) if mature else 0.
            row[f'continue_net_{h}']=(long if sign>0 else short) if sign else 0.
            row[f'fade_net_{h}']=(short if sign>0 else long) if sign else 0.
            row[f'abs_response_{h}']=abs(row[f'y_{h}'])
            path=sign*(mid[j:k+1]-mid[start])/.1 if mature and sign else np.array([])
            extension=sign*(mid[j:k+1]-mid[end])/.1 if mature and sign else np.array([])
            row[f'origin_retest_{h}']=bool(len(path) and np.any(path<=0))
            row[f'extends3ticks_{h}']=bool(len(extension) and np.any(extension>=3))
            row[f'both_landmarks_{h}']=row[f'origin_retest_{h}'] and row[f'extends3ticks_{h}']
        return row

    for event,rec in zip(events,recurrences):
        bi=event['bin']; match_id=f'{key}:{int(t[ends[bi]])}'
        meta={**{k:v for k,v in event.items() if k!='bin'},**rec}
        meta['retrospectively_isolated']=bool(event['event_class']=='fresh_impulse' and rec['retrospectively_no_repeat'])
        meta['recurrence_label_available_ns']=int(publication[ends[bi+36]]) if rec['recurrence_mature'] else 0
        row=make_row(bi,meta,'event',match_id)
        if row is None: continue
        row.update(spectral_diagnostic(d,int(ends[bi]))); rows.append(row)
        # One ordinary sample per event: same capture, <=30min apart, same local
        # volatility factor2, past unknown share within .20. No future outcome use.
        candidates=control_pool[(abs(control_pool-bi)*5<=1800)&(abs(control_pool-bi)*5>=180)]
        candidates=np.array([c for c in candidates if c not in controls_used],dtype=int)
        if len(candidates):
            vr=(volstate[candidates]+.01)/(volstate[bi]+.01)
            good=(vr>=.5)&(vr<=2)&(abs(up[candidates]-up[bi])<=.20)
            candidates=candidates[good]; vr=vr[good]
            if len(candidates):
                score=abs(np.log(vr))+abs(up[candidates]-up[bi])+abs(candidates-bi)*5/1800*.1
                for c in candidates[np.argsort(score)]:
                    control=make_row(int(c),dict(event_class=event['event_class'],activity=event['activity']), 'control',match_id)
                    if control is not None:
                        rows.append(control); controls_used.add(int(c)); break
    stem=key.replace('/','_'); path=folder/(stem+'.parquet')
    if rows: pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')
    result=dict(key=key,events=sum(r['row_kind']=='event' for r in rows),controls=len(controls_used),
                candidates=len(events),dropped=dropped,events_without_control=sum(r['row_kind']=='event' for r in rows)-len(controls_used),
                output_sha256=sha(path) if rows else None,
                source_seconds_sha256=receipt['market_payloads']['seconds.parquet'])
    save_json(folder/(stem+'.json'),result)
    return result


def day_interval(values,dates):
    daily=pd.DataFrame(dict(v=values,date=np.asarray(dates))).groupby('date').v.mean()
    if not len(daily): return dict(mean=None,ci95=None,days=0)
    rng=np.random.default_rng(6118)
    boot=rng.choice(daily.to_numpy(),size=(2000,len(daily)),replace=True).mean(axis=1)
    return dict(mean=float(daily.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),days=len(daily))


def summarize(frame,out):
    events=frame[frame.row_kind=='event'].copy(); controls=frame[frame.row_kind=='control'].copy()
    groups=[]
    for split,mask in split_dates(events.date).items():
        for cls,g in events[mask].groupby('event_class'):
            rec=g[g.recurrence_mature==True]
            item=dict(split=split,event_class=cls,n=len(g),days=g.date.nunique(),
                unknown_fraction_median=float(g.unknown_fraction.median()),
                robust_flow_share=float(g.flow_direction_robust.mean()),same_way_flow_share=float(g.same_way_flow_robust.mean()),
                opposite_depth_change_median=float(g.opposite_depth_change_fraction.median()),
                spread_change_median=float(g.spread_change.median()),
                recurrence_mature=len(rec),no_repeat_180_share=float(rec.retrospectively_no_repeat.mean()) if len(rec) else None,
                return_peak5_share_median=float(g.return_peak5_share.median()),
                coherence_median=float(g.loc[g.classified_flow_to_return_original_known==True,'classified_flow_to_return_original'].median()),
                coherence_remove5_median=float(g.loc[g.classified_flow_to_return_remove5_known==True,'classified_flow_to_return_remove5'].median()),
                coherence_interpolate15_median=float(g.loc[g.classified_flow_to_return_interpolate15_known==True,'classified_flow_to_return_interpolate15'].median()))
            for h in HORIZONS:
                z=g[(g[f'mature_{h}'])&(g.direction!=0)]
                item[str(h)]=dict(n=len(z),continue_net=day_interval(z[f'continue_net_{h}'],z.date),
                    fade_net=day_interval(z[f'fade_net_{h}'],z.date),
                    mean_abs_response_ticks=float(z[f'abs_response_{h}'].mean()) if len(z) else None,
                    origin_retest_share=float(z[f'origin_retest_{h}'].mean()) if len(z) else None,
                    extension_share=float(z[f'extends3ticks_{h}'].mean()) if len(z) else None,
                    both_share=float(z[f'both_landmarks_{h}'].mean()) if len(z) else None)
            groups.append(item)
    pairs=events.merge(controls,on='match_id',suffixes=('_event','_control'))
    matched=[]
    for split,mask in split_dates(pairs.date_event).items():
        for h in HORIZONS:
            z=pairs[mask & pairs[f'mature_{h}_event'] & pairs[f'mature_{h}_control']]
            matched.append(dict(split=split,horizon=h,pairs=len(z),
                abs_response_difference_ticks=day_interval(z[f'abs_response_{h}_event']-z[f'abs_response_{h}_control'],z.date_event)))
    descriptive=[]
    # Hindsight subdivision: explicit descriptive output only, not an action gate.
    mature=events[events.recurrence_mature==True]
    for isolated,g in mature.groupby('retrospectively_no_repeat'):
        descriptive.append(dict(retrospectively_no_repeat=bool(isolated),n=len(g),
            move_abs_median=float(g.move_ticks.abs().median()),volume_median=float(g.volume.median()),
            share_joint=float((g.activity=='joint').mean()),unknown_median=float(g.unknown_fraction.median()),
            peak_energy_median=float(g.return_peak5_share.median()),
            coherence_median=float(g.classified_flow_to_return_original.median()),
            coherence_removed_median=float(g.classified_flow_to_return_remove5.median())))
    summary=dict(status='COMPLETE_DESCRIPTIVE_EVENT_RESEARCH',events=len(events),controls=len(controls),
        captures=int(events.key.nunique()),with_unknown=int((events.unknown_volume>0).sum()),
        groups=groups,matched=matched,retrospective_only=descriptive,
        limitations=['Periods already development-exposed; no blind test.',
        'Event thresholds fixed before this run but this is exploratory with multiple comparisons.',
        'Day intervals weight days equally; not independent event confidence or corrected significance.',
        'Matched controls describe conditional activity, not a randomized causal effect. Can occur later in same session.',
        'Continue/fade are overlapping fixed-horizon 1oz bid/ask snapshot proxies, 0.30 USD roundtrip fee, not fills or equity.',
        'Recurrence is threshold event recurrence, not a stable oscillator. Isolated is a future label only.',
        'Unknown trades retained; volume and pressure bounds do not resolve aggressor identity.',
        'Spectral shock removal is a sensitivity with synthetic edges, not a causal test.'])
    save_json(out/'summary.json',summary)
    return summary


def run(root,output,workers):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'captures').mkdir()
    config=json.loads((root/'run_config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    catalog=json.loads((root/'training_catalog.json').read_text())
    selected={x['key'] for x in catalog['captures'] if x['eligible'] and x['date']!='2026-05-25'}
    receipts=sorted([r for r in manifest['captures'] if r['key'] in selected],key=lambda r:r['key'])
    protocol=dict(version=VERSION,bin_seconds=5,baseline_bins=120,threshold='2.5 x previous p95; floors 3 ticks / 10 contracts; strict >',
        rearm='two quiet bins; upward crossing of frozen current threshold required',
        recurrence='past180s; regular requires >=4 events, >=3 gaps, all15..90s, CV<=.5',
        direction='5s midpoint sign; if zero, classified flow only when abs(known)>unknown; otherwise no directional proxy',
        landmarks='5s event start/end midpoint; origin retest and +3 ticks beyond endpoint after recognition',
        targets='first second endpoint strictly after availability; horizons30/120s with continuous support and fresh quotes',
        isolated='fresh_impulse AND no future repeat180s; future label only, full support and label_available_ns; no_repeat alone also includes terminal bursts',
        activity='price_led/volume_led/joint denote fresh threshold crossings, not all elevated channels',
        captures=len(receipts),source_manifest_sha256=sha(root/'manifest.json'),source_catalog_sha256=sha(root/'training_catalog.json'),
        implementation_sha256=sha(__file__),created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        evaluation='already exposed; exploratory; no policy fitting',workers=workers)
    save_json(out/'protocol.json',protocol);start=time.perf_counter();reports=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(process_capture,[(r,config['reference'],str(out/'captures')) for r in receipts]):
            reports.append(r)
            if len(reports)%10==0: print(json.dumps(dict(captures=len(reports),events=sum(r['events'] for r in reports))),flush=True)
    frame=pd.concat([pq.read_table(p).to_pandas() for p in sorted((out/'captures').glob('*.parquet'))],ignore_index=True)
    frame=frame.sort_values(['date','t_ns','row_kind']).reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),out/'events.parquet',compression='zstd')
    summary=summarize(frame,out);summary.update(seconds=time.perf_counter()-start,events_sha256=sha(out/'events.parquet'),reports=reports)
    save_json(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','events','controls','captures','seconds')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--market',default='data/derived/full_mbo_v1')
    p.add_argument('--output',default='reports/event_landmarks_v1/run_001');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.market,a.output,a.workers)


if __name__=='__main__': main()
