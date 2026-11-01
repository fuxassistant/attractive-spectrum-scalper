"""Bounded descriptive GC day audit, not a strategy or predictive experiment."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.mechanical_lab.data import ROOT, source_for, digest, dump
from experiments.mechanical_lab.signals import build_features
from .memory import build_memory, NS, TICK
from .averages import compare_windows
from .mbo_evidence import load_evidence, summarize_price_evidence

MINUTE = 60 * NS
WIDTH = 5 * TICK


def heat_cells(quotes, trades, start, end,bin_seconds=60):
    """Sparse native-midpoint dwell and T quantity in absolute price/minute cells.

    Time coordinates are publication clocks, intervals [start,end). Quote age
    is capped at30s; gaps are not interpolated. No future quotes/trades retained.
    """
    if bin_seconds<30:raise ValueError('Heat bucket must be at least the quote age cap')
    period=int(bin_seconds*NS)
    clocks = quotes.available_ns.to_numpy(np.int64)
    first = max(0, int(np.searchsorted(clocks, start, 'right')) - 1)
    last = int(np.searchsorted(clocks, end, 'left'))
    q = quotes.iloc[first:last]
    qt = q.available_ns.to_numpy(np.int64)
    if not len(q): return []
    good = q.valid.to_numpy(bool) & (q.epoch.to_numpy() >= 0) & (q.bid_nano.to_numpy()>0) & (q.ask_nano.to_numpy()>q.bid_nano.to_numpy())
    st = np.maximum(qt, start)
    en = np.minimum(np.minimum(np.r_[qt[1:], end], qt+30*NS), end)
    good &= en>st
    st, en = st[good], en[good]
    price = ((q.bid_nano.to_numpy(np.int64)[good]+q.ask_nano.to_numpy(np.int64)[good])//2)//WIDTH*WIDTH
    boundary = (st//period+1)*period
    crosses = en>boundary
    dwell = pd.DataFrame(dict(minute=st//period, price=price, dwell=(np.minimum(en,boundary)-st)/NS))
    if crosses.any():
        dwell = pd.concat([dwell,pd.DataFrame(dict(minute=boundary[crosses]//period,price=price[crosses],dwell=(en[crosses]-boundary[crosses])/NS))],ignore_index=True)
    dwell = dwell.groupby(['minute','price']).dwell.sum()
    tr = trades.loc[(trades.available_ns>=start)&(trades.available_ns<end)&trades.supported].copy()
    tr['minute'] = tr.available_ns//period
    tr['price'] = tr.price_nano//WIDTH*WIDTH
    volume = tr.groupby(['minute','price']).quantity.sum()
    cells = pd.concat([dwell,volume.rename('quantity')],axis=1).fillna(0).reset_index()
    return [[int(r.minute)*bin_seconds,float(r.price)/1e9,round(float(r.dwell),5),int(r.quantity)] for r in cells.itertuples()]


def choose_scenes(features, memory, averages, day):
    """Fixed first-match rules based only on each candidate's known prefix."""
    ready=features.loc[features.ready].copy()
    scenes=[]
    def first(scene_id,title,rule,mask):
        eligible=ready.loc[mask]
        if len(eligible):
            row=eligible.iloc[0]
            scenes.append(dict(id=scene_id,title=title,reason=rule,query_ns=int(row.t_ns)))
            return int(row.t_ns)
        return None
    first('chop','Пила: 08:00 UTC','Первый готовый M1 после08:00 с efficiency20<0.12',
          (ready.bar_end_ns>=day+8*3600*NS)&(ready.efficiency20<.12))
    down=first('sharp_down','Снижение: 08:08–08:09 UTC','Первый M1 после07:30 с body/ATR14 < -1.2',
          (ready.bar_end_ns>=day+(7*3600+30*60)*NS)&(ready.body_atr< -1.2))
    if down:
        first('second_down','Снижение: 08:29–08:30 UTC','Следующий M1 с body/ATR14 < -1.2 минимум20мин позже первой сцены',
              (ready.t_ns>=down+20*60*NS)&(ready.body_atr< -1.2))
    first('trend','Направленный ход: 10:00 UTC','Первый M1 после10:00 с efficiency20>0.5',
          (ready.bar_end_ns>=day+10*3600*NS)&(ready.efficiency20>.5))
    av=averages.copy();av['gap']=av.vwap_trade-av.twa_mid
    gaps=av.pivot(index='query_ns',columns='window_seconds',values='gap')
    opposite=(gaps[1200]*gaps[3600]<0)
    first('horizon_contrast','Разные горизонты: 09:00 UTC','Первый M1 после09:00 с разными знаками VWAP−TWA на20/60мин',
          (ready.bar_end_ns>=day+9*3600*NS)&ready.t_ns.isin(opposite[opposite].index))
    # Arrival into a bin whose absence was already observed at the previous M1.
    grouped={int(t):g.set_index('price_bin_nano') for t,g in memory.groupby('query_ns',sort=False)}
    times=ready.t_ns.to_numpy(np.int64)
    for previous,current in zip(times[:-1],times[1:]):
        if current<day+11*3600*NS:continue
        now,old=grouped[int(current)],grouped[int(previous)]
        anchor=int(round(now.anchor_mid.iloc[0]*1e9))//WIDTH*WIDTH
        if anchor not in old.index:continue
        before,after=old.loc[anchor],now.loc[anchor]
        if before.observed_age_s>=900 and before.age_interval_known and after.arrivals>before.arrivals:
            scenes.append(dict(id='old_zone_return',title='Возвращение к старой зоне',query_ns=int(current),
                reason='Первый M1 после11:00: текущая зона отсутствовала >=15мин на предыдущем M1, затем новое наблюдаемое прибытие',
                prior_query_ns=int(previous),prior_observed_age_s=float(before.observed_age_s)))
            break
    return sorted(scenes,key=lambda s:s['query_ns'])


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,float)):return round(float(value),7) if np.isfinite(value) else None
    return value


def run(date,destination):
    started=time.perf_counter();timings={}
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    source=source_for(date,[1,41271,22]);market=Path(source['market'])
    if not market.is_absolute():market=ROOT/market
    manifest_path=market/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='COMPLETE' or digest(manifest_path)!=source['market_manifest_sha256']:raise ValueError('Market identity mismatch')
    for name in ('quotes.parquet','bars.parquet'):
        if digest(market/name)!=manifest['payloads'][name]:raise ValueError('Market payload mismatch')
    quotes=pd.read_parquet(market/'quotes.parquet')
    bars=pd.read_parquet(market/'bars.parquet')
    features=build_features(bars)
    day=int(pd.Timestamp(date,tz='UTC').value)
    # Current route is selected by07:00, all displayed snapshots occur later.
    features['ready'] &= features.t_ns>=int(source['entry']['selection']['selection_available_ns'])
    queries=features.loc[features.ready,'t_ns'].to_numpy(np.int64)
    raw=ROOT/'data/derived/market_representation_v1/raw'/date
    levels,trades,raw_receipt=load_evidence(raw,day,int(queries[-1]//NS*NS),int(queries[-1]))
    timings['verified_load_seconds']=time.perf_counter()-started
    began=time.perf_counter();memory=build_memory(quotes,queries)
    timings['memory_seconds']=time.perf_counter()-began
    began=time.perf_counter()
    # One partial final second is not needed by any earlier20/60min comparison;
    # last query's final milliseconds are explicitly outside trade coverage.
    # Use complete original T payload for exact publication-window arithmetic.
    all_trades=pd.read_parquet(raw/'trades.parquet')
    average_queries=features.loc[features.t_ns<=queries[-1],'t_ns'].to_numpy(np.int64)
    averages=compare_windows(quotes,all_trades,average_queries,coverage_start_ns=day,
        coverage_end_ns=int(manifest['input']['end_ns'])+1)
    timings['averages_seconds']=time.perf_counter()-began
    scenes=choose_scenes(features,memory,averages,day)
    dump(out/'selection.json',scenes)
    began=time.perf_counter()
    for scene in scenes:
        at=scene.pop('query_ns');physical_end=at//NS*NS
        mf=memory.loc[memory.query_ns==at].copy()
        ll=levels.loc[(levels.t_ns<=physical_end)&(levels.available_ns<=at)]
        tt=all_trades.loc[(all_trades.available_ns<=at)&(all_trades.t_ns<physical_end)]
        evidence=summarize_price_evidence(ll,tt)
        fields=['price_bin_nano','trade_buy_qty','trade_sell_qty','trade_unknown_qty','bid_passive_fill','bid_unlinked_remove','bid_refill','ask_passive_fill','ask_unlinked_remove','ask_refill']
        merged=mf.merge(evidence[fields],on='price_bin_nano',how='left',validate='one_to_one')
        merged[fields[1:]]=merged[fields[1:]].fillna(0)
        merged['price']=merged.price_low
        display_fields=['price','mid_dwell_s','arrivals','present_at_restart','pass_up','pass_down','reject_up','reject_down','jump_over','censored','active','observed_age_s','age_interval_known']+fields[1:]
        recent_features=features.loc[features.t_ns<=at]
        av20=averages.loc[averages.window_seconds==1200].set_index('query_ns')
        full_path=[]
        for r in recent_features.itertuples():
            a=av20.loc[r.t_ns] if r.t_ns in av20.index else None
            vw=float(a.vwap_trade) if a is not None and a.trade_known else np.nan
            vs=float(a.vwap_trade_sigma_population) if a is not None and a.trade_known else np.nan
            tw=float(a.twa_mid) if a is not None and a.quote_known else np.nan
            full_path.append([r.t_ns/NS,r.c,r.bb_mid,r.bb_upper,r.bb_lower,vw,vw+2*vs,vw-2*vs,tw])
        path=[row for row in full_path if row[0]>at/NS-3600]
        history_path=[row for i,row in enumerate(full_path) if i%5==0 or i==len(full_path)-1]
        near_levels=ll.loc[(ll.t_ns>physical_end-60*NS)]
        near_trades=tt.loc[tt.t_ns>=physical_end-60*NS]
        local=summarize_price_evidence(near_levels,near_trades)
        scene.update(asof_ns=str(at),asof_utc=pd.Timestamp(at,tz='UTC').isoformat(),
            mid=float(mf.anchor_mid.iloc[0]),levels=merged[display_fields].to_dict('records'),path=path,
            heat=heat_cells(quotes,all_trades,at-3600*NS,at),
            history_heat=heat_cells(quotes,all_trades,day,at,bin_seconds=300),history_path=history_path,history_bin_seconds=300,
            averages=averages.loc[averages.query_ns==at].drop(columns=['query_ns','left_ns']).to_dict('records'),
            evidence={k:float(local[k].sum()) for k in fields[1:]},
            coverage_fraction=float(mf.history_coverage.iloc[0]),
            history_start_utc=pd.Timestamp(int(quotes.available_ns.iloc[0]),tz='UTC').isoformat(),
            current_quote_known=bool(mf.current_quote_known.iloc[0]))
        # Persist unrestricted price-local observed activity too, beyond viewport.
        evidence.to_parquet(out/f'{scene["id"]}_price_evidence.parquet',index=False)
    timings['scene_seconds']=time.perf_counter()-began
    memory.to_parquet(out/'memory.parquet',index=False)
    averages.to_parquet(out/'averages.parquet',index=False)
    features.to_parquet(out/'features.parquet',index=False)
    data=dict(date=date,symbol='GCZ6',heat_bin_seconds=60,price_bin_width=.5,
        level_history='Observed UTC-day prefix; midpoint contact != execution; unknown history != no interest',
        path_clock='available_ns; completed M1 close. VWAP/TWA windows [asof-1200s,asof)',
        bb_definition='20 completed M1 midpoint closes, sample SD ddof1, multiplier2',
        vwap_definition='Exact accepted T prints, quantity-weighted population SD, multiplier2',scenes=scenes)
    dump(out/'visual_data.json',clean(data))
    receipt=dict(status='COMPLETE_DESCRIPTIVE_PROTOTYPE',date=date,route=[1,41271,22],symbol='GCZ6',
        source_manifest_sha256=digest(manifest_path),source_payloads=manifest['payloads'],raw_source=raw_receipt,
        quote_rows=len(quotes),map_rows=len(memory),snapshots=len(queries),scenes=len(scenes),
        known_average_rows=int(averages.known.sum()),average_rows=len(averages),
        max_decomposition_residual=float(averages.decomposition_residual.abs().max()),
        coverage_min=float(memory.history_coverage.min()),
        time_price_identity=['route','symbol','absolute_price_bin_nano','map_version=1'],
        geometry=dict(bin_ticks=5,hysteresis_ticks=2,viewport_radius_ticks=300,max_quote_age_seconds=30),
        timing_seconds=timings,elapsed_seconds=time.perf_counter()-started,
        limitations=['Single previously inspected date; no predictive or economic conclusion',
            'Day-prefix memory only; no cross-day/session persistence yet',
            'Dwell uses supported quotes; contacts are midpoint geometry, not executions',
            'Level mutations are observed flows, not absolute resting depth or participant intention',
            'BB close grid differs from native print VWAP; TWA added to isolate time weighting',
            'Sparse sidecar rows lack per-level continuity epoch; cumulative activity is observed-history only'],
        implementation_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'experiments/price_level_memory_v1').glob('*.py'))},
        artifact_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    dump(out/'receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],snapshots=len(queries),scenes=len(scenes),
        known_average_rows=receipt['known_average_rows'],timing_seconds=timings,out=str(out)),ensure_ascii=False),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--date',default='2026-08-24')
    parser.add_argument('--out',default='research_results/price_level_memory_v1/day_20260824')
    args=parser.parse_args();run(args.date,args.out)
