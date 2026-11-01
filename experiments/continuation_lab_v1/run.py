"""Frozen diagnostic week: causal phrase proposals, paired outcomes, accounts."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.mechanical_lab.data import dump, source_for
from experiments.mechanical_lab.execution import QuoteTape, simulate_pairs
from experiments.mechanical_lab.portfolio import account, prepare_outcomes
from experiments.regime_language_v1.measures import TimeAtPrice
from .context import context_features, quote_context, frozen_levels, coarse_texture, NS, MINUTE
from .episodes import generate, arrays

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_day(date, cfg):
    root = ROOT / cfg['source_root'] / date
    m = read(root / 'manifest.json')
    if m['status'] != 'COMPLETE' or m['symbol'] != cfg['symbol']:
        raise ValueError('Unadmitted session store')
    inputs = {str((root / 'manifest.json').relative_to(ROOT)): sha(root / 'manifest.json')}
    cp = root / 'context.parquet'
    if sha(cp) != m['files']['context.parquet']:
        raise ValueError('Context source drift')
    inputs[str(cp.relative_to(ROOT))] = m['files']['context.parquet']
    context = context_features(pd.read_parquet(cp))
    day = int(pd.Timestamp(date, tz='UTC').value)
    start, end, deadline = day + 7 * 3600 * NS, day + (20 * 3600 + 24 * 60) * NS, day + (20 * 3600 + 30 * 60) * NS
    parts = []
    for source in m['quote_sources']:
        p = Path(source['path'])
        if sha(p) != source['sha256']:
            raise ValueError('Quote source drift')
        inputs[str(p.relative_to(ROOT))] = source['sha256']
        q = pd.read_parquet(p, columns=['t_ns','available_ns','bid_nano','ask_nano','epoch','valid','source_ordinal'],
                            filters=[('available_ns', '>=', start - 31 * MINUTE), ('available_ns', '<=', deadline)])
        cutoff = source.get('skip_bootstrap_through_ordinal')
        if cutoff is not None:
            if source.get('handoff_version') != 'IDENTICAL_BOOK_HANDOFF_V1' or not source.get('book_sha256'):
                raise ValueError('Unproven book handoff')
            q = q[q.source_ordinal > cutoff].copy()
        q['epoch'] = q.epoch.replace({int(k): int(v) for k,v in source.get('epoch_map', {}).items()}).astype(np.int64)
        parts.append(q)
    quotes = pd.concat(parts, ignore_index=True).sort_values('available_ns', kind='stable').reset_index(drop=True)
    if (quotes.available_ns < quotes.t_ns).any():
        raise ValueError('Premature quote')
    selected = source_for(date, cfg['route'])
    available = int(selected['entry']['selection']['selection_available_ns'])
    if available > start:
        raise ValueError('Route selection after study start')
    inputs['data/derived/full_mbo_v1/manifest.json'] = sha(ROOT / 'data/derived/full_mbo_v1/manifest.json')
    return quotes, context, inputs, start, end, deadline


def enrich(ledger, quotes, context, date):
    t, p, good, epoch = arrays(quotes)
    av = context.available_ns.to_numpy(np.int64)
    rows = []
    mapper = TimeAtPrice(quotes)
    for x in ledger[ledger.phase == 'CONFIRMED'].itertuples(index=False):
        i = int(x.source_index); clock = int(x.available_ns); side = int(x.side); R = x.risk_nano / NS
        ci = int(av.searchsorted(clock, 'left')) - 1
        c = context.iloc[max(0, ci)]; frozen = context.iloc[x.context_index]
        known = bool(ci >= 0 and c.scale_known and c.epoch == epoch[i] and 0 <= clock - int(c.t_ns) <= 120 * NS)
        spread = (int(quotes.ask_nano.iloc[i]) - int(quotes.bid_nano.iloc[i])) / NS
        row = dict(date=date, t_ns=clock, source_index=i, side=side, story_id=x.story_id,
                   family=x.family, level_kind=x.level_kind, block=int(x.block), risk_hint=R,
                   start_ns=int(t[x.start_index]), peak_ns=int(t[x.peak_index]),
                   trough_ns=int(t[x.trough_index]), gate_ns=int(t[x.gate_index]),
                   start_price=p[x.start_index]/NS, peak_price=p[x.peak_index]/NS,
                   trough_price=p[x.trough_index]/NS, decision_price=p[i]/NS,
                   level=x.level_nano/NS if x.level_nano else np.nan,
                   impulse_r=side*(int(p[x.peak_index])-int(p[x.start_index]))/x.risk_nano,
                   pullback_r=side*(int(p[x.peak_index])-int(p[x.trough_index]))/x.risk_nano,
                   rebound_r=side*(int(p[i])-int(p[x.trough_index]))/x.risk_nano,
                   confirmation_seconds=(clock-int(t[x.gate_index]))/NS,
                   extreme_to_confirmation_seconds=(clock-int(t[x.trough_index]))/NS,
                   context_known=known, context_release_ns=int(c.available_ns),
                   frozen_release_ns=int(frozen.available_ns), spread=spread,
                   cost_over_r=(spread+.3)/R,
                   slope60_r=float(c.slope60/R) if known else np.nan,
                   slope240_r=float(c.slope240/R) if known else np.nan,
                   slope_ok=bool(known and np.isfinite(c.slope60) and side*c.slope60 > 0),
                   macro_alignment='UNKNOWN', regime='UNKNOWN', time_regime=pd.Timestamp(clock,tz='UTC').strftime('%H'))
        if known and np.isfinite(c.slope60) and np.isfinite(c.slope240):
            row['macro_alignment'] = 'ALIGNED' if c.slope60*c.slope240 > 0 else 'CONFLICT'
            row['regime'] = row['macro_alignment']
        for key in ('vwap60','sigma60','vwap240','sigma240','bb1','bb1_half','bb3','bb3_half','bb6','bb6_half'):
            row[key] = float(c[key]) if known else np.nan
            row['frozen_'+key] = float(frozen[key])
        for key in ('day_vwap','combined_vwap','previous_vwap'):
            row[key] = float(c[key]) if known and bool(c[key+'__known']) else np.nan
        row['near_bb'] = bool(np.isfinite(frozen.bb1) and abs(p[x.trough_index]/NS-frozen.bb1) <= .25*R)
        rail_distances = [abs(p[x.trough_index]/NS - frozen.vwap60 - k*frozen.sigma60)/R for k in (-1,0,1)]
        row['near_vwap_rail'] = bool(np.isfinite(rail_distances).all() and min(rail_distances) <= .25)
        row['nearest_rail'] = (-1,0,1)[int(np.argmin(rail_distances))] if np.isfinite(rail_distances).all() else 99
        row['origin_admitted'] = bool(known and good[i] and 0 < spread <= 1. and row['cost_over_r'] <= .75)
        left = max(int(t[0]), clock - 5*MINUTE)
        profile = mapper.facets(left, clock, 12)
        row.update(time_profile_known=profile['known'], time_center=profile['time_mean'],
                   time_migration=profile['center_migration_A']*1.2 if profile['known'] else np.nan,
                   time_overlap=profile['half_profile_overlap'], time_concentration=profile['largest_bin_fraction'])
        row.update(coarse_texture(quotes, int(t[x.start_index]), clock))
        rows.append(row)
    f = pd.DataFrame(rows).sort_values(['t_ns','family','level_kind','side'], kind='stable').reset_index(drop=True)
    return f


def recipes(frame):
    p = frame.family.eq('pullback')
    out = {'pullback_raw':p, 'pullback_slope':p & frame.slope_ok,
           'pullback_slope_bb':p & frame.slope_ok & frame.near_bb,
           'pullback_slope_vwap_rail':p & frame.slope_ok & frame.near_vwap_rail}
    for family in ('break_retest','repeat_bounce'):
        for kind in ('tpo','grid','shifted_tpo'):
            mask = frame.family.eq(family) & frame.level_kind.eq(kind)
            out[family+'_'+kind] = mask & frame.slope_ok
        out[family+'_tpo_raw'] = frame.family.eq(family) & frame.level_kind.eq('tpo')
    return out


def prepare(date, number, cfg, output):
    begun = time.perf_counter(); out = Path(output)/'days'/date; out.mkdir(parents=True, exist_ok=False)
    q, context, inputs, start, end, deadline = load_day(date, cfg)
    ci, risk = quote_context(q, context)
    levels = frozen_levels(q, context, start, end)
    ledger = generate(q, ci, risk, levels, start, end)
    f = enrich(ledger, q, context, date)
    f['candidate_id'] = np.arange(len(f), dtype=np.int64) + number*1_000_000
    f['origin_index'] = f.candidate_id
    f['split'] = 'exposed_week'
    ledger.to_parquet(out/'ledger.parquet', index=False)
    f.to_parquet(out/'features.parquet', index=False)
    context.to_parquet(out/'context.parquet', index=False)
    dump(out/'levels.json', levels)
    admitted = f[f.origin_admitted]
    tape = QuoteTape.from_frame(q, coverage_end_ns=deadline+1, entry_start_ns=start, deadline_ns=deadline)
    frames = []
    for scenario in ('base','stress'):
        ec = dict(cfg['execution'])
        if scenario == 'stress': ec.update(cfg['stress'])
        o = simulate_pairs(tape, admitted.t_ns.to_numpy(np.int64), admitted.risk_hint.to_numpy(float), cfg['exit'], ec)
        o['origin_index'] = admitted.origin_index.to_numpy(np.int64)[o.origin_index.to_numpy(int)]
        o['date'] = date; o['scenario'] = scenario
        frames.append(o)
    outcomes = pd.concat(frames, ignore_index=True); outcomes.to_parquet(out/'outcomes.parquet', index=False)
    # Prefix equality at noon: all real transitions and proposals must survive
    # deleting every later quote and context record. Administrative endings are
    # deliberately excluded; their clock is the supplied observation horizon.
    cutoff = int(pd.Timestamp(date+' 12:00',tz='UTC').value)
    pq = q[q.available_ns<=cutoff].copy(); pc = context[context.available_ns<cutoff].copy()
    pci, pr = quote_context(pq,pc)
    pl = [dict(v,end_ns=min(v['end_ns'],cutoff+1)) for v in levels if v['start_ns']<cutoff]
    prefix = generate(pq,pci,pr,pl,start,cutoff+1)
    cols = ledger.columns
    expected = ledger[(ledger.available_ns<=cutoff)&~ledger.phase_code.isin([13,14])].reset_index(drop=True)
    actual = prefix[(prefix.available_ns<=cutoff)&~prefix.phase_code.isin([13,14])].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected[cols],actual[cols])
    pf = enrich(prefix[prefix.available_ns<=cutoff],pq,pc,date)
    expected_f = f[f.t_ns<=cutoff].drop(columns=['candidate_id','origin_index','split']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_f,pf,check_dtype=True)
    # Save a bounded native visual stream, not a candle interpretation of events.
    scene_rows = []
    for family in ('pullback','break_retest','repeat_bounce'):
        pool = admitted[(admitted.family==family)&admitted.level_kind.isin(['none','tpo'])&admitted.slope_ok]
        if len(pool):
            # Lowest stable hash, selected without outcomes (one per day/family).
            keys = pd.util.hash_pandas_object(pool[['t_ns','side']],index=False)
            c = pool.loc[keys.idxmin()]; left=max(start-30*MINUTE,int(c.start_ns)-60*NS);right=min(deadline,int(c.t_ns)+125*NS)
            a=max(0,int(q.available_ns.searchsorted(left,'right'))-1);b=int(q.available_ns.searchsorted(right,'right'))
            name=str(int(c.candidate_id))+'_quotes.parquet'; q.iloc[a:b].to_parquet(out/name,index=False)
            scene_rows.append(dict(candidate_id=int(c.candidate_id),family=family,path=str((out/name).relative_to(Path(output)))))
    dump(out/'scenes.json',scene_rows)
    receipt=dict(date=date,status='COMPLETE',quote_rows=len(q),transitions=len(ledger),proposals=len(f),
                 admitted=int(f.origin_admitted.sum()),inputs=inputs,source_prefix_rows=len(expected),
                 feature_prefix_rows=len(pf),prefix_equal=True,elapsed_seconds=time.perf_counter()-begun,
                 phases={str(k):int(v) for k,v in ledger.phase.value_counts().items()},
                 artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    dump(out/'receipt.json',receipt)
    return receipt


def evaluate(root, cfg):
    f=pd.concat([pd.read_parquet(root/'days'/d/'features.parquet') for d in cfg['dates']],ignore_index=True)
    outcomes=pd.concat([pd.read_parquet(root/'days'/d/'outcomes.parquet') for d in cfg['dates']],ignore_index=True)
    f.to_parquet(root/'features.parquet',index=False);outcomes.to_parquet(root/'outcomes.parquet',index=False)
    rows=[];trades=[];coins=[];daily=[]
    lookups={s:prepare_outcomes(outcomes[outcomes.scenario==s]) for s in ('base','stress')}
    for name,mask in recipes(f).items():
        offered=f[mask&f.origin_admitted]
        results={}
        for scenario in ('base','stress'):
            for direction in ('signal','reverse'):
                r,t=account(offered,lookups[scenario],direction=direction)
                results[(scenario,direction)]=r
                rows.append(dict(recipe=name,scenario=scenario,direction=direction,**r))
                if len(t):
                    t['recipe']=name;t['scenario']=scenario;t['direction']=direction;trades.append(t)
                    if scenario=='base' and direction=='signal':
                        for d,group in t.groupby('date'):
                            daily.append(dict(recipe=name,date=d,trades=len(group),net_usd=float(group.pnl_usd.sum())))
        controls=[]
        for seed in range(cfg['coin_seeds']):
            r,_=account(offered,lookups['base'],direction='coin',seed=cfg['seed_offset']+seed)
            controls.append(r);coins.append(dict(recipe=name,seed=seed,**r))
        complete=[r['net_usd'] for r in controls if r['capital_curve_complete']]
        base=next(r for r in rows if r['recipe']==name and r['scenario']=='base' and r['direction']=='signal')
        base.update(coin_complete=len(complete),coin_median=float(np.median(complete)) if complete else None,
                    coin_p95=float(np.quantile(complete,.95)) if complete else None,
                    coin_p_descriptive=(1+sum(x>=base['net_usd'] for x in complete))/(1+len(complete)) if len(complete)==cfg['coin_seeds'] else None)
    pd.DataFrame(rows).to_parquet(root/'accounts.parquet',index=False)
    pd.concat(trades,ignore_index=True).to_parquet(root/'trades.parquet',index=False)
    pd.DataFrame(coins).to_parquet(root/'coins.parquet',index=False)
    pd.DataFrame(daily).to_parquet(root/'daily.parquet',index=False)
    # These independent paired trials describe gates; they are NOT a portfolio.
    base=outcomes[outcomes.scenario=='base'].merge(f[['origin_index','side','family','level_kind','slope_ok','near_bb','near_vwap_rail','macro_alignment']],on=['origin_index','side'],validate='many_to_one')
    strata=[]
    for keys,g in base.groupby(['family','level_kind','slope_ok','macro_alignment'],dropna=False):
        valid=g[g.status=='COMPLETE']
        strata.append(dict(family=str(keys[0]),level_kind=str(keys[1]),slope_ok=bool(keys[2]),macro_alignment=str(keys[3]),
                           n=len(g),complete=len(valid),mean_net_per_oz=float(valid.net_per_oz.mean()) if len(valid) else None))
    dump(root/'strata.json',strata)
    return dict(recipes=len(recipes(f)),proposals=len(f),admitted=int(f.origin_admitted.sum()),
                complete_outcomes=int((outcomes.status=='COMPLETE').sum()),outcome_rows=len(outcomes),
                status_counts={str(k):int(v) for k,v in outcomes.status.value_counts().items()},
                admission=cfg['economic_admission'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    root=ROOT/args.out;root.mkdir(parents=True,exist_ok=False);cfg=read(HERE/'config.json')
    sources=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[ROOT/'tests/test_continuation_phrases.py']
    sources += [ROOT/'experiments/mechanical_lab'/n for n in ('execution.py','portfolio.py','data.py')]
    sources += [ROOT/'experiments/regime_language_v1/measures.py']
    identity={str(p.relative_to(ROOT)):sha(p) for p in sources}
    for p in sources:
        dest=root/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    dump(root/'config.json',cfg);begun=time.perf_counter();days=[]
    dump(root/'manifest.json',dict(status='RUNNING',source_sha256=identity))
    try:
        with ProcessPoolExecutor(max_workers=max(1,min(args.workers,2))) as pool:
            futures=[pool.submit(prepare,d,i,cfg,str(root)) for i,d in enumerate(cfg['dates'])]
            for job in as_completed(futures):
                r=job.result();days.append(r)
                print(json.dumps({k:r[k] for k in ('date','quote_rows','proposals','admitted','prefix_equal','elapsed_seconds')},ensure_ascii=False),flush=True)
        summary=evaluate(root,cfg);dump(root/'summary.json',summary)
        if any(sha(ROOT/p)!=v for p,v in identity.items()):raise ValueError('Source changed during build')
        dump(root/'manifest.json',dict(status='COMPLETE',source_sha256=identity,days=sorted(days,key=lambda x:x['date']),
             artifacts={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json' and 'source_snapshot' not in p.parts},
             elapsed_seconds=time.perf_counter()-begun))
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as exc:
        dump(root/'manifest.json',dict(status='FAILED',source_sha256=identity,error=repr(exc),days=days));raise


if __name__=='__main__':main()
