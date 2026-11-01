"""Measured BBO costs and fixed-execution attribution on already-exposed days.

No strategy, threshold, stop, target, or execution is rerun. A midpoint add-back
is an accounting identity at the saved fills, not a zero-cost trading policy:
the saved exits and account admission were already chosen with costs present.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NS = 1_000_000_000
HOUR = 3600 * NS
CONTRACT_OZ = 100.
SOURCE = ROOT / 'research_results/zone_conditional_v1/verified_v1'
ZONE = ROOT / 'research_results/zone_episode_lab/verified_v3'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def dump(path, data):
    Path(path).write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Hash mismatch: {path}')
    return actual


def quote_arrays(q):
    for name in ('available_ns', 'bid_nano', 'ask_nano', 'epoch'):
        if q[name].dtype.kind not in 'iu':
            raise ValueError('Exact integer quote clocks/prices required')
    t = q.available_ns.to_numpy(np.int64)
    if not len(t) or np.any(t[1:] < t[:-1]):
        raise ValueError('Nonempty sorted quote tape required')
    bid = q.bid_nano.to_numpy(np.int64); ask = q.ask_nano.to_numpy(np.int64)
    good = q.valid.to_numpy(bool) & q.epoch.ge(0).to_numpy() & (bid > 0) & (ask > bid)
    return t, bid, ask, good


def _histogram(t, bid, ask, good, start, end, coverage, gap):
    """Exact supported nanosecond mass by spread over [start,end)."""
    lo = max(0, int(np.searchsorted(t, start, 'right'))-1)
    hi = int(np.searchsorted(t, min(end, coverage), 'left'))
    if hi <= lo:
        return pd.DataFrame({'spread_nano': np.array([], np.int64), 'duration_ns': np.array([], np.int64)})
    next_t = t[lo+1:hi+1] if hi < len(t) else np.r_[t[lo+1:hi], np.int64(coverage)]
    left = np.maximum(t[lo:hi], start)
    right = np.minimum(np.minimum(next_t, t[lo:hi]+gap), min(end, coverage))
    dt = np.where(good[lo:hi], np.maximum(0, right-left), 0)
    keep = dt > 0
    spread, inv = np.unique((ask[lo:hi]-bid[lo:hi])[keep], return_inverse=True)
    # Each hourly mass is <=3.6e12 ns, exactly representable in float64; convert
    # the integer-valued bincount back, and retain integer exposure thereafter.
    mass = np.bincount(inv, weights=dt[keep]).astype(np.int64)
    return pd.DataFrame(dict(spread_nano=spread, duration_ns=mass))


def spread_histogram(quotes, *, start_ns, end_ns, coverage_end_ns, max_age_seconds=30):
    for value in (start_ns, end_ns, coverage_end_ns):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError('Exact integer interval clocks required')
    if end_ns <= start_ns or max_age_seconds <= 0:
        raise ValueError('Positive interval/support required')
    return _histogram(*quote_arrays(quotes), int(start_ns), int(end_ns), int(coverage_end_ns), int(max_age_seconds*NS))


def histogram_summary(histogram):
    if histogram.empty or not histogram.duration_ns.sum():
        return dict(supported_ns=0, mean_spread_per_oz=np.nan, p10_spread_per_oz=np.nan,
                    p50_spread_per_oz=np.nan, p90_spread_per_oz=np.nan, p99_spread_per_oz=np.nan,
                    max_spread_per_oz=np.nan)
    h = histogram.groupby('spread_nano', as_index=False).duration_ns.sum().sort_values('spread_nano')
    mass = h.duration_ns.to_numpy(np.int64); prices = h.spread_nano.to_numpy(np.int64)
    total = int(mass.sum()); cumulative = mass.cumsum()
    result = dict(supported_ns=total, mean_spread_per_oz=float(np.average(prices/NS, weights=mass)),
                  max_spread_per_oz=float(prices[-1]/NS))
    for pct in (10, 50, 90, 99):
        result[f'p{pct}_spread_per_oz'] = float(prices[np.searchsorted(cumulative, total*pct/100, 'left')]/NS)
    return result


def distribution(values):
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    if not len(x):
        return dict(rows=0)
    return dict(rows=len(x), mean=float(x.mean()), minimum=float(x.min()), maximum=float(x.max()),
                **{f'p{p}':float(np.quantile(x, p/100)) for p in (5,25,50,75,90,95,99)})


def decompose_trades(quotes, trades, *, fee_per_oz=.3, slippage_ticks_per_fill=0.):
    """Exact native midpoint/spread identity at saved entry and exit ordinals."""
    t, bid, ask, good = quote_arrays(quotes)
    out = trades.copy()
    for col in ('entry_ordinal', 'exit_ordinal', 'entry_ns', 'exit_ns', 'side'):
        if out[col].dtype.kind not in 'iu':
            raise ValueError('Saved execution clocks/ordinals must remain exact integers')
    e = out.entry_ordinal.to_numpy(np.int64); x = out.exit_ordinal.to_numpy(np.int64)
    side = out.side.to_numpy(np.int64)
    if (e < 0).any() or (x < e).any() or (x >= len(t)).any() or not np.isin(side,[-1,1]).all():
        raise ValueError('Invalid saved execution index/direction')
    if not np.array_equal(t[e], out.entry_ns.to_numpy()) or not np.array_equal(t[x], out.exit_ns.to_numpy()):
        raise ValueError('Saved execution timestamp differs from native ordinal')
    if not (good[e] & good[x]).all():
        raise ValueError('Saved fills require supported native BBO')
    slip = slippage_ticks_per_fill * .1
    expected_entry = np.where(side == 1, ask[e]/NS+slip, bid[e]/NS-slip)
    expected_exit = np.where(side == 1, bid[x]/NS-slip, ask[x]/NS+slip)
    if not np.allclose(expected_entry, out.entry_price, atol=1e-10, rtol=0) or not np.allclose(expected_exit, out.exit_price, atol=1e-10, rtol=0):
        raise ValueError('Saved execution price differs from BBO plus frozen slippage')
    out['entry_spread_per_oz'] = (ask[e]-bid[e])/NS
    out['exit_spread_per_oz'] = (ask[x]-bid[x])/NS
    out['native_mid_move_per_oz'] = side*((bid[x]+ask[x])-(bid[e]+ask[e]))/(2*NS)
    out['entry_half_spread_per_oz'] = (ask[e]-bid[e])/(2*NS)
    out['exit_half_spread_per_oz'] = (ask[x]-bid[x])/(2*NS)
    out['spread_crossing_per_oz'] = out.entry_half_spread_per_oz+out.exit_half_spread_per_oz
    out['fee_per_oz'] = fee_per_oz
    out['slippage_per_oz'] = 2*slip
    out['total_explicit_cost_per_oz'] = out.spread_crossing_per_oz+fee_per_oz+2*slip
    out['reconstructed_net_per_oz'] = out.native_mid_move_per_oz-out.total_explicit_cost_per_oz
    out['net_identity_residual_per_oz'] = out.reconstructed_net_per_oz-out.net_per_oz
    out['fixed_fill_no_fee_per_oz'] = out.net_per_oz+fee_per_oz
    for col in ('fee_per_oz', 'spread_crossing_per_oz', 'slippage_per_oz', 'total_explicit_cost_per_oz'):
        out[col.removesuffix('_per_oz')+'_over_r'] = out[col]/out.risk_per_oz
    for col in ('native_mid_move', 'spread_crossing', 'fee', 'slippage', 'total_explicit_cost'):
        out[col+'_usd'] = out[col+'_per_oz']*out.qty*CONTRACT_OZ
    if 'source_ordinal' in quotes:
        raw = quotes.source_ordinal.to_numpy(np.int64)
        out['entry_raw_source_ordinal'] = raw[e]; out['exit_raw_source_ordinal'] = raw[x]
    if out.net_identity_residual_per_oz.abs().max() > 1e-10:
        raise ValueError('Cost identity failed')
    return out


def barrier_probe():
    from experiments.mechanical_lab.execution import QuoteTape, ExitPlan, simulate_pairs
    from spectrum.execution.barriers import SimulationConfig, scalar_reference
    t = np.rint(np.array([0,.11,.3,.4,.41,2.11,2.22])*NS).astype(np.int64)
    bid = np.rint(np.array([4000,4000,4001.7,4001.7,4001.7,4001.7,4001.7])*NS).astype(np.int64)
    q = QuoteTape(t,bid,bid+200_000_000,np.ones(len(t),bool),np.zeros(len(t),np.int64),100*NS,0,100*NS)
    plan = ExitPlan(risk_mode='fixed',fixed_stop_ticks=12.,take_r=1.,max_hold_seconds=2.)
    current = simulate_pairs(q,np.array([0],np.int64),np.ones(1),plan).iloc[0]
    legacy = scalar_reference(q,0,12*.1,1,SimulationConfig(gain_multiple=1.,deadline_seconds=2.,max_quote_gap_seconds=30.))
    assert current.reason == 'TAKE' and current.trigger_ns == 300_000_000
    library_passes = legacy['reason'] == 'TAKE' and int(legacy['trigger_ns']) == 300_000_000
    return dict(contract='Exact 12-tick net target at $4001.7 bid after $4000.2 ask entry and $0.30 fee',
                risk_float=12*.1,current=dict(reason=current.reason,trigger_ns=int(current.trigger_ns),net_per_oz=float(current.net_per_oz)),
                legacy=dict(reason=legacy['reason'],trigger_ns=int(legacy['trigger_ns']),net_per_oz=float(legacy['net'])),
                library_exact_line_pass=library_passes,
                interpretation=('Both current experiment and library pass this exact-line probe. The pre-0.4.1 TIME result is preserved in cost_audit_v1 execution_archaeology.json and source_snapshot.' if library_passes else
                                'At this producer version the mechanical experiment passes but library barriers misses the exact-line threshold. This audit changes neither implementation.'))


def git_text(*args):
    return subprocess.run(['git',*args],cwd=ROOT,check=True,capture_output=True,text=True,encoding='utf-8').stdout.strip()


def archaeology():
    prior = git_text('show','1fe5231:reports/cognition_audit_20261019/AUDIT_RU.md')
    return dict(head=git_text('rev-parse','HEAD'),branch=git_text('branch','--show-current'),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'pyproject.toml',ROOT/'spectrum/__init__.py',ROOT/'experiments/mechanical_lab/execution.py',ROOT/'spectrum/execution/barriers.py',ROOT/'spectrum/market/queue.py',ROOT/'tests/test_queue.py',ROOT/'tests/test_mechanical_execution.py']},
        barrier_probe=barrier_probe(),
        passive_history=dict(current_kernel='MBO_FIFO_QUEUE_PROBE_V5',
            current_source='spectrum/market/queue.py',
            historical_commits=[dict(commit=git_text('rev-parse',ref),subject=git_text('show','-s','--format=%s',ref)) for ref in ['b5c87ee','55fda8a','1fe5231','42878dd']],
            prior_audit_git_path='1fe5231:reports/cognition_audit_20261019/AUDIT_RU.md',
            prior_audit_sha256=hashlib.sha256(prior.encode()).hexdigest(),
            relevant_prior_audit_lines=[line for line in prior.splitlines() if any(term in line for term in ['520','неблагоприятный','PQ_30','108 train'])],
            scope='Read-only historical source/report verification. No old raw days or new holdout outcomes opened; 520/520 is a verified historical report, not a rerun today. Passive execution is not untouched.'))


def render_report(out, result, hours, groups):
    c=result['candidate_clock_distributions'];all_day=result['time_weighted_all_hours'];active=result['time_weighted_policy_session']
    probe=json.loads((out/'execution_archaeology.json').read_text(encoding='utf-8'))['barrier_probe']
    probe_text=('Both current mechanical and library implementations pass the exact-line 12-tick probe. The earlier pre-0.4.1 failure is retained in the separate frozen cost_audit_v1 history.'
                if probe['legacy']['reason']=='TAKE' and probe['legacy']['trigger_ns']==300_000_000 else
                'At this run version the mechanical engine passes the exact-line 12-tick probe while spectrum.execution.barriers misses it. The conditional experiment uses the mechanical engine. No implementation was changed by this audit.')
    head=f"""<h1>Observed costs on the already-exposed 22-day tape</h1>
<p>No new holdout was opened. No trading rule or execution was rerun. Components use saved native entry/exit quotes.</p>
<p>Time-weighted BBO spread: all supported calendar hours <b>{all_day['mean_spread_per_oz']:.4f}</b> USD/oz;
policy session 07:00–20:30 UTC <b>{active['mean_spread_per_oz']:.4f}</b> USD/oz.
Unique candidate-clock mean/median: <b>{c['spread_per_oz']['mean']:.4f}/{c['spread_per_oz']['p50']:.4f}</b> USD/oz.</p>
<p>Frozen round-trip commission assumption: <b>$0.30/oz = $30 per 100-oz GC contract</b>. It is not an audited broker tariff and not 0.3R.
Observed fee/R median: <b>{c['fee_over_r']['p50']:.4f}</b>; p5/p95: <b>{c['fee_over_r']['p5']:.4f}/{c['fee_over_r']['p95']:.4f}</b>.
Decision spread + fee proxy/R median: <b>{c['decision_cost_proxy_over_r']['p50']:.4f}</b>.</p>
<p><code>net = signed midpoint movement − entry half-spread − exit half-spread − fee − slippage</code>.
These costs are positive deductions. Latency affects the observed fill prices and is not charged twice.
Adding costs back at frozen fills is an identity, not the result of rerunning a zero-cost policy; net barriers and account caps can change exits/admission in such a rerun.</p>
<h2>UTC hour: elapsed-time spread and decision-clock spread</h2>"""
    show_hours=hours[['hour_utc','supported_hours','coverage_fraction','mean_spread_per_oz','p50_spread_per_oz','p90_spread_per_oz','candidate_clocks','candidate_mean_spread_per_oz','candidate_p50_spread_per_oz']]
    show_groups=groups[['block','arm','condition','trades','capital_curve_complete','mean_mid_move_per_oz','mean_spread_crossing_per_oz','mean_fee_per_oz','mean_slippage_per_oz','mean_net_per_oz','net_usd']]
    tail=f"""<h2>Saved account trades: fixed-fill attribution</h2><p>Each row is a separate account. Models share opportunities; rows must not be pooled as independent trials or a combined portfolio. Incomplete accounts show only their closed trades.</p>{show_groups.to_html(index=False,float_format=lambda x:f'{x:.4f}')}
<h2>Execution archaeology</h2><p>{probe_text}</p>
<p>Passive execution already exists: current <code>MBO_FIFO_QUEUE_PROBE_V5</code>, earlier learned resting orders, and sequential passive take-profit work are recorded in Git. The historical audit reports 520/520 FIFO parity and adverse selection; it does not establish passive profitability.</p>
<p>Lineage, raw capture/member hashes, exact configuration, numerical residuals and source versions are in <code>summary.json</code> and <code>manifest.json</code>. Raw staging was retired: stored raw hashes are lineage, while canonical manifests and quote payloads were freshly verified.</p>"""
    doc='<html><head><meta charset="utf-8"><title>Cost audit</title><style>body{font:15px/1.5 system-ui;margin:32px;color:#20313c}table{border-collapse:collapse;font-size:13px}td,th{padding:6px 10px;border-bottom:1px solid #ddd;text-align:right}p{max-width:1200px}h1,h2{font-weight:600}h2{margin-top:32px}</style></head><body>'+head+show_hours.to_html(index=False,float_format=lambda x:f'{x:.4f}')+tail+'</body></html>'
    (out/'report.html').write_text(doc,encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='research_results/edge_diagnostics_v1/cost_audit_v1');args=parser.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    sm=json.loads((SOURCE/'manifest.json').read_text());zm=json.loads((ZONE/'manifest.json').read_text())
    assert sm['status']==zm['status']=='COMPLETE_DIAGNOSTIC_ONLY'
    assert sm['source_manifest_sha256']==digest(ZONE/'manifest.json')
    inputs={str(SOURCE/'manifest.json'):digest(SOURCE/'manifest.json'),str(ZONE/'manifest.json'):digest(ZONE/'manifest.json')}
    names=['dataset','policy_clock','account_trades','policy_summary']
    tables={}
    for name in names:
        p=SOURCE/(name+'.parquet');inputs[str(p)]=verify(p,sm['artifacts'][p.name]);tables[name]=pd.read_parquet(p)
    config=json.loads((ZONE/'config.json').read_text());fee=float(config['execution']['fee_roundtrip_per_oz']);gap=int(config['execution']['max_quote_gap_seconds']*NS)
    data=tables['dataset'];clock=tables['policy_clock'].merge(data,on=['candidate_id','t_ns','date','block'],validate='one_to_one')
    assert len(clock)==clock.t_ns.nunique() and len(clock)==6881
    raw_cfg_path=ROOT/'data/derived/full_mbo_v1/run_config.json';raw_cfg=json.loads(raw_cfg_path.read_text());inputs[str(raw_cfg_path)]=digest(raw_cfg_path)
    members={r['key']:r for r in raw_cfg['members']};day_hours=[];histograms=[];active_histograms=[];candidate_rows=[];trade_rows=[];lineage=[]
    archived=archaeology();dump(out/'execution_archaeology.json',archived)
    dump(out/'manifest.json',dict(status='RUNNING',source_inputs=inputs,scope='Known22days only; fixed execution accounting, not zero-cost replay'))
    for r in sorted(sm['receipts'],key=lambda r:r['date']):
        date=r['date'];market=Path(r['source_market']);mp=market/'manifest.json'
        verify(mp,r['source_hashes'][str(mp)]);mm=json.loads(mp.read_text());qp=market/'quotes.parquet'
        qhash=verify(qp,mm['payloads']['quotes.parquet']);assert qhash==r['source_hashes'][str(qp)]
        q=pd.read_parquet(qp,columns=['available_ns','bid_nano','ask_nano','valid','epoch','source_ordinal'])
        arrays=quote_arrays(q);coverage=int(mm['input']['end_ns'])+1;day=int(pd.Timestamp(date,tz='UTC').value)
        for hour in range(24):
            start=day+hour*HOUR;end=start+HOUR
            hist=_histogram(*arrays,start,end,coverage,gap);stats=histogram_summary(hist)
            day_hours.append(dict(date=date,hour_utc=hour,**stats,coverage_fraction=stats['supported_ns']/HOUR))
            hist['date']=date;hist['hour_utc']=hour;histograms.append(hist)
        active_histograms.append(_histogram(*arrays,day+7*HOUR,day+20*HOUR+1800*NS,coverage,gap))
        c=clock.loc[clock.date.eq(date)].copy();qt,bid,ask,good=arrays;ix=np.searchsorted(qt,c.t_ns.to_numpy(np.int64),'right')-1
        assert (ix>=0).all() and good[ix].all() and (c.t_ns.to_numpy()-qt[ix]<=gap).all()
        spread=(ask[ix]-bid[ix])/NS;assert np.allclose(spread,c.spread_per_oz,atol=1e-12,rtol=0)
        c=c[['candidate_id','t_ns','date','block','side','risk_per_oz']].copy();c['hour_utc']=((c.t_ns-day)//HOUR).astype(int)
        c['quote_available_ns']=qt[ix];c['quote_age_ns']=c.t_ns.to_numpy()-qt[ix];c['spread_per_oz']=spread;c['fee_per_oz']=fee
        c['fee_over_r']=fee/c.risk_per_oz;c['spread_over_r']=spread/c.risk_per_oz
        c['decision_cost_proxy_per_oz']=spread+fee;c['decision_cost_proxy_over_r']=(spread+fee)/c.risk_per_oz;candidate_rows.append(c)
        trades=tables['account_trades'].loc[tables['account_trades'].date.eq(date)]
        for condition,g in trades.groupby('condition'):
            slip=config['stress']['slippage_ticks_per_fill'] if condition=='stress' else config['execution']['slippage_ticks_per_fill']
            trade_rows.append(decompose_trades(q,g,fee_per_oz=fee,slippage_ticks_per_fill=slip))
        member=members[mm['input']['member_key']]
        rawpath=ROOT/'data/derived/full_mbo_v1/staging'/mm['input']['member_key']/mm['input']['name']
        lineage.append(dict(date=date,canonical_manifest=str(mp),canonical_manifest_sha256=digest(mp),quotes_sha256=qhash,
            quote_rows=len(q),canonical_implementation_sha256=mm['implementation'],raw_capture_sha256=mm['input']['sha256'],
            source_partition_sha256=mm['source_partition_sha256'],raw_capture_records=mm['input']['records'],
            raw_first_ordinal=mm['input']['first_ordinal'],raw_last_ordinal=mm['input']['last_ordinal'],
            raw_archive=member['archive'],raw_member=member['member'],raw_member_sha256=member['sha256'],
            raw_staging_exists=rawpath.exists(),raw_hash_status='Lineage from freshly verified canonical manifest and recorded source config; raw archive not reread',
            selection_available_ns=int(mm['input']['selection']['selection_available_ns'])))
        print(json.dumps(dict(date=date,quote_rows=len(q),candidate_clocks=len(c),saved_trade_rows=len(trades))),flush=True)
    hourly=pd.DataFrame(day_hours);hist=pd.concat(histograms,ignore_index=True);candidates=pd.concat(candidate_rows,ignore_index=True);trades=pd.concat(trade_rows,ignore_index=True)
    hourly_aggregate=[]
    for hour,h in hist.groupby('hour_utc'):
        stat=histogram_summary(h);cd=candidates.loc[candidates.hour_utc.eq(hour)]
        hourly_aggregate.append(dict(hour_utc=int(hour),**stat,supported_hours=stat['supported_ns']/HOUR,
            coverage_fraction=stat['supported_ns']/(len(lineage)*HOUR),candidate_clocks=len(cd),
            candidate_mean_spread_per_oz=float(cd.spread_per_oz.mean()),candidate_p50_spread_per_oz=float(cd.spread_per_oz.median())))
    hours=pd.DataFrame(hourly_aggregate)
    groups=[]
    for key,g in trades.groupby(['block','arm','condition']):
        s=tables['policy_summary'].loc[tables['policy_summary'].block.eq(key[0])&tables['policy_summary'].arm.eq(key[1])&tables['policy_summary'].condition.eq(key[2])].iloc[0]
        assert len(g)==s.trades and abs(g.pnl_usd.sum()-s.net_usd)<1e-8
        row=dict(block=key[0],arm=key[1],condition=key[2],trades=len(g),capital_curve_complete=bool(s.capital_curve_complete),unresolved=int(s.unresolved),
            net_usd=float(g.pnl_usd.sum()),mean_net_per_oz=float(g.net_per_oz.mean()),median_total_cost_over_r=float(g.total_explicit_cost_over_r.median()))
        for name in ('native_mid_move','spread_crossing','fee','slippage','total_explicit_cost'):
            row['mean_'+('mid_move' if name=='native_mid_move' else name)+'_per_oz']=float(g[name+'_per_oz'].mean())
            row[name+'_usd']=float(g[name+'_usd'].sum())
        assert abs(row['native_mid_move_usd']-row['total_explicit_cost_usd']-row['net_usd'])<1e-8
        groups.append(row)
    groups=pd.DataFrame(groups)
    result=dict(status='COMPLETE_DIAGNOSTIC_ONLY',dates=[r['date'] for r in lineage],days=len(lineage),
        scope='Already exposed source days only; no new holdout, zero-cost replay, parameter search or passive experiment.',
        fee_assumption_per_oz=fee,fee_assumption_per_100oz_gc=fee*CONTRACT_OZ,fee_is_broker_tariff=False,
        fee_over_r_definition='0.30 / frozen risk_per_oz, not constant0.3R',
        time_weighted_semantics='Spread held over native publication intervals, clipped at next publication, invalid state, coverage, and30s quote age; exact integer-ns exposure. All-hours before07UTC are retrospective route context.',
        time_weighted_all_hours=histogram_summary(hist),time_weighted_policy_session=histogram_summary(pd.concat(active_histograms,ignore_index=True)),
        candidate_clock_distributions={name:distribution(candidates[name]) for name in ['risk_per_oz','spread_per_oz','fee_over_r','spread_over_r','decision_cost_proxy_per_oz','decision_cost_proxy_over_r']},
        candidate_clocks=len(candidates),saved_trade_rows=len(trades),unique_fill_paths=int(trades.drop_duplicates(['date','side','entry_ordinal','exit_ordinal','condition']).shape[0]),
        max_net_identity_residual_per_oz=float(trades.net_identity_residual_per_oz.abs().max()),
        accounting='net=side*(mid_exit-mid_entry)-halfspread_entry-halfspread_exit-fee-2*per_fill_slippage. Native integer price differences before float conversion.',
        counterfactual_limit='Midpoint and no-fee add-backs use cost-conditioned saved executions. They do not rerun barriers, timing, support, risk caps or admission. No assumption that passive orders fill or save half costs.',
        lineage=lineage,execution_archaeology_file='execution_archaeology.json',source_inputs=inputs,
        source_code_sha256=digest(__file__),elapsed_seconds=time.perf_counter()-started)
    outputs={'hourly_spread':hourly,'hourly_spread_pooled':hours,'spread_duration_histogram':hist,'candidate_clock_costs':candidates,'trade_cost_components':trades,'account_cost_summary':groups}
    for name,f in outputs.items():f.to_parquet(out/(name+'.parquet'),index=False)
    dump(out/'summary.json',result);render_report(out,result,hours,groups)
    dump(out/'manifest.json',dict(status='COMPLETE_DIAGNOSTIC_ONLY',source_code_sha256=digest(__file__),source_inputs=inputs,
        artifacts={p.name:digest(p) for p in out.iterdir() if p.name!='manifest.json'},days=len(lineage),candidate_clocks=len(candidates),saved_trade_rows=len(trades),elapsed_seconds=result['elapsed_seconds']))
    print(json.dumps(clean({k:result[k] for k in ['status','days','candidate_clocks','saved_trade_rows','time_weighted_all_hours','time_weighted_policy_session','candidate_clock_distributions','max_net_identity_residual_per_oz','elapsed_seconds']}),indent=2))


if __name__=='__main__':
    main()
