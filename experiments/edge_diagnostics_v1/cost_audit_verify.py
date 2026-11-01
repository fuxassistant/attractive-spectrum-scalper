"""Independent prefix-integral and direct-BBO audit of saved cost attribution."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_results/edge_diagnostics_v1/cost_audit_v1'
NS=10**9


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def integral(t, duration, prefix_time, prefix_value, spread, at):
    j=int(np.searchsorted(t,at,'right')-1)
    if j<0:return 0,0.
    partial=min(max(0,int(at)-int(t[j])),int(duration[j]))
    return int(prefix_time[j])+partial,float(prefix_value[j])+partial*float(spread[j])


def main():
    started=time.perf_counter();mp=OUT/'manifest.json';manifest_hash=sha(mp)
    manifest=json.loads(mp.read_text(encoding='utf-8'));assert manifest['status']=='COMPLETE_DIAGNOSTIC_ONLY'
    hashes=0
    for path,expected in manifest['source_inputs'].items():assert sha(Path(path))==expected;hashes+=1
    for name,expected in manifest['artifacts'].items():assert sha(OUT/name)==expected;hashes+=1
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    trades=pd.read_parquet(OUT/'trade_cost_components.parquet');candidates=pd.read_parquet(OUT/'candidate_clock_costs.parquet')
    hourly=pd.read_parquet(OUT/'hourly_spread.parquet');groups=pd.read_parquet(OUT/'account_cost_summary.parquet')
    zone=ROOT/'research_results/zone_episode_lab/verified_v3';zm=json.loads((zone/'manifest.json').read_text())
    # The run stores a pretty-printed copy; its manifest hashes the original
    # source config bytes, so validate source bytes and copied JSON semantics.
    source_config=ROOT/'experiments/zone_episode_lab/config.json'
    assert sha(source_config)==zm['config_sha256']
    config=json.loads((zone/'config.json').read_text())
    assert config==json.loads(source_config.read_text());hashes+=1
    fee=config['execution']['fee_roundtrip_per_oz'];gap=int(config['execution']['max_quote_gap_seconds']*NS)
    max_mean=0.;max_net=0.;max_component=0.;hour_checks=0;trade_checks=0;candidate_checks=0
    active_time=0;active_cost=0.;rows=[]
    for lineage in summary['lineage']:
        date=lineage['date'];source=Path(lineage['canonical_manifest']);assert sha(source)==lineage['canonical_manifest_sha256'];hashes+=1
        mm=json.loads(source.read_text());qp=source.parent/'quotes.parquet';assert sha(qp)==lineage['quotes_sha256'];hashes+=1
        assert mm['input']['sha256']==lineage['raw_capture_sha256']==lineage['source_partition_sha256']
        q=pd.read_parquet(qp,columns=['available_ns','bid_nano','ask_nano','valid','epoch'])
        t=q.available_ns.to_numpy(np.int64);bid=q.bid_nano.to_numpy(np.int64);ask=q.ask_nano.to_numpy(np.int64)
        good=q.valid.to_numpy(bool)&q.epoch.ge(0).to_numpy()&(bid>0)&(ask>bid)
        spread=(ask-bid)/NS;coverage=int(mm['input']['end_ns'])+1
        # Alternative to the producer's hour clipping/histograms: integrate all
        # native constant-spread segments once, then difference cumulative areas.
        duration=np.where(good,np.maximum(0,np.minimum(np.minimum(np.r_[t[1:],coverage],t+gap),coverage)-t),0)
        prefix_time=np.r_[0,np.cumsum(duration,dtype=np.int64)]
        prefix_value=np.r_[0.,np.cumsum(spread*duration)]
        day=int(pd.Timestamp(date,tz='UTC').value)
        for hour in range(24):
            a=integral(t,duration,prefix_time,prefix_value,spread,day+hour*3600*NS)
            b=integral(t,duration,prefix_time,prefix_value,spread,day+(hour+1)*3600*NS)
            dt=b[0]-a[0];saved=hourly.loc[hourly.date.eq(date)&hourly.hour_utc.eq(hour)].iloc[0]
            assert dt==saved.supported_ns
            if dt:
                residual=abs((b[1]-a[1])/dt-saved.mean_spread_per_oz);max_mean=max(max_mean,residual);assert residual<1e-8
            hour_checks+=1
        a=integral(t,duration,prefix_time,prefix_value,spread,day+7*3600*NS)
        b=integral(t,duration,prefix_time,prefix_value,spread,day+(20*3600+1800)*NS)
        active_time+=b[0]-a[0];active_cost+=b[1]-a[1]
        c=candidates.loc[candidates.date.eq(date)]
        ix=np.searchsorted(t,c.t_ns.to_numpy(np.int64),'right')-1
        assert np.array_equal(spread[ix],c.spread_per_oz.to_numpy())
        assert np.allclose(fee/c.risk_per_oz,c.fee_over_r,atol=1e-15,rtol=0)
        assert np.allclose((spread[ix]+fee)/c.risk_per_oz,c.decision_cost_proxy_over_r,atol=1e-15,rtol=0)
        candidate_checks+=len(c)
        for r in trades.loc[trades.date.eq(date)].itertuples(index=False):
            e=int(r.entry_ordinal);x=int(r.exit_ordinal);assert int(t[e])==r.entry_ns and int(t[x])==r.exit_ns
            slip=(config['stress']['slippage_ticks_per_fill'] if r.condition=='stress' else config['execution']['slippage_ticks_per_fill'])*.1
            raw_entry=int(ask[e] if r.side==1 else bid[e]);raw_exit=int(bid[x] if r.side==1 else ask[x])
            net=r.side*(raw_exit-raw_entry)/NS-fee-2*slip
            half_entry=(int(ask[e])-int(bid[e]))/(2*NS);half_exit=(int(ask[x])-int(bid[x]))/(2*NS)
            # Separate differences before summation, independent of producer's
            # difference of doubled midpoints.
            mid=r.side*((int(bid[x])-int(bid[e]))+(int(ask[x])-int(ask[e])))/(2*NS)
            max_net=max(max_net,abs(net-r.net_per_oz))
            max_component=max(max_component,abs(mid-r.native_mid_move_per_oz),abs(half_entry-r.entry_half_spread_per_oz),abs(half_exit-r.exit_half_spread_per_oz))
            assert abs(net-r.net_per_oz)<1e-12 and abs(mid-half_entry-half_exit-fee-2*slip-net)<1e-12
            rows.append(dict(block=r.block,arm=r.arm,condition=r.condition,
                mid=mid,entry_half_spread=half_entry,exit_half_spread=half_exit,
                fee=fee,slippage=2*slip,net=net,qty=r.qty))
            trade_checks+=1
    components=pd.DataFrame(rows);group_rows=[]
    for key,g in components.groupby(['block','arm','condition']):
        saved=groups.loc[groups.block.eq(key[0])&groups.arm.eq(key[1])&groups.condition.eq(key[2])].iloc[0]
        row=dict(block=key[0],arm=key[1],condition=key[2],trades=len(g),capital_curve_complete=bool(saved.capital_curve_complete),unresolved=int(saved.unresolved))
        for name in ['mid','entry_half_spread','exit_half_spread','fee','slippage','net']:
            row[name+'_mean_per_oz']=float(g[name].mean());row[name+'_usd']=float((g[name]*g.qty*100).sum())
        assert abs(row['net_usd']-saved.net_usd)<1e-8
        assert abs(row['mid_usd']-row['entry_half_spread_usd']-row['exit_half_spread_usd']-row['fee_usd']-row['slippage_usd']-row['net_usd'])<1e-8
        group_rows.append(row)
    assert active_time==summary['time_weighted_policy_session']['supported_ns']
    active_error=abs(active_cost/active_time-summary['time_weighted_policy_session']['mean_spread_per_oz']);assert active_error<1e-9
    group_path=OUT/'account_cost_summary_verified.parquet';pd.DataFrame(group_rows).to_parquet(group_path,index=False)
    result=dict(status='PASS',producer_manifest_sha256=manifest_hash,verifier_source_sha256=sha(Path(__file__)),
        verified_input_and_artifact_hashes=hashes,hourly_prefix_integral_checks=hour_checks,candidate_clock_checks=candidate_checks,
        fixed_trade_direct_bbo_checks=trade_checks,account_groups=len(group_rows),max_hourly_mean_error_per_oz=max_mean,
        active_window_mean_error_per_oz=active_error,max_net_error_per_oz=max_net,max_component_error_per_oz=max_component,
        independently_reconstructed_account_components=group_path.name,components_sha256=sha(group_path),
        caveats=['Recorded raw capture/member hashes are lineage; retired staging/raw archive was not newly rehashed.',
                 '07:00-20:00 UTC is candidate-entry window;07:00-20:30 is the execution horizon cutoff window.',
                 '21UTC is closure/rollover context, contains wide quotes and only10.58percent supported elapsed time; do not treat all-hours mean as executable-session cost.',
                 'Candidate population already has causal spread<=1.0USD/oz admission; clock costs are conditional on that fixed admission.',
                 'Every midpoint/no-fee add-back holds prior cost-conditioned fills and account selection fixed; not a zero-cost policy rerun.',
                 'Account groups overlap across models; incomplete accounts include only observed closed trades.'],
        elapsed_seconds=time.perf_counter()-started)
    assert sha(mp)==manifest_hash
    destination=OUT/'independent_reconstruction.json';destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(destination),sha256=sha(destination),**result),indent=2))


if __name__=='__main__':main()
