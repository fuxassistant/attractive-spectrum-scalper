"""Post-replay cost attribution at the SAME fills; never a zero-cost backtest."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.hypothesis_atlas_v1.run import checked,digest,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('out');args=ap.parse_args();out=Path(args.out)
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    for name in ('trades.parquet','base_outcomes.parquet','stress_outcomes.parquet'):
        checked(out/name,manifest['artifacts'][name])
    trades=pd.read_parquet(out/'trades.parquet');trades=trades[trades.policy.eq('raw_leader')]
    outcomes={k:pd.read_parquet(out/(k+'_outcomes.parquet')).set_index(['candidate_id','side']) for k in ('base','stress')}
    source=Path(config['execution_source']);em=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    receipts={r['date']:r for r in em['receipts']};rows=[];hashes={}
    for date,g in trades.groupby('date',sort=True):
        qp=Path(receipts[date]['source'])/'quotes.parquet';hashes[str(qp)]=checked(qp,receipts[date]['source_payloads']['quotes.parquet'])
        q=pd.read_parquet(qp,columns=['available_ns','bid_nano','ask_nano'])
        for r in g.itertuples():
            o=outcomes[r.condition].loc[(r.candidate_id,r.side)]
            en=q.iloc[int(o.entry_ordinal)];ex=q.iloc[int(o.exit_ordinal)]
            if int(en.available_ns)!=r.entry_ns or int(ex.available_ns)!=r.exit_ns:raise AssertionError('Fill identity drift')
            scale=r.qty*config['account']['contract_oz']
            mid=r.side*((int(ex.bid_nano)+int(ex.ask_nano))-(int(en.bid_nano)+int(en.ask_nano)))/(2e9)*scale
            spread=((int(en.ask_nano)-int(en.bid_nano))+(int(ex.ask_nano)-int(ex.bid_nano)))/(2e9)*scale
            fee=config['execution']['fee_roundtrip_per_oz']*scale
            slip=(config['stress']['slippage_ticks_per_fill'] if r.condition=='stress' else config['execution']['slippage_ticks_per_fill'])*.1*2*scale
            if not np.isclose(mid-spread-fee-slip,r.pnl_usd,rtol=0,atol=1e-7):raise AssertionError('Cost attribution identity failed')
            rows.append(dict(candidate_id=r.candidate_id,date=date,condition=r.condition,mid_move_usd=mid,
                             spread_usd=spread,commission_usd=fee,slippage_usd=slip,net_usd=r.pnl_usd))
    f=pd.DataFrame(rows);f.to_parquet(out/'cost_attribution.parquet',index=False)
    totals=f.groupby('condition')[['mid_move_usd','spread_usd','commission_usd','slippage_usd','net_usd']].sum()
    dump(out/'diagnostics.json',dict(scope='Post-replay diagnostics added after all-WAIT primary result. Same realized fills, exit timing, account halts and selected origins; not a no-cost counterfactual. Midpoint component is descriptive, not executable profit.',
         policy='raw_leader',cost_attribution=totals.reset_index().to_dict('records'),trades=len(f),
         source_hashes=hashes,artifact_sha256=digest(out/'cost_attribution.parquet'),implementation_sha256=digest(__file__)))
    print(totals.round(2).to_string())


if __name__=='__main__':main()
