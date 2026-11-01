"""Additional controls after the first run; no refit or change to selection."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.hypothesis_atlas_v1.run import dump,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();out=Path(a.run)
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'));targets=pd.read_parquet(out/'targets.parquet')
    decisions=pd.read_parquet(out/'decisions.parquet');atlas=pd.read_parquet(Path(cfg['atlas'])/'observations.parquet').set_index('candidate_id')
    rows=[];daily=[]
    for (target,policy),part in decisions.groupby(['target','policy']):
        y=targets[targets.target.eq(target)];dates=sorted(y.date.unique())
        for date,g in part[part.accepted&part.known].groupby('date'):
            before=dates[dates.index(date)-2:dates.index(date)];cutoff=pd.Timestamp(date,tz='UTC').value
            history=y[y.date.isin(before)&y.known&y.release_ns.lt(cutoff)]
            majority=-1 if history.event_sign.sum()>=0 else 1
            pastmove=atlas.loc[g.candidate_id,'past_net30_r'].to_numpy()*atlas.loc[g.candidate_id,'side'].to_numpy()
            candidates={'model':g.side.to_numpy(),'constant_long':np.ones(len(g)), 'constant_short':-np.ones(len(g)),
                        'prior_pair_majority_loser':np.full(len(g),majority),'momentum':np.where(pastmove>=0,1,-1),'anti_momentum':np.where(pastmove>=0,-1,1)}
            for name,side in candidates.items():
                reward=-side*g.event_sign.to_numpy()
                daily.append(dict(target=target,policy=policy,date=date,block=g.iloc[0].block,control=name,n=len(g),
                                  reward=float(reward.mean()),mean_bid_per_oz=float((side*g.bid_delta_per_oz.to_numpy()).mean()),
                                  feedback_before_ns=int(history.release_ns.max())))
    d=pd.DataFrame(daily)
    for (target,policy,block,control),g in d.groupby(['target','policy','block','control']):
        rows.append(dict(target=target,policy=policy,block=block,control=control,day_reward=float(g.reward.mean()),
                         weighted_reward=float(np.average(g.reward,weights=g.n)),n=int(g.n.sum()),days=len(g)))
    d.to_parquet(out/'matched_daily.parquet',index=False);pd.DataFrame(rows).to_parquet(out/'matched_controls.parquet',index=False)
    random=pd.read_parquet(out/'random_accounts.parquet');risk=[]
    for (policy,style),g in random.groupby(['policy','style']):
        risk.append(dict(policy=policy,style=style,n=len(g),depleted80_rate=float(g.depleted80.mean()),bankrupt_rate=float(g.bankrupt.mean()),
                         median_cash=float(g.terminal_usd.median()),p05_cash=float(g.terminal_usd.quantile(.05)),p95_cash=float(g.terminal_usd.quantile(.95))))
    dump(out/'diagnostics.json',dict(status='POSTHOC_DIAGNOSTICS_NO_MODEL_CHANGES',matched_controls=rows,random_risk=risk,
         interpretation='Same selected origins for all directional controls; prior-pair-majority uses only mature historical signs. These extra matched controls were added after the first run; they do not refit or alter any model. Random account tests were specified before the run.',
         implementation_sha256=digest(__file__)))
    print(pd.DataFrame(rows).query('target == @cfg["primary_target"] and policy == @cfg["primary_policy"]').to_string(index=False))


if __name__=='__main__':main()
