"""Independent streaming arithmetic for the post-result intraday stage."""
import argparse
import heapq
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.mechanical_lab.data import ROOT,digest
from experiments.loser_lab_v1.verify import need,verify_accounts
from .prepare import check,dump
from .engine import joined,contexts


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); args=ap.parse_args(); out=ROOT/args.run
    m=json.loads((out/'manifest.json').read_text()); config=json.loads((out/'config.json').read_text()); source=Path(m['source']); parent=Path(m['parent'])
    check(source/'manifest.json',m['source_manifest_sha256']); check(parent/'manifest.json',m['parent_manifest_sha256'])
    for name,sha in m['artifacts'].items(): check(out/name,sha)
    f=pd.read_parquet(source/'features.parquet'); y=pd.read_parquet(source/'targets.parquet'); y=y[y.target.eq('usd2_300')]
    post=pd.read_parquet(source/'post_cost.parquet'); post=post[post.target.eq('usd2_300')]
    f,_=contexts(joined(f,y),config); f=f.set_index('candidate_id'); pred=pd.read_parquet(out/'predictions.parquet')
    baseline=pd.read_parquet(parent/'predictions.parquet'); baseline=baseline[baseline.target.eq('usd2_300')&baseline.policy.eq('switch5_g10')].sort_values('t_ns')
    checked=0
    for policy,half_life,local in config['online_arms']:
        actual=pred[pred.policy.eq(policy)].set_index('candidate_id'); segment=None; pending=[]; observed={}; expected={}; times={}; total=0
        max_release=-1; max_origin=-1
        def decay(key,t):
            if key not in times: observed[key]=np.zeros(3); expected[key]=np.zeros(3); times[key]=t
            factor=2**(-(t-times[key])/(half_life*10**9)); observed[key]*=factor; expected[key]*=factor; times[key]=t
        for r in baseline.itertuples():
            x=f.loc[r.candidate_id]; a=actual.loc[r.candidate_id]
            if segment!=int(x.segment):
                segment=int(x.segment); pending=[]; observed={}; expected={}; times={}; total=0; max_release=-1; max_origin=-1
            while pending and pending[0][0]<r.t_ns:
                release,cid,t,prior,label,cell=heapq.heappop(pending)
                if label<0: continue
                for key in ['global',cell]: decay(key,release); observed[key][label]+=1; expected[key]+=prior
                total+=1; max_release=max(max_release,release); max_origin=max(max_origin,t)
            p=np.array([getattr(r,'p'+str(j)) for j in range(5)]); up=p[1]+p[2]; down=p[3]+p[4]; momentum=int(x.momentum_side)
            prior=np.array([p[0],up if momentum==1 else down,down if momentum==1 else up]); factors=[]
            for key in ['global']+([x.cell] if local else []):
                decay(key,r.t_ns); factors.append((observed[key]+20/3)/(expected[key]+20/3))
            q=prior*np.mean(factors,axis=0); q/=q.sum(); raw=q[1]-q[2]; side=-momentum*(1 if raw>=0 else -1)
            accept=bool(x.impulse_known and observed['global'].sum()>=12 and abs(raw)>=.1)
            need(abs(a.score-abs(raw))<1e-12 and a.side==side and a.accepted==accept,'Online decision reconstruction')
            need(a.feedback_count==total and a.max_feedback_release_ns==max_release and max_release<r.t_ns,'Online maturity')
            need(abs(a.global_effective_support-observed['global'].sum())<1e-11,'Online support')
            if x.impulse_known:
                label=0 if x.event_sign==0 else 1 if x.event_sign==momentum else 2
                heapq.heappush(pending,(int(x.release_ns),int(r.candidate_id),r.t_ns,prior,label if x.known else -1,x.cell))
            checked+=1
    ac=verify_accounts(pd.read_parquet(out/'accounts.parquet'),pd.read_parquet(out/'trades.parquet'),pred,y,post,config)
    dump(out/'verification.json',dict(status='PASS',independently_reconstructed_stream_decisions=checked,accounts=ac,common_feedback_maturity=True,
         limits='Second exploratory stage selected after primary results; causal arithmetic does not establish persistent prediction skill.'))
    print(dict(online_decisions=checked,accounts=ac))


if __name__=='__main__': main()
