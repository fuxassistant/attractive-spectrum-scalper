"""Independent vector path checks and scalar paper-account reconstruction."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.hypothesis_atlas_v1.run import digest,checked,dump

NS=1_000_000_000
ROOT=Path(__file__).resolve().parents[2]


def need(v,msg):
    if not v:raise AssertionError(msg)


def verify_paths(q,y,coverage,config):
    qt=q.available_ns.to_numpy(np.int64);bid=q.bid_nano.to_numpy(np.int64);epoch=q.epoch.to_numpy(np.int64)
    good=q.valid.to_numpy(bool)&(epoch>=0)&(bid>0)&(q.ask_nano.to_numpy()>bid)
    gap=int(config['post_cost']['quote_gap_seconds']*NS)
    for r in y.itertuples():
        a=int(np.searchsorted(qt,r.t_ns,'right')-1);b=int(np.searchsorted(qt,r.horizon_ns,'right'))
        valid=a>=0 and good[a] and r.t_ns-qt[a]<=gap
        if not valid:need(not r.known,'Unknown origin labelled known');continue
        delta=bid[a:b]-bid[a];threshold=round(r.threshold_per_oz*NS)
        hits=np.flatnonzero(np.abs(delta)>=threshold);j=int(hits[0]) if len(hits) else len(delta)-1
        path_good=good[a:a+j+1].all() and (epoch[a:a+j+1]==epoch[a]).all() and (np.diff(qt[a:a+j+1])<=gap).all()
        valid=path_good and (bool(len(hits)) or (r.horizon_ns<coverage and r.horizon_ns-qt[a+j]<=gap))
        need(bool(r.known)==bool(valid),f'Path support {r.candidate_id}/{r.target}')
        if not valid:continue
        sign=int(np.sign(delta[j])) if len(hits) else 0
        low=min(0,int(delta[:j+1].min()));high=max(0,int(delta[:j+1].max()))
        clean=(-low if sign==1 else high)<=config['clean_loss_max_favorable_x']*threshold+1e-9
        cls=0 if sign==0 else (1 if clean else 2) if sign==1 else (3 if clean else 4)
        need(r.event_sign==sign and r.class_index==cls,'Wrong directional/clean label')
        need(r.origin_ordinal==a and r.exit_ordinal==a+j,'Wrong quote identity')
        exit_ns=int(qt[a+j]) if len(hits) else int(r.horizon_ns)
        need(r.exit_ns==exit_ns and r.release_ns==r.horizon_ns,'Wrong exit or maturity')
        need(abs(r.bid_delta_per_oz-int(delta[j])/NS)<1e-12,'Wrong bid outcome')
        need(abs(r.bid_low_per_oz-low/NS)<1e-12 and abs(r.bid_high_per_oz-high/NS)<1e-12,'Wrong path extrema')
    return len(y)


def model_probability(f,receipt):
    x=f[receipt['columns']].to_numpy(float);missing=~np.isfinite(x)
    z=np.c_[np.where(missing,np.asarray(receipt['median']),x),missing]
    z=np.clip((z-np.asarray(receipt['mean']))/np.asarray(receipt['scale']),-8,8)
    prior=np.asarray(receipt['prior']);classes=receipt['classes']
    if not len(classes):return np.tile(prior,(len(f),1))
    logits=z@np.asarray(receipt['coef']).T+np.asarray(receipt['intercept'])
    if len(classes)==2:
        pos=1/(1+np.exp(-logits[:,0]));partial=np.c_[1-pos,pos]
    else:
        ex=np.exp(logits-logits.max(axis=1,keepdims=True));partial=ex/ex.sum(axis=1,keepdims=True)
    p=np.zeros((len(f),5));p[:,classes]=partial
    return receipt['n']/(receipt['n']+5)*p+5/(receipt['n']+5)*prior


def verify_fills(q,y,post,coverage,config):
    qt=q.available_ns.to_numpy(np.int64);bid=q.bid_nano.to_numpy(np.int64);ask=q.ask_nano.to_numpy(np.int64);ep=q.epoch.to_numpy(np.int64)
    good=q.valid.to_numpy(bool)&(ep>=0)&(bid>0)&(ask>bid);gap=int(config['post_cost']['quote_gap_seconds']*NS)
    bad=np.r_[0,np.cumsum(~good)];broken=np.r_[0,np.cumsum(np.r_[False,(np.diff(qt)>gap)|(np.diff(ep)!=0)])]
    lookup=y.set_index(['target','candidate_id']);pending=int(config['post_cost']['pending_seconds']*NS)
    for r in post.itertuples():
        t=lookup.loc[(r.target,r.candidate_id)];ec=config['post_cost'][r.mode];lat=int(ec['latency_ms']*1e6)
        a=int(np.searchsorted(qt,t.t_ns,'right')-1);en=int(np.searchsorted(qt,t.t_ns+lat,'right'))
        valid=bool(t.known) and a>=0 and en<len(qt) and qt[en]<=t.t_ns+pending
        if valid:
            request=max(int(t.exit_ns),int(qt[en]));ex=int(np.searchsorted(qt,request+lat,'right'))
            valid=ex<len(qt) and qt[ex]<coverage and qt[ex]<=request+pending
            if valid:valid=bad[ex+1]==bad[a] and broken[ex+1]==broken[a+1]
        need(bool(r.known)==bool(valid),'Post-cost source support')
        if valid:
            need(r.entry_ns==qt[en] and r.exit_ns==qt[ex],'Post-cost exact clock')
            entry=int(ask[en] if r.side==1 else bid[en]);exitp=int(bid[ex] if r.side==1 else ask[ex])
            expected=r.side*(exitp-entry)/NS-ec['fee_per_oz']-2*.1*ec['slippage_ticks_per_fill']
            need(abs(expected-r.net_per_oz)<1e-12,'Post-cost price identity')
    return len(post)


def verify_predictions(pred,source,models,config):
    checked_rows=0
    for (target,date),g in pred.groupby(['target','date']):
        x=source.set_index('candidate_id').loc[g[g.policy=='frozen_context'].candidate_id].reset_index()
        cutoff=int(pd.Timestamp(date,tz='UTC').value);receipts=models[target]
        for key,r in receipts.items():
            need(not any('equity' in c or 'future' in c or 'pnl' in c for c in r['columns']),'Forbidden learning feature')
        parts={}
        for key in ('frozen_geometry','frozen_context',date+'_pair',date+'_expanding'):
            need(receipts[key]['max_release_ns']<cutoff,'Unmatured model teacher')
            parts[key]=model_probability(x,receipts[key])
        w=config['pair_blend_weight'];blend=(1-w)*parts['frozen_context']+w*parts[date+'_pair']
        expected={'frozen_geometry':parts['frozen_geometry'],'frozen_context':parts['frozen_context'],
                  'pair_context':parts[date+'_pair'],'expanding_context':parts[date+'_expanding'],'blend_context':blend,'clean_blend':blend}
        for policy,p in expected.items():
            actual=g[g.policy==policy].set_index('candidate_id').loc[x.candidate_id]
            need(np.allclose(p,actual[['p'+str(j) for j in range(5)]],rtol=0,atol=2e-12),'Model probability reconstruction')
            up=p[:,1]+p[:,2];down=p[:,3]+p[:,4]
            ls=(p[:,3] if policy=='clean_blend' else down)-up;ss=(p[:,1] if policy=='clean_blend' else up)-down
            side=np.where(ls>ss,1,-1);score=np.maximum(ls,ss)
            need(np.array_equal(side,actual.side),'Wrong chosen side')
            need(np.array_equal(score>=config['confidence_gate'],actual.accepted),'Wrong confidence gate')
            checked_rows+=len(actual)
        need((g.selection_ns==cutoff).all() and (g.max_feedback_ns<g.selection_ns).all() and (g.selection_ns<g.t_ns).all(),'Selection clock')
    return checked_rows


def verify_accounts(accounts,trades,pred,targets,post,config):
    cfg=config['risk'];tlookup=targets.set_index(['target','candidate_id']);plookup=post.set_index(['target','candidate_id','mode','side'])
    for row in accounts.itertuples():
        d=pred[(pred.target==row.target)&(pred.policy==row.policy)].sort_values(['t_ns','candidate_id'])
        expected=[];cash=cfg['initial_deposit_usd'];size=1;busy=-1;unknown=0;overlap=0;unaffordable=0
        for r in d.itertuples():
            if not r.accepted:continue
            if r.t_ns<=busy:overlap+=1;continue
            t=tlookup.loc[(row.target,r.candidate_id)];planned=t.threshold_per_oz*cfg['contract_oz']*size
            if cash<planned:unaffordable+=1;continue
            side=int(r.side)*(-1 if row.reverse else 1)
            if not t.known:unknown+=1;break
            unit=side*t.bid_delta_per_oz;exit_ns=int(t.exit_ns)
            if row.mode!='bid':
                p=plookup.loc[(row.target,r.candidate_id,row.mode,side)]
                if not p.known:unknown+=1;break
                unit=p.net_per_oz;exit_ns=int(p.exit_ns)
            pnl=unit*cfg['contract_oz']*size;cash+=pnl;busy=exit_ns
            expected.append((r.candidate_id,size,pnl,cash,exit_ns))
            if row.style=='after_loss':size=min(cfg['max_multiplier'],2*size) if pnl<0 else 1
            elif row.style=='after_gain':size=min(cfg['max_multiplier'],2*size) if pnl>0 else 1
            if cash<=0:break
        actual=trades[(trades.target==row.target)&(trades.policy==row.policy)&(trades['mode']==row.mode)&(trades['style']==row.style)&(trades.reverse==row.reverse)]
        need(len(actual)==len(expected)==row.trades,'Account trade count')
        if expected:
            for j,col in ((0,'candidate_id'),(1,'multiplier'),(4,'exit_ns')):
                need(np.array_equal(actual[col].to_numpy(np.int64),np.array([v[j] for v in expected],dtype=np.int64)),'Exact account identity/clock')
            need(np.allclose(actual[['pnl_usd','equity_usd']].to_numpy(float),np.array([[v[2],v[3]] for v in expected]),rtol=0,atol=1e-7),'Independent account prices')
        need(abs(row.terminal_usd-cash)<1e-7 and row.unknown==unknown and row.complete==(unknown==0),'Account result')
        need(row.overlap==overlap and row.unaffordable==unaffordable,'Account constraints')
    return len(accounts)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();out=ROOT/a.run
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'));config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    for name,sha in manifest['artifacts'].items():checked(out/name,sha)
    for name,sha in manifest['source_hashes'].items():checked(name,sha)
    for name,sha in manifest['implementation_sha256'].items():checked(ROOT/name,sha)
    targets=pd.read_parquet(out/'targets.parquet');post=pd.read_parquet(out/'post_cost.parquet');pred=pd.read_parquet(out/'predictions.parquet')
    source=pd.read_parquet(ROOT/config['atlas']/'observations.parquet');receipt=json.loads((ROOT/config['market_manifest']).read_text(encoding='utf-8'))
    markets={r['date']:Path(r['source_market']) for r in receipt['receipts']};paths=0;fills=0
    for date,y in targets.groupby('date'):
        market=markets[date];q=pd.read_parquet(market/'quotes.parquet');mm=json.loads((market/'manifest.json').read_text(encoding='utf-8'))
        paths+=verify_paths(q,y,int(mm['input']['end_ns'])+1,config)
        fills+=verify_fills(q,y,post[post.date.eq(date)],int(mm['input']['end_ns'])+1,config)
    models=json.loads((out/'models.json').read_text(encoding='utf-8'));probas=verify_predictions(pred,source,models,config)
    ac=verify_accounts(pd.read_parquet(out/'accounts.parquet'),pd.read_parquet(out/'trades.parquet'),pred,targets,post,config)
    dump(out/'verification.json',dict(status='PASS_INDEPENDENT_PATH_MODEL_ACCOUNT_CHECK',native_paths=paths,post_cost_rows=fills,reconstructed_probabilities=probas,
         accounts=ac,source_hashes=len(manifest['source_hashes']),artifact_hashes=len(manifest['artifacts']),verifier_sha256=digest(__file__),
         limits=['Inherited X feature construction is not independently rebuilt.','Native bid targets, post-cost fill identities/support, and deterministic accounts are independently reconstructed.','Random controls and exploration on previously inspected dates are descriptive, not proof of a persistent loser.']))
    print(dict(native_paths=paths,probabilities=probas,accounts=ac))


if __name__=='__main__':main()
