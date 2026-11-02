"""Reconstruct native target paths, model probabilities, selection and accounts."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.mechanical_lab.data import ROOT,digest
from experiments.loser_lab_v1.verify import need,verify_paths,verify_fills,model_probability,verify_accounts
from .prepare import check,dump
from .engine import joined,contexts


def predictions_check(pred,features,targets,out,config):
    count=0
    for target,ys in targets.groupby('target'):
        f,_=contexts(joined(features,ys),config); models=json.loads((out/(target+'_models.json')).read_text())
        cache=json.loads((out/(target+'_cache.json')).read_text())
        check(out/(target+'_models.json'),cache['models']); check(out/(target+'_predictions.parquet'),cache['predictions'])
        for key,m in models.items():
            need(not any(any(x in c for x in ('equity','pnl','future','class_index','event_sign','h_')) for c in m['columns']),'Forbidden model input')
        for date,g in pred[pred.target.eq(target)].groupby('date'):
            x=f[f.date.eq(date)]; cutoff=int(pd.Timestamp(date,tz='UTC').value); probabilities={}
            for key in ('frozen','full2','full5','full10','no_flow','no_memory','coarse','multiscale'):
                modelkey='frozen' if key=='frozen' else date+'_'+key
                if modelkey not in models: continue
                receipt=models[modelkey]; need(receipt['max_release_ns']<cutoff,'Future model labels')
                if 'train_dates' in receipt: need(max(receipt['train_dates'])<date,'Future date in fit')
                probabilities[key]=model_probability(x,receipt)
            probabilities['blend2']=.5*(probabilities['frozen']+probabilities['full2'])
            for policy,actual in g.groupby('policy'):
                match=re.fullmatch(r'(fade|switch)(2|5|10)_g(10|20)',policy)
                if match:
                    mode=match[1]; key='full'+match[2]; gate=int(match[3])/100
                elif policy in ('blend2_fade','frozen_fade','clean5','no_flow5','no_memory5','coarse5','multiscale5'):
                    key={'blend2_fade':'blend2','frozen_fade':'frozen','clean5':'full5'}.get(policy,policy[:-1]); mode='clean' if policy=='clean5' else 'fade'; gate=.1
                else: continue
                actual=actual.set_index('candidate_id').loc[x.candidate_id]; p=probabilities[key]
                physical=p.copy(); neg=x.momentum_side.to_numpy()==-1; physical[neg]=p[neg][:,[0,3,4,1,2]]
                need(np.allclose(physical,actual[['p'+str(i) for i in range(5)]],rtol=0,atol=2e-12),'Probability reconstruction')
                raw=(p[:,1] if mode=='clean' else p[:,1]+p[:,2])-p[:,3]-p[:,4]
                score=abs(raw) if mode=='switch' else raw
                sign=np.where(raw>=0,1,-1) if mode=='switch' else 1
                side=-x.momentum_side.to_numpy()*sign
                need(np.array_equal(side,actual.side),'Side reconstruction')
                need(np.array_equal(x.impulse_known.to_numpy()&(score>=gate),actual.accepted),'Gate reconstruction')
                count+=len(actual)
    need((pred.max_feedback_ns<pred.selection_ns).all() and (pred.selection_ns<pred.t_ns).all(),'Prediction availability')
    return count


def selectors_check(pred,targets,out,config):
    d=pred.merge(targets[['candidate_id','target','known','event_sign','release_ns']],on=['candidate_id','target'],validate='many_to_one')
    d['reward']=np.where(d.accepted,-d.side*d.event_sign,0.)
    gr=pd.read_parquet(out/'guard_receipts.parquet'); sr=pd.read_parquet(out/'selection_receipts.parquet')
    gc=config['guard']; sc=config['selector']
    for r in gr.itertuples():
        p=d[d.target.eq(r.target)&d.policy.eq(r.policy.removeprefix('guard_'))]; dates=sorted(p.date.unique()); ix=dates.index(r.date)
        prior=dates[max(0,ix-gc['history_days']):ix]
        h=p[p.date.isin(prior)&p.accepted&p.known&p.release_ns.lt(r.cutoff_ns)&p.segment.eq(r.segment)&p.cell.eq(r.cell)]
        daily=h.groupby('date').reward.mean(); mean=daily.mean() if len(daily) else 0.; se=daily.std(ddof=1)/np.sqrt(len(daily)) if len(daily)>1 else 1.
        lower=mean*len(h)/(len(h)+gc['prior_rows'])-gc['day_se_penalty']*se
        licence=len(h)>=gc['minimum_rows'] and len(daily)>=gc['minimum_days'] and lower>0
        need(r.n==len(h) and r.days==len(daily) and abs(r.lower-lower)<1e-12 and r.licence==licence,'Licence reconstruction')
        now=p[p.date.eq(r.date)&p.cell.eq(r.cell)].set_index('candidate_id')
        actual=d[d.target.eq(r.target)&d.policy.eq(r.policy)&d.date.eq(r.date)&d.cell.eq(r.cell)].set_index('candidate_id').loc[now.index]
        need(np.array_equal(now.accepted&licence,actual.accepted),'Licence gate')
    for (target,week),group in sr.groupby(['target','week']):
        ranks=[]
        for r in group.itertuples():
            p=d[d.target.eq(target)&d.policy.eq(r.policy)]; dates=sorted(p.date.unique()); date=pd.Timestamp(r.cutoff_ns,tz='UTC').strftime('%Y-%m-%d'); ix=dates.index(date)
            past=dates[max(0,ix-sc['history_days']):ix]; h=p[p.date.isin(past)&p.known&p.release_ns.lt(r.cutoff_ns)]
            daily=h.groupby('date').reward.mean(); mean=daily.mean() if len(daily) else 0.; se=daily.std(ddof=1)/np.sqrt(len(daily)) if len(daily)>1 else 1.
            score=mean-sc['day_se_penalty']*se; active=h[h.accepted]
            support=len(past)==sc['history_days'] and len(active)>=sc['minimum_rows'] and active.date.nunique()>=sc['minimum_active_days']
            need(abs(score-r.score)<1e-12 and support==r.supported,'Weekly rank reconstruction')
            if support and score>0: ranks.append((score,r.policy))
        winner=sorted(ranks,key=lambda x:(-x[0],x[1]))[0][1] if ranks else 'WAIT'
        need(group.selection.eq(winner).all(),'Weekly winner identity')
        actual=d[d.target.eq(target)&d.policy.eq('weekly_selector')&(pd.to_datetime(d.date).dt.to_period('W-SUN').astype(str)==week)]
        need(actual.selected_recipe.eq(winner).all(),'Weekly deployment recipe')
        if winner=='WAIT': need(not actual.accepted.any(),'WAIT traded')
        else:
            base=d[d.target.eq(target)&d.policy.eq(winner)].set_index('candidate_id').loc[actual.candidate_id]
            need(np.array_equal(base.side,actual.side) and np.array_equal(base.accepted,actual.accepted),'Selected policy changed')
    return len(gr),len(sr)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); args=ap.parse_args(); out=ROOT/args.run
    m=json.loads((out/'manifest.json').read_text()); config=json.loads((out/'config.json').read_text()); source=Path(m['source'])
    check(source/'manifest.json',m['source_manifest_sha256']); im=json.loads((source/'manifest.json').read_text())
    for name,sha in m['artifacts'].items(): check(out/name,sha)
    for name,sha in im['artifacts'].items(): check(source/name,sha)
    f=pd.read_parquet(source/'features.parquet'); y=pd.read_parquet(source/'targets.parquet'); post=pd.read_parquet(source/'post_cost.parquet')
    pred=pd.read_parquet(out/'predictions.parquet'); need(not f.candidate_id.duplicated().any(),'Origin collision')
    for date,g in f.groupby('date'):
        need((np.diff(np.sort(g.t_ns))>=300*10**9).all(),'Overlapping origin horizon')
        for col in ['bar_context_available_ns','memory_available_ns','spectrum_available_ns','past_max_available_ns','source_selection_ns']:
            need((g[col]<=g.t_ns).all(),f'Future causal input {col}')
    paths=0; fills=0; hashes=0
    for r in im['receipts']:
        for name,sha in r['source_hashes'].items(): check(Path(name),sha); hashes+=1
        q=pd.read_parquet(Path(r['source'])/'quotes.parquet'); day=y[y.date.eq(r['date'])]
        paths+=verify_paths(q,day,r['coverage_end_ns'],config); fills+=verify_fills(q,day,post[post.date.eq(r['date'])],r['coverage_end_ns'],config)
    count=predictions_check(pred,f,y,out,config); licences,choices=selectors_check(pred,y,out,config)
    ac=verify_accounts(pd.read_parquet(out/'accounts.parquet'),pd.read_parquet(out/'trades.parquet'),pred,y,post,config)
    original=pd.read_parquet(ROOT/config['old_loser']/'targets.parquet'); now=y[y.candidate_id.isin(original.candidate_id)&y.target.isin(original.target)]
    cols=list(original.columns); order=['target','candidate_id']
    pd.testing.assert_frame_equal(original[cols].sort_values(order).reset_index(drop=True),now[cols].sort_values(order).reset_index(drop=True),check_exact=True)
    dump(out/'verification.json',dict(status='PASS',native_paths=paths,post_cost_rows=fills,reconstructed_probability_rows=count,
         licence_records=licences,weekly_rank_records=choices,accounts=ac,source_hashes=hashes,old_target_rows_identical=len(original),
         limits=['Causal feature builders reused; native target paths, probability coefficients, selection arithmetic and account replay reconstructed independently.',
                 'Finite exploratory comparisons and descriptive day bootstrap, no untouched holdout or persistent-market-edge claim.']))
    print(dict(paths=paths,fills=fills,probabilities=count,licences=licences,weekly_ranks=choices,accounts=ac))


if __name__=='__main__': main()
