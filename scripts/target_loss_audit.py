"""Measure saved target/loss/gradient geometry without changing the v2 fit."""
from collections import defaultdict
import json
from pathlib import Path
import pickle
import subprocess
import time
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.metrics import roc_auc_score
from spectrum.cognition import flow_context as fc, episodes as ep
from spectrum.cognition.target_loss_audit import (loss_gradient_hessian, head_targets,
    leaf_ids, lag_record, combine_lags, run_lengths)
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from scripts.flow_context_coverage import save, sha

SOURCE=Path('data/derived/flow_context_v2')
DATA=Path('data/derived/target_loss_audit_v1')
REPORT=Path('reports/target_loss_audit_v1')
MARKET=Path('data/derived/basket_multimonth_v1')
CONTRACT=Path('contracts/target_loss_audit_v1.json')
CLASSES=['TAKE','STOP','TIME','EMPTY']
FAMILIES=['baseline','rich_flow','rich_spectrum','shuffled']


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def progress(**values): print(json.dumps(values),flush=True)


def minute_certificate(day,t):
    tape=fc.minute_tape(day); j=np.searchsorted(tape['t'],t,side='left'); jj=np.minimum(j,len(tape['t'])-1)
    # For audit support only: enclose each origin by a certified completed minute.
    # No field from this certificate is supplied to a model.
    valid=(j<len(tape['t'])) & (t>tape['t'][jj]-60*ep.NS) & tape['valid'][jj]
    return valid,tape['segment'][jj]


def load_rows():
    man=read(SOURCE/'manifest.json');done=read(Path('reports/flow_context_v2/completion_receipt.json'))
    if sha(SOURCE/'manifest.json')!=done['manifest_sha256']: raise ValueError('Source manifest changed')
    frames=[]
    for d in man['days']:
        p=SOURCE/(d['date']+'.parquet')
        if sha(p)!=d['sha256']: raise ValueError('Source day changed')
        f=pd.read_parquet(p);day=MonthDay(MARKET/d['date'][:7]/d['date'])
        f['audit_support'],f['audit_segment']=minute_certificate(day,f.t_ns.to_numpy(np.int64))
        frames.append(f)
    out=pd.concat(frames,ignore_index=True);out['split']=np.where(out.date.str[:7]<'2026-01','FIT','B')
    return out,man,done


def class_rows(frame):
    rows=[]
    for (month,mode),f in frame.groupby([frame.date.str[:7],'mode']):
        n=len(f)
        for k in [-1,0,1,2,3]:
            d=f.loc[f.category==k]
            rows.append(dict(month=month,mode=mode,category='UNKNOWN' if k<0 else CLASSES[k],n=len(d),fraction=len(d)/n,
                 busy_s_quantiles=np.quantile(d.busy_s.dropna(),[.1,.5,.9,.99]).tolist() if len(d.busy_s.dropna()) else None))
    return rows


def gradient_group_records(f,head,phase,ll,g,h):
    g=np.asarray(g).reshape(len(f),-1);h=np.asarray(h).reshape(len(f),-1)
    out=[];norm=np.abs(g).sum(axis=1)
    groups={'class':f.category.map(dict(enumerate(CLASSES))).to_numpy(),
            'month':f.date.str[:7].to_numpy(),
            'spread':pd.cut(f.spread_ticks,[0,2,4,8,np.inf],labels=['0-2','2-4','4-8','8+']).astype(str).to_numpy(),
            'busy_time':pd.cut(f.busy_s,[-np.inf,5,30,120,np.inf],labels=['<=5s','5-30s','30-120s','120s+']).astype(str).to_numpy()}
    for split in ('FIT','B'):
        s=f.split.to_numpy()==split
        for name,values in groups.items():
            for value in np.unique(values[s]):
                m=s&(values==value);n=int(m.sum())
                out.append(dict(mode=str(f['mode'].iloc[0]),head=head,phase=phase,split=split,group=name,value=str(value),n=n,
                    loss_sum=float(ll[m].sum()),loss_mean=float(ll[m].mean()),grad_abs_sum=float(norm[m].sum()),
                    grad_l2sq_sum=float((g[m]**2).sum()),grad_signed_sum=g[m].sum(axis=0).tolist(),
                    grad_abs_quantiles=np.quantile(norm[m],[.1,.5,.9,.99]).tolist(),hessian_sum=h[m].sum(axis=0).tolist()))
    return out


def score_record(f,pf,pc):
    m=f.category.to_numpy()>=0;y=f.category.to_numpy()[m];filled=y<3
    fillp=pf[m];pathp=pc[m][filled];cy=y[filled]
    return dict(n_known=int(m.sum()),n_filled=int(filled.sum()),
        fill_loss=float(-np.where(filled,np.log(fillp),np.log1p(-fillp)).mean()),
        path_loss=float(-np.log(pathp[np.arange(len(cy)),cy]).mean()),
        conditional_take_auc=float(roc_auc_score(cy==0,pathp[:,0])),
        conditional_argmax_counts=np.bincount(pathp.argmax(axis=1),minlength=3).tolist(),
        conditional_accuracy=float((pathp.argmax(axis=1)==cy).mean()),
        majority_stop_accuracy=float((cy==1).mean()),
        fill_quantiles=np.quantile(fillp,[0,.01,.1,.5,.9,.99,1]).tolist(),
        take_quantiles=np.quantile(pathp[:,0],[0,.01,.1,.5,.9,.99,1]).tolist(),
        fill_std=float(fillp.std()),take_std=float(pathp[:,0].std()),
        class_observed=np.bincount(cy,minlength=3).tolist(),class_probability_mean=pathp.mean(axis=0).tolist())


def leaf_checks(model,f,head):
    mask,y=head_targets(f.category,head);x=f.loc[mask,model['columns']].to_numpy(np.float32);obj=model[head]
    z=np.repeat(obj._baseline_prediction,len(x),axis=0);ll,g,h=loss_gradient_hessian(y,z)
    ng,nh=obj._loss.gradient_hessian(y.astype(float),z)
    g=np.asarray(g).reshape(len(x),-1);h=np.asarray(h).reshape(len(x),-1)
    records=[]
    for k,tree in enumerate(obj._predictors[0]):
        ids=leaf_ids(obj,tree,x)
        for leaf in np.unique(ids):
            m=ids==leaf;G=float(g[m,k].sum());H=float(h[m,k].sum());expected=-obj.learning_rate*G/(H+obj.l2_regularization)
            actual=float(tree.nodes[leaf]['value']);count=int(tree.nodes[leaf]['count'])
            if count!=int(m.sum()) or abs(actual-expected)>2e-6: raise ValueError('First tree Newton value mismatch')
            records.append(dict(head=head,class_index=k,leaf=int(leaf),n=count,
                gradient_sum=G,gradient_abs_sum=float(np.abs(g[m,k]).sum()),hessian_sum=H,
                cancellation_ratio=abs(G)/max(float(np.abs(g[m,k]).sum()),1e-12),
                expected_update=expected,saved_update=actual,error=abs(actual-expected),
                target_counts=np.bincount(y[m],minlength=2 if head=='fill' else 3).tolist()))
    return records,dict(head=head,n=len(x),gradient_max_error=float(np.max(np.abs(g-np.asarray(ng).reshape(g.shape)))),
                        hessian_max_error=float(np.max(np.abs(h-np.asarray(nh).reshape(h.shape)))))


def staged_curves(model,f):
    records=[]
    for head in ('fill','path'):
        obj=model[head]
        for split in ('FIT','B'):
            sub=f.loc[f.split==split];mask,y=head_targets(sub.category,head);sub=sub.loc[mask]
            ix=np.linspace(0,len(sub)-1,min(len(sub),24000)).astype(int);y=y[ix];x=sub.iloc[ix][model['columns']].to_numpy(np.float32)
            stages=[(0,np.repeat(obj._baseline_prediction,len(x),axis=0))]
            stages.extend((i,z) for i,z in enumerate(obj._staged_raw_predict(x),1) if i in [1,5,10,25,50,75,100])
            for it,z in stages:
                ll,g,h=loss_gradient_hessian(y,z)
                records.append(dict(head=head,split=split,iteration=it,n=len(x),loss=float(ll.mean()),
                    grad_abs_mean=float(np.abs(g).mean()),grad_sq_mean=float((g*g).mean()),
                    hessian_mean=float(h.mean()),raw_abs_max=float(np.abs(z).max())))
    return records


def predict_audit(all_rows):
    results={};gradients=[];derivatives=[];leaves=[];curves=[];frames={};hashes={}
    for mode in ('P','T'):
        f=all_rows.loc[all_rows['mode']==mode].copy().reset_index(drop=True)
        columns=['date','t_ns','mode','side','category','status','net','busy_s','spread_ticks','log_sigma','split','audit_support','audit_segment']
        out=f[columns].copy()
        for family in FAMILIES:
            p=SOURCE/f'model_B_{mode}_{family}.pkl';hashes[p.name]=sha(p);model=pickle.loads(p.read_bytes())
            x=f[model['columns']].to_numpy(np.float32);pf=None;pc=None
            for head in ('fill','path'):
                obj=model[head];raw=obj._raw_predict(x)
                prob=expit(raw[:,0]) if head=='fill' else softmax(raw,axis=1)
                if head=='fill':pf=prob;out['pf_'+family]=pf
                else:
                    pc=prob
                    for k in range(3):out[f'pc{k}_{family}']=pc[:,k]
                if family=='rich_spectrum':
                    mask,y=head_targets(f.category,head);ff=f.loc[mask];z=raw[mask]
                    for phase,zz in [('initial',np.repeat(obj._baseline_prediction,len(y),axis=0)),('final',z)]:
                        ll,g,h=loss_gradient_hessian(y,zz)
                        gradients+=gradient_group_records(ff,head,phase,ll,g,h)
                        if phase=='final':
                            out.loc[mask,'loss_'+head]=ll
                            out.loc[mask,'g_'+head]=g if head=='fill' else g[:,0]
                    if mode=='P':
                        lr,dr=leaf_checks(model,f.loc[f.split=='FIT'],head);leaves+=lr;derivatives.append(dr)
            for split in ('FIT','B'):
                m=f.split==split;results[f'{mode}/{family}/{split}']=score_record(f.loc[m],pf[m],pc[m])
            if family=='rich_spectrum':
                initial_pf=float(expit(model['fill']._baseline_prediction[0,0]));initial_pc=softmax(model['path']._baseline_prediction,axis=1)[0]
                for split in ('FIT','B'):
                    sub=f.loc[f.split==split];results[f'{mode}/prior/{split}']=score_record(sub,np.full(len(sub),initial_pf),np.tile(initial_pc,(len(sub),1)))
                if mode=='P':curves=staged_curves(model,f)
            progress(stage='predictions',mode=mode,family=family)
        out.to_parquet(DATA/f'frame_{mode}.parquet',index=False);frames[mode]=out
    return dict(scores=results,gradient_groups=gradients,derivative_parity=derivatives,first_tree_leaves=leaves,staged_curves=curves,model_sha256=hashes),frames


def temporal_diagnostics(frames):
    accumulator=defaultdict(list);runs=defaultdict(list);rng=np.random.default_rng(61023)
    for mode,f in frames.items():
        for (date,side),d in f.groupby(['date','side']):
            d=d.sort_values('t_ns');cat=d.category.to_numpy();known=cat>=0;filled=known&(cat<3)
            support=d.audit_support.to_numpy()&known;t=d.t_ns.to_numpy();seg=d.audit_segment.to_numpy();split=d.split.iloc[0]
            values={'fill':np.where(known,(cat<3).astype(float),np.nan),
                    'take_joint':np.where(known,(cat==0).astype(float),np.nan),
                    'take_conditional':np.where(filled,(cat==0).astype(float),np.nan),
                    'class_conditional':np.where(filled,cat,np.nan),
                    'p_fill':d.pf_rich_spectrum.to_numpy(),
                    'p_take_joint':(d.pf_rich_spectrum*d.pc0_rich_spectrum).to_numpy(),
                    'p_take_conditional':np.where(filled,d.pc0_rich_spectrum,np.nan),
                    'g_take':d.g_path.to_numpy()}
            for name in ('fill','take_joint','take_conditional','class_conditional'):
                v=values[name].copy();ok=np.isfinite(v)&support;v[ok]=rng.permutation(v[ok]);values[name+'_permuted']=v
            for name,v in values.items():
                # Conditional continuity requires every interior origin to be
                # filled as well. EMPTY is an interruption, never a path class.
                used_support=support&filled if ('conditional' in name or name=='g_take') else support
                for lag in [30,60,120,300,600,1800]:
                    accumulator[f'{mode}/{split}/{name}/{lag}'].append(lag_record(t,v,seg,used_support,30,lag))
                if name in ('fill','take_conditional','class_conditional'):
                    runs[f'{mode}/{split}/{name}']+=run_lengths(t,v,seg,support,30)
    return {k:combine_lags(v) for k,v in accumulator.items()}, {k:dict(n_runs=len(v),median_observed_span_seconds=float(np.median([r[1]*30 for r in v])),
        p90_observed_span_seconds=float(np.quantile([r[1]*30 for r in v],.9))) for k,v in runs.items() if v}


def dense_scenes(man,frames):
    selected={};stats=defaultdict(list);records=[]
    for day in man['days']:selected.setdefault(day['date'][:7],day['date'])
    for date in selected.values():
        p=Path('data/derived/scalper_v1')/(date+'.parquet');day=MonthDay(MARKET/date[:7]/date);q=QuoteTape.from_day(day)
        columns=['t_ns','sigma600','sigma_known','spread_ticks']+[f'{m}_{s}_f24_12_{v}' for m in ('P','T') for s in ('L','S') for v in ('st','net','busy_s')]
        src=pd.read_parquet(p,columns=columns);t=src.t_ns.to_numpy();j=np.searchsorted(q.quote_t,t,side='right')-1
        gate=(j>=0)&(src.sigma_known.to_numpy()>0)&(src.sigma600.to_numpy()>=.3)&(src.spread_ticks.to_numpy()>0)&(t+335*ep.NS<q.deadline_ns)&q.valid[j]
        src=src.loc[gate].reset_index(drop=True);j=j[gate];t=src.t_ns.to_numpy();dense=[]
        audit_support,audit_segment=minute_certificate(day,t)
        for mode in ('P','T'):
            for side,sn in ((1,'L'),(-1,'S')):
                key=f'{mode}_{sn}_f24_12';cat=ep.categories(src[key+'_st'],src[key+'_net']);known=cat>=0;filled=known&(cat<3)
                d=pd.DataFrame(dict(date=date,t_ns=t,mode=mode,side=side,category=cat,net=src[key+'_net'],busy_s=src[key+'_busy_s'],
                    mid=(q.bid[j].astype(float)+q.ask[j])/2e9,bid=q.bid[j]/1e9,ask=q.ask[j]/1e9,epoch=q.epoch[j],spread_ticks=src.spread_ticks,
                    audit_support=audit_support,audit_segment=audit_segment))
                dense.append(d)
                origin=frames[mode].loc[(frames[mode].date==date)&(frames[mode].side==side),['t_ns','category']]
                joined=origin.merge(d[['t_ns','category']],on='t_ns',suffixes=('_v2','_dense'),validate='one_to_one')
                if len(joined)!=len(origin) or not np.array_equal(joined.category_v2,joined.category_dense):raise ValueError('Dense targets disagree at v2 roots')
                for name,v in {'fill':np.where(known,cat<3,np.nan),'take_conditional':np.where(filled,cat==0,np.nan),
                               'class_conditional':np.where(filled,cat,np.nan)}.items():
                    used_support=audit_support&(filled if 'conditional' in name else known)
                    for lag in [1,2,5,10,30,60]:stats[f'{mode}/{name}/{lag}'].append(lag_record(t,v,audit_segment,used_support,1,lag))
        output=pd.concat(dense,ignore_index=True);output.to_parquet(DATA/(date+'_dense.parquet'),index=False)
        records.append(dict(date=date,source_sha256=sha(p),rows=len(output),london_ns=str(day.meta['london_ns']),
                            dense_sha256=sha(DATA/(date+'_dense.parquet'))))
        QUOTE_CACHE.clear();progress(stage='dense',date=date,rows=len(output))
    return records,{k:combine_lags(v) for k,v in stats.items()}


def main():
    start=time.perf_counter();DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    all_rows,man,done=load_rows();progress(stage='loaded',rows=len(all_rows))
    result,frames=predict_audit(all_rows)
    result.update(class_distribution=class_rows(all_rows),source_completion_sha256=sha(Path('reports/flow_context_v2/completion_receipt.json')),
                  source_manifest_sha256=sha(SOURCE/'manifest.json'),contract_sha256=sha(CONTRACT),
                  code_sha256={p:sha(p) for p in ['scripts/target_loss_audit.py','spectrum/cognition/target_loss_audit.py']})
    result['temporal'],result['runs']=temporal_diagnostics(frames)
    result['dense_dates'],result['dense_temporal']=dense_scenes(man,frames)
    result['frame_sha256']={p.name:sha(p) for p in DATA.glob('frame_*.parquet')}
    result['git_head_at_run']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result['seconds']=round(time.perf_counter()-start,2);result['status']='COMPLETE'
    save(REPORT/'measurements.json',result);progress(stage='COMPLETE',seconds=result['seconds'])


if __name__=='__main__':main()
