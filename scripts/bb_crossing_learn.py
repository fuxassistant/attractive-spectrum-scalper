"""Chronological geometry/flow/event-weight controls and separate economic heads."""
from pathlib import Path
import json
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from spectrum.cognition import bb_crossing as bb, flow_context as fc, episodes as ep
from scripts.bb_crossing_build import DATA, REPORT, identity as build_identity
from scripts.flow_context_coverage import save, sha

BASE=[c for c in fc.FAMILIES['baseline'] if c!='side']
RICH=[c for c in fc.FAMILIES['rich_spectrum'] if c!='side'] + bb.GEOMETRY
FAMILIES={'geometry':BASE+bb.GEOMETRY,'rich':RICH,'rich_event':RICH}
HEADS=['m1','m3','m6','m9','global']


def probability(model,x,size):
    p=np.zeros((len(x),size));p[:,model.classes_.astype(int)]=model.predict_proba(x)
    return p


def scores(f,p,key):
    ok=f[key+'_y'].to_numpy()>=0;y=f.loc[ok,key+'_y'].to_numpy(int);pp=p[ok]
    if not len(y):return {'rows':0}
    w=bb.event_weights(f.loc[ok,key+'_id']);loss=-np.log(np.clip(pp[np.arange(len(y)),y],1e-12,1))
    correct=pp.argmax(1)==y;confidence=pp.max(1);cal=[]
    for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.01,.1)):
        a=(confidence>=lo)&(confidence<hi+1e-12)
        if a.any():cal.append(dict(lo=float(lo),n=int(a.sum()),predicted=float(confidence[a].mean()),accuracy=float(correct[a].mean())))
    return dict(rows=len(y),unknown=int((~ok).sum()),log_loss=float(loss.mean()),event_log_loss=float(np.average(loss,weights=w)),
        accuracy=float(correct.mean()),event_accuracy=float(np.average(correct,weights=w)),
        predicted_counts={str(k):int(v) for k,v in pd.Series(pp.argmax(1)).value_counts().items()},
        observed_counts={str(k):int(v) for k,v in pd.Series(y).value_counts().items()},
        calibration=cal,mean_probabilities=pp.mean(0).tolist())


def roughness(f,p,key):
    y=f[key+'_y'].to_numpy();t=f.t_ns.to_numpy();day=f.date.to_numpy();end=f[key+'_support_until_ns'].to_numpy()
    pair=(np.diff(t)==30*bb.NS)&(day[1:]==day[:-1])&(y[1:]>=0)&(y[:-1]>=0)&(t[1:]<end[:-1])
    a=np.flatnonzero(pair);same=f[key+'_id'].to_numpy()[a]==f[key+'_id'].to_numpy()[a+1]
    return dict(pairs=len(a),target_flip=float(np.mean(y[a]!=y[a+1])) if len(a) else None,
        event_id_flip=float(np.mean(~same)) if len(a) else None,
        mean_prediction_total_variation=float(np.abs(p[a+1]-p[a]).sum(1).mean()/2) if len(a) else None,
        argmax_flip=float(np.mean(p[a].argmax(1)!=p[a+1].argmax(1))) if len(a) else None)


def nearest(f,key):
    scales=bb.SCALES if key=='global' else [int(key[1:])]
    distances=[]
    for s in scales:
        position=f[f'bb{s}_position'].to_numpy();half=np.expm1(f[f'bb{s}_log_half_ticks'].to_numpy())
        distances.extend([np.abs(position-r)*half for r in [-1,0,1]])
    d=np.stack(distances,axis=1);d=np.where(np.isfinite(d),d,np.inf)
    return d.argmin(1)


def multiplicity(f,key):
    y=f[key+'_y'];ids=f.loc[(y>=0)&(f[key+'_event_ns']>0),key+'_id'];n=ids.value_counts()
    return dict(event_queries=len(ids),distinct_events=len(n),queries_per_event={str(q):float(n.quantile(q)) for q in [0,.5,.9,.99,1]},
        row_fraction_in_repeated_event=float(n[n>1].sum()/max(1,len(ids))),
        uniform_kish_event_units=float(n.sum()**2/np.sum(n.to_numpy(float)**2)) if len(n) else 0)


def classification(train,test,key,family,stage):
    ycol=key+'_y';columns=FAMILIES[family];size=14 if key=='global' else 5
    boundary=int(test.t_ns.min());mask=(train[ycol]>=0)&(train[key+'_maturity_ns']<boundary)
    tr=train.loc[mask];y=tr[ycol].to_numpy(int);x=tr[columns].to_numpy(np.float32)
    ids_train=set(tr.loc[tr[key+'_event_ns']>0,key+'_id']);ids_test=set(test.loc[test[key+'_event_ns']>0,key+'_id'])
    if ids_train&ids_test:raise ValueError('Shared future crossing across time split')
    w=bb.event_weights(tr[key+'_id']) if family=='rich_event' else np.ones(len(tr))
    model=HistGradientBoostingClassifier(random_state=bb.contract()['seed'],**bb.contract()['model']).fit(x,y,sample_weight=w)
    p=probability(model,test[columns].to_numpy(np.float32),size)
    prior=np.bincount(y,minlength=size).astype(float);prior/=prior.sum()
    baseline=np.broadcast_to(prior,(len(test),size))
    result=dict(columns=columns,train_rows=len(tr),purged=int(((train[ycol]>=0)&~mask).sum()),shared_test_event_ids=0,
        scores=scores(test,p,key),prior=scores(test,baseline,key),roughness=roughness(test,p,key),
        nearest_line_accuracy=float((nearest(test.loc[test[ycol]>=0],key)==test.loc[test[ycol]>=0,ycol]).mean()),
        multiplicity=multiplicity(test,key),months={m:scores(test.loc[test.date.str[:7]==m],p[test.date.str[:7]==m],key) for m in test.date.str[:7].unique()})
    pt=model.predict_proba(x);encoded=np.searchsorted(model.classes_,y)
    raw=model._raw_predict(x)
    analytical=bb.ce_gradient(pt,encoded,w)
    native=model._loss.gradient(y_true=encoded.astype(float),raw_prediction=raw,sample_weight=w)
    if native.shape!=analytical.shape:raise ValueError('Unexpected multiclass gradient layout')
    error=float(np.max(np.abs(native-analytical)))
    if error>1e-9:raise ValueError('Weighted native CE gradient mismatch')
    result['gradient']=dict(max_native_error=error,weight_min=float(w.min()),weight_max=float(w.max()),
        by_true_class=[dict(label=int(c),n=int((y==c).sum()),weight=float(w[y==c].sum()),
            l1_gradient=float(np.abs(analytical[y==c]).sum())) for c in model.classes_])
    out=test[['date','t_ns',ycol,key+'_id',key+'_event_ns',key+'_mask',key+'_delay_s',key+'_support_until_ns']].copy()
    for k in range(size):out[f'p{k}']=p[:,k]
    out.to_parquet(DATA/f'predictions_{stage}_{key}_{family}.parquet',index=False)
    with (DATA/f'model_{stage}_{key}_{family}.pkl').open('wb') as dst:pickle.dump(dict(model=model,columns=columns),dst)
    return result


def account(f,scale,days):
    capital=1500.;peak=capital;drawdown=0.;busy=0;trades=[];censor=None
    for r in f.itertuples():
        if r.t_ns<busy or not np.isfinite(getattr(r,f'bb{scale}_position')):continue
        side='L' if r.pred_L>=r.pred_S else 'S';score=max(r.pred_L,r.pred_S)
        if not np.isfinite(score) or score<=0:continue
        if r.execution_status<0:
            censor=dict(date=r.date,t_ns=str(r.t_ns),reason=r.termination,entry_ns=str(r.entry_ns));break
        if r.execution_status==0:
            busy=r.t_ns+2*bb.NS;continue
        net=getattr(r,'net_'+side);mae=getattr(r,'exit_mae_'+side)
        # This is an entry-anchored lower bound; full quote replay later gives
        # actual drawdown including peaks formed inside an open trade.
        drawdown=max(drawdown,peak-(capital+mae))
        before=capital;capital+=net;peak=max(peak,capital);busy=r.exit_ns
        trades.append(dict(date=r.date,t_ns=r.t_ns,entry_ns=r.entry_ns,exit_ns=r.exit_ns,scale=scale,side=side,
            predicted=score,net=net,mark=getattr(r,'mark_'+side),mae=mae,equity_before=before,equity=capital,
            cross_ns=r.cross_ns,cross_id=r.cross_id,entry_ordinal=r.entry_ordinal,exit_ordinal=r.exit_ordinal))
        if capital<=0:
            censor=dict(date=r.date,t_ns=str(r.exit_ns),reason='CAPITAL_EXHAUSTED');break
    trades=pd.DataFrame(trades)
    known_days=days if censor is None else [d for d in days if d<censor['date']]
    daily=(trades.groupby('date').net.sum().reindex(known_days,fill_value=0.) if len(trades) else pd.Series(0.,index=known_days))
    result=dict(fills=len(trades),censor=censor,net=float(capital-1500) if censor is None else None,
        observed_prefix_net=float(capital-1500),last_realized_equity=capital,
        drawdown_lower_bound=float(drawdown),daily_ci=ep.block_interval(daily.to_numpy()) if censor is None else [None,None],
        daily={d:float(v) for d,v in daily.items()})
    return result,trades


def economics(train,test,profiles,stage,admissions):
    out={};days=sorted(test.date.unique());boundary=int(test.t_ns.min())
    for scale in bb.SCALES:
        pr=profiles.loc[profiles.scale==scale]
        tr=train.merge(pr,on=['date','t_ns'],validate='one_to_one');te=test.merge(pr,on=['date','t_ns'],validate='one_to_one')
        fit=(tr.execution_status==1)&(tr.exit_ns<boundary)&np.isfinite(tr[f'bb{scale}_position'])
        rr=tr.loc[fit];models={}
        for side in ('L','S'):
            model=HistGradientBoostingRegressor(loss='squared_error',random_state=bb.contract()['seed'],**bb.contract()['model'])
            model.fit(rr[RICH].to_numpy(np.float32),rr['net_'+side].to_numpy())
            te['pred_'+side]=model.predict(te[RICH].to_numpy(np.float32));models[side]=model
        stats,trades=account(te,scale,days)
        if stage=='A2':admissions[str(scale)]=bool(stats['censor'] is None and stats['fills']>=30 and stats['daily_ci'][0]>0)
        good=te.execution_status==1;chosen=np.where(te.pred_L>=te.pred_S,te.net_L,te.net_S)
        out[str(scale)]=dict(train_rows=len(rr),candidate_account=stats,operational_admitted=admissions[str(scale)],
            operational_action='CANDIDATE' if admissions[str(scale)] else 'WAIT',
            known_opportunity_queries=int(good.sum()),mean_net_selected_side_all_known=float(chosen[good].mean()),
            mean_net_random_side_all_known=float((te.loc[good,'net_L']+te.loc[good,'net_S']).mean()/2),
            positive_prediction_fraction=float((te[['pred_L','pred_S']].max(axis=1)>0).mean()),
            caveat='All-known-query means overlap and are not account equity')
        te.to_parquet(DATA/f'economic_{stage}_m{scale}.parquet',index=False)
        trades.to_parquet(DATA/f'trades_{stage}_m{scale}.parquet',index=False)
        with (DATA/f'economic_model_{stage}_m{scale}.pkl').open('wb') as dst:pickle.dump(dict(models=models,columns=RICH),dst)
        print(json.dumps(dict(stage=stage,economic_scale=scale,**stats)),flush=True)
    return out


def main():
    man=json.loads((DATA/'manifest.json').read_text())
    if man['identity']!=build_identity():raise ValueError('Data identity changed')
    roots=[];profiles=[]
    for day in man['days']:
        for suffix,dest in [('roots',roots),('profiles',profiles)]:
            path=DATA/(day['date']+'_'+suffix+'.parquet')
            if sha(path)!=day['files'][suffix]:raise ValueError('Changed source table')
            dest.append(pd.read_parquet(path))
    roots=pd.concat(roots,ignore_index=True);profiles=pd.concat(profiles,ignore_index=True)
    result=dict(version=bb.contract()['version'],build_identity=man['identity'],script_sha256=sha(__file__),
        manifest_sha256=sha(DATA/'manifest.json'),families=FAMILIES,stages={},evaluation='DEVELOPMENT_EXPOSED_NOT_BLIND')
    admissions={}
    for stage,months,testmonths in [('A2',['2025-10','2025-11'],['2025-12']),('B',['2025-10','2025-11','2025-12'],['2026-01','2026-02'])]:
        train=roots.loc[roots.date.str[:7].isin(months)].reset_index(drop=True)
        test=roots.loc[roots.date.str[:7].isin(testmonths)].reset_index(drop=True)
        stage_result=dict(classification={},economics={})
        for key in HEADS:
            stage_result['classification'][key]={}
            for family in FAMILIES:
                start=time.perf_counter();res=classification(train,test,key,family,stage)
                stage_result['classification'][key][family]=res
                print(json.dumps(dict(stage=stage,head=key,family=family,loss=res['scores']['log_loss'],
                    event_loss=res['scores']['event_log_loss'],accuracy=res['scores']['accuracy'],seconds=round(time.perf_counter()-start,2))),flush=True)
        stage_result['economics']=economics(train,test,profiles,stage,admissions)
        result['stages'][stage]=stage_result;save(REPORT/'summary.running.json',result)
    result['admissions']=admissions
    save(REPORT/'summary.json',result)
    save(REPORT/'learning_receipt.json',dict(status='COMPLETE',summary_sha256=sha(REPORT/'summary.json'),
        manifest_sha256=sha(DATA/'manifest.json'),script_sha256=sha(__file__),
        predictions={p.name:sha(p) for p in DATA.glob('predictions_*.parquet')},
        models={p.name:sha(p) for p in DATA.glob('*model*.pkl')}))


if __name__=='__main__':main()
