"""Bounded favorable relabeling, event diagnostics and past-only probe."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import joblib
from spectrum.research_v6.favorable import qualify, origin_summary, adjacent_changes, probe_feature_names
from scripts.build_crossing_path_target_v6 import sha,save

CONFIG=Path('configs/research/favorable_series_v1.json')


def metrics(y,p):
    return dict(n=len(y),positives=int(y.sum()),prevalence=float(y.mean()),
        auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        ap=float(average_precision_score(y,p)) if y.sum() else None,
        brier=float(brier_score_loss(y,p)),logloss=float(log_loss(y,p,labels=[0,1])))


def fit_probes(origins,x,names,base,c,out,report):
    base_names,flow_names=probe_feature_names(names)
    base_idx=[names.index(n) for n in base_names]
    flow_idx=[names.index(n) for n in flow_names]
    use=(origins.common_known & origins.feature_available & origins.physical_support_available & origins.decision_eligible).to_numpy()
    train=use & (origins.split.to_numpy()=='train')
    if (origins.loc[use,'source_available_ns']>origins.loc[use,'t_ns']).any():
        raise ValueError('future source publication')
    pc=c['probe']
    params={k:pc[k] for k in ('max_iter','max_leaf_nodes','min_samples_leaf','learning_rate','l2_regularization','random_state','early_stopping')}
    prediction=origins[['date','t_ns','split','common_known','endpoint_L','endpoint_S']].copy()
    prediction['admitted']=use
    results={'protocol':pc,'base_features':base_names,'flow_features':flow_names,'per_side':{},
             'cohorts':{s:dict(all=int((origins.split==s).sum()),admitted=int((use&(origins.split.to_numpy()==s)).sum())) for s in base['splits']}}
    for side in ('L','S'):
        y=origins['endpoint_'+side].to_numpy()
        models={};side_result={}
        for arm,indices in (('base',base_idx),('flow',base_idx+flow_idx)):
            model=HistGradientBoostingClassifier(**params)
            with threadpool_limits(limits=4):
                model.fit(x[train][:,indices],y[train])
                p=model.predict_proba(x[:,indices])[:,1]
            models[arm]=model
            joblib.dump(model,out/f'probe_{side}_{arm}.joblib')
            prediction[f'p_{side}_{arm}']=p
            side_result[arm]={s:metrics(y[use&(origins.split.to_numpy()==s)],p[use&(origins.split.to_numpy()==s)]) for s in base['splits']}
        day_metrics=[]
        for date in base['splits']['test']:
            mask=use&(origins.date.to_numpy()==date)
            row={'date':date}
            for arm in ('base','flow'):row[arm]=metrics(y[mask],prediction[f'p_{side}_{arm}'].to_numpy()[mask])
            row['brier_improvement']=row['base']['brier']-row['flow']['brier']
            day_metrics.append(row)
        side_result['test_days']=day_metrics
        # Conditional permutation is descriptive; bins use TRAIN cutpoints only.
        cuts=np.unique(np.quantile(prediction.loc[train,f'p_{side}_base'],np.arange(.1,1,.1)))
        bins=np.searchsorted(cuts,prediction[f'p_{side}_base'].to_numpy())
        prediction[f'base_bin_{side}']=bins
        test=use&(origins.split.to_numpy()=='test')
        it=np.flatnonzero(test)
        z=x[it][:,base_idx+flow_idx].copy()
        groups=pd.DataFrame({'date':origins.date.to_numpy()[it],'bin':bins[it]}).groupby(['date','bin']).indices
        nulls=[]
        for seed in range(pc['flow_permutations']):
            zz=z.copy();rng=np.random.default_rng(1700+seed)
            for positions in groups.values():
                zz[positions,len(base_idx):]=z[rng.permutation(positions),len(base_idx):]
            with threadpool_limits(limits=4): pn=models['flow'].predict_proba(zz)[:,1]
            nulls.append(metrics(y[it],pn))
        side_result['test_flow_permutation']=nulls
        side_result['train_base_probability_cuts']=cuts.tolist()
        results['per_side'][side]=side_result
        print(json.dumps({'probe_side':side,'base_test':side_result['base']['test'],'flow_test':side_result['flow']['test']}),flush=True)
    # Geometry-only matching, across TEST days, no labels used for neighbour search.
    matching_names=base_names[:14]+['mbo__path_30s__signed_log1p','mbo__path_128s__signed_log1p']
    mi=[names.index(n) for n in matching_names]
    median=np.nanmedian(x[train][:,mi],axis=0)
    iqr=np.nanquantile(x[train][:,mi],.75,axis=0)-np.nanquantile(x[train][:,mi],.25,axis=0)
    iqr=np.where(iqr>1e-6,iqr,1.)
    z=np.clip((np.where(np.isfinite(x[:,mi]),x[:,mi],median)-median)/iqr,-10,10)
    eligible=use&(origins.split.to_numpy()=='test')&((origins.t_ns.to_numpy()%(60*10**9))==0)
    # Require observed matching coordinates, so missingness cannot masquerade as similarity.
    eligible &= np.isfinite(x[:,mi]).all(axis=1)
    ids=np.flatnonzero(eligible); pairs=[]
    position=[names.index(f'bb{s}_position') for s in (1,3,6)]
    width=[names.index(f'bb{s}_log_half_ticks') for s in (1,3,6)]
    minute=names.index('minutes_since_london');spread=names.index('spread_ticks')
    for q in ids:
        candidates=ids[(origins.date.to_numpy()[ids]!=origins.date.iloc[q]) &
            (abs(x[ids,minute]-x[q,minute])<=30) & (abs(x[ids,spread]-x[q,spread])<=1)]
        candidates=candidates[(abs(x[candidates][:,position]-x[q,position]).max(axis=1)<=.5)&
                              (abs(x[candidates][:,width]-x[q,width]).max(axis=1)<=.35)]
        if not len(candidates):continue
        distance=np.sqrt(((z[candidates]-z[q])**2).mean(axis=1));j=distance.argmin();d=distance[j]
        if d<=.5:pairs.append((min(int(q),int(candidates[j])),max(int(q),int(candidates[j])),float(d)))
    pairs=pd.DataFrame(pairs,columns=['left','right','distance']).drop_duplicates(['left','right'])
    for side in ('L','S'):
        pairs[f'y_left_{side}']=origins['endpoint_'+side].to_numpy()[pairs.left]
        pairs[f'y_right_{side}']=origins['endpoint_'+side].to_numpy()[pairs.right]
    pairs.to_parquet(out/'matched_pairs.parquet',index=False)
    results['matching']=dict(pairs=len(pairs),features=matching_names,caliper_rms=.5,
        median_distance=float(pairs.distance.median()),
        long_discordant_fraction=float((pairs.y_left_L!=pairs.y_right_L).mean()),
        short_discordant_fraction=float((pairs.y_left_S!=pairs.y_right_S).mean()),
        clock_caliper_minutes=30,spread_caliper_ticks=1,position_caliper=.5,log_half_caliper=.35,
        definition='Each 60s TEST origin seeks one nearest neighbour on another TEST day within explicit clock/spread/position/width calipers. Deduplicate symmetric pairs; pairs can share origins; future outcomes never used for matching. Descriptive, no independent-pair significance.')
    prediction.to_parquet(out/'probe_predictions.parquet',index=False)
    save(report/'probe_results.json',results)
    return results


def main():
    started=time.perf_counter()
    c=json.loads(CONFIG.read_text());base=json.loads(Path(c['base_config']).read_text())
    source=Path(c['source']);out=Path(c['output']);report=Path(c['report'])
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads((source/'manifest.json').read_text())
    if source_manifest['status']!='COMPLETE':raise ValueError('source not complete')
    bound={str(CONFIG):sha(CONFIG),c['base_config']:sha(c['base_config']),str(source/'manifest.json'):sha(source/'manifest.json'),
           'spectrum/research_v6/favorable.py':sha('spectrum/research_v6/favorable.py'),str(Path(__file__).resolve()):sha(__file__)}
    stats={'status':'RUNNING','version':c['version'],'ratio':c['ratio'],'numeric_margin_tolerance_R':1e-10,
        'days':[],'input_hashes':bound,'sensitivity':{},'series':[1,2,6,7,9,18]}
    save(report/'summary.json',stats)
    root_frames=[];x_frames=[];changes=[]
    data=Path(base['data']['root']);names=None
    for item in source_manifest['days']:
        date=item['date'];p=source/f'{date}_target.parquet'
        if sha(p)!=item['outputs'][p.name]:raise ValueError('V6 source drift')
        bound[str(p)]=item['outputs'][p.name]
        # Keep factual columns needed for this series, not all redundant wave coefficients.
        cols=['date','split','t_ns','scale','role','entry_ns','entry_ordinal','cross_ns','cross_ordinal',
            'v6_event_id','profile_known','hit_known','hit','ready','censored','target_available_ns',
            'R_event','elapsed_seconds','cause','price_move_R','band_move_R']
        cols += [f'{m}_{s}' for s in ('L','S') for m in ('net_R','mae_R','mfe_R','mae_before_cross_R',
            'max_drawdown_R','mean_negative_R','underwater_fraction','risk_before_cross','first_positive_seconds','first_risk_seconds')]
        original=pd.read_parquet(p,columns=cols)
        cells=qualify(original,c['ratio']);roots=origin_summary(cells)
        cells.to_parquet(out/f'{date}_cells.parquet',index=False)
        receipt=json.loads((data/f'{date}.json').read_text())
        for suffix in ('roots.parquet','x.npy','scene.parquet'):
            path=data/f'{date}_{suffix}'
            if sha(path)!=receipt['files'][path.name]:raise ValueError('frozen features changed')
            bound[str(path)]=receipt['files'][path.name]
        old=pd.read_parquet(data/f'{date}_roots.parquet')
        if not np.array_equal(roots.t_ns,old.t_ns):raise ValueError('origin alignment')
        for key in ('feature_available','physical_support_available','decision_eligible','source_available_ns'):
            roots[key]=old[key].to_numpy()
        root_frames.append(roots)
        xn=np.load(data/f'{date}_x.npy')
        if names is None:names=receipt['feature_names']
        if names!=receipt['feature_names'] or len(xn)!=len(roots):raise ValueError('feature schema/alignment')
        x_frames.append(xn)
        changes.append(adjacent_changes(cells))
        d=dict(date=date,split=item['split'],roots=len(roots),cells=len(cells),observed=int(cells.profile_known.sum()),
            common_known=int(roots.common_known.sum()),sides={},states={})
        for kind in ('mfe','endpoint'):d['states'][kind]={str(k):int(v) for k,v in roots[f'{kind}_state'].value_counts().items()}
        for s in ('L','S'):
            mq=cells[f'mfe_qualified_{s}']==1;eq=cells[f'endpoint_qualified_{s}']==1
            d['sides'][s]=dict(mfe=int(mq.sum()),endpoint=int(eq.sum()),
                mfe_nonpositive_endpoint=int((mq&(cells[f'net_R_{s}']<=0)).sum()),
                endpoint_risk_before=int((eq&(cells[f'risk_before_cross_{s}']==1)).sum()),
                endpoint_distinct_events=int(cells.loc[eq,'v6_event_id'].nunique()))
        for ratio in c['sensitivity_ratios']:
            alt=qualify(original,ratio)
            key=str(ratio)
            if key not in stats['sensitivity']:stats['sensitivity'][key]={f'{kind}_{s}':0 for kind in ('mfe','endpoint') for s in ('L','S')}
            for name in stats['sensitivity'][key]:stats['sensitivity'][key][name]+=int((alt[name.replace('_L','_qualified_L').replace('_S','_qualified_S')]==1).sum())
        stats['days'].append(d)
        print(json.dumps({'date':date,'known':d['common_known'],'endpoint_states':d['states']['endpoint']}),flush=True)
    origins=pd.concat(root_frames,ignore_index=True);x=np.concatenate(x_frames)
    origins.to_parquet(out/'origins.parquet',index=False)
    np.save(out/'x.npy',x)
    save(out/'feature_names.json',names)
    change=pd.concat(changes,ignore_index=True)
    change.to_parquet(out/'adjacent_changes.parquet',index=False)
    total=change.groupby(['side','kind'])[['pairs','flips','margin_abs_sum']].sum().reset_index()
    stats['adjacent_changes']=total.to_dict(orient='records')
    stats['roots']=len(origins);stats['cells']=sum(d['cells'] for d in stats['days'])
    stats['probe']=fit_probes(origins,x,names,base,c,out,report)
    for path,digest in bound.items():
        if sha(path)!=digest:raise ValueError('bound input changed during run '+path)
    stats['status']='COMPLETE';stats['seconds']=time.perf_counter()-started
    stats['outputs']={str(p):sha(p) for p in out.iterdir() if p.is_file()}
    save(report/'summary.json',stats)
    print(json.dumps({'status':'COMPLETE','seconds':stats['seconds'],'matching':stats['probe']['matching']}),flush=True)


if __name__=='__main__':main()
