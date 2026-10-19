"""Past-only flow/price response laboratory; never mutates the market dataset."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, json, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .full_history import ObservedCalendar
from .core.storage import sha
from .ingest import save_json

NS=10**9
WINDOW=512
HOP=128
SEGMENT=128
BANDS=((4,8),(8,16),(16,32))
HORIZONS=(30,120)
COUNTERS=('trade_buy','trade_sell','trade_unknown','path_ticks','add_bid','add_ask',
          'remove_bid','remove_ask','matched_fill_remove_bid','matched_fill_remove_ask',
          'post_trade_add_bid','post_trade_add_ask')


def ratio(a,b):
    return float(a/b) if b>1e-12 else 0.


def spectral_features(channels):
    """Seven overlapping past segments; smooth spectra BEFORE normalization."""
    names=list(channels);v=np.stack([channels[n] for n in names])
    z=np.lib.stride_tricks.sliding_window_view(v,SEGMENT,axis=-1)[:,::64,:]
    w=.5-.5*np.cos(2*np.pi*np.arange(SEGMENT)/SEGMENT)
    ft=np.fft.rfft((z-z.mean(axis=-1,keepdims=True))*w,axis=-1)
    psd=(abs(ft)**2).mean(axis=1)/(w@w)
    one_sided=np.ones(SEGMENT//2+1);one_sided[1:-1]=2
    psd*=one_sided
    f=np.fft.rfftfreq(SEGMENT)
    auto={};cross={}
    for k,n in enumerate(names):
        total=psd[k,1:].sum()
        auto[n+'_power']=float(np.log1p(total/SEGMENT))
        auto[n+'_power_known']=float(total>1e-12)
        prob=psd[k,1:]/total if total>1e-12 else np.zeros(SEGMENT//2)
        auto[n+'_entropy']=float(-(prob*np.log(np.maximum(prob,1e-300))).sum()/np.log(len(prob)))
        for a,b in BANDS:
            mask=(f>=1/b)&(f<1/a)
            auto[f'{n}_fraction_{a}_{b}']=ratio(psd[k,mask].sum(),total)
    for x,y in [('classified_flow','return'),('book_flow','return'),('volume','abs_return')]:
        i,j=names.index(x),names.index(y)
        xy=(ft[i].conj()*ft[j]).mean(axis=0)/(w@w)*one_sided
        den=psd[i]*psd[j]
        coherence=np.divide(abs(xy)**2,den,out=np.zeros_like(den),where=den>1e-20)
        for a,b in BANDS:
            mask=(f>=1/b)&(f<1/a);good=mask&(den>1e-20)
            stem=f'{x}_to_{y}_{a}_{b}'
            known=bool(np.any(good))
            cross[stem+'_known']=float(known)
            cross[stem+'_coherence']=float(np.clip(coherence[good].mean(),0,1)) if known else 0.
            # Sum cross spectra and input power within the band: weighted complex gain.
            gain=xy[good].sum()/psd[i,good].sum() if known else 0j
            cross[stem+'_gain_log']=float(np.log1p(abs(gain)))
            phase_known=known and abs(gain)>1e-12
            cross[stem+'_phase_known']=float(phase_known)
            cross[stem+'_phase_cos']=float(np.cos(np.angle(gain))) if phase_known else 0.
            cross[stem+'_phase_sin']=float(np.sin(np.angle(gain))) if phase_known else 0.
    return auto,cross


def window_features(d,end):
    """Only [end-512,end] is accessed. Unknown side is an observed separate channel."""
    z=d.iloc[end-WINDOW:end+1];deltas={n:np.diff(z['c_'+n].to_numpy()) for n in COUNTERS}
    ret=np.diff(z.mid.to_numpy())/.1
    buy,sell,u=(deltas[n] for n in ('trade_buy','trade_sell','trade_unknown'))
    q=buy-sell;v=buy+sell+u
    book=(deltas['add_bid']-deltas['remove_bid'])-(deltas['add_ask']-deltas['remove_ask'])
    base={}
    for n in (30,128,512):
        vol=float(v[-n:].sum());known=float((buy+sell)[-n:].sum());net=float(q[-n:].sum());unknown=float(u[-n:].sum())
        move=float(ret[-n:].sum());path=float(deltas['path_ticks'][-n:].sum())
        vals=dict(volume_log=np.log1p(vol),classified_flow=np.sign(net)*np.log1p(abs(net)),
            imbalance=ratio(net,known),unknown_fraction=ratio(unknown,vol),volume_known=float(vol>0),
            pressure_lower=ratio(net-unknown,vol),pressure_upper=ratio(net+unknown,vol),
            progress=move,path_log=np.log1p(max(path,0)),efficiency=ratio(abs(move),path),
            ticks_per_100=ratio(100*move,vol),return_std=float(ret[-n:].std()),
            flow_std=float(q[-n:].std()),corr=float(np.corrcoef(q[-n:],ret[-n:])[0,1]) if q[-n:].std()>0 and ret[-n:].std()>0 else 0.)
        for k in COUNTERS[4:]:vals[k+'_log']=np.log1p(max(0,float(deltas[k][-n:].sum())))
        for k,value in vals.items():base[f'{k}_{n}']=float(value)
    last=z.iloc[-1]
    base['spread_ticks']=float((last.ask-last.bid)/.1)
    for n in (1,4,12):
        bid=float(last[f'depth_bid_{n}']);ask=float(last[f'depth_ask_{n}'])
        base[f'depth_{n}_log']=float(np.log1p(bid+ask));base[f'depth_{n}_imbalance']=ratio(bid-ask,bid+ask)
    tm=pd.Timestamp(int(last.t_ns),tz='UTC');hour=tm.hour+tm.minute/60
    base['utc_hour_sin']=float(np.sin(hour*2*np.pi/24));base['utc_hour_cos']=float(np.cos(hour*2*np.pi/24))
    auto,cross=spectral_features({
        'return':ret,'classified_flow':q,'volume':v,'book_flow':book,'abs_return':abs(ret)})
    result={**{'A__'+k:v for k,v in base.items()},**{'B__'+k:v for k,v in auto.items()},**{'C__'+k:v for k,v in cross.items()}}
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite laboratory feature')
    return result


def materialize_capture(job):
    r,reference,out=job;out=Path(out);key=r['key'];stem=key.replace('/','_')
    market=Path(r['market']);feat=Path(r['features'])
    for p,h in [(market/'seconds.parquet',r['market_payloads']['seconds.parquet']),(feat/'features.parquet',r['feature_sha256'])]:
        if sha(p)!=h:raise ValueError('Input payload changed: '+str(p))
    d=pq.read_table(market/'seconds.parquet').to_pandas()
    f=pq.read_table(feat/'features.parquet',columns=['t_ns','entry_allowed']).to_pandas()
    if not np.array_equal(d.t_ns,f.t_ns):raise ValueError('Market/feature clock alignment')
    cal=ObservedCalendar(reference['path'],reference['sha256']);t=d.t_ns.to_numpy();av=d.available_ns.to_numpy();epoch=d.epoch.to_numpy()
    support=d.valid.to_numpy()&~cal.closed(t)&np.isfinite(d.mid.to_numpy())
    edge=support[1:]&support[:-1]&(np.diff(t)==NS)&(np.diff(epoch)==0)
    bad=np.r_[0,np.cumsum(~edge)]
    rows=[];counts=dict(candidate=0,unsupported=0,late_history=0,retained=0,with_unknown=0)
    for end in range(WINDOW,len(d),HOP):
        if not f.entry_allowed.iloc[end]:continue
        counts['candidate']+=1
        if bad[end]-bad[end-WINDOW]:counts['unsupported']+=1;continue
        decision=int(av[end])
        if np.max(av[end-WINDOW:end+1])>decision:counts['late_history']+=1;continue
        values=window_features(d,end)
        row=dict(key=key,date=r['entry']['selection']['date'],symbol=r['entry']['selection']['symbol'],
                 t_ns=int(t[end]),decision_ns=decision,**values)
        j=int(np.searchsorted(t,decision,side='right'))
        for h in HORIZONS:
            k=j+h;valid=(k<len(d) and f.entry_allowed.iloc[j] and bad[k]==bad[end] and
                        t[k]-t[j]==h*NS and epoch[k]==epoch[end] and d.age_ns.iloc[j]<=NS and d.age_ns.iloc[k]<=NS)
            row[f'mature_{h}']=bool(valid)
            row[f'y_{h}']=float((d.mid.iloc[k]-d.mid.iloc[j])/.1) if valid else 0.
            row[f'long_net_{h}']=float(d.bid.iloc[k]-d.ask.iloc[j]-.30) if valid else 0.
            row[f'short_net_{h}']=float(d.bid.iloc[j]-d.ask.iloc[k]-.30) if valid else 0.
            row[f'label_available_ns_{h}']=int(np.max(av[j:k+1])) if valid else 0
        rows.append(row);counts['retained']+=1;counts['with_unknown']+=int(values['A__unknown_fraction_512']>0)
    if rows:pq.write_table(pa.Table.from_pylist(rows),out/(stem+'.parquet'),compression='zstd')
    report=dict(key=key,date=r['entry']['selection']['date'],counts=counts,
                source_seconds_sha256=r['market_payloads']['seconds.parquet'],source_features_sha256=r['feature_sha256'],
                output_sha256=sha(out/(stem+'.parquet')) if rows else None)
    save_json(out/(stem+'.json'),report)
    return report


def split_dates(dates):
    dates=np.asarray(dates)
    return {'train':dates<'2026-03-01','validation':(dates>='2026-03-01')&(dates<'2026-05-01'),'evaluation':dates>='2026-05-01'}


def fit_compare(frame,out):
    splits=split_dates(frame.date.to_numpy());predictions=frame[['key','date','symbol','t_ns','decision_ns']].copy();results=[]
    rng=np.random.default_rng(613)
    for h in HORIZONS:
        y=frame[f'y_{h}'].to_numpy();mature=frame[f'mature_{h}'].to_numpy()
        masks={k:v&mature for k,v in splits.items()};tr,va,te=(masks[k] for k in ['train','validation','evaluation'])
        if min(tr.sum(),va.sum(),te.sum())<100:raise ValueError('Insufficient mature chronological splits')
        for n in ['mature','y','long_net','short_net']:predictions[f'{n}_{h}']=frame[f'{n}_{h}']
        models={}
        for group,prefixes in [('A',('A__',)),('B',('A__','B__')),('C',('A__','B__','C__'))]:
            cols=[n for n in frame if n.startswith(prefixes)];x=frame[cols].to_numpy(float)
            mean=x[tr].mean(axis=0);scale=x[tr].std(axis=0);scale[scale<1e-8]=1
            x=np.clip((x-mean)/scale,-8,8);x=np.column_stack([np.ones(len(x)),x]);yt=y[tr].mean()
            xx=x[tr].T@x[tr]/tr.sum();xy=x[tr].T@(y[tr]-yt)/tr.sum()
            trials=[]
            for alpha in (.01,.1,1.,10.):
                penalty=np.eye(x.shape[1])*alpha;penalty[0,0]=0
                coef=np.linalg.solve(xx+penalty,xy);pred=x@coef+yt
                trials.append((float(np.mean((pred[va]-y[va])**2)),alpha,coef,pred))
            _,alpha,coef,pred=min(trials,key=lambda z:z[0]);models[group]=pred
            predictions[f'{group}_{h}']=pred
            metrics={}
            for name,mask in masks.items():
                err=pred[mask]-y[mask];truth=y[mask]
                metrics[name]=dict(n=int(mask.sum()),days=int(frame.loc[mask,'date'].nunique()),mse=float(np.mean(err**2)),
                    mae=float(np.mean(abs(err))),r2=float(1-np.sum(err**2)/np.sum((truth-truth.mean())**2)),
                    zero_mse=float(np.mean(truth**2)),direction_accuracy=float(np.mean(np.sign(pred[mask])==np.sign(truth))))
            # Descriptive sign-of-prediction proxy; no threshold search or equity claim.
            net=np.where(pred[te]>=0,frame.loc[te,f'long_net_{h}'],frame.loc[te,f'short_net_{h}'])
            result=dict(group=group,horizon=h,features=len(cols),alpha=alpha,metrics=metrics,
                        evaluation_sign_proxy_mean_usd=float(np.mean(net)),wait_proxy_usd=0.,
                        proxy_warning='Overlapping origins can occur across targets; snapshot bid/ask proxy, not a fill simulator or portfolio equity.')
            results.append(result)
            save_json(out/f'model_{group}_{h}.json',dict(features=cols,mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),target_mean=float(yt),alpha=alpha,
                train_end_exclusive='2026-03-01',validation_end_exclusive='2026-05-01'))
        for group,baseline in [('B','A'),('C','A'),('C','B')]:
            base=(models[baseline][te]-y[te])**2;candidate=(models[group][te]-y[te])**2
            daily=pd.DataFrame(dict(date=frame.loc[te,'date'].to_numpy(),improvement=base-candidate)).groupby('date').improvement.mean()
            boot=rng.choice(daily.to_numpy(),size=(2000,len(daily)),replace=True).mean(axis=1)
            item=next(r for r in results if r['group']==group and r['horizon']==h)
            item[f'paired_day_mse_improvement_vs_{baseline}']=dict(mean=float(daily.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                method='exploratory paired day bootstrap, no correction across two horizons/three comparisons')
    pq.write_table(pa.Table.from_pandas(predictions,preserve_index=False),out/'predictions.parquet',compression='zstd')
    save_json(out/'comparison.json',dict(status='DEVELOPMENT_EVALUATION_NOT_BLIND',results=results,
        split=dict(train='before 2026-03-01',validation='2026-03-01 to 2026-04-30',evaluation='2026-05-01 onward'),
        limitations=['All periods already development-exposed.','Single linear model family; lack of improvement does not rule out other learners.',
                     'All groups use identical rows and target maturity masks.','Mean/std fitted on train only; alpha selected on validation only.',
                     'After-cost figures are snapshot proxies, not executable equity.']))
    return predictions,results


def run(root,output,workers):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'captures').mkdir()
    config=json.loads((root/'run_config.json').read_text());manifest=json.loads((root/'manifest.json').read_text());cat=json.loads((root/'training_catalog.json').read_text())
    # The independent full-session audit additionally excludes May 25.
    selected={x['key'] for x in cat['captures'] if x['eligible'] and x['date']!='2026-05-25'}
    receipts=sorted([r for r in manifest['captures'] if r['key'] in selected],key=lambda r:(r['entry']['selection']['date'],r['key']))
    protocol=dict(status='RUNNING',created_utc=pd.Timestamp.now(tz='UTC').isoformat(),window=WINDOW,hop=HOP,welch_segment=SEGMENT,welch_overlap=64,
        horizons=list(HORIZONS),unknown_policy='retain all supported windows; classified flow is partial, unknown volume and pressure interval explicit',
        target='midpoint change from first second endpoint strictly after decision availability; horizons 30/120 seconds; full supported path and fresh entry/exit',
        groups=dict(A='flow, depth, response ratios at 30/128/512s',B='A plus separate spectra',C='B plus averaged cross spectra'),
        source_manifest_sha256=sha(root/'manifest.json'),source_catalog_sha256=sha(root/'training_catalog.json'),
        implementation_sha256=sha(__file__),captures=len(receipts),excluded_by_session_audit=['2026-05-25'],
        split= dict(train_end='2026-03-01',validation_end='2026-05-01'),evaluation_status='all periods development-exposed')
    save_json(out/'protocol.json',protocol);start=time.perf_counter();reports=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for report in pool.map(materialize_capture,[(r,config['reference'],str(out/'captures')) for r in receipts]):
            reports.append(report)
            if len(reports)%10==0:print(json.dumps(dict(captures=len(reports),rows=sum(r['counts']['retained'] for r in reports))),flush=True)
    frame=pd.concat([pq.read_table(p).to_pandas() for p in sorted((out/'captures').glob('*.parquet'))],ignore_index=True)
    frame=frame.sort_values(['date','t_ns','key']).reset_index(drop=True)
    predictions,results=fit_compare(frame,out)
    summary=dict(status='COMPLETE',captures=len(reports),rows=len(frame),with_unknown=int((frame.A__unknown_fraction_512>0).sum()),
        candidate=sum(r['counts']['candidate'] for r in reports),unsupported=sum(r['counts']['unsupported'] for r in reports),
        late_history=sum(r['counts']['late_history'] for r in reports),seconds=time.perf_counter()-start,
        feature_counts={p:sum(n.startswith(p+'__') for n in frame) for p in ['A','B','C']},
        mature={str(h):int(frame[f'mature_{h}'].sum()) for h in HORIZONS},reports=reports,
        predictions_sha256=sha(out/'predictions.parquet'))
    save_json(out/'summary.json',summary)
    from .response_view import export_view
    export_view(root,out,frame,predictions,receipts)
    print(json.dumps({k:v for k,v in summary.items() if k!='reports'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--market',default='data/derived/full_mbo_v1');p.add_argument('--output',default='reports/response_lab_v1/run_001');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.market,a.output,a.workers)


if __name__=='__main__':main()
