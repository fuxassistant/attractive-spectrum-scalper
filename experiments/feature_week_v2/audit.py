"""Independent arithmetic, dependency clocks, batch/restart and loader audit."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,OUT,PREP,DATES,WEEK,FAMILIES,raw_for,source_for,read,NS,window_raw
import numpy as np
import pandas as pd
from spectrum.io import save_json,sha
from spectrum.features.pipeline import compute
from spectrum.features.stream import FeatureStream
from spectrum.features.families.spectral_activity import white_reference
from spectrum.representation.representation import scheduled_open,MINUTE
from spectrum.representation.session_store import open_session_store


def quantiles(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(zip(('p50','p95','p99','max'),map(float,np.quantile(a,[.5,.95,.99,1])))) if len(a) else {}


def day_audit(run,date):
    source=pd.read_parquet(PREP/date/'seconds.parquet');first=int(source.t_ns.iloc[0])
    i=DATES.index(date)
    left=pd.read_parquet(PREP/DATES[i-1]/'seconds.parquet',filters=[('t_ns','>=',first-3700*NS)])
    f=pd.concat([left,source],ignore_index=True)
    x=pd.read_parquet(run/'days'/date/'features.parquet');fast=pd.read_parquet(run/'days'/date/'fast.parquet')
    offset=len(left);names=[n for n in x if n+'__known' in x]
    assert len(names)==480
    assert f.t_ns.is_monotonic_increasing and f.t_ns.is_unique
    assert f.available_ns.is_monotonic_increasing and (f.available_ns>=f.t_ns).all()
    cadence=[];errors={};pair=source.t_ns.diff().eq(NS)&source.epoch.eq(source.epoch.shift())
    for name in names:
        a=x[name].to_numpy();k=x[name+'__known'].to_numpy(bool)
        assert np.isfinite(a).all() and np.all(a[~k]==0),name
        paired=pair.to_numpy()&k&np.r_[False,k[:-1]]
        changes=paired&~np.isclose(a,np.r_[a[0],a[:-1]],atol=1e-9,rtol=1e-10)
        cadence.append(dict(date=date,name=name,rows=len(a),known=int(k.sum()),pairs=int(paired.sum()),
                            changes=int(changes.sum()),distinct=int(pd.Series(a[k]).nunique())))
    def check(name,ref,tol=1e-6):
        ref=np.asarray(ref)[offset:];k=x[name+'__known'].to_numpy(bool)
        err=abs(x[name].to_numpy()[k]-ref[k]);maximum=float(err.max()) if len(err) else 0.
        assert maximum<tol,(date,name,maximum)
        errors[name]=dict(rows=len(err),max_error=maximum)
    check('spread_ticks',(f.ask-f.bid)/.1)
    check('depth_balance_12',(f.depth_bid_12-f.depth_ask_12)/(f.depth_bid_12+f.depth_ask_12))
    for n in (1,5,15,30,60,128):
        check(f'displacement_{n}s',(f.mid-f.mid.shift(n))/.1)
        check(f'path_{n}s',f.c_path_ticks-f.c_path_ticks.shift(n),1e-5)
    for side in ('buy','sell','unknown'):check(f'trade_{side}_30s',f['c_trade_'+side]-f['c_trade_'+side].shift(30))
    b,s,u=[f['c_trade_'+side].diff().rolling(30).sum() for side in ('buy','sell','unknown')]
    for name,ref in [('unknown_fraction',u/(b+s+u)),('observed_imbalance',(b-s)/(b+s+u)),
                     ('signed_lower_bound',(b-s-u)/(b+s+u)),('signed_upper_bound',(b-s+u)/(b+s+u))]:
        check('activity_30s_'+name,ref)
    tr=window_raw(date,'trades',first-3700*NS,int(source.t_ns.iloc[-1])-1)
    ti=tr.t_ns.to_numpy();price=tr.price_nano.to_numpy()/NS;q=tr.quantity.to_numpy(float)
    rng=np.random.default_rng(29);vwap=[];dft=[]
    for n in (60,300,900,3600):
        stem=f'vwap_exact_{n}s_';candidates=np.flatnonzero(x[stem+'center__known'])
        # Explicitly include the repaired midnight seam as well as random hours.
        indices=np.unique(np.r_[candidates[:5],rng.choice(candidates,min(25,len(candidates)),replace=False)])
        for j in indices:
            t=int(source.t_ns.iloc[j]);lo,hi=np.searchsorted(ti,[t-n*NS,t])
            p,w=price[lo:hi],q[lo:hi];mean=np.average(p,weights=w)
            sigma=np.sqrt(np.average((p-mean)**2,weights=w))/.1
            em,es=abs(mean-x[stem+'center'].iloc[j]),abs(sigma-x[stem+'sigma_ticks'].iloc[j])
            assert em<1e-7 and es<1e-5,(date,n,em,es)
            vwap.append((float(em),float(es)))
    for n in (32,64,128):
        for channel in ('mid_change','total_trade','observed_signed_trade','unknown_trade'):
            name=f'fft_{channel}_{n}s_power';candidates=np.flatnonzero(x[name+'__known'])
            for j in rng.choice(candidates,min(15,len(candidates)),replace=False):
                history=f.iloc[offset+j-n:offset+j+1]
                if channel=='mid_change':z=np.diff(history.mid.to_numpy())/.1
                else:
                    b,s,u=[np.diff(history['c_trade_'+side].to_numpy()) for side in ('buy','sell','unknown')]
                    z={'total_trade':b+s+u,'observed_signed_trade':b-s,'unknown_trade':u}[channel]
                w=.5-.5*np.cos(2*np.pi*np.arange(n)/n)
                p=abs(np.fft.rfft((z-z.mean())*w))**2/(w@w);p[1:-1]*=2
                err=abs(p[1:].sum()/n-x[name].iloc[j]);assert err<1e-7,(date,name,err)
                dft.append(float(err))
    unchanged=fast.ret1_ticks.eq(0)&fast.ledger_known
    churn=unchanged&((fast.trade_quantity>0)|(fast.bid_near_add>0)|(fast.ask_near_add>0)|(fast.bid_near_remove>0)|(fast.ask_near_remove>0))
    matrix=x[[n+'__known' for n in names]].to_numpy(bool)
    # Complete-case admission selects a different activity population.
    all_known=matrix.all(axis=1);admitted=x.admissible.to_numpy(bool)
    cohorts={}
    for label,k in [('all_480_known',all_known),('admitted_without_all',admitted&~all_known)]:
        cohorts[label]=dict(rows=int(k.sum()),trade_quantity=float(fast.trade_quantity[k].sum()))
    result=dict(date=date,rows=len(x),formulas=errors,vwap_checks=len(vwap),vwap_center_error=max(v[0] for v in vwap),
        vwap_sigma_ticks_error=max(v[1] for v in vwap),dft_checks=len(dft),dft_max_error=max(dft),
        price_known=int(fast.known.sum()),admitted=int(admitted.sum()),latency_ms=quantiles(((source.available_ns-source.t_ns)/1e6)[fast.known]),
        strict_signed128_known=int(x.fft_signed_trade_128s_power__known.sum()),
        total128_known=int(x.fft_total_trade_128s_power__known.sum()),all_features_known=int(all_known.sum()),
        unchanged_mid=int(unchanged.sum()),unchanged_with_flow=int(churn.sum()),cohorts=cohorts)
    return result,pd.DataFrame(cadence)


def coverage(run):
    frames=[pd.read_parquet(run/'days'/d/'fast.parquet',columns=['t_ns','available_ns','known','ledger_known','trade_known','epoch','trade_quantity']) for d in DATES]
    f=pd.concat(frames,ignore_index=True).set_index('t_ns')
    lo=int(pd.Timestamp('2026-08-23 22:00',tz='UTC').value);hi=int(pd.Timestamp('2026-08-28 21:00',tz='UTC').value)
    grid=np.arange(lo+NS,hi+NS,NS,dtype=np.int64);expected=scheduled_open(grid-1);a=f.reindex(grid)
    missing=expected&a.epoch.isna();unknown=expected&~a.known.eq(True)
    assert not missing.any()
    assert int(unknown.sum())==5,list(pd.to_datetime(grid[unknown],utc=True))
    tape=pd.concat([pd.read_parquet(raw_for(d)/'trades.parquet',filters=[('t_ns','>=',lo),('t_ns','<',hi)]) for d in DATES],ignore_index=True)
    in_session=scheduled_open(tape.t_ns)
    assert int(a.trade_quantity[expected].sum())==int(tape.quantity[in_session].sum())
    return dict(expected_open_seconds=int(expected.sum()),present_open_seconds=int((expected&~a.epoch.isna()).sum()),
        known_price_seconds=int((expected&a.known.eq(True)).sum()),ledger_known_seconds=int((expected&a.ledger_known.eq(True)).sum()),
        trade_known_seconds=int((expected&a.trade_known.eq(True)).sum()),
        unknown_price_endpoints=[str(t) for t in pd.to_datetime(grid[unknown],utc=True)],
        exact_trade_quantity=int(tape.quantity[in_session].sum()),exact_trade_rows=int(in_session.sum()),
        missing_endpoints=0,utc_handoffs=len(read(PREP/'manifest.json')['handoffs']))


def partitions(run):
    date=WEEK[0];left=pd.read_parquet(PREP/'2026-08-23/seconds.parquet').tail(4000)
    right=pd.read_parquet(PREP/date/'seconds.parquet').head(5000)
    f=pd.concat([left,right],ignore_index=True);reference=compute(f,FAMILIES)
    stream=FeatureStream(FAMILIES,source_id='v2-boundary');maximum=0.;pairs=0
    for lo,hi in ((0,4200),(4200,6500),(6500,9000)):
        r=stream.feed(f.iloc[lo:hi])
        for n in reference.values:
            assert np.array_equal(r['known'][n],reference.masks[n][lo:hi]),n
            err=float(np.max(abs(r['values'][n]-reference.values[n][lo:hi])))
            maximum=max(maximum,err);pairs+=len(r['values'][n])
            assert np.allclose(r['values'][n],reference.values[n][lo:hi],atol=2e-5,rtol=1e-6),(n,err)
        stream=FeatureStream.restore(stream.checkpoint(),expected_source_id='v2-boundary')
    stored=pd.read_parquet(run/'days'/date/'features.parquet').head(5000)
    for n in reference.values:
        assert np.array_equal(stored[n+'__known'],reference.masks[n][4000:]),n
        assert np.allclose(stored[n],reference.values[n][4000:],atol=2e-5,rtol=1e-6),n
    short=compute(f.iloc[:6000],FAMILIES)
    for n in reference.values:
        assert np.array_equal(short.masks[n],reference.masks[n][:6000]),n
        assert np.allclose(short.values[n],reference.values[n][:6000],atol=1e-8,rtol=1e-9),n
    return dict(status='PASS',channels=len(reference.values),comparisons=pairs,max_error=maximum,
                masks_equal=True,checkpoint=True,future_prefix=True,physical_midnight_overlap=True)


def minutes_audit(run):
    m=pd.read_parquet(run/'minutes.parquet');core=pd.read_parquet(run/'core_m1.parquet');errors=[];matched=[]
    for scale in (1,3,6):
        for j in np.flatnonzero(m['bb'+str(scale)].notna())[::41]:
            row=m.iloc[j];p=m[(m.t_ns<=int(m.t_ns.iloc[j]))&m.t_ns.mod(scale*MINUTE).eq(0)&m.segment.eq(row.segment)].tail(20)
            assert len(p)==20
            error=max(abs(p.close.mean()-row['bb'+str(scale)]),abs(2*p.close.std(ddof=1)-row['bb'+str(scale)+'_half']))
            assert error<1e-6,(scale,error)
            errors.append(float(error))
        # Same completed native time, sample SD and supported history.
        name=f'bb_m{scale}_width_scale'
        joined=core.merge(m[['t_ns','bb'+str(scale)+'_half']],on='t_ns')
        k=joined[name+'_known'].eq(1)&joined['bb'+str(scale)+'_half'].notna()&joined.t_ns.mod(scale*MINUTE).eq(0)
        a=joined.loc[k,name]*joined.loc[k,'scale'];b=joined.loc[k,'bb'+str(scale)+'_half']
        assert np.allclose(a,b,atol=3e-6,rtol=1e-7),(scale,float(abs(a-b).max()))
        matched.extend(abs(a-b).tolist())
    assert (core[[n for n in core if n.startswith('previous_week_')]]==0).all().all()
    return dict(independent_bb_checks=len(errors),max_error=max(errors),core_representation_pairs=len(matched),
                core_representation_max_error=max(matched),ddof=1,clock='COMPLETED_NATIVE',
                developing_crossing_bank='Separate clock: 19 completed closes plus current quote; unchanged')


def spectral_controls():
    rng=np.random.default_rng(2901);n=128;w=.5-.5*np.cos(2*np.pi*np.arange(n)/n);freq=np.fft.rfftfreq(n)
    selected=(freq>=1/8)&(freq<=1/4);ref=white_reference(n,selected)
    def power(a):
        p=abs(np.fft.rfft((a-a.mean(axis=-1,keepdims=True))*w,axis=-1))**2/(w@w)
        p[...,1:-1]*=2;p[...,0]=0;return p
    null=power(rng.normal(size=(8192,n)));p=power(np.cos(2*np.pi*np.arange(n)/6))
    ratio=float(null[:,selected].sum()/null.sum()/ref);positive=float(p[selected].sum()/p.sum()/ref)
    assert abs(ratio-1)<.02 and positive>1.7,(ratio,positive)
    return dict(null='8192 iid Gaussian windows; aggregate ratio of powers',white_reference=ref,
                null_relative_power=ratio,positive_6second_cycle=positive,
                interpretation='Control of band-width/window bias, not a probability, alpha or proof of tradable cycles')


def book_snapshot(out):
    from collections import defaultdict
    from spectrum.market.replay import MarketReplay
    from spectrum.market.continuity import book_identity
    path=Path(source_for('2026-08-23')['market'])/'checkpoint.json'
    replay=MarketReplay.restore(read(path));proof=read(PREP/'manifest.json')['handoffs']['2026-08-24']
    assert book_identity(replay.core.engine)==proof['book_sha256']
    levels=defaultdict(lambda:[0,0])
    for order in replay.core.engine.orders.values():
        value=levels[(order.side,order.price_nano)];value[0]+=order.size;value[1]+=1
    result=dict(available_ns=replay.core.last_record_ns,book_sha256=proof['book_sha256'],orders=len(replay.core.engine.orders),
        levels=[dict(side=s,price_nano=p,quantity=v[0],orders=v[1]) for (s,p),v in levels.items()],source=str(path),
        source_sha256=sha(path),identical_next_snapshot=True)
    save_json(out/'l3_snapshot.json',result)
    return {k:v for k,v in result.items() if k!='levels'}


QUERIES={'2026-08-24':['2026-08-23 22:02:00.5','2026-08-24 00:00:01.5','2026-08-24 02:30:00.5','2026-08-24 12:00:00.5'],
         '2026-08-25':['2026-08-25 12:00:00.5'],'2026-08-26':['2026-08-26 12:00:00.5'],
         '2026-08-27':['2026-08-27 12:00:00.5'],
         '2026-08-28':['2026-08-28 13:59:59.5','2026-08-28 14:00:03.5','2026-08-28 14:00:07.5','2026-08-28 14:03:00.5']}


def snapshots(run,out):
    directory=out/'snapshots';directory.mkdir(exist_ok=True);receipts=[]
    for date,queries in QUERIES.items():
        store=open_session_store(run/'stores'/date)
        for text in queries:
            t=int(pd.Timestamp(text,tz='UTC').value);o=store.observe(t)
            idx=int(np.searchsorted(store.quotes.available_ns.to_numpy(),t,side='right'))-1;q=store.quotes.iloc[idx]
            assert o.market['source_ordinal']==q.source_ordinal
            assert o.market['bid']==float(q.bid_nano)/NS and o.market['ask']==float(q.ask_nano)/NS
            for a in (o.fast,o.context):
                assert np.all(a['available_ns']<=t) and np.all(a['t_ns']<=t)
                assert np.isfinite(a['values']).all() and np.isfinite(a['normalized']).all()
                assert np.all(a['values'][~a['known']]==0)
            for a in (o.second_features,o.minute_features):
                assert a['available_ns']<=t and a['t_ns']<=t
            name=pd.Timestamp(t,tz='UTC').strftime('%Y%m%dT%H%M%S')
            arrays={}
            for group,a in [('fast',o.fast),('context',o.context),('second',o.second_features),('minute',o.minute_features)]:
                for n in ('values','known','normalized','t_ns','available_ns'):
                    if n in a:arrays[group+'_'+n]=np.asarray(a[n])
            np.savez_compressed(directory/(name+'.npz'),**arrays)
            r=dict(query_utc=text,t_ns=t,market=o.market,receipt=o.receipt,
                   shape={n:list(a['values'].shape) for n,a in [('fast',o.fast),('context',o.context)]},
                   normalization=o.fast['normalization'],names=dict(fast=o.fast['names'],context=o.context['names'],
                       second=o.second_features['names'],minute=o.minute_features['names']),
                   second_values=o.second_features['values'].tolist(),second_known=o.second_features['known'].tolist(),
                   file=name+'.npz',sha256=sha(directory/(name+'.npz')))
            r['context_quality']=o.context['quality']
            r['context_quality']['values']=r['context_quality']['values'].tolist()
            r['context_quality']['known']=r['context_quality']['known'].tolist()
            save_json(directory/(name+'.json'),r);receipts.append({k:v for k,v in r.items() if k not in ('names','second_values','second_known')})
        del store;gc.collect()
        print(json.dumps(dict(stage='loader',date=date)),flush=True)
    invalid=[r for r in receipts if r['query_utc']=='2026-08-28 14:00:03.5'][0]
    assert not invalid['market']['known'] and invalid['market']['reason']=='INVALID_BOOK'
    return dict(status='PASS',queries=len(receipts),asof_native_quote_exact=True,invalid_native_quote_not_skipped=True,receipts=receipts)


def main(name):
    run=OUT/name;out=OUT/'audit';out.mkdir(exist_ok=True);start=time.perf_counter()
    assert read(run/'manifest.json')['status']=='COMPLETE'
    results=[];cadence=[]
    for d in WEEK:
        r,c=day_audit(run,d);results.append(r);cadence.append(c)
        print(json.dumps(dict(stage='formulas',date=d,vwap=r['vwap_checks'],dft=r['dft_checks'])),flush=True)
        gc.collect()
    pd.concat(cadence,ignore_index=True).to_parquet(out/'cadence.parquet',index=False)
    r=dict(status='PASS',dataset=name,days=results,coverage=coverage(run),partitions=partitions(run),
           minutes=minutes_audit(run),spectral_controls=spectral_controls(),loader=snapshots(run,out),
           l3_snapshot=book_snapshot(out),
           lineage=verify_lineage(run),
           source_manifest_sha256=sha(run/'manifest.json'),elapsed_seconds=time.perf_counter()-start)
    save_json(out/'audit.json',r);print(json.dumps({k:r[k] for k in ('status','coverage','partitions','minutes','spectral_controls','elapsed_seconds')}),flush=True)
    compare_parallel(run,OUT/'benchmark_serial',out)


def verify_lineage(run):
    manifest=read(run/'manifest.json');prepared=read(PREP/'manifest.json');checked={}
    assert prepared['status']=='COMPLETE'
    assert sha(PREP/'manifest.json')==manifest['preparation_sha256']
    for path,digest in prepared['source_hashes'].items():
        assert sha(path)==digest,path
        checked[path]=digest
    for date,files in prepared['files'].items():
        for name,digest in files.items():
            path=PREP/date/name;assert sha(path)==digest,str(path);checked[str(path)]=digest
    for d in manifest['days']:
        receipt=run/'days'/d['date']/'manifest.json';assert read(receipt)==d
        for name,digest in d['files'].items():
            path=receipt.parent/name;assert sha(path)==digest,str(path);checked[str(path)]=digest
    for date in WEEK:
        s=read(run/'stores'/date/'manifest.json')
        assert s['source_identity']['preparation_sha256']==manifest['preparation_sha256']
        for day,digest in s['source_identity']['day_receipts'].items():
            assert sha(run/'days'/day/'manifest.json')==digest
        for name,digest in s['files'].items():
            path=run/'stores'/date/name;assert sha(path)==digest,str(path);checked[str(path)]=digest
    return dict(status='PASS',unique_payloads=len(checked),source_payloads=len(prepared['source_hashes']),
                complete_source_partitions=True,prepared_manifest_bound=True,day_and_session_receipts_bound=True,
                payload_sha256=checked)


def compare_parallel(run,serial,out):
    items=[]
    for d in DATES:
        for name in ('features.parquet','fast.parquet'):
            a=run/'days'/d/name;b=serial/'days'/d/name
            digest=sha(a);assert digest==sha(b),(d,name)
            items.append(dict(date=d,file=name,sha256=digest,bitwise_equal=True))
    parallel_manifest=read(run/'manifest.json');serial_manifest=read(serial/'manifest.json')
    result=dict(status='PASS',files=len(items),bitwise_equal=True,details=items,
        serial_day_seconds=serial_manifest['day_seconds'],parallel_day_seconds=parallel_manifest['day_seconds'],
        speedup=serial_manifest['day_seconds']/parallel_manifest['day_seconds'],full_parallel_seconds=parallel_manifest['elapsed_seconds'],
        context_seconds=parallel_manifest['context']['elapsed_seconds'])
    save_json(out/'parallel_parity.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='final_parallel2');a=p.parse_args();main(a.name)
