"""Frozen one-week line/band comparison, with genuine trade-price VWAP."""
from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
from spectrum.cognition import bb_crossing as bb
from spectrum.cognition.vwap_compare import TradePrefix,support_start,weighted_native_closes,statistics,NS
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay,QUOTE_CACHE
from scripts.flow_context_coverage import save,sha
from scripts.vwap_week_source import OUT,REPORT

DATES=[f'2026-02-{d:02}' for d in range(2,7)]
FACTORS=(.5,.75,1.,1.25,1.5,2.)


def main():
    source=json.loads((REPORT/'source_receipt.json').read_text())
    if source['status']!='PASS':raise ValueError('Unverified trades')
    for r in source['captures']:
        if sha(r['file'])!=r['sha256']:raise ValueError('Trade source changed')
    checks=[];receipts=[]
    for date in DATES:
        day=MonthDay(Path('data/derived/basket_multimonth_v1/2026-02')/date,verify=True)
        q=QuoteTape.from_day(day);k=bb.reduce_quotes(q)
        qt=q.quote_t[k];p=(q.bid[k].astype(float)+q.ask[k])/2e9
        valid=q.valid[k]&(q.ask[k]>q.bid[k])&(q.bid[k]>0)
        starts=support_start(qt,valid,q.epoch[k])
        grid=np.arange(int(day.meta['london_ns']),q.deadline_ns,NS,dtype=np.int64)
        j=np.searchsorted(qt,grid,side='right')-1
        frames=[]
        parts=[]
        for part in day.meta['quote_parts']:
            raw=json.loads((Path(part['root'])/'manifest.json').read_text())
            tr=pd.read_parquet(OUT/(raw['input']['selection']['date']+'_trades.parquet'))
            parts.append(tr.loc[(tr.available_ns>=part['start_ns'])&(tr.available_ns<part['end_ns'])])
        tr=pd.concat(parts,ignore_index=True)
        prefix=TradePrefix(tr.available_ns,tr.price_nano.to_numpy()/1e9,tr.quantity)
        proxy=TradePrefix(tr.available_ns,tr.mid,tr.quantity)
        for s in bb.SCALES:
            mid,half,seg=bb.native_geometry(qt,p,valid,q.epoch[k],s)
            f=pd.DataFrame(dict(date=date,t_ns=grid,scale=s,mid=p[j],bb=mid[j],bb_half=half[j],segment=seg[j]))
            left=(grid//(s*60*NS)-19)*(s*60*NS)
            for name,pref in [('vwap',prefix),('vwap_mid',proxy)]:
                mu,h,age,vol=pref.query(left,grid)
                known=(seg[j]>=0)&(left>=starts[j])&valid[j]&(grid-qt[j]<=60*NS)
                for suffix,v in [('',mu),('_half',h),('_age',age)]:f[name+suffix]=np.where(known,v,np.nan)
                f['volume']=np.where(known,vol,np.nan)
            mu,h,age,bbage=weighted_native_closes(qt,p,grid,s,prefix)
            f['bb_age']=np.where(seg[j]>=0,bbage,np.nan)
            for suffix,v in [('',mu),('_half',h),('_age',age)]:f['vwma'+suffix]=np.where(f.vwap.notna(),v,np.nan)
            for factor in FACTORS:
                window=int(s*20*60*factor);name=f'rolling_{window}'
                leftroll=grid-window*NS+1
                mu,h,age,vol=prefix.query(leftroll,grid)
                known=(leftroll>=starts[j])&valid[j]&(grid-qt[j]<=60*NS)
                for suffix,v in [('',mu),('_half',h),('_age',age)]:f[name+suffix]=np.where(known,v,np.nan)
            # Fixed calendar-spaced, independent raw-trade numerical checks.
            eligible=np.flatnonzero(f.vwap.notna())
            for idx in eligible[np.linspace(0,len(eligible)-1,20,dtype=int)]:
                sample=tr.loc[(tr.available_ns>=left[idx])&(tr.available_ns<=grid[idx])]
                weights=sample.quantity.to_numpy(float);prices=sample.price_nano.to_numpy()/1e9
                mean=np.average(prices,weights=weights);half0=2*np.sqrt(np.average((prices-mean)**2,weights=weights))
                err=max(abs(mean-f.vwap.iloc[idx]),abs(half0-f.vwap_half.iloc[idx]))
                if err>1e-6:raise ValueError('Direct T calculation mismatch')
                # Future truncation changes neither moments nor sampled result.
                cut=tr.available_ns<=grid[idx]
                pre=TradePrefix(tr.available_ns[cut],tr.price_nano.to_numpy()[cut]/1e9,tr.quantity[cut])
                causal=pre.query(np.array([left[idx]]),np.array([grid[idx]]))
                if abs(causal[0][0]-f.vwap.iloc[idx])>1e-9:raise ValueError('Future dependency')
                checks.append(dict(date=date,scale=s,t_ns=str(grid[idx]),max_error=err))
            frames.append(f)
        frame=pd.concat(frames,ignore_index=True);path=OUT/(date+'_comparison.parquet');frame.to_parquet(path,index=False)
        receipts.append(dict(date=date,rows=len(frame),file=str(path),sha256=sha(path),market_manifest_sha256=sha(day.path/'manifest.json')))
        print(json.dumps(dict(date=date,rows=len(frame))),flush=True)
        del q,day,frames,frame,tr;QUOTE_CACHE.clear();gc.collect()
    allf=pd.concat([pd.read_parquet(r['file']) for r in receipts],ignore_index=True)
    results={};matches={};daily={}
    for s in bb.SCALES:
        f=allf.loc[allf.scale==s].reset_index(drop=True)
        # Never compute slope differences across different sessions.
        f['segment']=pd.factorize(list(zip(f.date,f.segment)))[0]
        results[str(s)]={name:statistics(f,name) for name in ['vwma','vwap_mid','vwap',f'rolling_{s*1200}']}
        candidates=[f'rolling_{int(s*1200*factor)}' for factor in FACTORS]
        fit=f.date.isin(DATES[:2]);common=np.isfinite(f.bb)
        for name in candidates:common &= np.isfinite(f[name])
        errors={name:float(np.sqrt(np.mean((f.loc[fit&common,name]-f.loc[fit&common,'bb'])**2))) for name in candidates}
        chosen=min(errors,key=errors.get)
        held=f.loc[~fit&common].copy().reset_index(drop=True)
        # Metrics here use a discontinuous common cohort only for center RMSE.
        matches[str(s)]=dict(calendar_minutes=s*20,calibration_dates=DATES[:2],evaluation_dates=DATES[2:],
            common_calibration_rows=int((fit&common).sum()),common_evaluation_rows=len(held),calibration_rmse=errors,
            selected_minutes=int(chosen.split('_')[1])//60,
            heldout_selected_rmse=float(np.sqrt(np.mean((held[chosen]-held.bb)**2))),
            heldout_calendar_rmse=float(np.sqrt(np.mean((held[f'rolling_{s*1200}']-held.bb)**2))))
        daily[str(s)]={date:statistics(f.loc[f.date==date].reset_index(drop=True),'vwap') for date in DATES}
    summary=dict(status='COMPLETE',dates=DATES,symbol='GCJ6',sample_seconds=1,results=results,scale_matching=matches,daily=daily,
        source_receipt_sha256=sha(REPORT/'source_receipt.json'),files=receipts,independent_checks=len(checks),max_direct_error=max(x['max_error'] for x in checks),
        contracts_sha256=sha('contracts/vwap_bands_week_v1.json'),code_sha256={p:sha(p) for p in ['scripts/vwap_week_compare.py','spectrum/cognition/vwap_compare.py']})
    save(REPORT/'summary.json',summary);save(REPORT/'direct_checks.json',checks)
    print(json.dumps(dict(status='COMPLETE',checks=len(checks),matching=matches)),flush=True)


if __name__=='__main__':main()
