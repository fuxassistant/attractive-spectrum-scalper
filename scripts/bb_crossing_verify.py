"""Independent raw quote checks, saved prediction parity and account replay."""
import argparse
import gc
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition import bb_crossing as bb
from scripts.bb_crossing_build import DATA, REPORT, MARKET, SOURCE, identity
from scripts.flow_context_coverage import sha, save


def direct_geometry(q,indices,scale):
    """Independent batch bar-close formula using the uncompressed quote tape."""
    width=scale*bb.MINUTE;bucket=q.quote_t//width
    close=np.r_[np.flatnonzero(bucket[1:]!=bucket[:-1]),len(bucket)-1]
    cb=bucket[close];where=bucket[indices]
    p=(q.bid[indices].astype(float)+q.ask[indices])/2e9
    middle=np.full(len(indices),np.nan);half=middle.copy()
    for b in np.unique(where):
        at=np.searchsorted(cb,b,side='left')
        if at<19:continue
        js=close[at-19:at]
        if not np.array_equal(cb[at-19:at],np.arange(b-19,b)):continue
        history=(q.bid[js].astype(float)+q.ask[js])/2e9
        center=history.mean();z=history-center;ix=where==b;v=p[ix]-center
        middle[ix]=center+v/20
        half[ix]=2*np.sqrt((np.sum(z*z)+v*v-v*v/20)/19)
    return middle,half,p


def raw_checks():
    receipts=[]
    for date in bb.contract()['sensitivity_dates']:
        day=MonthDay(MARKET/date[:7]/date,verify=True);q=QuoteTape.from_day(day)
        roots=pd.read_parquet(DATA/(date+'_roots.parquet'));pr=pd.read_parquet(DATA/(date+'_profiles.parquet'))
        bank=pd.read_parquet(DATA/(date+'_bank.parquet'))
        checks=0;geom_error=0.;money_error=0.;crosses=0
        for s in bb.SCALES:
            candidates=roots.loc[(roots[f'm{s}_event_ns']>0)&(roots[f'm{s}_delay_s']<1800)]
            choose=candidates.iloc[np.linspace(0,len(candidates)-1,min(12,len(candidates))).astype(int)]
            for r in choose.itertuples():
                j=int(np.searchsorted(q.quote_t,r.t_ns,side='right')-1)
                stop=int(getattr(r,f'm{s}_ordinal'))
                ix=np.arange(j,stop+1);m,h,p=direct_geometry(q,ix,s)
                geom_error=max(geom_error,abs((p[0]-m[0])/h[0]-getattr(r,f'bb{s}_position')),
                    abs(np.log1p(h[0]/.1)-getattr(r,f'bb{s}_log_half_ticks')))
                delta=p[:,None]-m[:,None]-h[:,None]*np.array([-1,0,1])
                if np.any(abs(delta[0])<1e-8):continue
                side=np.sign(delta[0]);opposite=np.where(side>0,delta<-1e-9,delta>1e-9)
                hit=np.flatnonzero(opposite[1:].any(axis=1))+1
                if not len(hit) or j+hit[0]!=stop:raise ValueError(f'Raw first-cross mismatch {date} M{s} {r.t_ns}')
                mask=sum(1<<a for a in np.flatnonzero(opposite[hit[0]]))
                if mask!=getattr(r,f'm{s}_mask'):raise ValueError('Cross packet tie mask mismatch')
                crosses+=1
            pp=pr.loc[(pr.scale==s)&(pr.execution_status==1)]
            pp=pp.iloc[np.linspace(0,len(pp)-1,min(32,len(pp))).astype(int)]
            for r in pp.itertuples():
                e=int(np.searchsorted(q.quote_t,r.t_ns+100_000_000,side='right'))
                x=int(np.searchsorted(q.quote_t,r.request_ns+100_000_000,side='right'))
                if e!=r.entry_ordinal or x!=r.exit_ordinal:raise ValueError('Execution clock mismatch')
                if not q.valid[e:x+1].all() or np.any(q.epoch[e:x+1]!=q.epoch[e]) or np.any(np.diff(q.quote_t[e:x+1])>bb.MINUTE):
                    raise ValueError('Exposure crosses lost support')
                expected=[(q.bid[x]-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[x])/1e9-.3,
                    (q.bid[e:x+1].min()-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[e:x+1].max())/1e9-.3,
                    (q.bid[e:x+1].max()-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[e:x+1].min())/1e9-.3]
                got=[r.net_L,r.net_S,r.exit_mae_L,r.exit_mae_S,r.exit_mfe_L,r.exit_mfe_S]
                money_error=max(money_error,float(np.max(np.abs(np.array(expected)-got))))
                if r.cross_ordinal>=0:
                    c=r.cross_ordinal
                    if q.quote_t[c]<=q.quote_t[e]:raise ValueError('Pre-entry crossing used for mark')
                    want=[(q.bid[c]-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[c])/1e9-.3]
                    money_error=max(money_error,float(np.max(np.abs(np.array(want)-[r.mark_L,r.mark_S]))))
                    rows=bank.loc[(bank.scale==s)&(bank.t_ns>q.quote_t[e])]
                    if int(rows.iloc[0].ordinal)!=c:raise ValueError('Not first post-entry cross')
                checks+=1
        if geom_error>1e-7 or money_error>1e-9:raise ValueError('Independent numerical mismatch')
        receipts.append(dict(date=date,independent_first_cross_checks=crosses,raw_execution_checks=checks,
            geometry_max_error=geom_error,money_max_error=money_error))
        print(json.dumps(receipts[-1]),flush=True)
        del q,day,roots,pr,bank;QUOTE_CACHE.clear();gc.collect()
    save(REPORT/'raw_quote_checks.json',dict(status='PASS',script_sha256=sha(__file__),dates=receipts,
        scope='Five fixed calendar dates; first crossing and native formula from uncompressed quotes, direct extrema/fees/latency'))


def replay_accounts():
    alltrades={};states={}
    for stage in ['A2','B']:
        for s in bb.SCALES:
            key=f'{stage}/M{s}';f=pd.read_parquet(DATA/f'trades_{stage}_m{s}.parquet')
            alltrades[key]=f;states[key]=dict(peak=1500.,drawdown=0.,net=0.,fills=0,max_net_error=0.,points=[])
    dates=sorted(set(d for f in alltrades.values() if len(f) for d in f.date.unique()))
    for date in dates:
        day=MonthDay(MARKET/date[:7]/date);q=QuoteTape.from_day(day)
        for key,tr in alltrades.items():
            if not len(tr):continue
            for r in tr.loc[tr.date==date].itertuples():
                st=states[key];e=r.entry_ordinal;x=r.exit_ordinal
                if r.side=='L':net=(q.bid[e:x+1]-q.ask[e])/1e9-.3
                else:net=(q.bid[e]-q.ask[e:x+1])/1e9-.3
                equity=r.equity_before+net;peaks=np.maximum.accumulate(np.r_[st['peak'],equity])[1:]
                st['drawdown']=max(st['drawdown'],float(np.max(peaks-equity)))
                st['peak']=float(peaks[-1]);st['net']+=float(net[-1]);st['fills']+=1
                st['max_net_error']=max(st['max_net_error'],abs(float(net[-1])-r.net))
                loc=np.unique(np.r_[0,np.flatnonzero(np.diff(q.quote_t[e:x+1]//bb.NS))+1,len(equity)-1])
                # Include extrema and every fill/close; plotting is downsampled,
                # drawdown above is measured at every raw quote publication.
                loc=np.unique(np.r_[loc,np.argmin(equity),np.argmax(equity)])
                st['points'].extend(zip(q.quote_t[e+loc].tolist(),equity[loc].tolist()))
        QUOTE_CACHE.clear();del q,day;gc.collect()
    receipt={}
    for key,st in states.items():
        points=st.pop('points');stage,scale=key.split('/')
        pd.DataFrame(points,columns=['t_ns','equity']).to_parquet(DATA/f'equity_{stage}_{scale.lower()}.parquet',index=False)
        if st['max_net_error']>1e-9:raise ValueError('Account net replay mismatch')
        receipt[key]=st
    save(REPORT/'account_replay.json',dict(status='PASS',scope='Continuous raw-quote drawdown through completed trades only; selected censored trade is not silently completed',
        script_sha256=sha(__file__),accounts=receipt))


def hashes_and_predictions():
    man=json.loads((DATA/'manifest.json').read_text());learning=json.loads((REPORT/'learning_receipt.json').read_text())
    if man['identity']!=identity():raise ValueError('Data identity changed')
    checks=0;max_error=0.
    for r in man['days']:
        for suffix,expected in r['files'].items():
            if sha(DATA/(r['date']+'_'+suffix+'.parquet'))!=expected:raise ValueError('Data file hash drift')
        bank=pd.read_parquet(DATA/(r['date']+'_bank.parquet'))
        if bank.event_id.duplicated().any():raise ValueError('Duplicate physical event ID')
    roots=pd.concat([pd.read_parquet(DATA/(r['date']+'_roots.parquet')) for r in man['days']],ignore_index=True)
    for name,expected in learning['predictions'].items():
        path=DATA/name
        if sha(path)!=expected:raise ValueError('Prediction hash drift')
        pred=pd.read_parquet(path);suffix=name.removeprefix('predictions_').removesuffix('.parquet')
        modelpath=DATA/('model_'+suffix+'.pkl')
        if sha(modelpath)!=learning['models'][modelpath.name]:raise ValueError('Model hash drift')
        with modelpath.open('rb') as src:m=pickle.load(src)
        prohibited=['_id','_y','_event_ns','_ordinal','_maturity_ns','_support_until_ns','_delay_s']
        if any(c.endswith(tuple(prohibited)) for c in m['columns']):raise ValueError('Future metadata in X')
        joined=pred[['date','t_ns']].merge(roots,on=['date','t_ns'],validate='one_to_one',sort=False)
        pp=m['model'].predict_proba(joined[m['columns']].to_numpy(np.float32));classes=m['model'].classes_.astype(int)
        error=float(np.max(np.abs(pp-pred[[f'p{k}' for k in classes]].to_numpy())))
        max_error=max(max_error,error);checks+=1
    if max_error>1e-12:raise ValueError('Saved model probability parity')
    result=dict(status='PASS',prediction_pairs=checks,prediction_max_error=max_error,days=len(man['days']),
        manifest_sha256=sha(DATA/'manifest.json'),learning_receipt_sha256=sha(REPORT/'learning_receipt.json'),
        summary_sha256=sha(REPORT/'summary.json'),script_sha256=sha(__file__))
    save(REPORT/'numerical_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['raw','accounts','validate']);args=parser.parse_args()
    {'raw':raw_checks,'accounts':replay_accounts,'validate':hashes_and_predictions}[args.command]()
