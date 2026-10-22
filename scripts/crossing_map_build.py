"""Full-publication BB/VWAP line bank, paired path marks, and fixed risk queries."""
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from spectrum.cognition import bb_crossing as bb
from spectrum.cognition.crossing_map import SCALES,vwap_geometry,lookup_line,first_passage,geometry_id,NS
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay,QUOTE_CACHE
from scripts.bb_crossing_session import all_marks
from scripts.flow_context_coverage import save,sha

OUT=Path('data/derived/crossing_map_target_v1');REPORT=Path('reports/crossing_map_target_v1')
VW=Path('data/derived/vwap_bands_week_v1');OLD=Path('data/derived/bb_crossing_bank_v1')
DATES=[f'2026-02-{d:02}' for d in range(2,7)]


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    identity={p:sha(p) for p in ['contracts/crossing_map_target_v1.json','spectrum/cognition/crossing_map.py',__file__]}
    src=json.loads(Path('reports/vwap_bands_week_v1/source_receipt.json').read_text())
    for r in src['captures']:
        if sha(r['file'])!=r['sha256']:raise ValueError('Changed exact trades')
    receipts=[];checks=[]
    for date in DATES:
        rp=OUT/(date+'.json')
        if rp.exists():
            r=json.loads(rp.read_text())
            if r['identity']!=identity:raise ValueError('Build changed; preserve old run')
            for name,h in r['files'].items():
                if sha(OUT/name)!=h:raise ValueError('Changed output')
            receipts.append(r);checks.extend(r['checks']);continue
        day=MonthDay(Path('data/derived/basket_multimonth_v1/2026-02')/date,verify=True);q=QuoteTape.from_day(day)
        raw=[]
        for part in day.meta['quote_parts']:
            meta=json.loads((Path(part['root'])/'manifest.json').read_text())
            if sha(Path(part['root'])/'quotes.parquet')!=part['quote_file']['sha256']:raise ValueError('Changed raw quotes')
            a=pd.read_parquet(VW/(meta['input']['selection']['date']+'_trades.parquet'))
            raw.append(a.loc[(a.available_ns>=part['start_ns'])&(a.available_ns<part['end_ns'])])
        tr=pd.concat(raw,ignore_index=True)
        trade_ord=np.searchsorted(day.source_ordinal,tr.source_ordinal.to_numpy(),side='left')
        if not np.array_equal(q.quote_t[trade_ord],tr.available_ns):raise ValueError('Trade publication mapping')
        k=np.union1d(bb.reduce_quotes(q),trade_ord)
        qt=q.quote_t[k];price=(q.bid[k].astype(float)+q.ask[k])/2e9
        valid=q.valid&(q.bid>0)&(q.ask>q.bid)
        br=bb.execution_breaks(q.quote_t,valid,q.epoch);blocks=bb.extremum_blocks(q.bid,q.ask,k)
        times=np.arange(q.entry_start_ns,q.deadline_ns,30*NS,dtype=np.int64)
        decision=np.searchsorted(q.quote_t,times,side='right')-1
        entry=np.searchsorted(q.quote_t,times+100_000_000,side='right')
        safe=np.minimum(entry,len(q.quote_t)-1)
        eok=(decision>=0)&(entry<len(q.quote_t))&valid[np.maximum(decision,0)]&valid[safe]&(q.quote_t[safe]<=times+2*NS)&(entry<br[np.maximum(decision,0)])
        entry=np.where(eok,entry,-1);esafe=np.maximum(entry,0)
        roots=pd.DataFrame(dict(date=date,t_ns=times,entry_ordinal=entry,entry_ns=np.where(eok,q.quote_t[esafe],0),
            decision_spread=(q.ask[decision]-q.bid[decision])/1e9,entry_spread=np.where(eok,(q.ask[esafe]-q.bid[esafe])/1e9,np.nan)))
        obs_end=np.minimum(q.deadline_ns,np.minimum(q.quote_t[np.minimum(br[esafe],len(q.quote_t)-1)],q.quote_t[np.minimum(br[esafe]-1,len(q.quote_t)-1)]+60*NS))
        # Final source publication has explicit exclusive coverage beyond it.
        obs_end=np.where(br[esafe]==len(q.quote_t),min(q.deadline_ns,q.coverage_end_ns),obs_end)
        roots['quote_support_end_ns']=np.where(eok,obs_end,0)
        origin_j=np.searchsorted(qt,times,side='right')-1
        profiles=[];banks=[];scene=pd.DataFrame(dict(t_ns=times,mid=price[origin_j]));day_checks=[]
        for s in SCALES:
            middle,half,support=bb.native_geometry(qt,price,valid[k],q.epoch[k],s)
            if s==1:
                roots['risk_R']=np.maximum.reduce([2*(roots.decision_spread.to_numpy()+.3),.1*half[origin_j],np.full(len(times),.5)])
            vm,vh,vs=vwap_geometry(q.quote_t,k,trade_ord,tr.available_ns.to_numpy(),tr.price_nano.to_numpy()/1e9,tr.quantity.to_numpy(),s,support)
            for family,m,h,sup in [('BB',middle,half,support),('VWAP',vm,vh,vs)]:
                ends=bb.support_ends(qt,sup,q.deadline_ns)[origin_j]
                ready=(sup[origin_j]>=0)&eok&(q.quote_t[esafe]<ends)
                row,role,direction,cause,dp,dl=bb.crossing_rows(qt,price,m,h,sup)
                source_id=geometry_id(family,s,day.meta['quote_parts'],src['source_member_sha256'])
                bank=pd.DataFrame(dict(ordinal=k[row],t_ns=qt[row],role=role,direction=direction,mid=price[row],line=m[row]+(role-1)*h[row],
                    cause=[bb.CAUSES[c] for c in cause],family=family,scale=s))
                bank['event_id']=[f'{source_id}/{o}/{r}/{d}' for o,r,d in zip(bank.ordinal,role,direction)]
                banks.append(bank)
                scene[f'{family}{s}_middle']=m[origin_j];scene[f'{family}{s}_half']=h[origin_j]
                if family=='VWAP':
                    eligible=np.flatnonzero(sup>=0)
                    for idx in eligible[np.linspace(0,len(eligible)-1,8,dtype=int)]:
                        start=(qt[idx]//(s*60*NS)-19)*(s*60*NS)
                        use=(trade_ord<=k[idx])&(tr.available_ns.to_numpy()>=start)
                        p=tr.price_nano.to_numpy()[use]/1e9;w=tr.quantity.to_numpy()[use]
                        direct=np.average(p,weights=w);sd=2*np.sqrt(np.average((p-direct)**2,weights=w))
                        error=max(abs(direct-m[idx]),abs(sd-h[idx]))
                        if error>1e-6:raise ValueError('Ordinal VWAP check')
                        day_checks.append(dict(type='vwap_direct',scale=s,error=error))
                for r in range(3):
                    b=bank.loc[bank.role==r].reset_index(drop=True);ords=b.ordinal.to_numpy()
                    crosses=lookup_line(ords,entry,ready,ends,q.quote_t);known=crosses>=0
                    vals=all_marks(q.bid,q.ask,k,blocks,entry,crosses,known,br)
                    x=np.searchsorted(q.quote_t,q.quote_t[np.maximum(crosses,0)]+100_000_000,side='right');xsafe=np.minimum(x,len(q.quote_t)-1)
                    xok=known&(x<len(q.quote_t))&(x<br[esafe])&(q.quote_t[xsafe]<=q.quote_t[np.maximum(crosses,0)]+2*NS)
                    ix=np.searchsorted(ords,np.maximum(crosses,0));ix=np.minimum(ix,max(0,len(b)-1))
                    f=pd.DataFrame(dict(date=date,t_ns=times,family=family,scale=s,role=r,entry_ordinal=entry,entry_ns=roots.entry_ns,
                        ready=ready,support_end_ns=np.where(ready,ends,0),cross_ordinal=crosses,
                        cross_ns=np.where(known,q.quote_t[np.maximum(crosses,0)],0),
                        event_id=np.where(known,b.event_id.to_numpy()[ix],''),cross_direction=np.where(known,b.direction.to_numpy()[ix],0),
                        delay_s=np.where(known,(q.quote_t[np.maximum(crosses,0)]-q.quote_t[esafe])/NS,np.nan),
                        mid_delta=np.where(known,((q.bid[np.maximum(crosses,0)].astype(float)+q.ask[np.maximum(crosses,0)])-(q.bid[esafe].astype(float)+q.ask[esafe]))/2e9,np.nan),
                        net_exit_L=np.where(xok,(q.bid[xsafe]-q.ask[esafe])/1e9-.3,np.nan),
                        net_exit_S=np.where(xok,(q.bid[esafe]-q.ask[xsafe])/1e9-.3,np.nan)))
                    f[['mark_L','mark_S','mae_L','mae_S','mfe_L','mfe_S']]=vals
                    f['risk_R']=roots.risk_R.to_numpy();profiles.append(f)
                    selected=np.flatnonzero(known)
                    for z in selected[np.linspace(0,len(selected)-1,min(3,len(selected)),dtype=int)]:
                        e=entry[z];c=crosses[z]
                        assert c==ords[np.searchsorted(ords,e,side='right')]
                        direct=np.array([(q.bid[c]-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[c])/1e9-.3,
                            (q.bid[e:c+1].min()-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[e:c+1].max())/1e9-.3,
                            (q.bid[e:c+1].max()-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[e:c+1].min())/1e9-.3])
                        err=float(abs(direct-vals[z]).max())
                        if err>1e-9:raise ValueError('Raw path discrepancy')
                        day_checks.append(dict(type='profile_direct',family=family,scale=s,role=r,error=err))
        bank=pd.concat(banks,ignore_index=True).sort_values(['ordinal','family','scale','role']).reset_index(drop=True)
        old=pd.read_parquet(OLD/(date+'_bank.parquet'))
        a=bank.loc[(bank.family=='BB')&bank.scale.isin(bb.SCALES),['ordinal','scale','role','direction']].sort_values(['ordinal','scale','role']).reset_index(drop=True)
        b=old[['ordinal','scale','role','direction']].sort_values(['ordinal','scale','role']).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_dtype=False)
        hits,stops=first_passage(q.quote_t,q.bid,q.ask,k,entry,obs_end,roots.risk_R.to_numpy())
        for side,label in enumerate(['L','S']):
            roots['stop_'+label]=np.where(stops[:,side]>=0,q.quote_t[np.maximum(stops[:,side],0)],0)
            for g in range(3):roots[f'gain_{label}_{g}']=np.where(hits[:,side,g]>=0,q.quote_t[np.maximum(hits[:,side,g],0)],0)
        frames={'bank':bank,'map':pd.concat(profiles,ignore_index=True),'roots':roots,'scene':scene};files={}
        for name,f in frames.items():
            path=OUT/(date+'_'+name+'.parquet');f.to_parquet(path,index=False);files[path.name]=sha(path)
        rec=dict(date=date,identity=identity,files=files,origins=len(roots),cells=len(frames['map']),events=len(bank),
            quotes=len(q.quote_t),retained=len(k),bb_bank_parity_rows=len(a),checks=day_checks,
            manifest_sha256=sha(day.path/'manifest.json'))
        save(rp,rec);receipts.append(rec);checks.extend(day_checks)
        print(json.dumps({k:rec[k] for k in ['date','origins','cells','events','bb_bank_parity_rows']}),flush=True)
        del day,q,tr,k,blocks,br,frames,profiles,banks;QUOTE_CACHE.clear();gc.collect()
    save(OUT/'manifest.json',dict(status='COMPLETE',identity=identity,days=receipts,source_receipt_sha256=sha('reports/vwap_bands_week_v1/source_receipt.json')))
    save(REPORT/'build_receipt.json',dict(status='PASS',identity=identity,manifest_sha256=sha(OUT/'manifest.json'),
        dates=DATES,origins=sum(r['origins'] for r in receipts),cells=sum(r['cells'] for r in receipts),events=sum(r['events'] for r in receipts),
        bb_parity_rows=sum(r['bb_bank_parity_rows'] for r in receipts),checks=checks,days=receipts))


if __name__=='__main__':main()
