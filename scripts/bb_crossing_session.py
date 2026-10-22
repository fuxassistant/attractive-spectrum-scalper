"""Uncapped-by-30m factual first crossings and post-entry checkpoint profiles."""
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from spectrum.cognition import bb_crossing as bb
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay,QUOTE_CACHE
from scripts.bb_crossing_build import DATA,REPORT,MARKET
from scripts.bb_crossing_report import finish,plt
from scripts.flow_context_coverage import save,sha

OUT=DATA/'first_session'


@njit(cache=True)
def all_marks(bid,ask,k,blocks,entries,crosses,known,breaks):
    values=np.full((len(entries),6),np.nan)
    for i in range(len(entries)):
        if not known[i]:continue
        e=entries[i];x=crosses[i]
        if x>=breaks[e]:raise ValueError('Crossing after lost quote support')
        ext=bb.range_extrema(bid,ask,k,blocks,e,x)
        values[i,0]=(bid[x]-ask[e])/1e9-.3;values[i,1]=(bid[e]-ask[x])/1e9-.3
        values[i,2]=(ext[0]-ask[e])/1e9-.3;values[i,3]=(bid[e]+ext[3])/1e9-.3
        values[i,4]=(-ext[1]-ask[e])/1e9-.3;values[i,5]=(bid[e]-ext[2])/1e9-.3
    return values


def lookup(p,t,limit,ready,key):
    ix=np.searchsorted(p.t_ns.to_numpy(),t,side='right');safe=np.minimum(ix,max(0,len(p)-1))
    if not len(p):raise ValueError('Empty scale bank')
    event_t=p.t_ns.to_numpy()[safe];known=ready&(ix<len(p))&(event_t<limit)
    return pd.DataFrame({key+'_known':known,key+'_y':np.where(known,p.label.to_numpy()[safe],-1),
        key+'_id':np.where(known,p.packet_id.to_numpy()[safe],''),key+'_event_ns':np.where(known,event_t,0),
        key+'_ordinal':np.where(known,p.ordinal.to_numpy()[safe],-1),key+'_mask':np.where(known,p.line_mask.to_numpy()[safe],0),
        key+'_delay_s':np.where(known,(event_t-t)/bb.NS,np.nan)})


def main():
    OUT.mkdir(exist_ok=True);man=json.loads((DATA/'manifest.json').read_text());receipts=[];summaries={str(s):[] for s in bb.SCALES};all_profiles=[]
    ids={p:sha(p) for p in [__file__,'contracts/bb_crossing_session_projection_v1.json']}
    for r in man['days']:
        date=r['date'];rp=OUT/(date+'.json')
        if rp.exists():
            receipt=json.loads(rp.read_text())
            if receipt['identity']!=ids:raise ValueError('Session projection identity changed')
            for suffix,expected in receipt['files'].items():
                if sha(OUT/(date+'_'+suffix+'.parquet'))!=expected:raise ValueError('Projection changed')
            receipts.append(receipt);all_profiles.append(pd.read_parquet(OUT/(date+'_profiles.parquet')));continue
        root=pd.read_parquet(DATA/(date+'_roots.parquet'));bank=pd.read_parquet(DATA/(date+'_bank.parquet'));pr=pd.read_parquet(DATA/(date+'_profiles.parquet'))
        times=root.t_ns.to_numpy(np.int64);decision=root[['date','t_ns']].copy();ready={};ends={};frames=[]
        day=MonthDay(MARKET/date[:7]/date);q=QuoteTape.from_day(day);k=bb.reduce_quotes(q);blocks=bb.extremum_blocks(q.bid,q.ask,k);breaks=bb.execution_breaks(q.quote_t,q.valid,q.epoch)
        for s in bb.SCALES:
            ready[s]=np.isfinite(root[f'bb{s}_position'].to_numpy());ends[s]=root[f'm{s}_support_until_ns'].to_numpy(np.int64)
            packets=bb.packets(bank,s)
            decision=pd.concat([decision,lookup(packets,times,ends[s],ready[s],f'm{s}')],axis=1)
            f=pr.loc[pr.scale==s].reset_index(drop=True);entry=f.entry_ns.to_numpy(np.int64)
            eok=(f.entry_ordinal.to_numpy()>=0)&ready[s]&(entry<ends[s])
            target=lookup(packets,entry,ends[s],eok,'cross')
            p=f[['date','t_ns','scale','entry_ns','entry_ordinal','entry_bid','entry_ask']].copy()
            p=pd.concat([p,target],axis=1)
            p[['mark_L','mark_S','mae_L','mae_S','mfe_L','mfe_S']]=all_marks(q.bid,q.ask,k,blocks,
                f.entry_ordinal.to_numpy(np.int64),target.cross_ordinal.to_numpy(np.int64),target.cross_known.to_numpy(bool),breaks)
            p['later_than_30m']=p.cross_delay_s>1800
            p['status']=np.where(p.cross_known,'KNOWN_CROSS',np.where(eok,'UNRESOLVED_BEFORE_SUPPORT_OR_SESSION_END','NO_SUPPORTED_ENTRY_GEOMETRY'))
            frames.append(p)
        global_target=lookup(bb.packets(bank),times,np.minimum.reduce(list(ends.values())),np.logical_and.reduce(list(ready.values())),'global')
        decision=pd.concat([decision,global_target],axis=1);profile=pd.concat(frames,ignore_index=True)
        files={}
        for suffix,f in [('roots',decision),('profiles',profile)]:
            path=OUT/(date+'_'+suffix+'.parquet');f.to_parquet(path,index=False);files[suffix]=sha(path)
        receipt=dict(date=date,identity=ids,source_files=r['files'],files=files)
        save(rp,receipt);receipts.append(receipt);all_profiles.append(profile)
        del day,q,bank,root,pr,blocks,breaks,k;QUOTE_CACHE.clear();gc.collect()
        print(date,flush=True)
    profiles=pd.concat(all_profiles,ignore_index=True);stats={}
    for s in bb.SCALES:
        f=profiles.loc[profiles.scale==s];known=f.loc[f.cross_known]
        stats[str(s)]=dict(roots=len(f),known_crosses=len(known),unknown=len(f)-len(known),
            first_cross_after_30m=int(known.later_than_30m.sum()),
            delay_quantiles_s={str(q):float(known.cross_delay_s.quantile(q)) for q in [.1,.5,.9,.99]},
            mark_L_quantiles={str(q):float(known.mark_L.quantile(q)) for q in [.1,.5,.9]},
            mark_S_quantiles={str(q):float(known.mark_S.quantile(q)) for q in [.1,.5,.9]})
    save(OUT/'manifest.json',dict(status='COMPLETE',identity=ids,source_manifest_sha256=sha(DATA/'manifest.json'),days=receipts))
    save(REPORT/'full_session_projection.json',dict(status='COMPLETE',identity=ids,manifest_sha256=sha(OUT/'manifest.json'),statistics=stats,
        meaning='First future crossing without 30m cap, within same uninterrupted session support; marks are alternative checkpoints, not account returns; original models trained/evaluated on separate 30m target'))
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for s,color in zip(bb.SCALES,['#176b9b','#dd9224','#a53745','#5a56a6']):
        f=profiles.loc[(profiles.scale==s)&profiles.cross_known];delay=np.sort(f.cross_delay_s.to_numpy())/60
        axes[0].plot(delay,np.arange(1,len(delay)+1)/len(delay),color=color,label=f'M{s} n={len(f):,}')
    axes[0].axvline(30,color='black',ls='--',label='Separate learner / strategy cap')
    axes[0].set(xscale='log',xlabel='Minutes to first post-entry crossing',ylabel='CDF among observed crossings',title='Factual bank continues beyond 30 minutes');axes[0].legend(fontsize=8)
    x=np.arange(4);axes[1].bar(x,[stats[str(s)]['known_crosses'] for s in bb.SCALES],color='#bbd2de',label='All observed first crosses')
    axes[1].bar(x,[stats[str(s)]['first_cross_after_30m'] for s in bb.SCALES],color='#d1902b',label='First cross later than 30m')
    axes[1].set(xticks=x,xticklabels=[f'M{s}' for s in bb.SCALES],ylabel='Queries (overlap; not independent trades)',title='Late checkpoints retain their IDs, time and net');axes[1].legend()
    finish(fig,'09_full_session_profiles.png')


if __name__=='__main__':main()
