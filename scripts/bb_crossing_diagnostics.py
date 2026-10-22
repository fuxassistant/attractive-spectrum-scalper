"""Coverage, micro-recrossing, old-target comparison and paired day uncertainty."""
import argparse
import gc
import json
import numpy as np
import pandas as pd
from spectrum.cognition import bb_crossing as bb, episodes as ep
from spectrum.playground.data import MonthDay,QUOTE_CACHE
from spectrum.cognition.execution import QuoteTape
from scripts.bb_crossing_build import DATA,REPORT,MARKET
from scripts.bb_crossing_report import load_all,finish,clock,plt
from scripts.bb_crossing_verify import direct_geometry
from scripts.flow_context_coverage import save,sha


def bank_diagnostics():
    roots=load_all('roots');man=json.loads((DATA/'manifest.json').read_text());intervals={s:[] for s in bb.SCALES};used={s:[0,0] for s in bb.SCALES}
    coverage={}
    for month,f in roots.groupby(roots.date.str[:7]):
        coverage[month]={}
        for s in bb.SCALES:
            ready=np.isfinite(f[f'bb{s}_position']);known=f[f'm{s}_y']>=0
            coverage[month][str(s)]=dict(roots=len(f),origin_geometry_ready=int(ready.sum()),mature_target=int(known.sum()),
                no_geometry=int((~ready).sum()),future_censored=int((ready&~known).sum()),no_cross_by_h=int((f[f'm{s}_y']==4).sum()))
    for r in man['days']:
        date=r['date'];bank=pd.read_parquet(DATA/(date+'_bank.parquet'));rf=roots.loc[roots.date==date]
        if not len(rf):continue
        for s in bb.SCALES:
            f=bank.loc[bank.scale==s]
            dt=f.groupby(['role','support']).t_ns.diff().dropna().to_numpy()/bb.NS
            intervals[s].extend(dt.tolist())
            active=f.loc[f.t_ns.between(rf.t_ns.min()+1,rf.t_ns.max())]
            ids=set(active.packet_id+'/M'+str(s));used[s][0]+=len(ids)
            used[s][1]+=len(ids&set(rf[f'm{s}_id']))
    result=dict(coverage=coverage,interarrival={},script_sha256=sha(__file__))
    for s in bb.SCALES:
        a=np.array(intervals[s]);result['interarrival'][str(s)]=dict(pairs=len(a),
            quantiles_s={str(q):float(np.quantile(a,q)) for q in [.1,.5,.9,.99]},
            fraction_under_1s=float((a<1).mean()),fraction_under_5s=float((a<5).mean()),fraction_under_30s=float((a<30).mean()),
            active_window_scale_packets=used[s][0],first_target_packets=used[s][1],first_target_fraction=used[s][1]/used[s][0])
    # Same roots and side: old fixed T target vs BB on identical mature pairs.
    old=pd.read_parquet('data/derived/target_loss_audit_v1/frame_T.parquet')
    old=old.loc[(old.side>0)&(old.split=='B'),['date','t_ns','category']]
    matched=roots.merge(old,on=['date','t_ns'],validate='one_to_one').sort_values(['date','t_ns'])
    result['same_pair_old_target']={}
    for s in bb.SCALES:
        t=matched.t_ns.to_numpy();old_y=matched.category.to_numpy();new_y=matched[f'm{s}_y'].to_numpy();date=matched.date.to_numpy()
        pair=(np.diff(t)==30*bb.NS)&(date[1:]==date[:-1])&(old_y[1:]>=0)&(old_y[:-1]>=0)&(old_y[1:]<3)&(old_y[:-1]<3)&(new_y[1:]>=0)&(new_y[:-1]>=0)
        pair &= t[1:]<matched[f'm{s}_support_until_ns'].to_numpy()[:-1]
        a=np.flatnonzero(pair)
        result['same_pair_old_target'][str(s)]=dict(pairs=len(a),old_take_binary_flip=float(np.mean((old_y[a]==0)!=(old_y[a+1]==0))),
            old_three_class_flip=float(np.mean(old_y[a]!=old_y[a+1])),bb_class_flip=float(np.mean(new_y[a]!=new_y[a+1])),
            bb_event_id_flip=float(np.mean(matched[f'm{s}_id'].to_numpy()[a]!=matched[f'm{s}_id'].to_numpy()[a+1])))
    save(REPORT/'bank_diagnostics.json',result)
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    x=np.arange(4)
    for j,(name,color) in enumerate([('fraction_under_1s','#a53f49'),('fraction_under_5s','#cc8c32'),('fraction_under_30s','#23759b')]):
        axes[0].bar(x+(j-1)*.25,[result['interarrival'][str(s)][name] for s in bb.SCALES],.24,label=name.replace('fraction_under_','<'),color=color)
    axes[0].set(xticks=x,xticklabels=[f'M{s}' for s in bb.SCALES],ylim=(0,1),title='Same-line recrossings: many IDs share one local move',ylabel='Fraction of consecutive events, same support');axes[0].legend()
    for s,color in zip(bb.SCALES,['#176b9b','#dd9224','#a53745','#5a56a6']):
        axes[1].plot(list(coverage),[coverage[m][str(s)]['mature_target']/coverage[m][str(s)]['roots'] for m in coverage],'o-',label=f'M{s}',color=color)
    axes[1].set(title='Target coverage: long scales need a long supported history',ylabel='Known targets / all source roots',ylim=(0,1));axes[1].legend();axes[1].tick_params(axis='x',rotation=20)
    finish(fig,'07_microevents_and_coverage.png')

    date='2026-01-02';day=MonthDay(MARKET/date[:7]/date);q=QuoteTape.from_day(day)
    bank=pd.read_parquet(DATA/(date+'_bank.parquet'));m=bank.loc[(bank.scale==1)&(bank.role==1)&(bank.t_ns>=q.entry_start_ns)].reset_index(drop=True)
    delta=m.t_ns.to_numpy()[5:]-m.t_ns.to_numpy()[:-5]
    first=int(np.flatnonzero(delta<5*bb.NS)[0]);events=m.iloc[first:first+6]
    start=int(events.t_ns.iloc[0])-bb.NS;end=int(events.t_ns.iloc[-1])+bb.NS
    ix=np.arange(np.searchsorted(q.quote_t,start,'left'),np.searchsorted(q.quote_t,end,'right'))
    middle,half,p=direct_geometry(q,ix,1);elapsed=(q.quote_t[ix]-start)/bb.NS
    fig,axes=plt.subplots(2,1,figsize=(15,8),gridspec_kw={'height_ratios':[2,1.2]},layout='constrained')
    axes[0].step(elapsed,(p-middle)/.1,where='post',color='#176b9b',lw=1)
    axes[0].axhline(0,color='black',lw=.8)
    for n,r in enumerate(events.itertuples(),1):
        e=(r.t_ns-start)/bb.NS;axes[0].scatter([e],[(r.mid-r.line)/.1],s=35,color='#b43f43')
        axes[0].annotate(str(n),(e,(r.mid-r.line)/.1),xytext=(3,8),textcoords='offset points')
    axes[0].set(xlabel=f'Seconds since {clock([start])[0]}',ylabel='Price − developing M1 middle (ticks)',title='First six IDs labelled; full quote display includes further unlabelled reversals')
    axes[1].axis('off')
    lines=['#   timestamp UTC                    event ID']
    for n,r in enumerate(events.itertuples(),1):lines.append(f'{n}   {str(clock([r.t_ns])[0])[11:29]:<30}  {r.event_id}')
    axes[1].text(.01,.95,'\n'.join(lines),va='top',family='monospace',fontsize=10,linespacing=1.6)
    fig.suptitle('Physical event IDs identify reversals, not independent trading opportunities',fontsize=14,weight='bold')
    finish(fig,'08_micro_crossing_ids.png')
    save(REPORT/'micro_scene.json',dict(date=date,selection='first M1 middle cluster of six crossings within 5s after London on first B day',
        events=[dict(event_id=r.event_id,t_ns=str(r.t_ns),direction=int(r.direction),mid=r.mid,line=r.line) for r in events.itertuples()]))
    QUOTE_CACHE.clear();del day,q;gc.collect()


def paired_model_diagnostics():
    result={}
    for stage in ['A2','B']:
        for head in ['m1','m3','m6','m9','global']:
            frames={family:pd.read_parquet(DATA/f'predictions_{stage}_{head}_{family}.parquet') for family in ['geometry','rich','rich_event']}
            base=frames['geometry'];ok=base[head+'_y']>=0;days=sorted(base.date.unique());losses={}
            for family,f in frames.items():
                pd.testing.assert_frame_equal(base[['date','t_ns',head+'_y',head+'_id']],f[['date','t_ns',head+'_y',head+'_id']])
                y=f.loc[ok,head+'_y'].to_numpy(int);cols=[c for c in f if c.startswith('p') and c[1:].isdigit()]
                p=f.loc[ok,cols].to_numpy();losses[family]=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
            ids=base.loc[ok,head+'_id'];weights=bb.event_weights(ids)
            for newer,older in [('rich','geometry'),('rich_event','rich')]:
                gain=losses[older]-losses[newer]
                daily=pd.DataFrame(dict(date=base.loc[ok,'date'].to_numpy(),gain=gain,event_gain=gain*weights)).groupby('date').sum().reindex(days,fill_value=0.)
                result[f'{stage}/{head}/{newer}-{older}']=dict(mean_log_score_gain=float(gain.mean()),event_mean_gain=float(np.average(gain,weights=weights)),
                    block_day_sum_gain_ci=ep.block_interval(daily.gain.to_numpy()),event_block_day_sum_gain_ci=ep.block_interval(daily.event_gain.to_numpy()))
    save(REPORT/'paired_model_diagnostics.json',dict(status='COMPLETE',comparisons=result,
        note='Positive gain favors newer. Moving 5-day blocks; events still temporally dependent. Development-exposed comparisons, not confirmatory inference.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['bank','paired']);a=parser.parse_args()
    {'bank':bank_diagnostics,'paired':paired_model_diagnostics}[a.command]()
