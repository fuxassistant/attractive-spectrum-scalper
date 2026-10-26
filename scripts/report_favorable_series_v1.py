"""Visual series 1,2,6,7,9,18: fixed windows and reproducible counterexamples."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay,QUOTE_CACHE
from scripts.build_crossing_path_target_v6 import sha,save

C=json.loads(Path('configs/research/favorable_series_v1.json').read_text())
BASE=json.loads(Path(C['base_config']).read_text())
DATA=Path(C['output']);OUT=Path(C['report']);ROLES=['lower','middle','upper']
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'#f8fafc'})
IMAGES=[];CASES={};TAPES={}


def finish(fig,name):
    fig.savefig(OUT/name,dpi=145,bbox_inches='tight');plt.close(fig);IMAGES.append(name)


def tape(date):
    if date not in TAPES:
        TAPES.clear();QUOTE_CACHE.clear()
        TAPES[date]=QuoteTape.from_day(MonthDay(Path(BASE['data']['market_root'])/date[:7]/date,verify=True))
    return TAPES[date]


def trace(row,side):
    q=tape(row.date);e=int(row.entry_ordinal);x=int(row.cross_ordinal)
    tt=(q.quote_t[e:x+1]-q.quote_t[e])/1e9
    yy=((q.bid[e:x+1]-q.ask[e] if side=='L' else q.bid[e]-q.ask[e:x+1])/1e9-.3)/row.R_event
    return tt,yy


def row_receipt(row,side):
    keys=['date','scale','role','v6_event_id','elapsed_seconds',f'net_R_{side}',f'mae_R_{side}',f'mfe_R_{side}',
        f'endpoint_margin_{side}',f'mfe_margin_{side}',f'risk_before_cross_{side}']
    d={k:row[k].item() if isinstance(row[k],np.generic) else row[k] for k in keys}
    for k in ('t_ns','cross_ns','entry_ns','cross_ordinal','entry_ordinal'):d[k]=str(row[k])
    d['side']=side
    return d


def life_table(cells):
    a=cells[cells.profile_known].sort_values(['v6_event_id','t_ns']).copy()
    frames=[]
    for side in ('L','S'):
        same=a.v6_event_id.eq(a.v6_event_id.shift())
        adjacent=a.t_ns.diff().eq(5*10**9)
        flips=same & adjacent & a[f'endpoint_qualified_{side}'].ne(a[f'endpoint_qualified_{side}'].shift())
        a['flips']=flips.astype(int)
        b=a.groupby('v6_event_id').agg(date=('date','first'),scale=('scale','first'),role=('role','first'),
            origins=('t_ns','size'),first_t=('t_ns','min'),last_t=('t_ns','max'),
            qualifying=(f'endpoint_qualified_{side}','sum'),flips=('flips','sum'),
            cross_ns=('cross_ns','first'))
        b['side']=side;frames.append(b.reset_index())
    return pd.concat(frames,ignore_index=True)


def fixed_maps(cells,roots):
    for k,start in enumerate(C['fixed_windows']):
        t0=pd.Timestamp(start).value;t1=t0+C['window_minutes']*60*10**9
        a=cells[cells.t_ns.between(t0,t1)].copy()
        times=np.sort(a.t_ns.unique());xx=pd.to_datetime(times,utc=True)
        fig,axes=plt.subplots(3,2,figsize=(15,10),constrained_layout=True,sharex=True)
        fig.suptitle(f'Series 1 | {start[:10]} | fixed 30-minute clock | ratio 2:1',fontsize=16)
        for col,side in enumerate(('L','S')):
            for r,kind in enumerate(('mfe','endpoint')):
                z=a.pivot(index=['scale','role'],columns='t_ns',values=f'{kind}_margin_{side}').to_numpy()
                known=a.pivot(index=['scale','role'],columns='t_ns',values='hit_known').to_numpy()
                cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#dddddd')
                im=axes[r,col].imshow(z,aspect='auto',origin='upper',extent=[mdates.date2num(xx[0]),mdates.date2num(xx[-1]),8.5,-.5],cmap=cmap,vmin=-4,vmax=4,interpolation='nearest')
                unknown=np.where(~known,1,np.nan)
                axes[r,col].imshow(unknown,aspect='auto',extent=[mdates.date2num(xx[0]),mdates.date2num(xx[-1]),8.5,-.5],cmap=ListedColormap(['#b99bd4']),vmin=0,vmax=1,interpolation='nearest')
                axes[r,col].set(title=f'{side} | {kind}: (value - 2 adverse) / R',yticks=range(9),
                    yticklabels=[f'M{s} {ROLES[role]}' for s in (1,3,6) for role in range(3)])
            b=roots[roots.t_ns.between(t0,t1)]
            for kind,color in (('mfe','#ca772b'),('endpoint','#187c9d')):
                axes[2,col].step(pd.to_datetime(b.t_ns,utc=True),b[f'{kind}_{side}'].replace(-1,np.nan),where='post',label=kind,color=color)
            axes[2,col].set(ylabel='Any-line qualification',ylim=(-.1,1.15));axes[2,col].legend()
            axes[2,col].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.colorbar(im,ax=axes[:2,:].ravel().tolist(),label='Margin / R; display saturated at +/-4')
        fig.supxlabel('UTC | gray = complete no-hit, purple = unknown; positive margin is red')
        finish(fig,f'01_map_{k+1}.png')


def qualification(cells,stats):
    sides=['L','S'];fig,ax=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    fig.suptitle('Series 2 | excursion qualification versus crossing endpoint | all 20 days',fontsize=15)
    totals={}
    for s in sides:
        totals[s]={key:sum(d['sides'][s][key] for d in stats['days']) for key in stats['days'][0]['sides'][s]}
    x=np.arange(2)
    ax[0].bar(x-.18,[totals[s]['mfe'] for s in sides],.36,label='MFE-qualified',color='#ca772b')
    ax[0].bar(x+.18,[totals[s]['endpoint'] for s in sides],.36,label='Endpoint-qualified',color='#187c9d')
    ax[0].set(xticks=x,xticklabels=sides,ylabel='Crossing cells (dependent repeats)');ax[0].legend()
    for j,s in enumerate(sides,1):
        a=cells[cells.profile_known].iloc[::8]
        hb=ax[j].hexbin(a[f'mfe_margin_{s}'],a[f'endpoint_margin_{s}'],gridsize=60,bins='log',mincnt=1,cmap='viridis')
        ax[j].axhline(0,color='red',lw=.8);ax[j].axvline(0,color='red',lw=.8)
        ax[j].set(xlim=(-8,8),ylim=(-8,8),xlabel=f'{s} MFE margin / R',ylabel=f'{s} endpoint margin / R',title='TEST hit cells; sampled for display')
    finish(fig,'02_qualification.png')
    cases=[]
    fig,ax=plt.subplots(2,2,figsize=(14,9),constrained_layout=True,sharey=True)
    fig.suptitle('Series 2 counterexamples | earliest TEST case per side/category, elapsed >=5s',fontsize=15)
    for col,s in enumerate(sides):
        eligible=cells[cells.profile_known & (cells.elapsed_seconds>=5)]
        for r,mask in enumerate(((eligible[f'mfe_qualified_{s}']==1)&(eligible[f'net_R_{s}']<=0),eligible[f'endpoint_qualified_{s}']==1)):
            row=eligible[mask].sort_values(['t_ns','scale','role']).iloc[0]
            tt,yy=trace(row,s);a=ax[r,col]
            a.step(tt,yy,where='post',lw=1,color='#ca772b' if r==0 else '#187c9d')
            a.axhline(0,color='black',lw=.7)
            a.axhline(2*row[f'adverse_R_{s}'],color='#777',ls='--',label='2 x full-path adverse (future diagnostic)')
            a.plot(tt[-1],yy[-1],'o',color='red')
            a.set(title=f"{s} {row.date} {pd.to_datetime(row.t_ns,utc=True):%H:%M:%S} | M{row.scale} {ROLES[int(row.role)]}\n"
                f"MFE {row[f'mfe_R_{s}']:.2f} R | endpoint {row[f'net_R_{s}']:.2f} R | adverse {row[f'adverse_R_{s}']:.2f} R",
                xlabel='Seconds after entry',ylabel='Net / R')
            a.legend(fontsize=8);cases.append(row_receipt(row,s))
    finish(fig,'02_paths_and_counterexamples.png');CASES['series2']=cases
    return totals


def event_lives(cells):
    lives=life_table(cells)
    lives.to_parquet(DATA/'test_event_lives.parquet',index=False)
    pool=lives[(lives.origins>=12)&(lives.qualifying>0)].sort_values(['first_t','v6_event_id','side'])
    groups=[pool[pool.flips<=2],pool[pool.flips>=3]]
    if any(g.empty for g in groups):raise ValueError('prespecified event-life category empty')
    fig,ax=plt.subplots(3,2,figsize=(15,10),constrained_layout=True,sharex='col')
    fig.suptitle('Series 6 | one physical crossing, different starting decisions | examples + counterexample',fontsize=15)
    receipt=[]
    for col,g in enumerate(groups):
        selected=g.iloc[0];side=selected.side
        a=cells[cells.v6_event_id==selected.v6_event_id].sort_values('t_ns')
        tt=(a.t_ns-a.cross_ns)/1e9
        ax[0,col].plot(tt,a[f'net_R_{side}'],label='endpoint / R')
        ax[0,col].plot(tt,2*a[f'adverse_R_{side}'],label='2 x adverse / R')
        ax[0,col].set(title=f"{side} {selected.date} M{selected.scale} {ROLES[int(selected.role)]}\n"
            f"{selected.origins} origins | {selected.qualifying} qualified | {selected.flips} adjacent flips",ylabel='Outcome / R');ax[0,col].legend(fontsize=9)
        for kind,color in (('mfe','#ca772b'),('endpoint','#187c9d')):
            ax[1,col].plot(tt,a[f'{kind}_margin_{side}'],label=kind,color=color)
        ax[1,col].axhline(0,color='black',lw=.8);ax[1,col].set(ylabel='Qualification margin / R');ax[1,col].legend()
        ax[2,col].step(tt,a[f'endpoint_qualified_{side}'],where='post',label='endpoint qualifies')
        ax[2,col].plot(tt,a[f'underwater_fraction_{side}'],label='underwater fraction',alpha=.8)
        ax[2,col].set(xlabel='Decision seconds relative to the same future crossing',ylabel='Qualification / fraction',ylim=(-.05,1.1));ax[2,col].legend(fontsize=9)
        receipt.append(dict(event_id=selected.v6_event_id,side=side,date=selected.date,origins=int(selected.origins),flips=int(selected.flips),first_origin_ns=str(int(selected.first_t))))
    finish(fig,'06_event_lives.png');CASES['series6']=receipt
    stable=pool[(pool.qualifying>=6)&(pool.flips<=2)].iloc[0]
    a=cells[cells.v6_event_id==stable.v6_event_id].sort_values('t_ns');side=stable.side
    tt=(a.t_ns-a.cross_ns)/1e9
    fig,ax=plt.subplots(2,1,figsize=(13,7),sharex=True,constrained_layout=True)
    fig.suptitle('Series 6 | supplemental stable window | earliest event with >=6 qualifying origins and <=2 flips',fontsize=14)
    ax[0].plot(tt,a[f'net_R_{side}'],label='endpoint / R');ax[0].plot(tt,2*a[f'adverse_R_{side}'],label='2 x adverse / R')
    ax[0].legend();ax[0].set(ylabel='Outcome / R',title=f'{side} {stable.date} M{stable.scale} {ROLES[int(stable.role)]} | {stable.qualifying}/{stable.origins} qualifying origins')
    ax[1].plot(tt,a[f'endpoint_margin_{side}'],label='endpoint margin');ax[1].axhline(0,color='black',lw=.8)
    ax[1].fill_between(tt,a[f'endpoint_margin_{side}'],0,where=a[f'endpoint_qualified_{side}']==1,color='#187c9d',alpha=.2)
    ax[1].set(xlabel='Decision seconds relative to the same future crossing',ylabel='Margin / R');ax[1].legend()
    finish(fig,'06_stable_window.png')
    CASES['series6_stable']=dict(event_id=stable.v6_event_id,side=side,date=stable.date,origins=int(stable.origins),qualifying=int(stable.qualifying),flips=int(stable.flips))
    return dict(test_event_side_groups=len(lives),eligible_groups=len(pool),
        eligible_groups_with_3plus_flips=int((pool.flips>=3).sum()),
        eligible_fraction_with_3plus_flips=float((pool.flips>=3).mean()))


def switches(cells,stats):
    d=pd.DataFrame(stats['adjacent_changes'])
    fig,ax=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    fig.suptitle('Series 7 | qualification changes between adjacent 5-second origins | all 20 days',fontsize=15)
    kinds=['same_event','event_switch','hit_presence_change'];xx=np.arange(3)
    for j,s in enumerate(('L','S')):
        z=d[d.side==s].set_index('kind').loc[kinds]
        ax[0].bar(xx+(j-.5)*.32,z.flips/z.pairs,.32,label=s)
        ax[1].bar(xx+(j-.5)*.32,z.flips,.32,label=s)
    for a in ax:a.set(xticks=xx,xticklabels=['Same event ID','Different event ID','Hit / no-hit change']);a.legend()
    ax[0].set(ylabel='Qualification flip fraction within category');ax[1].set(ylabel='Number of flips')
    finish(fig,'07_switch_attribution.png')
    t0=pd.Timestamp(C['fixed_windows'][1]).value;t1=t0+30*60*10**9
    a=cells[(cells.scale==1)&(cells.role==1)&cells.t_ns.between(t0,t1)].sort_values('t_ns')
    tt=pd.to_datetime(a.t_ns,utc=True);ids=a.v6_event_id.to_numpy()
    switched=np.r_[False,ids[1:]!=ids[:-1]] & a.profile_known.to_numpy() & np.r_[False,a.profile_known.to_numpy()[:-1]]
    fig,ax=plt.subplots(3,1,figsize=(14,8),sharex=True,constrained_layout=True)
    fig.suptitle('Series 7 | fixed Sep01 13:30 UTC, M1 middle | lines mark observed event-ID switches',fontsize=14)
    for s in ('L','S'):ax[0].plot(tt,a[f'endpoint_margin_{s}'],label=s)
    ax[0].axhline(0,color='black',lw=.8);ax[0].set(ylabel='Endpoint margin / R');ax[0].legend()
    for s in ('L','S'):ax[1].step(tt,a[f'endpoint_qualified_{s}'].replace(-1,np.nan),where='post',label=s)
    ax[1].set(ylabel='Qualification');ax[1].legend()
    ax[2].plot(tt,a.elapsed_seconds);ax[2].set(ylabel='Seconds to crossing',xlabel='UTC | gaps are unobserved profiles')
    for axis in ax:
        for at in tt[switched]:axis.axvline(at,color='gray',alpha=.3,lw=.8)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    finish(fig,'07_fixed_switch_window.png')


def risk_order(cells,stats):
    fig,ax=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    fig.suptitle('Series 9 | ratio 2:1 is not a fixed absolute risk cap',fontsize=16)
    for j,s in enumerate(('L','S')):
        totals=[]
        for date in BASE['data']['dates']:
            a=pd.read_parquet(DATA/f'{date}_cells.parquet',columns=[f'endpoint_qualified_{s}',f'adverse_R_{s}'])
            totals.append(a.loc[a[f'endpoint_qualified_{s}']==1,f'adverse_R_{s}'].to_numpy())
        adverse=np.concatenate(totals)
        ax[j].hist(adverse,bins=np.linspace(0,8,65),color='#187c9d');ax[j].axvline(1,color='red',ls='--')
        ax[j].set(title=f'{s} qualified paths: adverse >=1R in {(adverse>=1).mean():.1%}',xlabel='Full-path adverse / R (display 0..8)',ylabel='Endpoint-qualified cells')
    finish(fig,'09_risk_population.png')
    fig,ax=plt.subplots(2,2,figsize=(14,9),constrained_layout=True,sharey=True);receipt=[]
    fig.suptitle('Series 9 | earliest TEST qualifying path: risk stayed inside versus breached before crossing',fontsize=14)
    for col,s in enumerate(('L','S')):
        for r,risk in enumerate((0,1)):
            pool=cells[(cells[f'endpoint_qualified_{s}']==1)&(cells[f'risk_before_cross_{s}']==risk)&(cells.elapsed_seconds>=5)]
            row=pool.sort_values(['t_ns','scale','role']).iloc[0];tt,yy=trace(row,s)
            a=ax[r,col];a.step(tt,yy,where='post',color='#187c9d' if risk==0 else '#ca772b',lw=1)
            a.axhline(-1,color='red',ls='--',label='Reference risk -1R');a.axhline(0,color='black',lw=.7)
            for field,marker,label in ((f'first_positive_seconds_{s}','^','First positive net'),(f'first_risk_seconds_{s}','x','First risk breach')):
                value=row[field]
                if np.isfinite(value):
                    ix=min(np.searchsorted(tt,value),len(tt)-1);a.plot(tt[ix],yy[ix],marker,ms=7,label=label)
            a.plot(tt[-1],yy[-1],'o',label='Crossing');a.legend(fontsize=8)
            a.set(title=f"{s} {row.date} {pd.to_datetime(row.t_ns,utc=True):%H:%M:%S} | net {row[f'net_R_{s}']:.2f}R | adverse {row[f'adverse_R_{s}']:.2f}R",
                xlabel='Seconds after entry',ylabel='Net / R');receipt.append(row_receipt(row,s))
    finish(fig,'09_risk_paths.png');CASES['series9']=receipt


def past_probe(cells,origins,stats):
    result=stats['probe'];pred=pd.read_parquet(DATA/'probe_predictions.parquet')
    fig,ax=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    fig.suptitle('Series 18 | fixed TRAIN-only predictive probe | all admitted TEST origins',fontsize=15)
    for col,s in enumerate(('L','S')):
        r=result['per_side'][s];labels=['base','+ flow','flow permuted']
        auc=[r['base']['test']['auc'],r['flow']['test']['auc'],np.mean([z['auc'] for z in r['test_flow_permutation']])]
        ax[0,col].bar(np.arange(3),auc,color=['#187c9d','#ca772b','#999999'])
        ax[0,col].axhline(.5,color='black',lw=.7);ax[0,col].set(xticks=range(3),xticklabels=labels,ylim=(.45,.65),ylabel='ROC AUC',title=f'{s}: geometry/time/price versus added flow')
        for j,v in enumerate(auc):ax[0,col].text(j,v+.003,f'{v:.4f}',ha='center')
        days=r['test_days'];delta=[d['brier_improvement'] for d in days]
        ax[1,col].bar(range(len(days)),delta,color=['#187c9d' if x>0 else '#ca772b' for x in delta])
        ax[1,col].axhline(0,color='black',lw=.7);ax[1,col].set(xticks=range(len(days)),xticklabels=[d['date'][5:] for d in days],
            ylabel='Base Brier minus Flow Brier',xlabel='TEST day | positive = flow improved error')
    finish(fig,'18_predictive_probe.png')
    # Calibration and risk-score strata: outcomes are observed, not selected examples.
    fig,ax=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    fig.suptitle('Series 18 | held-block calibration | equal-width probability bins, at least 50 roots',fontsize=15)
    test=pred[(pred.split=='test')&pred.admitted]
    for col,s in enumerate(('L','S')):
        for arm,color in (('base','#187c9d'),('flow','#ca772b')):
            p=test[f'p_{s}_{arm}'].to_numpy();y=test[f'endpoint_{s}'].to_numpy()
            rows=[]
            for j in range(10):
                m=(p>=j/10)&(p<(j+1)/10)
                if m.sum()>=50:rows.append((p[m].mean(),y[m].mean(),int(m.sum())))
            if rows:
                z=np.array(rows);ax[col].plot(z[:,0],z[:,1],'-o',label=arm,color=color)
                for px,py,n in rows:ax[col].annotate(str(n),(px,py),fontsize=7,xytext=(3,5),textcoords='offset points')
        ax[col].plot([0,.7],[0,.7],'--',color='gray');ax[col].set(xlim=(0,.7),ylim=(0,.7),title=s,xlabel='Mean forecast',ylabel='Observed favorable fraction');ax[col].legend()
    finish(fig,'18_calibration.png')
    pairs=pd.read_parquet(DATA/'matched_pairs.parquet')
    # Future labels choose the illustration category AFTER label-blind matching.
    categories=[('both favorable',(pairs.y_left_L==1)&(pairs.y_right_L==1)),
                ('different outcomes',pairs.y_left_L!=pairs.y_right_L),
                ('neither favorable',(pairs.y_left_L==0)&(pairs.y_right_L==0))]
    names=json.loads((DATA/'feature_names.json').read_text());x=np.load(DATA/'x.npy',mmap_mode='r')
    train=(origins.split=='train')&origins.common_known&origins.feature_available&origins.decision_eligible&origins.physical_support_available
    geo=result['matching']['features'];gi=[names.index(n) for n in geo]
    fn=['mbo__imbalance_l1','mbo__trade_buy_5s__signed_log1p','mbo__trade_sell_5s__signed_log1p',
        'mbo__add_bid_5s__signed_log1p','mbo__add_ask_5s__signed_log1p',
        'mbo__remove_bid_5s__signed_log1p','mbo__remove_ask_5s__signed_log1p']
    fi=[names.index(n) for n in fn]
    features=gi+fi;median=np.nanmedian(x[train][:,features],axis=0)
    iqr=np.nanquantile(x[train][:,features],.75,axis=0)-np.nanquantile(x[train][:,features],.25,axis=0)
    iqr=np.where(iqr>1e-6,iqr,1.)
    fig,axes=plt.subplots(3,3,figsize=(17,12),constrained_layout=True)
    fig.suptitle('Series 18 | matched past geometry: positive pattern and counterexamples | LONG target',fontsize=15)
    receipt=[]
    for r,(label,mask) in enumerate(categories):
        if not mask.any():raise ValueError('empty matched category '+label)
        pair=pairs[mask].sort_values(['left','right']).iloc[0]
        records=[]
        for color,i in zip(('#187c9d','#ca772b'),(int(pair.left),int(pair.right))):
            origin=origins.iloc[i];z=(x[i,features]-median)/iqr
            time_label=f'{origin.date} {pd.to_datetime(origin.t_ns,utc=True):%H:%M} y={int(origin.endpoint_L)}'
            axes[r,0].plot(range(len(gi)),np.clip(z[:len(gi)],-5,5),'-o',ms=3,color=color,label=time_label)
            axes[r,1].plot(range(len(fi)),np.clip(z[len(gi):],-5,5),'-o',ms=3,color=color,label=time_label)
            # Full fixed 300s future equity (not only qualifying witness) for both outcomes.
            card=cells[(cells.date==origin.date)&(cells.t_ns==origin.t_ns)].iloc[0]
            q=tape(origin.date);e=int(card.entry_ordinal)
            end=min(np.searchsorted(q.quote_t,q.quote_t[e]+300*10**9,side='right')-1,len(q.quote_t)-1)
            if not q.valid[e:end+1].all() or not (q.epoch[e:end+1]==q.epoch[e]).all() or (np.diff(q.quote_t[e:end+1])>60*10**9).any():
                raise ValueError('matched full-horizon visual has unsupported quotes')
            tt=(q.quote_t[e:end+1]-q.quote_t[e])/1e9;yy=((q.bid[e:end+1]-q.ask[e])/1e9-.3)/card.R_event
            axes[r,2].step(tt,yy,where='post',lw=.8,color=color,label=time_label)
            crossings=cells[(cells.date==origin.date)&(cells.t_ns==origin.t_ns)&cells.profile_known]
            axes[r,2].scatter(crossings.elapsed_seconds,crossings.net_R_L,s=15,color=color,marker='o')
            qualifying=crossings[crossings.endpoint_qualified_L==1]
            axes[r,2].scatter(qualifying.elapsed_seconds,qualifying.net_R_L,s=70,marker='*',color=color)
            records.append(dict(date=origin.date,t_ns=str(int(origin.t_ns)),endpoint_L=int(origin.endpoint_L),
                p_base=float(pred.iloc[i].p_L_base),p_flow=float(pred.iloc[i].p_L_flow),
                geometry={n:float(x[i,names.index(n)]) for n in geo},
                flow={n:float(x[i,names.index(n)]) if np.isfinite(x[i,names.index(n)]) else None for n in fn}))
        axes[r,0].set(title=f'{label} | geometry RMS {pair.distance:.3f}',ylabel='TRAIN-IQR coordinates (clipped +/-5)',xticks=range(len(gi)),
            xticklabels=[n.replace('bb','').replace('mbo__','') for n in geo])
        axes[r,0].tick_params(axis='x',labelrotation=75,labelsize=7);axes[r,0].legend(fontsize=7)
        axes[r,1].set(title='Past flow differs; difference alone is not an explanation',xticks=range(len(fi)),
            xticklabels=['imbalance','buy5s','sell5s','addBid','addAsk','remBid','remAsk'])
        axes[r,1].tick_params(axis='x',labelrotation=45,labelsize=8)
        axes[r,2].axhline(0,color='black',lw=.7);axes[r,2].set(title='Actual future path; dots = crossings, stars = qualified',xlabel='Seconds after entry',ylabel='LONG net / R')
        receipt.append(dict(category=label,distance=float(pair.distance),origins=records))
    finish(fig,'18_matched_counterexamples.png');CASES['series18']=receipt


def main():
    stats=json.loads((OUT/'summary.json').read_text())
    if stats['status']!='COMPLETE':raise ValueError('incomplete run')
    for path,digest in stats['outputs'].items():
        if sha(path)!=digest:raise ValueError('materialized output changed '+path)
    origins=pd.read_parquet(DATA/'origins.parquet')
    cells=pd.concat([pd.read_parquet(DATA/f'{date}_cells.parquet') for date in BASE['splits']['test']],ignore_index=True)
    fixed_maps(cells,origins)
    totals=qualification(cells,stats)
    life=event_lives(cells)
    switches(cells,stats)
    risk_order(cells,stats)
    past_probe(cells,origins,stats)
    findings=dict(totals=totals,event_life=life,
        source_summary_sha256=sha(OUT/'summary.json'),images={name:sha(OUT/name) for name in IMAGES},
        disclaimer='Visual cases are future-selected illustrations under declared rules, never performance estimates; fixed windows also retained. All input features are causal; targets and future case selection are not features.')
    save(OUT/'visual_cases.json',CASES);save(OUT/'visual_findings.json',findings)
    sections=[
        ('1 · Карты favorable', ['01_map_1.png','01_map_2.png'], 'Два фиксированных окна. Цвет — непрерывный запас до 2:1. Серый: crossing не наблюдался за полный горизонт; фиолетовый: исход неизвестен. Сходство двух цветовых карт не означает одинаковую семантику.'),
        ('2 · MFE и результат на crossing', ['02_qualification.png','02_paths_and_counterexamples.png'], 'Первый вид favorable означает, что благоприятная экскурсия существовала. Второй — что crossing сохранил достаточный результат. Показаны первые по времени TEST-примеры, включая MFE-qualified с неположительным endpoint.'),
        ('6 · Одна возможность, разные входы', ['06_event_lives.png','06_stable_window.png'], 'Один физический event ID наблюдается из множества исходных точек. Слева первый подходящий случай с не более чем двумя переключениями; справа — контрпример с тремя и более. Отдельно первый устойчивый пример с минимум шестью положительными origins. Это будущая диагностика, идентификатор будущего события недоступен модели.'),
        ('7 · Почему меняется разметка', ['07_switch_attribution.png','07_fixed_switch_window.png'], 'Отдельно измерены соседние точки одного события, смена event ID и появление/исчезновение crossing в горизонте. Изменение метки не обязательно вызвано сменой события: меняются вход, издержки и adverse от нового входа.'),
        ('9 · Пропорция и абсолютный риск', ['09_risk_population.png','09_risk_paths.png'], '2:1 не ограничивает абсолютную просадку. Точная equity сохраняет порядок первого положительного net, первого нарушения −R и crossing. Пути продолжаются после нарушения риска и не являются исполненными сделками со стопом.'),
        ('18 · Различимость по прошлому', ['18_predictive_probe.png','18_calibration.png','18_matched_counterexamples.png'], 'TRAIN-only одинаковые небольшие модели: 32 признака геометрии/времени/прошлой цены против 69 с потоком. Параметры фиксированы. TEST ранее исследован, это исследовательский перенос. Matching без чтения будущих меток; категории иллюстраций выбираются после matching. Пары могут разделять origins, независимость не предполагается.')]
    html='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Favorable 2:1 · visual series</title>
<style>body{max-width:1450px;margin:30px auto;padding:0 24px;background:#f8fafc;color:#182a3d;font:17px/1.55 system-ui}article{background:white;padding:22px;margin:20px 0;border-radius:12px}img{width:100%}a{color:#126b93}nav a{margin-right:20px}code{background:#e8eef5;padding:2px 4px}</style>
<h1>Favorable 2:1 · первая серия визуальных проверок</h1>
<p>20 сохранённых сессий V6 · BB M1/M3/M6 · первые crossing каждой из 9 линий · H=300s · LONG и SHORT после spread и fee.</p>
<article><b>Условие:</b> MFE-qualified: MFE ≥ 2·adverse; endpoint-qualified: net(crossing) ≥ 2·adverse. Adverse — полный путь от входа до crossing. Положительное значение обязательно; числовой допуск 1e-10 R только для округления. Отдельно хранится непрерывный запас до порога. Нет нового торгового policy или оценки PnL.</article>
'''
    l=stats['probe']['per_side']['L'];s=stats['probe']['per_side']['S']
    html+=f'<article><b>Проверка прошлого:</b> TEST AUC LONG {l["base"]["test"]["auc"]:.4f} → {l["flow"]["test"]["auc"]:.4f}; SHORT {s["base"]["test"]["auc"]:.4f} → {s["flow"]["test"]["auc"]:.4f}. Дополнительные текущие признаки потока не показали общего улучшения AUC/Brier в этой фиксированной постановке. Это не проверка всех способов представления потока и не причинное доказательство.</article>'
    html+='<nav>'+''.join(f'<a href="#s{i}">{title}</a>' for i,(title,_,_) in enumerate(sections))+'</nav>'
    for i,(title,images,caption) in enumerate(sections):
        html+=f'<article id="s{i}"><h2>{title}</h2><p>{caption}</p>'
        for name in images:html+=f'<a href="{name}"><img src="{name}" alt="{title}"></a>'
        html+='</article>'
    html+='<p><a href="summary.json">Полный расчёт</a> · <a href="visual_cases.json">Точные примеры и ID</a> · <a href="probe_results.json">Предиктивная проверка</a> · <a href="visual_findings.json">Измерения и SHA изображений</a> · <a href="sol61_visual_audit.json">Аудит Sol 6.1</a> · <a href="completion_receipt.json">Финальная проверка</a></p></html>'
    (OUT/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps({'images':IMAGES,'totals':totals,'event_life':life},indent=2),flush=True)
    TAPES.clear();QUOTE_CACHE.clear()


if __name__=='__main__':main()
