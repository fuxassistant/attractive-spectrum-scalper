"""Standalone research figures. Uses matplotlib only for rendering, not labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

NAMES={'time_30m':'30 min','time_60m':'60 min','time_30m_shift15':'30 min +15',
       'events_512':'512 raw pubs','events_2048':'2048 raw pubs','opportunities_128':'128 setup pubs',
       'time_30m_h60':'30m / H60','time_60m_h60':'60m / H60'}
COLORS={'event':'#2563eb','physical_exit':'#8b5cf6','cluster':'#ea580c',
        'unique_entry_greedy':'#059669','nonoverlap_greedy':'#dc2626'}


def read(path):
    """Explicit plotting-only read of future Y, never an X input."""
    frame=pd.read_parquet(path)
    return frame.rename(columns={c:c.removeprefix('future_') for c in frame})


def save(fig,root,name):
    fig.savefig(root/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(root/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def price_panel(ax,source,rows,start,end,title,limit=5):
    tape=pd.read_parquet(source/'tape.parquet')
    bank=pd.read_parquet(source/'events.parquet')
    q=tape[(tape.t_ns>=start)&(tape.t_ns<end)]
    dt=pd.to_datetime(q.t_ns,utc=True)
    ax.plot(dt,q.bid,color='#64748b',lw=.6,label='bid',alpha=.8)
    ax.plot(dt,q.ask,color='#a5b4fc',lw=.5,label='ask',alpha=.7)
    selected=rows.head(limit)
    for j,r in enumerate(selected.itertuples(),1):
        color='#059669' if r.side=='LONG' else '#dc2626'
        entry=r.entry_ask if r.side=='LONG' else r.entry_bid
        exit_price=r.exit_bid if r.side=='LONG' else r.exit_ask
        a,b=pd.to_datetime([r.entry_ns,r.exit_ns],utc=True)
        ax.annotate('',xy=(b,exit_price),xytext=(a,entry),arrowprops=dict(arrowstyle='->',color=color,lw=1.4,alpha=.9))
        ax.scatter([a],[entry],s=25,color=color,zorder=5)
        ax.scatter([b],[exit_price],s=35,marker='x',color=color,zorder=5)
        ax.annotate(f'#{r.rank}',(b,exit_price),xytext=(-14,6+j*3),textcoords='offset points',fontsize=9,color=color)
        ax.text(1.01,.96-(j-1)*.065,f'#{r.rank} {r.side[0]} M{r.scale}/{r.role[0]} +{r.net:.2f}',
                transform=ax.transAxes,fontsize=8,color=color,va='top')
    ax.set_title(title,loc='left',fontsize=11)
    ax.set_ylabel('GC USD / oz')
    clock_format='%H:%M:%S' if end-start<10*60*1_000_000_000 else '%H:%M'
    ax.xaxis.set_major_formatter(mdates.DateFormatter(clock_format,tz=__import__('datetime').timezone.utc))
    ax.grid(alpha=.2)
    return bank


def plot(root,bank_root):
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    windows=read(root/'windows.parquet');curves=read(root/'curves.parquet');top=read(root/'top.parquet')
    matching=read(root/'matching/curves.parquet') if (root/'matching/curves.parquet').exists() else pd.DataFrame()
    active=windows[(windows.origins>0)&~windows.partial]
    modes=list(NAMES)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axs=plt.subplots(2,2,figsize=(18,10))
    for ax,column,label,scale in [(axs[0,0],'duration_s','Window duration (minutes)','log'),
                                  (axs[0,1],'n80','Unique outcomes needed for 80% mass','log'),
                                  (axs[1,0],'top10_share','Top 10 / all positive event mass','linear'),
                                  (axs[1,1],'top10_same_entry_fraction','Repeated origins within top 10','linear')]:
        data=[active.loc[active['mode']==m,column].dropna().to_numpy()/(60 if column=='duration_s' else 1) for m in modes]
        ax.boxplot(data,tick_labels=[NAMES[m] for m in modes],showfliers=False)
        ax.tick_params(axis='x',rotation=25);ax.set_ylabel(label);ax.set_yscale(scale);ax.grid(axis='y',alpha=.2)
        if 'share' in column or 'fraction' in column:ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('First raw-crossing top-N | GC sessions 4–8 May 2026\nFull quote origins; spread + $0.30/oz fee; descriptive oracle, not portfolio profit',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.93]);save(fig,out,'01_week_overview')

    fig,axs=plt.subplots(2,4,figsize=(19,9),sharex=True,sharey=True)
    for ax,mode in zip(axs.flat,modes):
        full=active[active['mode']==mode][['day','window']]
        sub=curves[curves['mode']==mode].merge(full,on=['day','window'])
        for control,color in COLORS.items():
            g=sub[sub.control==control].groupby('k').coverage_of_event_mass.mean()
            ax.plot(g.index,g.values,label=control,color=color,lw=1.6)
        if len(matching) and mode in set(matching['mode']):
            exact=matching[matching['mode']==mode].merge(active[active['mode']==mode][['day','window','oracle_mass']],on=['day','window'])
            exact['coverage']=exact.mass/exact.oracle_mass.replace(0,np.nan)
            g=exact.groupby('k').coverage.mean()
            ax.plot(g.index,g.values,color='black',ls='--',lw=1.3,label='exact_origin_event')
        ax.set_title(NAMES[mode]);ax.grid(alpha=.2);ax.set_ylim(0,1.02);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel('Top N');ax.set_ylabel('Mean coverage of full event oracle mass')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Exact distinct-origin matching retains almost all independent event mass\nGreedy controls discard alternatives; their loss is NOT the cost of enforcing uniqueness',fontsize=14)
    fig.tight_layout(rect=[0,.08,1,.92]);save(fig,out,'02_coverage_controls')

    base=active[(active['mode']=='time_30m')&(active.positive_outcomes>=10)]
    ordinary=base.iloc[(base.oracle_mass-base.oracle_mass.median()).abs().argmin()]
    rich=base.loc[base.oracle_mass.idxmax()]
    bursts=active[(active['mode']=='events_512')&(active.positive_outcomes>=10)]
    burst=bursts.loc[bursts.top10_flicker_fraction.idxmax()]
    cases=[]
    for name,record in [('ordinary',ordinary),('largest_oracle_mass',rich),('raw_flicker',burst)]:
        mode=record['mode']
        rows=top[(top.day==record.day)&(top['mode']==mode)&(top.window==record.window)]
        fig,axs=plt.subplots(2,1,figsize=(15,9),gridspec_kw={'height_ratios':[2,1]})
        source=Path(bank_root)/record.day
        bank=price_panel(axs[0],source,rows,record.start,record.end,
            f'{name}: {record.day}, {NAMES[mode]} | top 5 independent outcome representatives')
        best=rows[~rows.opportunity].iloc[0] if name=='raw_flicker' and (~rows.opportunity).any() else rows.iloc[0]
        q=pd.read_parquet(source/'tape.parquet')
        left=max(record.start,best.entry_ns-15*1_000_000_000);right=min(record.end,best.exit_ns+30*1_000_000_000)
        q=q[(q.t_ns>=left)&(q.t_ns<right)]
        axs[1].plot(pd.to_datetime(q.t_ns,utc=True),q.bid,color='#64748b',lw=.7,label='bid')
        axs[1].plot(pd.to_datetime(q.t_ns,utc=True),q.ask,color='#a5b4fc',lw=.7,label='ask')
        events=bank[(bank.scale==best.scale)&(bank.role==('lower','middle','upper').index(best.role))&(bank.t_ns>=left)&(bank.t_ns<right)]
        intervening=events[(events.t_ns>best.entry_ns)&(events.t_ns<best.exit_ns)&(events.support==best.support)]
        if len(intervening):raise ValueError('Plot candidate skips a raw first crossing')
        axs[1].scatter(pd.to_datetime(events.t_ns,utc=True),events.line,s=30,marker='x',color='#ea580c',label='every raw crossing of selected line')
        for stamp,label,color in [(best.entry_ns,'chosen entry','#059669'),(best.exit_ns,'FIRST future crossing','#dc2626')]:
            axs[1].axvline(pd.to_datetime(stamp,utc=True),color=color,ls='--',label=label)
        axs[1].set_title(f'First-crossing audit: #{int(best["rank"])} {best.side}, M{best.scale}/{best.role}, net +{best.net:.2f} USD/oz | crosses in between = 0',loc='left')
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'));axs[1].grid(alpha=.2);axs[1].legend(fontsize=8,ncol=4)
        axs[1].set_xlabel('UTC; orange markers show crossing line value, not execution price')
        fig.suptitle('Each arrow is an alternative. Overlapping arrows cannot be added as single-position capital.',fontsize=13)
        fig.tight_layout(rect=[0,0,.91,.95]);save(fig,out,'03_'+name)
        cases.append(dict(name=name,day=record.day,window=int(record.window),mode=mode,start=int(record.start),end=int(record.end)))

    # A common physical hour with 60m, two 30m, and shifted 30m selections.
    record=rich;hour=(int(record.start)//(3600*1_000_000_000))*(3600*1_000_000_000)
    fig,axs=plt.subplots(3,1,figsize=(15,12),sharex=True)
    for ax,mode in zip(axs,['time_60m','time_30m','time_30m_shift15']):
        subset=top[(top.day==record.day)&(top['mode']==mode)&(top.entry_ns>=hour)&(top.exit_ns<hour+3600*1_000_000_000)&(top['rank']<=3)]
        price_panel(ax,Path(bank_root)/record.day,subset,hour,hour+3600*1_000_000_000,NAMES[mode]+' | up to top 3 per window, arrows fully inside displayed hour',limit=9)
    axs[-1].set_xlabel('UTC');fig.suptitle('Window-boundary sensitivity on the same market hour',fontsize=15)
    fig.tight_layout(rect=[0,0,.91,.96]);save(fig,out,'04_window_boundaries')

    fig,axs=plt.subplots(1,2,figsize=(16,6))
    x=np.arange(len(modes));bottom=np.zeros(len(modes))
    for column,label,color in [('crossed','First crossing observed','#059669'),('window_censored','Window/horizon limit','#ea580c'),
                               ('support_end','Support / session end','#dc2626'),('not_ready','Line not ready','#94a3b8')]:
        values=np.array([active[active['mode']==m][column].sum()/active[active['mode']==m].queries.sum() for m in modes])
        axs[0].bar(x,values,bottom=bottom,label=label,color=color);bottom+=values
    axs[0].set_xticks(x,[NAMES[m] for m in modes],rotation=30);axs[0].yaxis.set_major_formatter(PercentFormatter(1))
    axs[0].set_ylabel('All eligible origin × line queries');axs[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2)
    axs[0].set_title('Window choice changes which outcomes can mature')
    for mode,color in [('events_512','#2563eb'),('events_2048','#ea580c'),('opportunities_128','#059669')]:
        data=np.sort(active[active['mode']==mode].duration_s.to_numpy()/60)
        axs[1].plot(data,np.arange(1,len(data)+1)/len(data),label=NAMES[mode],color=color)
    axs[1].axvline(30,color='black',ls='--',alpha=.5);axs[1].axvline(60,color='black',ls=':',alpha=.5)
    axs[1].set_xscale('log');axs[1].set_xlabel('Window duration (minutes, log scale)');axs[1].set_ylabel('Cumulative fraction of windows')
    axs[1].set_title('Event-count windows do not keep a fixed time scale');axs[1].legend();axs[1].grid(alpha=.2)
    fig.tight_layout(rect=[0,.04,1,1]);save(fig,out,'05_horizons_and_event_clock')

    if len(matching):
        fig,ax=plt.subplots(figsize=(11,6))
        for mode,g in matching[~matching.partial].groupby('mode'):
            means=g.groupby('k').retention.mean();ax.plot(means.index,means.values,label=NAMES[mode],lw=2)
        ax.set_ylim(.95,1.002);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('K distinct openings AND outcomes')
        ax.set_ylabel('Exact matching / independent event-top sum');ax.grid(alpha=.2);ax.legend()
        ax.set_title('Distinct quote timestamps do not create distinct market movements\nExact maximum-weight assignments, K=1..32; overlapping positions still allowed')
        fig.tight_layout();save(fig,out,'06_exact_matching')
    meta=dict(cases=cases,renderer=dict(python=sys.version,matplotlib=matplotlib.__version__),
              files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.png'))})
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');p.add_argument('--bank-root',default=str(ROOT/'data/derived/bb_crossing_bank_v3'))
    a=p.parse_args();plot(a.output,a.bank_root)
