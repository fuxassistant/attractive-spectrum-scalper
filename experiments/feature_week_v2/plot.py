"""Standalone research figures from verified V2 payloads; no fitted outcomes."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/derived/feature_week_v2';RUN=OUT/'final_parallel2';AUDIT=OUT/'audit'
FIG=AUDIT/'figures';FIG.mkdir(exist_ok=True)
NS=10**9;DATES=('2026-08-23','2026-08-24','2026-08-25','2026-08-26','2026-08-27','2026-08-28')
plt.rcParams.update({'figure.facecolor':'#f8fafc','axes.facecolor':'#ffffff','axes.spines.top':False,'axes.spines.right':False,
                     'font.size':10,'axes.grid':True,'grid.alpha':.18,'savefig.facecolor':'#f8fafc','font.family':'DejaVu Sans'})


def times(f):return pd.to_datetime(f.available_ns,utc=True)
def finish(fig,name,title,axes):
    fig.suptitle(title,fontsize=16,fontweight='bold',x=.08,ha='left')
    for ax in np.ravel(axes):
        ax.xaxis.set_major_formatter(md.DateFormatter('%d Aug\n%H:%M' if name=='01_week.png' else '%H:%M',tz=__import__('datetime').timezone.utc))
        if name=='04_same_price.png':
            ax.xaxis.set_major_locator(md.SecondLocator(bysecond=[0,30]))
            ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S',tz=__import__('datetime').timezone.utc))
    np.ravel(axes)[-1].set_xlabel('UTC · actual publication time')
    fig.align_ylabels();fig.tight_layout(rect=(0,0,1,.94));fig.savefig(FIG/name,dpi=150);plt.close(fig)


def read_window(date,lo,hi):
    a,b=pd.Timestamp(lo,tz='UTC').value,pd.Timestamp(hi,tz='UTC').value
    return pd.read_parquet(RUN/'stores'/date/'fast.parquet',filters=[('t_ns','>=',a),('t_ns','<=',b)])


def week():
    m=pd.read_parquet(RUN/'minutes.parquet');start=pd.Timestamp('2026-08-23 22:00',tz='UTC').value
    m=m[m.t_ns>=start];t=times(m)
    fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
    axes[0].plot(t,m.close.where(m.known),color='#164e63',lw=.8,label='Completed M1 close')
    axes[0].plot(t,m.bb1+ m.bb1_half,color='#ea580c',lw=.6,label='Completed SMA20 ± 2 sample SD')
    axes[0].plot(t,m.bb1- m.bb1_half,color='#ea580c',lw=.6);axes[0].set_ylabel('GCZ6 · USD');axes[0].legend(loc='upper left')
    axes[1].plot(t,m.volume,color='#0e7490',lw=.6,label='All T prints');axes[1].plot(t,m.unknown,color='#c026d3',lw=.6,label='Unknown aggressor')
    axes[1].set_yscale('symlog',linthresh=10);axes[1].set_ylabel('Contracts / minute');axes[1].legend(loc='upper left')
    parts=[]
    for d in DATES:
        x=pd.read_parquet(RUN/'days'/d/'features.parquet',columns=['t_ns','available_ns','admissible','fft_total_trade_128s_power__known','fft_signed_trade_128s_power__known'])
        x=x[x.t_ns>=start];x['minute']=x.t_ns//(60*NS)*(60*NS)
        parts.append(x.groupby('minute')[['admissible','fft_total_trade_128s_power__known','fft_signed_trade_128s_power__known']].mean())
    z=pd.concat(parts).groupby(level=0).mean()
    for n,label,color in [('admissible','Decision admission','#64748b'),('fft_total_trade_128s_power__known','Observed total activity','#0d9488'),('fft_signed_trade_128s_power__known','Strict signed activity','#be123c')]:
        axes[2].plot(pd.to_datetime(z.index,utc=True),z[n]*100,label=label,lw=.7,color=color)
    axes[2].set_ylabel('Known seconds / min · %');axes[2].legend(loc='upper left',ncol=3)
    finish(fig,'01_week.png','Week coverage: price, executed volume, and separate feature masks',axes)


def midnight():
    f=read_window('2026-08-24','2026-08-23 23:55','2026-08-24 00:10')
    x=pd.read_parquet(RUN/'stores/2026-08-24/features.parquet',columns=['t_ns','available_ns','vwap_exact_60s_center','vwap_exact_60s_center__known','vwap_exact_3600s_center__known'])
    x=x[(x.t_ns>=f.t_ns.min())&(x.t_ns<=f.t_ns.max())]
    old=[]
    for d in ('2026-08-23','2026-08-24'):
        p=ROOT/'data/derived/feature_week_v1/serial_v2/days'/d/'features.parquet'
        old.append(pd.read_parquet(p,columns=['t_ns','available_ns','vwap_exact_60s_center__known','vwap_exact_3600s_center__known'],filters=[('t_ns','>=',int(f.t_ns.min())),('t_ns','<=',int(f.t_ns.max()))]))
    old=pd.concat(old,ignore_index=True);t=times(f)
    fig,axes=plt.subplots(4,1,figsize=(13,9),sharex=True)
    axes[0].plot(t,f.mid,color='#164e63',label='Last completed second mid',lw=.8)
    axes[0].plot(times(x),x.vwap_exact_60s_center.where(x.vwap_exact_60s_center__known),color='#ea580c',label='Exact T VWAP 60s',lw=1);axes[0].legend(loc='upper left');axes[0].set_ylabel('USD')
    axes[1].bar(t,f.trade_quantity,width=1/86400,color='#0e7490');axes[1].bar(t,f.trade_unknown,width=1/86400,color='#c026d3');axes[1].set_ylabel('T contracts / s')
    for a,n,title in [(axes[2],'vwap_exact_60s_center__known','60s VWAP'),(axes[3],'vwap_exact_3600s_center__known','3600s VWAP')]:
        a.plot(times(x),x[n].astype(float),label='V2 continuous history',color='#059669',lw=1.5)
        a.plot(times(old),old[n].astype(float),label='V1 reset by UTC file',color='#b91c1c',lw=1,ls='--')
        a.set_ylim(-.2,1.2);a.set_ylabel(title+' known');a.legend(loc='lower left',ncol=2)
    for a in axes:a.axvline(pd.Timestamp('2026-08-24',tz='UTC'),color='#475569',ls='--',lw=1)
    finish(fig,'02_midnight.png','Identical-book handoff preserves history; source endpoint is not a new market session',axes)


def event_gap():
    f=read_window('2026-08-28','2026-08-28 13:58:30','2026-08-28 14:04:00');t=times(f)
    fig,axes=plt.subplots(4,1,figsize=(13,10),sharex=True)
    axes[0].plot(t,f.mid.where(f.known),color='#164e63',lw=1);axes[0].set_ylabel('Known mid · USD')
    axes[1].bar(t,f.trade_quantity,width=1/86400,color='#0e7490',label='All T');axes[1].bar(t,f.trade_unknown,width=1/86400,color='#c026d3',label='Unknown side');axes[1].set_ylabel('Contracts / s');axes[1].legend(loc='upper left')
    axes[2].plot(t,f.depth_bid_12.where(f.known),color='#059669',label='Bid depth L12');axes[2].plot(t,f.depth_ask_12.where(f.known),color='#dc2626',label='Ask depth L12');axes[2].set_ylabel('Visible contracts');axes[2].legend(loc='upper left')
    for i,(n,label) in enumerate([('known','Price'),('trade_known','T volume'),('ledger_known','Interaction'),('mechanism_known','30s descriptor')]):
        axes[3].plot(t,f[n].astype(float)*.7+i,label=label,lw=1)
    axes[3].set_yticks(np.arange(4)+.35,['Price','T volume','Interaction','30s descriptor']);axes[3].set_ylabel('Separate masks')
    for a in axes:a.axvspan(pd.Timestamp('2026-08-28 14:00:02',tz='UTC'),pd.Timestamp('2026-08-28 14:00:07',tz='UTC'),color='#94a3b8',alpha=.3)
    finish(fig,'03_real_gap.png','Real Friday support break stays unknown; histories recover at their own pace',axes)


def same_price():
    f=pd.read_parquet(RUN/'days/2026-08-26/fast.parquet')
    good=f[f.mechanism_known & f.known & f.ret30_ticks.abs().le(1) & f.range30_ticks.le(8) & f.ret300_ticks.notna()].copy()
    good['price_key']=np.rint(good.mid*20).astype(int)
    low=good.groupby('price_key').trade_quantity30.idxmin();high=good.groupby('price_key').trade_quantity30.idxmax()
    gap=good.loc[high,'trade_quantity30'].to_numpy()-good.loc[low,'trade_quantity30'].to_numpy()
    j=int(np.argmax(gap));indices=[int(low.iloc[j]),int(high.iloc[j])]
    fig,axes=plt.subplots(3,2,figsize=(14,9),sharex='col');records=[]
    for col,idx in enumerate(indices):
        r=f.loc[idx];w=f[(f.t_ns>r.t_ns-120*NS)&(f.t_ns<=r.t_ns)];t=times(w)
        axes[0,col].plot(t,(w.mid-r.mid)/.1,color='#164e63');axes[0,col].set_title(('Quiet' if col==0 else 'Active')+' prefix → same endpoint '+f'{r.mid:.2f} USD')
        axes[0,col].set_ylabel('Mid displacement · ticks')
        for a in axes[:,col]:a.axvspan(pd.Timestamp(int(r.t_ns)-30*NS,tz='UTC'),pd.Timestamp(int(r.t_ns),tz='UTC'),color='#94a3b8',alpha=.15)
        axes[1,col].bar(t,w.trade_quantity,width=1/86400,color='#0891b2');axes[1,col].bar(t,w.trade_unknown,width=1/86400,color='#c026d3');axes[1,col].set_ylabel('All / unknown T · contracts')
        for n,label,c in [('bid_near_fill30','Bid executions','#059669'),('ask_near_fill30','Ask executions','#dc2626'),('bid_near_withdrawal30','Bid withdrawals','#65a30d'),('ask_near_withdrawal30','Ask withdrawals','#f97316')]:
            axes[2,col].plot(t,w[n].where(w.mechanism_known),label=label,color=c,lw=.9)
        axes[2,col].set_ylabel('Near-book quantities / 30s');axes[2,col].legend(fontsize=8,loc='upper left',ncol=2)
        records.append(dict(t_ns=int(r.t_ns),available_ns=int(f.available_ns.loc[idx]),utc=str(pd.Timestamp(int(r.t_ns),tz='UTC')),
            mid=float(r.mid),trade_quantity30=float(r.trade_quantity30),unknown_fraction30=float(r.unknown_fraction30),
            bid_depth12=float(r.depth_bid_12),ask_depth12=float(r.depth_ask_12),ret30_ticks=float(r.ret30_ticks),
            withdrawal30=float(r.bid_near_withdrawal30+r.ask_near_withdrawal30)))
    finish(fig,'04_same_price.png','Matched endpoint and near-zero 30s displacement: different observed flow',axes)
    (AUDIT/'same_price_cases.json').write_text(json.dumps(dict(selection='Largest 30s volume contrast at identical midpoint, absolute 30s return <=1 tick and 30s range <=8 ticks; exploratory descriptive selection, no outcomes',cases=records),indent=2))


def spectral():
    n=128;rng=np.random.default_rng(2901);w=.5-.5*np.cos(2*np.pi*np.arange(n)/n);freq=np.fft.rfftfreq(n);band=(freq>=1/8)&(freq<=1/4)
    p=abs(np.fft.rfft((lambda z:z-z.mean(axis=1,keepdims=True))(rng.normal(size=(8192,n)))*w))**2/(w@w)
    p[:,1:-1]*=2;p[:,0]=0;frac=p[:,band].sum(axis=1)/p.sum(axis=1)
    a=json.loads((AUDIT/'audit.json').read_text())['spectral_controls'];ref=a['white_reference']
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].hist(frac,bins=70,color='#64748b',alpha=.9);axes[0].axvline(ref,color='#ea580c',label=f'Ratio of expected powers = {ref:.3f}');axes[0].set_xlabel('Raw energy fraction in 4–8 second band');axes[0].set_ylabel('IID Gaussian windows');axes[0].legend()
    axes[1].hist(frac/ref,bins=70,color='#0e7490');axes[1].axvline(1,color='#ea580c',label='White reference');axes[1].axvline(a['positive_6second_cycle'],color='#059669',label='6 second sinusoid');axes[1].set_xlabel('Band power / matching white reference');axes[1].legend()
    fig.suptitle('Large band energy is not by itself a cycle; bandwidth and taper need a reference',fontsize=14);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(FIG/'05_spectral_controls.png',dpi=150);plt.close(fig)


def cadence():
    c=pd.read_parquet(AUDIT/'cadence.parquet').groupby('name')[['rows','known','pairs','changes']].sum()
    names=['spread_ticks','depth_balance_12','trade_buy_1s','trade_buy_30s','vwap_exact_60s_center','vwap_exact_3600s_center','fft_mid_change_128s_power','fft_signed_trade_128s_power','fft_total_trade_128s_power','activity_30s_unknown_fraction']
    names=[n for n in names if n in c.index];a=c.loc[names]
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    axes[0].barh(names,100*a.known/a.rows,color='#0e7490');axes[0].set_xlim(0,100);axes[0].set_xlabel('Known / all weekday UTC rows · %')
    axes[1].barh(names,100*a.changes/a.pairs,color='#ea580c');axes[1].set_xlim(0,100);axes[1].set_xlabel('Changed / adjacent known pairs · %')
    fig.suptitle('Coverage and update rate are separate: changing every second is not independent information',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(FIG/'06_cadence.png',dpi=150);plt.close(fig)


def book():
    receipt=json.loads((AUDIT/'l3_snapshot.json').read_text());b=pd.DataFrame(receipt['levels']);b['price']=b.price_nano/NS
    bids=b[b.side=='B'].sort_values('price',ascending=False).head(12);asks=b[b.side=='A'].sort_values('price').head(12)
    fig,ax=plt.subplots(figsize=(10,8));ax.barh(bids.price,-bids.quantity,height=.075,color='#059669',label='Bid contracts');ax.barh(asks.price,asks.quantity,height=.075,color='#dc2626',label='Ask contracts')
    for part,sign in ((bids,-1),(asks,1)):
        for r in part.itertuples():ax.annotate(f'{r.quantity} · {r.orders} orders',(sign*r.quantity,r.price),xytext=(5*sign,0),textcoords='offset points',ha='left' if sign>0 else 'right',va='center',fontsize=9)
    ax.set_xlim(-bids.quantity.max()*1.55,asks.quantity.max()*1.55);ax.axvline(0,color='#94a3b8',lw=1)
    ax.set_yticks(sorted(set(bids.price)|set(asks.price)));ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.set_ylabel('Absolute price · USD');ax.set_xlabel('Bid ← visible contracts → ask');ax.legend(loc='lower right')
    stamp=str(pd.Timestamp(receipt['available_ns'],tz='UTC'))
    fig.suptitle('Actual L3 book: top 12 levels on each side',fontweight='bold',fontsize=16)
    ax.set_title(f'{stamp}\n{receipt["orders"]:,} order IDs verified identical across the UTC file snapshot',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(FIG/'07_l3_snapshot.png',dpi=150);plt.close(fig)


def model_masks():
    receipts=[json.loads(p.read_text()) for p in sorted((AUDIT/'snapshots').glob('*.json'))]
    wanted=['spread_ticks','depth_balance_12','displacement_30s','path_30s','trade_buy_30s','trade_sell_30s','trade_unknown_30s',
            'activity_30s_unknown_fraction','activity_30s_signed_lower_bound','activity_30s_signed_upper_bound',
            'fft_signed_trade_128s_power','fft_total_trade_128s_power','fft_unknown_trade_128s_power','fft_unknown_trade_128s_entropy',
            'vwap_exact_60s_center','vwap_exact_300s_center','vwap_exact_3600s_center']
    names=['Native BBO tradable']+wanted;columns=[]
    for r in receipts:
        known=dict(zip(r['names']['second'],r['second_known']));columns.append([r['market']['known']]+[known[n] for n in wanted])
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(figsize=(13,7));ax.imshow(np.array(columns).T,aspect='auto',cmap=ListedColormap(['#e2e8f0','#0d9488']),vmin=0,vmax=1)
    ax.set_yticks(range(len(names)),names);labels=[pd.Timestamp(r['t_ns'],tz='UTC').strftime('%d Aug\n%H:%M:%S') for r in receipts]
    ax.set_xticks(range(len(receipts)),labels,fontsize=8);ax.grid(False)
    ax.set_title('Actual as-of observations · green = known, grey = unknown\nHistorical features can remain visible while the current native book is invalid',fontsize=13)
    fig.tight_layout();fig.savefig(FIG/'08_model_masks.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    for fn in (week,midnight,event_gap,same_price,spectral,cadence,book,model_masks):fn();print(fn.__name__,flush=True)
