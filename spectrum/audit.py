"""Reproducible random visual samples and saved-data arithmetic checks."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from .ingest import save_json
from .core.storage import sha
from .calendar import MarketCalendar


def _shade(ax,t,closed):
    edges=np.diff(np.r_[False,closed,False].astype(int))
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        ax.axvspan(t[a],t[min(b,len(t)-1)],color='#64748b',alpha=.18)


def scene(capture,features,out,start_ns,end_ns,label):
    filt=[('t_ns','>=',start_ns),('t_ns','<=',end_ns)]
    f=pq.read_table(Path(features)/'features.parquet',filters=filt).to_pandas()
    q=pq.read_table(Path(capture)/'quotes.parquet',filters=filt).to_pandas()
    b=pq.read_table(Path(capture)/'bars.parquet',filters=filt).to_pandas()
    t=pd.to_datetime(f.t_ns,unit='ns',utc=True);qt=pd.to_datetime(q.t_ns,unit='ns',utc=True)
    fig,ax=plt.subplots(8,1,figsize=(15,16),sharex=True,layout='constrained')
    fig.suptitle(label+' | UTC | midpoint from complete MBO publications',fontsize=13)
    # Downsample solely for display; extrema and OHLC oracle use every quote.
    step=max(1,len(q)//12000);display=q.iloc[::step];dt=qt.iloc[::step]
    display_good=display.valid.to_numpy()&~MarketCalendar().closed(display.t_ns.to_numpy())
    # A plotting line must not bridge unsupported silence or a scheduled break.
    display_good&=~np.r_[False,np.diff(display.t_ns.to_numpy())>30*10**9]
    ax[0].plot(dt,np.where(display_good,display.bid_nano/1e9,np.nan),lw=.6,label='bid')
    ax[0].plot(dt,np.where(display_good,display.ask_nano/1e9,np.nan),lw=.6,label='ask')
    ax[0].set_ylabel('GC price');ax[0].legend(loc='upper left',ncol=2)
    bt=pd.to_datetime(b.t_ns,unit='ns',utc=True)
    good=b.valid.to_numpy()
    ax[1].vlines(bt[good],b.l[good],b.h[good],lw=1,color='#334155',label='M1 low-high')
    ax[1].plot(bt,np.where(good,b.c,np.nan),'.-',ms=3,lw=.6,label='M1 close');ax[1].set_ylabel('M1 midpoint');ax[1].legend(loc='upper left')
    for name in ('path_30s','displacement_30s'):
        ax[2].plot(t,np.where(f[name+'__known'],f[name],np.nan),label=name,lw=.8)
    ax[2].set_ylabel('ticks');ax[2].legend(loc='upper left',ncol=2)
    for name in ('trade_buy_30s','trade_sell_30s','trade_unknown_30s'):
        ax[3].plot(t,np.where(f[name+'__known'],f[name],np.nan),label=name,lw=.8)
    ax[3].set_ylabel('contracts');ax[3].legend(loc='upper left',ncol=3)
    for name in ('fft_mid_change_32s_power','fft_mid_change_128s_power'):
        ax[4].plot(t,np.where(f[name+'__known'],f[name],np.nan),label=name,lw=.8)
    ax[4].set_ylabel('power (tick²)');ax[4].legend(loc='upper left',ncol=2)
    ax[5].step(t,f.admissible.astype(int),where='post',label='admissible',lw=1)
    ax[5].step(t,f.fft_signed_trade_128s_power__known.astype(int)*.8,where='post',label='signed-trade FFT known',lw=.8)
    ax[5].set_ylim(-.1,1.2);ax[5].legend(loc='upper left',ncol=2)
    for name in ('depth_bid_12','depth_ask_12'):
        ax[6].plot(t,np.where(f[name+'__known'],f[name],np.nan),label=name,lw=.8)
    ax[6].set_ylabel('depth contracts');ax[6].legend(loc='upper left',ncol=2)
    for name in ('add_bid_30s','add_ask_30s','remove_bid_30s','remove_ask_30s'):
        ax[7].plot(t,np.where(f[name+'__known'],f[name],np.nan),label=name,lw=.7)
    ax[7].set_ylabel('gross contracts');ax[7].legend(loc='upper left',ncol=4)
    for a in ax:
        a.grid(alpha=.2);_shade(a,t,f.scheduled_closed.to_numpy())
    ax[-1].xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz=__import__('datetime').timezone.utc))
    fig.savefig(out,dpi=120);plt.close(fig)
    # Independent OHLC extrema/close from exact completed quote prices.
    checks=0;max_error=0.
    for row in b.itertuples():
        left=row.t_ns-60*10**9
        if not row.valid or left<start_ns:continue
        event=q[(q.t_ns>left)&(q.t_ns<=row.t_ns)&q.valid]
        prior=q[(q.t_ns<=left)&q.valid].tail(1)
        if not len(event) or not len(prior) or left-int(prior.t_ns.iloc[0])>30*10**9:continue
        values=np.r_[(prior.bid_nano.to_numpy()+prior.ask_nano.to_numpy())/2e9,
                     (event.bid_nano.to_numpy()+event.ask_nano.to_numpy())/2e9]
        error=float(np.max(abs(np.array([row.o,row.h,row.l,row.c])-np.array([values[0],values.max(),values.min(),values[-1]]))))
        max_error=max(max_error,error);checks+=1
    return dict(image=Path(out).name,sha256=sha(out),start_ns=str(start_ns),end_ns=str(end_ns),
                quote_rows=len(q),feature_rows=len(f),independent_ohlc_checks=checks,max_ohlc_error=max_error,
                display_stride=step,source_capture=Path(capture).name)


def audit(market,features,output,seed=20261015,intervals=None):
    market=Path(market);features=Path(features);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();m=json.loads((market/'manifest.json').read_text());rng=np.random.default_rng(seed)
    scenes=[];coverage=[];all_bars=[];all_support=[];cal=MarketCalendar()
    periods=intervals or [('2026-05-03T22:00Z','2026-05-08T21:00Z'),('2026-05-10T22:00Z','2026-05-15T21:00Z')]
    if any(pd.Timestamp(lo)>=pd.Timestamp(hi) for lo,hi in periods):raise ValueError('Invalid audit interval')
    for capture_index,e in enumerate(m['captures']):
        name=Path(e['input']['name']).stem;cap=market/name;feat=features/name
        # Verify payloads before interpreting pictures.
        for file,h in e['payloads'].items():
            if sha(cap/file)!=h:raise ValueError('Market artifact changed '+str(cap/file))
        fm=json.loads((feat/'manifest.json').read_text())
        if sha(feat/'features.parquet')!=fm['feature_sha256']:raise ValueError('Feature artifact changed')
        f=pq.read_table(feat/'features.parquet',columns=['t_ns','available_ns','epoch','admissible','scheduled_closed','fft_mid_change_128s_power__known']).to_pandas()
        all_support.append(f[['t_ns','admissible','scheduled_closed']])
        b=pq.read_table(cap/'bars.parquet').to_pandas();all_bars.append(b)
        eligible=np.flatnonzero(f.admissible.to_numpy())
        if len(eligible):
            idx=int(rng.choice(eligible));origin=int(f.t_ns.iloc[idx]);left=max(int(f.t_ns.iloc[0]),origin-450*10**9);right=min(int(f.t_ns.iloc[-1]),origin+450*10**9)
            scenes.append(scene(cap,feat,output/f'random_{name}.png',left,right,f'Random sample {name} | '+str(pd.Timestamp(origin,tz='UTC'))))
        open_rows=~f.scheduled_closed.to_numpy()
        coverage.append(dict(capture=name,rows=len(f),open_seconds=int(open_rows.sum()),admitted_open=int(f.admissible[open_rows].sum()),
                             spectral_open=int(f.fft_mid_change_128s_power__known[open_rows].sum()),first_ns=str(int(f.t_ns.iloc[0])),last_ns=str(int(f.t_ns.iloc[-1]))))
        if capture_index in (1,7):
            day=pd.Timestamp(e['input']['start_ns'],tz='UTC').normalize().value
            scenes.append(scene(cap,feat,output/f'break_{name}.png',day+int(20.5*3600*1e9),day+int(22.5*3600*1e9),'Daily break 21:00–22:00 UTC | '+name))
    bars=pd.concat(all_bars,ignore_index=True).sort_values('t_ns');bars.to_parquet(output/'audit_bars.parquet',index=False)
    fig,axes=plt.subplots(3,1,figsize=(16,10),sharex=True,layout='constrained')
    t=pd.to_datetime(bars.t_ns,unit='ns',utc=True)
    price=bars.c.where(bars.valid).to_numpy();breaks=np.r_[True,np.diff(bars.t_ns)>60*10**9];price[breaks]=np.nan
    axes[0].plot(t,price,lw=.6);axes[0].set_ylabel('GC MBO midpoint');axes[0].set_title('Two research weeks | source resets retained | UTC')
    coverage_values=bars.coverage.to_numpy(copy=True);coverage_values[breaks]=np.nan
    axes[1].plot(t,coverage_values,lw=.5,label='MBO M1 observed coverage');axes[1].set_ylim(-.05,1.05);axes[1].legend()
    grid=pd.date_range(periods[0][0],periods[-1][1],freq='min');closed=cal.closed(grid.asi8)
    axes[2].step(grid,(~closed).astype(int),where='post',label='frozen observed-hours calendar');axes[2].legend()
    for a in axes:a.grid(alpha=.2);_shade(a,grid,closed)
    axes[-1].xaxis.set_major_formatter(md.DateFormatter('%m-%d\n%H:%M'))
    fig.savefig(output/'two_weeks.png',dpi=120);plt.close(fig)
    weeks=[];support=pd.concat(all_support,ignore_index=True)
    for lo,hi in periods:
        begin=pd.Timestamp(lo).value;end=pd.Timestamp(hi).value
        # Bars are close-labelled; calendar support applies to the preceding minute.
        bb=bars[(bars.t_ns>begin)&(bars.t_ns<=end)]
        opened=~cal.closed(bb.t_ns.to_numpy()-1)
        expected=int((~cal.closed(pd.date_range(lo,hi,freq='min',inclusive='left').asi8)).sum())
        ss=support[(support.t_ns>=begin)&(support.t_ns<end)&~support.scheduled_closed]
        weeks.append(dict(start=lo,end=hi,expected_open_minutes=expected,observed_open_bar_rows=int(opened.sum()),valid_open_bars=int(bb.valid[opened].sum()),
                          valid_open_minute_fraction=float(bb.valid[opened].sum()/expected),
                          expected_open_seconds=expected*60,observed_open_second_rows=len(ss),admitted_open_seconds=int(ss.admissible.sum()),
                          admitted_open_second_fraction=float(ss.admissible.sum()/(expected*60)),
                          basis='midpoint from MBO; no cTrader OHLC substitution'))
    checks=sum(x['independent_ohlc_checks'] for x in scenes)
    enough=all(w['valid_open_minute_fraction']>=.95 and w['admitted_open_second_fraction']>=.8 for w in weeks)
    passed=checks>=100 and all(x['independent_ohlc_checks']>0 and x['max_ohlc_error']<1e-8 for x in scenes) and enough
    result=dict(status='PASS' if passed else 'FAIL',seed=seed,
                sampling='one uniformly random admissible origin per source-reset capture; fixed seed; no outcome filter',
                scenes=scenes,coverage=coverage,weeks=weeks,elapsed_seconds=time.perf_counter()-start,
                overview_sha256=sha(output/'two_weeks.png'),calendar=cal.contract(),
                independent_ohlc_checks=checks,coverage_gate=dict(min_valid_open_minute_fraction=.95,min_admitted_open_second_fraction=.8,min_ohlc_checks=100),
                completeness='Measured against every expected open minute/second; missing boundary rows remain missing, not filled')
    save_json(output/'audit.json',result)
    html='<!doctype html><meta charset="utf-8"><title>Market reconstruction audit</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;background:#f8fafc;color:#0f172a}img{width:100%;height:auto}article{margin:25px 0;background:white;padding:15px;border:1px solid #cbd5e1}a{color:#0369a1}</style><h1>Two-week market replay and feature audit</h1><p>UTC. Grey: scheduled breaks. Missing features remain masked. These are representation checks, not trading evidence.</p><article><img src="two_weeks.png"></article>'
    for s in scenes:html+=f'<article><h2>{s["image"]}</h2><p>OHLC checks: {s["independent_ohlc_checks"]}; max error {s["max_ohlc_error"]:.3g}</p><img loading="lazy" src="{s["image"]}"></article>'
    (output/'index.html').write_text(html,encoding='utf8')
    return result
