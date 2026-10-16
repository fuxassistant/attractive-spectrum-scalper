"""Static audit scenes from the same exported series as the interactive UI.
These are scientific plots, not browser screenshots.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectrum.ingest import save_json

def payload(path):
    s=path.read_text(encoding='utf-8');return json.loads(s[len('window.loadDataset('):-2])

def build(root='.',run='run_001'):
    out=Path(root)/'reports/player_v2'/run
    files=[f'61017__selected_test__2026-05-{d}.js' for d in ('11','12','15')]
    files+=['61015__sampled_test_101__2026-05-11.js','61016__selected_test__2026-05-13.js','61015__validation_epoch_060__2026-05-08.js']
    receipts=[];plt.rcParams.update({'figure.dpi':130,'text.parse_math':False,'axes.grid':True,'grid.alpha':.2})
    for n,file in enumerate(files):
        d=payload(out/'data'/file);p=np.asarray(d['intent']);gate=np.asarray(d['gate']);t=np.asarray(d['t']);candles=np.asarray(d['candles'],float)
        ix=np.flatnonzero(gate>0);center=int(ix[np.argmax(p[ix])]) if len(ix) else len(p)//2
        if d['trades']:center=d['trades'][0]['decision_index']
        a=max(0,center-90);b=min(len(p),center+150);x=(t[a:b]-t[center])/60000
        fig,axs=plt.subplots(5,1,figsize=(14,11),layout='constrained',sharex=True,gridspec_kw={'height_ratios':[2,1,1,1,1]})
        axs[0].plot(x,candles[a:b,3],label='Closed M1 midpoint',color='#3e658a');axs[0].fill_between(x,candles[a:b,2],candles[a:b,1],alpha=.12,color='#3e658a')
        for tr in d['trades']:
            if a<=tr['decision_index']<b:
                xx=(tr['entry_ms']-t[center])/60000;end=(tr['exit_ms']-t[center])/60000
                axs[0].scatter([xx],[tr['entry_nano']/1e9],marker='^' if tr['side']>0 else 'v',color='#009980')
                axs[0].scatter([end],[tr['exit_nano']/1e9],marker='x',color='#d65345')
                axs[0].annotate(f'{tr["reason"]}\n${tr["net"]:+.2f}'+(' censored' if tr['censored'] else ''),(end,tr['exit_nano']/1e9),fontsize=7,xytext=(3,8),textcoords='offset points')
        axs[1].plot(x,p[a:b],label='Raw participation intention',color='#008b78');axs[1].plot(x,gate[a:b],label='After action mask',color='#d48636',ls='--');axs[1].axhline(.5,color='grey',ls=':');axs[1].set_ylim(0,1)
        direction=np.asarray(d['direction']);plans=np.asarray(d['plans']);chosen=np.argmax(direction,axis=1)
        axs[2].plot(x,direction[a:b,0],label='P(LONG | trade)',color='#477fc7')
        for k,label in enumerate(('fast','medium','wide')):axs[2].plot(x,plans[np.arange(a,b),chosen[a:b],k],label='P('+label+' | chosen direction)',ls='--')
        axs[2].set_ylim(0,1);axs[3].plot(x,np.asarray(d['equity'])[a:b],label='Liquidation PnL before decision',color='#7851a6')
        fx=np.asarray(d['features'])
        for k in (0,1,2,3):axs[4].plot(x,fx[a:b,k],label=d['feature_names'][k])
        for ax in axs:ax.axvline(0,color='grey',ls=':');ax.legend(loc='upper left',fontsize=8,ncol=2)
        axs[0].set_ylabel('USD/oz');axs[3].set_ylabel('USD');axs[4].set_ylabel('Train-scaled X');axs[4].set_xlabel('Minutes from chosen center (first trade, else eligible peak)')
        from datetime import datetime,timezone
        stamp=datetime.fromtimestamp(int(t[center])/1000,timezone.utc).isoformat()
        fig.suptitle(f'Seed {d["seed"]} · {d["simulation"]} · {d["date"]} · center {stamp}\nExact UI data series; static audit plot, not browser screenshot')
        name=f'signal_scene_{n:02d}.png';fig.savefig(out/name);plt.close(fig)
        receipts.append(dict(file=name,data=file,center_index=center,method='First trade or eligible peak',data_sha256=__import__('hashlib').sha256((out/'data'/file).read_bytes()).hexdigest()))
    # Compare actual Friday policy paths at fixed epochs, not a fabricated interpolation.
    fig,axs=plt.subplots(3,1,figsize=(14,9),layout='constrained',sharex=True)
    for ax,seed in zip(axs,(61015,61016,61017)):
        for epoch in (0,1,10,30,60):
            d=payload(out/'data'/f'{seed}__validation_epoch_{epoch:03d}__2026-05-08.js');x=(np.asarray(d['t'])-d['london_ms'])/3600000
            ax.plot(x,d['intent'],label=f'epoch {epoch}')
        ax.axhline(.5,color='grey',ls=':');ax.set_title(f'{seed}: raw participation by epoch on own account trajectory');ax.set_ylim(0,1);ax.legend(ncol=5,fontsize=8)
    axs[-1].set_xlabel('Hours relative to London open; negative hours are context-only');fig.savefig(out/'epoch_signal_overlay.png');plt.close(fig)
    save_json(out/'signal_scene_receipt.json',receipts)

if __name__=='__main__':build()
