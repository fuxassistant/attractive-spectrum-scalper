"""Bounded, calendar-selected pilot of the optional volatility/VWAP plugin.

Preserves raw/core datasets. Saves feature rows only at existing event origins,
plus decimated plotting samples; never trains a policy or joins future labels.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .lab import Context,REGISTRY,compute
from .plugins.volatility_vwap import proxy,SCALES,PROXY
from .full_history import ObservedCalendar
from .core.storage import sha
from .ingest import save_json


def process(job):
    receipt,reference,event_path,output=job;out=Path(output);key=receipt['key'];stem=key.replace('/','_')
    p=Path(receipt['market'])/'seconds.parquet'
    if sha(p)!=receipt['market_payloads']['seconds.parquet']: raise ValueError('Source hash drift')
    started=time.perf_counter();f=pq.read_table(p).to_pandas()
    cal=ObservedCalendar(reference['path'],reference['sha256']);c=Context(f,cal)
    proxy(c);elapsed=time.perf_counter()-started
    events=pq.read_table(event_path,filters=[('key','=',key)]).to_pandas()
    positions=np.searchsorted(f.t_ns.to_numpy(),events.t_ns.to_numpy())
    if not np.array_equal(f.t_ns.to_numpy()[positions],events.t_ns): raise ValueError('Event clock join failed')
    data={n:events[n].to_numpy() for n in ('key','date','symbol','t_ns','decision_ns','event_class','activity')}
    for name,v in c.values.items():data[name]=v[positions];data[name+'__known']=c.masks[name][positions]
    table=pa.table(data);pq.write_table(table,out/'captures'/(stem+'.parquet'),compression='zstd')
    if not pq.read_table(out/'captures'/(stem+'.parquet')).equals(table):raise ValueError('Feature roundtrip mismatch')
    # Choose time by calendar/order only, never by attractive future outcomes.
    center=int(events.iloc[len(events)//2].t_ns) if len(events) else int(f.t_ns.iloc[len(f)//2])
    ix=np.flatnonzero((abs(f.t_ns.to_numpy()-center)<=1200*10**9)&(f.t_ns.to_numpy()%(5*10**9)==0))
    names=['t_ns','available_ns','mid','epoch'];plot={n:f[n].to_numpy()[ix] for n in names}
    plot['mid']=np.where(c.supported[ix],plot['mid'],np.nan)
    for name,v in c.values.items():plot[name]=np.where(c.masks[name][ix],v[ix],np.nan)
    pq.write_table(pa.table(plot),out/'plots'/(stem+'.parquet'),compression='zstd')
    checks=[]
    # Independent weighted central formula at 16 predetermined positions.
    mid=f.mid.to_numpy();volume=f.c_trade_buy.diff()+f.c_trade_sell.diff()+f.c_trade_unknown.diff()
    for n in SCALES:
        mask=c.masks[f'vwap_proxy_{n}s_center'];eligible=np.flatnonzero(mask)
        selected=eligible[np.linspace(0,len(eligible)-1,min(4,len(eligible))).astype(int)] if len(eligible) else []
        for end in selected:
            px=mid[end-n+1:end+1];w=volume.iloc[end-n+1:end+1].to_numpy()
            mean=np.average(px,weights=w);sigma=np.sqrt(np.average((px-mean)**2,weights=w))/.1
            error_mu=abs(c.values[f'vwap_proxy_{n}s_center'][end]-mean)
            error_sigma=abs(c.values[f'vwap_proxy_{n}s_sigma_ticks'][end]-sigma)
            if error_mu>1e-8 or error_sigma>1e-6:raise ValueError('Independent central-moment check failed')
            checks.append(dict(window=n,end=int(end),center_error=float(error_mu),sigma_error=float(error_sigma)))
    report=dict(key=key,date=receipt['entry']['selection']['date'],symbol=receipt['entry']['selection']['symbol'],
        seconds_rows=len(f),event_rows=len(events),features=len(c.values),seconds=elapsed,
        source_sha256=sha(p),features_sha256=sha(out/'captures'/(stem+'.parquet')),
        coverage_at_events={name:float(c.masks[name][positions].mean()) if len(positions) else None for name in c.values},
        independent_checks=checks,units=c.units)
    save_json(out/'captures'/(stem+'.json'),report)
    return report


def render(out,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    out=Path(out);(out/'png').mkdir(exist_ok=True)
    for i,r in enumerate(reports):
        d=pq.read_table(out/'plots'/(r['key'].replace('/','_')+'.parquet')).to_pandas()
        ts=pd.to_datetime(d.t_ns.to_numpy(),unit='ns',utc=True);fig,axes=plt.subplots(5,1,figsize=(16,12),sharex=True,
            gridspec_kw={'height_ratios':[2,2,1,1,1]})
        colors={60:'#be7331',300:'#17877c',900:'#526bb0',3600:'#9556aa'}
        for ax,scales in [(axes[0],(900,3600)),(axes[1],(60,300))]:
            ax.plot(ts,d.mid,label='GC midpoint',color='#263348',lw=1)
            for n in scales:
                ax.plot(ts,d[f'vwap_proxy_{n}s_center'],color=colors[n],label=f'Volume-weighted midpoint {n//60}min')
                ax.fill_between(ts,d[f'vwap_proxy_{n}s_lower2'],d[f'vwap_proxy_{n}s_upper2'],color=colors[n],alpha=.12,label=f'+/-2 price sigma {n//60}min')
            ax.set_ylabel('USD');ax.legend(loc='upper left',fontsize=8,ncol=2)
        for n in SCALES:axes[2].plot(ts,d[f'vwap_proxy_{n}s_sigma_ticks'],color=colors[n],label=f'{n//60}min')
        axes[2].set_ylabel('Price sigma, ticks');axes[2].legend(loc='upper left',fontsize=8,ncol=4)
        axes[3].plot(ts,d.vwap_proxy_60s_return_rms_1s,label='60s RMS of 1s change',color=colors[60])
        axes[3].plot(ts,d.vwap_proxy_300s_return_rms_1s,label='300s RMS of 1s change',color=colors[300])
        axes[3].set_ylabel('Return RMS, ticks');axes[3].legend(loc='upper left',fontsize=8,ncol=2)
        axes[4].plot(ts,d.vwap_proxy_60s_volume_rate,label='All T volume / second, 60s',color=colors[60])
        axes[4].plot(ts,d.vwap_proxy_300s_volume_rate,label='All T volume / second, 300s',color=colors[300])
        axes[4].set_ylabel('Contracts / second');axes[4].legend(loc='upper left',fontsize=8,ncol=2)
        ax2=axes[4].twinx();ax2.plot(ts,d.vwap_proxy_60s_unknown_fraction,color='#777',ls=':',alpha=.8);ax2.set_ylim(0,1);ax2.set_ylabel('Unknown share (dotted)')
        for ax in axes:ax.grid(alpha=.15)
        axes[-1].xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz=ts.tz));axes[-1].set_xlabel('UTC endpoint; features usable only at publication time. Gaps masked.')
        fig.suptitle(f'{r["date"]} {r["symbol"]} | Multi-scale volatility + volume\nPROXY: second endpoint midpoint weighted by total T volume. NOT trade-price VWAP. Calendar-selected scene.')
        fig.tight_layout();fig.savefig(out/'png'/f'scene_{i:02d}.png',dpi=110);plt.close(fig)


def run(root,events,output,workers):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    for folder in ('captures','plots'):(out/folder).mkdir()
    config=json.loads((root/'run_config.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    e=pq.read_table(events,columns=['key','date']).to_pandas().drop_duplicates()
    selected=[]
    for month,g in e.groupby(e.date.str[:7]):
        g=g.sort_values(['date','key']);selected.append(g.iloc[len(g)//2]['key'])
    receipts=sorted([r for r in manifest['captures'] if r['key'] in selected],key=lambda r:r['entry']['selection']['date'])
    plugin_file=Path(__file__).with_name('plugins')/'volatility_vwap.py'
    protocol=dict(status='PILOT',basis='MIDPOINT_VOLUME_PROXY_NOT_TRADE_VWAP',scales_seconds=list(SCALES),
        multipliers=[1,2],sigma='volume-weighted population central price dispersion; not return volatility or confidence interval',
        reset='full contiguous support required per window; observed calendar / epoch / finite price / nonnegative counters',
        selection='one chronological midpoint capture per calendar month, from existing admitted event catalog',
        source_manifest_sha256=sha(root/'manifest.json'),source_events_sha256=sha(events),
        plugin_sha256=sha(plugin_file),runner_sha256=sha(__file__),lab_sha256=sha(Path(__file__).with_name('lab.py')),
        no_training=True,workers=workers,selected=selected,exact_trade_moments_in_dataset=False)
    save_json(out/'protocol.json',protocol);started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reports=list(pool.map(process,[(r,config['reference'],str(events),str(out)) for r in receipts]))
    event_features=pa.concat_tables([pq.read_table(out/'captures'/(r['key'].replace('/','_')+'.parquet')) for r in reports])
    pq.write_table(event_features,out/'event_features.parquet',compression='zstd')
    summarize_seconds=time.perf_counter()-started
    render(out,reports)
    names=[n for n in event_features.column_names if n.startswith('vwap_proxy_') and not n.endswith('__known')]
    save_json(out/'feature_contract.json',dict(name=PROXY.name,version=PROXY.version,features=names,units=reports[0]['units'],
        halo_seconds=PROXY.lookback_seconds,x_only=True,target_columns=[],basis=protocol['basis'],
        input_columns_exact=list(__import__('spectrum.plugins.volatility_vwap',fromlist=['EXACT_COLUMNS']).EXACT_COLUMNS),
        caveat='Exact plugin is implemented/tested but not run on real exact moments: the current dataset lacks them.'))
    summary=dict(status='COMPLETE_PROXY_INTEGRATION_PILOT',captures=len(reports),rows=event_features.num_rows,features=len(names),
        seconds_rows=sum(r['seconds_rows'] for r in reports),compute_and_read_seconds=summarize_seconds,
        including_plots_seconds=time.perf_counter()-started,independent_checks=sum(len(r['independent_checks']) for r in reports),
        max_sigma_error=max(z['sigma_error'] for r in reports for z in r['independent_checks']),
        event_features_sha256=sha(out/'event_features.parquet'),reports=reports)
    save_json(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='reports'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--market',default='data/derived/full_mbo_v1')
    p.add_argument('--events',default='reports/event_landmarks_v1/run_003/causal_events.parquet')
    p.add_argument('--output',default='reports/volatility_vwap_v1/pilot_001');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.market,a.events,a.output,a.workers)
