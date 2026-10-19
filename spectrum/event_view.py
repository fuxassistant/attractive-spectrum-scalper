"""Event landmark visual evidence; explicitly separates recognition and hindsight."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .core.storage import sha
from .full_history import ObservedCalendar
from .ingest import save_json


def export(root,out):
    root=Path(root);out=Path(out)
    events=pq.read_table(out/'events.parquet').to_pandas()
    events=events[events.row_kind=='event'].sort_values(['date','t_ns']).reset_index(drop=True)
    manifest=json.loads((root/'manifest.json').read_text());bykey={r['key']:r for r in manifest['captures']}
    cfg=json.loads((root/'run_config.json').read_text());cal=ObservedCalendar(cfg['reference']['path'],cfg['reference']['sha256'])
    summary=json.loads((out/'summary.json').read_text())
    chosen=[]
    for cls,g in events.groupby('event_class'):
        chosen.append((g.iloc[len(g)//2], 'chronological midpoint of '+cls))
    # Transparent selection for contrast; isolated is a FUTURE descriptive label.
    fresh=events[(events.event_class=='fresh_impulse')&(events.recurrence_mature==True)]
    for isolated,g in fresh.groupby('retrospectively_isolated'):
        chosen.append((g.iloc[len(g)//2], 'retrospective contrast: '+('no repeat' if isolated else 'later repeat')))
    rng=np.random.default_rng(6118)
    for month,g in events.groupby(events.date.str[:7]):
        if month in ('2025-10','2025-12','2026-02','2026-04','2026-06','2026-08'):
            chosen.append((g.iloc[int(rng.integers(len(g)))], 'seed6118 sample '+month))
    scenes=[]
    (out/'png').mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    plt.rcParams.update({'font.size':9})
    for k,(row,reason) in enumerate(chosen):
        center=int(row.decision_ns);left=center-720*10**9;right=center+240*10**9
        source=Path(bykey[row.key]['market'])/'seconds.parquet'
        d=pq.read_table(source,filters=[('t_ns','>=',left-5*10**9),('t_ns','<=',right)]).to_pandas()
        t=d.t_ns.to_numpy();ok=d.valid.to_numpy()&~cal.closed(t)
        edge=np.r_[False,ok[1:]&ok[:-1]&(np.diff(t)==10**9)&(np.diff(d.epoch)==0)]
        good=pd.Series(edge).rolling(5).sum().to_numpy()==5
        q=d.c_trade_buy.diff(5)-d.c_trade_sell.diff(5);u=d.c_trade_unknown.diff(5)
        v=d.c_trade_buy.diff(5)+d.c_trade_sell.diff(5)+u
        tape=[]
        for i in np.flatnonzero((t%(5*10**9)==0)&(t>=left)):
            tape.append([int(t[i]//10**6),float(d.mid.iloc[i]) if ok[i] else None,
                float(q.iloc[i]) if good[i] else None,float(u.iloc[i]) if good[i] else None,
                float(d.depth_bid_12.iloc[i]) if ok[i] else None,float(d.depth_ask_12.iloc[i]) if ok[i] else None,
                float(v.iloc[i]) if good[i] else None,float((d.ask.iloc[i]-d.bid.iloc[i])/.1) if ok[i] else None,
                int(d.available_ns.iloc[i]//10**6)])
        nearby=events[(events.key==row.key)&(events.decision_ns>=left)&(events.decision_ns<=right)]
        markers=[]
        for e in nearby.itertuples():
            markers.append(dict(ms=int(e.decision_ns//10**6),price=e.event_end_price,event_class=e.event_class,
                                activity=e.activity,volume=e.volume,move=e.move_ticks))
        fields=['event_class','activity','move_ticks','volume','unknown_fraction','pressure_lower','pressure_upper',
                'prior_events_180','interval_cv','interval_cv_known','last_gap_s','move_threshold','volume_threshold',
                'event_start_price','event_end_price','direction','spread_before','spread_after','opposite_depth_change_fraction',
                'return_peak5_share','flow_peak5_share','classified_flow_to_return_original','classified_flow_to_return_remove5',
                'classified_flow_to_return_interpolate15','recurrence_mature','retrospectively_isolated','future_events_180',
                'mature_30','mature_120','continue_net_30','continue_net_120','fade_net_30','fade_net_120',
                'origin_retest_120','extends3ticks_120','both_landmarks_120']
        info={n:row[n].item() if isinstance(row[n],np.generic) else row[n] for n in fields}
        scene=dict(id=k,date=row.date,symbol=row.symbol,key=row.key,reason=reason,center_ms=center//10**6,tape=tape,markers=markers,info=info)
        scenes.append(scene)
        a=np.array([[np.nan if x is None else x for x in z] for z in tape],float);ts=pd.to_datetime(a[:,0],unit='ms',utc=True)
        fig,axes=plt.subplots(4,1,figsize=(15,9),sharex=True,gridspec_kw={'height_ratios':[2,1,1,1]})
        axes[0].plot(ts,a[:,1],color='#202c4c');axes[0].set_ylabel('Midprice USD')
        for field,color in [('event_start_price','#c0771e'),('event_end_price','#397c98')]:
            axes[0].axhline(info[field],color=color,lw=.8,ls='--',label=field)
        for mark in markers:
            mt=pd.Timestamp(mark['ms'],unit='ms',tz='UTC')
            axes[0].scatter(mt,mark['price'],marker='v',color='#a92854' if mark['event_class']=='fresh_impulse' else '#7954b3',s=30)
        axes[0].legend(loc='upper left',fontsize=8)
        axes[1].plot(ts,a[:,2],label='classified buy - sell /5s',color='#157a69')
        axes[1].fill_between(ts,a[:,2]-a[:,3],a[:,2]+a[:,3],alpha=.2,color='#d19b32',label='unknown assignment bounds')
        axes[1].legend(loc='upper left',fontsize=8);axes[1].set_ylabel('Contracts')
        axes[2].plot(ts,a[:,4],label='bid depth12');axes[2].plot(ts,a[:,5],label='ask depth12');axes[2].legend(loc='upper left',fontsize=8)
        axes[2].set_ylabel('Depth contracts')
        axes[3].plot(ts,a[:,6],color='#7563ad',label='all trades /5s');axes[3].set_ylabel('Volume')
        origin=pd.Timestamp(center,unit='ns',tz='UTC')
        for ax in axes:
            ax.axvline(origin,color='black',lw=1);ax.axvspan(origin,ts[-1],alpha=.07,color='red');ax.grid(alpha=.15)
        axes[-1].xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz=ts.tz));axes[-1].set_xlabel('UTC; red shade = FUTURE after selected recognition (audit only)')
        fig.suptitle(f'{row.date} {row.symbol} | {info["event_class"]} / {info["activity"]}\n{reason}; known flow partial, unknown retained; no periodicity or profit claim')
        fig.tight_layout();fig.savefig(out/'png'/f'scene_{k:02d}.png',dpi=120);plt.close(fig)
    payload=dict(summary={k:v for k,v in summary.items() if k!='reports'},scenes=scenes,
                 events_sha256=sha(out/'events.parquet'),summary_sha256=sha(out/'summary.json'))
    save_json(out/'ui_data.json',payload)
    refresh(out)
    save_json(out/'ui_receipt.json',dict(scenes=len(scenes),selection='4 class midpoints + 2 hindsight contrasts + 6 seeded calendar samples',
        events_sha256=sha(out/'events.parquet'),html_sha256=sha(out/'index.html'),data_sha256=sha(out/'ui_data.json'),
        png_sha256={p.name:sha(p) for p in sorted((out/'png').glob('*.png'))}))


def refresh(out):
    out=Path(out);payload=json.loads((out/'ui_data.json').read_text());audit=out.parent/'sol61_audit'/'verdict.json'
    payload['audit']=None
    if audit.exists():
        a=json.loads(audit.read_text())
        if a.get('events_sha256')==payload['events_sha256'] and a.get('summary_sha256')==payload['summary_sha256']:payload['audit']=a
    template=Path(__file__).resolve().parents[1]/'scripts/event_landmarks_ui.html'
    (out/'index.html').write_text(template.read_text(encoding='utf8').replace('__PAYLOAD__',json.dumps(payload,allow_nan=False).replace('</',r'<\/')),encoding='utf8')
    if (out/'ui_receipt.json').exists():
        receipt=json.loads((out/'ui_receipt.json').read_text());receipt['html_sha256']=sha(out/'index.html')
        save_json(out/'ui_receipt.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--market',default='data/derived/full_mbo_v1');p.add_argument('--output',default='reports/event_landmarks_v1/run_003')
    a=p.parse_args();export(a.market,a.output)
