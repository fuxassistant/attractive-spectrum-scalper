"""Self-contained price/flow/response inspector and reproducible audit PNGs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .core.storage import sha
from .ingest import save_json
from .full_history import ObservedCalendar


def refresh_ui(out):
    """Attach only an independently audited result with matching payload identity."""
    out=Path(out);payload=json.loads((out/'ui_data.json').read_text())
    audit_path=out.parent/'sol61_audit'/'verdict.json';payload['independent_audit']=None
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        if audit.get('run_predictions_sha256')==sha(out/'predictions.parquet') and audit.get('run_comparison_sha256')==sha(out/'comparison.json'):
            payload['independent_audit']=audit
    save_json(out/'ui_data.json',payload)
    template=Path(__file__).resolve().parents[1]/'scripts/response_lab_ui.html'
    html=template.read_text(encoding='utf8').replace('__PAYLOAD__',json.dumps(payload,allow_nan=False).replace('</',r'<\/'))
    (out/'index.html').write_text(html,encoding='utf8')
    if (out/'ui_receipt.json').exists():
        receipt=json.loads((out/'ui_receipt.json').read_text())
        receipt.update(html_sha256=sha(out/'index.html'),data_sha256=sha(out/'ui_data.json'))
        save_json(out/'ui_receipt.json',receipt)


def export_view(root,out,frame,predictions,receipts):
    root=Path(root);out=Path(out)
    cfg=json.loads((root/'run_config.json').read_text());cal=ObservedCalendar(cfg['reference']['path'],cfg['reference']['sha256'])
    chosen=[]
    # Fixed calendar coverage, not a selection of attractive price outcomes.
    for month,g in frame.groupby(frame.date.str[:7]):
        dates=sorted(g.date.unique());day=dates[len(dates)//2]
        candidates=g[g.date==day];row=candidates.iloc[len(candidates)//2]
        chosen.append((row,'calendar midpoint '+month))
    for field,reason in [('A__unknown_fraction_512','maximum unknown share (exploratory)'),('C__classified_flow_to_return_8_16_coherence','maximum coherence (exploratory)')]:
        chosen.append((frame.loc[frame[field].idxmax()],reason))
    bykey={r['key']:r for r in receipts};scenes=[]
    for row,reason in chosen:
        r=bykey[row.key];center=int(row.t_ns);left=center-3600*10**9;right=center+3600*10**9
        d=pq.read_table(Path(r['market'])/'seconds.parquet',filters=[('t_ns','>=',left-30*10**9),('t_ns','<=',right)]).to_pandas()
        support=d.valid.to_numpy()&~cal.closed(d.t_ns.to_numpy())
        edge=np.r_[False,support[1:]&support[:-1]&(np.diff(d.t_ns)==10**9)&(np.diff(d.epoch)==0)]
        good=pd.Series(edge).rolling(30,min_periods=30).sum().to_numpy()==30
        buy=d.c_trade_buy.diff(30);sell=d.c_trade_sell.diff(30);unknown=d.c_trade_unknown.diff(30)
        tape=[]
        for i in range(30,len(d),5):
            tape.append([int(d.t_ns.iloc[i]//10**6),float(d.mid.iloc[i]) if support[i] else None,
                float(buy.iloc[i]-sell.iloc[i]) if good[i] else None,float(unknown.iloc[i]) if good[i] else None,
                float(d.depth_bid_12.iloc[i]) if support[i] else None,float(d.depth_ask_12.iloc[i]) if support[i] else None])
        f=frame[(frame.key==row.key)&(frame.t_ns>=left)&(frame.t_ns<=right)]
        pr=predictions[(predictions.key==row.key)&(predictions.t_ns>=left)&(predictions.t_ns<=right)].set_index('t_ns')
        windows=[]
        for z in f.itertuples(index=False):
            p=pr.loc[z.t_ns]
            windows.append(dict(ms=int(z.decision_ns//10**6),endpoint_ms=int(z.t_ns//10**6),available_ms=int(z.decision_ns//10**6),
                pressure=(z.A__pressure_lower_512+z.A__pressure_upper_512)/2,lower=z.A__pressure_lower_512,upper=z.A__pressure_upper_512,
                unknown=z.A__unknown_fraction_512,response=z.A__ticks_per_100_512,
                coherence=[getattr(z,f'C__classified_flow_to_return_{a}_{b}_coherence') for a,b in [(4,8),(8,16),(16,32)]],
                coherence_known=[bool(getattr(z,f'C__classified_flow_to_return_{a}_{b}_known')) for a,b in [(4,8),(8,16),(16,32)]],
                gain=float(np.expm1(z.C__classified_flow_to_return_8_16_gain_log)),
                gain_known=bool(z.C__classified_flow_to_return_8_16_known),
                A=float(p.A_30),B=float(p.B_30),C=float(p.C_30),
                future=float(p.y_30) if p.mature_30 else None))
        split='train' if row.date<'2026-03-01' else 'validation' if row.date<'2026-05-01' else 'development evaluation'
        scenes.append(dict(id=len(scenes),date=row.date,symbol=row.symbol,reason=reason,split=split,center_ms=center//10**6,tape=tape,windows=windows))
    summary=json.loads((out/'summary.json').read_text());comparison=json.loads((out/'comparison.json').read_text())
    payload=dict(scenes=scenes,summary={k:v for k,v in summary.items() if k!='reports'},comparison=comparison)
    save_json(out/'ui_data.json',payload)
    refresh_ui(out)
    render_pngs(scenes,out)
    save_json(out/'ui_receipt.json',dict(scenes=len(scenes),html_sha256=sha(out/'index.html'),data_sha256=sha(out/'ui_data.json'),
        plotted='5-second sampled price/depth and trailing 30-second flow; spectral windows every 128 seconds, trailing 512 seconds',
        warnings=['All plotted future outcomes are retrospective, separate from input features.','Charts show classified flow and unknown volume separately; pressure bounds apply to total-window sums, not spectral confidence.']))


def render_pngs(scenes,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    (out/'png').mkdir(exist_ok=True)
    for scene in scenes:
        a=np.array([[np.nan if v is None else v for v in row] for row in scene['tape']],float)
        t=pd.to_datetime(a[:,0],unit='ms',utc=True);w=scene['windows'];tw=pd.to_datetime([r['ms'] for r in w],unit='ms',utc=True)
        fig,axs=plt.subplots(5,1,figsize=(14,10),sharex=True,layout='constrained')
        fig.suptitle(f"{scene['date']} {scene['symbol']} | {scene['split']} | {scene['reason']}")
        axs[0].plot(t,a[:,1],color='#20639b',lw=1);axs[0].set_ylabel('Midpoint USD')
        axs[1].plot(t,a[:,2],label='classified buy-sell / 30s',lw=.8);axs[1].plot(t,a[:,3],label='unknown volume / 30s',color='orange',lw=.8);axs[1].legend(loc='upper right');axs[1].set_ylabel('Contracts')
        axs[2].plot(tw,[r['pressure'] for r in w],label='known net / total volume',color='purple');axs[2].fill_between(tw,[r['lower'] for r in w],[r['upper'] for r in w],alpha=.25,label='total-flow bounds');axs[2].set_ylim(-1.05,1.05);axs[2].legend(loc='upper right');axs[2].set_ylabel('Pressure / 512s')
        for i,band in enumerate(['4-8s','8-16s','16-32s']):axs[3].plot(tw,[r['coherence'][i] if r['coherence_known'][i] else np.nan for r in w],label=band)
        axs[3].set_ylim(0,1);axs[3].legend(loc='upper right');axs[3].set_ylabel('Coherence')
        axs[4].plot(tw,[r['response'] for r in w],label='net ticks / 100 total contracts');axs[4].axhline(0,color='grey',lw=.6);axs[4].legend(loc='upper right');axs[4].set_ylabel('Realized response');axs[4].xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz=t.tz));axs[4].set_xlabel('UTC; features plotted at publication time; trailing 512-second windows')
        for ax in axs:ax.grid(alpha=.2)
        fig.savefig(out/'png'/f"scene_{scene['id']:02d}.png",dpi=130);plt.close(fig)
