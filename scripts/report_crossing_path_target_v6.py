"""Deterministic visual semantic audit. Case selection uses future labels only
for diagnostics and is never a performance estimate or a training split.
"""
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition import bb_crossing as bb
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from scripts.build_crossing_path_target_v6 import sha, save

ROOT = Path('data/derived/crossing_path_target_v6')
OUT = Path('reports/crossing_path_target_v6')
BASE = json.loads(Path('configs/research/opportunity_pipeline_v5.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2,
                     'figure.facecolor': '#f8fafc', 'axes.facecolor': 'white'})
COLORS = ['#087e8b', '#d95d39']
IMAGES = []


def finish(fig, name):
    fig.savefig(OUT/name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    IMAGES.append(name)


def load_tape(date):
    return QuoteTape.from_day(MonthDay(Path(BASE['data']['market_root'])/date[:7]/date, verify=True))


def trace(row):
    q = load_tape(row.date)
    e, x = int(row.entry_ordinal), int(row.cross_ordinal)
    t = (q.quote_t[e:x+1]-q.quote_t[e])/1e9
    y = ((q.bid[e:x+1]-q.ask[e])/1e9-.3)/row.R_event
    return t, y


def case_dict(row):
    names = ['date', 'split', 'scale', 'role', 'v6_event_id', 'cause', 'elapsed_seconds',
        'net_R_L', 'mae_R_L', 'max_drawdown_R_L', 'mean_negative_R_L', 'underwater_fraction_L',
        'velocity_R_per_T_L', 'risk_before_cross_L']
    result = {k: row[k].item() if isinstance(row[k], np.generic) else row[k] for k in names}
    for key in ('t_ns', 'entry_ns', 'cross_ns', 'entry_ordinal', 'cross_ordinal'):
        result[key] = str(row[key])
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETE'
    columns = ['date', 'split', 't_ns', 'entry_ns', 'cross_ns', 'entry_ordinal', 'cross_ordinal',
        'scale', 'role', 'profile_known', 'v6_event_id', 'cause', 'R_event', 'elapsed_seconds',
        'price_move_R', 'band_move_R', 'relative_move_R', 'price_step', 'line_step', 'y_crossing']
    for side in ('L', 'S'):
        columns += [f'{k}_{side}' for k in ('net_R', 'mae_R', 'max_drawdown_R', 'mean_positive_R',
            'mean_negative_R', 'underwater_fraction', 'velocity_R_per_T', 'risk_before_cross', 'efficiency')]
    columns += [f'path_q{i:02d}_R_L' for i in range(17)]
    frames = []
    for d in manifest['days']:
        p = ROOT/f"{d['date']}_target.parquet"
        assert sha(p) == d['outputs'][p.name]
        f = pd.read_parquet(p, columns=columns)
        frames.append(f.loc[f.profile_known])
    a = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()
    stats = dict(status='COMPLETE', version='CROSSING_PATH_TARGET_V6',
        roots=sum(x['roots'] for x in manifest['days']), cells=sum(x['cells'] for x in manifest['days']),
        hits=len(a), no_hit=sum(x['no_hit'] for x in manifest['days']),
        censored=sum(x['censored'] for x in manifest['days']), not_ready=sum(x['not_ready'] for x in manifest['days']),
        zero_time_hits=sum(x['zero_time_hits'] for x in manifest['days']),
        unique_events=int(a.v6_event_id.nunique()),
        independent_raw_paths=sum(x['checks']['independent_paths'] for x in manifest['days']),
        max_probe_error=max(x['checks']['max_probe_error'] for x in manifest['days']),
        causes={k: int(v) for k, v in a.cause.value_counts().items()}, splits={},
        units='Price units per one unit exposure; not contract dollar PnL or executable exit.',
        no_fit=True)
    for split in BASE['splits']:
        b = a.loc[a.split == split]
        receipts = [d for d in manifest['days'] if d['split'] == split]
        stats['splits'][split] = dict(cells=sum(d['cells'] for d in receipts), hits=len(b),
            unique_events=int(b.v6_event_id.nunique()),
            positive_long_fraction=float((b.net_R_L>0).mean()),
            positive_long_with_prior_risk_fraction=float(b.loc[b.net_R_L>0, 'risk_before_cross_L'].mean()))
    stats['delay_quantiles_s'] = {str(p): float(a.elapsed_seconds.quantile(p)) for p in (.01, .1, .5, .9, .99)}
    stats['subsecond_fraction'] = float((a.elapsed_seconds<1).mean())
    abs_v = abs(a.velocity_R_per_T_L.to_numpy())
    short = a.elapsed_seconds.to_numpy()<1
    stats['subsecond_share_squared_velocity'] = float((abs_v[short]**2).sum()/(abs_v**2).sum())
    stats['max_absolute_velocity_R_per_minute'] = float(np.max(abs_v))
    stats['positive_long_with_prior_risk_fraction'] = float(a.loc[a.net_R_L>0, 'risk_before_cross_L'].mean())
    samples = a[[f'path_q{i:02d}_R_L' for i in range(17)]].to_numpy()
    missed = samples.min(axis=1)-a.mae_R_L.to_numpy()
    stats['sampled_17point_mae_underestimate_over_half_R'] = float((missed>.5).mean())
    stats['sampled_17point_mae_underestimate_q99_R'] = float(np.quantile(missed, .99))

    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle('V6 target population | 20 frozen sessions | conditional on first-line hit', fontsize=16)
    for split, color in zip(BASE['splits'], ['#186a8b', '#7e57c2', '#e38d2c', '#b43b4e']):
        b = a[a.split==split]
        ax[0, 0].hist(b.elapsed_seconds, bins=np.linspace(0,300,61), density=True, histtype='step', label=split, color=color)
        ax[0, 1].hist(b.net_R_L.clip(-10,10), bins=np.linspace(-10,10,81), density=True, histtype='step', label=split, color=color)
    ax[0,0].set(xlabel='Seconds from aligned entry to crossing', ylabel='Conditional density'); ax[0,0].legend()
    ax[0,1].set(xlabel='LONG net / causal R (display clipped at +/-10)', ylabel='Conditional density')
    every = a.iloc[::max(1, len(a)//40000)]
    hb=ax[1,0].hexbin(every.net_R_L, every.mean_negative_R_L, gridsize=65, mincnt=1, bins='log', cmap='viridis')
    ax[1,0].set(xlabel='LONG terminal net / R', ylabel='Mean underwater magnitude / R', xlim=(-10,10), ylim=(0,6))
    fig.colorbar(hb, ax=ax[1,0], label='Sample count (log color)')
    table = a.groupby(['scale','role']).agg(hit=('v6_event_id','size'), positive=('net_R_L',lambda x: (x>0).mean()))
    x=np.arange(9); ax[1,1].bar(x, table['hit']/stats['roots'], color='#186a8b')
    ax[1,1].set(xticks=x, xticklabels=[f'M{s}\n{bb.ROLES[r]}' for s,r in table.index], ylabel='Fraction of all origins with a hit', ylim=(0,1))
    finish(fig,'01_population.png')

    fig, ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    fig.suptitle('Raw profit / time is unstable near zero elapsed time',fontsize=16)
    ax[0].scatter(every.elapsed_seconds, every.velocity_R_per_T_L, s=2, alpha=.2)
    ax[0].set(xscale='log', yscale='symlog', xlabel='Elapsed seconds (log)', ylabel='Net / R per minute (symlog)')
    ax[1].hist(np.log10(np.maximum(abs_v,1e-12)),bins=90,color='#186a8b')
    ax[1].set(xlabel='log10 |net / R per minute|',ylabel='Hit cells')
    ax[1].text(.04,.95, f"<1 s: {stats['subsecond_fraction']:.1%} of hits\n"
        f"{stats['subsecond_share_squared_velocity']:.6%} of raw velocity squared\n"
        'Squared magnitude is a zero-prediction MSE diagnostic,\nnot a trained model gradient.', transform=ax[1].transAxes,va='top',fontsize=10,
        bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    finish(fig,'02_velocity_instability.png')

    # Matched examples from one existing TEST day: future-selected diagnostic.
    pool=a[(a.date=='2026-09-01') & a.net_R_L.between(1,5) & a.elapsed_seconds.between(30,240)].copy()
    pool['net_bin']=np.floor(pool.net_R_L/.2).astype(int)
    pool['time_bin']=np.floor(pool.elapsed_seconds/10).astype(int)
    best=None
    for _, group in pool.groupby(['scale','role','net_bin','time_bin']):
        lo=group.loc[group.mean_negative_R_L.idxmin()]
        alternatives=group[group.v6_event_id != lo.v6_event_id]
        if alternatives.empty: continue
        hi=alternatives.loc[alternatives.mean_negative_R_L.idxmax()]
        difference=hi.mean_negative_R_L-lo.mean_negative_R_L
        if best is None or difference>best[0]: best=(difference,lo,hi)
    if best is None: raise ValueError('no prespecified matched pair')
    pair=[best[1],best[2]]
    cases={'selection':'2026-09-01, LONG net 1..5R, duration 30..240s, same scale/role, net bin .2R, time bin 10s, different event IDs; maximize mean-negative contrast',
           'matched_pair':[case_dict(r) for r in pair]}
    fig,ax=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
    fig.suptitle('Similar endpoint and time, different exposure | future-selected diagnostic pair',fontsize=15)
    for col,r in enumerate(pair):
        tt,yy=trace(r)
        ax[0,col].step(tt,yy,where='post',color=COLORS[col],lw=1,label='Full quote liquidation equity')
        ax[0,col].fill_between(tt,yy,0,where=yy<0,step='post',alpha=.18,color='#d95d39')
        ss=np.array([r[f'path_q{i:02d}_R_L'] for i in range(17)])
        ax[0,col].plot(np.linspace(0,r.elapsed_seconds,17),ss,'o',ms=3,color='black',label='17 shape samples')
        ax[0,col].axhline(-1,color='red',ls='--',lw=.8,label='Reference risk -1R')
        ax[0,col].set(title=f"{r.date} {pd.to_datetime(r.entry_ns,utc=True):%H:%M:%S} UTC | M{r.scale} {bb.ROLES[int(r.role)]}",
            xlabel='Seconds after entry',ylabel='LONG net / R')
        ax[0,col].legend(fontsize=8,loc='upper left')
        ax[1,col].axis('off')
        ax[1,col].text(.02,.94,
            f"Net {r.net_R_L:.3f} R   |   time {r.elapsed_seconds:.3f} s\n"
            f"Average velocity {r.velocity_R_per_T_L:.3f} R/min\n"
            f"MAE {r.mae_R_L:.3f} R   |   max drawdown {r.max_drawdown_R_L:.3f} R\n"
            f"Mean negative exposure {r.mean_negative_R_L:.3f} R\n"
            f"Time underwater {r.underwater_fraction_L:.1%}\n"
            f"Reference risk crossed BEFORE event: {bool(r.risk_before_cross_L)}\n"
            f"Crossing cause: {r.cause}",va='top',fontsize=12,linespacing=1.5)
        from spectrum.research_v6.targets import haar_details
        hh=haar_details(ss[None,:])[0]
        ax[2,col].bar(np.arange(15),hh,color=COLORS[col])
        ax[2,col].set(xlabel='Signed Haar details: 1 | 2 | 4 | 8 (coarse to fine)',ylabel='Shape coefficients / R')
        for cut in (.5,2.5,6.5): ax[2,col].axvline(cut,color='black',alpha=.3,lw=.7)
    for axes in (ax[0],ax[2]):
        limits=(min(axis.get_ylim()[0] for axis in axes),max(axis.get_ylim()[1] for axis in axes))
        for axis in axes: axis.set_ylim(limits)
    finish(fig,'03_matched_paths.png')

    # Fixed unselected origin: full 9-cell map, including missing outcomes.
    full=pd.read_parquet(ROOT/'2026-09-01_target.parquet')
    clock=pd.Timestamp('2026-09-01T13:30:00Z').value
    origin=int(full.t_ns.iloc[np.argmin(abs(full.t_ns.to_numpy()-clock))])
    card=full[full.t_ns==origin].sort_values(['scale','role'])
    fig,ax=plt.subplots(1,5,figsize=(16,6),constrained_layout=True)
    fig.suptitle('Fixed origin: 2026-09-01 13:30 UTC | nine line outcomes | blank = no observed hit by 300 s',fontsize=13)
    labels=[f'M{r.scale} {bb.ROLES[int(r.role)]}' for r in card.itertuples()]
    specs=[('elapsed_seconds','Time, s'),('net_R_L','LONG net / R'),('net_R_S','SHORT net / R'),
        ('mae_R_L','LONG MAE / R'),('mean_negative_R_L','LONG mean negative / R')]
    for j,(name,title) in enumerate(specs):
        vals=card[name].to_numpy(float)
        cmap=plt.get_cmap('coolwarm' if j in(1,2,3) else 'viridis').copy();cmap.set_bad('#e7e9ec')
        finite=vals[np.isfinite(vals)]
        extent=max(abs(finite).max(),.01) if len(finite) else 1
        if j in (1,2,3): extent=float(np.nanmax(abs(card[['net_R_L','net_R_S','mae_R_L']].to_numpy())))
        im=ax[j].imshow(vals[:,None],aspect='auto',cmap=cmap,
            vmin=-extent if j in(1,2,3) else 0,vmax=extent)
        ax[j].set(yticks=np.arange(9),yticklabels=labels if j==0 else [],xticks=[],title=title)
        for i,v in enumerate(vals): ax[j].text(0,i,f'{v:.2f}' if np.isfinite(v) else '—',ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=1))
    finish(fig,'04_nine_line_map.png')
    cases['fixed_origin_ns']=str(origin)

    fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    fig.suptitle('Crossing mechanisms | moving line and price are distinct coordinates',fontsize=15)
    for cause,col in zip(('PRICE','LINE','JOINT_NEEDED','BOTH_SUFFICIENT'),('#186a8b','#e38d2c','#b43b4e','#7e57c2')):
        sub=every[every.cause==cause]
        if len(sub): ax[0].scatter(sub.price_step,sub.line_step,s=4,alpha=.3,label=f'{cause} (all hits: {stats["causes"].get(cause,0):,})',color=col)
    ax[0].set(xlabel='Price change at strict crossing anchor',ylabel='Line change at strict crossing anchor',xscale='symlog',yscale='symlog'); ax[0].legend(fontsize=8)
    hb=ax[1].hexbin(every.price_move_R,every.band_move_R,gridsize=75,mincnt=1,bins='log',cmap='viridis')
    ax[1].set(xlabel='Price move entry to cross / R',ylabel='Band move entry to cross / R',xlim=(-10,10),ylim=(-10,10))
    fig.colorbar(hb,ax=ax[1],label='Sample count (log color)')
    finish(fig,'05_price_vs_band.png')

    # Fixed time window, reference gradients only, no invented fitted predictions.
    window=full[(full.scale==1)&(full.role==2)&full.t_ns.between(clock,clock+30*60*10**9)].copy()
    xx=pd.to_datetime(window.t_ns,utc=True)
    fig,ax=plt.subplots(4,1,figsize=(14,11),sharex=True,constrained_layout=True)
    fig.suptitle('Fixed 30-minute window | target continuity and reference gradients | no model fit',fontsize=15)
    ax[0].step(xx,np.where(window.y_crossing>=0,window.y_crossing>0,np.nan),where='post',label='V5 any-line amplitude event',lw=1)
    ax[0].step(xx,np.where(window.hit_known,window.hit,np.nan),where='post',label='V6 M1 upper first-line hit by 300s',alpha=.7)
    ax[0].set(ylabel='Binary labels');ax[0].legend(loc='upper right',fontsize=8)
    ax[1].plot(xx,window.net_R_L,label='net / R');ax[1].plot(xx,window.mae_R_L,label='MAE / R');ax[1].plot(xx,window.mean_negative_R_L,label='mean negative / R')
    ax[1].set(ylabel='Conditional profile');ax[1].legend(loc='upper right',fontsize=8)
    ax[2].plot(xx,window.elapsed_seconds,color='#186a8b');ax[2].set(ylabel='Time to hit, s')
    ax[3].plot(xx,np.where(window.hit_known,.5-window.hit,np.nan),label='BCE hit: p=.5, grad=.5-y',alpha=.8)
    ax[3].plot(xx,-np.clip(window.net_R_L,-1,1),label='Huber net at prediction=0, delta=1',alpha=.8)
    ax[3].plot(xx,-np.clip(window.mean_negative_R_L,-1,1),label='Huber negative exposure at prediction=0',alpha=.8)
    ax[3].set(ylabel='Unweighted per-cell derivative',xlabel='UTC | gaps are masks, not zero labels');ax[3].legend(loc='upper right',fontsize=8)
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=xx.dt.tz))
    finish(fig,'06_target_and_reference_gradient.png')
    known=full.hit_known
    hit=full.profile_known
    stats['reference_loss_diagnostics_sep01'] = dict(
        hit_positive_fraction=float(full.loc[known,'hit'].mean()),
        long_net_huber_delta1_saturated_fraction=float((abs(full.loc[hit,'net_R_L'])>=1).mean()),
        mean_negative_huber_delta1_saturated_fraction=float((full.loc[hit,'mean_negative_R_L']>=1).mean()),
        definition='p=.5 BCE and zero-prediction unit Huber; not fitted gradients or an accepted loss recipe')
    # Additional contrast controls MAE as well as endpoint and duration.
    pool['net_bin']=np.floor(pool.net_R_L/.5).astype(int)
    pool['time_bin']=np.floor(pool.elapsed_seconds/20).astype(int)
    pool['mae_bin']=np.floor(pool.mae_R_L/.5).astype(int)
    best_area=None
    for _, group in pool.groupby(['scale','role','net_bin','time_bin','mae_bin']):
        lo=group.loc[group.mean_negative_R_L.idxmin()]
        alternatives=group[group.v6_event_id!=lo.v6_event_id]
        if alternatives.empty: continue
        hi=alternatives.loc[alternatives.mean_negative_R_L.idxmax()]
        difference=hi.mean_negative_R_L-lo.mean_negative_R_L
        if best_area is None or difference>best_area[0]: best_area=(difference,lo,hi)
    if best_area is None: raise ValueError('no MAE-controlled pair')
    area_pair=[best_area[1],best_area[2]]
    cases['area_pair_selection']='Same Sep01 pool and scale/role; net .5R bin, duration 20s bin, MAE .5R bin; different IDs; maximize mean-negative contrast'
    cases['area_pair']=[case_dict(r) for r in area_pair]
    stats['area_pair']=cases['area_pair']
    fig,ax=plt.subplots(2,2,figsize=(14,8),constrained_layout=True,sharey='row')
    fig.suptitle('Similar endpoint, duration AND MAE: area distinguishes persistence of loss',fontsize=15)
    for col,r in enumerate(area_pair):
        tt,yy=trace(r)
        ax[0,col].step(tt,yy,where='post',color=COLORS[col],lw=1)
        ax[0,col].fill_between(tt,yy,0,where=yy<0,step='post',alpha=.2,color=COLORS[col])
        ax[0,col].set(title=f"{pd.to_datetime(r.entry_ns,utc=True):%H:%M:%S} UTC | net {r.net_R_L:.2f} R | {r.elapsed_seconds:.1f}s | MAE {r.mae_R_L:.2f} R",
            xlabel='Seconds after entry',ylabel='LONG net / R')
        rr=full[(full.t_ns==r.t_ns)&(full.scale==r.scale)&(full.role==r.role)].iloc[0]
        v=np.array([rr[f'quarter_velocity_{i}_R_per_T_L'] for i in range(4)])
        ax[1,col].bar(np.arange(4),v,color=COLORS[col])
        ax[1,col].set(xticks=np.arange(4),xticklabels=['0–25%','25–50%','50–75%','75–100%'],
            xlabel='Future event interval quarters (not known at entry)',ylabel='Change velocity, R/min',
            title=f'Mean negative {r.mean_negative_R_L:.2f} R | underwater {r.underwater_fraction_L:.1%}')
    finish(fig,'07_area_beyond_mae.png')
    stats['matched_pair']=cases['matched_pair']
    stats['artifact_inputs']={str(ROOT/'manifest.json'):sha(ROOT/'manifest.json')}
    stats['images']={name:sha(OUT/name) for name in IMAGES}
    save(OUT/'summary.json',stats)
    save(OUT/'visual_cases.json',cases)
    save(OUT/'data_receipt.json',manifest)
    text=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Crossing path target V6</title>
<style>body{{font:17px/1.55 system-ui;margin:32px auto;max-width:1300px;padding:0 24px;background:#f8fafc;color:#142638}}img{{width:100%;background:white;margin:16px 0}}article{{background:white;padding:20px;margin:18px 0;border-radius:12px}}code{{background:#edf2f7;padding:2px 5px}}</style>
<h1>V6 · карта исхода и пути до crossing</h1>
<p>20 сессий · {stats['roots']:,} origins · {stats['cells']:,} ячеек · {stats['hits']:,} наблюдаемых crossing · {stats['unique_events']:,} уникальных event ID.</p>
<article><b>Это проверка семантики нового таргета, без обучения модели.</b> BB M1/M3/M6, 3 линии, H=300 s. Результат LONG/SHORT после spread и fee, время, риск, площадь под equity и форма пути остаются отдельными величинами. Все будущие поля запрещены в X. VWAP не подменён приближением. Условные профили отсутствующего crossing замаскированы.</article>
<article><b>Главная проверка:</b> менее секунды занимают {stats['subsecond_fraction']:.2%} пересечений, но они дают {stats['subsecond_share_squared_velocity']:.6%} суммы квадратов сырой скорости. Поэтому скорость сохранена как диагностика, не включена в основной regression target. Среди прибыльных LONG-crossing {stats['positive_long_with_prior_risk_fraction']:.1%} уже проходили риск −1R до события. Это условные mark-to-market пути, не результат исполнения выхода с задержкой.</article>
<article><b>Форма:</b> точные интегралы считаются по всем публикациям. 17 точек и signed Haar — сжатое описание будущего пути, не замена extrema: в {stats['sampled_17point_mae_underestimate_over_half_R']:.1%} случаев 17 точек недооценивают MAE более чем на 0.5R. Haar выражен в долях длительности события, это не STFT и не частота рынка в герцах. Циклический UTC-clock сохранён отдельно как причинный вход; влияние на модель ещё не проверено.</article>
<article><b>Чтение графиков:</b> распределения условны на hit, ячейки зависимы через event ID. Пара примеров выбрана постфактум по заранее описанному правилу контраста, это иллюстрация, не оценка доходности. Карта и окно градиента выбраны фиксированным временем 13:30 UTC. Reference gradients рассчитаны при постоянном прогнозе, не получены обучением.</article>
'''
    captions=[
        '01. Распределения и покрытие. Простое наступление crossing существенно отличается от прибыльной возможности.',
        '02. Деление на время. Короткие интервалы усиливают в том числе отрицательную стоимость входа.',
        '03. Сходный результат и время: полная equity показывает различную просадку и время под водой. Signed Haar описывает форму, но не заменяет точный риск.',
        '04. Один фиксированный origin: все девять линий. Серый означает отсутствие наблюдаемого пересечения за горизонт, не нулевую доходность.',
        '05. Движение цены и линии разделены. Cause относится к последнему strict anchor, entry-to-cross delta — к полному пути.',
        '06. Таргет не становится автоматически плавным: переключение идентичности будущего crossing сохраняет скачки. Градиенты только диагностические.',
        '07. Дополнительная пара с близкими исходом, временем и MAE. Площадь различает длительное пребывание в убытке и короткую просадку. Скорости четвертей — изменение equity, стоимость входа остаётся отдельно в net.'
    ]
    for name, caption in zip(IMAGES,captions): text+=f'<article><p>{caption}</p><img src="{name}" alt="{caption}"></article>'
    text+='<p><a href="summary.json">Измерения</a> · <a href="visual_cases.json">Точные идентификаторы примеров</a> · <a href="data_receipt.json">Источники и SHA256</a> · <a href="sol61_visual_audit.json">Независимый визуальный аудит Sol 6.1</a> · <a href="completion_receipt.json">Финальная проверка</a> · <a href="../../contracts/crossing_path_target_v6.json">Контракт таргета</a></p></html>'
    (OUT/'index.html').write_text(text,encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
    QUOTE_CACHE.clear()


if __name__=='__main__': main()
