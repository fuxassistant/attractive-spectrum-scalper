"""Bound visual and numerical report for the fixed flow/context ablation."""
from pathlib import Path
import html
import json
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectrum.cognition import flow_context as fc, episodes as ep
from spectrum.cognition.flow_context_lab import DATA, REPORT, MARKET, compare
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from scripts.flow_context_coverage import save, sha

COLORS = dict(baseline='#3465a4', flow='#148a78', flow_episode='#be941e', rich_flow='#5e43a6', rich_day='#cf6e28', rich_spectrum='#bf3e75', shuffled='#999999')
MAIN = list(COLORS)[:-1]
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18})


def figure_save(fig, name):
    fig.canvas.draw(); fig.savefig(REPORT / name, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)


def coverage_plot():
    cov = json.loads((REPORT / 'coverage_audit.json').read_text()); data = json.loads((REPORT / 'data_receipt.json').read_text())
    rows = [d for d in cov['days'] if not d.get('excluded')]; dates = pd.to_datetime([r['date'] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    fig.suptitle('Coverage audit: February has source data; the <=2 tick gate discarded 99.6% of eligible roots\nV2 keeps every positive supported spread; wide-spread results remain separate', fontsize=13)
    for name, label, color in [('uncapped_roots', 'V2 uncapped supported', '#148a78'), ('legacy_roots', 'V1 <=2 ticks', '#3465a4')]:
        ax[0, 0].plot(dates, [r[name] for r in rows], label=label, color=color)
    ax[0, 0].set(title='Causal 30-second roots before L/S and mode duplication', ylabel='Roots / source day'); ax[0, 0].legend()
    ax[0, 1].plot(dates, [r['spread_quantiles']['0.5'] for r in rows], label='median', color='#3465a4')
    ax[0, 1].plot(dates, [r['spread_quantiles']['0.9'] for r in rows], label='p90', color='#cf6e28')
    ax[0, 1].axhline(2, color='#999999', ls='--'); ax[0, 1].set(ylabel='Spread (ticks)', title='Observed BBO spread; 16,541 calendar probes matched'); ax[0, 1].legend()
    months = list(cov['monthly']); bottom = np.zeros(len(months))
    for bucket, color in zip(('0-2', '2-4', '4-8', '8+'), ('#3465a4', '#148a78', '#cf6e28', '#bf3e75')):
        v = np.array([cov['monthly'][m]['spread_bins'][bucket] / cov['monthly'][m]['uncapped_roots'] for m in months])
        ax[1, 0].bar(months, v, bottom=bottom, color=color, label=bucket); bottom += v
    ax[1, 0].set(title='Distribution shift remains after recovering coverage', ylabel='Fraction of eligible roots'); ax[1, 0].legend(ncol=4)
    groups = ['previous_known_roots', 'day_vwap', 'stft_price_b0', 'stft_volume_b0']; matrix = []
    for key in groups:
        a = []
        for month in months:
            ds = [r for r in data['daily'] if r['date'][:7] == month]; n = sum(r['roots'] for r in ds)
            a.append(sum((r[key] if key == 'previous_known_roots' else r['known_fraction'][key] * r['roots']) for r in ds) / n)
        matrix.append(a)
    ax[1, 1].imshow(matrix, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    ax[1, 1].set(xticks=range(5), xticklabels=months, yticks=range(4), yticklabels=['previous weekday context', 'current London VWAP proxy', '128-min price spectrum', '128-min volume spectrum'], title='Actually available to V2 at the decision')
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax[1, 1].text(j, i, f'{v:.1%}', ha='center', va='center', color='white' if v > .7 else 'black')
    for a in ax.flat:
        a.tick_params(axis='x', rotation=20)
    figure_save(fig, '01_coverage.png')


def quality_plot(s):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
    fig.suptitle('Separate decision-time fill and conditional path heads | B: Jan-Feb 2026 development data\nSame roots, uniform weights; conditional path evaluated only on observed fills', fontsize=13)
    b = s['stages']['B']['P']
    for ax, key, title in ((axes[0, 0], 'fill', 'P fill/EMPTY binary log-loss'), (axes[0, 1], 'path', 'P TAKE/STOP/TIME given fill log-loss')):
        values = [b[f]['all']['fill']['log_loss'] if key == 'fill' else b[f]['all']['path_log_loss'] for f in MAIN]
        ax.plot(range(len(MAIN)), values, 'o-', color='#3465a4')
        ax.set(xticks=range(len(MAIN)), xticklabels=MAIN, title=title, ylabel='Lower is better')
        ax.tick_params(axis='x', rotation=35)
    pairs = fc.contract()['primary_comparisons']
    for ax, part in ((axes[1, 0], 'fill'), (axes[1, 1], 'path')):
        for i, pair in enumerate(pairs):
            r = s['comparisons']['B/P/' + pair][part]; lo, hi = r['daily_ci']; color = COLORS[pair.split('-')[0]]
            ax.plot([lo, hi], [i, i], color=color, lw=3); ax.plot(r['daily_mean'], i, 'o', color=color)
        ax.axvline(0, color='black', ls='--'); ax.set(yticks=range(len(pairs)), yticklabels=pairs, xlabel='Daily log-score gain, positive = better', title=part + ': paired 95% 5-day-block intervals (unadjusted)')
    for mode, ax in (('P', axes[0, 2]), ('T', axes[1, 2])):
        for family in ('baseline', 'flow', 'rich_flow', 'rich_day', 'rich_spectrum'):
            rr = s['stages']['B'][mode][family]
            ax.plot(['Jan', 'Feb'], [rr['months'][m]['path_log_loss'] for m in ('2026-01', '2026-02')], 'o-', label=family, color=COLORS[family])
        base = s['stages']['B'][mode]['baseline']['months']
        ax.set_xticks([0, 1], [f"{m}\n{base[k]['n_filled']:,} fills" for m, k in (('Jan', '2026-01'), ('Feb', '2026-02'))])
        ax.set(title=mode + ' conditional path by month', ylabel='Log-loss (base rates differ by month)'); ax.legend(fontsize=7)
    figure_save(fig, '02_separate_heads.png')


def model_usage():
    result = {}; groups = [ep.BASE, fc.FLOW, fc.RICH, fc.DAY, fc.SPECTRUM]; group_names = ['baseline6', 'short flow16', 'extra flow15', 'day context21', 'spectrum8']
    matrix = []
    for head in ('fill', 'path'):
        for family in MAIN:
            model = pickle.loads((DATA / f'model_B_P_{family}.pkl').read_bytes()); columns = model['columns']; obj = model[head]; counts = Counter()
            if not isinstance(obj, int):
                for iteration in obj._predictors:
                    for predictor in iteration:
                        for row in predictor.nodes:
                            if not row['is_leaf']:
                                counts[columns[int(row['feature_idx'])]] += 1
            result[head + '/' + family] = dict(offered=len(columns), used=len(counts), split_counts=dict(counts),
                 semantics='Used in at least one saved tree split; frequency is not predictive value or causal importance')
            if family == 'rich_spectrum':
                matrix.append([sum(c in counts for c in g) for g in groups])
    save(REPORT / 'model_feature_usage.json', result)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    fig.suptitle('What the saved P models actually read | tree usage is not proof of useful information', fontsize=13)
    for offset, head, color in ((-.18, 'fill', '#148a78'), (.18, 'path', '#bf3e75')):
        axes[0].bar(np.arange(len(MAIN)) + offset, [result[head + '/' + f]['used'] for f in MAIN], width=.36, label=head, color=color)
    axes[0].plot(range(len(MAIN)), [len(fc.FAMILIES[f]) for f in MAIN], 'k--', label='offered')
    axes[0].set(xticks=range(len(MAIN)), xticklabels=MAIN, ylabel='Distinct features used by saved splits'); axes[0].legend(); axes[0].tick_params(axis='x', rotation=25)
    axes[1].imshow(matrix, aspect='auto', cmap='YlGnBu'); axes[1].set(xticks=range(5), xticklabels=group_names, yticks=[0, 1], yticklabels=['fill head', 'conditional path head'], title='Rich_spectrum: used / offered in each group')
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            axes[1].text(j, i, f'{v}/{len(groups[j])}', ha='center', va='center', color='white' if v > 10 else 'black')
    axes[1].tick_params(axis='x', rotation=20)
    figure_save(fig, '03_feature_usage.png')
    return result


def calibration_economics(s):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    fig.suptitle('Calibration and money are separate checks | all results are exposed development evidence', fontsize=13)
    for family in ('baseline', 'flow', 'rich_flow', 'rich_spectrum'):
        f = pd.read_parquet(DATA / f'predictions_B_P_{family}.parquet'); f = f.loc[f.category >= 0]
        for a, p, y in ((ax[0, 0], f.p_fill.to_numpy(), (f.category < 3).to_numpy()), (ax[0, 1], f.loc[f.category < 3, 'c_TAKE'].to_numpy(), (f.loc[f.category < 3, 'category'] == 0).to_numpy())):
            bins = np.minimum((p*10).astype(int), 9); xs, ys = [], []
            for i in range(10):
                mask = bins == i
                if mask.any():
                    xs.append(p[mask].mean()); ys.append(y[mask].mean())
                    if family == 'rich_spectrum':
                        a.annotate(str(mask.sum()), (xs[-1], ys[-1]), xytext=(3, 4), textcoords='offset points', fontsize=7)
            a.plot(xs, ys, 'o-', color=COLORS[family], label=family)
        p = s['stages']['B']['P'][family]['policy']; d = pd.Series(p['daily']).sort_index()
        ax[1, 0].plot(pd.to_datetime(d.index), d.cumsum(), color=COLORS[family], label=f"{family}, {p['attempts']} attempts")
    for a, title in ((ax[0, 0], 'P(fill) on known attempts'), (ax[0, 1], 'P(TAKE | fill) on known fills')):
        a.plot([0, 1], [0, 1], 'k--'); a.set(xlabel='Predicted', ylabel='Observed', xlim=(0, 1), ylim=(0, 1), title=title + '; bin counts, no bin CI'); a.legend(fontsize=7)
    ax[1, 0].set(title='Policy E[net]>0; check A2 admission separately', ylabel='Realized net ($ / 1 oz)'); ax[1, 0].legend(fontsize=7); ax[1, 0].tick_params(axis='x', rotation=20)
    rows = []
    for family in MAIN:
        p = s['stages']['B']['P'][family]['forced_top10']
        cutoff = p['censor']['date'][5:] if p['censor'] else 'complete'
        rows.append([family, str(p['attempts']), cutoff, f"{p['observed_prefix_net']:+.1f}", 'unknown' if p['net'] is None else f"{p['net']:+.1f}"])
    ax[1, 1].axis('off')
    table = ax[1, 1].table(cellText=rows, colLabels=['Model', 'Attempts', 'Censor date', 'Prefix $', 'Full B $'],
                          colWidths=[.27, .16, .21, .17, .19], cellLoc='center', bbox=[0, .22, 1, .65])
    table.auto_set_font_size(False); table.set_fontsize(8)
    for i, family in enumerate(MAIN, 1):
        table[(i, 0)].get_text().set_color(COLORS[family])
    ax[1, 1].set_title('Forced A2 top10% diagnostic: each account is incomplete')
    ax[1, 1].text(.5, .08, 'Different censor dates in 2026; no full-period net ranking.\nObserved prefixes are not comparable B-period returns.',
                  transform=ax[1, 1].transAxes, ha='center', va='center', fontsize=9)
    figure_save(fig, '04_calibration_economics.png')


def context_scene():
    man = json.loads((DATA / 'manifest.json').read_text()); chosen = None
    for r in man['days']:
        date = r['date']; prev = r['previous_date']
        if not date.startswith('2026-02') or (pd.Timestamp(date)-pd.Timestamp(prev)).days != 1 or r['previous_known_roots'] == 0:
            continue
        f = pd.read_parquet(DATA / (date+'.parquet')); f = f.loc[(f['mode']=='P') & (f.side==1) & (f.minutes_since_london>=180) & (f.previous_known>.5)]
        if len(f): chosen = f.iloc[0]; break
    if chosen is None: raise ValueError('No supported two-day context scene')
    date = chosen.date; intent = int(chosen.t_ns); previous_date = str((pd.Timestamp(date)-pd.offsets.BDay()).date())
    current = fc.minute_tape(MonthDay(MARKET/date[:7]/date)); previous = fc.minute_tape(MonthDay(MARKET/previous_date[:7]/previous_date))
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.subplots_adjust(left=.08, right=.97, bottom=.08, top=.90, hspace=.25)
    fig.suptitle(f'{date} context at {pd.Timestamp(intent,unit="ns",tz="UTC")}\nPublished past only: previous London + current session; scalar summaries offered, full trace not fed to HGB', fontsize=13)
    for tape, label, color in ((previous, 'previous session from London', '#3465a4'), (current, 'current session prefix', '#148a78')):
        mask = tape['available'] <= intent
        if tape is previous: mask &= tape['t'] > tape['london']
        tt = pd.to_datetime(tape['t'][mask], utc=True); mid = tape['mid'][mask]*.1; vol = tape['volume'][mask]
        valid = tape['valid'][mask]
        weighted = np.cumsum(np.where(valid, mid*vol, 0)) / np.maximum(np.cumsum(np.where(valid, vol, 0)), 1)
        # Retain invalid endpoints and insert NaN rows for missing minutes, so
        # the audit trace never visually bridges unsupported data.
        frame = pd.DataFrame(dict(mid=np.where(valid, mid, np.nan),
                                  vwap=np.where(valid, weighted, np.nan),
                                  volume=np.where(valid, np.log1p(vol), np.nan)), index=tt)
        frame = frame.reindex(pd.date_range(tt.min(), tt.max(), freq='min'))
        axes[0].plot(frame.index, frame.mid, color=color, lw=.8, label=label)
        axes[0].plot(frame.index, frame.vwap, color=color, ls='--', lw=1.1, label=label+' VWAP proxy')
        axes[1].plot(frame.index, frame.volume, color=color, lw=.6, label=label)
    sp = fc.trailing_spectra(current); mask = current['available'] <= intent
    for j in range(3):
        axes[2].plot(pd.to_datetime(current['t'][mask],utc=True), sp[mask,j], label=f'price power band {j}')
    for ax in axes:
        for tape, label in ((previous,'previous London'),(current,'current London')):
            ax.axvline(pd.Timestamp(tape['london'],unit='ns',tz='UTC'),color='gray',ls=':',lw=.8)
        ax.axvline(pd.Timestamp(intent,unit='ns',tz='UTC'),color='black',lw=1)
        ax.legend(fontsize=7,loc='upper left')
    axes[0].set_ylabel('Minute-end mid ($/oz)'); axes[1].set_ylabel('log(1 + minute volume)'); axes[2].set_ylabel('128-min spectral fraction'); axes[2].set_xlabel('UTC; invalid/missing minutes and exchange session gaps are not interpolated')
    figure_save(fig,'05_day_context.png')
    save(REPORT/'context_scene.json',dict(date=date,t_ns=str(intent),previous_date=previous_date,selection='first February date with immediately previous calendar day supported, first root >=3h after London',
        scalar_features={k:(None if not np.isfinite(chosen[k]) else float(chosen[k])) for k in fc.DAY+fc.SPECTRUM},claim='The trace is an audit visualization. Only the listed scalar summaries enter HGB.'))


def execution_scenes():
    selection=json.loads((REPORT/'scene_selection.json').read_text()); facts=json.loads((DATA/'audit_execution_facts.json').read_text())
    lookup={(r['date'],r['t_ns'],r['mode'],r['side']):r for r in facts}
    preds={f:pd.read_parquet(DATA/f'predictions_B_P_{f}.parquet') for f in ('baseline','flow','rich_spectrum')}; files=[]
    for number,s in enumerate(selection['scenes']):
        date,side,t=s['date'],s['side'],int(s['t_ns']); fact=lookup[(date,s['t_ns'],'P',side)]
        q=QuoteTape.from_day(MonthDay(MARKET/date[:7]/date)); fill=int(fact['entry_ns']) if fact['entry_ns'] else None
        end_ns=int(fact['exit_ns']) if fact['exit_ns'] else t+330*ep.NS; end=max(90,min(390,(end_ns-t)/ep.NS+25))
        mask=(q.quote_t>=t-90*ep.NS)&(q.quote_t<=t+end*ep.NS); ix=np.flatnonzero(mask); xx=(q.quote_t[ix]-t)/ep.NS
        cat=ep.CLASSES[fact['category']] if fact['category']>=0 else 'CENSORED'
        fig,ax=plt.subplots(4,1,figsize=(14,12),sharex=True,gridspec_kw={'height_ratios':[2,1,1,1.2]});fig.subplots_adjust(left=.10,right=.97,bottom=.07,top=.91,hspace=.35)
        fig.suptitle(f"{date} {pd.Timestamp(t,unit='ns',tz='UTC').strftime('%H:%M:%S UTC')} | {'LONG' if side==1 else 'SHORT'} | {cat} | {s['selection']}\nDiagnostic candidate; future shown for replay audit; not an admitted policy trade",fontsize=12)
        for price,color,label in ((q.bid,'#147d92','bid'),(q.ask,'#d79535','ask')):
            ax[0].plot(xx,np.where(q.valid[ix],price[ix]/1e9,np.nan),color=color,lw=.8,drawstyle='steps-post',label=label)
        at=np.searchsorted(q.quote_t,t,side='right')-1; limit=(q.bid[at] if side==1 else q.ask[at])/1e9
        ax[0].plot([.1,(fill-t)/ep.NS if fill else 30.1],[limit]*2,':',color='gray',label='submitted limit')
        if fill is not None:
            px=int(fact['entry_px_nano'])/1e9; xf=(fill-t)/ep.NS; xe=(end_ns-t)/ep.NS
            ax[0].plot(xf,px,'ko',ms=5,label='queue fill'); ax[0].plot([xf,xe],[px+side*2.4]*2,'--',color='#148a78',label='TP');ax[0].plot([xf,xe],[px-side*1.2]*2,'--',color='#bf3e75',label='SL')
            held=mask&(q.quote_t>=fill)&(q.quote_t<=end_ns)&q.valid
            if fact['exit_px_nano'] is not None:
                exit_px = int(fact['exit_px_nano'])/1e9
                ax[0].plot(xe,exit_px,'X',color='#bf3e75',mec='white',mew=.8,ms=9,zorder=8,label='taker exit')
                if cat == 'TAKE':
                    overshoot = side*(exit_px-px)-2.4
                    ax[0].annotate(f'Taker TP exit\nquote overshoot {overshoot:+.2f}', (xe,exit_px), xytext=(12,18),
                                   textcoords='offset points', fontsize=8, bbox=dict(facecolor='white',alpha=.9),
                                   arrowprops=dict(arrowstyle='->', color='#bf3e75'))
            if cat=='CENSORED':
                horizon=fill+305*ep.NS; marker=min(end_ns,horizon); held&=q.quote_t<marker
                reason='No exit quote inside 300s + 5s grace' if end_ns>=horizon else 'Quote validity/epoch boundary'
                for a in ax[:2]:a.axvline((marker-t)/ep.NS,color='#bf3e75',ls=':')
                ax[0].text(.98,.04,reason,transform=ax[0].transAxes,ha='right',fontsize=8,bbox=dict(facecolor='white',alpha=.9))
            pnl=side*((q.bid[held] if side==1 else q.ask[held])/1e9-px)-.3
            ax[1].plot((q.quote_t[held]-t)/ep.NS,pnl,color='#148a78',drawstyle='steps-post')
            ax[1].set_title(f"Realized net {fact['net']:+.3f} USD / 1 oz" if fact['net'] is not None else 'Outcome unknown; no final net assigned')
            if fact['exit_px_nano'] is not None and 0<=xe-xf<1:
                ins=ax[0].inset_axes([.65,.56,.32,.36]); margin=max((end_ns-fill)*.6,1e6); a=max(0,np.searchsorted(q.quote_t,fill-margin)-1); b=np.searchsorted(q.quote_t,end_ns+margin,side='right');tt=(q.quote_t[a:b]-fill)/1e6
                ins.plot(tt,q.bid[a:b]/1e9,color='#147d92',drawstyle='steps-post');ins.plot(tt,q.ask[a:b]/1e9,color='#d79535',drawstyle='steps-post');dur=(end_ns-fill)/1e6
                ins.plot(0,px,'ko',ms=3);ins.plot(dur,int(fact['exit_px_nano'])/1e9,'kx');ins.axhline(px-side*1.2,color='#bf3e75',ls='--',lw=.8);ins.set_xlim(-margin/1e6,dur+margin/1e6);ins.set_title(f'Hold {dur:.3f} ms',fontsize=8);ins.set_xlabel('ms from fill',fontsize=7);ins.tick_params(labelsize=6)
        else:ax[1].text(.5,.5,'No certified fill; '+('EMPTY' if cat=='EMPTY' else 'outcome unknown'),ha='center',transform=ax[1].transAxes)
        raw=pd.read_parquet(Path('data/derived/scalper_v1')/(date+'.parquet'),columns=['t_ns','flow_pressure30_low','flow_pressure30_high','flow_pressure30_known','flow_book30_pressure','flow_book30_known']);raw=raw.loc[(raw.t_ns>=t-90*ep.NS)&(raw.t_ns<=t+end*ep.NS)];rx=(raw.t_ns.to_numpy()-t)/ep.NS
        ax[2].fill_between(rx,np.where(raw.flow_pressure30_known,raw.flow_pressure30_low,np.nan),np.where(raw.flow_pressure30_known,raw.flow_pressure30_high,np.nan),alpha=.3,label='trade pressure bounds incl unknown')
        ax[2].plot(rx,np.where(raw.flow_book30_known,raw.flow_book30_pressure,np.nan),color='#5e43a6',label='book pressure');ax[2].set_ylim(-1.05,1.05)
        for family,p in preds.items():
            p=p.loc[(p.date==date)&(p.side==side)&(p.t_ns>=t-90*ep.NS)&(p.t_ns<=t+end*ep.NS)]
            ax[3].scatter((p.t_ns.to_numpy()-t)/ep.NS,p.c_TAKE,s=14,color=COLORS[family],label=family+' P(TAKE|fill)')
            if family=='rich_spectrum':
                ax[3].scatter((p.t_ns.to_numpy()-t)/ep.NS,p.p_fill,s=18,marker='x',color='black',label='rich P(fill)')
                now=p.loc[p.t_ns==t].iloc[0];ax[3].set_title(f"At decision: P(fill)={now.p_fill:.3f}, P(TAKE|fill)={now.c_TAKE:.3f}\nE[net]={now.expected_net:+.3f}; spread={now.spread_ticks:g} ticks", fontsize=9)
        for a in ax:a.axvline(0,color='black',lw=1);a.axvspan(0,end,color='#eeeeee',alpha=.35,zorder=-5);a.set_xlim(-90,end)
        ax[0].set_ylabel('GC ($/oz)');ax[1].set_ylabel('Liquidation net');ax[2].set_ylabel('Buy-positive pressure');ax[3].set(xlabel='Seconds from decision; forecasts are separate points with their own past',ylabel='Probability',ylim=(0,1))
        for a in (ax[0],ax[2],ax[3]):a.legend(fontsize=7,loc='upper left',ncol=3)
        filename=f'scene_{number+1:02d}_{cat.lower()}.png';figure_save(fig,filename);files.append(filename);QUOTE_CACHE.clear()
    return files


def write_report(s,usage,scenes):
    lines=['# Flow context v2: проверенный результат','', 'Протокол и код зафиксированы до fit: 55f8b0e. Все периоды ранее исследованы; development evidence, не blind OOS. 106 дней, 173724 roots, 694896 строк P/T × L/S. Две головы получают X в момент решения; новые сведения после fill им не передаются.','',
      '**Итог: поток без эпизода улучшает прогноз исполнения, но не подтверждает преимущество по исходу после входа над baseline.** Расширенный короткий поток восстанавливает часть потери conditional качества варианта flow, оставаясь практически на уровне baseline. Добавление дневных сводок ухудшило conditional результат; восемь спектральных сводок добавочной пользы не показали. Все 14 вариантов P/T остаются без допуска A2, при E[net]>0 все дают WAIT. Это отрицательный результат заданной абляции, не опровержение полезности полного контекста дня.','',
      '## Покрытие февраля','', 'Исходные данные присутствуют. Спред <=2 тика оставлял 135 из 32870 допустимых февральских roots. Медиана дневных медиан спреда выросла с 3 до 7 тиков. Все 108 source hashes проверены; 16541 roots десяти календарных дней точно совпали с BBO, а при build совпали спреды всех 173724 выбранных roots. Gate v2 не имеет верхней границы; торговый план 24/12/300 и затраты не изменены. Широкий спред не объявлен торговой возможностью.','',
      '## Раздельные головы, B, пассивный вход','', '| Семейство | Полей | Fill log-loss ↓ | Path given fill log-loss ↓ | AUC TAKE given fill | B попытки E[net]>0 | Допуск A2 |','|---|---:|---:|---:|---:|---:|---|']
    for family in MAIN:
        r=s['stages']['B']['P'][family];m=r['all'];lines.append(f"| {family} | {len(fc.FAMILIES[family])} | {m['fill']['log_loss']:.6f} | {m['path_log_loss']:.6f} | {m['take_given_fill']['auc']:.4f} | {r['policy']['attempts']} | {'да' if r['operational_admitted'] else 'нет, WAIT'} |")
    lines+=['', 'P: пассивный вход с доказательством очереди V5; T: вторичный контроль с тейкерским входом. На B для P известны 135129 side-attempts и 101665 fills; в феврале 65721 и 46740 соответственно. Две стороны одного root и перекрывающиеся будущие пути зависимы.','',
      '## Вторичный контроль: тейкерский вход','', '| Семейство T | Fill log-loss ↓ | Path given fill log-loss ↓ | B попытки E[net]>0 |','|---|---:|---:|---:|']
    for family in MAIN:
        r=s['stages']['B']['T'][family];m=r['all'];lines.append(f"| {family} | {m['fill']['log_loss']:.6f} | {m['path_log_loss']:.6f} | {r['policy']['attempts']} |")
    lines+=['', 'T также не дал улучшения conditional path над baseline или открытий. Возможность EMPTY у T обусловлена ограничением ожидания поддержанной исполнимой котировки, поэтому fill не подменён константой 1.','']
    lines+=['','## Заранее заданные парные сравнения','', 'Ниже 95% moving-block интервалы **средней дневной суммы log-score gain**, 5-дневные блоки, 2000 resamples. Положительно — лучше родителя; это не интервал ошибки на одной попытке. Интервалы не скорректированы на множество сравнений и не являются blind подтверждением.','', '| Сравнение P на B | Fill daily gain CI | Conditional path daily gain CI |','|---|---|---|']
    for pair in fc.contract()['primary_comparisons']:
        r=s['comparisons']['B/P/'+pair];lines.append(f"| {pair} | {[round(v,3) for v in r['fill']['daily_ci']]} | {[round(v,3) for v in r['path']['daily_ci']]} |")
    lines+=['','## Экономика и цензура','', '| Модель P | Top10% A2: попытки B | Fills | Полный net B | Наблюдённый prefix net | Первая цензура |','|---|---:|---:|---:|---:|---|']
    for family in MAIN:
        p=s['stages']['B']['P'][family]['forced_top10'];lines.append(f"| {family} | {p['attempts']} | {p['fills']} | {'unknown' if p['net'] is None else p['net']} | {p['observed_prefix_net']:.3f} | {p['censor']['date'] if p['censor'] else 'нет'} |")
    lines+=['', 'Top10% — зафиксированный на A2 диагностический порог, не выбранная по B политика. После первой выбранной цензуры полный net и интервалы неизвестны. Префиксы заканчиваются в разные даты: их нельзя считать итогом всего B или ранжировать по ним модели. Калибровка на рисунке относится к общей известной популяции, а не отдельно к выбранному хвосту. Rich_spectrum прогнозирует TAKE given fill в среднем 23.50%, наблюдается 20.76%.','',
      '## Что модель видит','', 'Flow22 не содержит episode/reaction-onset и не использует эпизодные веса. Rich_flow37 расширяет объёмы/давление/книгу до 300 секунд. Rich_day58 получает 21 координату наблюдаемого сегодняшнего/предыдущего дня. Rich_spectrum66 получает ещё восемь долей спектральной мощности/энтропии по 128 минутам. Координаты ориентированы относительно предлагаемой стороны. HGB остаётся моделью скалярного состояния: сами последовательности секунд/M1, order ID, полный спектр, фазу и когерентность он не получает.','',
      'Вчерашняя часть — предыдущий weekday, тот же контракт, >=95% покрытия; отсутствующие дни не перепрыгиваются. Combined VWAP proxy включает предыдущий London→закрытие плюс текущую сессию до решения, включая ночь. Отдельный today anchor начинается с первой наблюдаемой завершённой M1 после London, не с точной opening quote. Это объёмно-взвешенные midpoint-прокси, не точный trade-price VWAP. STFT требует129 endpoints/128 приращений, одинаковый quote segment и отсутствие внутренних invalid/epoch переходов.','',
      '| Богатая модель P | Подано | Использовано хотя бы в одном split |','|---|---:|---:|']
    for head in ('fill','path'):
        u=usage[head+'/rich_spectrum'];lines.append(f"| {head} | {u['offered']} | {u['used']} |")
    lines+=['', 'У rich_spectrum conditional-голова использовала 16 из 21 дневного поля и все 8 спектральных. Она получает и использует эти координаты, но полезного ориентирования в контексте опыт не подтвердил. В феврале предыдущий день доступен на 89.86% roots, STFT — на 92.78%, текущий day VWAP proxy — на 99.88%. Missingness остаётся явной и не заполняется будущим.','',
      'Split counts подтверждают, что признаки прочитаны деревьями; они не доказывают пользу, направление причинности или экономическую значимость. Добавочная польза проверяется отдельными вложенными абляциями. Счёт 66 не является процентом 182 фич: определения/маски/производные различаются. Подробный независимый inventory — FEATURE_CONTEXT_AUDIT_RU.md.','',
      '## Курс следующего исследования','',
      '1. Сохранить отдельные задачи исполнения, условного пути после fill и выбора действия. Не возвращать их к одной регрессии точного исхода или net. Проверять улучшение каждой головы и итог after-cost отдельно.','2. Проверить фактическую смешанную память: подробный поток/книга/объёмы на коротком горизонте плюс причинная M1-последовательность от предыдущего London до сейчас. Сравнить её с теми же скалярными baseline; сам график длинного пути ещё не означает, что learner его видел. Ночь, закрытия, roll и отсутствующие минуты требуют явных масок.','3. На длинном масштабе проверить геометрию и динамику VWAP, последовательность пересечений и сжатия/расширения, спектральную мощность и переходы режимов. Нынешние 8 STFT summaries не проверяют фазу, когерентность или весь спектр. Добавлять каналы отдельными абляциями, не объявлять все доступные фичи полезными заранее.','4. Для управления после входа нужен отдельный опыт: новые X после fill, совпадающие сохранённые состояния и альтернативы HOLD/CLOSE/REDUCE, с раздельными записями ожидания/цены исполнения. Настоящий v2 оценивает исход после fill из X до входа и этого опыта не заменяет.','5. До экономического вывода провести полный положительный контроль через replay, сохраняющий волатильность null и проверку на неиспользованном периоде. Все текущие A1/A2/B уже исследованы.','',
      '## Проверки и визуальный аудит','', '315 тестов прошли после добавления v2. Source/feature identities заморожены; future mutation, publication clock, exact previous weekday, quote epoch, неизвестные targets и factorization проверены тестами. Независимый свежий MBO replay и quote-scan выходов записаны в execution_audit.json. Сцены выбираются календарно/по первым категориям и спредам, не по лучшему net. Это диагностические кандидаты; торговая политика их не обязательно выбирает.','',
      'В независимом execution audit 824 сравнения, 0 расхождений на пяти календарных датах. TP исполняется тейкером по котировке, поэтому возможен overshoot относительно номинального барьера. Sol 6.1 отдельно сверил splits сохранённых P/B моделей и воспроизвёл прогнозы богатой модели на 3334 строках первого B-дня с максимальным расхождением 0.','',
      'Визуальный независимый вердикт Sol 6.1 — SOL61_VISUAL_AUDIT_RU.md. Ограничения: фиксированный план, исходные исследовательские единицы1oz/$0.30, development-only периоды, сжатый контекст, отсутствие полного planted-positive replay и нового volatility-preserving null. Полученные результаты не доказывают бесполезность полной дневной последовательности, которую модель не получала.','']
    (REPORT/'RESULTS_RU.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    images=['01_coverage.png','02_separate_heads.png','03_feature_usage.png','04_calibration_economics.png','05_day_context.png']+scenes
    body=''.join(f'<figure><img src="{html.escape(f)}" alt="{html.escape(f)}"><figcaption>{html.escape(f)}</figcaption></figure>' for f in images)
    page='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Flow context v2</title><style>body{max-width:1500px;margin:32px auto;padding:0 24px;font:17px system-ui;background:#f3f5f7;color:#17232b}figure{background:white;margin:24px 0;padding:12px;border-radius:12px}img{width:100%;height:auto}figcaption{padding:8px;color:#52616d}a{color:#146b86}</style><h1>Поток, исполнение и дневной контекст — v2</h1><p>106 дней. Покрытие февраля восстановлено новым gate. Две отдельные головы; development evidence.</p><p><a href="RESULTS_RU.md">Результаты</a> · <a href="FEATURE_CONTEXT_AUDIT_RU.md">Какие признаки реально доступны</a> · <a href="SOL61_VISUAL_AUDIT_RU.md">Sol 6.1: визуальный аудит</a></p>'+body+'</html>'
    (REPORT/'index.html').write_text(page,encoding='utf-8',newline='\n')
    save(REPORT/'visual_receipt.json',dict(images={f:sha(REPORT/f) for f in images},summary_sha256=sha(REPORT/'summary.json'),scene_selection_sha256=sha(REPORT/'scene_selection.json'),script_sha256=sha(__file__)))


if __name__=='__main__':
    summary=json.loads((REPORT/'summary.json').read_text());coverage_plot();quality_plot(summary);usage=model_usage();calibration_economics(summary);context_scene();scene_files=execution_scenes();write_report(summary,usage,scene_files)
