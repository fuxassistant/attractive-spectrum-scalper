"""Fixed-calendar crossing case; no fitted selection and no invented tick path."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path('data/derived/scientific_pipeline_v2/dataset')
OUT = Path('reports/scientific_pipeline_v2')
DATE = '2026-05-11'
NS = 1_000_000_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    day_receipt_path = ROOT/(DATE+'.json')
    day_receipt = json.loads(day_receipt_path.read_text(encoding='utf-8'))
    if day_receipt['status'] != 'COMPLETE':
        raise ValueError('Complete dataset day required')
    sources = {str(day_receipt_path): sha(day_receipt_path)}
    frames = {}
    for name in ('roots', 'map', 'bank', 'scene', 'execution'):
        path = ROOT/(DATE+'_'+name+'.parquet')
        actual = sha(path)
        if actual != day_receipt['files'][path.name]:
            raise ValueError('Changed bound case source '+name)
        sources[str(path)] = actual
        frames[name] = pd.read_parquet(path)
    anchor = int(frames['roots'].t_ns.min())+3600*NS
    roots = frames['roots'].loc[frames['roots'].t_ns == anchor]
    if len(roots) != 1:
        raise ValueError('Exact preregistered calendar anchor missing')
    root = roots.iloc[0]
    cells = frames['map'].loc[frames['map'].t_ns == anchor].sort_values(['scale', 'role']).reset_index(drop=True)
    trades = frames['execution'].loc[frames['execution'].t_ns == anchor].copy()
    trades['plan_order'] = trades.plan.map({'cost2': 0, 'cost4': 1, 'cost6': 2})
    trades = trades.sort_values(['plan_order', 'side'], ascending=[True, False]).reset_index(drop=True)
    if len(cells) != 9 or len(trades) != 6:
        raise ValueError('Nine line cells and six execution alternatives required')
    bank = frames['bank'].set_index('event_id')
    resolved = cells.cross_ordinal >= 0
    mapped = bank.reindex(cells.loc[resolved, 'event_id'])
    if not np.array_equal(mapped.ordinal.to_numpy(), cells.loc[resolved, 'cross_ordinal'].to_numpy()):
        raise ValueError('Case crossing identity mismatch')
    state_names = {-1: 'UNKNOWN', 0: 'NONE', 1: 'LONG_ONLY', 2: 'SHORT_ONLY', 3: 'BOTH'}
    gain = float(root.G_event)
    event_r = float(root.R_event)
    entry = int(root.entry_ns)
    labels = [f'M{int(r.scale)} · '+('lower', 'middle', 'upper')[int(r.role)] for r in cells.itertuples()]
    y = np.arange(9)
    colors = {'L': '#1769aa', 'S': '#e17822'}
    scale_colors = {1: '#167fa0', 3: '#8c6a20', 6: '#7856a6'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(18, 13.5), facecolor='#f6f8fb')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.26, 1], left=.085, right=.96,
                          top=.84, bottom=.13, wspace=.19, hspace=.38)
    ax_t = fig.add_subplot(grid[0, 0])
    ax_n = fig.add_subplot(grid[0, 1], sharey=ax_t)
    ax_p = fig.add_subplot(grid[1, :])
    for ax in (ax_t, ax_n, ax_p):
        ax.set_facecolor('white')
        ax.grid(axis='x', color='#dce2eb', linewidth=.7)
        ax.set_axisbelow(True)
    ax_t.axvspan(0, 300, color='#edf4fa', alpha=.65)
    ax_t.axvline(300, color='#9d3a4a', linestyle='--', linewidth=1.3)
    for i, row in cells.iterrows():
        if not bool(row.ready):
            ax_t.text(15, i, 'геометрия неизвестна', va='center', color='#7d8590', fontsize=10)
        elif row.cross_ordinal < 0:
            known_to_horizon = int(row.support_end_ns) > entry+300*NS
            ax_t.scatter([300], [i], marker='>' if known_to_horizon else 'x', s=75,
                         facecolors='none' if known_to_horizon else '#7d8590', edgecolors='#7d8590')
            ax_t.text(310, i, '>300 s' if known_to_horizon else 'support lost', va='center', fontsize=9, color='#69717c')
        elif row.delay_s <= 300:
            ax_t.hlines(i, 0, row.delay_s, color=scale_colors[int(row.scale)], alpha=.4, linewidth=2)
            ax_t.scatter([row.delay_s], [i], color=scale_colors[int(row.scale)], s=65, zorder=3)
            ax_t.text(row.delay_s-5, i-.18, f'{row.delay_s:.1f} s', ha='right', fontsize=9)
        else:
            ax_t.scatter([300], [i], marker='>', s=75, facecolors='none', edgecolors='#7b8591', linewidth=1.4)
            ax_t.text(310, i, f'>300 s ({row.delay_s/60:.1f} min)', va='center', fontsize=9, color='#69717c')
    ax_t.set_yticks(y, labels)
    ax_t.set_ylim(8.65, -.65)
    ax_t.set_xlim(0, 420)
    ax_t.set_xticks([0, 60, 120, 180, 240, 300])
    ax_t.set_xlabel('Секунды после фактического входа')
    ax_t.set_title('A. Первое пересечение каждой линии', loc='left', fontsize=14, pad=15)
    ax_t.text(0, 1.015, 'Стрелка: правое цензурирование временем 300 s; в скобках — фактическое позднее событие.',
              transform=ax_t.transAxes, fontsize=9, color='#56616e')
    values = cells[['mark_L', 'mark_S']].to_numpy(float)
    lo = min(-1., float(np.nanmin(values))-1)
    hi = max(gain+1, float(np.nanmax(values))+1)
    ax_n.axvspan(gain, hi, color='#e6f3ea')
    ax_n.axvline(0, color='#7d8794', linewidth=1)
    ax_n.axvline(gain, color='#22874c', linestyle='--', linewidth=1.6)
    for i, row in cells.iterrows():
        for si, side in enumerate(('L', 'S')):
            value = float(row['mark_'+side])
            if not np.isfinite(value):
                continue
            yy = i+(-.14 if si == 0 else .14)
            within = bool(row.within_horizon)
            ax_n.scatter([value], [yy], s=53, marker='o' if si == 0 else 's',
                facecolors=colors[side] if within else 'none', edgecolors=colors[side],
                alpha=1 if within else .42, linewidth=1.3, zorder=3)
            ax_n.text(value+(.14 if si == 0 else -.14), yy+(-.1 if si == 0 else .14),
                f'{value:+.1f}', ha='left' if si == 0 else 'right', va='center', fontsize=8.7,
                color=colors[side] if within else '#8a929d')
    ax_n.set_xlim(lo, hi)
    ax_n.set_xlabel('Net при пересечении, USD / условная унция')
    ax_n.tick_params(axis='y', labelleft=False)
    ax_n.set_title('B. Одна цель для всех бюджетов риска', loc='left', fontsize=14, pad=15)
    ax_n.text(gain, -.92, f'фиксированный G = +{gain:.2f}', color='#22874c', ha='center', fontsize=10)
    handles = [Line2D([], [], marker='o', linestyle='', color=colors['L'], label='LONG'),
               Line2D([], [], marker='s', linestyle='', color=colors['S'], label='SHORT'),
               Line2D([], [], marker='o', linestyle='', color='#7d8794', markerfacecolor='none', label='вне 300 s: в Y не входит')]
    ax_n.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.29), ncol=3, fontsize=9, frameon=False)
    plan_labels = []
    for i, row in trades.iterrows():
        side = 'L' if row.side == 1 else 'S'
        name = 'LONG' if row.side == 1 else 'SHORT'
        plan_labels.append(f'{row.plan} · {name}\nR_trade = {row.R_trade:.2f}')
        if row.status != 'COMPLETE':
            ax_p.text(0, i, row.status+' / '+row.reason, va='center', color='#7d8794')
            continue
        ax_p.hlines(i, row.mae, row.mfe, color=colors[side], linewidth=7, alpha=.25)
        ax_p.plot([row.mae, row.mfe], [i, i], '|', color=colors[side], markersize=14)
        ax_p.scatter([row.net], [i], color=colors[side], edgecolors='white', linewidth=1.2, s=95, zorder=4)
        ax_p.plot([-row.R_trade], [i], marker='|', color='#b84955', markersize=25, markeredgewidth=2)
        ax_p.text(row.net, i-.22, f'net {row.net:+.2f}', ha='center', fontsize=9, color=colors[side])
        dt_trigger = (int(row.trigger_ns)-int(row.entry_ns))/NS
        dt_exit = (int(row.exit_ns)-int(row.entry_ns))/NS
        ax_p.text(3.45, i, f'{row.reason}  {dt_trigger:.3f} s → выход {dt_exit:.3f} s',
                  va='center', fontsize=10, color='#3e4854')
    ax_p.axvline(0, color='#7d8794', linewidth=1)
    ax_p.axvline(gain, color='#22874c', linestyle='--', linewidth=1.5)
    ax_p.set_yticks(np.arange(len(trades)), plan_labels)
    ax_p.set_ylim(5.7, -.7)
    ax_p.set_xlim(-5, 6.5)
    ax_p.set_xticks(np.arange(-5, 4, 1))
    ax_p.set_xlabel('Диапазон MAE—MFE до фактического выхода; точка — net, красная отметка — −R_trade')
    ax_p.set_title('C. Меняется только STOP-бюджет; G и crossing-event остаются прежними',
                   loc='left', fontsize=14, pad=15)
    stamp = pd.Timestamp(anchor, tz='UTC').strftime('%d.%m.%Y · %H:%M:%S UTC')
    fig.text(.085, .95, 'Один календарный вход: событие и риск имеют разные роли', fontsize=22, weight='bold', color='#233449')
    fig.text(.085, .913, f'{stamp}  |  R_event = {event_r:.2f}  |  G = {gain:.2f}  |  fixed crossing label: {state_names[int(root.state)]}',
             fontsize=14, color='#31465c')
    fig.text(.085, .883, 'До 300 s пересеклись две линии; ни одна не дала net ≥ G. Более поздние пересечения не меняют этот Y.',
             fontsize=12, color='#56616e')
    fig.text(.085, .071, 'Выбор точки: первый origin 11 мая + 1 час. Исходы не участвовали в выборе. Все линии — BB; VWAP здесь не строился.',
             fontsize=11, color='#485663')
    fig.text(.085, .045, 'Показаны альтернативные сделки по одной условной унции, не equity портфеля. MAE—MFE — диапазон, а не восстановленная траектория.',
             fontsize=11, color='#485663')
    png = OUT/'crossing_case.png'
    fig.savefig(png, dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)
    cell_records = []
    for row in cells.itertuples():
        cell_records.append(dict(scale=int(row.scale), role=int(row.role), event_id=str(row.event_id),
            cross_ordinal=int(row.cross_ordinal), cross_ns=str(int(row.cross_ns)),
            delay_seconds=float(row.delay_s) if np.isfinite(row.delay_s) else None,
            ready=bool(row.ready), support_end_ns=str(int(row.support_end_ns)),
            within_horizon=bool(row.within_horizon),
            horizon_status='OBSERVED_BY_300' if row.within_horizon else 'RIGHT_CENSORED_AT_300' if row.ready and row.support_end_ns > entry+300*NS else 'UNKNOWN_SUPPORT',
            mark_L=float(row.mark_L) if np.isfinite(row.mark_L) else None,
            mark_S=float(row.mark_S) if np.isfinite(row.mark_S) else None))
    trade_records = []
    for row in trades.itertuples():
        trade_records.append(dict(plan=row.plan, side=int(row.side), status=row.status, reason=row.reason,
            R_trade=float(row.R_trade), G_event=float(row.G_event),
            entry_ns=str(int(row.entry_ns)), trigger_ns=str(int(row.trigger_ns)), exit_ns=str(int(row.exit_ns)),
            entry_ordinal=int(row.entry_ordinal), trigger_ordinal=int(row.trigger_ordinal), exit_ordinal=int(row.exit_ordinal),
            net=float(row.net) if np.isfinite(row.net) else None,
            mae=float(row.mae) if np.isfinite(row.mae) else None, mfe=float(row.mfe) if np.isfinite(row.mfe) else None))
    receipt = dict(status='COMPLETE', anchor_rule='2026-05-11 minimum saved origin + 3600 seconds, selected independently of outcomes',
        date=DATE, origin_ns=str(anchor), entry_ns=str(entry), R_event=event_r, G_event=gain,
        y_crossing=int(root.state), state=state_names[int(root.state)], all_known=bool(root.all_known),
        input_sha256=sources, script_sha256=sha(__file__), png=str(png), png_sha256=sha(png),
        cells=cell_records, executions=trade_records,
        interpretation='Fixed first-crossing opportunity target; future beyond300s shown hollow and excluded from Y. Stop budgets change executed alternatives. MAE/MFE are saved ranges, not a fabricated raw tick path.')
    (OUT/'crossing_case.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(status='COMPLETE', origin_ns=str(anchor), state=receipt['state'], png=str(png), png_sha256=receipt['png_sha256'])))


if __name__ == '__main__':
    main()
