"""Visual audit pack for martingale series risk and changing capital bases."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build(output='reports/player_v3/martingale_series_001', data='data/research/player_v3/memory_001'):
    output = Path(output)
    data = Path(data)
    population = pd.read_csv(output / 'population.csv')
    frozen = json.loads((output / 'frozen_before_evaluation.json').read_text())
    evaluation = json.loads((output / 'evaluation/summary.json').read_text())
    protocol = json.loads((output / 'protocol.json').read_text())
    eligible = population.train_complete & population.validation_complete
    top = population[eligible].sort_values(['validation_score', 'train_score', 'index'], ascending=[False, False, True]).head(10)
    top.to_csv(output / 'top_validation_sensitivity.csv', index=False)
    summary = dict(risk_version=protocol['risk_version'], risk_rule=protocol['series_floor_rule'],
                   score=protocol['score'], periods='Development-exposed chronological comparison, not a new blind holdout.',
                   grid_size=len(population), uncensored_eligible=int(eligible.sum()),
                   train_censored=int((~population.train_complete).sum()),
                   validation_censored=int((~population.validation_complete).sum()),
                   positive_validation_scores=int((population.loc[eligible, 'validation_score'] > 0).sum()),
                   train_any_population_ruined=int(population.train_ruined.sum()),
                   validation_ruined=int(population.validation_ruined.sum()),
                   selected=frozen['config'], selected_index=frozen['index'],
                   selected_validation_score=frozen['validation']['score'],
                   selected_validation_series=frozen['validation'].get('series_rows', []),
                   deployment=frozen['deployment_choice'], search_seconds=frozen['elapsed_seconds'],
                   test={k: {n: r[n] for n in ('net', 'final_equity', 'max_drawdown', 'max_series_loss_fraction', 'trades', 'wins',
                                              'ruined', 'ruin_ns', 'complete', 'survival_fraction', 'lifetime_eligible_seconds', 'score')}
                         for k, r in evaluation['results'].items()})
    for label in ('selected_martingale', 'matched_1oz', 'wait'):
        cycles = pd.read_csv(output / f'evaluation/{label}/cycles.csv')
        summary['test'][label].update(
            series_count=len(cycles), recovered_series=int((cycles.status == 'recovered').sum()),
            ruined_series=int((cycles.status == 'ruined').sum()),
            censored_series=int((cycles.status == 'censored').sum()),
            series_bases=cycles.start_equity.tolist(), series_floors=cycles.floor.tolist())
        try:
            trades = pd.read_csv(output / f'evaluation/{label}/trades.csv')
        except pd.errors.EmptyDataError:
            trades = pd.DataFrame()
        if len(trades):
            summary['test'][label].update(
                gross=float(trades.gross.sum()), fees=float(trades.fee.sum()),
                average_size_oz=float(trades.size_oz.mean()),
                median_holding_seconds=float(((trades.exit_ns - trades.entry_ns) / 1e9).median()),
                exit_reasons=trades.reason.value_counts().to_dict(),
                last_close_utc=str(pd.to_datetime(trades.exit_ns.iloc[-1], utc=True)))
    (output / 'audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    plt.rcParams.update({'figure.dpi': 130, 'axes.grid': True, 'grid.alpha': .2, 'font.size': 10})
    fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
    complete = population[eligible]
    points = ax.scatter(complete.validation_survival_fraction * 100, complete.validation_net,
                        c=complete.cap_oz, cmap='viridis', s=45, alpha=.8, label='Complete train + validation')
    censored = population[~eligible]
    ax.scatter(censored.validation_survival_fraction * 100, censored.validation_net,
               marker='x', color='gray', alpha=.35, label='Censored: excluded from selection')
    winner = population.iloc[frozen['index']]
    ax.scatter([winner.validation_survival_fraction * 100], [winner.validation_net],
               marker='*', s=230, facecolors='none', edgecolors='red', linewidths=1.5, label='Frozen experimental winner')
    ax.scatter([100], [0], marker='D', color='black', s=65, label='WAIT')
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(xlabel='Validation observed trading lifetime (% of week)', ylabel='Validation net ($)',
           title='180 fixed policies: death strictly beyond 50% loss of the current series\nThe series base changes only after recovery and a new fill')
    ax.legend(loc='lower right', fontsize=8)
    fig.colorbar(points, ax=ax, label='Position cap (oz)')
    fig.savefig(output / 'population_survival.png')
    plt.close(fig)

    date = '2026-05-18'
    meta = json.loads((data / date / 'manifest.json').read_text())
    minute_t = np.load(data / date / 'decision_t.npy', mmap_mode='r')
    price = np.load(data / date / 'decision_mid.npy', mmap_mode='r')
    london = int(meta['london_ns'])
    horizon = london + 4 * 3600 * 10**9
    visible = (minute_t >= london) & (minute_t <= horizon)
    minute_axis = (minute_t[visible] - london) / 60e9
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout='constrained')
    axes[0].plot(minute_axis, price[visible], color='#222222', linewidth=.9)
    trades = pd.read_csv(output / 'evaluation/selected_martingale/trades.csv')
    for side, marker, color in ((1, '^', '#15803d'), (-1, 'v', '#b91c1c')):
        group = trades[trades.side == side]
        axes[0].scatter((group.entry_ns - london) / 60e9, group.entry_nano / 1e9,
                        marker=marker, s=22 + group.size_oz * 7, color=color,
                        label=('Long' if side > 0 else 'Short') + ' (marker size = oz)', alpha=.75)
    axes[0].set(ylabel='GC quote price ($/oz)', title='Corrected series-risk rule on 18 May: exact quote equity and dynamic floor')
    axes[0].legend(fontsize=8)
    cfg = frozen['config']
    for label, color, title, axis in (
            ('selected_martingale', '#b91c1c', f"Selected {cfg['multiplier']:g}x, cap {cfg['cap_oz']:g} oz", axes[1]),
            ('matched_1oz', '#2563eb', 'Same entry/exit rule, fixed 1 oz', axes[2])):
        directory = output / 'evaluation' / label
        quote_t = np.load(data / date / 'quote_t.npy', mmap_mode='r')
        equity = np.load(directory / f'quote_equity_{date}.npy', mmap_mode='r')
        floor = np.load(directory / f'quote_series_floor_{date}.npy', mmap_mode='r')
        visible_q = (quote_t >= london) & (quote_t <= horizon) & np.isfinite(equity)
        idx = np.flatnonzero(visible_q)
        if len(idx):
            # Show a bounded sample but always preserve floor transitions and
            # minimum equity. Risk itself was checked on every exact quote.
            changed = np.flatnonzero(np.diff(np.nan_to_num(floor, nan=-1.)) != 0)
            changed = np.unique(np.r_[changed, changed + 1])
            changed = changed[(changed >= idx[0]) & (changed <= idx[-1])]
            idx = np.unique(np.r_[idx[::max(1, len(idx) // 12000)], changed, idx[-1], idx[np.argmin(equity[idx])]])
            axis.plot((quote_t[idx] - london) / 60e9, equity[idx], color=color, label=title)
            axis.plot((quote_t[idx] - london) / 60e9, floor[idx], color='#a16207', linestyle='--',
                      label='50% of active series base; death strictly below')
        axis.axhline(200, color='#222222', linewidth=.8, label='WAIT equity $200')
        result = evaluation['results'][label]
        stop_ns = int(result['ruin_ns'] if result['ruined'] else result['censor_ns'] if result['censored'] else 0)
        if london <= stop_ns <= horizon:
            minutes = (stop_ns - london) / 60e9
            axis.axvline(minutes, color=color, linestyle=':', linewidth=1.)
            text = ('Series loss >50%; stopped' if result['ruined'] else 'Support censored; no score')
            axis.annotate(text, (minutes, result['final_equity']), xytext=(8, 20),
                          textcoords='offset points', color=color, fontsize=9)
        axis.set(ylabel='Liquidation equity ($)', xlim=(0, 240))
        axis.legend(fontsize=8)
    axes[2].set(xlabel='Minutes since London 08:00 local (07:00 UTC)')
    fig.savefig(output / 'test_price_equity.png')
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout='constrained')
    index = np.arange(1, len(trades) + 1)
    axes[0].step(index, trades.size_oz, where='mid', color='#7c3aed')
    axes[0].set(ylabel='Position size (oz)', title='Series sizing: partial wins hold size; whole-series recovery resets the next entry')
    axes[1].bar(index, trades.net, color=np.where(trades.net > 0, '#15803d', '#b91c1c'))
    axes[1].plot(index, -trades.fee, color='#222222', linewidth=.8, label='Round-trip fee')
    axes[1].set(ylabel='Trade net ($)')
    axes[1].legend(fontsize=8)
    axes[2].plot(index, trades.series_start_equity + trades.series_cumulative_net, color='#b91c1c', label='Cash after closed leg')
    axes[2].step(index, trades.series_start_equity, where='mid', color='#15803d', label='Current series base')
    axes[2].step(index, trades.series_floor, where='mid', color='#a16207', linestyle='--', label='Dynamic 50% floor')
    recovered = trades.series_status == 'recovered'
    axes[2].scatter(index[recovered], (trades.series_start_equity + trades.series_cumulative_net)[recovered],
                    color='#15803d', s=20, label='Series recovered')
    for axis in axes:
        for start in index[np.r_[True, np.diff(trades.series_id) != 0]]:
            axis.axvline(start - .5, color='#777777', linewidth=.7, alpha=.25)
    axes[2].set(xlabel='Closed leg number; vertical lines begin a new series', ylabel='Cash / series base ($)')
    axes[2].legend(fontsize=8)
    fig.savefig(output / 'test_trade_sizing.png')
    plt.close(fig)

    validation_series = frozen['validation'].get('series_rows', [])
    if validation_series:
        # This plot deliberately shows series states, not an invented
        # intra-series equity interpolation. Exact-quote traces are above.
        starts = np.asarray([int(row['series_start_ns']) for row in validation_series], dtype=np.int64)
        ends = np.asarray([int(row.get('end_ns') or row.get('last_trade_exit_ns') or row['series_start_ns'])
                           for row in validation_series], dtype=np.int64)
        origin = starts[0]
        fig, ax = plt.subplots(figsize=(11, 4.8), layout='constrained')
        for i, row in enumerate(validation_series):
            x0, x1 = (starts[i] - origin) / 60e9, (ends[i] - origin) / 60e9
            floor = row['series_start_equity'] * .5
            ax.plot([x0, x1], [floor, floor], color='#a16207', linewidth=2.5)
            ax.scatter([x0], [floor], color='#a16207', s=25)
            ax.annotate(f"Series {row['series_id']}: base USD {row['series_start_equity']:.2f}, floor USD {floor:.2f}",
                        (x0, floor), xytext=(8, 12), textcoords='offset points', fontsize=9)
            if row['series_status'] in ('ruined', 'censored'):
                mark = row['liquidation_equity']
                ax.scatter([x1], [mark], color='#b91c1c', marker='X', s=75)
                ax.annotate(f"{row['series_status']}: terminal mark ${mark:.2f}", (x1, mark),
                            xytext=(-8, -22), ha='right', textcoords='offset points', color='#b91c1c')
        ax.set(xlabel='Minutes since first validation fill', ylabel='Series loss boundary ($)',
               title='Frozen validation candidate: a recovered gain changes the next series base\nRecorded series boundaries and terminal mark; not an intra-series equity path')
        ax.margins(x=.08, y=.4)
        fig.savefig(output / 'validation_series_bases.png')
        plt.close(fig)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/player_v3/martingale_series_001')
    parser.add_argument('--data', default='data/research/player_v3/memory_001')
    args = parser.parse_args()
    build(args.output, args.data)
