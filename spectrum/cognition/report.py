"""Reproducible report of paired targets and separately replayed exit policies.

Reads completed experiment artifacts only. It never tunes a threshold, selects
a seed by test performance, or adds overlapping option profits into equity.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAIN = 'full_61021'
COLORS = {'full': '#137dba', 'account_only': '#c07824', 'permuted_targets': '#9272ae'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def fmt(value, digits=3):
    if value is None or not np.isfinite(value):
        return '—'
    return f'{value:,.{digits}f}'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def figure(path, caption, alt=None):
    return f'<figure><img src="{html.escape(path)}" alt="{html.escape(alt or caption)}" loading="lazy"><figcaption>{html.escape(caption)}</figcaption></figure>'


def weights(frame):
    roots = frame.groupby('date').root_id.transform('nunique').to_numpy(float)
    ages = frame.groupby('root_id').root_id.transform('size').to_numpy(float)
    w = 1 / (roots * ages)
    return w / w.mean()


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def training_plot(out, protocol, summary):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex='col')
    for col, variant in enumerate(COLORS):
        models = [s for s in summary if s['model'].rsplit('_', 1)[0] == variant]
        for i, model in enumerate(models):
            history = read_json(out / (model['model'] + '_history.json'))
            x = [h['epoch'] for h in history]
            alpha = 1 if i == 0 else .38
            train = [h['train_loss'] * protocol['target_scale'] ** 2 for h in history]
            val = [h['weighted_mse'] for h in history]
            axes[0, col].plot(x, train, color=COLORS[variant], alpha=alpha, linestyle='--', label='train online MSE' if i == 0 else None)
            axes[0, col].plot(x, val, color=COLORS[variant], alpha=alpha, label='validation MSE' if i == 0 else None)
            axes[0, col].scatter([model['best_epoch']], [history[model['best_epoch'] - 1]['weighted_mse']], s=35, color=COLORS[variant])
            axes[1, col].plot(x, [h['close_fraction'] * 100 for h in history], color=COLORS[variant], alpha=alpha, label=model['model'])
        axes[0, col].set_title(variant)
        axes[0, col].set_ylabel('Weighted MSE, USD²')
        axes[0, col].legend(fontsize=8)
        axes[1, col].set_ylabel('Validation CLOSE, weighted %')
        axes[1, col].set_xlabel('Epoch')
        axes[1, col].legend(fontsize=7)
        axes[1, col].set_ylim(0, 102)
        for row in range(2):
            axes[row, col].grid(alpha=.2)
    fig.suptitle('Training dynamics — validation chooses checkpoint; seed 61021 fixed in advance', fontsize=14)
    fig.tight_layout()
    save_figure(fig, out / 'training.png')


def calibration(out, frame):
    p = frame[MAIN].to_numpy(float)
    y = frame.target_advantage.to_numpy(float)
    w = weights(frame)
    # Equal-count prediction bins. Tied predictions remain together.
    bins = pd.qcut(p, q=min(10, len(np.unique(p))), duplicates='drop')
    records = []
    for bucket in bins.categories:
        m = np.asarray(bins == bucket)
        if not m.any():
            continue
        records.append(dict(bin=str(bucket), n=int(m.sum()), predicted=float(np.average(p[m], weights=w[m])),
            observed=float(np.average(y[m], weights=w[m])), observed_hold_positive=float(np.average(y[m] > 0, weights=w[m]))))
    fig, ax = plt.subplots(figsize=(8, 5))
    xp = [r['predicted'] for r in records]
    yp = [r['observed'] for r in records]
    ax.plot(xp, yp, '-o', color=COLORS['full'], label='Observed paired advantage in prediction bins')
    limits = [min(xp + yp + [0.]), max(xp + yp + [0.])]
    ax.plot(limits, limits, '--', color='gray', label='Perfect mean calibration')
    ax.axhline(0, color='black', linewidth=.7)
    ax.axvline(0, color='black', linewidth=.7)
    for r in records:
        ax.annotate(f"n={r['n']}", (r['predicted'], r['observed']), fontsize=7, xytext=(3, 4), textcoords='offset points')
    ax.set(xlabel='Predicted HOLD_fixed − CLOSE, USD', ylabel='Observed HOLD_fixed − CLOSE, USD', title='August paired calibration — full seed 61021')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    save_figure(fig, out / 'calibration.png')
    write_json(out / 'calibration.json', records)
    return records


def attach_account_slices(frame, data, manifest):
    rows = []
    for day in manifest['days']:
        if day['split'] != 'test':
            continue
        x = np.load(data / day['date'] / 'x.npy', mmap_mode='r')
        part = frame.loc[frame.date == day['date']]
        idx = part.day_row.to_numpy(int)
        if len(idx) and (idx.min() < 0 or idx.max() >= len(x)):
            raise ValueError('Pair index outside causal feature matrix')
        for name in ('unrealized_usd', 'time_left_fraction', 'position_mfe_usd', 'position_mae_usd'):
            if name not in frame:
                frame[name] = np.nan
            frame.loc[part.index, name] = x[idx, day['feature_names'].index(name)]
        rows.extend(part.index)
    if len(rows) != len(frame) or frame.unrealized_usd.isna().any():
        raise ValueError('Incomplete account slice alignment')
    return frame


def semantic_slices(out, frame):
    f = frame.copy()
    f['w'] = weights(f)
    f['side_label'] = f.side.map({-1: 'SHORT', 1: 'LONG'})
    f['plan_label'] = f.plan_id.map({0: 'P0: 120 s', 1: 'P1: 300 s', 2: 'P2: 600 s'})
    f['age_label'] = pd.cut(f.age_seconds, [0, 15, 60, 180, 10000], labels=['5–15 s', '30–60 s', '120–180 s', '240+ s'])
    f['pnl_label'] = pd.cut(f.unrealized_usd, [-np.inf, -.3, 0, .3, np.inf], labels=['gross unrealized < −0.30', '−0.30..0', '0..+0.30', '> +0.30'])
    records = []
    for axis in ('side_label', 'plan_label', 'age_label', 'pnl_label'):
        for value, g in f.groupby(axis, observed=True):
            y = g.target_advantage.to_numpy(float)
            p = g[MAIN].to_numpy(float)
            w = g.w.to_numpy(float)
            records.append(dict(axis=axis, group=str(value), pairs=len(g), roots=g.root_id.nunique(),
                close_fraction=float(np.average(p < 0, weights=w)), predicted=float(np.average(p, weights=w)),
                observed=float(np.average(y, weights=w)), gain_vs_hold=float(np.average(np.where(p < 0, -y, 0), weights=w))))
    write_json(out / 'behavior_slices.json', records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    labels = [r['group'] for r in records]
    yaxis = np.arange(len(records))
    axes[0].barh(yaxis, [100 * r['close_fraction'] for r in records], color=COLORS['full'])
    axes[0].set(xlabel='Weighted CLOSE, %', xlim=(0, 102), yticks=yaxis, yticklabels=labels)
    gain = [r['gain_vs_hold'] for r in records]
    axes[1].barh(yaxis, gain, color=['#24886d' if v >= 0 else '#ba4c49' for v in gain])
    axes[1].axvline(0, color='black', linewidth=.7)
    axes[1].set_xlabel('Paired single-decision gain vs fixed HOLD, USD')
    axes[0].invert_yaxis()
    fig.suptitle('Operational behavior slices — descriptive, overlapping, not discovered causal schemes')
    fig.tight_layout()
    save_figure(fig, out / 'behavior_slices.png')
    return records


def sequential_plot(out, summary, options):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    labels = [r['controller'] for r in summary]
    for i, r in enumerate(summary):
        color = '#137dba' if r['controller'] == MAIN else '#75849b'
        axes[0].barh(i, r['mean_net'], color=color)
        interval = r['day_block_interval']
        if interval is not None:
            axes[1].plot([interval['lower'], interval['upper']], [i, i], color=color, linewidth=3)
            axes[1].scatter([interval['mean']], [i], color=color, s=30)
        axes[2].plot([r['equity_min'] - 1500, r['equity_max'] - 1500], [i, i], color=color, linewidth=3)
        axes[2].scatter([r['mean_net']], [i], color=color, s=25)
    axes[0].set(yticks=np.arange(len(labels)), yticklabels=labels, xlabel='Mean option net, USD (all paired-known roots)')
    axes[1].set_xlabel('Day-balanced delta vs HOLD, USD\n95% exploratory 5-observed-day block interval')
    axes[2].set_xlabel('Exact observed option equity range − $1500\nDot = mean terminal net; resets for each option')
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axvline(0, color='black', linewidth=.7)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Repeated decisions on independent options — not a continuous trading account')
    fig.tight_layout()
    save_figure(fig, out / 'sequential.png')
    f = options.loc[(options.controller == MAIN) & options.paired_known].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(f.net, bins=40, color=COLORS['full'], alpha=.8)
    axes[0].axvline(0, color='black', linewidth=.7)
    axes[0].set(xlabel='Net per option, USD', ylabel='Options', title='Repeated full_61021: realized outcomes')
    axes[1].scatter(f.equity_min - 1500, f.equity_max - 1500, s=8, alpha=.15, color=COLORS['full'])
    axes[1].set(xlabel='Min equity − $1500, USD', ylabel='Max equity − $1500, USD', title='Exact min/max from all replayed quotes, each option')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    save_figure(fig, out / 'option_ranges.png')


def case_plots(out, cases, options):
    rendered = []
    for index, case in enumerate(cases[:12], 1):
        price = pd.DataFrame(case['price'])
        decisions = pd.DataFrame(case['decisions'])
        matching = options.loc[(options.root_id == case['root_id']) & (options.controller == case['controller'])]
        if len(matching) != 1:
            raise ValueError('Case must match exactly one sequential option')
        record = matching.iloc[0]
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        axes[0].plot(price.seconds, price.bid, color='#237aaa', label='bid', linewidth=1)
        axes[0].plot(price.seconds, price.ask, color='#c18a4b', label='ask', linewidth=1, alpha=.8)
        axes[0].axhline(case['entry_price'], color='gray', linestyle=':', label='entry fill')
        axes[0].legend(loc='best', fontsize=8)
        axes[0].set_ylabel('GC quote, USD/oz')
        if len(decisions):
            axes[1].plot(decisions.seconds, decisions.advantage, '-o', color=COLORS['full'], markersize=4)
            for _, d in decisions.iterrows():
                axes[1].annotate(d.action, (d.seconds, d.advantage), fontsize=7, xytext=(0, 6), textcoords='offset points')
            # Decision equity is exact at saved states; no fictitious dense path.
            axes[2].scatter(decisions.seconds, decisions.equity - 1500, color=COLORS['full'], s=25, label='exact equity at decision')
        axes[1].axhline(0, color='black', linewidth=.7)
        axes[1].set_ylabel('Predicted advantage\nHOLD_fixed − CLOSE, USD')
        axes[2].axhspan(record.equity_min - 1500, record.equity_max - 1500, color=COLORS['full'], alpha=.1, label='exact whole-option min/max range')
        axes[2].scatter([case['exit_seconds']], [case['net']], color='#ca4845', marker='x', s=55, label='executed terminal net')
        axes[2].axhline(0, color='black', linewidth=.7)
        axes[2].set(ylabel='Equity − $1500, USD', xlabel='Seconds since entry fill (price picture sampled only)')
        axes[2].legend(fontsize=7, loc='best')
        for ax in axes:
            ax.axvline(0, color='gray', linewidth=.6)
            ax.axvline(case['exit_seconds'], color='#ca4845', linestyle='--', linewidth=.8)
            ax.grid(alpha=.2)
        side = 'LONG' if case['side'] == 1 else 'SHORT'
        fig.suptitle(f"{case['root_id']} | {side}, plan {case['plan_id']}\nnet {case['net']:+.2f}, fixed HOLD {case['reference_net']:+.2f}, delta {case['net'] - case['reference_net']:+.2f} USD", fontsize=12)
        fig.tight_layout()
        name = f'case_{index:02d}.png'
        save_figure(fig, out / name)
        rendered.append(dict(file=name, root_id=case['root_id'], side=side, net=case['net'], reference_net=case['reference_net'],
            delta=case['net'] - case['reference_net'], exit_reason=record.exit_reason,
            min_equity=float(record.equity_min), max_equity=float(record.equity_max)))
    return rendered


def build(data='data/derived/hold_close_v1', output='reports/hold_close_v1'):
    data, out = Path(data), Path(output)
    required = ['training_summary.json', 'training_protocol.json', 'sequential_summary.json', 'sequential_options.parquet',
                'paired_predictions.parquet', 'cases.json', 'evaluation_receipt.json']
    missing = [str(out / name) for name in required if not (out / name).exists()]
    if missing:
        raise FileNotFoundError('Completed inputs required: ' + ', '.join(missing))
    manifest = read_json(data / 'manifest.json')
    protocol = read_json(out / 'training_protocol.json')
    evaluation = read_json(out / 'evaluation_receipt.json')
    comparisons = read_json(out / 'sequential_comparisons.json') if (out / 'sequential_comparisons.json').exists() else []
    audit = read_json(out / 'behavior_audit.json') if (out / 'behavior_audit.json').exists() else None
    if any(x['status'] != 'COMPLETE' for x in (manifest, protocol, evaluation)):
        raise ValueError('Incomplete experiment cannot become a result report')
    if protocol['data_manifest_sha256'] != digest(data / 'manifest.json') or evaluation['source_manifest_sha256'] != digest(data / 'manifest.json'):
        raise ValueError('Training/evaluation data identities differ')
    summary = read_json(out / 'training_summary.json')
    sequential = read_json(out / 'sequential_summary.json')
    options = pd.read_parquet(out / 'sequential_options.parquet')
    paired = pd.read_parquet(out / 'paired_predictions.parquet')
    test = paired.loc[(paired.split == 'test') & paired.target_known].copy()
    if not len(test) or not len(options):
        raise ValueError('Empty test')
    if (test.market_available_ns > test.decision_ns).any():
        raise ValueError('Future market features')
    if paired.groupby('root_id').split.nunique().max() != 1:
        raise ValueError('Root crossed split')
    cases = read_json(out / 'cases.json')
    if len(cases) < 6:
        raise ValueError('At least six deterministic visual cases required')
    training_plot(out, protocol, summary)
    bins = calibration(out, test)
    test = attach_account_slices(test, data, manifest)
    slices = semantic_slices(out, test)
    sequential_plot(out, sequential, options)
    images = case_plots(out, cases, options)

    by_model = {r['controller']: r for r in sequential}
    main = by_model[MAIN]
    account = by_model['account_only_61021']
    shuffled = by_model['permuted_targets_61021']
    interval = main['day_block_interval']
    interval_text = f"[{fmt(interval['lower'])}; {fmt(interval['upper'])}]" if interval else 'не определён'
    development = main['mean_delta_vs_hold']
    opening = ('Повторяемая политика улучшила средний исход относительно фиксированного HOLD на этих опциях.' if development > 0
               else 'Повторяемая политика не улучшила средний исход относительно фиксированного HOLD на этих опциях.')
    market = ('Полный вход дал более высокий средний исход, чем контроль только со состоянием позиции.' if main['mean_net'] > account['mean_net']
              else 'Полный вход не превзошёл контроль только со состоянием позиции по среднему исходу.')
    train_rows = []
    for r in summary:
        m = r['splits']['test']
        train_rows.append([r['model'], r['best_epoch'], fmt(r['splits']['train']['weighted_mse']), fmt(r['splits']['validation']['weighted_mse']),
            fmt(m['weighted_mse']), fmt(m['close_fraction'] * 100, 1) + '%', fmt(m['paired_gain_vs_hold']), fmt(m['hindsight_regret'])])
    seq_rows = []
    for r in sequential:
        interval_r = r['day_block_interval']
        ci = f"{fmt(interval_r['mean'])} [{fmt(interval_r['lower'])}; {fmt(interval_r['upper'])}]" if interval_r else '—'
        seq_rows.append([r['controller'], r['known_comparisons'], r['zero_decision_roots'], fmt(r['mean_net']), fmt(r['mean_delta_vs_hold']), ci,
                        r['roots_with_close'], r['censored'], f"{fmt(r['equity_min'], 2)}…{fmt(r['equity_max'], 2)}"])
    comparisons_html = ''
    if comparisons:
        comparison_rows = [[r['control'], r['roots'], fmt(r['root_mean_delta']),
            f"[{fmt(r['interval']['lower'])}; {fmt(r['interval']['upper'])}]" if r['interval'] else '—'] for r in comparisons]
        comparisons_html = '<h3>Разность с каждым контролем на одних и тех же корнях</h3>' + table(['Контроль', 'Корней', 'Full 61021 − контроль $', '95% блоковый интервал $'], comparison_rows)
        comparisons_html += '<p>Каждая разность рассчитана внутри одного root, затем агрегирована. Интервал, включающий ноль, не поддерживает уверенный вывод о превосходстве над данным контролем.</p>'
    audit_html = ''
    if audit:
        perm = audit['family_permutations']
        perm_rows = [[r['family'], fmt(r['mse_increase']), fmt(100 * r['decision_disagreement'], 1) + '%',
            fmt(r['paired_gain_vs_hold'])] for r in perm]
        surrogate = audit['surrogate']
        audit_html = '<h3>Использование входов и простое описание решений</h3>' + table(['Перемешанное семейство', 'Δ MSE $²', 'Сменилось решений', 'Парный gain к HOLD $'], perm_rows)
        audit_html += f'''<p>Перемешивание семейства разрушает его связи с состоянием. Это диагностическое вмешательство, не причинный экономический эффект: оно может создавать неестественные комбинации входов. Если разрушение рыночного контекста не ухудшает результат, нет оснований приписывать модели полезное понимание потока или спектров.</p>
<p>Дерево глубины 3 обучено только на train имитировать решения сети. На тесте совпадение {100 * surrogate['fidelity']:.1f}%, у постоянного CLOSE — {100 * surrogate['always_close_fidelity']:.1f}%; balanced accuracy дерева {100 * surrogate['balanced_accuracy']:.1f}%. Это приближённое описание выходов сети, не новое торговое правило. True = CLOSE; False = HOLD.</p>
<details><summary>Правила приближённого дерева решений</summary><pre>{html.escape(surrogate['rule_text'])}</pre></details>
<p>Аудит повторно воспроизвёл {len(audit['target_reconciliation'])} равномерно выбранных пар из сохранённых состояний: {sum(bool(r['pass_']) for r in audit['target_reconciliation'])} совпадений. Постоянными в учебных состояниях были {len(audit['risk_coverage']['constant_account_features'])} полей счёта; наличие полей в интерфейсе не означает освоенную реакцию на их новые значения.</p>'''
    coverage = []
    for split in ('train', 'validation', 'test'):
        days = [d for d in manifest['days'] if d['split'] == split]
        coverage.append([split, min(d['date'] for d in days), max(d['date'] for d in days), len(days), sum(d['roots'] for d in days),
            sum(d['pairs'] for d in days), sum(d['known_pairs'] for d in days), sum(d['zero_decision_roots'] for d in days)])
    slice_rows = [[r['group'], r['pairs'], r['roots'], fmt(100 * r['close_fraction'], 1) + '%', fmt(r['predicted']), fmt(r['observed']), fmt(r['gain_vs_hold'])] for r in slices]
    negative = options.loc[(options.controller == MAIN) & options.paired_known & (options.delta_vs_hold < 0)]
    positive = options.loc[(options.controller == MAIN) & options.paired_known & (options.delta_vs_hold > 0)]
    neutral = options.loc[(options.controller == MAIN) & options.paired_known & (options.delta_vs_hold.abs() <= 1e-10)]
    sections = [f'''<header><p class="eyebrow">HOLD / CLOSE · эксперимент v1 · 19 октября 2026</p><h1>Что научился делать менеджер позиции</h1>
<p class="lede">{opening} {market} Это проверка одного умения на независимых попытках, а не прибыль непрерывного счёта.</p></header>
<nav><a href="#result">Результат</a><a href="#contract">Контракт</a><a href="#learning">Обучение</a><a href="#behavior">Поведение</a><a href="#cases">Примеры</a><a href="#limits">Границы вывода</a></nav>
<section id="result"><h2>Результат повторяемого управления</h2><div class="cards">
<div><b>{fmt(main['mean_net'])} $</b><span>средний net на опцию · full 61021</span></div>
<div><b>{fmt(main['mean_delta_vs_hold'])} $</b><span>изменение к фиксированному HOLD</span></div>
<div><b>{interval_text}</b><span>95% исследовательский блоковый интервал · среднее по дням</span></div>
<div><b>{main['roots_with_close']} / {main['roots']}</b><span>опций с решением CLOSE</span></div></div>
<p>Главный seed 61021 закреплён заранее; остальные seed показывают чувствительность. Прирост к отрицательному контролю ещё не означает положительный net. Контроль account-only: {fmt(account['mean_net'])} $; перемешанные обучающие цели: {fmt(shuffled['mean_net'])} $ на опцию.</p>
{figure('sequential.png', 'Слева средний результат всех опций с известной парой. В центре среднее изменение по дням и чувствительность к выборке пяти последовательных наблюдаемых дней. Справа полный наблюдённый диапазон equity отдельных опций, не доверительный интервал и не траектория счёта.')}
{table(['Контроллер', 'Известных пар опций', 'Без решения', 'Средний net $', 'Δ к HOLD $', 'Δ по дням [95% блок] $', 'Опций CLOSE', 'Цензур.', 'Min…max equity $'], seq_rows)}
{comparisons_html}
<p>Опции без доступного решения не удалены: защитный план мог закрыть их раньше первого решения на пятой секунде. Интервалы учитывают локальную зависимость дней, но не прошлый подбор по этой истории и не дают доказательства будущей прибыли.</p>
{figure('option_ranges.png', 'Каждая точка min/max принадлежит одной попытке со свежим капиталом $1500; экстремумы рассчитаны replay по всем котировкам, а не по M1 или только моментам решений.')}</section>''',
f'''<section id="contract"><h2>Какой смысл поставлен перед моделью</h2>
<p class="formula">A(s) = итоговые cash при HOLD фиксированного плана − итоговые cash при CLOSE сейчас</p>
<p>Модель регрессирует парную разность в долларах. Если прогноз отрицателен — отправляет CLOSE, иначе HOLD. Это supervised-обучение оценки Q относительно фиксированного продолжения, не PPO и не оценка собственной многократно применяемой политики. Последняя отдельно воспроизведена в симуляции выше.</p>
<p>Обе ветви начинают из полностью одинакового сериализованного состояния и используют одну поддержанную ленту, общий горизонт, bid/ask, 1 oz и $0.30 за круг. CLOSE исполняется после задержки 100 ms при наличии допустимой котировки; после закрытия деньги остаются неизменными до общего горизонта. Повторных входов в опции нет. Механические stop/target/time/session/barrier срабатывают немедленно на публикации по объявленному контракту и имеют приоритет над ожидающим CLOSE; это асимметрия модели исполнения.</p>
<p>Позиции открываются на независимой от будущей цены сетке раз в полчаса, в обе стороны и с тремя планами: target/stop/max time = 12/8 ticks/120 s, 24/12 ticks/300 s, 40/20 ticks/600 s. Tick = $0.10/oz. Это разнообразие учебных попыток, не обученные входы. Каждый корень начинает со свежих $1500, без накопленного долга предыдущей серии.</p>
{table(['Выборка', 'Первая дата', 'Последняя дата', 'Дней', 'Опций', 'Пар решений', 'Известных целей', 'Опций без решения'], coverage)}
<p>Всего {len(manifest['days'])} доступных дней мая–августа. 31 июля отсутствует в подготовленном источнике и не заменяется синтетическими свечами. Август — хронологическая техническая проверка на ранее исследованной проектом истории, не новый слепой holdout. Веса обучения выравнивают дни, корни внутри дня и возрасты внутри корня; соседние решения и перекрывающиеся попытки зависимы.</p></section>''',
f'''<section id="learning"><h2>Как проходило обучение</h2>
<p>{html.escape(protocol['network'])}. {protocol['epochs']} эпох; {html.escape(protocol['optimizer'])}. Нормализация обучена только на мае–июне, известность признаков учитывается масками. Входы ограничены после нормализации; денежные исходы не обрезаются. Выбор checkpoint — минимальная MSE июля; порог CLOSE = 0 заранее фиксирован.</p>
<p>Полный вход содержит компактные рыночные признаки, причинные сводки 5/15/60 минут и полное наблюдение позиции. Account-only сохраняет план, PnL, возраст, расстояния до защит, остаток времени и состояние счёта, без рыночного контекста. Перемешивание обучающих целей проверяет, нужен ли корректный сигнал обучения.</p>
{figure('training.png', 'Train loss приведён к USD² через квадрат train-only масштаба цели; это средняя online minibatch loss по обновлявшейся модели, а validation — оценка модели конца эпохи. Слабые линии — дополнительные seed. Частота CLOSE показана для validation, не для train.')}
{table(['Модель', 'Выбрана эпоха', 'Train MSE $²', 'Val MSE $²', 'Test MSE $²', 'Test CLOSE', 'Парный gain к HOLD $', 'Hindsight regret $'], train_rows)}
<p>Парный gain — результат одного выбора относительно фиксированного продолжения; это не средняя прибыль повторяемой политики. Hindsight regret использует известный после опыта лучший исход двух ветвей как диагностический ориентир, не как доступное модели знание.</p>
{figure('calibration.png', 'Для заранее выбранной full_61021: группы по прогнозу, затем средний фактический парный эффект с теми же весами. Отклонение от диагонали показывает ошибку калибровки; число строк не равно числу независимых рыночных ситуаций.')}</section>''',
f'''<section id="behavior"><h2>Какие операционные схемы проявились</h2>
<p>Здесь «схема» означает наблюдаемую зависимость решения от стороны, плана, возраста и PnL. Мы не приписываем скрытым нейронам понимание паттерна без проверки. Группы пересекаются; отличия между ними могут объясняться составом состояний, не причинным эффектом одного признака.</p>
{figure('behavior_slices.png', 'Избирательность CLOSE и фактический парный эффект в заранее определённых срезах августа. Это описание поведения на фиксированных состояниях, отдельно от повторяемой политики.')}
{table(['Срез', 'Пар', 'Корней', 'CLOSE', 'Прогноз A $', 'Факт A $', 'Парный gain к HOLD $'], slice_rows)}
{audit_html}
<p>Повторяемая full_61021 дала лучший результат, чем фиксированный HOLD, на {len(positive)} опциях, худший — на {len(negative)}, тот же — на {len(neutral)}. Средний Δ среди ухудшенных опций: {fmt(negative.delta_vs_hold.mean() if len(negative) else None)} $. Негативные исходы сохраняются: одно улучшение средней метрики не отменяет пропущенные восстановления и раннее прекращение выгодного удержания.</p>
<p>Проверка рыночного смысла требует преимущества над account-only, устойчивости между seed и согласованности калибровки. Даже преимущество полного входа не доказывает конкретную историю вроде «поглощение потока»: для такой формулировки нужен следующий контролируемый опыт с удалением семейства признаков.</p></section>''',
'''<section id="cases"><h2>Детерминированные примеры: цена, оценка, решение, equity</h2><p>Первые подходящие корни каждого сочетания side/plan на первых двух днях теста, прожившие хотя бы 30 секунд по reference. Отбор задан до просмотра прибыли, примеры не ранжировались по успешности. Условие 30 секунд означает, что это визуальная проверка живых решений, а не представительная выборка всех, в том числе мгновенно закрывшихся, опций.</p><p>Цена на рисунках прорежена до последней котировки секунды; исполнение использовало все котировки. На нижней панели показаны точные equity в моменты решений, терминальный net и полный диапазон min/max опции. Между этими точками не выдумана непрерывная кривая капитала. Красная вертикаль — фактический выход; цена после неё показана только как последующий ход рынка.</p>'''
    ]
    for item in images:
        caption = f"{item['root_id']} · {item['exit_reason']} · net {item['net']:+.2f} $, Δ к HOLD {item['delta']:+.2f} $, equity [{item['min_equity']:.2f}; {item['max_equity']:.2f}] $."
        sections.append('<details><summary>' + html.escape(caption) + '</summary>' + figure(item['file'], caption) + '</details>')
    sections.append('''</section><section id="limits"><h2>Что установлено и чего этот опыт не учил</h2>
<p>Установлены воспроизводимые парные цели и измеряемое поведение обученного менеджера на отложенных по времени датах. Успех модели определяется цифрами выше, а не фактом снижения train loss. Сохранённые оценки фиксированного плана не становятся Q повторяемой новой политики автоматически.</p>
<p>Этот опыт не обучал входам, направлению, размеру, ADD/REDUCE, мартингейлу, реакции на накопленный долг серии или непрерывной жизни счёта. Начальные условия не содержат таких вариантов. Он также не испытывает собственное воздействие заявок на исторический рынок. Пустая событийная память обрабатывается корректно, но отсутствующие события не выдуманы и сама событийная ветка здесь не обучалась.</p>
<p>Следующие выводы должны опираться на сравнение замороженных правил, новые поддержанные даты, проверку исполнения и отдельно расширенный диапазон состояний счёта. Положительная средняя разность к HOLD может означать только уменьшение убытка; положительный результат независимых опций не равен заработку одновременно торгуемого портфеля.</p>
<p class="links"><a href="training_protocol.json">Протокол обучения</a> · <a href="training_summary.json">Метрики моделей</a> · <a href="sequential_summary.json">Повторяемая политика</a> · <a href="behavior_slices.json">Срезы поведения</a> · <a href="evaluation_receipt.json">Квитанция replay</a> · <a href="report_receipt.json">Проверки отчёта</a></p></section>''')
    css = '''body{margin:0;background:#f3f5f7;color:#182633;font:16px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:30px}header{padding:25px 0}h1{font-size:42px;line-height:1.15;margin:10px 0 20px}h2{font-size:27px;line-height:1.3}p{max-width:1100px}.eyebrow{color:#4b6a7b;letter-spacing:.12em;font-size:13px;font-weight:700}.lede{font-size:21px;max-width:1050px}nav{display:flex;flex-wrap:wrap;gap:20px;padding:16px 0;border-top:1px solid #ccd4da;border-bottom:1px solid #ccd4da}a{color:#146b9b}section{background:white;padding:26px;margin:28px 0;border:1px solid #dde4e8;border-radius:12px}figure{margin:28px 0}img{max-width:100%;height:auto;display:block}figcaption{font-size:14px;color:#506273;margin-top:10px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.cards div{background:#edf5f9;padding:18px;border-radius:8px}.cards b{display:block;font-size:25px;line-height:1.2;margin-bottom:12px}.cards span{font-size:14px}.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid #e1e6e9;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#eef3f6}tr:hover{background:#f6fafc}.formula{font-size:21px;font-weight:650;background:#edf5f9;padding:20px}details{border-top:1px solid #dce3e8;padding:14px 0}summary{cursor:pointer;font-weight:600}.links{font-size:14px}@media(max-width:800px){main{padding:12px}section{padding:16px}h1{font-size:32px}.cards{grid-template-columns:repeat(2,1fr)}}'''
    document = '<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HOLD/CLOSE — обучение и поведение</title><style>' + css + '</style></head><body><main>' + '\n'.join(sections) + '</main></body></html>'
    (out / 'index.html').write_text(document, encoding='utf8')
    sources = {str(out / name): digest(out / name) for name in required}
    for name in ('behavior_audit.json', 'sequential_comparisons.json'):
        if (out / name).exists():
            sources[str(out / name)] = digest(out / name)
    for entry in summary:
        path = out / (entry['model'] + '_history.json')
        sources[str(path)] = digest(path)
    sources[str(data / 'manifest.json')] = digest(data / 'manifest.json')
    sources[str(Path(__file__))] = digest(Path(__file__))
    all_images = re.findall(r'<img src="([^"]+)"', document)
    for name in all_images:
        if not (out / name).is_file():
            raise ValueError('Broken image link: ' + name)
    for name in re.findall(r'<a href="([^"]+)"', document):
        if not name.startswith('#') and name != 'report_receipt.json' and not (out / name).is_file():
            raise ValueError('Broken local artifact link: ' + name)
    write_json(out / 'report_receipt.json', dict(status='COMPLETE',source_hashes=sources,
        checks=dict(completed_inputs=True, matching_data_identity=True, no_cross_split_roots=True,
                    causal_availability=True, local_image_links=len(all_images), deterministic_cases=len(images),
                    no_external_scripts=True, no_summed_option_equity=True),
        interpretation='Paired fixed-plan target and repeated-policy evaluation kept separate. All roots independently reset to $1500.',
        visual_qa='Static plots generated. Human/agent image inspection is separate from these automated checks.'))
    print(json.dumps(dict(report=str(out / 'index.html'),images=len(all_images),cases=len(images),main_mean_net=main['mean_net'],main_delta=main['mean_delta_vs_hold']), ensure_ascii=False))
    return out / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/derived/hold_close_v1')
    parser.add_argument('--output', default='reports/hold_close_v1')
    args = parser.parse_args()
    build(args.data, args.output)
