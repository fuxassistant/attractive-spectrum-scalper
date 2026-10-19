"""Self-contained mixed-memory gym audit, with exact clocks and fixed-sample plots.

The charts deliberately show the account *before* a decision at its timestamp.
Equity after a step can include later quote events and is not moved backwards
onto that decision. Routing weights are descriptive instrumentation, not an
explanation of a market cause or a calibrated probability of a profitable trade.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


BRANCHES = ('fast', 'slow', 'anchors', 'events')
COLORS = ('#4ca6ff', '#ffbd4a', '#b789ff', '#4bd1ad')
DECISION_COLUMNS = (
    't_ns', 'mid', 'action', 'equity_before', 'equity_after', 'intent', 'gate',
    'p_long', 'plan', 'can_enter', 'fast_count', 'slow_count', 'event_count',
    'branch_fast', 'branch_slow', 'branch_anchors', 'branch_events',
    'event_top1_ns', 'event_top1_weight', 'event_top1_class', 'policy_evaluated',
    'previous_M1_attention_mass',
    'series_floor', 'series_start_equity', 'series_id', 'series_loss_fraction',
    'series_active', 'risk_dead', 'series_net',
    'series_cumulative_net', 'series_status', 'liquidation_equity', 'completed_series',
    'series_trade_count', 'series_min_equity', 'max_series_loss_fraction', 'series_start_ns',
)
SERIES_FIELDS = ('series_floor', 'series_start_equity', 'series_id',
                 'series_loss_fraction', 'series_active', 'risk_dead')


def _clean(value, key=''):
    """JSON-safe objects; all nanosecond clocks remain decimal strings."""
    if isinstance(value, dict):
        return {str(k): _clean(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v, key) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return str(int(value)) if key.endswith('_ns') or key.endswith('_nano') else int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        # Clocks should enter as int64; a non-null float clock is rejected.
        if key.endswith('_ns') or key.endswith('_nano'):
            raise ValueError(f'Clock/price integer was converted to float: {key}')
        return float(value)
    if value is pd.NA or value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def _risk_contract(protocol, replays):
    """Read the declared risk contract; never invent a constant equity floor."""
    contract = protocol.get('contract', {})
    declaration = protocol.get('risk_contract', protocol.get('series_risk', protocol.get('risk', {})))
    if not isinstance(declaration, dict):
        declaration = {'description': declaration}
    version = declaration.get('version', protocol.get('risk_version', contract.get('risk_version', protocol.get('version', 'UNSPECIFIED'))))
    present = [all(name in replay['decisions'] for name in SERIES_FIELDS) for replay in replays]
    declared_series = 'series' in (str(version)+' '+json.dumps(declaration)).lower()
    mode = ('SERIES' if present and all(present) else 'MIXED_OR_INCOMPLETE_SERIES' if any(present) or declared_series else 'LEGACY')
    fraction = declaration.get('loss_limit_fraction', declaration.get('max_loss_fraction'))
    # If the declaration omits its numeric ratio, describe only the saved
    # ratio when it is constant. This is reporting, not an execution rule.
    ratios = []
    if fraction is None:
        for replay in replays:
            d = replay['decisions']
            if 'series_floor' not in d or 'series_start_equity' not in d:
                continue
            base, floor = _column(d, 'series_start_equity'), _column(d, 'series_floor')
            use = np.isfinite(base) & np.isfinite(floor) & (base > 0) & (floor > 0)
            ratios.extend((1-floor[use]/base[use]).tolist())
        if ratios and np.ptp(ratios) < 1e-9:
            fraction = float(np.median(ratios))
    comparison = declaration.get('comparison')
    if comparison is None and 'strictly more than' in str(declaration.get('death', '')).lower():
        comparison = '>'
    return dict(mode=mode, version=str(version), declaration=declaration,
                initial_capital=declaration.get('initial_capital', contract.get('initial_capital', protocol.get('initial_capital'))),
                loss_limit_fraction=fraction, comparison=comparison,
                replay_series_fields=present,
                legacy='Historical risk contract: series state is absent; no series floor is reconstructed.' if mode == 'LEGACY' else None)


def _read_csv_records(path):
    if not path.exists() or not path.stat().st_size:
        return []
    try:
        names = pd.read_csv(path, nrows=0).columns
        types = {name: 'string' for name in names if name.endswith('_ns') or name.endswith('_nano')}
        return [_clean(row) for row in pd.read_csv(path, dtype=types).to_dict('records')]
    except pd.errors.EmptyDataError:
        return []


def _normalize_risk_view(frame):
    """Normalize saved *before-step* aliases without changing source artifacts.

    Neural exports use unprefixed risk_snapshot names; martingale exports prefix
    them with risk_. Explicit *_after columns are never read as origin state.
    Ruin is a certified status, never inferred from a low diagnostic equity.
    """
    frame = frame.copy()
    for name in DECISION_COLUMNS:
        prefixed = 'risk_' + name
        if name not in frame and prefixed in frame:
            frame[name] = frame[prefixed]
    if 'series_status' in frame:
        frame['risk_dead'] = frame.series_status.eq('ruined')
    if 'series_cumulative_net' in frame:
        frame['series_net'] = frame.series_cumulative_net
    if 'equity_before' not in frame and 'liquidation_equity' in frame:
        frame['equity_before'] = frame.liquidation_equity
    if 'equity_after' not in frame and 'equity' in frame:
        # evaluate_policy documents this field as the after-step equity.
        frame['equity_after'] = frame.equity
    return frame


def _read_replay(root, path):
    frame = _normalize_risk_view(pd.read_parquet(path))
    if 't_ns' not in frame:
        raise ValueError(f'Missing t_ns in {path}')
    if not pd.api.types.is_integer_dtype(frame.t_ns.dtype):
        raise ValueError(f't_ns must be an integer column in {path}')
    if not frame.t_ns.is_monotonic_increasing:
        raise ValueError(f'Unordered decisions in {path}')
    columns = [name for name in DECISION_COLUMNS if name in frame]
    frame = frame.copy()
    if 'mid' in frame:
        # Zero is the source's missing-price sentinel, never a GC market print.
        frame.loc[frame.mid <= 0, 'mid'] = np.nan
    if 'policy_evaluated' in frame:
        for name in ('fast_count', 'slow_count', 'event_count'):
            if name in frame:
                frame.loc[~frame.policy_evaluated.astype(bool), name] = np.nan
    decisions = {name: _clean(frame[name].tolist(), name) for name in columns}
    trade_path = path.parent / 'trades.csv'
    trades = []
    if trade_path.exists() and trade_path.stat().st_size:
        for item in _read_csv_records(trade_path):
            # Use the saved execution prices where available. No future price
            # inference is needed for the trade markers.
            for prefix in ('entry', 'exit'):
                nano = item.get(prefix + '_nano')
                if nano is not None and nano is not pd.NA and str(nano) != '<NA>':
                    item[prefix + '_price'] = int(nano) / 1e9
            trades.append(_clean(item))
    meta = _read_json(path.parent / 'metrics.json', {})
    cycles = []
    for filename in ('series.csv', 'cycles.csv', 'series_cycles.csv'):
        if (path.parent / filename).exists():
            cycles = _read_csv_records(path.parent / filename)
            break
    return dict(key=path.parent.relative_to(root / 'replays').as_posix(),
                decisions=decisions, trades=trades, cycles=cycles, metrics=_clean(meta),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _column(decisions, name, fallback=np.nan):
    n = len(decisions.get('t_ns', []))
    return np.asarray(decisions.get(name, [fallback] * n), dtype=float)


def _series_column(decisions, name):
    values = _column(decisions, name)
    visible = (_column(decisions, 'series_active', 0) > 0) | (_column(decisions, 'risk_dead', 0) > 0)
    return np.where(visible & (values > 0), values, np.nan)


def _png_replay(replay, target, focus_active=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    d = replay['decisions']
    t = pd.to_datetime(np.asarray(d['t_ns'], dtype=np.int64), utc=True)
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.2, 1.5, 1.2, 1.2]})
    axes[0].plot(t, _column(d, 'mid'), color='#26374a', lw=.85, label='Mid at decision')
    for trade in replay['trades']:
        if not trade.get('entry_ns') or not trade.get('exit_ns'):
            continue
        times = pd.to_datetime([int(trade['entry_ns']), int(trade['exit_ns'])], utc=True)
        side = int(trade.get('side', 0))
        ep, xp = trade.get('entry_price'), trade.get('exit_price')
        if ep is None or xp is None:
            continue
        color = '#168458' if float(trade.get('net') or 0) > 0 else '#c03b43'
        axes[0].plot(times, [ep, xp], color=color, alpha=.65, lw=.7)
        axes[0].scatter(times[0], ep, marker='^' if side > 0 else 'v', color=color, s=24, zorder=4)
        axes[0].scatter(times[1], xp, marker='x', color=color, s=15, zorder=4)
    axes[0].set_ylabel('GC quote / oz')
    axes[0].set_title(replay['key'] + ' | Executed entries/exits; green net > 0, red net <= 0', loc='left')
    for name, color, label in (('intent', '#7e62bb', 'Raw participation intent'),
                               ('gate', '#217eae', 'Executable participation'),
                               ('p_long', '#c79528', 'Conditional P(LONG)')):
        if name in d:
            axes[1].plot(t, _column(d, name), color=color, label=label, lw=.8)
    axes[1].axhline(.5, color='#999', ls=':', lw=.7)
    axes[1].set_ylim(-.03, 1.03)
    axes[1].set_ylabel('Policy probability')
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(loc='upper left', ncol=3, fontsize=8)
    axes[2].plot(t, _column(d, 'equity_before'), color='#243d8b', label='Liquidation equity before decision')
    if all(name in d for name in SERIES_FIELDS):
        axes[2].step(t, _series_column(d, 'series_floor'), where='post', color='#c03b43', ls='--',
                     label='Saved series floor (strict breach; equality alive)')
        axes[2].step(t, _series_column(d, 'series_start_equity'), where='post', color='#a17b19', ls=':',
                     label='Equity before first series fill')
        risk_label = 'SERIES RISK | ' + replay.get('risk_version', 'version unspecified')
    else:
        risk_label = 'LEGACY RISK CONTRACT | series fields unavailable; no floor inferred'
    fig.suptitle(risk_label, fontsize=10, color='#864a34' if 'LEGACY' in risk_label else '#26374a')
    axes[2].set_ylabel('USD')
    axes[2].legend(loc='upper left', fontsize=8)
    for branch, color in zip(BRANCHES, COLORS):
        name = 'branch_' + branch
        if name in d:
            axes[3].plot(t, _column(d, name), color=color, label=branch, lw=.85)
    if 'previous_M1_attention_mass' in d:
        axes[3].plot(t, _column(d, 'previous_M1_attention_mass'), color='#cc6f93', ls='--',
                     label='Previous M1 attention mass', lw=.85)
    axes[3].set_ylim(-.03, 1.03)
    axes[3].set_ylabel('Router weights')
    if axes[3].lines:
        axes[3].legend(loc='upper left', ncol=4, fontsize=8)
    for name, color in zip(('fast_count', 'slow_count', 'event_count'), COLORS):
        if name in d:
            axes[4].plot(t, _column(d, name), color=color, label=name, lw=.85)
    axes[4].set_ylabel('Available rows')
    axes[4].set_xlabel('UTC; equity and memory align to the decision publication timestamp')
    if axes[4].lines:
        axes[4].legend(loc='upper left', ncol=3, fontsize=8)
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
    if focus_active and 'policy_evaluated' in d:
        known = np.flatnonzero(np.asarray(d['policy_evaluated'], dtype=bool))
        if len(known) > 1:
            axes[4].set_xlim(t[known[0]], t[known[-1]])
    for ax in axes:
        ax.grid(alpha=.18)
    fig.text(.01, .003, 'Router weights describe model computation, not causal importance. Probabilities are action preferences, not win probabilities.', fontsize=9)
    fig.tight_layout(rect=(0, .015, 1, .98))
    fig.savefig(target, dpi=135)
    plt.close(fig)


def _png_history(history, target, risk_contract=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    frame = pd.DataFrame(history)
    groups = [k for k in ('variant', 'seed') if k in frame]
    iterable = frame.groupby(groups, dropna=False) if groups else [('all', frame)]
    for key, rows in iterable:
        rows = rows.sort_values('epoch') if 'epoch' in rows else rows
        x = rows['epoch'] if 'epoch' in rows else np.arange(len(rows))
        label = str(key)
        for name in ('val_score', 'validation_score', 'validation_net'):
            if name in rows:
                axes[0].plot(x, pd.to_numeric(rows[name], errors='coerce'), label=label + ' / ' + name)
                break
        for name in ('val_equity', 'validation_equity', 'validation_final_equity'):
            if name in rows:
                axes[1].plot(x, pd.to_numeric(rows[name], errors='coerce'), label=label + ' / equity')
                break
    axes[0].set_ylabel('Validation selection metric')
    axes[1].set_ylabel('Validation equity, USD')
    axes[1].set_xlabel('Epoch')
    initial = (risk_contract or {}).get('initial_capital')
    if initial is not None:
        axes[1].axhline(initial, color='#aaa', ls=':', lw=.8, label='Declared initial capital')
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7, ncol=2)
    risk_label = ' | ' + str((risk_contract or {}).get('mode', 'UNSPECIFIED'))
    fig.suptitle('Validation curves; evaluation is not used for epoch selection' + risk_label)
    fig.tight_layout()
    fig.savefig(target, dpi=135)
    plt.close(fig)


def build_report(reportroot):
    """Build index.html plus up to four randomly sampled replay PNGs and epochs.

    Input: summary.json, optional protocol.json/history.csv, and
    replays/**/decisions.parquet with sibling trades.csv. Empty and WAIT-only
    replays are valid. Returns artifact paths and counts; writes ui_receipt.json.
    """
    root = Path(reportroot).resolve()
    root.mkdir(parents=True, exist_ok=True)
    replays = [_read_replay(root, p) for p in sorted((root / 'replays').rglob('decisions.parquet'))]
    summary = _read_json(root / 'summary.json', {})
    protocol = _read_json(root / 'protocol.json', _read_json(root / 'contract.json', {}))
    memory_usage = _read_json(root / 'memory_usage.json', {})
    risk_contract = _risk_contract(protocol, replays)
    for replay in replays:
        replay['risk_version'] = risk_contract['version']
        replay['risk_mode'] = 'SERIES' if all(name in replay['decisions'] for name in SERIES_FIELDS) else 'LEGACY'
    history = []
    if (root / 'history.csv').exists():
        try:
            history = _clean(pd.read_csv(root / 'history.csv').to_dict('records'))
        except pd.errors.EmptyDataError:
            pass
    payload = dict(summary=summary, protocol=protocol, history=history, replays=replays, memory_usage=memory_usage,
                   risk_contract=risk_contract,
                   generated_utc=datetime.now(timezone.utc).isoformat())
    encoded = json.dumps(_clean(payload), ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    encoded = encoded.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = root / 'index.html'
    html.write_text(HTML.replace('__AUDIT_DATA__', encoded), encoding='utf-8', newline='\n')
    figures = []
    eligible = [r for r in replays if r['decisions'].get('t_ns')]
    rng = np.random.default_rng(61018)
    chosen = sorted(rng.choice(len(eligible), size=min(4, len(eligible)), replace=False).tolist()) if eligible else []
    samples = []
    for j, i in enumerate(chosen):
        name = f'audit_replay_{j + 1:02d}.png'
        _png_replay(eligible[i], root / name)
        figures.append(str(root / name))
        samples.append(dict(replay=eligible[i]['key'], figure=name))
    # Deliberate companion views supplement, never replace, the random sample.
    # They expose actual exploratory trades alongside the selected WAIT policy.
    for label, filename in (('mixed_full_s61018_sample101', 'audit_mixed_stochastic.png'),
                            ('mixed_full_s61018', 'audit_mixed_greedy.png')):
        candidates = [r for r in eligible if r['key'].split('/')[0] == label]
        if candidates:
            selected = sorted(candidates, key=lambda r: r['key'])[0]
            _png_replay(selected, root / filename, focus_active=True)
            figures.append(str(root / filename))
    if history:
        path = root / 'audit_epochs.png'
        _png_history(history, path, risk_contract)
        figures.append(str(path))
    receipt = dict(html=str(html), figures=figures, replay_count=len(replays),
                   decision_count=sum(len(r['decisions'].get('t_ns', [])) for r in replays),
                   trade_count=sum(len(r['trades']) for r in replays),
                   figure_sampling=dict(seed=61018, method='uniform_without_replacement', samples=samples),
                   clock_contract='nanosecond integers serialized as decimal strings; JS BigInt before milliseconds',
                   equity_contract='equity_before is plotted at decision time; equity_after is not backdated',
                   risk_contract=risk_contract,
                   risk_visualization='Saved per-origin series floor only; no fixed floor inferred for legacy replay.',
                   html_sha256=hashlib.sha256(html.read_bytes()).hexdigest())
    (root / 'ui_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return receipt


HTML = r'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Смешанная память · аудит игрока</title>
<style>
:root{color-scheme:dark;--bg:#0c1420;--panel:#132032;--line:#294059;--text:#e6edf5;--muted:#a4b6ca}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:27px;margin:0 0 9px}h2{font-size:18px}p{line-height:1.5;color:var(--muted)}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0}.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}select,button{padding:9px 12px;background:#1c3048;border:1px solid #46617e;border-radius:6px;color:var(--text);max-width:100%}label{display:flex;gap:8px;align-items:center}.chips{display:flex;gap:9px;flex-wrap:wrap;margin:14px 0}.chip{padding:8px 11px;background:#1c3048;border-radius:7px}.muted{color:var(--muted)}canvas{width:100%;display:block}#chart{height:900px}#epochs{height:340px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid var(--line);padding:9px;text-align:left;white-space:nowrap}th{color:#b2c8e0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px ui-monospace,monospace;line-height:1.5}details{margin:12px 0}summary{cursor:pointer}.good{color:#6bddaa}.bad{color:#ff8d93}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px}.swatch{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px}#hover{min-height:43px;font-family:ui-monospace,monospace;font-size:12px;line-height:1.5}.note{border-left:3px solid #e6b65d;padding-left:12px}.section-title{margin:3px 0 14px}.empty{padding:32px;text-align:center;color:var(--muted)}a{color:#80bbff}
</style><body><main>
<h1>Память рынка и поведение игрока</h1>
<p>Последние 15 минут по секундам · вчера и сегодня в M1 · открытия · банк событий. Цена, решение, счёт и доступная память показаны на одной временной оси.</p>
<div class="panel"><h2 class="section-title">Сравнение зафиксированных политик</h2><div id="summaryTable" class="scroll"></div><p class="note" id="riskContract"></p><p class="muted">Размер и комиссии берутся из сохранённых сделок. Унция на котировках GC — условная модель исполнения, а не фьючерсный контракт с биржевой маржой.</p><details><summary>Точные результаты и ограничения</summary><pre id="summaryJson"></pre></details></div>
<div class="panel"><div class="controls"><label>Модель / проверка <select id="model"></select></label><label>Сессия <select id="day"></select></label><button id="reset">Вся сессия</button><label>Масштаб <select id="zoom"><option value="1">Вся сессия</option><option value="2">½ сессии</option><option value="4">¼ сессии</option><option value="12">Около часа</option></select></label></div><p class="note" id="replayRisk"></p><div class="chips" id="chips"></div><div class="legend" id="legend"></div><div id="hover" class="muted">Наведите курсор на график. Колесо мыши приближает выбранное время; Shift + колесо перемещает окно.</div><canvas id="chart"></canvas><p class="note">Intent — намерение участвовать; gate — вероятность участия с учётом разрешённых действий; P(LONG) — выбор направления при участии. Это предпочтения политики, а не вероятность прибыли. Веса ветвей описывают вычисление модели; об использовании памяти судят также по отключению ветвей у зафиксированной модели. Счёт показан <b>до</b> решения: последующий исход не перенесён назад во времени.</p><details><summary>Сделки выбранной сессии</summary><div class="scroll" id="trades"></div></details><details><summary>Сохранённые циклы выбранной сессии</summary><div class="scroll" id="cycles"></div></details></div>
<div class="panel"><h2 class="section-title">Обучение по эпохам</h2><div class="legend" id="epochLegend"></div><canvas id="epochs"></canvas><p>На графике — сохранённая validation-метрика выбора эпохи. Финальная проверка не должна использоваться для подбора эпохи или победившего seed.</p><details><summary>Таблица эпох</summary><div class="scroll" id="history"></div></details></div>
<div class="panel"><h2 class="section-title">Контроль памяти и martingale</h2><h3>Парные отключения при одинаковом счёте</h3><p>До 16 legal origins на день. Previous M1 отключает только минутную последовательность вчера; прошлые открытия и события сохраняются. Изменения вероятности приведены в процентных пунктах.</p><div id="pairedTable" class="scroll"></div><h3>Полный replay с отключением ветвей</h3><p>Здесь решения меняют последующую историю счёта. Эти результаты дополняют парные пробы и не заменяют их.</p><div id="ablationTable" class="scroll"></div><p id="memoryReceipt"></p><details><summary>Протокол, интервалы и ограничения</summary><pre id="protocol"></pre></details><p id="stamp"></p></div>
</main><script id="audit-data" type="application/json">__AUDIT_DATA__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('audit-data').textContent), $=id=>document.getElementById(id);
const C=['#4ca6ff','#ffbd4a','#b789ff','#4bd1ad','#ff7e89','#b5d06b','#e898ca','#d6e2f0'];
const fmt=(x,n=2)=>x==null?'—':typeof x==='number'?(Number.isFinite(x)?x.toFixed(n):'—'):String(x);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ms=s=>Number(BigInt(String(s))/1000000n), utc=s=>new Date(ms(s)).toISOString().replace('T',' ').replace('.000Z',' UTC');
const isNum=x=>x!==null&&x!==undefined&&Number.isFinite(Number(x));
const risk=D.risk_contract||{mode:'LEGACY',version:'UNSPECIFIED'},riskLimit=isNum(risk.loss_limit_fraction)?fmt(risk.loss_limit_fraction*100,2)+'%':'по сохранённому контракту';
$('riskContract').textContent=risk.mode==='LEGACY'?'LEGACY — исторический контракт риска. В этих данных отсутствует состояние серии; результат не соответствует текущему правилу контроля одной серии. Граница серии не реконструируется.':'Контракт '+risk.version+'. Начальный капитал: '+(isNum(risk.initial_capital)?'$'+fmt(risk.initial_capital):'см. протокол')+'. База и граница каждой серии взяты из сохранённого replay. Условие потери: '+(risk.comparison||'см. протокол')+' '+riskLimit+'. '+(risk.comparison==='>'?'Равенство порогу оставляет серию живой. ':'')+'Новая база включает накопленный заработок; неполный прибыльный откат не означает восстановление серии.';
function plotValue(d,key,i){const v=d[key]?.[i];if(key==='series_floor'||key==='series_start_equity')return (d.series_active?.[i]||d.risk_dead?.[i])&&isNum(v)&&Number(v)>0?v:null;return v;}
function seriesStatus(d,i){const state=d.series_status?.[i]||(d.risk_dead?.[i]?'ruined':d.series_active?.[i]?'active':'none');return ({none:'ещё не начата',active:'активна',recovered:'восстановлена',ruined:'превышен лимит',censored:'цензурирована; исход неизвестен'})[state]||state;}
function seriesNet(d,i){if(!(Number(d.series_id?.[i])>0))return null;const saved=d.series_cumulative_net?.[i]??d.series_net?.[i];if(isNum(saved))return saved;const base=d.series_start_equity?.[i],eq=d.liquidation_equity?.[i]??d.equity_before?.[i];return isNum(base)&&Number(base)>0&&isNum(eq)?Number(eq)-Number(base):null;}
function seriesInfo(d,i){if(!d.series_floor)return 'LEGACY: состояние серии отсутствует';return 'серия '+fmt(d.series_id?.[i],0)+' / '+seriesStatus(d,i)+' | база $'+fmt(d.series_start_equity?.[i])+' | floor $'+fmt(d.series_floor?.[i])+' | потеря '+fmt(Number(d.series_loss_fraction?.[i])*100,3)+'% | net серии $'+fmt(seriesNet(d,i));}
function cycleRows(cycles){const when=t=>t&&String(t)!=='0'?utc(t):null;return cycles.map(c=>({'Серия':c.series_id,'Первый fill UTC':when(c.series_start_ns??c.start_ns),'Окончание UTC':when(c.end_ns),'База':c.series_start_equity??c.start_equity,'Floor':c.series_floor??c.floor,'Статус':c.series_status??c.status,'Net серии':c.series_cumulative_net??c.net,'Сделок':c.series_trade_count??c.trades,'Минимум equity':c.series_min_equity??c.min_equity}));}

function table(rows,container){if(!rows.length){container.innerHTML='<div class="empty">Нет сохранённых строк.</div>';return;}const keys=[...new Set(rows.flatMap(Object.keys))];container.innerHTML='<table><thead><tr>'+keys.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+keys.map(k=>'<td>'+esc(typeof r[k]==='object'?JSON.stringify(r[k]):fmt(r[k],k.includes('п.п.')?6:3))+'</td>').join('')+'</tr>').join('')+'</tbody></table>';}
function flattened(rows){return rows.map(r=>{const out={};for(const[k,v]of Object.entries(r)){if(v&&typeof v==='object'&&!Array.isArray(v)){for(const[a,b]of Object.entries(v))if(typeof b!=='object')out[k+'.'+a]=b;}else if(!Array.isArray(v))out[k]=v;}return out;});}
const models=Array.isArray(D.summary)?D.summary:D.summary.models||D.summary.results||[];
const modelRows=Array.isArray(models)?models:Object.entries(models).map(([variant,v])=>({variant,...v}));table(modelRows.map(m=>({'Модель':m.variant,'Seed':m.seed,'Эпоха':m.selected_epoch,'Параметров':m.parameters,'Equity, $':m.evaluation?.final_equity,'Сделок':m.evaluation?.trades,'Score':m.evaluation?.score,'Полный replay':m.evaluation?.complete?'да':'нет'})),$('summaryTable'));
$('summaryJson').textContent=JSON.stringify(D.summary,null,2);$('protocol').textContent=JSON.stringify(D.protocol,null,2);$('stamp').textContent='Создано: '+D.generated_utc+' · Самодостаточный файл: сетевые запросы не нужны.';
const ablations=modelRows.flatMap(m=>Object.entries(m.ablations||{}).map(([branch,v])=>({'Модель / seed':m.variant+' / '+m.seed,'Отключение':branch,'Equity, $':v.final_equity,'Сделок':v.trades,'Score':v.score,'Полный replay':v.complete?'да':'нет'}))),mart=D.summary.martingale||{};table(ablations,$('ablationTable'));const paired=D.memory_usage?.paired_ablations||D.summary.paired_memory_ablations||[];table(paired.map(a=>({'Модель / отключение':a.variant+' / '+a.seed+' / '+a.ablation,'Проб':a.probes,'Сред. |Δ intent|, п.п.':a.mean_abs_intent_delta*100,'Макс. |Δ intent|, п.п.':a.max_abs_intent_delta*100,'Смена действия, %':a.action_change_fraction*100})),$('pairedTable'));$('memoryReceipt').textContent='Проверено неизменных decision timestamps: '+(D.memory_usage?.causal_stable_timestamps_count??'—')+'; нарушений доступности будущей памяти: '+(D.memory_usage?.causal_future_violations??'—')+'. Веса attention/router сами по себе не доказывают использование памяти для прибыльного решения.';
if(Object.keys(mart).length){const details=document.createElement('details'),head=document.createElement('summary'),pre=document.createElement('pre');head.textContent='Martingale: сохранённый подбор и проверка';pre.textContent=JSON.stringify(mart,null,2);details.append(head,pre);$('ablationTable').after(details);}
table(D.history,$('history'));
const replayModels=[...new Set(D.replays.map(r=>r.key.split('/').slice(0,-1).join('/')||r.key))];
$('model').innerHTML=replayModels.map(k=>'<option value="'+esc(k)+'">'+esc(k)+(k.includes('sample101')?' · stochastic trial':k.includes('epoch')?' · diagnostic epoch':' · greedy replay')+'</option>').join('');if(replayModels.includes('mixed_full_s61018_sample101'))$('model').value='mixed_full_s61018_sample101';
let replay=null,range=[0,1],hoverIndex=-1,chartGeometry=null;
function resizeCanvas(canvas){const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));const c=canvas.getContext('2d');c.setTransform(ratio,0,0,ratio,0,0);return[c,rect.width,rect.height];}
function series(c,xx,values,px,py,color,width=1.2,step=false){c.strokeStyle=color;c.lineWidth=width;c.beginPath();let active=false,lastY=0;for(let i=0;i<xx.length;i++){if(!isNum(values[i])){active=false;continue;}const x=px(xx[i]),y=py(Number(values[i]));if(active){if(step)c.lineTo(x,lastY);c.lineTo(x,y);}else{c.moveTo(x,y);active=true;}lastY=y;}c.stroke();}
function bounds(arr,forced){if(forced)return forced;let lo=Infinity,hi=-Infinity;for(const x of arr)if(isNum(x)){lo=Math.min(lo,Number(x));hi=Math.max(hi,Number(x));}if(!Number.isFinite(lo))return[0,1];const pad=(hi-lo)*.08||Math.max(.1,Math.abs(hi)*.0001);return[lo-pad,hi+pad];}
function draw(){const[c,w,h]=resizeCanvas($('chart'));c.fillStyle='#132032';c.fillRect(0,0,w,h);if(!replay||!replay.decisions.t_ns.length){c.fillStyle='#a4b6ca';c.fillText('Нет сохранённого replay.',20,40);return;}const d=replay.decisions,all=d.t_ns.map(ms),start=all[0],span=Math.max(1,all.at(-1)-start),minT=start+span*range[0],maxT=start+span*range[1],indices=all.map((v,i)=>i).filter(i=>all[i]>=minT&&all[i]<=maxT),xx=indices.map(i=>all[i]);const left=74,right=w-20,top=12,gap=30,heights=[230,130,140,115,100],names=['Цена GC / oz · ▲ LONG / ▼ SHORT · × выход','Участие и направление','Equity до решения, USD','Веса ветвей памяти','Доступные строки памяти'];let y=top;const px=t=>left+(t-minT)/(maxT-minT)*(right-left);chartGeometry={left,right,minT,maxT};
const defs=[[['mid','#dbe7f4']], [['intent','#b789ff'],['gate','#4ca6ff'],['p_long','#ffbd4a']], [['equity_before','#78a9ff'],['series_floor','#ff7e89'],['series_start_equity','#e5b865']], [['branch_fast',C[0]],['branch_slow',C[1]],['branch_anchors',C[2]],['branch_events',C[3]],['previous_M1_attention_mass','#e898ca']], [['fast_count',C[0]],['slow_count',C[1]],['event_count',C[3]]]];
for(let k=0;k<heights.length;k++){const ht=heights[k],bot=y+ht,vals=defs[k].flatMap(([key])=>indices.map(i=>plotValue(d,key,i)));const[lo,hi]=bounds(vals,k===1||k===3?[-.03,1.03]:null),py=v=>bot-12-(v-lo)/(hi-lo)*(ht-34);c.fillStyle='#a4b6ca';c.font='12px system-ui';c.fillText(names[k],left,y+12);for(let j=0;j<=3;j++){const v=lo+(hi-lo)*j/3,yy=py(v);c.strokeStyle='#263a50';c.lineWidth=.6;c.beginPath();c.moveTo(left,yy);c.lineTo(right,yy);c.stroke();c.fillText(fmt(v,k===0?1:k===4?0:2),5,yy+4);}c.save();c.beginPath();c.rect(left,y+18,right-left,ht-20);c.clip();for(const[key,color]of defs[k])if(d[key])series(c,xx,indices.map(i=>plotValue(d,key,i)),px,py,color,1.2,key==='series_floor'||key==='series_start_equity');
if(k===0)for(const tr of replay.trades){if(!tr.entry_ns||!tr.exit_ns)continue;const et=ms(tr.entry_ns),xt=ms(tr.exit_ns);if(xt<minT||et>maxT)continue;const nearest=t=>{let best=0;while(best+1<all.length&&all[best+1]<=t)best++;return d.mid?.[best];};const ep=tr.entry_price??nearest(et),xp=tr.exit_price??nearest(xt);if(!isNum(ep)||!isNum(xp))continue;const col=Number(tr.net)>0?'#5cd59e':'#ff7e89';c.strokeStyle=col;c.lineWidth=.8;c.beginPath();c.moveTo(px(et),py(ep));c.lineTo(px(xt),py(xp));c.stroke();c.fillStyle=col;const x=px(et),yy=py(ep),sign=Number(tr.side)>0?1:-1;c.beginPath();c.moveTo(x,yy-sign*5);c.lineTo(x-4,yy+sign*4);c.lineTo(x+4,yy+sign*4);c.closePath();c.fill();const ex=px(xt),ey=py(xp);c.beginPath();c.moveTo(ex-3,ey-3);c.lineTo(ex+3,ey+3);c.moveTo(ex-3,ey+3);c.lineTo(ex+3,ey-3);c.stroke();}
if(k===1){const line=.5;c.strokeStyle='#8091a8';c.setLineDash([5,4]);c.beginPath();c.moveTo(left,py(line));c.lineTo(right,py(line));c.stroke();c.setLineDash([]);}if(hoverIndex>=0){const x=px(all[hoverIndex]);c.strokeStyle='#fff8';c.setLineDash([3,4]);c.beginPath();c.moveTo(x,y+20);c.lineTo(x,bot);c.stroke();c.setLineDash([]);}c.restore();if(k===heights.length-1){for(let j=0;j<=5;j++){const t=minT+(maxT-minT)*j/5;c.fillStyle='#a4b6ca';c.fillText(new Date(t).toISOString().slice(11,16),left+(right-left)*j/5-14,bot+20);}}y=bot+gap;}}
function selectReplay(){const key=$('day').value;replay=D.replays.find(r=>r.key===key)||null;range=[0,1];hoverIndex=-1;$('zoom').value='1';if(!replay){draw();return;}const d=replay.decisions,n=d.t_ns.length,net=replay.trades.reduce((s,t)=>s+Number(t.net||0),0),last=d.equity_after?.at(-1)??d.equity_before?.at(-1);$('replayRisk').textContent=replay.risk_mode==='SERIES'?'Серия контролируется относительно equity перед первым fill. Floor показан по состоянию до решения; завершение требует восстановления базы без открытой позиции. Частичный выигрыш не сбрасывает серию.':'LEGACY — исторический контракт риска: серийные поля отсутствуют. Этот replay не пересчитан по текущему правилу.';$('chips').innerHTML=[['Режим',replay.key.includes('sample101')?'Stochastic trial 101':replay.key.includes('epoch')?'Diagnostic epoch':'Greedy policy'],['Состояние',replay.metrics.complete===false?'CENSORED: market support':(replay.metrics.risk_dead||d.risk_dead?.at(-1))?'Серия превысила лимит':replay.metrics.ruined?'LEGACY: остановка старого контракта':'Завершено'],['Решений',n],['Сделок',replay.trades.length],['Net сделок, $',fmt(net)],['Конечный equity, $',fmt(last)],['Первый origin',n?utc(d.t_ns[0]):'—'],...(d.series_floor?[['Серия / последний origin',Number(d.series_id?.at(-1))>0?fmt(d.series_id?.at(-1),0):'ещё не начата'],['Серия / статус',seriesStatus(d,n-1)],['База серии, $',fmt(Number(d.series_start_equity?.at(-1))>0?d.series_start_equity.at(-1):null)],['Floor серии, $',fmt(Number(d.series_floor?.at(-1))>0?d.series_floor.at(-1):null)],['Потеря серии, %',fmt(Number(d.series_start_equity?.at(-1))>0?Number(d.series_loss_fraction?.at(-1))*100:null,3)],['Net серии на origin, $',fmt(seriesNet(d,n-1))]]:[])].map(([k,v])=>'<span class="chip">'+esc(k)+': <b>'+esc(v)+'</b></span>').join('');$('legend').innerHTML=[['Секунды',C[0]],['M1',C[1]],['Открытия',C[2]],['События',C[3]],['Attention к вчерашнему M1','#e898ca'],['Floor серии','#ff7e89'],['База серии','#e5b865']].filter((item,i)=>i<4||(i===4?d.previous_M1_attention_mass:d.series_floor)).map(([s,col])=>'<span><i class="swatch" style="background:'+col+'"></i>'+s+'</span>').join('');table(replay.trades.map(t=>({entry_UTC:t.entry_ns?utc(t.entry_ns):null,exit_UTC:t.exit_ns?utc(t.exit_ns):null,side:Number(t.side)>0?'LONG':'SHORT',size_oz:t.size_oz,plan:t.plan,entry:t.entry_price,exit:t.exit_price,net:t.net,reason:t.reason,censored:t.censored,series_id:t.series_id,series_status:t.series_status,series_base:t.series_start_equity,series_net:t.series_cumulative_net??t.series_net})),$('trades'));table(cycleRows(replay.cycles||[]),$('cycles'));draw();}
function selectModel(){const key=$('model').value;const rows=D.replays.filter(r=>(r.key.split('/').slice(0,-1).join('/')||r.key)===key);$('day').innerHTML=rows.map(r=>'<option value="'+esc(r.key)+'">'+esc(r.key.split('/').at(-1))+'</option>').join('');selectReplay();}
$('model').onchange=selectModel;$('day').onchange=selectReplay;$('reset').onclick=()=>{range=[0,1];$('zoom').value='1';draw();};$('zoom').onchange=()=>{const width=1/Number($('zoom').value),center=(range[0]+range[1])/2,start=Math.max(0,Math.min(1-width,center-width/2));range=[start,start+width];draw();};
$('chart').addEventListener('mousemove',ev=>{if(!replay||!chartGeometry)return;const x=ev.offsetX,g=chartGeometry,t=g.minT+(x-g.left)/(g.right-g.left)*(g.maxT-g.minT),d=replay.decisions,ts=d.t_ns.map(ms);let best=0;for(let i=1;i<ts.length;i++)if(Math.abs(ts[i]-t)<Math.abs(ts[best]-t))best=i;hoverIndex=best;$('hover').textContent=utc(d.t_ns[best])+' | price '+fmt(d.mid?.[best])+' | intent '+fmt(d.intent?.[best],3)+' | gate '+fmt(d.gate?.[best],3)+' | P(long) '+fmt(d.p_long?.[best],3)+' | action '+(d.action?.[best]??'—')+' | equity before $'+fmt(d.equity_before?.[best])+' | '+seriesInfo(d,best)+' | memory '+[d.fast_count?.[best],d.slow_count?.[best],d.event_count?.[best]].map(v=>fmt(v,0)).join(' / ')+(d.event_top1_ns?.[best]&&d.event_top1_ns[best]!=='0'?' | event focus '+utc(d.event_top1_ns[best])+' '+(d.event_top1_class?.[best]??'')+' weight '+fmt(d.event_top1_weight?.[best],3):'');draw();});
$('chart').addEventListener('mouseleave',()=>{hoverIndex=-1;draw();});
$('chart').addEventListener('wheel',ev=>{if(!replay||!chartGeometry)return;ev.preventDefault();const width=range[1]-range[0],g=chartGeometry,a=Math.max(0,Math.min(1,(ev.offsetX-g.left)/(g.right-g.left)));if(ev.shiftKey){const start=Math.max(0,Math.min(1-width,range[0]+Math.sign(ev.deltaY)*width*.15));range=[start,start+width];}else{const nw=Math.max(.015,Math.min(1,width*(ev.deltaY>0?1.2:1/1.2))),center=range[0]+a*width,start=Math.max(0,Math.min(1-nw,center-a*nw));range=[start,start+nw];}draw();},{passive:false});
function drawEpochs(){const[c,w,h]=resizeCanvas($('epochs'));c.fillStyle='#132032';c.fillRect(0,0,w,h);const groups={};for(const r of D.history){const key=(r.variant??'model')+' / '+(r.seed??'');(groups[key]??=[]).push(r);}const all=D.history.map(r=>r.val_score??r.validation_score??r.validation_net).filter(isNum),[lo,hi]=bounds(all),epochs=D.history.map(r=>Number(r.epoch||0)),max=Math.max(1,...epochs),left=65,right=w-20,top=20,bot=h-35,px=e=>left+e/max*(right-left),py=v=>bot-(v-lo)/(hi-lo)*(bot-top);c.font='12px system-ui';for(let j=0;j<=4;j++){const v=lo+(hi-lo)*j/4;c.strokeStyle='#294059';c.beginPath();c.moveTo(left,py(v));c.lineTo(right,py(v));c.stroke();c.fillStyle='#a4b6ca';c.fillText(fmt(v,3),2,py(v)+4);}for(let j=0;j<=5;j++){const e=max*j/5;c.fillText(fmt(e,0),px(e)-4,bot+22);}Object.entries(groups).forEach(([key,rows],i)=>{rows.sort((a,b)=>a.epoch-b.epoch);series(c,rows.map(r=>r.epoch),rows.map(r=>r.val_score??r.validation_score??r.validation_net),px,py,C[i%C.length],1.4);});$('epochLegend').innerHTML=Object.keys(groups).map((key,i)=>'<span><i class="swatch" style="background:'+C[i%C.length]+'"></i>'+esc(key)+'</span>').join('');if(!all.length){c.fillStyle='#a4b6ca';c.fillText('Нет сохранённой validation-метрики по эпохам.',left,40);}}
window.addEventListener('resize',()=>{draw();drawEpochs();});selectModel();drawEpochs();
</script></body></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reportroot', type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.reportroot), ensure_ascii=False, indent=2))
