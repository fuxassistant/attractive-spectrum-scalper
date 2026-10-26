"""Readability atlas and frozen chronological distinctness probes."""
import sys, json, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from spectrum.market_repr.representation import NS, TICK, config, MarketObservationStore
from spectrum.market_repr.zones import MECHANISMS
from spectrum.market_repr.probes import query_tables, fit_probes, BASE, BOOK
from scripts.build_market_representation_v1 import ROOT, REPORT
from scripts.build_crossing_path_target_v6 import save, sha

LABELS = {'REFILL_UNDER_PRESSURE': 'Малая реакция + восполнение', 'EXECUTED_DEPLETION': 'Движение с исполнениями',
          'WITHDRAWAL': 'Движение со снятием объёма', 'AGGRESSION_DECAY': 'Затухание агрессии'}
COLORS = dict(mid='#152e45', buy='#2578ad', sell='#cf644b', vwap='#b58b13', refill='#158d70')


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def load():
    manifest = json.loads((ROOT / 'manifest.json').read_text()); assert manifest['status'] == 'COMPLETE'
    cfg = config(); out = ROOT / 'observations'
    fast = pd.concat([pd.read_parquet(out / (d + '_fast.parquet')) for d in manifest['dates']], ignore_index=True)
    dates = [r['date'] for r in manifest['inventory'] if r['status'] == 'MATERIALIZED']
    contexts = {d: pd.read_parquet(out / (d + '_context.parquet')) for d in dates}
    zones = {d: pd.read_parquet(out / (d + '_zones.parquet')) for d in dates}
    events = {d: pd.read_parquet(out / (d + '_events.parquet')) for d in dates}
    return manifest, cfg, fast, contexts, zones, events


def plot_scene(scene, fast, context, zones, future=False):
    t = int(scene['origin_ns']); end = t + (120 * NS if future else 0)
    local = fast[(fast.t_ns >= t - 900 * NS) & (fast.available_ns <= end)].copy()
    day = context[context.available_ns <= t].copy()
    x = pd.to_datetime(local.t_ns, utc=True); dx = pd.to_datetime(day.t_ns, utc=True)
    origin = pd.Timestamp(t, tz='UTC')
    fig, axes = plt.subplots(5, 1, figsize=(13.8, 12.2), gridspec_kw={'height_ratios': [1.05, 1.55, 1, 1, 1]}, layout='constrained')
    fig.patch.set_facecolor('#f7f9fb')
    for ax in axes: ax.grid(alpha=.15); ax.set_facecolor('white'); ax.spines[['top', 'right']].set_visible(False)
    title = LABELS.get(scene['mechanism'], scene['mechanism'])
    mode = (f"БУДУЩЕЕ РАСКРЫТО · Δ120s {scene['future']['delta_ticks']:+.1f} ticks · {scene['future']['relative_outcome']}" if future else 'ТОЛЬКО ДОСТУПНОЕ ПРОШЛОЕ')
    if future and scene['future'].get('selected_model_error'): title = 'Уверенная ошибка probe'
    fig.suptitle(f"{scene['id']} · {title} · {origin.strftime('%d.%m %H:%M:%S')} UTC\n{mode}", fontsize=14, weight='bold', color='#152e45')
    axes[0].plot(dx, day.close.where(day.known), color=COLORS['mid'], lw=1, label='Midpoint, M1 all-event close')
    axes[0].plot(dx, day.combined_vwap, color=COLORS['vwap'], lw=1.3, label='Trade VWAP от вчерашнего London 08:00')
    axes[0].fill_between(dx, (day.combined_vwap - 2 * day.combined_sigma).to_numpy(),
                         (day.combined_vwap + 2 * day.combined_sigma).to_numpy(), color=COLORS['vwap'], alpha=.1)
    axes[0].set_ylabel('Контекст\n$/oz'); axes[0].legend(fontsize=8, loc='upper left', ncol=2)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M', tz='UTC'))
    axes[1].plot(x, local.mid.where(local.known), color=COLORS['mid'], lw=1.3, label='Midpoint, 1s')
    cm = context[(context.t_ns >= t - 900 * NS) & (context.available_ns <= t)]
    mx = pd.to_datetime(cm.available_ns, utc=True)
    price_lo, price_hi = local.mid.where(local.known).min(), local.mid.where(local.known).max()
    pad = max(.4, (price_hi - price_lo) * .12)
    axes[1].set_ylim(price_lo - pad, price_hi + pad)
    if cm.day_vwap.between(price_lo - pad, price_hi + pad).any():
        axes[1].plot(mx, cm.day_vwap, color=COLORS['vwap'], lw=1.3, label='Текущий trade VWAP')
    elif cm.day_vwap.notna().any():
        axes[1].text(.99, .96, f"Текущий VWAP {cm.day_vwap.dropna().iloc[-1]:.2f} $/oz (вне локального масштаба)",
                     transform=axes[1].transAxes, ha='right', va='top', fontsize=8, color='#947010')
    axes[1].plot(mx, cm.bb1 + cm.bb1_half, color='#778db0', lw=.8, ls='--', label='BB M1, SMA20 ±2σ')
    axes[1].plot(mx, cm.bb1 - cm.bb1_half, color='#778db0', lw=.8, ls='--')
    current_zones = zones[zones.available_ns <= t].groupby('zone_id').tail(1)
    near = current_zones[~current_zones.is_control & (current_zones.price - scene['past']['mid']).abs().lt(5)]
    for z in near.itertuples():
        axes[1].axhline(z.price, color='#ac8d9e', lw=.7, ls=':', label=z.kind)
    axes[1].set_ylabel('Локальная цена\n$/oz'); axes[1].legend(fontsize=7, loc='upper left', ncol=3)
    axes[2].plot(x, local.trade_buy30, color=COLORS['buy'], label='Buy T volume / 30s')
    axes[2].plot(x, local.trade_sell30, color=COLORS['sell'], label='Sell T volume / 30s')
    axes[2].plot(x, local.trade_unknown30, color='#999', ls=':', label='Unknown side')
    axes[2].set_ylabel('Агрессия\nконтрактов / 30s'); axes[2].legend(fontsize=8, ncol=3, loc='upper left')
    axes[3].plot(x, local.bid_near_fill30, color=COLORS['buy'], label='Bid: linked execution')
    axes[3].plot(x, local.ask_near_fill30, color=COLORS['sell'], label='Ask: linked execution')
    axes[3].plot(x, local.bid_near_refill30, color=COLORS['buy'], ls='--', label='Bid: refill ≤2s')
    axes[3].plot(x, local.ask_near_refill30, color=COLORS['sell'], ls='--', label='Ask: refill ≤2s')
    axes[3].set_ylabel('Исполнение / refill\nконтрактов / 30s'); axes[3].legend(fontsize=8, ncol=2, loc='upper left')
    axes[4].plot(x, local.bid_near_withdrawal30, color=COLORS['buy'], label='Bid: unlinked removal / 30s')
    axes[4].plot(x, local.ask_near_withdrawal30, color=COLORS['sell'], label='Ask: unlinked removal / 30s')
    depth = axes[4].twinx()
    depth.plot(x, local.depth_bid_12, color=COLORS['buy'], alpha=.35, lw=.65, label='Bid depth')
    depth.plot(x, local.depth_ask_12, color=COLORS['sell'], alpha=.35, lw=.65, label='Ask depth')
    depth.set_ylabel('Глубина 12 ticks\nполупрозрачные линии', fontsize=8, color='#777')
    axes[4].set_ylabel('Снятия proxy\nконтрактов / 30s'); axes[4].legend(fontsize=8, ncol=2, loc='upper left')
    for ax in axes[1:]:
        ax.axvline(origin, color='#222', ls='--', lw=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz='UTC'))
        ax.set_xlim(pd.Timestamp(t - 900 * NS, tz='UTC'), pd.Timestamp(end + (3 if not future else 0) * NS, tz='UTC'))
        if future: ax.axvspan(origin, pd.Timestamp(end, tz='UTC'), color='#eacb8a', alpha=.16)
    axes[-1].set_xlabel('UTC. Near book: первые 12 ценовых тиков от своей лучшей цены. F не прибавляется к T-объёму.', fontsize=9)
    p = scene['prediction']
    fig.supxlabel(f"Probe P(down/flat/up): {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}. Вероятности 120s midpoint; не приказ и не PnL.\nСнятие — proxy, а восполнение не доказывает одного и того же участника.", fontsize=9)
    path = REPORT / 'figures' / (scene['id'] + ('_future' if future else '_past') + '.png')
    fig.savefig(path, dpi=115); plt.close(fig)
    return str(path.relative_to(REPORT)).replace('\\', '/')


def select_scenes(X, Y, predictions, fast):
    origins = X.origin_ns.to_numpy()
    state = pd.merge_asof(X[['origin_ns']].sort_values('origin_ns'), fast.sort_values('available_ns'), left_on='origin_ns', right_on='available_ns')
    sign = np.sign(state.aggression30.fillna(0).to_numpy())
    sign = np.where(sign == 0, np.sign(state.ret30_ticks.fillna(0).to_numpy()), sign)
    signed = sign * Y.delta_ticks.to_numpy()
    outcome = np.where(signed > 2, 'CONTINUE', np.where(signed < -2, 'RETURN', 'NEUTRAL'))
    selected, slots = [], []
    for mechanism in MECHANISMS:
        for result in ('CONTINUE', 'RETURN'):
            candidates = np.flatnonzero(Y.known.to_numpy() & state[mechanism].to_numpy(bool) & (outcome == result))
            chosen = []
            for i in candidates:
                if not chosen or origins[i] - origins[chosen[-1]] >= 900 * NS: chosen.append(int(i))
                if len(chosen) == 3: break
            slots.append(dict(mechanism=mechanism, outcome=result, requested=3, available=int(len(candidates)), selected=len(chosen), empty=3-len(chosen)))
            selected.extend((i, mechanism, 'retrospective_stratum') for i in chosen)
    for date in ('2026-08-25', '2026-08-27'):
        for hour in ('10:00', '14:00'):
            t = int(pd.Timestamp(date + ' ' + hour, tz='UTC').value)
            ix = np.flatnonzero((origins == t) & Y.known.to_numpy())
            if len(ix): selected.append((int(ix[0]), 'Фиксированное окно', 'fixed_clock'))
    # Failure views are explicitly diagnostic selection, excluded from frequency claims.
    probs = predictions[['all_down', 'all_flat', 'all_up']].to_numpy()
    for date in ('2026-08-27', '2026-08-28'):
        mask = (X.date == date).to_numpy() & Y.known.to_numpy() & (probs.argmax(1) != Y.outcome.to_numpy())
        candidates = np.flatnonzero(mask)
        if len(candidates):
            i = int(candidates[np.argmax(probs[candidates].max(1))])
            selected.append((i, 'Диагностический случай probe', 'probe_case'))
    scenes = []
    for j, (i, mechanism, selection) in enumerate(selected):
        r = state.iloc[i]
        scenes.append(dict(id=f'scene_{j+1:02d}', date=str(X.date.iloc[i]), origin_ns=int(origins[i]), mechanism=mechanism,
            selection=selection, root_id=X.root_id.iloc[i],
            past={k: float(r[k]) for k in ('mid', 'ret30_ticks', 'aggression30', 'trade_quantity30', 'bid_near_refill30', 'ask_near_refill30', 'bid_near_withdrawal30', 'ask_near_withdrawal30')},
            prediction=probs[i].tolist(),
            future=dict(delta_ticks=float(Y.delta_ticks.iloc[i]), relative_outcome=str(outcome[i]),
                        maturity_ns=int(Y.mature_ns.iloc[i]), class_id=int(Y.outcome.iloc[i]),
                        selected_model_error=selection == 'probe_case')))
    return scenes, slots, state, outcome


def distinctness(X, Y, state, outcome):
    counts = {}
    for mech in MECHANISMS:
        mask = state[mech].to_numpy(bool) & Y.known.to_numpy()
        counts[mech] = dict(n=int(mask.sum()),
            continuation=int((mask & (outcome == 'CONTINUE')).sum()), return_=int((mask & (outcome == 'RETURN')).sum()),
            neutral=int((mask & (outcome == 'NEUTRAL')).sum()),
            note='Tags can overlap; deterministic past descriptors, not ground-truth causal classes.')
    # Nearest price/time/volatility geometry across different days. No Y in matching.
    mask = Y.known.to_numpy() & X[BASE].notna().all(axis=1).to_numpy()
    ix = np.flatnonzero(mask); values = StandardScaler().fit_transform(X.loc[mask, BASE])
    nn = NearestNeighbors(n_neighbors=min(40, len(ix))).fit(values)
    distance, neighbor = nn.kneighbors(values)
    pairs = []
    for k, original in enumerate(ix):
        for d, j in zip(distance[k, 1:], neighbor[k, 1:]):
            other = ix[j]
            if X.date.iloc[original] != X.date.iloc[other]:
                a, b = state.iloc[original], state.iloc[other]
                pairs.append(dict(a=int(original), b=int(other), distance=float(d),
                    aggression_difference=float(abs(a.aggression30-b.aggression30)),
                    refill_difference=float(abs((a.bid_near_refill30+a.ask_near_refill30)-(b.bid_near_refill30+b.ask_near_refill30))),
                    same_future_class=bool(Y.outcome.iloc[original] == Y.outcome.iloc[other])))
                break
    pairs = sorted(pairs, key=lambda r: (r['distance'], r['a']))
    top = pairs[:100]
    return clean(dict(mechanisms=counts, matched_geometry=dict(rule='First 100 nearest cross-day pairs in standardized past price/time/volatility coordinates; no future in matching. Pairs may share roots, so not independent trials.',
        pairs=len(top), median_distance=float(np.median([r['distance'] for r in top])),
        median_aggression_difference=float(np.nanmedian([r['aggression_difference'] for r in top])),
        median_refill_difference=float(np.median([r['refill_difference'] for r in top])),
        fraction_same_future_class=float(np.mean([r['same_future_class'] for r in top])), examples=top[:8])))


def summary_plots(manifest, contexts, state, Y, probe, distinct):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
    families = list(probe['results']); xx = np.arange(len(families))
    for offset, split, color in ((-.18, 'validation', '#739eb7'), (.18, 'test', '#d49e57')):
        ax[0].bar(xx + offset, [probe['results'][f][split]['log_loss'] for f in families], width=.36, label=split, color=color)
    ax[0].set_xticks(xx, families); ax[0].set_ylabel('Multiclass log loss · меньше лучше'); ax[0].legend()
    for offset, split, color in ((-.18, 'validation', '#739eb7'), (.18, 'test', '#d49e57')):
        ax[1].bar(xx + offset, [probe['results'][f][split]['directional_auc'] for f in families], width=.36, label=split, color=color)
    ax[1].set_xticks(xx, families); ax[1].set_ylabel('AUC направления · без FLAT'); ax[1].axhline(.5, color='black', ls='--'); ax[1].set_ylim(.4, .65)
    fig.suptitle('Фиксированный probe: следующий 120s midpoint. Один train-день; два test-дня. Не торговая экономика.', fontsize=12)
    fig.savefig(REPORT / 'figures/probe_comparison.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    sample = Y.known.to_numpy() & state.mechanism_known.to_numpy()
    sc = ax[0].scatter(state.aggression30[sample], state.ret30_ticks[sample], c=(state.bid_near_refill30 + state.ask_near_refill30)[sample],
                       cmap='viridis', s=7, alpha=.5, vmin=0, vmax=50)
    fig.colorbar(sc, ax=ax[0], label='Refill near-book / 30s (цвет до 50)')
    ax[0].set(xlabel='Асимметрия агрессии за 30s', ylabel='Уже наблюдённое Δprice, ticks', ylim=(-80, 80))
    ax[0].axhline(0, color='#777', lw=.8)
    labels = [LABELS[m] for m in MECHANISMS]
    vals = np.array([[distinct['mechanisms'][m][x] for x in ('continuation', 'return_', 'neutral')] for m in MECHANISMS], float)
    bottom = np.zeros(len(vals))
    for j, (name, color) in enumerate((('Продолжение', '#6994af'), ('Возврат', '#c77968'), ('Нейтрально', '#babdc1'))):
        fraction = np.divide(vals[:, j], vals.sum(1), out=np.zeros(len(vals)), where=vals.sum(1)>0)
        ax[1].barh(np.arange(4), fraction, left=bottom, label=name, color=color); bottom += fraction
    ax[1].set_yticks(np.arange(4), labels, fontsize=9); ax[1].set_xlabel('120s исход относительно текущей агрессии / движения')
    for i, n in enumerate(vals.sum(1)): ax[1].text(1.01, i, 'n=' + str(int(n)), va='center', fontsize=8)
    ax[1].set_xlim(0, 1.18); ax[1].legend(loc='lower left', bbox_to_anchor=(0, -.26), ncol=3, fontsize=8)
    fig.suptitle('Одинаковое малое движение может сопровождаться разным усилием. Ни один тег не гарантирует исход.', fontsize=12)
    fig.savefig(REPORT / 'figures/mechanism_distinctness.png', dpi=140); plt.close(fig)
    c = contexts['2026-08-25']; at = int(pd.Timestamp('2026-08-25 17:00', tz='UTC').value)
    c = c[c.available_ns <= at]; x = pd.to_datetime(c.t_ns, utc=True)
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True, layout='constrained')
    ax[0].plot(x, c.close.where(c.known), lw=.8, label='Midpoint M1'); ax[0].plot(x, c.combined_vwap, lw=1.4, label='Exact trade VWAP')
    ax[0].plot(x, c.combined_midpoint_proxy, lw=.8, ls='--', label='Volume × M1 midpoint close proxy'); ax[0].legend(fontsize=8, ncol=3)
    ax[0].set_ylabel('$/oz')
    ax[1].plot(x, (c.combined_trade_vwap_on_proxy_support - c.combined_midpoint_proxy) / TICK, color='#a57d22'); ax[1].set_ylabel('Exact − proxy\nобщий объём, ticks'); ax[1].axhline(0, color='#777', lw=.5)
    for name in ('stft_2_8m', 'stft_8_32m', 'stft_32_64m'): ax[2].plot(x, c[name], label=name)
    ax[2].set_ylabel('Энергия STFT'); ax[2].legend(fontsize=8, ncol=3)
    for a in ax:
        a.grid(alpha=.15)
        a.fill_between(x, 0, 1, where=~c.scheduled_open.to_numpy(), transform=a.get_xaxis_transform(), color='#ccc', alpha=.3)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M', tz='UTC')); ax[2].set_xlabel('UTC · серое = плановая пауза; STFT не интерполируется через разрывы')
    fig.suptitle('Реальная двухдневная память, точный VWAP и спектр завершённых 128 минут', fontsize=13)
    fig.savefig(REPORT / 'figures/two_day_context.png', dpi=140); plt.close(fig)


def main():
    manifest, cfg, fast, contexts, zones, events = load()
    (REPORT / 'figures').mkdir(parents=True, exist_ok=True)
    X, Y = query_tables(fast, contexts, cfg)
    X.to_parquet(ROOT / 'queries_X.parquet', index=False); Y.to_parquet(ROOT / 'diagnostic_Y.parquet', index=False)
    with threadpool_limits(limits=2): predictions, probe = fit_probes(X, Y, cfg)
    predictions.to_parquet(REPORT / 'probe_predictions.parquet', index=False)
    save(REPORT / 'probe_scores.json', clean(probe))
    scenes, slots, state, outcome = select_scenes(X, Y, predictions, fast)
    separation = distinctness(X, Y, state, outcome)
    save(REPORT / 'distinctness.json', separation)
    summary_plots(manifest, contexts, state, Y, probe, separation)
    for s in scenes:
        d = s['date']; s['past_image'] = plot_scene(s, fast[fast.date == d], contexts[d], zones[d], False)
        s['future_image'] = plot_scene(s, fast[fast.date == d], contexts[d], zones[d], True)
    selection = dict(scope='Retrospective strata are illustrations, not frequency or predictive evaluation; first chronological examples with 15-minute spacing within each slot.',
                     slots=slots, scenes=scenes, regular_queries=len(X), mature_queries=int(Y.known.sum()))
    save(REPORT / 'scene_selection.json', clean(selection))
    view = []
    for s in scenes:
        view.append(f'<article><h3>{s["id"]} · {html.escape(LABELS.get(s["mechanism"],s["mechanism"]))} · {s["date"]}</h3>'
            f'<p class="muted">{s["selection"]} · {s["root_id"]}</p><button onclick="flip(this)">Раскрыть следующие 120 секунд</button>'
            f'<img loading="lazy" src="{s["past_image"]}" data-past="{s["past_image"]}" data-future="{s["future_image"]}" alt="{s["id"]}"></article>')
    scores = []
    for name, results in probe['results'].items():
        r = results['test']; scores.append(f'<tr><td>{name}</td><td>{r["n"]}</td><td>{r["log_loss"]:.4f}</td><td>{r["brier"]:.4f}</td><td>{r["directional_auc"]:.4f}</td></tr>')
    rows = ''.join(f'<tr><td>{d}</td><td>{sum(contexts[d].known)}</td><td>{len(contexts[d])}</td><td>{len(events[d])}</td></tr>' for d in contexts)
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Репрезентация рынка v1 — читаемость и различимость</title><style>
body{font:16px/1.55 system-ui,sans-serif;background:#edf1f5;color:#183044;margin:0}main{max-width:1120px;margin:auto;padding:30px 24px}h1{font-size:30px;line-height:1.2}h2{margin-top:35px}.intro,article{background:white;padding:22px;border-radius:10px;margin:22px 0}img{display:block;width:100%;height:auto;margin:18px 0}.muted{color:#657786;font-size:13px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px 13px;border-bottom:1px solid #dde4e9;text-align:left}button{padding:11px 18px;background:#173f5a;color:white;border:0;border-radius:5px;cursor:pointer}a{color:#176f99}code{background:#e9eef2;padding:2px 5px}.notice{border-left:4px solid #c29047;padding-left:16px}
</style><main><h1>Репрезентация рынка: что видно до решения</h1><p>GCZ6 · 24–28 августа 2026 · наблюдаемая борьба агрессии и ликвидности в памяти текущего и прошлого дня.</p>
<div class="intro"><p><b>Собран вход:</b> 15 минут по 1 секунде, минутная история от вчерашнего London 08:00, доступные уровни предыдущего цикла, последовательность контактов и наблюдаемых состояний. Объём — контракты GC; цена — доллары за унцию.</p><p>Исполнение, снятие и восполнение измеряются раздельно. Наблюдаемые роли не раскрывают личность, намерение или общий инвентарь участника.</p><p class="notice">Это описательная исследовательская версия. Сквозная проверка: <a href="verification.json">verification.json</a>. Независимый визуальный аудит: <a href="sol61_visual_audit.json">Sol 6.1</a>. Подготовленный loader ещё не является обученным игроком.</p></div>
<h2>Память, VWAP и поддержка спектра</h2><img src="figures/two_day_context.png" alt="Память двух дней"><p>Спектр — trailing Hann по 128 завершённым минутам, диапазоны 2–8, 8–32 и 32–64 минут. Периоды 64–128 минут не объявляются устойчивыми по одному-двум циклам. Энергия не кодирует направление.</p>
<table><tr><th>День решений</th><th>Известных M1</th><th>Физических M1 в файле</th><th>Токенов событий</th></tr>''' + rows + '''</table><p class="muted">Файл может содержать более поздние строки; observe(t) возвращает только available_ns ≤ t. Плановые закрытия и неизвестные интервалы остаются на физической сетке с масками. UTC reset не склеивается в непрерывный локальный поток.</p>
<h2>Описательная различимость и её границы</h2><img src="figures/mechanism_distinctness.png" alt="Механизмы и разные исходы"><p>Теги определены по прошлым измерениям; они могут пересекаться. Их видимое разделение само по себе не доказывает прогноз. Частоты исходов ниже считаются по общей временной сетке, а не по выбранным картинкам.</p>
<h2>Хронологический probe</h2><p>TRAIN: 25 августа; VALIDATION: 26 августа; TEST: 27–28 августа. Одна фиксированная конфигурация. Исход — midpoint через 120s: DOWN / FLAT ±2 ticks / UP. Это отдельная диагностика представления; существующие crossing target и BCE не менялись.</p><p>Probe — небольшой gradient boosting на сводках. Он проверяет вклад групп признаков, но не проверяет нейросетевую обработку всей двухдневной последовательности. Даты уже использовались для разработки.</p><img src="figures/probe_comparison.png" alt="Сравнение probe"><table><tr><th>Вход</th><th>Test n</th><th>Log loss ↓</th><th>Brier ↓</th><th>Direction AUC ↑</th></tr>''' + ''.join(scores) + '''</table><p class="muted">price: цена/время/волатильность; book: поток/глубина/снятия/refill; context: координаты дня и вчерашнего цикла, VWAP, спектр. На всех вариантах один cohort. Никакая колонка Y не передаётся в X.</p>
<h2>Атлас: сначала прошлое, затем проверка ожидания</h2><p>Первые хронологические примеры в каждой ретроспективной группе «механизм × продолжение/возврат». Между выбранными примерами одного слота ≥15 минут, чтобы не показывать один фрагмент многократно. Это правило показа, не cooldown обучения. Пустые слоты сохраняются в <a href="scene_selection.json">реестре</a>. Добавлены фиксированные часы и наиболее уверенные ошибки probe на test.</p>''' + ''.join(view) + '''<h2>Следующий допуск</h2><p>Для перехода к игроку ещё нужны обучение sequence encoder через этот loader, повторение на неиспользованных датах и общий оценщик действий с исполнением и издержками. Текущие вероятности и изменения midpoint не являются доходностью сделки.</p><p><a href="source_to_tensor_inventory.json">Происхождение данных и фактический вход</a> · <a href="probe_scores.json">Все метрики probe</a> · <a href="distinctness.json">Контрпримеры и похожая геометрия</a> · <a href="scene_selection.json">Отбор сцен</a></p></main><script>
function flip(button){const img=button.parentElement.querySelector('img');const future=img.dataset.revealed!=='yes';img.src=future?img.dataset.future:img.dataset.past;img.dataset.revealed=future?'yes':'no';button.textContent=future?'Снова скрыть будущее':'Раскрыть следующие 120 секунд';}
</script></html>'''
    page = page.replace('Добавлены фиксированные часы и наиболее уверенные ошибки probe на test.',
        'Добавлены фиксированные часы и диагностические случаи probe; причина их отбора раскрывается вместе с будущим.')
    (REPORT / 'index.html').write_text(page, encoding='utf-8')
    output_paths = [p for p in REPORT.iterdir() if p.is_file() and p.suffix in ('.html', '.json', '.parquet') and p.name != 'report_manifest.json']
    output_paths += list((REPORT / 'figures').glob('*.png'))
    save(REPORT / 'report_manifest.json', dict(status='COMPLETE_PENDING_VISUAL_AUDIT', dataset_manifest_sha256=sha(ROOT / 'manifest.json'),
        scenes=len(scenes), images=len(list((REPORT / 'figures').glob('*.png'))), output_hashes={str(p): sha(p) for p in output_paths},
        query_hashes={str(ROOT / n): sha(ROOT / n) for n in ('queries_X.parquet', 'diagnostic_Y.parquet')},
        implementation={p: sha(p) for p in ('scripts/report_market_representation_v1.py', 'spectrum/market_repr/probes.py')},
        limitations=manifest['limitations']))
    print(json.dumps(dict(status='COMPLETE_PENDING_VISUAL_AUDIT', scenes=len(scenes), queries=len(X), mature=int(Y.known.sum()), test=probe['results']['all']['test'])), flush=True)


if __name__ == '__main__': main()
