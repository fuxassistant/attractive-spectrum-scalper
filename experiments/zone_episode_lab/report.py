"""Render a frozen zone-episode audit; never executes or selects a strategy.

Run with the plotting environment:
  .venv/Scripts/python.exe -m experiments.zone_episode_lab.report OUTDIR
Only COMPLETE run manifests and hash-verified saved inputs are accepted.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .spectral import zone_spectrum

NS = 1_000_000_000
POLICIES = {
    "return_frequent": "Часто посещаемая зона",
    "return_sticky": "Зона с накопленным пребыванием",
    "return_permeable": "Быстро проходимая зона",
    "return_sticky_spectral": "Пребывание + спектральное вето",
    "return_sticky_interval": "Пребывание + CV интервалов",
    "inside_now": "Вход внутри сейчас",
    "wait_outside": "Ждать выхода и возврата",
}
SPLITS = {"development": "Разработка", "validation": "Валидация",
          "exposed_diagnostic": "Ранее просмотренный", "extension": "Продолжение"}
SHORT_SPLITS = {"development": "Разраб.", "validation": "Валид.",
                "exposed_diagnostic": "Просмотр.", "extension": "Продолж."}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return actual


def fmt(value, places=0):
    if value is None or (not isinstance(value, (str, bool)) and pd.isna(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "да" if value else "нет"
    if isinstance(value, str):
        return value
    return f"{value:,.{places}f}".replace(",", " ")


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>'+html.escape(str(x))+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def summary_row(frame, policy, split, direction="toward", stress=False):
    selected = frame.loc[frame.policy.eq(policy) & frame.split.eq(split) & frame.direction.eq(direction) & frame.stress.eq(stress)]
    if len(selected) > 1:
        raise ValueError("Duplicate account summary")
    return selected.iloc[0] if len(selected) else None


def select_episode(episodes, candidates):
    ids = candidates.loc[candidates.parent_episode_id.ge(0), ["date", "parent_episode_id"]].drop_duplicates().rename(columns={"parent_episode_id": "episode_id"})
    eligible = episodes.merge(ids, on=["date", "episode_id"], validate="one_to_one")
    eligible = eligible.loc[eligible.completed & eligible.entry_side.ne(0)]
    if eligible.empty:
        eligible = episodes.loc[episodes.completed & episodes.entry_side.ne(0)]
    if eligible.empty:
        raise ValueError("No completed episode available for a boundary audit")
    date = "2026-08-24" if eligible.date.eq("2026-08-24").any() else sorted(eligible.date.unique())[0]
    selected = eligible.loc[eligible.date.eq(date)].sort_values(["arrival_ns", "episode_id"])
    after = selected.loc[selected.arrival_ns.ge(pd.Timestamp(date, tz="UTC").value + 8*3600*NS)]
    return (after if len(after) else selected).iloc[0]


def synthetic_controls():
    t = np.arange(5, 3605, 5, dtype=np.int64)*NS
    periodic = ((t//NS-5) % 300 < 75).astype(float)
    path = np.cumsum(np.random.default_rng(811).normal(size=len(t)))
    walk = (abs(path) < 2).astype(float)
    arrivals = (np.flatnonzero((walk[1:] > 0) & (walk[:-1] == 0))+1)*5*NS
    inputs = [("Периодические посещения", periodic, np.arange(0, 3600, 300, dtype=np.int64)*NS),
              ("Один проход", np.r_[np.zeros(180), np.ones(360), np.zeros(180)], np.array([900*NS], np.int64)),
              ("Случайное блуждание", walk, arrivals)]
    return [(name, zone_spectrum(t, values, 3600*NS, arrival_ns=a, include_spectrum=True)) for name, values, a in inputs]


def return_profiles(episodes, trials):
    joined = trials.merge(episodes[["date", "episode_id", "outcome", "core_dwell_seconds"]], on=["date", "episode_id"], validate="one_to_one")
    joined["profile"] = np.where(joined.outcome.str.startswith("PASS"), "PASS", np.where(joined.outcome.str.startswith("REJECT"), "REJECT", "UNKNOWN"))
    joined["dwell_bucket"] = pd.cut(joined.core_dwell_seconds, [-np.inf, 5, 30, np.inf], right=False, labels=["0–<5 с", "5–<30 с", "30+ с"])
    rows = []
    for (profile, dwell), group in joined.groupby(["profile", "dwell_bucket"], observed=True):
        returned = group.status.eq("RETURN")
        complete = group.status.isin(["RETURN", "NO_RETURN_300S"])
        exposure = ((group.loc[complete, "available_ns"]-group.loc[complete, "departure_ns"])/NS).clip(0, 300).sum()
        rows.append(dict(profile=profile, dwell=str(dwell), trials=len(group), supported=int(complete.sum()),
                         censored=int((~complete).sum()), returns=int(returned.sum()),
                         return_rate=float(returned.sum()/complete.sum()) if complete.any() else None,
                         median_seconds=float(group.loc[returned, "seconds_to_return"].median()) if returned.any() else None,
                         median_approach_efficiency=float(group.loc[returned, "approach_efficiency"].median()) if group.loc[returned, "approach_efficiency"].notna().any() else None,
                         median_peak_distance_ticks=float(group.loc[returned, "absence_peak_distance_ticks"].median()) if group.loc[returned, "absence_peak_distance_ticks"].notna().any() else None,
                         known_exposure_hours=float(exposure/3600)))
    return rows


def plot_audit(out, cfg, summary, chosen, events, quote_path, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.8))
    ax = axes[0, 0]
    low, high = chosen.zone_low_nano/NS, chosen.zone_high_nano/NS
    pad = cfg["geometry"]["departure_ticks"]*.1
    q = quote_path
    time = pd.to_datetime(q.available_ns, unit="ns", utc=True)
    ax.plot(time, q.plot_mid, color="#243746", lw=.8, label="Native midpoint")
    ax.axhspan(low, high, color="#7296ae", alpha=.16, label="Ядро зоны")
    ax.axhline(low-pad, color="#8d6c42", lw=.8, ls="--")
    ax.axhline(high+pad, color="#8d6c42", lw=.8, ls="--", label="Граница подтверждения")
    marks = [("ARRIVAL", "o", "Прибытие", "#246b47"), ("EDGE_EXIT", "x", "Выход за край", "#b47b28"),
             ("EDGE_RETURN", "^", "Возврат за край", "#587da0"), ("DEPARTURE_CONFIRMED", "D", "Уход подтверждён", "#984539")]
    for kind, marker, label, color in marks:
        part = events.loc[events.event_type.eq(kind)]
        if len(part):
            ax.scatter(pd.to_datetime(part.available_ns, unit="ns", utc=True), part.mid_nano/NS, marker=marker, c=color, s=32, label=label, zorder=5)
    ax.axvline(pd.Timestamp(int(chosen.end_ns), unit="ns", tz="UTC"), color="#984539", lw=.8, alpha=.7)
    ax.set(title=f"1. Граница эпизода: {chosen.date}, ID {int(chosen.episode_id)}\n{low:.1f}–{high:.1f}; исход {chosen.outcome} известен только справа", xlabel="Публикация · UTC", ylabel="Цена · USD/oz")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.legend(fontsize=8, ncol=2, loc="best", frameon=True, facecolor="white", edgecolor="none", framealpha=.95)
    ax = axes[0, 1]
    for (name, result), color in zip(synthetic, ["#356c9c", "#b4772c", "#47836b"]):
        freq, power = np.asarray(result["frequency_hz"]), np.asarray(result["power_fraction"])
        period = 1/freq
        mask = period >= 30
        order = np.argsort(period[mask])
        ax.plot(period[mask][order], power[mask][order], lw=1.4, color=color,
                label=f"{name}; C={result['main_lobe_power_fraction']:.2f}")
    ax.set_xscale("log")
    ax.set_xticks([30, 60, 120, 300, 1200, 3600], ["30", "60", "120", "300", "1200", "3600"])
    ax.set(title="2. Концентрация спектра не доказывает цикличность\nОдин проход концентрирован сильнее периодического контроля", xlabel="Период · с (логарифмическая шкала)", ylabel="Доля мощности в частотном отсчёте")
    ax.legend(fontsize=8.5, frameon=False)
    splits = [s for s in cfg["splits"] if summary.split.eq(s).any()]
    positions = np.arange(len(splits))
    ax = axes[1, 0]
    for offset, direction, stress, label, color in [(-.24, "toward", False, "К зоне", "#356c9c"), (0, "away", False, "От зоны", "#b4772c"), (.24, "toward", True, "К зоне: стресс", "#984539")]:
        records = [summary_row(summary, "return_sticky", s, direction, stress) for s in splits]
        bars = ax.bar(positions+offset, [r.net_usd for r in records], .22, color=color, label=label)
        for bar, record in zip(bars, records):
            if not record.capital_curve_complete:
                bar.set_hatch("//")
    median = [float(summary_row(summary, "return_sticky", s).coin_median_usd) if pd.notna(summary_row(summary, "return_sticky", s).coin_median_usd) else np.nan for s in splits]
    p95 = [float(summary_row(summary, "return_sticky", s).coin_p95_usd) if pd.notna(summary_row(summary, "return_sticky", s).coin_p95_usd) else np.nan for s in splits]
    ax.scatter(positions, median, color="#24282d", marker="_", s=160, label="Монетка, base: медиана", zorder=4)
    ax.scatter(positions, p95, color="#24282d", marker="D", s=23, label="Монетка, base: p95", zorder=4)
    ax.set(title="3. Return sticky: последовательный счёт после затрат", ylabel="Чистый результат · USD", xticks=positions, xticklabels=[SHORT_SPLITS[s] for s in splits])
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, -.13))
    ax = axes[1, 1]
    for offset, policy, color in [(-.17, "inside_now", "#356c9c"), (.17, "wait_outside", "#47836b")]:
        records = [summary_row(summary, policy, s) for s in splits]
        bars = ax.bar(positions+offset, [r.net_usd for r in records], .32, color=color, label=POLICIES[policy])
        for bar, record in zip(bars, records):
            if not record.capital_curve_complete:
                bar.set_hatch("//")
    ax.set(title="4. Общие исходные контакты, разные правила участия\nКаждая ветвь: отдельный последовательный счёт", ylabel="Чистый результат · USD", xticks=positions, xticklabels=[SHORT_SPLITS[s] for s in splits])
    ax.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(0, -.13))
    for i, ax in enumerate(axes.flat):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
        if i >= 2:
            ax.axhline(0, color="#555555", lw=.7)
    fig.suptitle("Память эпизодов зоны: наблюдение, повторяемость и торговый результат", x=.065, ha="left", fontsize=17)
    fig.text(.065, .035, "Эпизод выбран по времени, не по прибыли; рисунок заканчивается на подтверждении ухода. Блоки начинают счёт заново.\nМонетка: базовые затраты; штриховка = неполная кривая. Геометрический возврат не равен прибыльной сделке.", fontsize=10)
    fig.subplots_adjust(left=.065, right=.975, top=.90, bottom=.18, hspace=.43, wspace=.22)
    path = out/"zone_episode_audit.png"
    fig.savefig(path, dpi=155)
    plt.close(fig)
    return path


def render(directory):
    out = Path(directory).resolve()
    manifest = read_json(out/"manifest.json")
    if not str(manifest.get("status", "")).startswith("COMPLETE"):
        raise ValueError("Report requires a completed run; partial outputs are not read")
    cfg = read_json(out/"config.json")
    quarantined = []
    for name in ("verified_v1", "verified_v2", "smoke_v1", "smoke_v2"):
        previous = out.parent/name/"manifest.json"
        if previous.exists() and previous.parent != out:
            old = read_json(previous)
            status = str(old.get("status", ""))
            if any(word in status for word in ("SUPERSEDED", "INVALID", "QUARANTIN")):
                quarantined.append([name, status, str(old.get("quarantine_reason", old.get("invalidation_reason", "См. manifest предыдущего прогона")))])
    inputs = {"manifest.json": digest(out/"manifest.json"), "config.json": digest(out/"config.json")}
    root = Path(__file__).resolve().parents[2]
    for name, expected in manifest["implementation_sha256"].items():
        inputs["implementation:"+name] = verify(root/name, expected)
    if cfg != read_json(Path(__file__).with_name("config.json")):
        raise ValueError("Saved configuration differs from the hash-verified frozen source")
    for name in ("independent_review.json", "diagnostic_gates.json", "execution_review.json"):
        if (out/name).exists():
            inputs[name] = digest(out/name)
    for name, expected in manifest["artifacts"].items():
        inputs[name] = verify(out/name, expected)
    names = ["policy_summary", "account_trades", "candidates", "spectral", "episodes", "return_trials", "parent_comparison", "coin_controls"]
    if any(name+".parquet" not in manifest["artifacts"] for name in names):
        raise ValueError("A report input is missing from the artifact hash manifest")
    data = {name: pd.read_parquet(out/(name+".parquet")) for name in names}
    summary, episodes, candidates = data["policy_summary"], data["episodes"], data["candidates"]
    chosen = select_episode(episodes, candidates)
    counts = Counter()
    fast_returns = 0
    selected_events = None
    selected_receipt = None
    for receipt in manifest["receipts"]:
        day = out/"days"/receipt["date"]
        events_path = day/"events.parquet"
        inputs[str(events_path.relative_to(out))] = verify(events_path, receipt["artifact_sha256"]["events.parquet"])
        edge_returns = pd.read_parquet(events_path, columns=["event_type", "duration_ns"], filters=[("event_type", "==", "EDGE_RETURN")])
        fast_returns += int(edge_returns.duration_ns.le(cfg["geometry"]["confirmation_seconds"]*NS).sum())
        counts.update(receipt["event_counts"])
        if receipt["date"] == chosen.date:
            selected_events = pd.read_parquet(events_path, filters=[("episode_id", "==", int(chosen.episode_id))])
            selected_receipt = receipt
    source = Path(selected_receipt["source"])
    source_manifest = source/"manifest.json"
    inputs[str(source_manifest)] = verify(source_manifest, selected_receipt["source_manifest_sha256"])
    qpath = source/"quotes.parquet"
    inputs[str(qpath)] = verify(qpath, selected_receipt["source_payloads"]["quotes.parquet"])
    start, stop = int(chosen.arrival_ns)-30*NS, int(chosen.end_ns)
    quote_path = pd.read_parquet(qpath, columns=["available_ns", "bid_nano", "ask_nano", "valid", "epoch"], filters=[("available_ns", ">=", start), ("available_ns", "<=", stop)])
    supported = quote_path.valid & quote_path.epoch.ge(0) & quote_path.bid_nano.gt(0) & quote_path.ask_nano.gt(quote_path.bid_nano)
    discontinuity = quote_path.available_ns.diff().gt(cfg["geometry"]["max_quote_age_seconds"]*NS) | quote_path.epoch.diff().fillna(0).ne(0)
    quote_path["plot_mid"] = ((quote_path.bid_nano+quote_path.ask_nano)/2e9).where(supported & ~discontinuity)
    # Preserve local extrema and chronology when a high-activity episode is long.
    if len(quote_path) > 8000:
        quote_path["bucket"] = ((quote_path.available_ns-start)//max(1, (stop-start)//2000)).astype(np.int64)
        extrema = pd.DataFrame({"bucket": quote_path.bucket, "mid": quote_path.plot_mid}).dropna().groupby("bucket").mid.agg(["idxmin", "idxmax"])
        ids = np.unique(np.r_[quote_path.index[0], extrema.to_numpy().ravel(), quote_path.index[quote_path.plot_mid.isna()], quote_path.index[-1]])
        quote_path = quote_path.loc[ids].sort_values("available_ns")
    synthetic = synthetic_controls()
    figure = plot_audit(out, cfg, summary, chosen, selected_events, quote_path, synthetic)
    profiles = return_profiles(episodes, data["return_trials"])
    splits = [s for s in cfg["splits"] if summary.split.eq(s).any()]
    rows = []
    for split in splits:
        for policy in POLICIES:
            base = summary_row(summary, policy, split)
            away = summary_row(summary, policy, split, "away")
            stress = summary_row(summary, policy, split, stress=True)
            coins = data["coin_controls"].loc[data["coin_controls"].split.eq(split) & data["coin_controls"].policy.eq(policy)]
            rows.append([SPLITS[split], POLICIES[policy], fmt(base.trades), fmt(base.active_days), fmt(base.net_usd), fmt(away.net_usd), fmt(stress.net_usd), fmt(base.coin_median_usd), fmt(base.coin_p95_usd), f"{int(coins.complete.sum())}/{len(coins)}", fmt(base.mean_usd, 1), fmt(base.win_rate*100, 1)+"%", fmt(base.max_drawdown_usd), fmt(base.unresolved)])
    account_table = table(["Блок", "Правило", "Сделки", "Дни", "К зоне $", "От зоны $", "Стресс $", "Монетка med $", "Монетка p95 $", "Полные монетки", "Среднее $", "Победы", "Max DD $", "Незакрыто"], rows)
    parent_rows = []
    for split, group in data["parent_comparison"].groupby("split", sort=False):
        supported = group.inside_pnl_usd.notna() & group.outside_pnl_usd.notna()
        pair = group.loc[supported]
        parent_rows.append([SPLITS.get(split, split), fmt(len(group)), fmt(len(pair)), fmt((~supported).sum()), fmt(group.inside_status.eq("WAIT").sum()), fmt(group.outside_status.eq("WAIT").sum()), fmt(pair.inside_pnl_usd.mean(), 2), fmt(pair.outside_pnl_usd.mean(), 2), fmt((pair.outside_pnl_usd-pair.inside_pnl_usd).mean(), 2)])
    parent_table = table(["Блок", "Все контакты", "Обе ветви известны", "Неизвестна ветвь", "WAIT внутри", "WAIT снаружи", "Внутри $/контакт", "Снаружи $/контакт", "Разность $/контакт"], parent_rows)
    spec_rows = []
    for split, group in data["spectral"].groupby("split", sort=False):
        spec_rows.append([SPLITS.get(split, split), fmt(len(group)), fmt(group.known.sum()), fmt(group.spectrum_known.sum()), fmt(group.period_candidate_supported.sum()), fmt(group.spectral_gate.sum()), fmt(group.interval_gate.sum()), fmt(group.coverage_fraction.mean()*100, 2)+"%"])
    spectral_table = table(["Блок", "Внешние кандидаты", "Полное окно", "Ненулевой спектр", "≥3 цикла + интервалы", "Спектральное вето", "CV-вéто", "Среднее покрытие"], spec_rows)
    return_table = table(["Исход ухода", "Dwell ядра", "Попытки", "Известен исход 300с", "Цензурировано", "Возвраты", "Возврат / известные", "Медиана возврата, с", "Прямота пути, med", "Удаление, med тики", "Время завершённых испытаний, ч"], [[r["profile"], r["dwell"], fmt(r["trials"]), fmt(r["supported"]), fmt(r["censored"]), fmt(r["returns"]), fmt(r["return_rate"]*100, 1)+"%" if r["return_rate"] is not None else "—", fmt(r["median_seconds"], 1), fmt(r["median_approach_efficiency"], 3), fmt(r["median_peak_distance_ticks"], 1), fmt(r["known_exposure_hours"], 1)] for r in profiles])
    memory_table = ""
    inside = candidates.loc[candidates.date.eq(chosen.date) & candidates.inside & candidates.parent_episode_id.eq(chosen.episode_id)]
    if "history_tokens.parquet" in manifest["artifacts"] and len(inside):
        candidate = inside.sort_values("t_ns").iloc[0]
        tokens = pd.read_parquet(out/"history_tokens.parquet", filters=[("candidate_id", "==", int(candidate.candidate_id))]).sort_values("position")
        if tokens.available_ns.gt(candidate.t_ns).any():
            raise ValueError("A saved memory token was unavailable at its decision clock")
        memory_table = '<h2>Что было известно до показанного контакта</h2><p>Последние завершённые эпизоды той же абсолютной зоны на момент решения внутри. Показанный выше будущий для этого решения исход в память не включён.</p>' + table(["Эпизод", "Вход UTC", "Доступен UTC", "Сторона входа", "Сторона ухода", "Исход", "Dwell, с", "Длительность, с", "Предыдущая пауза, с", "Прямота подхода"], [[fmt(r.episode_id), pd.Timestamp(r.arrival_ns,unit="ns",tz="UTC").strftime("%H:%M:%S"), pd.Timestamp(r.available_ns,unit="ns",tz="UTC").strftime("%H:%M:%S"), fmt(r.entry_side), fmt(r.exit_side), r.outcome, fmt(r.core_dwell_seconds,1), fmt(r.duration_seconds,1), fmt(r.previous_absence_seconds,1), fmt(r.approach_efficiency,3)] for r in tokens.itertuples()])
    pass_count = int(episodes.outcome.str.startswith("PASS").sum())
    reject_count = int(episodes.outcome.str.startswith("REJECT").sum())
    flow_table = table(["Наблюдение за полные UTC-дни", "Количество"], [["Выход за край ядра: EDGE_EXIT", fmt(counts["EDGE_EXIT"])], ["Возврат за край внутри того же эпизода: EDGE_RETURN", fmt(counts["EDGE_RETURN"])], [f"Из них возврат за ≤{cfg['geometry']['confirmation_seconds']} с", fmt(fast_returns)], ["Подтверждённые уходы", fmt(counts["DEPARTURE_CONFIRMED"])], ["Эпизоды PASS", fmt(pass_count)], ["Эпизоды REJECT", fmt(reject_count)], ["Эпизоды UNKNOWN_ENTRY", fmt(episodes.outcome.eq("UNKNOWN_ENTRY").sum())], ["Цензурированные эпизоды", fmt(episodes.outcome.eq("CENSORED").sum())], ["Пропуск ценовой ячейки: JUMP_OVER", fmt(counts["JUMP_OVER"])]])
    baseline = summary.loc[summary.policy.eq("return_sticky") & summary.direction.eq("toward") & ~summary.stress]
    positive = int((baseline.net_usd.gt(0) & baseline.capital_curve_complete).sum())
    lead = f"Базовое правило return_sticky имеет положительный чистый результат с полной кривой в {positive} из {len(baseline)} блоков. Эти даты уже использовались в исследовании; независимого подтверждения преимущества нет."
    image = base64.b64encode(figure.read_bytes()).decode("ascii")
    definitions = [
        f"Адрес зоны: фиксированная ячейка {cfg['geometry']['bin_ticks']*.1:g} USD/oz. Уход требует отступления ещё на {cfg['geometry']['departure_ticks']*.1:g} USD/oz и {cfg['geometry']['confirmation_seconds']} с непрерывного подтверждения. Короткий EDGE_RETURN не является новым завершённым визитом.",
        "PASS: вход с одной стороны и подтверждённый уход через другую; REJECT: подтверждённый уход обратно. Эти исходы известны только после подтверждения ухода. UNKNOWN_ENTRY и CENSORED сохраняются отдельно.",
        "Возврат за 300 с считается после подтверждённого ухода. Его частота делится только на испытания RETURN + NO_RETURN_300S; неизвестное продолжение не превращается в отсутствие возврата. В таблице экспозиция суммируется только для испытаний с известным исходом, до возврата или горизонта.",
        "Торговые кандидаты допускаются по опубликованным признакам и завершённой истории; решения вне зоны принимаются на готовых M1. Одна позиция, один контракт GC, комиссия, bid/ask, задержка и ограничения счёта. Каждый блок начинает счёт заново.",
        f"Основная задержка {cfg['execution']['latency_ms']:g} мс; стресс {cfg['stress']['latency_ms']:g} мс и {cfg['stress']['slippage_ticks_per_fill']:g} тик проскальзывания на исполнение. Наблюдённый возврат цены сам по себе не покрывает эти затраты.",
        "Спектр: фиксированная зона, завершённые 5-секундные доли присутствия за последний час, причинное Hann-окно. Разрывы запрещают расчёт; период требует ≥3 циклов и ≥3 завершённых интервалов возвращения. Сильный пик — описательная величина, не тест значимости и не торговая команда.",
        "Спектральное вето сравнивается с простым CV интервалов на тех же зафиксированных правилах. Регрессия не обучалась; эти результаты не устанавливают условное предсказательное приращение спектра.",
    ]
    model_rows = [[code, description] for code, description in POLICIES.items()]
    synthetic_rows = [[name, fmt(r["raw_peak_period_seconds"]), fmt(r["main_lobe_power_fraction"], 3), fmt(r["return_interval_count"]), fmt(r["return_interval_cv"], 3), fmt(r["dominant_period_seconds"])] for name, r in synthetic]
    status_rows = data["spectral"].groupby("status").size().sort_values(ascending=False)
    quarantine_section = ""
    correction_note = ""
    if quarantined:
        quarantine_section = '<h2>Исключённые предыдущие артефакты</h2><p>Этот отчёт использует только '+html.escape(out.name)+'. Ранее сохранённые результаты ниже не являются основанием для текущих выводов. В частности, накопление длительности в дробных секундах могло дать occupancy чуть выше единицы и ошибочно исключить полное окно спектра. Исправление интегрирует целочисленные наносекунды и делит один раз в конце; пороги спектрального и интервального фильтров не изменялись.</p>'+table(["Прогон","Статус","Причина"],quarantined)
        if any("NUMERICAL_OCCUPANCY" in row[1] for row in quarantined):
            spectral_trades = summary.loc[summary.policy.eq("return_sticky_spectral") & summary.direction.eq("toward") & ~summary.stress, "trades"].sum()
            correction_note = '<p><strong>Исправление предыдущего вывода.</strong> После точного интегрирования времени полное покрытие имеют '+fmt(data["spectral"].known.sum())+' из '+fmt(len(data["spectral"]))+' окон кандидатов. Спектральное правило исполнило '+fmt(spectral_trades)+' сделки по всем блокам. Прежний вывод об отсутствии сделок этого правила отменён: его вызвала численная ошибка поддержки данных.</p>'
    sections = f"""<h1>Память эпизодов ценовой зоны</h1>
<p class="subtitle">{html.escape(cfg['symbol'])} · {manifest['days']} торговых дней · {fmt(manifest['quote_rows'])} native quotes · завершённый диагностический прогон</p>
<p class="lead">{html.escape(lead)}</p><p><strong>Экономический допуск: NOT_ESTABLISHED.</strong> Здесь отдельно показаны геометрическая повторяемость, исполнимая механическая игра и неизвестные исходы. Блок «продолжение» содержит только два дня и не может выполнить порог четырёх активных дней.</p>
{correction_note}
<img src="data:image/png;base64,{image}" alt="Четыре панели: границы эпизода, спектральные контроли, торговля к зоне против контролей, вход внутри против ожидания снаружи">
<h2>Что именно измеряется</h2>{''.join('<p>'+html.escape(x)+'</p>' for x in definitions)}
<h2>Края, завершённые эпизоды и последующие возвраты</h2>{flow_table}
<p>Количество событий у края и количество эпизодов имеют разные знаменатели. Таблица ниже группирует уже подтверждённые уходы по исходу PASS/REJECT и времени в ядре, известным на момент ухода; возврат после него является последующим исходом.</p>{return_table}
<p>Прямота пути — модуль смещения, делённый на пройденный путь; удаление — максимальное расстояние от ядра в период отсутствия. Оба показателя описывают уже состоявшиеся возвраты и не доступны в будущем исходе заранее.</p>{memory_table}
<h2>Семь заранее зафиксированных правил</h2>{table(['Код','Описание'], model_rows)}{account_table}
<p>Монетка выбирает направление на тех же предложенных моментах при базовых затратах; последовательные ограничения могут менять число исполненных сделок. Полнота каждой монетки указана явно; при неполной кривой хотя бы одного seed квантили не определены. P95 — эмпирический квантиль этих контролей, без коррекции за перебор гипотез. Для стресс-счёта проверяется собственный положительный результат; базовая монетка не является стресс-контролем. Незакрытая позиция прекращает счёт; частичный результат не является полной кривой капитала.</p>
<h2>Одинаковые исходные контакты: войти внутри или ждать снаружи</h2>{parent_table}
<p>WAIT включён как нулевой результат. Средние и разность посчитаны по контактам, у которых известны обе ветви; цензурированные ветви указаны отдельно. Это диагностическое сравнение правил с разными моментами входа, целями и выплатами. Оно не оценивает причинный эффект одинаковой сделки. Родительские контакты могут перекрываться: сумма их гипотетических исходов не заменяет последовательные счета на рисунке.</p>
<h2>Доступность спектра и простой контроль интервалов</h2>{spectral_table}{table(['Статус','Кандидаты'], [[k,fmt(v)] for k,v in status_rows.items()])}{table(['Контроль','Пик, с','Концентрация','Полные интервалы','CV','Поддержанный период, с'], synthetic_rows)}
<p>Отсутствие подходящих кандидатов после вето — отдельный результат. Нулевые сделки не дают основания назвать фильтр прибыльным. Для полезности спектра требуется приращение к базовой истории эпизодов и CV на новой записи с неизменными правилами.</p>
<h2>Происхождение и воспроизводимость</h2><p>Статус исходного прогона: {html.escape(manifest['status'])}. Исходный вычислительный код и все корневые parquet сверены с manifest; журналы событий каждого дня сверены с дневными квитанциями. Сохранённая конфигурация совпадает с зафиксированной исходной. Для граничного рисунка дополнительно сверен исходный файл quotes. Эпизод выбран хронологически: первый завершённый контакт квалифицированной зоны после 08:00 UTC 24 августа (при отсутствии — ближайший доступный день); прибыль в выборе не участвует.</p>
<p>Выбранный эпизод {html.escape(str(chosen.date))}/{int(chosen.episode_id)}; доступность завершения {pd.Timestamp(int(chosen.end_ns),unit='ns',tz='UTC').isoformat()}. {html.escape(str(manifest['economic_admission']))}</p>
{quarantine_section}
<details><summary>Сохранённая конфигурация</summary><pre>{html.escape(json.dumps(cfg,ensure_ascii=False,indent=2))}</pre></details>
<p class="subtitle">Renderer: {html.escape(sys.executable)}. Хеши и результаты проверки находятся в renderer_receipt.json рядом с отчётом.</p>"""
    document = '<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Память эпизодов зоны</title><style>body{font:16px/1.5 system-ui,sans-serif;color:#24313b;background:#f6f7f8;margin:0}main{max-width:1450px;margin:auto;padding:36px 24px}h1,h2{line-height:1.2;font-weight:600}h2{margin-top:36px}p{max-width:1100px}.subtitle{color:#566572}.lead{font-size:20px}img{width:100%;height:auto;background:white;margin:20px 0}.table-wrap{overflow:auto;margin:16px 0}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:9px 10px;text-align:right;border-bottom:1px solid #dce2e6;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#e9eef1;font-weight:600}pre{white-space:pre-wrap;font-size:13px}details{margin:24px 0}@media(max-width:600px){main{padding:20px 12px}.lead{font-size:18px}h1{font-size:27px}}</style></head><body><main>'+sections+'</main></body></html>'
    report = out/"report.html"
    report.write_text(document, encoding="utf-8", newline="\n")
    receipt = dict(status="RENDERED_PENDING_VISUAL_REVIEW", input_sha256=inputs,
                   renderer_sha256=digest(__file__), spectral_renderer_sha256=digest(Path(__file__).with_name("spectral.py")),
                   output_sha256={report.name:digest(report),figure.name:digest(figure)},
                   selected_episode=dict(date=str(chosen.date), episode_id=int(chosen.episode_id), available_ns=int(chosen.end_ns), selection="FIRST_COMPLETED_QUALIFIED_AFTER_08_UTC_NOT_PNL"),
                   synthetic_controls=[dict(name=name, **{k:v for k,v in result.items() if k not in ("frequency_hz","power_fraction")}) for name,result in synthetic],
                   return_profiles=profiles, economic_admission="NOT_ESTABLISHED", renderer_python=sys.executable,
                   quarantined_previous_runs=quarantined,
                   figure=str(figure), report=str(report), visual_inspection="PENDING")
    (out/"renderer_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return {"report":str(report), "figure":str(figure), "receipt":str(out/"renderer_receipt.json"), "status":receipt["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(render(args.directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
