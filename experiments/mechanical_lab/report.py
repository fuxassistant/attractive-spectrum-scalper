"""Render a completed mechanical-lab run as a self-contained Russian report.

Run with the plotting environment, for example:
  .venv/Scripts/python.exe -m experiments.mechanical_lab.report RUN_DIRECTORY
The renderer only reads frozen experiment outputs; editing recipe JSON in the
HTML never changes or recomputes the historical metrics.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = {
    "discovery": "Поиск · 3–14 августа",
    "validation": "Выбор · 17–21 августа",
    "diagnostic_test": "Диагностика · 24–28 августа",
    "extension": "Продолжение · 31 августа–1 сентября",
}
SIZING = {"fixed": "1 контракт", "martingale_capped": "Мартингейл 1→2→4", "anti_martingale": "Антимартингейл 1→2→4"}
DIRECTIONS = {"signal": "Сигнал", "reverse": "Обратная сторона", "stress": "Стресс исполнения"}
SHORT = {
    "bb_breakout": "BB breakout", "bb_reentry": "BB reentry", "bb_squeeze_breakout": "BB squeeze",
    "range_breakout": "Range breakout", "range_failure": "Range failure", "momentum_second": "Momentum 2",
    "large_candle_fade": "Large candle fade", "wick_rejection": "Wick rejection", "ema_pullback": "EMA pullback",
    "opening_range_breakout": "Opening range break", "opening_range_failure": "Opening range failure",
    "inside_bar_breakout": "Inside bar break", "three_bar_reversal": "3-bar reversal",
    "trend_pullback_resume": "Pullback resume", "stretch_reversal": "Stretch reversal",
    "compression_release": "Compression release",
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _stop_trace(quotes, outcome, plan, execution):
    """Independent display replay, checked against saved execution results.

    The returned thresholds are observable BBO prices, not promised fills.
    Updating ceases once any exit is requested; pending quotes cannot retract it.
    """
    side = int(outcome.side)
    e = int(outcome.entry_ordinal)
    trigger = int(outcome.trigger_ordinal)
    fee = float(execution["fee_roundtrip_per_oz"])
    slip = float(execution["slippage_ticks_per_fill"]) * .1
    risk = float(outcome.risk_per_oz)
    qt = quotes.available_ns.to_numpy(np.int64)
    bid = quotes.bid_nano.to_numpy(np.int64)
    ask = quotes.ask_nano.to_numpy(np.int64)
    valid = quotes.valid.to_numpy(bool)
    entry_nano = int(ask[e] if side > 0 else bid[e])
    end = trigger + 1 if trigger >= 0 else int(np.searchsorted(qt, int(outcome.trigger_ns), "left"))
    stop, peak, changes = -risk, 0., 0
    times, stops = [int(outcome.entry_ns)], [stop]
    observed_trigger = None
    for j in range(e, end):
        if not valid[j]:
            raise ValueError("Cannot display a stop through an invalid quote")
        exit_nano = int(bid[j] if side > 0 else ask[j])
        mark = side * (exit_nano - entry_nano) / 1e9 - fee - 2 * slip
        peak = max(peak, mark)
        if mark <= stop + 1e-12 or mark >= plan["take_r"] * risk - 1e-12:
            observed_trigger = j
            break
        new_stop = stop
        if plan["stop_mode"] in ("breakeven", "ratchet") and mark >= plan["be_trigger_r"] * risk - 1e-12:
            new_stop = max(new_stop, plan["be_lock_r"] * risk)
        if plan["stop_mode"] == "ratchet" and peak >= plan["trail_start_r"] * risk - 1e-12:
            lock = np.floor((peak / risk - plan["trail_distance_r"]) / plan["trail_step_r"] + 1e-12) * plan["trail_step_r"] * risk
            new_stop = max(new_stop, lock)
        if plan["stop_mode"] == "chandelier" and peak >= plan["trail_start_r"] * risk - 1e-12:
            new_stop = max(new_stop, peak - plan["trail_distance_r"] * risk)
        if new_stop > stop + 1e-12:
            stop = new_stop
            changes += 1
            times.append(int(qt[j]))
            stops.append(stop)
    if trigger >= 0 and observed_trigger != trigger:
        raise ValueError("Independent stop replay disagrees with the recorded barrier event")
    if trigger < 0 and observed_trigger is not None:
        raise ValueError("Independent stop replay found an earlier barrier than recorded timer")
    if not np.isclose(stop, outcome.final_stop_net_per_oz, atol=1e-10, rtol=0) or changes != int(outcome.stop_updates):
        raise ValueError("Independent migrating stop does not match the saved final stop and update count")
    times.extend([int(outcome.trigger_ns), int(outcome.exit_ns)])
    stops.extend([stop, stop])
    prices = float(outcome.entry_price) + side * (np.asarray(stops) + fee + slip)
    take_price = float(outcome.entry_price) + side * (plan["take_r"] * risk + fee + slip)
    return np.asarray(times, dtype=np.int64), prices, float(take_price), dict(
        verified=True, stop_updates=changes, final_stop_net_per_oz=float(stop),
        trigger_ordinal=trigger, display_update_count=len(times),
        pending_exit_stop_frozen=True)


def _records(frame):
    # Pandas handles NaN/NumPy scalars without leaking non-standard JSON NaN.
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def _embed(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _style(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.facecolor": "#fbfcfe", "figure.facecolor": "white",
                         "axes.grid": True, "grid.alpha": .15, "savefig.facecolor": "white"})


def _plots(root, results, coins, trades, regimes, selected, config, catalog):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    import pyarrow.parquet as pq

    _style(plt)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    signals = [r["signal"] for r in selected]
    shown = results.loc[(results.split == "diagnostic_test") & (results.sizing == "fixed") & (results.direction == "signal")].set_index("signal").reindex(signals)
    coin_test = coins.loc[coins.split == "diagnostic_test"]
    fig, ax = plt.subplots(figsize=(14.4, 8.4))
    ys = np.arange(len(signals))
    complete = shown.capital_curve_complete.fillna(False).to_numpy(bool)
    colors = np.where(complete, np.where(shown.net_usd >= 0, "#178369", "#c8544d"), "#c98a28")
    bars = ax.barh(ys, shown.net_usd, color=colors, alpha=.80, height=.57)
    for i, signal in enumerate(signals):
        if not complete[i]:
            bars[i].set_hatch("///")
        c = coin_test.loc[(coin_test.signal == signal) & coin_test.capital_curve_complete]
        if len(c):
            median, high = c.net_usd.median(), c.net_usd.quantile(.95)
            ax.plot([median, high], [i, i], color="#283b55", lw=2.1)
            ax.scatter([median, high], [i, i], marker="|", s=150, c="#283b55", zorder=4)
        ax.text(1.01, i, f"n={int(shown.iloc[i].trades)} · coin {len(c)}/{config['coin_seeds']}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.5)
    ax.set_yticks(ys, [SHORT.get(s, s) for s in signals])
    ax.invert_yaxis()
    ax.axvline(0, color="#586579", lw=.8)
    ax.set_xlabel("Результат закрытых сделок, USD · 1 GC-контракт · комиссия и BBO включены")
    ax.set_title("Одна выбранная конфигурация на идею: диагностическая неделя", loc="left", weight="bold", pad=15)
    ax.legend(handles=[Line2D([0], [0], color="#283b55", lw=2, marker="|", label="Медиана → p95 только завершённых монеток")], loc="lower left", fontsize=8.5)
    fig.text(.03, .018, "Штриховка: счёт прерван неразрешённой позицией; показан лишь закрытый префикс. При неполных монетках p95 не является порогом допуска.", fontsize=9, color="#825015")
    fig.subplots_adjust(left=.20, right=.80, top=.91, bottom=.12)
    fig.savefig(figures / "comparison.png", dpi=145)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14.4, 7.0))
    colors = plt.cm.tab20(np.linspace(0, 1, len(signals)))
    initial = config["account"]["initial_capital_usd"]
    start = pd.Timestamp(config["splits"]["diagnostic_test"][0], tz="UTC") + pd.Timedelta(hours=7)
    for i, signal in enumerate(signals):
        t = trades.loc[(trades.signal == signal) & (trades.split == "diagnostic_test") & (trades.sizing == "fixed") & (trades.direction == "signal")].sort_values("exit_ns")
        if t.empty:
            continue
        clocks = [start] + pd.to_datetime(t.exit_ns, utc=True).tolist()
        values = np.r_[0., t.equity_usd.to_numpy() - initial]
        ax.step(clocks, values, where="post", color=colors[i], lw=1.2,
                linestyle="-" if complete[i] else "--", label=SHORT.get(signal, signal))
        if not complete[i]:
            ax.scatter(clocks[-1], values[-1], color=colors[i], marker="x", s=55, zorder=5)
    ax.axhline(0, color="#586579", lw=.8)
    ax.set_title("Траектории закрытого результата · выбранные рецепты · 1 контракт", loc="left", weight="bold", pad=13)
    ax.set_ylabel("Изменение денежных средств, USD")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M", tz="UTC"))
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False, fontsize=8)
    fig.text(.03, .025, "Кривые обновляются при закрытиях, поэтому внутрисделочная просадка здесь не видна. × — последний закрытый результат перед неразрешённой позицией.", fontsize=9)
    fig.subplots_adjust(bottom=.25, left=.09, right=.98, top=.90)
    fig.savefig(figures / "equity.png", dpi=145)
    plt.close(fig)

    r = regimes.loc[regimes.split == "diagnostic_test"]
    regime_names = sorted(r.regime.unique())
    matrix = r.pivot(index="signal", columns="regime", values="mean_usd").reindex(index=signals, columns=regime_names)
    counts = r.pivot(index="signal", columns="regime", values="trades").reindex(index=signals, columns=regime_names).fillna(0)
    cap = max(1., float(np.nanmax(np.abs(matrix.to_numpy()))) if matrix.notna().any().any() else 1.)
    fig, ax = plt.subplots(figsize=(14.4, 9.5))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#e7ebef")
    heat = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), cmap=cmap, vmin=-cap, vmax=cap, aspect="auto")
    for i in range(len(signals)):
        for j in range(len(regime_names)):
            n = int(counts.iloc[i, j])
            label = f"{matrix.iloc[i,j]:+.0f}\nn={n}" if n else "—"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="#192634")
    ax.set_xticks(range(len(regime_names)), regime_names, rotation=25, ha="right")
    ax.set_yticks(range(len(signals)), [SHORT.get(s, s) for s in signals])
    ax.grid(False)
    ax.set_title("Режим на входе: средний результат сделки, USD, и число сделок", loc="left", weight="bold", pad=15)
    fig.colorbar(heat, ax=ax, shrink=.7, label="USD / закрытую сделку")
    fig.text(.03, .015, "Режимы причинные: эффективность движения за 20 минут × ATR14 / средний ATR предыдущих 100 минут. Малые n не доказывают устойчивость.", fontsize=9)
    fig.subplots_adjust(left=.20, bottom=.15, right=.95, top=.91)
    fig.savefig(figures / "regimes.png", dpi=145)
    plt.close(fig)

    # Deterministic examples: the first completed diagnostic trade in these
    # four predeclared families, never largest win/loss selected after outcome.
    preferred = ["bb_breakout", "bb_reentry", "range_breakout", "large_candle_fade"]
    fixed = trades.loc[(trades.split == "diagnostic_test") & (trades.sizing == "fixed") & (trades.direction == "signal")].sort_values("entry_ns")
    examples = []
    for signal in preferred + [s for s in signals if s not in preferred]:
        part = fixed.loc[fixed.signal == signal]
        if len(part):
            examples.append(part.iloc[0])
        if len(examples) == 4:
            break
    nrows = 1 if len(examples) <= 2 else 2
    ncols = 1 if len(examples) <= 1 else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.4, 5.8 if nrows == 1 else 9.3), squeeze=False)
    snapshot_records = []
    quote_cache = {}
    for ax, trade in zip(axes.flat, examples):
        day = root / "days" / str(trade.date)
        receipt = _read_json(day / "receipt.json")
        source = Path(receipt["source"])
        for name in ("features.parquet", "outcomes.parquet"):
            expected = receipt.get("artifact_sha256", {}).get(name)
            if expected and _digest(day / name) != expected:
                raise ValueError(f"Snapshot input identity mismatch: {day / name}")
        f = pd.read_parquet(day / "features.parquet")
        a = int(trade.entry_ns) - 15 * 60 * 10**9
        b = int(trade.exit_ns) + 10 * 60 * 10**9
        f = f.loc[f.bar_end_ns.between(a, b)]
        times = pd.to_datetime(f.bar_end_ns, utc=True)
        ax.plot(times, f.c, color="#485e78", lw=1.5, label="M1 midpoint close")
        ax.plot(times, f.bb_upper, color="#8b9bac", lw=.9, linestyle="--", label="BB20 ±2σ")
        ax.plot(times, f.bb_lower, color="#8b9bac", lw=.9, linestyle="--")
        # Native BBO gives execution context inside the held interval; capped
        # point count is a display sampling choice, never a simulation input.
        qa, qb = int(trade.entry_ns) - 60 * 10**9, int(trade.exit_ns) + 60 * 10**9
        if str(source) not in quote_cache:
            if _digest(source / "quotes.parquet") != receipt["payloads"]["quotes.parquet"]:
                raise ValueError("Snapshot native quote identity mismatch")
            quote_cache[str(source)] = pq.read_table(source / "quotes.parquet", columns=["available_ns", "bid_nano", "ask_nano", "valid"]).to_pandas()
        qfull = quote_cache[str(source)]
        q = qfull.loc[qfull.available_ns.between(qa, qb)]
        q = q.loc[q.valid]
        stride = max(1, int(np.ceil(len(q) / 8000)))
        q = q.iloc[::stride]
        qt = pd.to_datetime(q.available_ns, utc=True)
        ax.plot(qt, q.bid_nano / 1e9, color="#2f8f81", lw=.6, alpha=.75, label="BBO bid / ask")
        ax.plot(qt, q.ask_nano / 1e9, color="#c17838", lw=.6, alpha=.75)
        outcomes = pd.read_parquet(day / "outcomes.parquet")
        outcome = outcomes.loc[(outcomes.origin_index == trade.origin_index) & (outcomes.side == trade.side) & (outcomes.plan == trade.plan)].iloc[0]
        plan = catalog["exits"][trade.plan]
        if config.get("recipe"):
            plan = dict(plan, **config["recipe"]["exit"].get("parameters", {}))
        stop_times, stop_prices, take_price, stop_review = _stop_trace(qfull, outcome, plan, config["execution"])
        entry_time, exit_time = pd.to_datetime(int(trade.entry_ns), utc=True), pd.to_datetime(int(trade.exit_ns), utc=True)
        ax.scatter([entry_time], [outcome.entry_price], c="#157c60", marker="^" if trade.side > 0 else "v", s=70, zorder=7, label="Фактический entry")
        ax.scatter([exit_time], [outcome.exit_price], c="#b94743", marker="x", s=65, linewidths=2, zorder=7, label="Фактический exit")
        ax.axvspan(entry_time, exit_time, color="#8bb8e0", alpha=.10)
        ax.step(pd.to_datetime(stop_times, utc=True), stop_prices, where="post", color="#a9437f", lw=1.5, label="Стоп: BBO-порог")
        ax.axvline(pd.to_datetime(int(outcome.trigger_ns), utc=True), color="#a9437f", lw=.8, linestyle=":", alpha=.7)
        ymin, ymax = ax.get_ylim()
        if ymin - .30 * (ymax - ymin) <= take_price <= ymax + .30 * (ymax - ymin):
            ax.plot([entry_time, exit_time], [take_price, take_price], color="#288c5f", lw=1., linestyle="--", label="TP: BBO-порог")
        else:
            ax.text(.98, .03, f"TP вне кадра: {take_price:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#237e55", bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
        if stop_review["stop_updates"] > 0:
            # A minute-scale plot can hide sub-second stop migration. Show
            # the first tightening through the actual exit at native cadence.
            first_update = int(stop_times[1])
            qa = first_update - min(200_000_000, max(1, int(outcome.exit_ns) - first_update) // 3)
            qb = int(outcome.exit_ns) + min(100_000_000, max(1, int(outcome.exit_ns) - first_update) // 3)
            zoom = qfull.loc[qfull.available_ns.between(qa, qb)]
            inset = ax.inset_axes([.055, .62, .43, .30])
            relative = (zoom.available_ns.to_numpy(np.int64) - first_update) / 1e9
            inset.plot(relative, zoom.bid_nano / 1e9, color="#2f8f81", lw=.65)
            inset.plot(relative, zoom.ask_nano / 1e9, color="#c17838", lw=.65)
            inset.step((stop_times[1:] - first_update) / 1e9, stop_prices[1:], where="post", color="#a9437f", lw=1.4)
            inset.axvline((int(outcome.trigger_ns) - first_update) / 1e9, color="#a9437f", lw=.7, linestyle=":")
            inset.scatter([(int(outcome.exit_ns) - first_update) / 1e9], [outcome.exit_price], c="#b94743", marker="x", s=20, zorder=5)
            inset.set_title(f"Стоп: {stop_review['stop_updates']} сдвига · секунды от первого", fontsize=7, loc="left", pad=3)
            inset.tick_params(labelsize=6, pad=1)
            inset.set_facecolor("white")
            inset.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_title(f"{SHORT.get(trade.signal, trade.signal)} · {'LONG' if trade.side > 0 else 'SHORT'}\n{trade.date} · {trade.plan} · {trade.reason} · {trade.pnl_usd:+.0f} USD", loc="left", fontsize=10.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
        ax.set_ylabel("USD/oz")
        ax.tick_params(axis="x", labelsize=8)
        snapshot_records.append(dict(signal=trade.signal, date=str(trade.date), origin_index=int(trade.origin_index), side=int(trade.side),
                                     selection_rule="first completed diagnostic fixed-size trade in predeclared family", display_quote_stride=stride,
                                     entry_ns=int(trade.entry_ns), exit_ns=int(trade.exit_ns), entry_price=float(outcome.entry_price), exit_price=float(outcome.exit_price), stop_trace=stop_review))
    for ax in list(axes.flat)[len(examples):]:
        ax.set_visible(False)
    if examples:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, fontsize=7.6)
    fig.suptitle(f"Воспроизводимые снимки сделки: {len(examples)} · ex-post визуальный аудит", x=.055, ha="left", weight="bold", fontsize=14)
    fig.text(.055, .047, "Это иллюстрации исполнения после события. Будущие цены не были признаками. BBO может быть прорежен только при рисовании; время на оси UTC.", fontsize=9)
    fig.subplots_adjust(left=.07, right=.98, top=.85 if nrows == 1 else .89,
                        bottom=.20 if nrows == 1 else .12, hspace=.35, wspace=.20)
    fig.savefig(figures / "snapshots.png", dpi=145)
    plt.close(fig)
    (root / "snapshot_review.json").write_text(json.dumps(snapshot_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {name: figures / f"{name}.png" for name in ("comparison", "equity", "regimes", "snapshots")}


def render(directory):
    root = Path(directory).resolve()
    required = ["results.parquet", "admission.json", "trades.parquet", "coins.parquet", "regimes.parquet", "selected.json", "evaluation.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"Wait for the completed evaluation before rendering: {missing}")
    evaluation = _read_json(root / "evaluation.json")
    if _read_json(root / "manifest.json")["status"] != "COMPLETE" or evaluation["status"] != "COMPLETE":
        raise RuntimeError("Only COMPLETE generation and evaluation may be rendered")
    for name, expected in evaluation.get("artifact_sha256", {}).items():
        if Path(name).name != name or _digest(root / name) != expected:
            raise ValueError(f"Evaluation artifact identity mismatch: {name}")
    if (root / "stress").exists() and _read_json(root / "stress/manifest.json")["status"] != "COMPLETE":
        raise RuntimeError("Stress generation is not COMPLETE")
    results = pd.read_parquet(root / "results.parquet")
    coins = pd.read_parquet(root / "coins.parquet")
    trades = pd.read_parquet(root / "trades.parquet")
    regimes = pd.read_parquet(root / "regimes.parquet")
    selected = _read_json(root / "selected.json")
    config = _read_json(root / "frozen_config.json")
    catalog = _read_json(root / "catalog.json")
    admission = _read_json(root / "admission.json")
    plots = _plots(root, results, coins, trades, regimes, selected, config, catalog)
    coin_summary = []
    for (signal, split), group in coins.groupby(["signal", "split"]):
        complete = group.loc[group.capital_curve_complete]
        coin_summary.append(dict(signal=signal, split=split, complete=len(complete), total=len(group),
            median=float(complete.net_usd.median()) if len(complete) else None,
            p95=float(complete.net_usd.quantile(.95)) if len(complete) else None))
    grouped = trades.groupby(["signal", "split", "sizing", "direction", "regime"], as_index=False).agg(
        trades=("pnl_usd", "size"), mean_usd=("pnl_usd", "mean"), net_usd=("pnl_usd", "sum"), win_rate=("pnl_usd", lambda x: (x > 0).mean()))
    data = dict(results=_records(results), coins=coin_summary, regimes=_records(grouped), selected=selected,
                admission=admission, config=config, catalog=catalog, labels=SHORT, splits=SPLITS, sizing=SIZING, directions=DIRECTIONS)
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    pngs = {key: _embed(path) for key, path in plots.items()}
    template = r'''<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Механические идеи · лаборатория GCZ6</title><style>
:root{--ink:#182838;--muted:#617185;--line:#dce4ec;--green:#167d65;--red:#b64643;--amber:#a56a18}*{box-sizing:border-box}body{margin:0;background:#f0f4f8;color:var(--ink);font:15px/1.55 system-ui,Segoe UI,sans-serif}main{max-width:1460px;margin:auto;padding:30px}h1{font-size:38px;line-height:1.12;margin:10px 0 16px;letter-spacing:-1px}h2{font-size:24px;margin:0 0 16px}h3{font-size:17px;margin:0 0 10px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:2px;color:var(--muted)}.lead{max-width:1040px;font-size:18px;color:#405269}.panel{background:#fff;border:1px solid var(--line);border-radius:14px;padding:24px;margin:20px 0}.notice{border-left:5px solid var(--amber);background:#fff7e8;padding:15px 18px;border-radius:7px}.controls{display:grid;grid-template-columns:2fr 1.5fr 1.4fr 1.2fr;gap:12px}label{display:block;color:var(--muted);font-size:12px;font-weight:600}select,textarea,button{font:inherit}select{width:100%;margin-top:6px;padding:10px;border:1px solid #becbd8;border-radius:7px;background:white;color:var(--ink)}.metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:18px 0}.metric{border:1px solid var(--line);border-radius:9px;padding:13px}.metric small{display:block;color:var(--muted);font-size:12px}.metric strong{font-size:22px;font-weight:650;display:block;margin-top:4px}.muted{color:var(--muted)}.good{color:var(--green)}.bad{color:var(--red)}.amber{color:var(--amber)}.layout{display:grid;grid-template-columns:1.1fr 1fr;gap:24px}.modules{line-height:1.8;overflow-wrap:anywhere}.modules code{font-size:12px;background:#f0f4f8;padding:3px 5px;border-radius:4px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f5f7fa;color:#506278;font-weight:600}tbody tr:hover{background:#f2f6fa}.scroll{overflow:auto}.plot{display:block;width:100%;height:auto;margin:8px 0}.caption{font-size:13px;color:var(--muted)}textarea{display:block;width:100%;min-height:280px;resize:vertical;border:1px solid #bccbd9;border-radius:7px;padding:13px;background:#f8fafc;font:13px/1.5 Consolas,monospace}button{border:0;border-radius:7px;background:#234f75;color:white;padding:10px 16px;cursor:pointer;margin:12px 8px 0 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f7;border-radius:7px;padding:13px;font-size:12px}.pill{display:inline-block;border-radius:20px;padding:3px 10px;background:#fff0d2;color:#85530e;font-size:12px}.gates{display:flex;gap:7px;flex-wrap:wrap}.gate{font-size:12px;padding:4px 8px;background:#f2f4f7;border-radius:5px}.footer{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.compact{margin:8px 0}.hidden{display:none}@media(max-width:1000px){.controls,.layout{grid-template-columns:1fr 1fr}.metrics{grid-template-columns:repeat(3,1fr)}main{padding:16px}}@media(max-width:620px){h1{font-size:30px}.controls,.layout{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.panel{padding:16px}}
</style></head><body><main>
<div class="eyebrow">GCZ6 · M1-предложения · native BBO execution</div><h1>Лаборатория механических идей</h1>
<p class="lead">__LEAD__</p>
<div class="notice"><strong>Экономический допуск не установлен.</strong> Неделя 24–28 августа уже рассматривалась в предыдущем аудите; она является диагностической, а не нетронутым слепым тестом. Пять дней диагностики и два дня продолжения недостаточны для утверждения устойчивого преимущества. Незавершённая позиция прерывает счёт; её нельзя заменить нулевой доходностью.</div>
<section class="panel"><h2>Сохранённые результаты и конструктор</h2><div class="controls">
<label>Идея<select id="signal"></select></label><label>Период<select id="split"></select></label><label>Размер позиции<select id="sizing"></select></label><label>Проверка стороны / исполнения<select id="direction"></select></label></div>
<div id="status" class="compact"></div><div class="metrics" id="metrics"></div>
<div class="layout"><div><h3>Выбранные модули</h3><div id="modules" class="modules"></div><p id="coin" class="notice"></p><h3>Причинные режимы на входе</h3><div id="regimes" class="scroll"></div><p class="caption">Среднее считается по закрытым сделкам выбранного счёта. При прерванном счёте это только наблюдённый префикс, а небольшие группы не являются самостоятельным подтверждением.</p></div>
<div><h3>Рецепт для следующего запуска</h3><textarea id="recipe" spellcheck="false"></textarea><button id="download">Скачать recipe.json</button><button id="restore">Вернуть сохранённый рецепт</button><div id="editorStatus" class="caption"></div><p class="caption">Редактирование JSON подготавливает новый эксперимент. Показанные метрики принадлежат сохранённому запуску и не пересчитываются в браузере.</p><pre>.tmp\baseline\venv\Scripts\python.exe -m experiments.mechanical_lab.run run custom_v1 --recipe recipe.json --workers 2</pre><h3>Проверки выбранной идеи</h3><div id="gates" class="gates"></div><p class="caption">Даже прохождение всех исследовательских проверок не меняет статус NOT_ESTABLISHED: необходимы новые, ранее не исследованные даты.</p></div></div></section>
<section class="panel"><h2>Все выбранные идеи</h2><p class="caption">Таблица следует выбранным периоду, размеру и проверке. __RECIPE_SELECTION__ Сортировка по названию, а не по поздней прибыли.</p><div id="allResults" class="scroll"></div></section>
<section class="panel"><h2>Сравнение с монеткой</h2><img class="plot" src="__COMPARISON__" alt="Результаты шестнадцати выбранных идей и распределения монеток"><p class="caption">Одинаковые исходные возможности и выбранный выход; монетка меняет сторону. Из-за разного времени удержания, перекрытий, ограничений риска и цензурирования фактическое число сделок может различаться. Показанные квантили используют только завершённые счёта; если завершены не все 199, порог допуска по монетке не определён. WAIT даёт 0 при отсутствии позиции.</p></section>
<section class="panel"><h2>Закрытый результат во времени</h2><img class="plot" src="__EQUITY__" alt="Траектории результата закрытых сделок"><p class="caption">Каждый период начинается со свежего счёта 100 000 USD. Показанная кривая денежных средств и максимальная просадка — разные величины: просадка учитывает native BBO внутри сделок. Пунктир заканчивается перед неразрешённой позицией, её дальнейшая стоимость неизвестна.</p></section>
<section class="panel"><h2>Результаты по режимам</h2><img class="plot" src="__REGIMES__" alt="Матрица причинных режимов с числом сделок и средним результатом"></section>
<section class="panel"><h2>Визуальный аудит исполнения</h2><img class="plot" src="__SNAPSHOTS__" alt="Снимки сделок с фактическими заполнениями и ценами"><p class="caption">__SNAPSHOT_RULE__ Это правило не выбирает лучшие или худшие исходы. Точки входа и выхода берутся из сохранённых результатов native BBO. Фиолетовый стоп независимо восстановлен на каждой исходной публикации; финальный уровень и число изменений совпали с сохранённым исполнением. После запроса выхода стоп больше не меняется. Врезка раскрывает быстрые сдвиги, невидимые на минутном масштабе. Графики после момента решения предназначены только для аудита.</p></section>
<section class="panel"><h2>Что именно проверялось</h2><div class="layout"><div><p><strong>Причинность.</strong> Сигнал доступен после публикации завершённой M1. ATR14, BB20 с ddof=1, EMA и диапазоны используют доступное прошлое. Разрыв, неполная минута и смена epoch запускают прогрев заново. Диапазон пробоя исключает текущую свечу. Отбор контракта должен быть известен до решения.</p><p><strong>Исполнение.</strong> Вход и все выходы требуют новой публикации после задержки 100 мс; take/stop — запросы, а не гарантированные цены. Комиссия 0.30 USD/oz за оборот, то есть 30 USD за GC-контракт, является предположением. Измеренный bid/ask уже включает спред. Стресс: 250 мс и дополнительный тик на каждое заполнение.</p></div><div><p><strong>Размер позиции.</strong> Один контракт — базовое сравнение. Мартингейл ограничен 1→2→4 и ожиданием до следующего дня после убытка на четырёх; антимартингейл увеличивает объём после выигрыша. Лимиты риска применяются к обеим схемам. Эти модули меняют распределение риска и не создают доказательство преимущества входа.</p><p><strong>Ограничения.</strong> Предполагается доступное исполнение по BBO без моделирования глубины, очереди, рыночного воздействия и маржи. Проверено фиксированное UTC-окно 07:00–20:30. Поздние положительные значения и отдельные режимы требуют новой проверки; многократный перебор не исчезает от хорошего p-value.</p></div></div></section>
<p class="footer">Артефакт: __ROOT__ · Все изображения и данные встроены. Источник параметров: frozen_config.json; рецептов: selected.json; допуска: admission.json. Внешние библиотеки и сеть для просмотра не требуются.</p>
</main><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,n=0)=>v==null?'—':Number(v).toLocaleString('ru-RU',{minimumFractionDigits:n,maximumFractionDigits:n});
const money=v=>num(v,0)+' $',pct=v=>num(100*v,1)+'%';
function options(id,entries,selected){$(id).innerHTML=entries.map(([v,l])=>`<option value="${esc(v)}">${esc(l)}</option>`).join('');if(selected)$(id).value=selected;}
options('signal',D.selected.map(x=>[x.signal,D.labels[x.signal]||x.signal]));options('split',Object.entries(D.splits),'diagnostic_test');options('sizing',Object.entries(D.sizing),D.config.recipe?.sizing||'fixed');options('direction',Object.entries(D.directions),'signal');
const names={selection_eligible:'Пригоден ранний отбор',n_and_days:'Число сделок и дней',positive_validation_and_test:'Плюс выбор + диагностика',complete_accounts:'Завершённые счета',beats_coin_p95:'Выше p95 монеток',drawdown:'Лимит просадки',stress_positive_complete:'Плюс завершённый стресс'};
function recipe(){const s=$('signal').value,r=D.selected.find(x=>x.signal===s),custom=D.config.recipe,entryOverrides=custom?.entry?.id===s?custom.entry.parameters||{}:{},exitOverrides=custom?.exit?.id===r.plan?custom.exit.parameters||{}:{};return {entry:{id:s,parameters:{...D.catalog.entries[s].parameters,...entryOverrides}},veto:{id:r.veto},exit:{id:r.plan,parameters:{...exitOverrides}},sizing:$('sizing').value};}
function resetRecipe(){$('recipe').value=JSON.stringify(recipe(),null,2);$('editorStatus').textContent='Редактируемый рецепт нового запуска. Исторические метрики не меняются.';}
function table(headers,rows){return '<table><thead><tr>'+headers.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';}
function render(){const signal=$('signal').value,split=$('split').value,sizing=$('sizing').value,direction=$('direction').value,r=D.selected.find(x=>x.signal===signal),a=D.admission.find(x=>x.signal===signal),z=D.results.find(x=>x.signal===signal&&x.split===split&&x.sizing===sizing&&x.direction===direction);
$('status').innerHTML=z?(z.capital_curve_complete?'<span class="pill good">Счёт завершён</span>':'<span class="pill">Счёт прерван: результат только закрытого префикса</span>'):'<span class="pill">Это сочетание не рассчитывалось</span>';
const metrics=z?[['Закрытый результат',money(z.net_usd),z.net_usd>=0?'good':'bad'],['Сделки',num(z.trades),''],['Средняя сделка',money(z.mean_usd),z.mean_usd>=0?'good':'bad'],['Доля прибыльных',pct(z.win_rate),''],['Макс. просадка',money(z.max_drawdown_usd),''],['Цензурировано / открыто',num(z.censored||0)+' / '+num(z.unresolved||0),(z.unresolved||0)?'amber':'']]:[];
$('metrics').innerHTML=metrics.map(([l,v,c])=>`<div class="metric"><small>${esc(l)}</small><strong class="${c}">${esc(v)}</strong></div>`).join('');
const savedRecipe=recipe(),shownPlan={...D.catalog.exits[r.plan],...savedRecipe.exit.parameters};$('modules').innerHTML='<div><code>'+esc(signal)+'</code> '+esc(D.catalog.entries[signal].description_ru)+' '+esc(JSON.stringify(savedRecipe.entry.parameters))+'</div><div><code>'+esc(r.veto)+'</code> '+esc(D.catalog.vetos[r.veto])+'</div><div><code>'+esc(r.plan)+'</code> '+esc(JSON.stringify(shownPlan))+'</div><div><code>'+esc(sizing)+'</code> '+esc(D.sizing[sizing])+'</div>';
const c=D.coins.find(x=>x.signal===signal&&x.split===split);$('coin').textContent=sizing!=='fixed'?'Монетка рассчитана для одного контракта; её денежный результат нельзя напрямую сравнивать с увеличенным объёмом.':c?`Монетка: завершено ${c.complete}/${c.total}; медиана ${money(c.median)}, p95 ${money(c.p95)}. ${c.complete===c.total?'Все контрольные счета завершены.':'Квантили условны по завершённой подвыборке; критерий p95 заблокирован.'}`:'Монетка рассчитана для диагностического периода и продолжения.';
const rr=D.regimes.filter(x=>x.signal===signal&&x.split===split&&x.sizing===sizing&&x.direction===direction);$('regimes').innerHTML=rr.length?table(['Режим','n','Среднее','Сумма','Плюсовые'],rr.map(x=>[x.regime,num(x.trades),money(x.mean_usd),money(x.net_usd),pct(x.win_rate)])):'<p class="muted">Нет сохранённых закрытых сделок для этой проверки.</p>';
$('gates').innerHTML=Object.entries(a.gates).map(([k,v])=>`<span class="gate ${v?'good':'bad'}">${v?'✓':'×'} ${esc(names[k]||k)}</span>`).join('');
const rows=D.results.filter(x=>x.split===split&&x.sizing===sizing&&x.direction===direction).sort((a,b)=>a.signal.localeCompare(b.signal));$('allResults').innerHTML=table(['Идея','Вето / выход','n','Дни','Среднее','Сумма','Плюсовые','Просадка','Открыто','Счёт'],rows.map(x=>[D.labels[x.signal]||x.signal,x.veto+' / '+x.plan,num(x.trades),num(x.active_days),money(x.mean_usd),money(x.net_usd),pct(x.win_rate),money(x.max_drawdown_usd),num(x.unresolved||0),x.capital_curve_complete?'завершён':'префикс']));resetRecipe();}
for(const id of ['signal','split','sizing','direction'])$(id).addEventListener('change',render);$('restore').addEventListener('click',resetRecipe);
$('download').addEventListener('click',()=>{try{const r=JSON.parse($('recipe').value);if(!D.catalog.entries[r.entry?.id]||!D.catalog.vetos[r.veto?.id]||!D.catalog.exits[r.exit?.id]||!D.sizing[r.sizing])throw Error('Неизвестный модуль');const blob=new Blob([JSON.stringify(r,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='recipe.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('editorStatus').textContent='JSON скачан. Числовые ограничения дополнительно проверит CLI.';}catch(e){$('editorStatus').textContent='Ошибка рецепта: '+e.message;}});render();
</script></body></html>'''
    custom = bool(config.get("recipe"))
    substitutions = {"__DATA__": payload, "__ROOT__": html.escape(str(root)),
        "__LEAD__": ("Фиксированный пользовательский рецепт объединяет вход, вето, выход и размер позиции. Сохранены изменённые числовые параметры; периоды показывают результат, обратную сторону, монетку и стресс исполнения." if custom else "Вход, вето, выход и размер позиции собраны в отдельные модули. Исторический перебор выбирает одну конфигурацию каждой идеи на ранних периодах; следующие даты показывают её результат, обратную сторону, монетку и стресс исполнения."),
        "__RECIPE_SELECTION__": ("Конфигурация входа/вето/выхода задана пользовательским JSON до запуска." if custom else "Конфигурации входа/вето/выхода остаются зафиксированными ранним отбором."),
        "__SNAPSHOT_RULE__": ("Показана первая закрытая диагностическая сделка пользовательского рецепта при одном контракте." if custom else "Выбор примеров: первая закрытая диагностическая сделка BB breakout, BB reentry, Range breakout и Fade большой свечи.")}
    substitutions.update({f"__{key.upper()}__": value for key, value in pngs.items()})
    for marker, value in substitutions.items():
        template = template.replace(marker, value)
    path = root / "REPORT_RU.html"
    path.write_text(template, encoding="utf-8")
    receipt = dict(status="COMPLETE", report=str(path), images={k: str(v) for k, v in plots.items()},
                   renderer_python=sys.executable, renderer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   input_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in required},
                   economic_admission="NOT_ESTABLISHED", report_bytes=path.stat().st_size,
                   validation="Self-contained data, saved metrics only; figures require visual inspection")
    (root / "report_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(render(args.directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
