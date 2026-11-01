"""Reproducible, outcome-blind selection of raw-leader market scenes.

The robust adaptive account can choose WAIT. These pictures describe actual
executions of the separately reported, ungated raw-leader diagnostic only.
Before pictures contain no future quotes, fills, outcomes, or future-derived
axis limits. A canonical cumulative T counter supplies global route volume;
matched F removals are never counted as a second trade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .engine import CATALOG

ROOT = Path(__file__).resolve().parents[2]
NS = 1_000_000_000
COLORS = {"ink": "#193542", "past": "#365f73", "future": "#dc7b31",
          "zone": "#55a89b", "buy": "#25877f", "sell": "#d66a52",
          "unknown": "#9b9fa8"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return actual


def utc(ns):
    return pd.Timestamp(int(ns), unit="ns", tz="UTC").isoformat()


def volume_frame(seconds, start_ns, end_ns):
    """Difference adjacent supported one-second cumulative counter endpoints.

    Inclusion and plotting use publication time, including a delayed endpoint
    only when it has actually become available. Epoch crossings, invalid
    endpoints, missing seconds and counter reversals produce unknown intervals.
    """
    cols = ["c_trade_buy", "c_trade_sell", "c_trade_unknown"]
    delta = seconds[cols].diff()
    known = (seconds.valid.astype(bool) & seconds.valid.shift(1, fill_value=False).astype(bool)
             & seconds.epoch.eq(seconds.epoch.shift(1))
             & seconds.t_ns.diff().eq(NS)
             & delta.ge(0).all(axis=1))
    selected = seconds.available_ns.between(start_ns, end_ns)
    result = seconds.loc[selected, ["t_ns", "available_ns"]].copy()
    result[cols] = delta.loc[selected].where(known.loc[selected], np.nan)
    result["known"] = known.loc[selected].to_numpy()
    return result


def grouped_volume(volume, origin, cutoff):
    f = volume.loc[volume.available_ns.le(cutoff)].copy()
    # Fixed bins relative to the known decision time, never to an outcome.
    f["bin"] = np.floor((f.available_ns.to_numpy(np.int64) - origin) / (10 * NS)).astype(int)
    cols = ["c_trade_buy", "c_trade_sell", "c_trade_unknown"]
    sums = f.groupby("bin")[cols].sum(min_count=1)
    unknown = f.groupby("bin").known.all()
    sums.loc[~unknown, :] = np.nan
    return sums


def draw_scene(path, *, quote, volume, observation, trade, outcome, pair,
               limits, after, mobile=False):
    origin = int(observation.t_ns)
    end = int(trade.exit_ns) + 30 * NS if after else origin
    f = quote.loc[quote.available_ns.between(origin - 600 * NS, end)].copy()
    x = (f.available_ns.to_numpy(np.int64) - origin) / NS
    mid = (f.bid_nano.to_numpy(float) + f.ask_nano.to_numpy(float)) / 2e9
    valid = f.valid.to_numpy(bool) & (f.bid_nano.to_numpy() > 0) & (f.ask_nano.to_numpy() > f.bid_nano.to_numpy())
    # Do not connect a line through invalid publications or source epoch changes.
    valid &= f.epoch.eq(f.epoch.shift(1, fill_value=f.epoch.iloc[0])).to_numpy()
    mid[~valid] = np.nan
    width, height = ((6.0, 8.2) if mobile else (12.0, 7.5))
    fig, (ax, av) = plt.subplots(2, 1, figsize=(width, height), dpi=120,
        gridspec_kw={"height_ratios": [3.5, 1.25]}, sharex=True)
    fig.patch.set_facecolor("#f8fafb")
    for a in (ax, av):
        a.set_facecolor("white")
        a.grid(axis="y", alpha=.16)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=9)
    before = x <= 0
    ax.step(x[before], mid[before], where="post", color=COLORS["past"], linewidth=1.25, label="Mid до решения")
    if after:
        exit_seconds = (int(trade.exit_ns) - origin) / NS
        future = (x >= 0) & (x <= exit_seconds)
        # Include the last already-known quote to show the causal boundary.
        if before.any():
            future[np.flatnonzero(before)[-1]] = True
        ax.step(x[future], mid[future], where="post", color=COLORS["future"], linewidth=1.45, label="Mid после решения")
        post_exit = x >= exit_seconds
        ax.step(x[post_exit], mid[post_exit], where="post", color="#8b9098", linestyle=":", linewidth=1.25, label="После закрытия")
    low, high = float(observation.zone_low_nano) / 1e9, float(observation.zone_high_nano) / 1e9
    ax.axhspan(low, high, color=COLORS["zone"], alpha=.17, label="Прежняя область")
    decision_price = float(observation.decision_mid_per_oz)
    ax.scatter([0], [decision_price], s=38, color=COLORS["ink"], zorder=6)
    for a in (ax, av):
        a.axvline(0, color=COLORS["ink"], linestyle=":", linewidth=1.3)
    if after:
        target = int(outcome.target_nano) / 1e9
        # The stop is a threshold on costed liquidation BBO, not on midpoint.
        fee = float(pair["fee_roundtrip_per_oz"])
        stop = float(outcome.entry_price) + int(trade.side) * (-float(observation.risk_per_oz) + fee)
        tx = (int(trade.exit_ns) - origin) / NS
        ex = (int(trade.entry_ns) - origin) / NS
        ax.hlines(target, 0, tx, color="#268473", linestyle="--", linewidth=1.1, label="Цель: порог bid/ask")
        ax.hlines(stop, ex, tx, color="#bb4c55", linestyle="--", linewidth=1.1, label="Стоп: порог bid/ask с fee")
        ax.scatter([ex], [float(outcome.entry_price)], s=72, marker="^" if trade.side == 1 else "v", color="#216ea3", zorder=7, label="Фактический вход BBO")
        ax.scatter([tx], [float(outcome.exit_price)], s=65, marker="X", color="#9d3350", zorder=7, label="Фактический выход BBO")
        ax.axvline(tx, color="#9d3350", alpha=.35, linewidth=.9)
        clipped = int(np.sum(np.isfinite(mid[~before]) & ((mid[~before] < limits[0]) | (mid[~before] > limits[1]))))
        outcome_text = f"{trade.reason} · {float(trade.pnl_usd):+.0f} USD · {float(trade.holding_seconds):.1f} с"
        if clipped:
            outcome_text += f" · вне прежней шкалы: {clipped} публикаций"
        ax.text(.02, .97, outcome_text, transform=ax.transAxes, va="top", fontsize=10,
                bbox={"facecolor": "white", "alpha": .92, "edgecolor": "none"})
    else:
        ax.text(.02, .97, "Будущее и исполнения скрыты", transform=ax.transAxes, va="top", fontsize=10,
                bbox={"facecolor": "white", "alpha": .92, "edgecolor": "none"})
    ax.set_ylim(*limits)
    ax.set_xlim(-600, max(35., (end - origin) / NS + 3.) if after else 5.)
    ax.set_ylabel("GCZ6 · USD / oz", fontsize=10)
    ax.legend(loc="lower left", fontsize=7.5 if mobile else 8, ncols=1 if mobile else 3, framealpha=.91)
    totals = grouped_volume(volume, origin, end)
    bx = (totals.index.to_numpy() + .5) * 10
    bottom = np.zeros(len(totals))
    for column, label, color in [("c_trade_buy", "Покупки", COLORS["buy"]),
                                  ("c_trade_sell", "Продажи", COLORS["sell"]),
                                  ("c_trade_unknown", "Сторона неизвестна", COLORS["unknown"])]:
        values = totals[column].to_numpy(float)
        av.bar(bx, values, width=9, bottom=bottom, color=color, alpha=.85, label=label)
        bottom += np.nan_to_num(values)
    av.set_ylabel("Объём T\nза 10 с", fontsize=9)
    av.set_xlabel("Секунды относительно решения · время публикации", fontsize=9)
    av.legend(loc="upper left", fontsize=7.5, ncols=1 if mobile else 3, framealpha=.9)
    # The before volume axis also depends solely on already published history.
    vp = grouped_volume(volume, origin, origin).sum(axis=1, min_count=1)
    vy = max(1., float(vp.max()) * 1.2) if len(vp) else 1.
    av.set_ylim(0, vy)
    clipped_volume = int(np.sum(bottom > vy))
    if after and clipped_volume:
        av.text(.99, .96, f"За шкалой: {clipped_volume} столб.", transform=av.transAxes,
                ha="right", va="top", fontsize=8, color="#9d3350",
                bbox={"facecolor": "white", "alpha": .92, "edgecolor": "none"})
    title = CATALOG[trade.selected_strategy]["title"]
    side = "LONG" if int(trade.side) == 1 else "SHORT"
    stamp = pd.Timestamp(origin, unit="ns", tz="UTC").strftime("%d.%m.%Y %H:%M:%S UTC")
    fig.suptitle(f"{'ПОСЛЕ' if after else 'ДО'} · {title}\n{stamp} · {side}", x=.09, ha="left", fontsize=13 if mobile else 15, color=COLORS["ink"])
    footer = ("Сырой лидер пары: диагностический счёт. Адаптивная политика — отдельно.\n"
              "Объём всего маршрута, не у зоны; T один раз, F не прибавляется.\n"
              "Шкалы цены и объёма заданы по прошлому; BBO-пороги отличаются от mid.")
    fig.text(.09, .02, footer, fontsize=7.2 if mobile else 8, color="#5b6770", va="bottom")
    fig.subplots_adjust(left=.13 if mobile else .09, right=.97, top=.85, bottom=.20 if mobile else .17, hspace=.10)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = ROOT / args.out
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    files = ["trades.parquet", "observations.parquet", "base_outcomes.parquet", "selections.parquet", "config.json"]
    receipts = {name: checked(out / name, manifest["artifacts"][name]) for name in files}
    trades = pd.read_parquet(out / "trades.parquet")
    trades = trades[trades.policy.eq("raw_leader") & trades.condition.eq("base")]
    chosen = trades.sort_values(["signal_ns", "candidate_id"]).groupby("selected_strategy", sort=False).head(1)
    observations = pd.read_parquet(out / "observations.parquet").set_index("candidate_id")
    outcomes = pd.read_parquet(out / "base_outcomes.parquet").set_index(["candidate_id", "side"])
    selections = pd.read_parquet(out / "selections.parquet").set_index("next_date")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    sources = {}
    for path, sha in manifest["source_hashes"].items():
        if Path(path).name != "manifest.json" or "full_mbo_v1" not in path:
            continue
        checked(path, sha)
        market = json.loads(Path(path).read_text(encoding="utf-8"))
        sources[market["input"]["selection"]["date"]] = (Path(path).parent, market, sha)
    scene_dir = out / "scenes"
    scene_dir.mkdir(exist_ok=True)
    scenes = []
    for trade in chosen.itertuples(index=False):
        obs = observations.loc[trade.candidate_id]
        consequence = outcomes.loc[(trade.candidate_id, trade.side)]
        pair = selections.loc[trade.date]
        if pair.raw_leader != trade.selected_strategy or int(pair.selection_ns) >= int(trade.signal_ns):
            raise AssertionError("Scene strategy was not selected before this execution")
        for name in ("entry_ns", "exit_ns", "reason", "net_per_oz"):
            if getattr(trade, name) != consequence[name]:
                raise AssertionError(f"Execution identity mismatch: {name}")
        source, market, msha = sources[trade.date]
        qsha = checked(source / "quotes.parquet", market["payloads"]["quotes.parquet"])
        if qsha != manifest["source_hashes"][str(source / "quotes.parquet")]:
            raise AssertionError("Native source differs from experiment")
        ssha = checked(source / "seconds.parquet", market["payloads"]["seconds.parquet"])
        origin = int(obs.t_ns)
        end = int(trade.exit_ns) + 30 * NS
        quotes = pd.read_parquet(source / "quotes.parquet")
        quotes = quotes[quotes.available_ns.between(origin - 600 * NS, end)].copy()
        seconds = pd.read_parquet(source / "seconds.parquet")
        volume = volume_frame(seconds, origin - 600 * NS, end)
        past = quotes[quotes.available_ns.le(origin) & quotes.valid]
        if past.empty:
            raise AssertionError("No known price history")
        pvalues = (past.bid_nano.to_numpy(float) + past.ask_nano.to_numpy(float)) / 2e9
        p0, r = float(obs.decision_mid_per_oz), float(obs.risk_per_oz)
        # Fixed observed-history range with a causal risk envelope; never reads future extrema.
        vmin = min(float(pvalues.min()), float(obs.zone_low_nano) / 1e9, p0 - 2.5 * r)
        vmax = max(float(pvalues.max()), float(obs.zone_high_nano) / 1e9, p0 + 2.5 * r)
        pad = max(.2, (vmax - vmin) * .08)
        limits = (vmin - pad, vmax + pad)
        name = trade.selected_strategy + "_" + str(trade.candidate_id)
        images = {}
        for mobile in (False, True):
            for after in (False, True):
                key = ("after" if after else "before") + ("_mobile" if mobile else "") + "_image"
                image = f"scenes/{name}_{'after' if after else 'before'}{'_mobile' if mobile else ''}.png"
                draw_scene(out / image, quote=quotes, volume=volume, observation=obs,
                           trade=trade, outcome=consequence,
                           pair={"fee_roundtrip_per_oz": config["execution"]["fee_roundtrip_per_oz"]},
                           limits=limits, after=after, mobile=mobile)
                images[key] = image
        features = {
            "Расстояние до зоны / R": round(float(obs.distance_r), 3),
            "Продвижение к зоне за 30 с / R": round(float(obs.past_net30_r), 3),
            "Смещение средних к зоне за 10 мин / R": round(float(obs.past_migration_r), 3),
            "Дисбаланс сделок к зоне, 30 с": round(float(obs.x_mbo_mbo_now_trade_imbalance), 3),
            "Доля объёма с неизвестной стороной, 30 с": round(float(obs.x_mbo_mbo_now_trade_unknown_fraction), 3),
            "Дисбаланс лучшей глубины к зоне": round(float(obs.x_mbo_mbo_now_depth_1_toward_imbalance), 3),
            "Доля прежней зоны: первые / последние 5 мин": f"{float(obs.past_zone_share_old):.1%} / {float(obs.past_zone_share_new):.1%}",
            "Созревших посещений": int(obs.completed_visits),
        }
        scene = dict(id=name, strategy=trade.selected_strategy,
            strategy_label=CATALOG[trade.selected_strategy]["title"], date=trade.date,
            t_ns=origin, decision_time_utc=utc(origin), pair_start=pair.pair_start,
            pair_end=pair.pair_end, side="LONG" if trade.side == 1 else "SHORT",
            entry_ns=int(trade.entry_ns), exit_ns=int(trade.exit_ns),
            entry_time_utc=utc(trade.entry_ns), exit_time_utc=utc(trade.exit_ns),
            entry_price=float(consequence.entry_price), exit_price=float(consequence.exit_price),
            net_usd=float(trade.pnl_usd), reason=trade.reason,
            adaptive_strategy=pair.selected_strategy, features=features, **images,
            receipt=dict(candidate_id=int(trade.candidate_id), source=str(source),
                market_manifest_sha256=msha, quotes_sha256=qsha, seconds_sha256=ssha,
                source_publication_clock="available_ns", selection_ns=int(pair.selection_ns),
                max_before_quote_available_ns=int(past.available_ns.max()),
                max_before_volume_available_ns=int(volume.loc[volume.available_ns.le(origin), "available_ns"].max()),
                before_y_limits=list(limits), before_window_seconds=600,
                volume_scope="Global route canonical T counters; 10s publication bins; no F double-count; not zone-local",
                unknown_second_intervals=int((~volume.known).sum()),
                selection_uses_outcome=False,
                png_sha256={key: digest(out / filename) for key, filename in images.items()}))
        scenes.append(scene)
        print(json.dumps({"scene": name, "date": trade.date, "net_usd": trade.pnl_usd}), flush=True)
    result = dict(version="two_day_strategy_scenes_v1", policy="raw_leader", condition="base",
        selection_rule="Earliest actually executed base raw_leader trade per distinct selected_strategy, sorted by signal_ns then candidate_id; no outcome-based selection",
        scope="Raw-leader diagnostic account. Robust adaptive selection and its WAIT decisions are separate. Futures revealed only in after images.",
        source_manifest_sha256=digest(out / "manifest.json"), source_artifact_sha256=receipts,
        script_sha256=digest(Path(__file__)), scenes=scenes)
    (out / "scenes.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
