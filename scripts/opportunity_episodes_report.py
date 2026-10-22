"""Trader/ML audit figures for the frozen opportunity experiment, no selection."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectrum.cognition import episodes as ep
from spectrum.cognition.episodes_lab import DATA, REPORT, save, sha
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition.signatures import DEFAULT_DATA
from spectrum.playground.data import MonthDay, QUOTE_CACHE

COLORS = dict(baseline="#3465a4", episode="#b78b18", response="#107b6b", response_unweighted="#9b59b6", shuffled="#999999")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .18, "figure.facecolor": "white", "savefig.facecolor": "white"})


def fig_save(fig, filename):
    fig.canvas.draw()
    fig.savefig(REPORT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def overview(summary):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    fig.suptitle("Executable opportunity: Jan-Feb 2026 development evaluation\nFixed 24/12 ticks, 300 s; queue entry, taker exits; $0.30 / 1 oz", fontsize=15)
    families = list(COLORS); b = summary["stages"]["B"]["P"]
    ax[0, 0].bar(families, [b[f]["all"]["log_loss"] for f in families], color=[COLORS[f] for f in families])
    ax[0, 0].set(ylabel="Multiclass log-loss (lower is better)", title="Same known attempts, all four outcomes")
    ax[0, 0].tick_params(axis="x", rotation=20)
    for f in ("baseline", "episode", "response", "response_unweighted"):
        cal = b[f]["all"]["calibration"]
        ax[0, 1].plot([r["predicted"] for r in cal], [r["observed"] for r in cal], "o-", color=COLORS[f], label=f)
        if f == "response":
            for r in cal:
                ax[0, 1].annotate(f"n={r['n']:,}", (r["predicted"], r["observed"]), xytext=(3, 5), textcoords="offset points", fontsize=8)
    ax[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0, 1].set(xlim=(0, .8), ylim=(0, .8), xlabel="Predicted P(TAKE)", ylabel="Observed TAKE fraction", title="Reliability; counts label response bins")
    ax[0, 1].legend(fontsize=8)
    for i, f in enumerate(families[1:4]):
        r = summary["comparisons"]["B/P/" + f]; lo, hi = r["daily_ci"]
        ax[1, 0].plot([lo, hi], [i, i], color=COLORS[f], lw=3)
        ax[1, 0].plot(r["daily_gain"], i, "o", color=COLORS[f])
    ax[1, 0].axvline(0, color="black", ls="--")
    ax[1, 0].set(yticks=range(3), yticklabels=families[1:4], xlabel="Daily total log-score gain vs baseline (41 calendar days)", title="Paired 95% moving-block interval (5 days)")
    ax[1, 0].set_ylim(-.25, 2.55)
    sh = summary["comparisons"]["B/P/shuffled"]
    ax[1, 0].text(.02, .97, f"Shuffled control outside zoom: {sh['daily_gain']:.2f}\n95% CI [{sh['daily_ci'][0]:.2f}, {sh['daily_ci'][1]:.2f}]", transform=ax[1, 0].transAxes, va="top", fontsize=8)
    months = ["2026-01", "2026-02"]
    for f in families[:4]:
        ax[1, 1].plot(months, [b[f]["months"][m]["log_loss"] for m in months], "o-", color=COLORS[f], label=f)
    ax[1, 1].set(ylabel="Log-loss", title="Transfer by month; no refit within evaluation")
    ax[1, 1].set_xticks(range(2), [f"{m}\nn={b['response']['months'][m]['n']:,} known side-attempts" for m in months])
    ax[1, 1].legend(fontsize=8)
    fig_save(fig, "01_prediction_quality.png")


def all_class_calibration(summary):
    frames = {f: pd.read_parquet(DATA / f"predictions_B_P_{f}.parquet") for f in ("baseline", "response")}
    best = frames["response"].loc[frames["response"].groupby("t_ns").expected_net.idxmax()]
    threshold = summary["coverage"]["P/response"]["0.1"]
    tail = best.loc[best.expected_net > threshold]
    groups = {**frames, "response A2 top10% tail": tail}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    fig.suptitle("All-class reliability: known attempts, P mode, B development data\nTail uses frozen A2 score threshold, best side per root, before account occupancy filtering", fontsize=13)
    records = {}
    for name, frame in groups.items():
        known = frame.loc[frame.category >= 0]
        records[name] = dict(n_known=len(known), n_unknown=len(frame) - len(known), classes={})
        for k, cls in enumerate(ep.CLASSES):
            ax = axes.flat[k]; prob = known["p_" + cls].to_numpy(); bins = np.minimum((prob * 10).astype(int), 9)
            points = []
            for i in range(10):
                mask = bins == i
                if mask.any():
                    points.append(dict(n=int(mask.sum()), predicted=float(prob[mask].mean()), observed=float((known.category.to_numpy()[mask] == k).mean())))
            records[name]["classes"][cls] = points
            is_tail = name.endswith("tail")
            color = "#bd3d43" if is_tail else COLORS[name]
            ax.scatter([r["predicted"] for r in points], [r["observed"] for r in points], marker="s" if is_tail else "o", facecolors="none" if is_tail else color, edgecolors=color, s=48, label=f"{name}, n={len(known):,}")
            if name != "baseline":
                for r in points:
                    ax.annotate(str(r["n"]), (r["predicted"], r["observed"]), xytext=(3, -11 if is_tail else 5), textcoords="offset points", color=color, fontsize=7)
            else:
                for r in points:
                    if r["n"] < 30:
                        ax.annotate(f"b n={r['n']}", (r["predicted"], r["observed"]), xytext=(3, 5), textcoords="offset points", color=color, fontsize=7)
    for ax, cls in zip(axes.flat, ep.CLASSES):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel=f"Predicted P({cls})", ylabel="Observed fraction", title=cls + " | labels = bin counts; no bin confidence intervals")
        ax.legend(fontsize=7, loc="upper left")
    fig_save(fig, "03_all_class_calibration.png")
    save(REPORT / "calibration_audit.json", dict(groups=records, tail_threshold=threshold,
        semantics="Fixed 0.1 probability bins; unknown outcomes excluded only from reliability, not from policy; descriptive, not calibration admission"))


def score_decomposition(summary):
    """Post-hoc accounting identity, not another selection or admission test."""
    names = ["baseline", "response"]
    frames = [pd.read_parquet(DATA / f"predictions_B_P_{f}.parquet") for f in names]
    a, b = frames
    pd.testing.assert_frame_equal(a[["date", "t_ns", "side", "category"]], b[["date", "t_ns", "side", "category"]])
    known = a.category >= 0; a, b = a.loc[known], b.loc[known]
    y = a.category.to_numpy(int); filled = y != ep.CLASSES.index("EMPTY")
    pa, pb = [f[["p_" + c for c in ep.CLASSES]].to_numpy() for f in (a, b)]
    joint = np.log(pb[np.arange(len(y)), y]) - np.log(pa[np.arange(len(y)), y])
    fill = np.where(filled, np.log1p(-pb[:, 3]) - np.log1p(-pa[:, 3]), np.log(pb[:, 3]) - np.log(pa[:, 3]))
    cond = np.zeros(len(y))
    cond[filled] = np.log(pb[filled, y[filled]] / (1 - pb[filled, 3])) - np.log(pa[filled, y[filled]] / (1 - pa[filled, 3]))
    np.testing.assert_allclose(joint, fill + cond, rtol=1e-10, atol=1e-14)
    days = summary["stages"]["B"]["P"]["response"]["calendar_days"]
    parts = {}
    for name, values in (("joint", joint), ("fill", fill), ("conditional_path", cond)):
        daily = pd.DataFrame(dict(date=a.date, gain=values)).groupby("date").gain.sum().reindex(days, fill_value=0.)
        parts[name] = dict(mean_gain_per_attempt=float(values.mean()), total=float(values.sum()),
                           daily_ci=ep.block_interval(daily.to_numpy()), days=len(days))
    save(REPORT / "posthoc_score_decomposition.json", dict(status="POST_HOC_DIAGNOSTIC", parts=parts,
        n_known=len(y), n_filled=int(filled.sum()), fill_share_of_joint=float(fill.sum() / joint.sum()),
        identity_max_abs_error=float(np.max(np.abs(joint - fill - cond))),
        prediction_sha256={f: sha(DATA / f"predictions_B_P_{f}.parquet") for f in names},
        semantics="Exact log-score decomposition; conditional evaluation uses observed fills diagnostically, never to select policy rows; no admission or model changed"))


def coverage_and_economics(summary):
    man = json.loads((DATA / "manifest.json").read_text()); rows = man["days"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    fig.suptitle("Coverage, dependence and economic limits\nAll periods previously researched; these are development diagnostics", fontsize=15)
    dates = pd.to_datetime([r["date"] for r in rows])
    ax[0, 0].plot(dates, [r["roots"] for r in rows], label="eligible 30 s decision roots", color="#3465a4")
    ax[0, 0].plot(dates, [r["active_roots"] for r in rows], label="roots inside causal episodes", color="#107b6b")
    for boundary in ("2025-12-01", "2026-01-01"):
        ax[0, 0].axvline(pd.Timestamp(boundary), color="black", ls="--", lw=1)
    ax[0, 0].set(ylabel="Roots per day (before L/S duplication)", title="A1 Oct-Nov | A2 Dec | B Jan-Feb")
    ax[0, 0].legend(fontsize=8); ax[0, 0].tick_params(axis="x", rotation=20)
    fit = summary["stages"]["B"]["P"]["response"]["fit"]
    ax[0, 1].bar(["episode rows", "episode fit weight"], [fit["active_fraction"], fit["active_weight_fraction"]], color=["#107b6b", "#9b59b6"])
    ax[0, 1].set(ylim=(0, 1), ylabel="Fraction of fitting population", title="Explicit effect of episode de-duplication")
    b = summary["stages"]["B"]["P"]
    for family in ("baseline", "episode", "response", "response_unweighted"):
        p = b[family]["policy"]
        daily = pd.Series(p["daily"], dtype=float).sort_index()
        if len(daily):
            ax[1, 0].plot(pd.to_datetime(daily.index), daily.cumsum(), color=COLORS[family], label=f"{family}: {p['attempts']} attempts")
    ax[1, 0].axhline(0, color="black", ls="--", label="WAIT = 0")
    ax[1, 0].set(ylabel="Cumulative realized net ($ / 1 oz)", title="Threshold E[net] > 0; rejected candidates are diagnostic")
    ax[1, 0].tick_params(axis="x", rotation=20); ax[1, 0].legend(fontsize=8)
    for family in ("baseline", "episode", "response", "response_unweighted"):
        cs = b[family]["coverage"]
        xs, ys = [], []
        for key in ("0.1", "0.05", "0.01"):
            r = cs[key]
            if r["net_per_day"] is not None:
                xs.append(100 * r["actual_score_coverage"]); ys.append(r["net_per_day"])
                ax[1, 1].annotate(str(r["attempts"]), (xs[-1], ys[-1]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax[1, 1].plot(xs, ys, "o-", color=COLORS[family], label=family)
    ax[1, 1].axhline(0, color="black", ls="--")
    ax[1, 1].set(xlabel="Actual B score coverage (%)", ylabel="Mean daily net ($ / 1 oz)",
                 title="A2 top-10/5/1% thresholds frozen; labels = attempts")
    ax[1, 1].legend(fontsize=8)
    fig_save(fig, "02_coverage_and_economics.png")


def scenes():
    selection = json.loads((REPORT / "scene_selection.json").read_text()); facts = json.loads((DATA / "audit_execution_facts.json").read_text())
    lookup = {(r["date"], r["t_ns"], r["mode"], r["side"]): r for r in facts}
    predictions = {fam: pd.read_parquet(DATA / f"predictions_B_P_{fam}.parquet") for fam in ("baseline", "response")}
    files = []
    for number, scene in enumerate(selection["scenes"]):
        date, side, t = scene["date"], scene["side"], int(scene["t_ns"])
        fact = lookup[(date, scene["t_ns"], "P", side)]
        md = MonthDay(Path(DEFAULT_DATA) / date[:7] / date); q = QuoteTape.from_day(md)
        fill = int(fact["entry_ns"]) if fact["entry_ns"] is not None else None
        exit_ns = int(fact["exit_ns"]) if fact["exit_ns"] is not None else t + 330 * ep.NS
        end = max(90., min(390., (exit_ns - t) / ep.NS + 30)); lo, hi = t - 90 * ep.NS, t + int(end * ep.NS)
        keep = (q.quote_t >= lo) & (q.quote_t <= hi); idx = np.flatnonzero(keep)
        xx = (q.quote_t[idx] - t) / ep.NS
        fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1.2, 1.2]})
        fig.subplots_adjust(left=.085, right=.98, bottom=.07, top=.92, hspace=.25)
        cat = ep.CLASSES[fact["category"]] if fact["category"] >= 0 else "CENSORED"
        clock = pd.Timestamp(t, unit="ns", tz="UTC").strftime("%H:%M:%S UTC")
        fig.suptitle(f"{date} {clock} | {'LONG' if side == 1 else 'SHORT'} | {cat} | {scene['selection']}\nDiagnostic candidate; future shown only for audit; not a policy trade", fontsize=13)
        ax[0].plot(xx, np.where(q.valid[idx], q.bid[idx] / 1e9, np.nan), color="#147d92", lw=.8, drawstyle="steps-post", label="bid")
        ax[0].plot(xx, np.where(q.valid[idx], q.ask[idx] / 1e9, np.nan), color="#d79535", lw=.8, alpha=.85, drawstyle="steps-post", label="ask")
        at = int(np.searchsorted(q.quote_t, t, side="right") - 1)
        limit = (q.bid[at] if side == 1 else q.ask[at]) / 1e9
        ax[0].plot([.1, (fill - t) / ep.NS if fill else 30.1], [limit, limit], color="#666666", ls=":", lw=2, label="submitted limit")
        if fill is not None:
            px = int(fact["entry_px_nano"]) / 1e9; xf = (fill - t) / ep.NS; xe = (exit_ns - t) / ep.NS
            ax[0].plot(xf, px, "o", color="black", ms=7, label="V5 entry fill")
            ax[0].plot([xf, xe], [px + side * 2.4] * 2, "--", color="#168151", label="executable TP")
            ax[0].plot([xf, xe], [px - side * 1.2] * 2, "--", color="#bd3d43", label="executable SL")
            if fact["exit_px_nano"] is not None:
                ax[0].plot(xe, int(fact["exit_px_nano"]) / 1e9, "x", color="black", ms=9, label="taker exit")
            held = keep & (q.quote_t >= fill) & (q.quote_t <= exit_ns) & q.valid
            if cat == "CENSORED":
                horizon = fill + 305 * ep.NS
                stop_i = int(np.searchsorted(q.quote_t, exit_ns))
                if exit_ns >= horizon:
                    reason = "No exit quote within 300s + 5s grace"
                    marker = (horizon - t) / ep.NS
                    held &= q.quote_t < horizon
                elif not q.valid[stop_i]:
                    reason = "Quote support lost"; marker = xe
                elif q.epoch[stop_i] != fact["quote_epoch"]:
                    reason = "Quote epoch changed"; marker = xe
                else:
                    raise ValueError("Unexplained censor in scene")
                for a in ax[:2]:
                    a.axvline(marker, color="#bd3d43", ls=":", lw=1.5)
                ax[0].text(.98, .05, f"CENSORED at +{marker:.3f}s\n{reason}\nFirst later quote: +{xe:.3f}s", transform=ax[0].transAxes, ha="right", fontsize=8, color="#a62b32", bbox=dict(facecolor="white", edgecolor="#bd3d43", alpha=.9))
            if cat != "CENSORED" and 0 <= xe - xf < 1:
                # A true-time inset makes very short executions visible without
                # extending their TP/SL lifetimes on the main time axis.
                inset = ax[0].inset_axes([.64, .54, .33, .39])
                margin = max((exit_ns - fill) * .6, 1e6)
                left, right = fill - int(margin), exit_ns + int(margin)
                a = max(0, int(np.searchsorted(q.quote_t, left)) - 1)
                b = int(np.searchsorted(q.quote_t, right, side="right"))
                tx = (q.quote_t[a:b] - fill) / 1e6
                inset.plot(tx, q.bid[a:b] / 1e9, color="#147d92", lw=.8, drawstyle="steps-post")
                inset.plot(tx, q.ask[a:b] / 1e9, color="#d79535", lw=.8, drawstyle="steps-post")
                duration = (exit_ns - fill) / 1e6
                inset.plot([0, duration], [px + side * 2.4] * 2, "--", color="#168151")
                inset.plot([0, duration], [px - side * 1.2] * 2, "--", color="#bd3d43")
                inset.plot(0, px, "ko", ms=4)
                inset.plot(duration, int(fact["exit_px_nano"]) / 1e9, "kx", ms=5)
                inset.set_xlim(-margin / 1e6, duration + margin / 1e6)
                inset.set_title(f"Execution zoom: hold {duration:.3f} ms", fontsize=8)
                inset.set_xlabel("Milliseconds from fill", fontsize=7)
                inset.tick_params(labelsize=7)
            pnl = side * ((q.bid[held] if side == 1 else q.ask[held]) / 1e9 - px) - .30
            ax[1].plot((q.quote_t[held] - t) / ep.NS, pnl, color="#107b6b", label="liquidation net after fee")
            ax[1].axhline(2.1, color="#168151", ls="--", lw=.8); ax[1].axhline(-1.5, color="#bd3d43", ls="--", lw=.8)
            ax[1].set_title(f"Observed net = {fact['net']:+.3f} USD / 1 oz" if fact["net"] is not None else "Unknown outcome: no fabricated PnL", fontsize=10)
        else:
            ax[1].text(.5, .5, "NO FILL — no position, no TP/SL outcome", ha="center", transform=ax[1].transAxes)
        raw = pd.read_parquet(Path("data/derived/scalper_v1") / f"{date}.parquet", columns=["t_ns", "flow_pressure30_low", "flow_pressure30_high", "flow_pressure30_known", "flow_book30_pressure", "flow_book30_known"])
        raw = raw.loc[(raw.t_ns >= lo) & (raw.t_ns <= hi)]
        rx = (raw.t_ns.to_numpy(np.int64) - t) / ep.NS
        low = np.where(raw.flow_pressure30_known, raw.flow_pressure30_low, np.nan)
        high = np.where(raw.flow_pressure30_known, raw.flow_pressure30_high, np.nan)
        ax[2].fill_between(rx, low, high, alpha=.25, color="#3465a4", label="trade pressure bounds incl. unknown")
        ax[2].plot(rx, (low + high) / 2, color="#3465a4", lw=1)
        ax[2].plot(rx, np.where(raw.flow_book30_known, raw.flow_book30_pressure, np.nan), color="#9b59b6", label="book pressure (30 s)")
        ax[2].axhline(0, color="black", lw=.6); ax[2].set(ylim=(-1.05, 1.05), ylabel="pressure")
        for family in ("baseline", "response"):
            p = predictions[family]; p = p.loc[(p.date == date) & (p.side == side) & (p.t_ns >= lo) & (p.t_ns <= hi)]
            ax[3].scatter((p.t_ns.to_numpy(np.int64) - t) / ep.NS, p.p_TAKE, s=16, marker="o" if family == "baseline" else "x", color=COLORS[family], label=family + " P(TAKE)")
            current = p.loc[p.t_ns == t]
            if family == "response" and len(current):
                r = current.iloc[0]
                age = f"{r.episode_age * 120:.0f}s" if np.isfinite(r.episode_age) else "inactive"
                ax[3].set_title(f"At decision: P(TAKE)={r.p_TAKE:.3f}, P(EMPTY)={r.p_EMPTY:.3f}, E[net]={r.expected_net:+.3f}; episode age={age}", fontsize=10)
        for a in ax:
            a.axvline(0, color="black", lw=1, label="decision" if a is ax[0] else None)
            a.axvspan(0, end, color="#eeeeee", alpha=.35, zorder=-5)
            a.set_xlim(-90, end)
        ax[0].set_ylabel("GC price ($ / oz)"); ax[1].set_ylabel("Net ($ / 1 oz)")
        ax[3].set(xlabel="Seconds relative to decision (0); every forecast uses only its own past", ylabel="probability", ylim=(0, 1))
        for a in (ax[0], ax[2], ax[3]):
            a.legend(loc="upper left", fontsize=7, ncol=3 if a is ax[0] else 2)
        filename = f"scene_{number + 1:02d}_{cat.lower()}.png"; fig_save(fig, filename); files.append(filename); QUOTE_CACHE.clear()
    return files


def report(summary, scene_files):
    lines = ["# Исполнимая возможность и причинные эпизоды: результат", "", "Дата: 2026-10-22. Протокол зафиксирован до обучения. Все периоды ранее исследованы; **development evidence, не blind OOS**.", "",
             "Основной режим P: очередь V5 на входе, **тейкерский** выход и тейк, цель 24 тика / стоп 12 тиков / 300 секунд. 1 oz и $0.30 за круг — исследовательские единицы. Стоимость исходов оценивается только на fit-периоде.", "",
             "## Проверяемый результат на B (январь–февраль)", "", "| Модель P | Log-loss ↓ | AUC TAKE | Средний выигрыш log-score/попытку против базы | 95% интервал дневного выигрыша |", "|---|---:|---:|---:|---|"]
    for family in COLORS:
        r = summary["stages"]["B"]["P"][family]["all"]; cmp = summary["comparisons"].get("B/P/" + family)
        lines.append(f"| {family} | {r['log_loss']:.6f} | {r['auc_take']:.4f} | {cmp['mean_gain_per_attempt'] if cmp else 0:+.6f} | {str([round(x, 3) for x in cmp['daily_ci']]) if cmp else 'опора'} |")
    comp = summary["comparisons"]["B/P/response"]
    verdict = "Положительная разность с базой подтверждается данным интервалом." if comp["daily_ci"][0] > 0 else "Положительный добавочный навык response сверх базы не подтверждён данным интервалом."
    lines += ["", verdict, "", "В B январь содержит 12 885 известных side-attempts, февраль только 270 (около 2.1% B). Основная масса свидетельства — январская; февраль не подтверждает устойчивый перенос с сопоставимой мощностью. AUC TAKE response в феврале около 0.535. Общая калибровка и хвост показаны отдельно для всех четырёх классов; это описательные точки без доверительных интервалов, не допуск по калибровке.", "",
              "**Post-hoc разложение объясняет природу улучшения:** joint log-score точно равен Bernoulli log-score исполнения плюс условному log-score TAKE/STOP/TIME после fill. Из +0.002450 nats/attempt на исполнение приходится +0.002087 (85.2%), на условный путь +0.000363. Дневной 95% интервал условной части [-0.129; +0.418] включает ноль. Следовательно, улучшение главным образом в EMPTY/fill; навык распознавать выгодный исход после входа не подтверждён. Это диагностическое тождество после результата, не новый допуск и не отбор сделок по будущему. См. posthoc_score_decomposition.json.", "",
              "Семейства различаются информацией, но получают одинаковые origins. response_unweighted показывает эффект повторного вклада эпизода. shuffled — переставленные обучающие метки. Секундные агрегаты не доказывают поглощение на уровне order ID.", "",
              "## Последовательный счёт, E[net] > 0", "", "| Модель / режим | A2 net | Допущена по A2 | B попытки / fills | B net | B дневной 95% интервал |", "|---|---:|---|---:|---:|---|"]
    for mode in ("P", "T"):
        for family in list(COLORS)[:4]:
            a = summary["stages"]["A2"][mode][family]; b = summary["stages"]["B"][mode][family]; p = b["policy"]
            lines.append(f"| {family} / {mode} | {a['policy']['net']} | {'да' if b['operational_admitted'] else 'нет, WAIT'} | {p['attempts']} / {p['fills']} | {p['net']} | {p['daily_ci']} |")
    lines += ["", "Если вариант не прошёл A2, его кривая B — диагностический прогон отклонённого кандидата. Она не меняет заранее принятого WAIT. Top-10/5/1% в summary и на графике — фиксированные диагностические пороги из A2, не выбранные победители.", "",
              "Занятость счёта включает ожидание исполнения; это не чистая рыночная экспозиция. Парный random-side контроль сравнивает те же корни и не является второй непрерывной кривой капитала. При цензуре net и интервалы не достраиваются нулями.", "",
              "## Данные, ограничения и проверка", "", f"Опубликовано {summary['days']} дней, {summary['source_rows']:,} строк с отдельными режимами и сторонами. Отбор по узкому спреду — причинный, но меняет календарное покрытие; график показывает фактические counts.", "",
              "Дневные знаменатели и интервалы B используют все 41 непустой исходный день, включая пять дней без eligible roots. Это исправление отчётной ошибки выполнено после результата, без переобучения, изменения порогов и admission. Предыдущие summary и completion сохранены; см. AMENDMENTS_RU.md.", "",
              "Сохранённый scalper.py принадлежал старому коммиту a68c7aa. Текущий файл отличается выделением entry_points. После обратной подстановки этой функции AST всего модуля совпадает; исходный хеш сохранён, не подменён текущим. См. source_code_compatibility.json.", "",
              "Независимый quote-scan и свежий replay MBO проверяют категории, net, время выхода и реальные точки входа: execution_audit.json. Сцены календарные и первые по категориям, поэтому иллюстрируют механику, а не типичную прибыльность. Отсутствующие категории перечислены в scene_selection.json.", "",
              "Положительный контроль проверяет только способность HGB выучить известную вероятностную зависимость, не весь торговый реплей. " + json.dumps(summary["synthetic_control"], ensure_ascii=False), "",
              "## Визуальные материалы", "", "- 01_prediction_quality.png — качество и калибровка.", "- 02_coverage_and_economics.png — покрытие, веса и экономика.", "- 03_all_class_calibration.png — четыре класса и диагностический хвост с counts."]
    lines += ["- " + f for f in scene_files]
    (REPORT / "RESULTS_RU.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    images = ["01_prediction_quality.png", "02_coverage_and_economics.png", "03_all_class_calibration.png"] + scene_files
    body = "".join(f'<figure><img src="{html.escape(f)}" alt="{html.escape(f)}"><figcaption>{html.escape(f)}</figcaption></figure>' for f in images)
    page = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Opportunity episodes — visual audit</title><style>body{max-width:1450px;margin:32px auto;padding:0 24px;font:17px system-ui;color:#17232b;background:#f3f5f7}h1{font-size:30px}figure{margin:24px 0;background:white;padding:12px;border-radius:12px}img{width:100%;height:auto}figcaption{padding:8px;color:#52616d}a{color:#146b86}</style><h1>Исполнимая возможность и причинные эпизоды</h1><p>Development evidence. P: очередь на входе, тейкер на выходе. Графики будущего — только для аудита. Календарные и классовые сцены не являются выборкой прибыльных сделок.</p><p><a href="RESULTS_RU.md">Численный отчёт</a> · <a href="SOL61_VISUAL_AUDIT_RU.md">Независимый визуальный аудит Sol 6.1</a></p>' + body + '</html>'
    (REPORT / "index.html").write_text(page, encoding="utf-8")
    save(REPORT / "visual_receipt.json", dict(images={f: sha(REPORT / f) for f in images}, summary_sha256=sha(REPORT / "summary.json"),
                                             scene_selection_sha256=sha(REPORT / "scene_selection.json")))


if __name__ == "__main__":
    s = json.loads((REPORT / "summary.json").read_text()); overview(s); coverage_and_economics(s); all_class_calibration(s); score_decomposition(s)
    report(s, scenes())
