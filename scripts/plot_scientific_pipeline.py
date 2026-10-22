"""Render source-bound scientific diagnostics without altering trained runs.

python -m scripts.plot_scientific_pipeline
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


RUNS = ("tcn_execution_bce_seed17", "mlp_execution_bce_seed17", "tcn_crossing_bce_seed17")
NAMES = ("TCN / execution", "MLP / execution", "TCN / crossing")
COLORS = ("#2563eb", "#d97706", "#7c3aed")
STATE_NAMES = ("UNKNOWN", "NONE", "LONG only", "SHORT only", "BOTH")
STATE_COLORS = ("#d1d5db", "#475569", "#059669", "#dc2626", "#8b5cf6")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--runs-root", default="data/derived/scientific_pipeline_v1/runs")
    parser.add_argument("--out", default="reports/scientific_pipeline_v1")
    args = parser.parse_args()
    root, out = Path(args.runs_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    input_hashes, bundles, saved = {}, [], []
    for run in RUNS:
        path = root / run
        receipt = read_json(path / "receipt.json")
        if receipt["status"] != "COMPLETE":
            raise ValueError(f"Incomplete run: {run}")
        # Verify every saved output, including the selected checkpoint and
        # preprocessing, rather than trusting a summary status in isolation.
        for relative, expected in receipt["outputs"].items():
            actual = sha(path / relative)
            if actual != expected:
                raise ValueError(f"Changed run output: {run}/{relative}")
            input_hashes[str(path / relative)] = actual
        input_hashes[str(path / "receipt.json")] = sha(path / "receipt.json")
        bundles.append(dict(path=path, summary=read_json(path / "summary.json"),
                            config=read_json(path / "config.json"),
                            history=pd.DataFrame(read_json(path / "market/history.json")),
                            predictions=pd.read_parquet(path / "predictions.parquet")))
    base = bundles[0]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "figure.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})

    def save(fig, filename, caption):
        fig.get_layout_engine().set(rect=(0, .055, 1, .945))
        fig.text(.015, .012, caption, fontsize=9, color="#334155", va="bottom")
        fig.savefig(out / filename, dpi=160, bbox_inches="tight", pad_inches=.18)
        plt.close(fig)
        saved.append(dict(file=filename, sha256=sha(out / filename), caption=caption))

    # 1. Intentionally easy controls versus actual held-out market learning.
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), layout="constrained")
    fig.suptitle("Learning works on designed controls; market generalization is a separate test", fontsize=16)
    controls = read_json(base["path"] / "controls.json")
    ax = axes[0, 0]
    for sub, color, name in (("positive", "#059669", "Persistent drift"),
                              ("shuffled", "#b91c1c", "Shuffled TRAIN labels")):
        h = pd.DataFrame(read_json(base["path"] / sub / "history.json"))
        ax.plot(h.epoch, h.validation_accuracy, color=color, label=name + " / validation")
        ax.plot(h.epoch, h.train_accuracy, color=color, ls=":", alpha=.7, label=name + " / train")
    ax.axhline(1 / 3, color="grey", ls="--", lw=1, label="Three-state chance")
    ax.set(title="Synthetic controls: separate train, validation and test seeds", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    ax.legend(fontsize=8, loc="lower right")
    ax.text(.02, .87, f"Held-out test: positive {controls['positive']['test']['accuracy']:.1%}; "
            f"shuffled {controls['shuffled']['test']['accuracy']:.1%}", transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    ax = axes[0, 1]
    h = pd.DataFrame(read_json(base["path"] / "memorize/history.json"))
    ax.plot(h.epoch, h.train_accuracy, color="#059669", lw=2)
    ax.axhline(.95, color="grey", ls="--", label="Predeclared 95% check")
    ax.set(title="Memorize one small real TRAIN batch", xlabel="Epoch", ylabel="Same-batch accuracy", ylim=(0, 1.05))
    ax.text(.04, .18, "Same samples for fitting and checking.\nDropout and weight decay disabled.\nCapacity/backprop check; no generalization claim.",
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor="white", edgecolor="#cbd5e1"))
    ax.legend(loc="lower right", fontsize=8)
    for k, b in enumerate(bundles[:2]):
        ax, h, s = axes[1, k], b["history"], b["summary"]
        ax.plot(h.epoch, h.train_nll, color=COLORS[k], label="Train NLL")
        ax.plot(h.epoch, h.validation_nll, color="#b91c1c", label="Validation NLL")
        ax.axhline(s["evaluation"]["validation"]["prior"]["joint_nll"], color="black", ls=":", label="Train-prior / validation")
        ax.axvline(s["training"]["best_epoch"], color="#059669", ls="--", label=f"Selected epoch {s['training']['best_epoch']}")
        ax.set(title=NAMES[k] + " | identical execution target/cohort", xlabel="Epoch", ylabel="Joint NLL (lower is better)")
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    save(fig, "01_learning_and_controls.png", "Baseline TCN controls. TRAIN-shuffle retains true validation labels for epoch selection: weak contrast, NOT a strict null/leakage proof. Test never selects an epoch.")

    # 2. Denominators and target disagreement, retaining unknown clock roots.
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
    fig.suptitle("Execution and crossing answer different questions", fontsize=16)
    p = base["predictions"]
    days = sorted(p.date.unique())
    for row, column, title in ((0, "execution_target", "Fixed-plan actual delayed net > 0"),
                               (1, "crossing_target", "Any BB checkpoint net >= 2R within 300 s")):
        ax = fig.add_subplot(gs[row, 0])
        counts = np.array([[int((p.loc[p.date == d, column] == state).sum()) for state in [-1, 0, 1, 2, 3]] for d in days])
        bottom = np.zeros(len(days))
        for j, (name, color) in enumerate(zip(STATE_NAMES, STATE_COLORS)):
            ax.bar(np.arange(len(days)), counts[:, j], bottom=bottom, color=color, label=name, width=.7)
            bottom += counts[:, j]
        ax.set(xticks=np.arange(len(days)), xticklabels=[d[5:] for d in days], ylabel="Clock roots (all, before X admission)", title=title)
        ax.set_ylim(0, bottom.max() * 1.12)
        for j in range(len(days)):
            ax.text(j, bottom[j] + 15, f"n={int(bottom[j])}", ha="center", fontsize=8)
        if row == 0:
            ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12))
    ax = fig.add_subplot(gs[:, 1])
    matrix = np.asarray(base["summary"]["diagnostics"]["crossing_rows_execution_columns"])
    row_sum = matrix.sum(axis=1, keepdims=True)
    fractions = np.divide(matrix, row_sum, out=np.zeros(matrix.shape, float), where=row_sum > 0)
    img = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{matrix[i,j]:,}\n{fractions[i,j]:.1%} of row", ha="center", va="center",
                    color="white" if fractions[i,j] > .55 else "#0f172a", fontsize=11)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=STATE_NAMES[1:], yticklabels=STATE_NAMES[1:],
           xlabel="Execution outcome", ylabel="Crossing outcome", title=f"Common known labels + available X: n={matrix.sum():,}\nCounts and row percentages; all five days")
    fig.colorbar(img, ax=ax, shrink=.55, label="Fraction within crossing-state row")
    save(fig, "02_cohorts_and_target_relation.png", "Unknown is retained in clock-root bars. Confusion uses only common known outcomes with available X; it does not compare predictive model errors.")

    # 3. Fixed early test interval, selected solely by the clock before plotting.
    day = "2026-02-06"
    day_rows = p.loc[p.date == day]
    begin = int(day_rows.t_ns.min())
    end = begin + 2 * 3600 * 1_000_000_000
    fixed = p.loc[(p.t_ns >= begin) & (p.t_ns < end)].copy()
    minutes = (fixed.t_ns.to_numpy() - begin) / 60e9
    fig, axes = plt.subplots(4, 1, figsize=(17, 11), sharex=True,
                              gridspec_kw={"height_ratios": [.8, 1.2, 1.2, 1.2]}, layout="constrained")
    start_text = pd.Timestamp(begin, tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    fig.suptitle(f"Fixed first two hours of the test session: {start_text} | 30-second roots", fontsize=15)
    states = np.vstack((fixed.execution_target, fixed.crossing_target)) + 1
    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(np.arange(-.5, 5.5), cmap.N)
    axes[0].imshow(states, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm,
                   extent=[0, 120, 1.5, -.5])
    axes[0].set(yticks=[0, 1], yticklabels=["Execution", "Crossing"], title="Observed target states: discrete cells, no smoothing")
    axes[0].legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(STATE_COLORS, STATE_NAMES)],
                   ncol=5, fontsize=9, loc="upper right", bbox_to_anchor=(1, 1.55))
    for k, b in enumerate(bundles):
        ax = axes[k + 1]
        sample = b["predictions"].loc[(b["predictions"].t_ns >= begin) & (b["predictions"].t_ns < end)]
        if not np.array_equal(sample.t_ns, fixed.t_ns):
            raise ValueError("Fixed case prediction clocks differ")
        valid = sample.feature_available.to_numpy(bool)
        for value, label, color in ((1 - sample.p_none.to_numpy(), "pE", "#111827"),
                                    ((sample.p_long + sample.p_both).to_numpy(), "P(LONG success)", "#059669"),
                                    ((sample.p_short + sample.p_both).to_numpy(), "P(SHORT success)", "#dc2626")):
            value = value.copy()
            value[~valid] = np.nan
            ax.plot(minutes, value, color=color, label=label, lw=1.1)
        unknown = sample.target.to_numpy() < 0
        for t in minutes[unknown]:
            ax.axvspan(t, t + .5, color="#d1d5db", alpha=.45, lw=0)
        ax.set(title=NAMES[k] + " | gray intervals: this target unknown", ylabel="Probability / score", ylim=(0, 1))
        ax.grid(alpha=.15)
        ax.legend(loc="upper right", ncol=3, fontsize=9)
    axes[-1].set(xlabel="Minutes since first scheduled test-session origin", xlim=(0, 120), xticks=np.arange(0, 121, 15))
    save(fig, "03_fixed_test_timeline.png", "Interval selected by session clock, without outcome search. Scores are shown only where X is available. Unknown future labels are not NONE.")
    fixed_receipt = dict(selection="First 2 hours from first scheduled Feb 6 origin; no outcome selection",
                         start_ns=str(begin), end_exclusive_ns=str(end), origins=len(fixed), cadence_seconds=30)

    # 4. Calibration day reliability and strictly held-out test distributions.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")
    fig.suptitle("Calibration diagnostics and held-out test scores | no fitted probability calibrator (T=1)", fontsize=15)
    for k, b in enumerate(bundles):
        ax = axes[0, k]
        m = b["summary"]["evaluation"]["calibration"]["model"]
        bins = [v for v in m["reliability"] if v["count"]]
        ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
        xx, yy = [v["probability"] for v in bins], [v["frequency"] for v in bins]
        ax.scatter(xx, yy, s=[30 + 250 * v["count"] / m["n"] for v in bins], color=COLORS[k], alpha=.8)
        for xval, yval, v in zip(xx, yy, bins):
            ax.annotate(str(v["count"]), (xval, yval), xytext=(6, 5), textcoords="offset points", fontsize=8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted pE in bin", ylabel="Observed event frequency",
               title=f"{NAMES[k]} | calibration day\nn={m['n']:,}; ECE={m['event_ece']:.3f}")
        ax.grid(alpha=.15)
    for k, b in enumerate(bundles):
        pred = b["predictions"]
        sel = (pred["split"] == "test") & pred.feature_available & (pred.target >= 0)
        ax = axes[1, 0] if k < 2 else axes[1, 1]
        ep = 1 - pred.loc[sel, "p_none"].to_numpy()
        ax.hist(ep, bins=np.linspace(0, 1, 31), histtype="step", lw=2, color=COLORS[k],
                weights=np.ones(len(ep)) / len(ep), label=f"{NAMES[k]} (n={len(ep):,})")
        ax.set(xlim=(0, 1), xlabel="Predicted pE on known test roots", ylabel="Fraction per bin")
        ax.legend(fontsize=8)
    axes[1, 0].set_title("Execution target: same test cohort")
    axes[1, 1].set_title("Crossing target: separate test cohort")
    ax = axes[1, 2]
    for k, b in enumerate(bundles):
        e = b["summary"]["evaluation"]["test"]
        ax.bar(k - .16, e["model"]["joint_nll"], width=.3, color=COLORS[k])
        ax.bar(k + .16, e["prior"]["joint_nll"], width=.3, color="#cbd5e1")
        ax.text(k, max(e["model"]["joint_nll"], e["prior"]["joint_nll"]) + .025,
                f"n={e['model']['n']:,}", ha="center", fontsize=8)
    ax.set(xticks=range(3), xticklabels=["TCN\nexecution", "MLP\nexecution", "TCN\ncrossing"],
           ylabel="Joint NLL (lower is better)", title="Compare each model with its own train-prior")
    ax.axvline(1.5, color="#64748b", ls="--", lw=1)
    nll_max = max(b["summary"]["evaluation"]["test"][kind]["joint_nll"] for b in bundles for kind in ("model", "prior"))
    ax.set_ylim(0, nll_max * 1.25)
    ax.legend(handles=[Patch(facecolor="#2563eb", label="Model"), Patch(facecolor="#cbd5e1", label="Own train-prior")], fontsize=8, loc="upper left", ncol=2)
    save(fig, "04_calibration_and_distributions.png", "Top: calibration-day reliability; numbers are bin counts, marker area tracks count. Bottom: TEST. NLL across different targets/cohorts is not an architecture ranking.")

    # 5. Optimization mechanics, separate from trading success.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")
    fig.suptitle("Gradient scale, shared-encoder alignment and clipping", fontsize=16)
    for k, b in enumerate(bundles):
        s, h, ax = b["summary"], b["history"], axes[0, k]
        before, after = s["training"]["gradients_before"], s["training"]["gradients_selected"]
        ax.bar(np.array([0, 1]) - .17, [before["event_bce_encoder_norm"], after["event_bce_encoder_norm"]], width=.32, color="#2563eb", label="Event BCE")
        ax.bar(np.array([0, 1]) + .17, [before["direction_encoder_norm"], after["direction_encoder_norm"]], width=.32, color="#d97706", label="Conditional direction CE")
        ax.set(xticks=[0, 1], xticklabels=["Before fit", "Selected checkpoint"], ylabel="Unclipped encoder gradient L2 norm",
               title=NAMES[k] + f" | fixed {before['n']}-root TRAIN probe")
        ax.text(.02, .97, f"E/D cosine: {before['cosine_event_direction']:+.3f} -> {after['cosine_event_direction']:+.3f}",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.set_ylim(0, max(before["event_bce_encoder_norm"], before["direction_encoder_norm"],
                           after["event_bce_encoder_norm"], after["direction_encoder_norm"]) * 1.28)
        ax.legend(fontsize=8, loc="upper right")
        ax = axes[1, k]
        ax.plot(h.epoch, h.gradient_norm_mean, color="#2563eb", label="Mean pre-clip norm")
        ax.plot(h.epoch, h.gradient_norm_max, color="#d97706", label="Max pre-clip norm")
        ax.axhline(b["config"]["training"]["gradient_clip"], color="#dc2626", ls="--", lw=1, label="Clip threshold")
        ax.axvline(s["training"]["best_epoch"], color="#059669", ls=":", label="Selected epoch")
        ax.set(xlabel="Epoch", ylabel="All-parameter gradient L2 norm")
        right = ax.twinx()
        right.plot(h.epoch, h.gradient_clip_fraction, color="#7c3aed", ls="--", alpha=.7)
        right.set(ylim=(-.02, 1.05), ylabel="Clipped batches / all batches")
        right.yaxis.label.set_color("#7c3aed")
        right.tick_params(axis="y", colors="#7c3aed")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=.15)
    save(fig, "05_gradients_and_clipping.png", "Top excludes the final output layer and uses a shared root denominator. A negative cosine at one probe is a diagnostic, not proof that two separate models are superior.")

    receipt = dict(status="COMPLETE_VISUAL_DIAGNOSTICS", script_sha256=sha(__file__),
                   input_sha256=input_hashes, run_ids=list(RUNS), figures=saved, fixed_case=fixed_receipt,
                   exclusions="No account-equity chart; no empty WAIT projection presented as profit.")
    (out / "visual_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status="COMPLETE", out=str(out), figures=[v["file"] for v in saved])))


if __name__ == "__main__":
    main()
