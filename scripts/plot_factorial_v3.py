"""Render prespecified paired TEST effects after the independent v3 audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CONTRASTS = (
    ("context_at_uniform", "Restored history\nat uniform weights"),
    ("weighting_at_restored", "Crossing-ID weights\nat restored history"),
    ("interaction", "Interaction:\nextra combined benefit"),
)
COLORS = {"17": "#2774A6", "29": "#D47925"}


def plot(audit, output):
    if audit.get("status") != "COMPLETE" or len(audit.get("runs", {})) != 8:
        raise ValueError("Final independent eight-fit audit required")
    selected = audit["factorial"]["by_split"]["test"]
    metrics = (("joint_nll", "Uniform-root joint NLL", "All known admitted roots: NONE / LONG / SHORT / BOTH"),
               ("event_id_positive_side_nll", "Positive-side crossing-ID NLL", "Each (side, ID) equal; roots averaged within that group"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "#F5F4F0", "axes.facecolor": "#FFFFFF"})
    figure, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=150, sharey=True)
    dates_reference = None
    for ax, (metric, title, subtitle) in zip(axes, metrics):
        cell = selected[metric]
        if cell["paired_dates"] != 10:
            raise ValueError("Exactly ten matched TEST days required for this fixed report")
        extent = []
        for j, (contrast, label) in enumerate(CONTRASTS):
            y = 2-j
            combined = cell["mean_seed_effect_paired_by_date"][contrast]
            dates = combined["dates"]
            if dates_reference is None:
                dates_reference = dates
            if dates != dates_reference:
                raise ValueError("Metric/contrast date cohorts differ")
            for seed, offset in (("17", .16), ("29", -.16)):
                effect = cell["by_seed"][seed][contrast]
                if effect["dates"] != dates:
                    raise ValueError("Seed date cohorts differ")
                values = np.asarray([row["effect"] for row in effect["by_day"]])
                # A fixed day-dependent vertical offset makes overlapping days visible.
                jitter = np.linspace(-.055, .055, len(values))
                ax.scatter(values, y+offset+jitter, s=45, color=COLORS[seed], alpha=.7,
                    edgecolors="white", linewidth=.5, zorder=3)
                ax.plot(effect["mean"], y+offset, marker="|", markersize=17, color=COLORS[seed],
                    markeredgewidth=2.7, zorder=4)
                extent.extend(values.tolist())
            lo, hi = combined["percentile_95_interval"]
            center = combined["mean"]
            ax.errorbar(center, y, xerr=[[max(0, center-lo)], [max(0, hi-center)]], fmt="D",
                color="#18242C", markersize=6, elinewidth=2.2, capsize=4, zorder=5)
            ax.text(.98, (y+.88)/3, f"mean {center:+.4f}   95% [{lo:+.4f}, {hi:+.4f}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#34434B",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=2))
            extent.extend([lo, hi])
        ax.axvline(0, color="#68777E", linewidth=1.1)
        ax.set_ylim(-.5, 2.5)
        lower, upper = min(extent+[0]), max(extent+[0])
        padding = max((upper-lower)*.14, .002)
        ax.set_xlim(lower-padding, upper+padding)
        ax.grid(axis="x", alpha=.16)
        for boundary in (.5, 1.5):
            ax.axhline(boundary, color="#DDE2E4", linewidth=.7)
        ax.set_title(title+"\n"+subtitle, loc="left", fontsize=13, pad=19)
        ax.set_xlabel("NLL improvement (positive is better)", labelpad=12)
        ax.set_yticks([2, 1, 0], [label for _, label in CONTRASTS])
        ax.tick_params(axis="y", length=0, pad=10)
    legend = [Line2D([], [], marker="o", color="none", markerfacecolor=COLORS[s], markeredgecolor="white",
               markersize=8, label=f"seed {s}: 10 daily effects (bar = seed mean)") for s in ("17", "29")]
    legend.append(Line2D([], [], marker="D", color="#18242C", markersize=6,
        label="Mean across seeds within each date + descriptive 95% day-bootstrap interval"))
    figure.suptitle("Context × crossing-ID weighting: matched TEST-day effects", x=.05, y=.97,
        ha="left", fontsize=21, fontweight="bold", color="#18242C")
    figure.legend(handles=legend, loc="lower left", bbox_to_anchor=(.05, .125), frameon=False, fontsize=10)
    figure.text(.05, .09, "10 dates: "+", ".join(d[5:] for d in dates_reference)+". 20,000 paired day resamples; the two seeds do not double the day count.",
        fontsize=10, color="#45545C")
    figure.text(.05, .035, "Right panel is POSITIVE-ONLY: it cannot assess false positives, full-population calibration, or profit.\n"
        "Intervals are descriptive; days/events may be dependent. All eight prespecified fits remain reported; no TEST winner selection.",
        fontsize=10, color="#45545C", linespacing=1.6)
    figure.subplots_adjust(left=.18, right=.98, top=.82, bottom=.29, wspace=.14)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--audit", default="reports/scientific_pipeline_v3/statistical_audit.json")
    parser.add_argument("--output", default="reports/scientific_pipeline_v3/08_paired_effects.png")
    parser.add_argument("--receipt", default="reports/scientific_pipeline_v3/paired_figure_receipt.json")
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    plot(audit, args.output)
    hashes = {str(Path(path)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for path in (Path(__file__), args.audit, args.output)}
    receipt = dict(status="COMPLETE", source_hashes=hashes, output=args.output,
        split="test", metrics=["joint_nll", "event_id_positive_side_nll"],
        contrasts=[contrast for contrast, _ in CONTRASTS], seeds=[17, 29],
        dates=audit["factorial"]["by_split"]["test"]["joint_nll"]["mean_seed_effect_paired_by_date"]["context_at_uniform"]["dates"],
        definition="All 10 paired daily effects per seed; diamond averages seeds within day, then averages days. Bars are descriptive 95% paired day-bootstrap intervals, not independent-seed intervals.",
        limitation="Side-ID panel is positive-only and cannot assess false positives, full-population calibration, or profit; no TEST winner selection.")
    Path(args.receipt).write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(output=args.output, sha256=hashes[str(Path(args.output))], receipt=args.receipt)))


if __name__ == "__main__":
    main()
