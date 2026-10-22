"""Source-bound visual report for the completed scientific opportunity v2 suite.

python -m scripts.report_opportunity_v2

This script only reads completed datasets/runs and writes report artifacts. It
does not fit models, select a candidate, or smooth observed target labels.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


STATE_NAMES = ("UNKNOWN", "NONE", "LONG only", "SHORT only", "BOTH")
STATE_COLORS = ("#cbd5e1", "#475569", "#059669", "#dc2626", "#8b5cf6")
FAMILY_COLORS = dict(geometry="#2563eb", micro="#d97706", rich="#7c3aed")
FIXED_CASES = (("2026-05-11", 0, "03a_test_open_may11.png"),
               ("2026-05-18", 5, "03b_test_afternoon_may18.png"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _safe_relative(root, name):
    p = (root / name).resolve()
    if root.resolve() not in p.parents:
        raise ValueError("Artifact path escapes its run/dataset root")
    return p


def _run_entries(suite):
    runs = suite["runs"]
    if isinstance(runs, dict):
        return [dict(run_id=k, **({"role": v.get("role", "development")} if isinstance(v, dict) else {})) for k, v in runs.items()]
    return [dict(run_id=v, role="development") if isinstance(v, str) else v for v in runs]


def load_inputs(suite_path, out):
    suite = read(suite_path)
    if suite.get("status") != "COMPLETE":
        raise ValueError("Report requires a COMPLETE suite, not intermediate results")
    dataset = Path(suite["dataset_root"])
    runs_root = Path(suite["runs_root"])
    manifest = read(dataset / "manifest.json")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Dataset is not COMPLETE")
    manifest_sha = sha(dataset / "manifest.json")
    if suite.get("dataset_manifest_sha256", manifest_sha) != manifest_sha:
        raise ValueError("Suite dataset manifest changed")
    inputs = {str(Path(suite_path)): sha(suite_path), str(dataset / "manifest.json"): manifest_sha}
    bundles = []
    for entry in _run_entries(suite):
        run = _safe_relative(runs_root, entry["run_id"])
        receipt = read(run / "receipt.json")
        if receipt.get("status") != "COMPLETE" or receipt.get("dataset_manifest_sha256") != manifest_sha:
            raise ValueError("Run is incomplete or uses a different dataset: " + entry["run_id"])
        for filename, expected in receipt["outputs"].items():
            path = _safe_relative(run, filename)
            if sha(path) != expected:
                raise ValueError("Run artifact changed: " + str(path))
            inputs[str(path)] = expected
        inputs[str(run / "receipt.json")] = sha(run / "receipt.json")
        bundles.append(dict(run_id=entry["run_id"], role=entry.get("role", "development"), path=run,
                            summary=read(run / "summary.json"), config=read(run / "config.json"),
                            history=pd.DataFrame(read(run / "history.json")), gradient=read(run / "gradient_artifact.json")))
    by_id = {b["run_id"]: b for b in bundles}
    primary, confirmation = by_id[suite["primary_run_id"]], by_id[suite["confirmation_run_id"]]
    development = [b for b in bundles if b["role"] == "development"]
    common = development[0]["summary"]["cohorts"]
    shared_scaler = development[0]["summary"]["scaler_sha256"]
    if any(b["summary"]["cohorts"] != common or b["summary"]["scaler_sha256"] != shared_scaler for b in bundles):
        raise ValueError("Cohorts/scalers differ: cannot present a matched family comparison")
    # Labels, identity and fixed time cuts come from saved dataset rows, not
    # from scores or observed profitability.
    root_parts = []
    day_receipts = {}
    for day in manifest["days"]:
        day_receipts[day["date"]] = day
        path = _safe_relative(dataset, day["roots_file"])
        if sha(path) != day["files"][day["roots_file"]]:
            raise ValueError("Root dataset changed: " + str(path))
        inputs[str(path)] = day["files"][day["roots_file"]]
        root_parts.append(pd.read_parquet(path))
    roots = pd.concat(root_parts, ignore_index=True)
    if not roots.t_ns.is_monotonic_increasing or roots.t_ns.duplicated().any():
        raise ValueError("Origin clock is not unique and chronological")
    for b in (primary, confirmation):
        pred = pd.read_parquet(b["path"] / "predictions.parquet")
        if (not np.array_equal(pred.t_ns, roots.t_ns) or not np.array_equal(pred.root_index, np.arange(len(roots)))
                or not np.array_equal(pred.y_crossing, roots.y_crossing)):
            raise ValueError("Prediction/root/target alignment changed")
        b["predictions"] = pred
    diagnostics_path = out / "dataset_diagnostics.json"
    diagnostics = read(diagnostics_path) if diagnostics_path.exists() else None
    if diagnostics is not None:
        if diagnostics["manifest_sha256"] != manifest_sha:
            raise ValueError("Dataset diagnostics use another dataset")
        inputs[str(diagnostics_path)] = sha(diagnostics_path)
    return dict(suite=suite, dataset=dataset, manifest=manifest, bundles=bundles, primary=primary,
                confirmation=confirmation, development=development, roots=roots, days=day_receipts,
                diagnostics=diagnostics, input_sha256=inputs, shared_scaler_sha256=shared_scaler)


def _label(bundle):
    c = bundle["config"]
    return f"{c['model']['architecture'].upper()} {c['data']['family']} / seed {c['seed']}"


def _save_figure(fig, out, filename, title, caption, figures):
    wrapped = textwrap.fill(caption, width=185)
    footer_height = max(.055, (wrapped.count("\n") + 1) * .018)
    fig.get_layout_engine().set(rect=(0, footer_height, 1, .945))
    fig.text(.015, .012, wrapped, fontsize=9, color="#334155", va="bottom")
    fig.savefig(out / filename, dpi=160, bbox_inches="tight", pad_inches=.18)
    plt.close(fig)
    figures.append(dict(file=filename, title=title, caption=caption, sha256=sha(out / filename)))


def plot_coverage(data, out, figures):
    manifest, roots, days = data["manifest"], data["roots"], data["manifest"]["dates"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), layout="constrained")
    fig.suptitle("Full feature families share the same origin clock, target and TRAIN-only scaler", fontsize=16)
    groups = manifest["groups"]
    families = [f for f in ("geometry", "micro", "rich") if f in groups]
    ax = axes[0, 0]
    numbers = np.array([len(groups[f]) for f in families])
    ax.bar(np.arange(len(families)) - .18, numbers, width=.35, label="Source value channels", color="#2563eb")
    ax.bar(np.arange(len(families)) + .18, 2 * numbers + 1, width=.35, label="Model: values + masks + row flag", color="#94a3b8")
    for i, n in enumerate(numbers):
        ax.text(i - .18, n + 3, str(n), ha="center", fontsize=9)
        ax.text(i + .18, 2*n + 4, str(2*n+1), ha="center", fontsize=9)
    ax.set(xticks=range(len(families)), xticklabels=families, ylabel="Channels", title="Included information; not proof of model utilization")
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    counts = []
    for d in days:
        r = roots.loc[roots.date == d]
        available = r.feature_available.to_numpy(bool)
        known = r.y_crossing.to_numpy() >= 0
        counts.append([int((available & known).sum()), int((available & ~known).sum()), int((~available).sum())])
    counts = np.array(counts)
    bottom = np.zeros(len(days))
    for j, (name, color) in enumerate((("Known Y + X: common fit cohort", "#059669"),
                                      ("X available, Y unknown", "#d97706"), ("X unavailable", "#cbd5e1"))):
        ax.bar(np.arange(len(days)), counts[:, j], bottom=bottom, color=color, label=name, width=.9)
        bottom += counts[:, j]
    tick = np.arange(0, len(days), max(1, len(days)//9))
    ax.set(xticks=tick, xticklabels=[days[i][5:] for i in tick], ylabel="Clock roots", title=f"All {len(days)} days: unknowns are retained")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1, 0]
    coverage = []
    for family in families:
        coverage.append([1 - float(np.mean([data["days"][d]["feature_receipt"]["missing_fraction"][n]
                                            for n in groups[family]])) for d in days])
    coverage = np.array(coverage)
    image = ax.imshow(coverage, aspect="auto", vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    ax.set(yticks=range(len(families)), yticklabels=[f"{f} ({len(groups[f])} values)" for f in families],
           xticks=tick, xticklabels=[days[i][5:] for i in tick], title="Mean finite feature fraction over ALL daily roots")
    fig.colorbar(image, ax=ax, shrink=.7, label="Fraction finite (masks retain missingness)")
    ax = axes[1, 1]
    role_names = ["train", "validation", "calibration", "test"]
    baseline = data["primary"]["summary"]["cohorts"]
    n = [baseline[r]["known_feature_available_roots"] for r in role_names]
    ax.bar(role_names, n, color=["#2563eb", "#7c3aed", "#d97706", "#059669"])
    for i, size in enumerate(n):
        ax.text(i, size + max(n)*.02, f"{size:,}", ha="center", fontsize=10)
    ax.set_ylim(0, max(n)*1.3)
    ax.set(ylabel="Known roots admitted for metrics/loss", title=f"Identical cohort hashes in all {len(data['bundles'])} runs")
    ax.text(.02, .94, "Same scaler SHA across families and seed replication\nUNKNOWN does not reset causal sequence history",
            transform=ax.transAxes, va="top", fontsize=9)
    _save_figure(fig, out, "01_features_and_cohorts.png", "Признаки, покрытие и общая выборка",
                 "Availability is causal and shared by every family. Presence of a feature is not evidence that the model uses it. Unknown future outcomes remain on the clock.", figures)


def plot_learning(data, out, figures):
    families = [f for f in ("geometry", "micro", "rich") if f in data["manifest"]["groups"]]
    fig, axes = plt.subplots(2, len(families), figsize=(18, 10), squeeze=False, layout="constrained")
    fig.suptitle("Matched development comparison: validation selects the model; test does not", fontsize=16)
    for row, architecture in enumerate(("mlp", "tcn")):
        for col, family in enumerate(families):
            ax = axes[row, col]
            matches = [b for b in data["development"] if b["config"]["model"]["architecture"] == architecture and b["config"]["data"]["family"] == family]
            if len(matches) != 1:
                ax.text(.5, .5, "No single completed development run", transform=ax.transAxes, ha="center")
                continue
            b, = matches
            h, summary = b["history"], b["summary"]
            ax.plot(h["update"], h.train_probe_nll, color="#2563eb", label="Fixed TRAIN probe NLL")
            ax.plot(h["update"], h.validation_nll, color="#dc2626", label="All validation known roots")
            ax.axhline(summary["evaluation"]["validation"]["prior"]["joint_nll"], color="black", ls=":", label="TRAIN-prior on validation")
            selected = summary["training"]["best_update"]
            ax.axvline(selected, color="#059669", ls="--", label=f"Selected update {selected}")
            ax.set(title=f"{architecture.upper()} / {family} — {summary['training']['parameter_count']:,} parameters\nmax real-token span {summary['training']['detailed_history_seconds']:g} s; actual: see 07",
                   xlabel="Optimizer updates (0 = untrained)", ylabel="Joint NLL")
            ax.grid(alpha=.15)
            ax.legend(fontsize=8)
    _save_figure(fig, out, "02_learning_by_family.png", "Обучение при смене семейства и архитектуры",
                 "TRAIN curves use the same fixed probe, not every training root. A selected update 0 is explicitly an untrained network, not successful learning.", figures)


def plot_test_comparison(data, out, figures):
    bundles = data["bundles"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.7), layout="constrained")
    fig.suptitle("Checkpoint and family selected by VAL; TEST is descriptive after that selection", fontsize=15)
    yy = np.arange(len(bundles))
    labels = [_label(b) + (" * primary" if b is data["primary"] else " * replication" if b is data["confirmation"] else "") for b in bundles]
    for ax, role in zip(axes[:2], ("validation", "test")):
        model = np.array([b["summary"]["evaluation"][role]["model"]["joint_nll"] for b in bundles])
        prior = np.array([b["summary"]["evaluation"][role]["prior"]["joint_nll"] for b in bundles])
        delta = model - prior
        ax.barh(yy, delta, color=["#059669" if d < 0 else "#dc2626" for d in delta])
        ax.axvline(0, color="black", lw=1)
        ax.set(yticks=yy, yticklabels=labels if role == "validation" else [""] * len(labels),
               xlabel="Model NLL - own TRAIN-prior NLL", title=f"{role.upper()} | negative is improvement")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    ax = axes[2]
    for b, color in ((data["primary"], "#2563eb"), (data["confirmation"], "#d97706")):
        bins = [r for r in b["summary"]["evaluation"]["test"]["model"]["reliability"] if r["count"]]
        ax.plot([r["probability"] for r in bins], [r["frequency"] for r in bins], "o-", color=color, label=_label(b), lw=1)
        for r in bins:
            ax.annotate(str(r["count"]), (r["probability"], r["frequency"]), xytext=(4, 4), textcoords="offset points", fontsize=7, color=color)
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted pE in bin", ylabel="Observed event frequency",
           title="Primary + prespecified seed replication: TEST")
    ax.legend(fontsize=8)
    _save_figure(fig, out, "03_selection_and_test.png", "Выбор по validation и проверка на test",
                 "The seed replication is not a second test-selected winner. Bin labels are root counts; correlated roots do not supply independent confidence intervals.", figures)


def _shade_unknown(ax, minutes, unknown, cadence_seconds):
    if not np.any(unknown):
        return
    edges = np.flatnonzero(np.diff(np.r_[False, unknown, False]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(minutes[a], minutes[b - 1] + cadence_seconds / 60, color="#cbd5e1", alpha=.55, lw=0)


def plot_fixed_case(data, out, date, offset_hours, filename, figures):
    day = data["days"][date]
    roots = data["roots"].loc[data["roots"].date == date]
    begin = int(roots.t_ns.min()) + offset_hours * 3600 * 1_000_000_000
    end = begin + 2 * 3600 * 1_000_000_000
    scene_name = f"{date}_scene.parquet"
    scene_path = data["dataset"] / scene_name
    expected = day["files"][scene_name]
    if sha(scene_path) != expected:
        raise ValueError("Fixed-case causal scene changed")
    data["input_sha256"][str(scene_path)] = expected
    scene = pd.read_parquet(scene_path)
    primary, confirmation = data["primary"], data["confirmation"]
    pred = primary["predictions"]
    mask = (pred.t_ns >= begin) & (pred.t_ns < end)
    p = pred.loc[mask].copy()
    c = confirmation["predictions"].loc[mask].copy()
    scene = p[["t_ns"]].merge(scene, on="t_ns", how="left", validate="one_to_one", sort=False)
    if len(p) == 0 or not np.array_equal(p.t_ns, c.t_ns) or scene.mid.isna().all():
        raise ValueError("Fixed time case has no aligned observations")
    x = (p.t_ns.to_numpy() - begin) / 60e9
    unknown = p.y_crossing.to_numpy() < 0
    admitted = p.feature_available.to_numpy(bool)
    y = p.y_crossing.to_numpy()
    target_e = np.where(y >= 0, (y > 0).astype(float), np.nan)
    ep = 1 - p.p_none.to_numpy()
    ep[~admitted] = np.nan
    cp = 1 - c.p_none.to_numpy()
    cp[~admitted] = np.nan
    dt = primary["config"]["data"]["cadence_seconds"]
    fig, axes = plt.subplots(5, 1, figsize=(19, 14), sharex=True,
        gridspec_kw={"height_ratios": [2., 1.1, .38, 1.1, 1.2]}, layout="constrained")
    start = pd.Timestamp(begin, tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"Fixed TEST clock interval: {start} to +2 h | {dt:g}-second roots | no outcome selection", fontsize=15)
    ax = axes[0]
    ax.plot(x, scene.mid, color="#111827", lw=1, label="Causal midpoint")
    for scale, color in ((1, "#2563eb"), (3, "#d97706"), (6, "#7c3aed")):
        middle, half = scene[f"BB{scale}_middle"].to_numpy(), scene[f"BB{scale}_half"].to_numpy()
        ax.plot(x, middle, color=color, lw=1, label=f"BB M{scale} middle")
        ax.plot(x, middle + half, color=color, lw=.6, alpha=.65)
        ax.plot(x, middle - half, color=color, lw=.6, alpha=.65)
    ax.set(ylabel="Price, USD", title="Price and moving BB bands at the decision clock")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax = axes[1]
    ax.step(x, target_e, where="post", color="#94a3b8", lw=1, alpha=.8, label="Observed E (binary; no smoothing)")
    ax.plot(x, ep, color="#2563eb", lw=1.2, label="Primary sigmoid pE")
    ax.plot(x, cp, color="#d97706", lw=1, ls="--", label="Seed replication pE")
    _shade_unknown(ax, x, unknown, dt)
    ax.set(ylabel="Event", ylim=(-.05, 1.05), title="The binary target and probabilistic forecast are different objects")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    axes[2].imshow((y + 1)[None, :], interpolation="nearest", aspect="auto",
        cmap=ListedColormap(STATE_COLORS), norm=BoundaryNorm(np.arange(-.5, 5.5), 5), extent=[0, 120, .5, -.5])
    axes[2].set(yticks=[0], yticklabels=["Joint target"])
    axes[2].legend(handles=[Patch(facecolor=col, label=name) for name, col in zip(STATE_NAMES, STATE_COLORS)],
                   ncol=5, fontsize=8, loc="upper right", bbox_to_anchor=(1, 1.6))
    ax = axes[3]
    for values, color, label in (((p.p_long + p.p_both).to_numpy(), "#059669", "P(LONG crossing event)"),
                                 ((p.p_short + p.p_both).to_numpy(), "#dc2626", "P(SHORT crossing event)")):
        values[~admitted] = np.nan
        ax.plot(x, values, color=color, label=label, lw=1.1)
    _shade_unknown(ax, x, unknown, dt)
    ax.set(ylabel="Side probability", ylim=(-.05, 1.05), title="pL=pE(qL+qBOTH), pS=pE(qS+qBOTH); neither is 1 minus the other")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax = axes[4]
    gradient = p.g_event.to_numpy().copy()
    gradient[~admitted] = np.nan
    ax.plot(x, gradient, color="#7c3aed", lw=.8, label="Actual unreduced dL_i / dzE_i")
    ax.axhline(0, color="black", lw=.6)
    _shade_unknown(ax, x, unknown, dt)
    ax.set(ylabel="Loss-logit derivative", xlabel="Minutes from fixed interval start", xlim=(0, 120),
           title="BCE: dL/dzE = pE - E; this is NOT a temporal target derivative")
    ax.legend(fontsize=9, loc="upper right")
    for ax in (axes[0], axes[1], axes[3], axes[4]):
        ax.grid(alpha=.12)
    axes[-1].set_xticks(np.arange(0, 121, 15))
    _save_figure(fig, out, filename, f"Фиксированный участок {date}, London +{offset_hours} ч",
                 "Gray marks unknown outcomes. Forecasts use only admitted X. No target smoothing, profitable-case search or account-equity interpretation is applied.", figures)
    return dict(date=date, offset_from_first_origin_hours=offset_hours, start_ns=str(begin), end_exclusive_ns=str(end),
                roots=len(p), selection="Frozen clock offset from first scheduled London origin; no price/outcome search")


def plot_temporal(data, out, figures):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5), layout="constrained")
    fig.suptitle("Target changes, probability changes and loss gradients must not be confused", fontsize=16)
    bundles = data["bundles"]
    ax = axes[0, 0]
    tv = [b["gradient"]["temporal"]["event"]["test"]["probability_mean_absolute_step"] for b in bundles]
    reference = data["primary"]["gradient"]["temporal"]["event"]["test"]["target_flip_fraction"]
    ax.barh(np.arange(len(bundles)), tv, color=[FAMILY_COLORS[b["config"]["data"]["family"]] for b in bundles])
    ax.axvline(reference, color="#dc2626", ls="--", label=f"Binary E flip fraction = {reference:.3f}")
    ax.set(yticks=np.arange(len(bundles)), yticklabels=[_label(b) for b in bundles],
           xlabel="Mean absolute step on matched known adjacent roots", title="5-second TEST variation; a constant forecast also has low TV")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    primary, confirm = data["primary"], data["confirmation"]
    pred = primary["predictions"]
    test = pred.loc[pred["split"] == "test"]
    days = sorted(test.date.unique())
    ax = axes[0, 1]
    values = []
    for d in days:
        part = test.loc[test.date == d]
        values.append([part.delta_target_event.abs().mean(), part.delta_probability_event.abs().mean()])
    values = np.array(values)
    ax.plot(range(len(days)), values[:, 0], "o-", color="#dc2626", label="Observed E flip fraction")
    ax.plot(range(len(days)), values[:, 1], "o-", color="#2563eb", label="Primary probability TV")
    other = confirm["predictions"]
    cv = [other.loc[other.date == d, "delta_probability_event"].abs().mean() for d in days]
    ax.plot(range(len(days)), cv, "o--", color="#d97706", label="Replication probability TV")
    ax.set(xticks=range(len(days)), xticklabels=[d[5:] for d in days], ylabel="Mean absolute step", title="Every test day; same-support adjacent known roots")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    valid = test.delta_target_event.notna() & test.delta_probability_event.notna()
    for changed, color, name in ((False, "#2563eb", "E unchanged"), (True, "#dc2626", "E flips")):
        mask = valid & (test.delta_target_event != 0).eq(changed)
        val = test.loc[mask, "delta_probability_event"].abs().to_numpy()
        if len(val):
            ax.hist(val, bins=np.linspace(0, max(.05, float(test.delta_probability_event.abs().max())), 45),
                    weights=np.ones(len(val))/len(val), histtype="step", lw=1.5, color=color, label=f"{name} (n={len(val):,})")
    ax.set(xlabel="Absolute change in pE over 5 s", ylabel="Fraction per bin", title="Primary TEST: response at actual target transitions")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    known = test.loc[test.feature_available & (test.y_crossing >= 0)]
    sampled = known.iloc[np.linspace(0, len(known) - 1, min(4000, len(known)), dtype=int)] if len(known) else known
    for event, color in ((False, "#dc2626"), (True, "#059669")):
        part = sampled.loc[(sampled.y_crossing > 0) == event]
        ax.scatter(1 - part.p_none, part.g_event, s=3, alpha=.25, color=color, label=f"E={int(event)}")
    ax.set(xlabel="Predicted pE", ylabel="Unreduced actual dL/dzE", title="Loss derivative uses label error, not elapsed-time variation")
    ax.axhline(0, color="grey", lw=.7)
    ax.legend(fontsize=9)
    _save_figure(fig, out, "04_temporal_vs_loss_gradients.png", "Изменения во времени и градиенты loss",
                 "Indicator TV is computed without ordinal class distances. Low probability TV is not evidence of predictive skill; assess NLL, calibration and execution separately.", figures)


def plot_gradients(data, out, figures):
    primary, confirmation = data["primary"], data["confirmation"]
    groups = primary["gradient"]["full_clock_groups"]["test"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), layout="constrained")
    fig.suptitle("Actual loss gradients: example counts, gradient mass and shared encoder", fontsize=16)
    names = ["NONE", "LONG_ONLY", "SHORT_ONLY", "BOTH"]
    count = [groups[k]["count"] for k in names]
    ax = axes[0, 0]
    ax.bar(range(4), count, color=STATE_COLORS[1:])
    for i, n in enumerate(count):
        ax.text(i, n + max(count)*.02, f"{n:,}", ha="center")
    ax.set(xticks=range(4), xticklabels=names, ylabel="Known TEST roots with available X", title=_label(primary))
    ax.set_ylim(0, max(count)*1.15)
    ax = axes[0, 1]
    xx = np.arange(4)
    ax.bar(xx-.17, [groups[k]["event_absolute_gradient_mass_fraction"] for k in names], width=.32, color="#2563eb", label="Event |dL/dzE| mass")
    ax.bar(xx+.17, [groups[k]["direction_l2_mass_fraction"] for k in names], width=.32, color="#d97706", label="Direction gradient L2 mass")
    ax.set(xticks=xx, xticklabels=names, ylim=(0, 1.05), ylabel="Fraction of each head's total mass", title="NONE does not train conditional direction")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    keys = ["hard_p_true_lt_0.4", "intermediate_p_true_0.4_to_0.8", "easy_p_true_ge_0.8"]
    for j, (key, label, color) in enumerate((("fraction_known", "Example fraction", "#94a3b8"),
                                           ("event_absolute_gradient_mass_fraction", "Event gradient mass", "#2563eb"),
                                           ("direction_l2_mass_fraction", "Direction gradient mass", "#d97706"))):
        ax.bar(np.arange(3)+(j-1)*.24, [groups[k][key] for k in keys], width=.22, color=color, label=label)
    ax.set(xticks=range(3), xticklabels=["Hard\np(true)<.4", "Intermediate\n.4 <= p(true) < .8", "Easy\np(true)>=.8"],
           ylim=(0, 1.05), ylabel="Fraction", title="Difficulty is model-specific; counts and masses are separate")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    entries = [(b, phase) for b in (primary, confirmation) for phase in ("before", "selected")]
    for j, (b, phase) in enumerate(entries):
        g = b["gradient"][phase]
        ax.bar(j-.16, g["event_encoder_gradient_l2"], width=.3, color="#2563eb")
        ax.bar(j+.16, g["direction_encoder_gradient_l2"], width=.3, color="#d97706")
        cosine = g["cosine_event_direction"]
        text = "undefined" if cosine is None else f"{cosine:+.3f}"
        ax.annotate("cos " + text, (j, max(g["event_encoder_gradient_l2"], g["direction_encoder_gradient_l2"])),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xticks=range(4), xticklabels=[f"seed {b['config']['seed']}\n{phase}" for b, phase in entries],
           ylabel="Shared encoder gradient L2 norm", title="Frozen TRAIN probes; output layer excluded")
    ax.margins(y=.25)
    ax.legend(handles=[Patch(facecolor="#2563eb", label="Event loss"), Patch(facecolor="#d97706", label="Direction CE")], fontsize=8)
    _save_figure(fig, out, "05_gradient_mass_and_alignment.png", "Масса градиента и взаимодействие голов",
                 "Per-root logit derivatives are unreduced. Difficulty bins use joint p(true class): hard <0.4, intermediate 0.4-0.8, easy >=0.8. For positive events the joint probability includes pE, so hard bins are not independent evidence of difficult direction classification. Encoder gradients use the shared known-root mean; negative cosine alone does not establish that separate networks improve results.", figures)


def plot_economics(data, out, figures):
    primary, confirmation = data["primary"], data["confirmation"]
    path = primary["path"] / "economic_evaluation.json"
    if not path.exists():
        return None
    economics = read(path)
    other = read(confirmation["path"] / "economic_evaluation.json")
    plans = list(economics["paired_diagnostics"]["plans"])
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5), layout="constrained")
    fig.suptitle("Fixed crossing event, different stop budgets: risk diagnostics are separate from account returns", fontsize=15)
    paired = economics["paired_diagnostics"]
    ax = axes[0, 0]
    for j, (side, color) in enumerate((("LONG", "#059669"), ("SHORT", "#dc2626"))):
        values = [paired["plans"][plan]["by_split"]["test"]["conditional_crossing"][side]["positive_fraction"] for plan in plans]
        values = [np.nan if v is None else v for v in values]
        ax.bar(np.arange(len(plans)) + (j-.5)*.32, values, width=.3, color=color, label=side)
    ax.set(xticks=range(len(plans)), xticklabels=plans, ylim=(0, 1.05), ylabel="Fraction with actual delayed-exit net > 0",
           title="Among true favorable crossing labels on common observed TEST cohort")
    ax.legend()
    ax = axes[0, 1]
    for key, color, label in (("decision_cost_R_trade", "#dc2626", "Decision cost / trade risk"),
                              ("fixed_gain_R_trade", "#2563eb", "Fixed event gain / trade risk")):
        vals = [paired["plans"][plan]["by_split"]["test"][key]["q50"] for plan in plans]
        ax.plot(range(len(plans)), vals, "o-", color=color, label=label)
    ax.set(xticks=range(len(plans)), xticklabels=plans, ylabel="Median ratio", title="Wider stops alter payoff geometry; they do not alter E/D labels")
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    for plan, color in zip(plans, ("#2563eb", "#d97706", "#7c3aed", "#059669")):
        rows = [r for r in economics["calibration_grid"] if r["plan"] == plan and r["completed"]]
        if rows:
            ax.plot([r["threshold"] for r in rows], [r["net"] for r in rows], "o-", label=plan, color=color)
    ax.axhline(0, color="grey", lw=.8)
    selected = economics["summary"]["selected"]
    ax.set(xlabel="Marginal side probability threshold", ylabel="Completed calibration closed net, USD / oz",
           title=f"CAL only: selected {selected['plan']}, threshold {selected['threshold']:g}")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    any_trade = False
    notes = []
    for bundle, result, color in ((primary, economics, "#2563eb"), (confirmation, other, "#d97706")):
        test = result["summary"]["test"]
        rows = [r for r in result["ledgers"]["test"] if r["status"] == "COMPLETE"]
        label = f"seed {bundle['config']['seed']}: {test['trades']} trades; net {test['net']:+.2f}"
        if rows:
            any_trade = True
            when = pd.to_datetime([int(r["exit_ns"]) for r in rows], unit="ns", utc=True)
            ax.step(when, [r["closed_cash"] for r in rows], where="post", color=color,
                    label=label + (" (PARTIAL / censored)" if not test["completed"] else ""))
        else:
            notes.append(label + "; WAIT/no completed trades")
    if any_trade:
        ax.set(xlabel="Actual exit time UTC", ylabel="Closed cash, USD / reference ounce", title="Selected TEST replay; intratrade drawdown not shown")
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(fontsize=8)
    else:
        ax.set_axis_off()
        ax.text(.04, .75, "Selected TEST policies made no completed trades.\nNo zero-profit equity curve is drawn.\nWAIT does not establish a trading edge.",
                transform=ax.transAxes, fontsize=13, va="top")
    if notes:
        ax.text(.04, .12, "\n".join(notes), transform=ax.transAxes, fontsize=9, va="bottom")
    _save_figure(fig, out, "06_risk_and_execution.png", "Риск, издержки и отдельное исполнение",
                 "Top panels condition on future-observed outcomes and are oracle diagnostics, not a strategy. Bottom uses CAL-selected one-position policy; CENSORED halts replay.", figures)
    return dict(primary=economics["summary"], confirmation=other["summary"])


def actual_context_history(data):
    """Reproduce frozen FeatureStore reset semantics, without importing Torch.

    Real tokens include the current root. Therefore 61 tokens at 5 s means
    max(61-1,0)*5=300 elapsed seconds, not 305 seconds.
    """
    roots, config = data["roots"], data["primary"]["config"]
    cadence = float(config["data"]["cadence_seconds"])
    supplied = int(config["data"]["sequence_length"])
    t, date = roots.t_ns.to_numpy(), roots.date.to_numpy()
    available, tokens = roots.feature_available.to_numpy(bool), roots.segment_token.to_numpy()
    day_change = np.ones(len(roots), bool)
    day_change[1:] = date[1:] != date[:-1]
    gap = np.zeros(len(roots), bool)
    gap[1:] = np.diff(t) != int(round(cadence * 1e9))
    unavailable = ~available
    unavailable[1:] |= ~available[:-1]
    token_change = np.zeros(len(roots), bool)
    token_change[1:] = tokens[1:] != tokens[:-1]
    reset = day_change | gap | unavailable | token_change
    starts = np.maximum.accumulate(np.where(reset, np.arange(len(roots)), 0))
    real_tokens = np.arange(len(roots)) - starts + 1
    real_tokens[~available] = 0
    # Audit the TCN architecture used in this suite even if another candidate
    # were selected. MLP reads only its current token.
    tcns = [b for b in data["development"] if b["config"]["model"]["architecture"] == "tcn"]
    rfs = {b["summary"]["training"]["architecture_receptive_field"] for b in tcns}
    if len(rfs) != 1:
        raise ValueError("Context report requires one explicit TCN RF across families")
    receptive, = rfs
    effective_limit = min(supplied, receptive)
    effective = np.minimum(real_tokens, effective_limit)
    seconds = np.maximum(effective - 1, 0) * cadence
    known = available & (roots.y_crossing.to_numpy() >= 0)
    def stats(a):
        q = np.quantile(a, [.1, .5, .9])
        return dict(n=len(a), mean=float(np.mean(a)), q10=float(q[0]), q50=float(q[1]), q90=float(q[2]))
    by_split = {}
    role_masks = {"all": known}
    role_masks.update({role: roots.date.isin(dates).to_numpy() & known for role, dates in config["splits"].items()})
    for role, mask in role_masks.items():
        ids = np.flatnonzero(mask)
        by_split[role] = dict(known_feature_roots=len(ids), effective_tokens=stats(effective[ids]),
            detailed_history_seconds=stats(seconds[ids]),
            full_receptive_field_fraction=float((real_tokens[ids] >= receptive).mean()),
            full_supplied_window_fraction=float((real_tokens[ids] >= supplied).mean()))
    # Exclusive precedence categories explain the loader behavior, not the
    # upstream physical cause of a source/quote support-token change.
    exclusive = dict(date_or_first_root=int(day_change.sum()),
        within_day_root_clock_gap=int((~day_change & gap).sum()),
        current_or_previous_X_unavailable=int((~day_change & ~gap & unavailable).sum()),
        support_token_change_with_X_available=int((~day_change & ~gap & ~unavailable & token_change).sum()))
    audit_path = Path("reports/scientific_pipeline_v2/statistical_audit.json")
    upstream = None
    if audit_path.exists():
        audit = read(audit_path)
        data["statistical_audit"] = audit
        expected = audit.get("actual_context_history", {}).get("by_split", {})
        for role, values in expected.items():
            actual = by_split[role]
            for key in ("full_receptive_field_fraction", "full_supplied_window_fraction"):
                if not np.isclose(actual[key], values[key], atol=1e-12, rtol=0):
                    raise ValueError("Context audit/full-window discrepancy")
            for key in ("q10", "q50", "q90"):
                if not np.isclose(actual["detailed_history_seconds"][key], values["detailed_history_seconds"][key], atol=1e-12, rtol=0):
                    raise ValueError("Context audit/elapsed-history discrepancy")
        if expected:
            data["input_sha256"][str(audit_path)] = sha(audit_path)
            upstream = audit.get("actual_context_history")
    receipt = dict(cadence_seconds=cadence, supplied_tokens=supplied, tcn_receptive_field_tokens=receptive,
        theoretical_max_elapsed_seconds=(effective_limit-1)*cadence, by_split=by_split,
        reset_counts_exclusive_all_roots=exclusive,
        formula="min(RF, supplied_window, root_index-segment_start+1) real tokens; (tokens-1)*cadence elapsed seconds",
        reset_definition="date, cadence gap, current/previous X unavailable, or support token change; unknown Y never resets",
        interpretation="Consecutive real sequence history is not total information horizon: current-token rolling features, BB geometry and rich-only London summaries may summarize older observations",
        upstream_statistical_context_audit=upstream)
    data["context_history"] = receipt
    data["context_history_arrays"] = dict(seconds=seconds, real_tokens=real_tokens, known=known)
    return receipt


def plot_context_history(data, out, figures):
    receipt = actual_context_history(data)
    arrays, roots = data["context_history_arrays"], data["roots"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5), layout="constrained")
    fig.suptitle("Actual consecutive sequence history is much shorter than the theoretical TCN maximum", fontsize=16)
    roles = ["train", "validation", "calibration", "test"]
    colors = ["#2563eb", "#7c3aed", "#d97706", "#059669"]
    ax = axes[0, 0]
    for role, color in zip(roles, colors):
        mask = arrays["known"] & roots.date.isin(data["primary"]["config"]["splits"][role]).to_numpy()
        values, counts = np.unique(arrays["seconds"][mask], return_counts=True)
        ax.step(values, np.cumsum(counts)/counts.sum(), where="post", color=color, label=f"{role} (n={counts.sum():,})")
    ax.axvline(receipt["theoretical_max_elapsed_seconds"], color="#475569", ls=":")
    ax.set(xlabel="Elapsed span of real TCN input tokens, seconds", ylabel="Fraction of known X+Y origins at or below span",
           xlim=(-5, 305), ylim=(0, 1.03), title="Empirical distribution; no padding counted as history")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[0, 1]
    medians = np.array([receipt["by_split"][r]["detailed_history_seconds"]["q50"] for r in roles])
    lower = np.array([receipt["by_split"][r]["detailed_history_seconds"]["q10"] for r in roles])
    upper = np.array([receipt["by_split"][r]["detailed_history_seconds"]["q90"] for r in roles])
    for i, color in enumerate(colors):
        ax.errorbar(i, medians[i], yerr=[[medians[i]-lower[i]], [upper[i]-medians[i]]], color=color,
                    fmt="o", capsize=8, ms=9, lw=2)
        ax.annotate(f"median {medians[i]:g}s", (i, medians[i]), xytext=(9, 8), textcoords="offset points", fontsize=9)
    ax.axhline(receipt["theoretical_max_elapsed_seconds"], color="#475569", ls=":", label="Theoretical maximum")
    ax.set(xticks=range(4), xticklabels=roles, ylabel="Seconds", ylim=(-10, 340), title="Median and q10-q90; whiskers are NOT confidence intervals")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    a = np.array([receipt["by_split"][r]["full_receptive_field_fraction"] for r in roles])
    b = np.array([receipt["by_split"][r]["full_supplied_window_fraction"] for r in roles])
    ax.bar(np.arange(4)-.17, a*100, width=.32, color="#2563eb", label=f"Full TCN RF ({receipt['tcn_receptive_field_tokens']} tokens)")
    ax.bar(np.arange(4)+.17, b*100, width=.32, color="#94a3b8", label=f"Full supplied window ({receipt['supplied_tokens']} tokens)")
    for i, v in enumerate(a):
        ax.text(i-.17, v*100+.6, f"{v:.1%}", ha="center", fontsize=9)
    ax.set(xticks=range(4), xticklabels=roles, ylabel="Percent of known X+Y origins", ylim=(0, max(a.max(), b.max())*100+8),
           title="Same causal history availability for every feature family")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    resets = receipt["reset_counts_exclusive_all_roots"]
    labels = ["Date boundary / first root", "Within-day 5s root-clock gap", "Current or previous X unavailable", "Support token changes; X available"]
    counts = np.array(list(resets.values()))
    ax.barh(np.arange(4), counts, color="#64748b")
    for i, n in enumerate(counts):
        ax.text(n+max(counts)*.01, i, f"{n:,}", va="center", fontsize=9)
    ax.set(yticks=range(4), yticklabels=labels, xlabel="Exclusive reset counts over all clock roots",
           xlim=(0, max(counts)*1.18), title="Token reset does NOT automatically mean missing physical quotes")
    ax.invert_yaxis()
    for ax in axes.flat:
        ax.grid(alpha=.12)
    _save_figure(fig, out, "07_context_history.png", "Фактически доступная последовательная история",
                 "61 tokens allow AT MOST 300 s. Current-token rolling statistics can summarize older history; rich-only London/FFT features are absent from the selected micro model.", figures)
    return receipt


def _relative(path, out):
    return html.escape(os.path.relpath(path, out).replace("\\", "/"), quote=True)


def compact_metrics(data):
    """Portable result evidence: all metrics, no source payloads or ledgers."""
    runs = []
    policy_keys = ("plan", "threshold", "completed", "net", "closed_max_drawdown", "wins", "win_rate",
                   "origin_candidates", "causal_feature_origins", "unprocessed_after_censor", "halt",
                   "exit_reasons", "trades", "ineligible", "empty", "censored", "skipped_busy")
    for b in data["bundles"]:
        s, c = b["summary"], b["config"]
        tr = s["training"]
        tkeys = ("best_update", "selected_checkpoint", "best_validation_nll", "updates_completed", "examples_seen",
                 "validation_checks", "seconds", "parameter_count", "device", "architecture_receptive_field",
                 "supplied_tokens", "effective_tokens", "detailed_history_seconds", "selection", "beats_train_prior_on_validation")
        policy = s.get("selected_policy", {})
        compact_policy = {"selected": policy.get("selected"),
                          "calibration": {k: v for k, v in policy.get("calibration", {}).items() if k in policy_keys},
                          "test": {k: v for k, v in policy.get("test", {}).items() if k in policy_keys}}
        runs.append(dict(run_id=b["run_id"], role=b["role"], seed=c["seed"], family=c["data"]["family"],
                         model=c["model"], loss=c["loss"], training_config=c["training"],
                         training={**{k: tr[k] for k in tkeys if k in tr},
                                   "detailed_history_seconds_is_theoretical_max": True}, evaluation=s["evaluation"],
                         cohorts=s["cohorts"], scaler_sha256=s["scaler_sha256"],
                         checkpoint_sha256=sha(b["path"] / "checkpoint.pt"),
                         summary_sha256=sha(b["path"] / "summary.json"), selected_policy=compact_policy))
    manifest, c = data["manifest"], data["primary"]["config"]
    diagnostics = data["diagnostics"]
    return dict(status="COMPLETE_PORTABLE_RESULTS", primary_run_id=data["suite"]["primary_run_id"],
                confirmation_run_id=data["suite"]["confirmation_run_id"], selection=data["suite"]["selection"],
                dataset_manifest_sha256=sha(data["dataset"] / "manifest.json"),
                source_dates=manifest["dates"], roots=len(data["roots"]),
                families={k: dict(feature_names=v, source_value_channels=len(v), model_input_channels=2*len(v)+1)
                          for k, v in manifest["groups"].items()},
                by_day=diagnostics["by_day"] if diagnostics else None,
                by_role=diagnostics["by_role"] if diagnostics else None,
                feature_finite_fraction=diagnostics.get("feature_finite_fraction") if diagnostics else None,
                full_feature_order=manifest["feature_names"],
                event_query=c["event_query"], execution=c["execution"], risk=c["risk"], policy=c["policy"],
                experiment=c["experiment"], actual_context_history=data["context_history"], runs=runs,
                secondary_validation_prior_diagnostic={rid: value.get("validation_prior_posthoc_diagnostic")
                    for rid, value in data.get("statistical_audit", {}).get("runs", {}).items()},
                omissions="No checkpoints, raw feature arrays, large prediction tables or trade ledgers are embedded; SHA values bind those external artifacts.")


def include_crossing_case(data, out, figures):
    """Include the independently rendered, clock-selected nine-line case."""
    path = out / "crossing_case.json"
    if not path.exists():
        return None
    receipt = read(path)
    png = out / "crossing_case.png"
    if receipt.get("status") != "COMPLETE" or sha(png) != receipt["png_sha256"]:
        raise ValueError("Independent crossing-case receipt/PNG mismatch")
    for source, expected in receipt["input_sha256"].items():
        if sha(source) != expected:
            raise ValueError("Independent crossing-case source changed")
        data["input_sha256"][source] = expected
    data["input_sha256"][str(path)] = sha(path)
    figures.append(dict(file=png.name, title="Одна фиксированная точка: девять первых пересечений и отдельные исполнения",
        caption="Anchor is May 11 first origin +1 hour. Late hollow crossings do not enter the 300-second label. MAE/MFE ranges are saved excursions, not invented equity paths.",
        sha256=receipt["png_sha256"]))
    return dict(receipt_file=path.name, receipt_sha256=sha(path), anchor_rule=receipt["anchor_rule"],
                origin_ns=receipt["origin_ns"], state=receipt["state"])


def make_html(data, out, figures, economics):
    primary, confirmation, manifest = data["primary"], data["confirmation"], data["manifest"]
    rows = []
    for b in data["bundles"]:
        summary, training = b["summary"], b["summary"]["training"]
        ev = summary["evaluation"]
        tag = "основной, выбор по VAL" if b is primary else "реплика seed" if b is confirmation else "кандидат"
        rows.append(f"<tr><td>{html.escape(_label(b))}</td><td>{tag}</td><td>{training['best_update']}</td>"
                    f"<td>{ev['validation']['model']['joint_nll']:.4f}</td><td>{ev['test']['model']['joint_nll']:.4f}</td>"
                    f"<td>{ev['test']['prior']['joint_nll']:.4f}</td><td>{ev['test']['model']['event_auc']:.3f}</td>"
                    f"<td><a href='report_metrics.json'>metrics JSON</a></td></tr>")
    split_rows = []
    for role, c in primary["summary"]["cohorts"].items():
        split_rows.append(f"<tr><td>{role}</td><td>{c['dates'][0]} — {c['dates'][-1]}</td><td>{len(c['dates'])}</td>"
                          f"<td>{c['known_feature_available_roots']:,}</td><td>{html.escape(str(c['state_counts']))}</td></tr>")
    family_rows = []
    for family, names in manifest["groups"].items():
        family_rows.append(f"<tr><td>{family}</td><td>{len(names)}</td><td>{2*len(names)+1}</td></tr>")
    context = data["context_history"]
    history_rows = []
    for role in ("train", "validation", "calibration", "test"):
        v = context["by_split"][role]
        h = v["detailed_history_seconds"]
        history_rows.append(f"<tr><td>{role}</td><td>{h['q50']:g} s</td><td>{h['q10']:g} / {h['q90']:g} s</td>"
                            f"<td>{v['full_receptive_field_fraction']:.2%}</td><td>{v['full_supplied_window_fraction']:.2%}</td></tr>")
    primary_family = primary["config"]["data"]["family"]
    section_figures = "\n".join(f"<section><h2>{html.escape(f['title'])}</h2><a href='{f['file']}'><img loading='lazy' src='{f['file']}' alt='{html.escape(f['title'])}'></a><p class='caption'>{html.escape(f['caption'])}</p></section>" for f in figures)
    economic_text = "Экономическая оценка в этом наборе артефактов отсутствует; из классификационных метрик нельзя заключать о доходности."
    if economics:
        chunks = []
        for name, v in economics.items():
            s, t = v["selected"], v["test"]
            state = "полный replay" if t["completed"] else "частичный replay: остановлен на неизвестном исполнении"
            behavior = "WAIT / нет завершённых сделок" if t["trades"] == 0 else f"{t['trades']} сделок, closed net {t['net']:+.2f} USD на условную унцию"
            chunks.append(f"{name}: {s['plan']}, threshold={s['threshold']:g}; {behavior}; {state}.")
        economic_text = " ".join(chunks)
    n = len(data["roots"])
    best = primary["summary"]["evaluation"]["test"]
    replicated = confirmation["summary"]["evaluation"]["test"]
    delta = best["model"]["joint_nll"] - best["prior"]["joint_nll"]
    wait_count = sum(b["summary"].get("selected_policy", {}).get("selected", {}).get("threshold", 0) > 1
                     for b in data["bundles"])
    val_prior = data.get("statistical_audit", {}).get("runs", {}).get(primary["run_id"], {}).get("validation_prior_posthoc_diagnostic")
    prior_context = ""
    if val_prior:
        prior_context = (f"<p>Часть выигрыша относительно TRAIN-prior связана с изменением частоты event. "
                         f"Дополнительный constant prior, оценённый только по VAL, даёт на TEST joint NLL "
                         f"{val_prior['test']['joint_nll']:.4f} вместо {best['prior']['joint_nll']:.4f} у TRAIN-prior; "
                         f"основная модель сохраняет NLL {best['model']['joint_nll']:.4f}. "
                         "Это вторичная post-hoc проверка сдвига частот; исходный benchmark остаётся TRAIN-prior.</p>")
    doc = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scientific opportunity pipeline v2</title><style>
body{{margin:0;background:#f1f5f9;color:#15243a;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1480px;margin:auto;padding:32px 24px 80px}}h1{{font-size:32px;line-height:1.2}}h2{{font-size:24px;margin-top:0}}section,.card{{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:24px;margin:24px 0}}img{{width:100%;height:auto;border:1px solid #e2e8f0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e2e8f0}}th{{background:#f8fafc}}a{{color:#1d4ed8}}.caption,.subtle{{font-size:14px;color:#526175}}.notice{{border-left:5px solid #d97706;padding-left:18px}}code{{background:#e8eef5;padding:2px 5px}}.table-wrap{{overflow:auto}}p{{max-width:1150px}}ul{{max-width:1150px}}
</style><main><h1>Многоблочный научный pipeline: событие пересечения, контекст и градиенты</h1>
<p>{len(manifest['dates'])} торговых дней; {n:,} исходных точек на 5-секундном clock. Полные логические datasets geometry/micro/rich используют общие immutable значения, метки и индексы без копирования окон N×T×F.</p>
<div class="card"><h2>Что зафиксировано</h2><p>Основной run: <b>{html.escape(_label(primary))}</b>, выбран только по validation joint NLL. Реплика: <b>{html.escape(_label(confirmation))}</b>; она не является вторым победителем, выбранным по test. Разница test NLL основного run и собственного train-prior: <b>{delta:+.4f}</b>, отрицательное значение означает улучшение вероятностной метрики.</p>
<p><b>Модель различает наличие crossing-event лучше, чем его направление.</b> TEST event AUC: {best['model']['event_auc']:.4f} и {replicated['model']['event_auc']:.4f} у двух seed. Conditional direction NLL: {best['model']['direction_nll_given_event']:.4f} / {replicated['model']['direction_nll_given_event']:.4f} против {best['prior']['direction_nll_given_event']:.4f} у prior — значительно более слабое улучшение. Это поддерживает отдельную оценку event и direction; прибыльность из этих чисел не следует.</p>
<p><b>CAL выбрал WAIT в {wait_count} из {len(data['bundles'])} прогонов.</b> В заданной сетке риска и порогов не найдено положительного полного CAL-replay с минимальным числом сделок. Основной micro-вариант не получает 29 London-context и 126 FFT-признаков; добавление rich-пакета не дало согласованного улучшения VAL между архитектурами. Главный обнаруженный bottleneck последовательности — короткая реальная история после причинных сбросов: TEST-медиана {context['by_split']['test']['detailed_history_seconds']['q50']:g} секунд при максимуме {context['theoretical_max_elapsed_seconds']:g}.</p>
{prior_context}
<p class="notice">Наличие признаков, обучаемость и экономическая пригодность — отдельные утверждения. Более плавный pE не означает более предсказуемую возможность. Исторические даты новые относительно февральского scientific-v1, но часть могла встречаться в прежних экспериментах репозитория.</p>
<p><a href="report_metrics.json">Все результаты в переносимом JSON</a> · <a href="suite_summary.json">Suite и зафиксированный выбор</a> · <a href="model_selection.json">Selection receipt</a> · <a href="methodology.json">Методология и гиперпараметры</a> · <a href="dataset_diagnostics.json">Покрытие признаков</a> · <a href="visual_receipt.json">SHA визуального отчёта</a> · <a href="sol61_visual_audit.json">Визуальный аудит Sol 6.1</a></p></div>
<section><h2>Общая задача и границы наблюдаемости</h2><p>Primary target сохраняет BB M1/M3/M6 и первые пересечения всех девяти отдельных линий — lower/middle/upper каждого масштаба: фиксированное достаточно большое движение за 300 секунд. NONE/LONG_ONLY/SHORT_ONLY/BOTH описывают будущую карту. Event-head обучается BCE, направление — conditional CE только при известном положительном event. UNKNOWN не становится NONE и не участвует в supervised loss.</p>
<p>R_event и G_event заморожены для всех риск-планов. Изменение стоп-бюджета R_trade не меняет event-метку. Исполнение использует отдельный причинный replay, задержки входа/выхода, комиссии и поддержку источника. Перебор риска и порога допускается на CAL; TEST не выбирает параметры.</p>
<p><code>R_event=max(2×(spread+0.30), 0.10×BB1_half, 0.50)</code>; <code>G_event=2×R_event</code>. В риск-сетке коэффициент стоимости у R_trade равен 2/4/6, а G_event остаётся тем же.</p>
<p>TCN RF={context['tcn_receptive_field_tokens']} при пятисекундных отсчётах допускает максимум {context['theoretical_max_elapsed_seconds']:g} секунд между первым и текущим реальным токеном. Это верхняя граница, а не фактически доступная история каждого примера. В известной TEST-выборке медиана составляет <b>{context['by_split']['test']['detailed_history_seconds']['q50']:g} секунд</b>, полные {context['tcn_receptive_field_tokens']} токенов доступны лишь в <b>{context['by_split']['test']['full_receptive_field_fraction']:.2%}</b> точек. Padding не считается наблюдаемой историей. Это свойство причинных сегментов и общего loader, одинаковое для всех семейств признаков.</p>
<p>Основная модель использует <b>{html.escape(primary_family)}</b>. Только семейство rich добавляет 29 признаков текущего и предыдущего London-контекста и 126 спектральных признаков; выбранная micro-модель их не получает. Даже у rich это агрегаты и M1-спектры, а не полный двухдневный микроструктурный путь. Короткий span последовательных токенов не означает столь же короткий суммарный информационный горизонт: текущий токен содержит rolling-статистики micro до 128 секунд и BB-геометрию. Контекстный VWAP остаётся midpoint/volume proxy, а не точным VWAP цен всех T-сделок.</p>
<p>Историю сбрасывают смена даты, разрыв пятисекундного clock, недоступность текущего или предыдущего X и смена support token; неизвестный Y её не сбрасывает. Смена support token сама по себе не доказывает пропуск физических котировок. Независимый <a href="statistical_audit.json">аудит источника и причин сброса</a> отделяет эти механизмы. Увеличение nominal sequence length без исправления обоснованных ограничений поддержки не добавит модели реальной истории.</p>
<div class="table-wrap"><table><tr><th>Split</th><th>Медиана реальной истории</th><th>q10 / q90</th><th>Полный RF, {context['tcn_receptive_field_tokens']} токенов</th><th>Полное окно, {context['supplied_tokens']} токена</th></tr>{''.join(history_rows)}</table></div>
<div class="table-wrap"><table><tr><th>Split</th><th>Диапазон</th><th>Дней</th><th>Известные Y + X</th><th>NONE / L / S / BOTH</th></tr>{''.join(split_rows)}</table></div></section>
<section><h2>Сменяемые признаки и модели</h2><table><tr><th>Семейство</th><th>Значения</th><th>Вход с масками</th></tr>{''.join(family_rows)}</table>
<p>Все сравнения прошли проверку одинаковых cohort hashes и scaler SHA. Scaler использует только допустимые TRAIN-строки независимо от известности их Y. Для temporal windows важны пропуски и support token; неизвестный будущий исход сам по себе не обрывает историю. Начальный checkpoint update=0 явно обозначается как необученная сеть.</p>
<p>Бюджет ограничен 1500 optimizer updates с validation каждые 150 и patience=5 проверок. Это ограниченный baseline, а не доказательство сходимости всех вариантов. Hidden width одинаков, но число параметров меняется с входной размерностью. MLP видит текущий токен, TCN — последовательность: сравнение одновременно меняет архитектуру и доступную подробную историю.</p>
<div class="table-wrap"><table><tr><th>Run</th><th>Роль</th><th>Update</th><th>VAL NLL</th><th>TEST NLL</th><th>Prior NLL</th><th>TEST event AUC</th><th>Артефакт</th></tr>{''.join(rows)}</table></div></section>
<section><h2>Как читать градиенты</h2><ul><li>Изменение бинарного event во времени: разность E(t)−E(t−5s). Для категорий нельзя использовать расстояние между кодами 0/1/2/3.</li><li>Производная loss по логиту event для BCE: pE−E. Она выражает ошибку текущего прогноза и не является скоростью изменения цены или таргета.</li><li>Encoder gradient измеряет воздействие loss на общие параметры. Cosine event/direction на фиксированном TRAIN-probe является диагностикой взаимодействия, а не доказательством необходимости двух сетей.</li><li>pL и pS получены из согласованного совместного распределения; наличие BOTH не позволяет положить pS=1−pL. Это вероятности соответствующих событий пересечения, а не выигрыша исполнения.</li></ul><p>Easy/hard-разрезы показывают отдельно количество примеров и их вклад в абсолютную массу градиента. Группы определены по <b>совместной</b> вероятности истинного класса: hard &lt;0.4, intermediate 0.4–0.8, easy ≥0.8. У положительного event она содержит множитель pE, поэтому попадание почти всех положительных примеров в hard само по себе не доказывает отдельную сложность направления. Редкий большой убыток не объявляется ошибочной меткой только потому, что он труден для сети.</p></section>
{section_figures}
<section><h2>Экономическая граница результата</h2><p>{html.escape(economic_text)}</p><p>Paired risk diagnostics ограничены общей выборкой с известными будущими исходами и служат измерению сохранения возможности. Это oracle-диагностика, не кривая капитала. В последовательном replay неизвестность не отфильтровывается задним числом: CENSORED останавливает оценку. Closed-cash drawdown не содержит внутрисделочную переоценку, а отсутствие сделок не доказывает положительного edge.</p></section>
<p class="notice">Хвосты высоких вероятностей с единичными наблюдениями или несколькими десятками коррелированных roots нельзя считать надёжной калибровкой торгового порога. На reliability-графике числа рядом с bins — объём выборки, а не число независимых событий; bins с n=1, n=3 или n=23 не подтверждают торговый edge.</p>
<p><a href="shadow_parity.json">Независимая сверка target/replay</a> · <a href="statistical_audit.json">Статистический аудит и зависимость примеров</a> · <a href="crossing_case.json">Числа и ID фиксированного crossing-case</a></p>
<p class="subtle">Основной baseline на графиках — TRAIN-prior. Если статистический аудит показывает дополнительный VAL-only prior, это вторичная post-hoc диагностика изменения частоты event, а не замена исходному benchmark или подбор по TEST.</p>
<p class="subtle">Отчёт построен только по COMPLETE receipts и проверенным SHA. Два временных участка заданы до просмотра результата: May 11 London open +0…2h и May 18 +5…7h. Никакие модели этим скриптом не обучаются.</p></main></html>"""
    (out / "index.html").write_text(doc, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--suite", default="reports/scientific_pipeline_v2/suite_summary.json")
    parser.add_argument("--out", default="reports/scientific_pipeline_v2")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_inputs(Path(args.suite), out)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False})
    figures = []
    plot_coverage(data, out, figures)
    plot_learning(data, out, figures)
    plot_test_comparison(data, out, figures)
    cases = [plot_fixed_case(data, out, date, hours, filename, figures) for date, hours, filename in FIXED_CASES]
    plot_temporal(data, out, figures)
    plot_gradients(data, out, figures)
    economics = plot_economics(data, out, figures)
    context_history = plot_context_history(data, out, figures)
    crossing_case = include_crossing_case(data, out, figures)
    save_json(out / "report_metrics.json", compact_metrics(data))
    make_html(data, out, figures, economics)
    save_json(out / "visual_receipt.json", dict(status="COMPLETE_VISUAL_REPORT", script_sha256=sha(__file__),
        input_sha256=data["input_sha256"], figures=figures, fixed_cases=cases, html_sha256=sha(out / "index.html"),
        report_metrics_sha256=sha(out / "report_metrics.json"),
        primary_run_id=data["suite"]["primary_run_id"], confirmation_run_id=data["suite"]["confirmation_run_id"],
        common_cohort_and_scaler_verified=True, shared_scaler_sha256=data["shared_scaler_sha256"],
        temporal_and_loss_gradient_definitions_separate=True, targets_smoothed=False,
        economic_figure_present=economics is not None, independent_crossing_case=crossing_case,
        actual_context_history=context_history))
    print(json.dumps(dict(status="COMPLETE", figures=len(figures), out=str(out))))


if __name__ == "__main__":
    main()
