"""Frozen OPPORTUNITY_EPISODES_V1 experiment. See the committed preregistration.

Run build, then run. Heavy arrays remain local; compact receipts enter Git.
"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import json
import pickle
import subprocess
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from spectrum.cognition.signatures import DEFAULT_DATA
from spectrum.cognition.execution import QuoteTape
from . import episodes as ep

DATA = Path("data/derived/opportunity_episodes_v1")
REPORT = Path("reports/opportunity_episodes_v1")
CONTRACT = Path("contracts/opportunity_episodes_v1.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


def identity():
    files = [CONTRACT, Path(ep.__file__), Path(__file__)]
    return {str(p): sha(p) for p in files}


def source_columns(path):
    cols = pq.ParquetFile(path).schema_arrow.names
    return [c for c in cols if c in ("date", "t_ns") or c.startswith(("sigma", "spread", "minutes", "ret", "flow_"))] + [
        f"{mode}_{side}_f24_12_{field}" for mode in ("P", "T") for side in ("L", "S") for field in ("st", "net", "busy_s")]


def verify_source_code(code):
    """Accept the historical extraction of entry_points only with AST equality.

    The cached data predates this refactor. Never replace its source receipt with
    the current hash or treat an arbitrary changed execution as compatible.
    """
    checks = {}
    for filename, expected in code.items():
        path = Path(ep.__file__).with_name(filename); current = sha(path)
        if current == expected:
            checks[filename] = dict(source=expected, current=current, exact=True); continue
        if filename != "scalper.py":
            raise ValueError("Source execution code identity drift: " + filename)
        old = subprocess.check_output(["git", "show", "a68c7aa:spectrum/cognition/scalper.py"])
        if hashlib.sha256(old).hexdigest() != expected:
            raise ValueError("Historical source hash does not match receipt")
        old_ast = ast.parse(old); current_ast = ast.parse(path.read_text(encoding="utf-8"))
        entry = next(n for n in current_ast.body if isinstance(n, ast.FunctionDef) and n.name == "entry_points")
        labels = next(n for n in current_ast.body if isinstance(n, ast.FunctionDef) and n.name == "labels")
        call = next(i for i, n in enumerate(labels.body) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name) and n.value.func.id == "entry_points")
        labels.body[call:call + 1] = copy.deepcopy(entry.body[3:-1])
        current_ast.body.remove(entry)
        if ast.dump(old_ast, include_attributes=False) != ast.dump(current_ast, include_attributes=False):
            raise ValueError("Execution refactor not semantically identical after inlining")
        checks[filename] = dict(source=expected, current=current, exact=False, historical_commit="a68c7aa",
                                proof="entire module AST identical after inlining extracted entry_points")
    save(REPORT / "source_code_compatibility.json", checks)


def build():
    c = ep.contract(); DATA.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    ids = identity(); source = Path(c["source"])
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "COMPLETE":
        raise ValueError("Source not COMPLETE")
    code = json.loads((source / "protocol.json").read_text())["code"]
    verify_source_code(code)
    months = c["train_months"] + c["development_months"] + c["evaluation_months"]
    dates = [d for d in manifest["days"] if d[:7] in months]
    receipts = []; excluded = []
    for date in dates:
        started = time.perf_counter(); src = source / f"{date}.parquet"
        receipt = json.loads(src.with_suffix(".json").read_text())
        actual = sha(src)
        if actual != receipt["sha256"]:
            raise ValueError("Source hash mismatch: " + date)
        dest = DATA / f"{date}.parquet"; rp = dest.with_suffix(".json")
        if rp.exists():
            r = json.loads(rp.read_text())
            if r.get("excluded"):
                if r["identity"] != ids or r["source_sha256"] != actual:
                    raise ValueError("Excluded-day identity drift")
                excluded.append(r); continue
            if r["identity"] != ids or r["source_sha256"] != actual or sha(dest) != r["sha256"]:
                raise ValueError("Cached episode day differs: " + date)
            receipts.append(r); continue
        d = pd.read_parquet(src, columns=source_columns(src))
        if d.empty:
            r = dict(date=date, excluded="NO_SOURCE_DECISIONS", identity=ids, source_sha256=actual, rows=0)
            save(rp, r); excluded.append(r)
            print(json.dumps(dict(date=date, excluded=r["excluded"])), flush=True)
            continue
        md = MonthDay(Path(DEFAULT_DATA) / date[:7] / date); q = QuoteTape.from_day(md)
        ix = np.searchsorted(q.quote_t, d.t_ns.to_numpy(np.int64), side="right") - 1
        if (ix < 0).any():
            raise ValueError("Origin before quote support")
        segment = q.epoch[ix]
        x, meta = ep.features(d, segment)
        t = d.t_ns.to_numpy(np.int64)
        allowed = ((t // ep.NS) % c["decision_stride_seconds"] == 0) & (d.sigma_known.to_numpy() > 0)
        allowed &= (d.sigma600.to_numpy() >= c["minimum_sigma600"]) & (d.spread_ticks.to_numpy() > 0)
        allowed &= d.spread_ticks.to_numpy() <= c["max_spread_ticks"]
        allowed &= t + 335 * ep.NS < q.deadline_ns
        allowed &= q.valid[ix] & (segment >= 0)
        k = np.flatnonzero(allowed); frames = []
        for mode in ("P", "T"):
            for side, sn in ((1, "L"), (-1, "S")):
                key = f"{mode}_{sn}_f24_12"
                f = ep.orient(x.iloc[k], side).reset_index(drop=True)
                f["date"] = date; f["t_ns"] = t[k]; f["mode"] = mode
                f["episode_id"] = meta["episode_id"][k]; f["episode_start"] = meta["episode_start"][k]
                f["status"] = d[key + "_st"].to_numpy()[k]
                f["net"] = d[key + "_net"].to_numpy()[k]; f["busy_s"] = d[key + "_busy_s"].to_numpy()[k]
                f["category"] = ep.categories(f.status, f.net)
                frames.append(f)
        out = pd.concat(frames, ignore_index=True)
        for col in ep.BASE + ep.EPISODE + ep.RESPONSE:
            out[col] = out[col].astype(np.float32)
        out.to_parquet(dest, index=False)
        r = dict(date=date, rows=len(out), roots=len(k), episodes=int(meta["episode_start"].sum()),
                 active_roots=int((meta["episode_id"][k] > 0).sum()), source_sha256=actual,
                 source_receipt_sha256=sha(src.with_suffix(".json")), day_manifest_sha256=sha(Path(DEFAULT_DATA) / date[:7] / date / "manifest.json"),
                 identity=ids, sha256=sha(dest), seconds=round(time.perf_counter() - started, 3))
        save(rp, r); receipts.append(r); QUOTE_CACHE.clear()
        print(json.dumps({k: r[k] for k in ("date", "roots", "episodes", "active_roots", "seconds")}), flush=True)
    save(DATA / "manifest.json", dict(status="COMPLETE", identity=ids, source_manifest_sha256=sha(source / "manifest.json"), days=receipts, excluded=excluded))
    save(REPORT / "data_receipt.json", dict(status="COMPLETE", days=len(receipts), roots=sum(r["roots"] for r in receipts),
                                           rows=sum(r["rows"] for r in receipts), source_hashes_checked=len(receipts) + len(excluded), excluded=excluded, identity=ids,
                                           source_manifest_sha256=sha(source / "manifest.json")))


def fit_model(train, family, seed):
    c = ep.contract(); known = train.category >= 0; f = train.loc[known]
    cols = ep.FAMILIES[family]; y = f.category.to_numpy(int)
    if family == "shuffled":
        y = np.random.default_rng(seed).permutation(y)
    weights = ep.episode_weights(f) if family != "response_unweighted" else np.ones(len(f))
    params = dict(c["model"]); params.pop("class")
    m = HistGradientBoostingClassifier(random_state=seed, **params)
    m.fit(f[cols].to_numpy(np.float32), y, sample_weight=weights)
    values = np.asarray([2.1, -1.5, -.30, 0.])
    for k in range(4):
        a = f.loc[f.category == k, "net"]
        if len(a):
            values[k] = a.mean()
    info = dict(rows=len(f), columns=cols, weight_sum=float(weights.sum()),
                active_fraction=float((f.episode_id > 0).mean()),
                active_weight_fraction=float(weights[f.episode_id.to_numpy() > 0].sum() / weights.sum()), class_values=values,
                class_counts={ep.CLASSES[k]: int((y == k).sum()) for k in range(4)})
    return m, values, info


def predictions(model, frame, family, values):
    p = np.zeros((len(frame), 4)); raw = model.predict_proba(frame[ep.FAMILIES[family]].to_numpy(np.float32))
    p[:, model.classes_.astype(int)] = raw
    out = frame.copy()
    for k, name in enumerate(ep.CLASSES):
        out["p_" + name] = p[:, k]
    out["expected_net"] = p @ values
    return out


def metrics(frame):
    f = frame.loc[frame.category >= 0]; y = f.category.to_numpy(int)
    if not len(f):
        return dict(n=0)
    p = f[["p_" + x for x in ep.CLASSES]].to_numpy(float)
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))
    brier = np.sum((p - np.eye(4)[y]) ** 2, 1)
    daily = pd.DataFrame(dict(date=f.date.to_numpy(), loss=loss, brier=brier)).groupby("date").agg(["sum", "count"])
    calibration = []
    bins = np.minimum((p[:, 0] * 10).astype(int), 9)
    for k in range(10):
        sel = bins == k
        if sel.any():
            calibration.append(dict(bin=k, n=int(sel.sum()), predicted=float(p[sel, 0].mean()), observed=float((y[sel] == 0).mean())))
    return dict(n=len(f), log_loss=float(loss.mean()), brier=float(brier.mean()),
                auc_take=float(roc_auc_score(y == 0, p[:, 0])) if len(np.unique(y == 0)) == 2 else None,
                class_rates={name: float((y == k).mean()) for k, name in enumerate(ep.CLASSES)},
                daily={d: dict(loss_sum=float(r[("loss", "sum")]), n=int(r[("loss", "count")]), brier_sum=float(r[("brier", "sum")])) for d, r in daily.iterrows()},
                calibration=calibration)


def policy_stats(frame, threshold=0.):
    tr, censor = ep.sequential(frame, threshold=threshold); days = sorted(frame.date.unique())
    # No censor means all calendar days, including WAIT days, are observed.
    known_days = [d for d in days if censor is None or d < censor["date"]]
    daily = tr.groupby("date").net.sum().reindex(known_days, fill_value=0.)
    ci = ep.block_interval(daily.to_numpy()) if censor is None else [None, None]
    # At identical roots only; not a hypothetical continuous account.
    lookup = frame.set_index(["t_ns", "side"])
    paired = []
    for r in tr.itertuples():
        opposite = lookup.loc[(r.t_ns, -r.side)]
        if int(opposite.category) >= 0 and np.isfinite(opposite.net):
            paired.append(dict(date=r.date, selected=float(r.net), random_side=.5 * (float(r.net) + float(opposite.net))))
    pair = pd.DataFrame(paired)
    pdaily = (pair.groupby("date")[["selected", "random_side"]].sum().reindex(days, fill_value=0.) if len(pair) else pd.DataFrame())
    pair_diff = (pdaily.selected - pdaily.random_side).to_numpy() if len(pdaily) else np.zeros(len(days))
    return dict(attempts=len(tr), fills=int((tr.status == ep.COMPLETE).sum()), empty=int((tr.status == ep.EMPTY).sum()),
                net=float(tr.net.sum()) if censor is None else None, observed_prefix_net=float(tr.net.sum()),
                net_per_day=float(daily.mean()) if censor is None and len(daily) else None, daily_ci=ci, censor=censor,
                occupied_account_hours=float(tr.busy_s.sum() / 3600),
                daily={d: float(v) for d, v in daily.items()}, threshold=float(threshold),
                paired_random_side=dict(roots=len(pair), difference=float(pair_diff.sum()),
                                        daily_ci=ep.block_interval(pair_diff) if censor is None else [None, None],
                                        observed_prefix_only=censor is not None,
                                        semantics="matched selected roots only, not another account curve")), tr


def synthetic_control(seed):
    rng = np.random.default_rng(seed); x = rng.normal(size=(50000, 6)).astype(np.float32)
    p = np.where(x[:, 0] > 0, .65, .25); y = (rng.random(len(x)) < p).astype(int)
    params = dict(ep.contract()["model"]); params.pop("class")
    m = HistGradientBoostingClassifier(random_state=seed, **params).fit(x[:35000], y[:35000])
    pred = m.predict_proba(x[35000:])[:, 1]
    return dict(n_test=15000, p_positive=.65, p_negative=.25, auc=float(roc_auc_score(y[35000:], pred)),
                predicted_positive=float(pred[x[35000:, 0] > 0].mean()), predicted_negative=float(pred[x[35000:, 0] <= 0].mean()),
                scope="fit/predict probability control only; no replay or economic claim")


def run():
    c = ep.contract(); ids = identity(); man = json.loads((DATA / "manifest.json").read_text())
    if man["status"] != "COMPLETE" or man["identity"] != ids:
        raise ValueError("Dataset not complete for this code")
    frames = []
    for r in man["days"]:
        p = DATA / (r["date"] + ".parquet")
        if sha(p) != r["sha256"]:
            raise ValueError("Dataset hash drift")
        frames.append(pd.read_parquet(p))
    all_rows = pd.concat(frames, ignore_index=True); all_rows["month"] = all_rows.date.str[:7]
    summary = dict(version=c["version"], contract_sha256=sha(CONTRACT), identity=ids,
                   evaluation_status=c["evaluation_status"], stages={}, admissions={}, coverage={}, comparisons={},
                   synthetic_control=synthetic_control(c["seed"]))
    thresholds = {}; admissions = {}
    families = list(ep.FAMILIES)
    for stage, train_months, test_months in (("A2", c["train_months"], c["development_months"]),
                                            ("B", c["train_months"] + c["development_months"], c["evaluation_months"])):
        summary["stages"][stage] = {}
        for mode in ("P", "T"):
            train = all_rows.loc[all_rows.month.isin(train_months) & (all_rows["mode"] == mode)].reset_index(drop=True)
            test = all_rows.loc[all_rows.month.isin(test_months) & (all_rows["mode"] == mode)].reset_index(drop=True)
            summary["stages"][stage][mode] = {}; predicted = {}
            for family in families:
                started = time.perf_counter(); model, values, fit_info = fit_model(train, family, c["seed"])
                f = predictions(model, test, family, values); predicted[family] = f
                pol, trades = policy_stats(f)
                res = dict(fit=fit_info, all=metrics(f), episodes=metrics(f.loc[f.episode_id > 0]), policy=pol,
                           months={month: metrics(f.loc[f.month == month]) for month in test_months},
                           unknown_counts={str(k): int(v) for k, v in f.loc[f.category < 0, "status"].value_counts().items()})
                key = mode + "/" + family
                if stage == "A2":
                    admissions[key] = bool(pol["censor"] is None and pol["fills"] >= 30 and pol["daily_ci"][0] > 0)
                    roots = f.groupby("t_ns").expected_net.max()
                    thresholds[key] = {str(cov): float(roots.quantile(1 - cov)) for cov in c["diagnostic_coverages"]}
                res["operational_admitted"] = admissions[key]
                res["operational_action"] = "CANDIDATE" if admissions[key] else "WAIT"
                res["coverage"] = {}
                for cov, threshold in thresholds[key].items():
                    roots = f.groupby("t_ns").expected_net.max()
                    ps, _ = policy_stats(f, threshold=threshold)
                    ps["actual_score_coverage"] = float((roots > threshold).mean()); res["coverage"][cov] = ps
                summary["stages"][stage][mode][family] = res
                f.to_parquet(DATA / f"predictions_{stage}_{mode}_{family}.parquet", index=False)
                trades.to_parquet(DATA / f"trades_{stage}_{mode}_{family}.parquet", index=False)
                with (DATA / f"model_{stage}_{mode}_{family}.pkl").open("wb") as dest:
                    pickle.dump(dict(model=model, values=values, columns=ep.FAMILIES[family]), dest)
                print(json.dumps(dict(stage=stage, mode=mode, family=family, log_loss=res["all"].get("log_loss"),
                                      net=pol["net"], attempts=pol["attempts"], admitted=admissions[key],
                                      seconds=round(time.perf_counter() - started, 2))), flush=True)
            for family in families[1:]:
                a, b = predicted["baseline"], predicted[family]
                ok = a.category.to_numpy() >= 0; y = a.category.to_numpy(int)[ok]
                pa = a[["p_" + k for k in ep.CLASSES]].to_numpy()[ok]
                pb = b[["p_" + k for k in ep.CLASSES]].to_numpy()[ok]
                gain = np.log(np.clip(pb[np.arange(len(y)), y], 1e-9, 1)) - np.log(np.clip(pa[np.arange(len(y)), y], 1e-9, 1))
                g = pd.DataFrame(dict(date=a.loc[ok, "date"].to_numpy(), gain=gain)).groupby("date").gain.sum()
                summary["comparisons"][f"{stage}/{mode}/{family}"] = dict(mean_gain_per_attempt=float(gain.mean()),
                    daily_gain=float(g.mean()), daily_ci=ep.block_interval(g.to_numpy()), days=len(g),
                    daily={d: float(v) for d, v in g.items()})
            save(REPORT / "summary.running.json", summary)
    summary["admissions"] = admissions; summary["coverage"] = thresholds
    summary["source_rows"] = len(all_rows); summary["days"] = len(man["days"])
    summary["git_head_at_run"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save(REPORT / "summary.json", summary)
    save(REPORT / "completion_receipt.json", dict(status="COMPLETE", summary_sha256=sha(REPORT / "summary.json"),
        contract_sha256=sha(CONTRACT), identity=ids, completed_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        datasets_sha256=sha(DATA / "manifest.json"), prediction_files={p.name: sha(p) for p in DATA.glob("predictions_*.parquet")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("build", "run")); a = p.parse_args()
    (build if a.command == "build" else run)()
