"""Independent exit scan and fresh MBO entry proofs for fixed calendar probes.

No fitting or parameter selection. Writes exact entry/exit facts for visual QA.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pandas as pd
from spectrum.cognition import episodes as ep
from spectrum.cognition.episodes_lab import DATA, REPORT, save, sha
from spectrum.cognition import scalper as sc
from spectrum.cognition.passive import day_tapes, captures_by_date, Quotes, LATENCY, TICK
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition.signatures import DEFAULT_DATA
from spectrum.playground.data import MonthDay, QUOTE_CACHE


def direct_exit(q, start, entry_price, side, intent, mode):
    """Simple Python quote loop, independent of scan_exits and label category logic."""
    if start < 0:
        return dict(status=ep.EMPTY, category=3, net=0., exit_ns=int(intent + (30.1 if mode == "P" else 2.1) * ep.NS))
    fill = int(q.t[start]); epoch = q.epoch[start]
    horizon = (fill if mode == "P" else intent) + 305 * ep.NS
    for j in range(start, len(q.t)):
        now = int(q.t[j])
        if now >= horizon or not q.ok[j] or q.epoch[j] != epoch:
            return dict(status=ep.CENSORED, category=-1, net=None, exit_ns=now)
        price = int(q.bid[j] if side == 1 else q.ask[j]); gross_ticks = side * (price - entry_price) / TICK
        if now >= q.deadline:
            category = 2
        elif gross_ticks <= -12:
            category = 1
        elif gross_ticks >= 24:
            category = 0
        elif now - fill >= 300 * ep.NS:
            category = 2
        else:
            continue
        return dict(status=ep.COMPLETE, category=category, net=side * (price - entry_price) / 1e9 - .30,
                    exit_ns=now, exit_px=price, exit_ordinal=j)
    return dict(status=ep.CENSORED, category=-1, net=None, exit_ns=int(q.t[-1]))


def scene_candidates(days):
    """Declared deterministic calendar + first category cases, never maximum PnL."""
    evaluation = [d for d in days if d[:7] in ep.contract()["evaluation_months"]]
    scenes = []; seen = set(); categories = set()
    for date in evaluation:
        f = pd.read_parquet(DATA / f"{date}.parquet")
        f = f.loc[f["mode"] == "P"].sort_values(["t_ns", "side"], ascending=[True, False])
        if f.empty:
            continue
        if len([s for s in scenes if s["selection"].startswith("calendar")]) < 2:
            month = date[:7]
            if not any(s["selection"] == "calendar_" + month for s in scenes):
                near = f.loc[f.minutes_since_london >= 60]
                row = (near if len(near) else f).iloc[0]
                key = (date, int(row.t_ns), int(row.side)); seen.add(key)
                scenes.append(dict(date=date, t_ns=str(int(row.t_ns)), side=int(row.side), mode="P", selection="calendar_" + month))
        for cat in (0, 1, 2, 3, -1):
            if cat in categories:
                continue
            g = f.loc[(f.category == cat) & (f.episode_id > 0)]
            if len(g):
                row = g.iloc[0]; key = (date, int(row.t_ns), int(row.side)); categories.add(cat)
                if key not in seen:
                    seen.add(key)
                    scenes.append(dict(date=date, t_ns=str(int(row.t_ns)), side=int(row.side), mode="P",
                                       selection="first_episode_" + (ep.CLASSES[cat] if cat >= 0 else "CENSORED")))
    return scenes, dict(missing_categories=[ep.CLASSES[k] if k >= 0 else "CENSORED" for k in (0, 1, 2, 3, -1) if k not in categories])


def main():
    man = json.loads((DATA / "manifest.json").read_text()); days = [r["date"] for r in man["days"] if r["roots"] > 0]
    scenes, scene_info = scene_candidates(days)
    calendar = [next(d for d in days if d[:7] == m) for m in ep.contract()["train_months"] + ep.contract()["development_months"] + ep.contract()["evaluation_months"]]
    dates = sorted(set(calendar + [s["date"] for s in scenes])); by = captures_by_date(months=tuple(sorted({d[:7] for d in dates})))
    groups = defaultdict(list)
    for d in dates:
        groups[by[d][0]["member_key"]].append(d)
    facts = []; checks = []; source_hashes = {}
    for member, ds in sorted(groups.items()):
        print(json.dumps(dict(member=member, dates=ds, phase="decode_MBO")), flush=True)
        for date, tape, captures in day_tapes(member, ds):
            md = MonthDay(Path(DEFAULT_DATA) / date[:7] / date); qtape = QuoteTape.from_day(md); q = Quotes(qtape)
            for part in md.meta["quote_parts"]:
                fp = Path(part["root"]) / "quotes.parquet"
                actual = sha(fp)
                if actual != part["quote_file"]["sha256"]:
                    raise ValueError("Canonical quote hash drift")
                source_hashes[str(fp)] = actual
            df = pd.read_parquet(DATA / f"{date}.parquet")
            roots = np.sort(df.t_ns.unique()); pick = roots[np.linspace(0, len(roots) - 1, min(40, len(roots))).astype(int)] if date in calendar else np.array([], np.int64)
            intent = np.unique(np.r_[pick, [int(s["t_ns"]) for s in scenes if s["date"] == date]]).astype(np.int64)
            # Queue entries are replayed from raw MBO using the audited V5 kernel,
            # with fill proofs enforced by entry_points; no midpoint touch proxy.
            entries = sc.entry_points(q, tape, intent, np.ones(len(intent), bool))
            lookup = df.set_index(["t_ns", "mode", "side"])
            for i, t in enumerate(intent):
                for mode in ("P", "T"):
                    for side in (1, -1):
                        j, px, st0, ref = [a[i] for a in entries[(mode, side)]]
                        if mode == "T":
                            # Entry BBO is checked without invoking taker_entries.
                            jj = np.flatnonzero((q.t > t + LATENCY) & (q.t <= t + 2 * ep.NS) & q.ok)
                            independent_j = int(jj[0]) if len(jj) else -1
                            if independent_j != j:
                                raise AssertionError("Taker entry index mismatch")
                        if st0 == ep.CENSORED:
                            result = dict(status=ep.CENSORED, category=-1, net=None, exit_ns=None)
                        else:
                            result = direct_exit(q, int(j), int(px), side, int(t), mode)
                        row = lookup.loc[(t, mode, side)]
                        bad = int(row.status) != result["status"] or int(row.category) != result["category"]
                        if result["net"] is not None:
                            bad |= abs(float(row.net) - result["net"]) > 2e-5
                            busy = (result["exit_ns"] - int(t)) / ep.NS
                            bad |= abs(float(row.busy_s) - busy) > 5e-5
                        if bad:
                            raise AssertionError(f"Saved outcome mismatch {date} {t} {mode} {side}: {result} vs {row[['status','category','net','busy_s']].to_dict()}")
                        fact = dict(date=date, t_ns=str(int(t)), mode=mode, side=side, status=result["status"], category=result["category"],
                                    net=result["net"], submit_ns=str(int(t) + LATENCY),
                                    entry_ns=str(int(q.t[j])) if j >= 0 else None, entry_px_nano=str(int(px)) if j >= 0 else None,
                                    exit_ns=str(result["exit_ns"]) if result["exit_ns"] is not None else None,
                                    exit_px_nano=str(result["exit_px"]) if "exit_px" in result else None,
                                    entry_ordinal=int(j) if j >= 0 else None, exit_ordinal=result.get("exit_ordinal"),
                                    quote_epoch=int(q.epoch[j]) if j >= 0 else None)
                        facts.append(fact)
            checks.append(dict(date=date, roots=len(intent), compared=len(intent) * 4, mismatches=0, captures=captures))
            print(json.dumps(checks[-1]), flush=True); QUOTE_CACHE.clear()
    save(DATA / "audit_execution_facts.json", facts)
    save(REPORT / "scene_selection.json", dict(scenes=scenes, **scene_info,
        semantics="calendar and first class-conditioned diagnostic candidates; not a representative random sample or selected policy trades"))
    save(REPORT / "execution_audit.json", dict(status="PASS", checks=checks, compared=sum(r["compared"] for r in checks),
        quote_hashes=source_hashes, entry_semantics="fresh MBO replay, queue kernel V5 proof; T independently checked", 
        exit_semantics="independent Python quote loop; status, category, net and exit time compared",
        facts_sha256=sha(DATA / "audit_execution_facts.json"), missing_scene_categories=scene_info["missing_categories"]))


if __name__ == "__main__":
    main()
