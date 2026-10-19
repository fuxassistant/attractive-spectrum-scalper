"""Sequential schema experiment on train months with controls.

Variants share one origin stream and identical outcomes:
  full             assimilation + accommodation + quarantine
  assimilate_only  root schemas updated, no new schemas
  frozen_after_A   everything learned in block A, no updates in block B
  no_forgetting    full, infinite half-life (stability over plasticity)
  shuffled_context full, coordinates permuted within each day (breaks meaning)
One 1 oz account per variant; equity = $1500 + cumulative net in time order.
"""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.player.contract import NS
from .qmodel import dataset
from .signature_report import load, day_boot
from .signatures import DEFAULT_OUTPUT as SIGNATURES, MENU_PLANS
from .representation import DEFAULT_OUTPUT as REPRESENTATION
from . import schemas as sm

REPORT = 'reports/schemas_v1'
VARIANTS = dict(full=dict(), assimilate_only=dict(accommodation=False), frozen_after_A=dict(freeze_at_block='B'),
                no_forgetting=dict(half_life_days=float('inf')), shuffled_context=dict())
# Positive control (REGIME_CHANGE_AND_RETURN): a known edge exists only in Dec-Jan.
PLANTED = dict(start='2025-12-01', end='2026-01-31', coordinate=('range', 'low'), plan=3, usd=2.5)
PLANTED_VARIANTS = ('full', 'assimilate_only', 'frozen_after_A', 'no_forgetting')


def plant_edge(outcomes, frame, spec=PLANTED):
    """Long gains, short loses spec.usd where the causal coordinate holds in the window."""
    coords = sm.coordinates(frame); name, value = spec['coordinate']
    hit = (coords[name] == value) & (frame.date >= spec['start']).to_numpy() & (frame.date <= spec['end']).to_numpy()
    planted = dict(outcomes)
    for oid in frame.index[hit]:
        for side in (-1, 1):
            key = (oid, spec['plan'], side)
            if key in planted and planted[key][0] == 'COMPLETE':
                status, net, known = planted[key]
                planted[key] = (status, net + side * spec['usd'], known)
    return planted, int(hit.sum())


def monthly(trades):
    if not len(trades):
        return {}
    t = trades[trades.net.notna()]
    return {m: dict(trades=len(g), net=float(g.net.sum()), schemas=g['describe'].value_counts().head(4).to_dict())
            for m, g in t.groupby(t.date.str[:7])}


def outcome_table(signatures):
    _, _, _, b = load(signatures)
    b = b[b.plan_id.isin(sm.PLANS)]
    out = {}
    for r in b.itertuples():
        if r.status == 'COMPLETE':
            out[(r.origin_id, r.plan_id, r.side)] = ('COMPLETE', float(r.net_usd), int(r.exit_ns))
        elif r.status in ('CENSORED', 'NOT_FILLED'):
            out[(r.origin_id, r.plan_id, r.side)] = (r.status, None, None)
    return out


def fix_unknown_times(outcomes, frame):
    """Censored/unfilled outcomes resolve at their declared branch horizon."""
    intents = frame.intent_ns.to_dict()
    for key, (status, net, known) in list(outcomes.items()):
        if status != 'COMPLETE':
            outcomes[key] = (status, net, int(intents[key[0]]) + (MENU_PLANS[key[1]][2] + 5) * NS)
    return outcomes


def block_metrics(trades, frame, block):
    dates = sorted(frame.date[frame.block == block].unique())
    t = trades[trades.block == block] if len(trades) else trades
    done = t[t.net.notna()] if len(t) else t
    daily = pd.DataFrame(dict(date=dates)).merge(
        done.groupby('date', as_index=False).net.sum() if len(done) else pd.DataFrame(columns=['date', 'net']),
        on='date', how='left')
    daily['net'] = pd.to_numeric(daily.net, errors='coerce').fillna(0.)
    equity = done.sort_values('intent_ns').net.cumsum().to_numpy() if len(done) else np.zeros(0)
    peak = np.maximum.accumulate(np.r_[0., equity]) if len(equity) else np.zeros(1)
    return dict(days=len(dates), trades=int(len(t)), completed=int((t.status == 'COMPLETE').sum()) if len(t) else 0,
                censored=int((t.status == 'CENSORED').sum()) if len(t) else 0,
                not_filled=int((t.status == 'NOT_FILLED').sum()) if len(t) else 0,
                net=float(done.net.sum()) if len(done) else 0., net_per_trade=float(done.net.mean()) if len(done) else None,
                win_rate=float((done.net > 0).mean()) if len(done) else None,
                max_drawdown=float((peak - np.r_[0., equity]).max()) if len(equity) else 0.,
                daily=day_boot(daily, 'net'), active_days=int((daily.net != 0).sum()))


def library_view(lib):
    rows = []
    for k in lib.schemas:
        s = k.stat
        rows.append(dict(id=k.id, describe=k.describe(), program=k.program, plan=list(MENU_PLANS[k.plan]),
                         context=[list(c) for c in k.context], parent=k.parent, depth=k.depth, status=k.status,
                         n=s.n, days=len(s.days), total_net=s.total, mean_decayed=s.mean,
                         sd=s.sd if np.isfinite(s.sd) else None, mean_all=s.total / s.n if s.n else None,
                         acted=k.acted, acted_net=k.acted_net, surprises=k.surprises, trigger=k.trigger,
                         confirm=dict(n=k.confirm.n, mean=k.confirm.mean) if k.parent >= 0 else None,
                         history=k.history))
    return rows


def expectation_error(lib):
    """Prequential squared error of schema means vs a context-free running mean."""
    months = {}
    for k in lib.schemas:
        for m, (se, n, se_g) in k.err.items():
            v = months.setdefault(m, [0., 0, 0.]); v[0] += se; v[1] += n; v[2] += se_g
    return {m: dict(n=n, mse=se / n, mse_context_free=g / n, skill=1 - se / g) for m, (se, n, g) in sorted(months.items())}


def fingerprint(trades):
    if not len(trades):
        return {}
    out = {}
    for block, t in trades.groupby('block'):
        out[block] = dict(programs=t.program.value_counts(normalize=True).round(3).to_dict(),
                          plans=t.plan.map(lambda p: str(MENU_PLANS[p])).value_counts(normalize=True).round(3).to_dict(),
                          long_share=float((t.side > 0).mean()), child_share=float((t.depth > 0).mean()),
                          distinct_schemas=int(t.schema.nunique()),
                          top=t.groupby('describe').net.agg(['size', 'sum']).sort_values('size', ascending=False)
                              .head(6).round(2).reset_index().to_dict('records'))
    return out


def experiment(signatures=SIGNATURES, representation=REPRESENTATION, output=REPORT):
    started = time.perf_counter(); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    frame, _, _ = dataset(signatures, representation)
    frame = frame.sort_values('intent_ns', kind='stable')
    frame['day_index'] = frame.date.rank(method='dense').astype(int)
    outcomes = fix_unknown_times(outcome_table(signatures), frame)
    planted, planted_origins = plant_edge(outcomes, frame)
    runs = [(name, overrides, outcomes) for name, overrides in VARIANTS.items()]
    runs += [('planted_' + name, VARIANTS[name], planted) for name in PLANTED_VARIANTS]
    results = {}
    for name, overrides, stream in runs:
        t0 = time.perf_counter(); params = replace(sm.Params(), **overrides)
        lib, trades = sm.run(frame, stream, params, shuffle_seed=61051 if name == 'shuffled_context' else None)
        statuses = pd.Series([k.status for k in lib.schemas]).value_counts().to_dict()
        results[name] = dict(params={k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v)
                                     for k, v in asdict(params).items()},
                             A=block_metrics(trades, frame, 'A'), B=block_metrics(trades, frame, 'B'),
                             schemas=len(lib.schemas), statuses=statuses,
                             spawned=sum(e['kind'] == 'spawn' for e in lib.events),
                             promoted=sum(e['kind'] == 'active' for e in lib.events),
                             refuted=sum(e['kind'] == 'retired' for e in lib.events),
                             expectation_error=expectation_error(lib), fingerprint=fingerprint(trades),
                             monthly=monthly(trades), events=lib.events[:40],
                             seconds=time.perf_counter() - t0)
        save_json(out / f'library_{name}.json', dict(schemas=library_view(lib), events=lib.events))
        if len(trades):
            trades.to_parquet(out / f'trades_{name}.parquet', index=False)
        else:
            (out / f'trades_{name}.parquet').unlink(missing_ok=True)  # never leave a stale earlier run
        print(name, json.dumps({b: {k: results[name][b][k] for k in ('trades', 'net', 'net_per_trade')} for b in 'AB'}),
              results[name]['spawned'], results[name]['promoted'], results[name]['refuted'], flush=True)
    summary = dict(version='SCHEMA_LIBRARY_V1', seconds=time.perf_counter() - started,
                   signatures_manifest_sha256=sha(Path(signatures) / 'manifest.json'),
                   representation_manifest_sha256=sha(Path(representation) / 'manifest.json'),
                   code_sha256={p: sha(Path(__file__).with_name(p)) for p in ('schemas.py', 'schema_run.py')},
                   coordinates=list(sm.COORDINATES), programs=list(sm.PROGRAMS),
                   plans=[list(MENU_PLANS[p]) for p in sm.PLANS], variants=results,
                   planted_control=dict(PLANTED, coordinate=list(PLANTED['coordinate']), origins=planted_origins,
                                        meaning='Synthetic positive control on the real stream; never economic evidence'),
                   scope='Train months only (A=Oct-Dec 2025, B=Jan-Feb 2026), prequential; development evidence.')
    save_json(out / 'schema_summary.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=REPORT)
    a = p.parse_args(); experiment(output=a.output)
