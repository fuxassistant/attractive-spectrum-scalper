"""Outcome signatures: the full fixed-1oz action menu from one shared origin.

Each origin (a causal event or a price-independent clock point) starts from one
flat ExecutionReplay state; every sibling branch is an O(1) clone over the same
immutable tape, so the only difference between siblings is the action. WAIT is
flat cash, exactly zero. Branches never re-enter. Origins are independent
experiments with a fresh account; they are not one continuous equity curve.

X (causal at intent), Y (opportunity future) and branch outcomes are written to
separate files. Event fields are the event study's declared X allowlist; its
future labels are never read here.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.player.contract import PLANS, NS
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from .execution import ExecutionReplay, ExecutionConfig, QuoteTape
from . import opportunity as op

VERSION = 'OPPORTUNITY_SIGNATURES_V1'
DEFAULT_DATA = 'data/derived/basket_multimonth_v1'
DEFAULT_EVENTS = 'reports/event_landmarks_v1/run_003'
DEFAULT_OUTPUT = 'data/derived/opportunity_signatures_v1'
DEFAULT_MARKET = 'data/derived/full_mbo_v1'
TRAIN_MONTHS = ('2025-10', '2025-11', '2025-12', '2026-01', '2026-02')
BLOCKS = dict(A=('2025-10', '2025-11', '2025-12'), B=('2026-01', '2026-02'))
MENU_PLANS = tuple(PLANS) + ((80, 40, 1800),)
CLOCK_SECONDS = 300
EVENT_LOOKBACK_SECONDS = 300
SESSION_SLACK_SECONDS = 5
STATUSES = ('COMPLETE', 'CENSORED', 'NOT_FILLED', 'NOT_ELIGIBLE', 'EXCLUDED_SESSION_TIME')


def block_for(date):
    for name, months in BLOCKS.items():
        if date[:7] in months:
            return name
    raise ValueError('Date outside declared train blocks: ' + date)


def event_allowlist(events_dir):
    return list(json.loads((Path(events_dir) / 'feature_contract.json').read_text(encoding='utf8'))['x_allowlist'])


def origins_for(day, events, bank_known=True):
    """Event origins at their decision time plus a fixed 300s clock; both causal."""
    lo, cutoff = int(day.meta['london_ns']), int(day.meta['ny_close_ns'])
    e = events[(events.symbol == day.meta['symbol']) & (events.decision_ns >= lo) & (events.decision_ns < cutoff)]
    if not bank_known:
        e = e.iloc[:0]  # a partially covered day must not mix event and missing-event semantics
    e = e.sort_values('decision_ns')
    rows = [dict(origin_id=f'{day.meta["date"]}:event:{r.match_id}', kind='event', intent_ns=int(r.decision_ns),
                 match_id=r.match_id) for r in e.itertuples()]
    for t in range(lo + CLOCK_SECONDS * NS, cutoff, CLOCK_SECONDS * NS):
        rows.append(dict(origin_id=f'{day.meta["date"]}:clock:{t}', kind='clock', intent_ns=int(t), match_id=None))
    return pd.DataFrame(rows), e


def event_context(intent, event_times, bank_known=True):
    """Strictly earlier recognised events only; the origin's own event is excluded.

    Without a certified event job for the day, counts are masked, not a known zero.
    """
    intent = np.asarray(intent, np.int64); event_times = np.asarray(event_times, np.int64)
    j = np.searchsorted(event_times, intent, side='left')
    lo = np.searchsorted(event_times, intent - EVENT_LOOKBACK_SECONDS * NS, side='left')
    previous = event_times[np.clip(j - 1, 0, None)] if len(event_times) else np.zeros(len(intent), np.int64)
    known = (j > 0) & bank_known
    return dict(prior_events_300s=np.where(bank_known, j - lo, 0).astype(float),
                since_prior_event_s=np.where(known, (intent - previous) / NS, 0.), since_prior_event_known=known,
                event_bank_known=np.full(len(intent), bool(bank_known)))


def event_coverage(market, events_dir):
    """date -> True iff every capture selected for that date was in the event job."""
    done = {json.loads(p.read_text(encoding='utf8'))['key'] for p in (Path(events_dir) / 'captures').glob('*.json')}
    by_date = {}
    for c in json.loads((Path(market) / 'manifest.json').read_text(encoding='utf8'))['captures']:
        if c['status'] == 'COMPLETE':
            by_date.setdefault(c['entry']['selection']['date'], []).append(c['key'])
    return {date: all(k in done for k in keys) for date, keys in by_date.items()}


def branch(base, tape, side, plan_id, intent, mid_ticks):
    plan = MENU_PLANS[plan_id]
    row = dict(side=side, plan_id=plan_id, take_ticks=plan[0], stop_ticks=plan[1], max_hold_s=plan[2])
    horizon = intent + (plan[2] + SESSION_SLACK_SECONDS) * NS
    if horizon >= tape.deadline_ns:
        return dict(row, status='EXCLUDED_SESSION_TIME')
    r = base.clone()
    if not r.open_at(intent, side, plan_id):
        return dict(row, status='NOT_ELIGIBLE')
    s = r.reference_exit(horizon)
    if s.entry_ns <= 0:
        return dict(row, status='NOT_FILLED')
    out = dict(row, entry_ns=int(s.entry_ns), entry_latency_ms=(s.entry_ns - intent) / 1e6,
               entry_cost_ticks=side * (s.entry_price_nano / op.TICK_NANO - mid_ticks),
               mfe_usd=float(s.position_mfe), mae_usd=float(s.position_mae), fees_usd=float(s.total_fees))
    if s.censored:
        return dict(out, status='CENSORED', exit_reason=s.exit_reason)
    return dict(out, status='COMPLETE', exit_reason=s.exit_reason, net_usd=float(s.cash - s.initial_capital),
                gross_usd=float(s.cash - s.initial_capital + s.total_fees), hold_s=(s.exit_ns - s.entry_ns) / NS,
                exit_ns=int(s.exit_ns))


def prepare_day(path, events, allowlist, output, protocol_sha, bank_known):
    started = time.perf_counter(); path = Path(path); date = path.name
    dest = Path(output) / date; manifest = dest / 'manifest.json'
    identity = dict(day_manifest=sha(path / 'manifest.json'), protocol_sha256=protocol_sha)
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding='utf8'))
        if m['identity'] != identity:
            raise ValueError('Completed signature day identity changed: ' + date)
        for name, digest in m['files'].items():
            if sha(dest / name) != digest:
                raise ValueError('Completed signature payload changed: ' + date)
        return m
    day = MonthDay(path, verify=True); config = ExecutionConfig(plans=MENU_PLANS)
    tape = QuoteTape.from_day(day)
    origins, day_events = origins_for(day, events, bank_known)
    grid = op.second_grid(tape.quote_t, tape.bid, tape.ask, tape.valid, tape.epoch,
                          int(day.meta['start_ns']), int(day.meta['end_ns']))
    intents = origins.intent_ns.to_numpy(np.int64)
    x, y = op.measure(grid, intents)
    ctx = event_context(intents, np.sort(day_events.decision_ns.to_numpy(np.int64)), bank_known)
    good = np.flatnonzero(np.asarray(tape.valid, bool) & (tape.bid > 0) & (tape.ask >= tape.bid))
    q = np.searchsorted(tape.quote_t[good], intents, side='right') - 1
    if len(good):  # a closed-exchange day can hold no valid book at all
        g = good[np.clip(q, 0, None)]
        mid = np.where(q >= 0, (tape.bid[g] + tape.ask[g]) / (2 * op.TICK_NANO), np.nan)
    else:
        mid = np.full(len(intents), np.nan)
    lo = int(day.meta['london_ns'])
    xf = pd.DataFrame(dict(origin_id=origins.origin_id, kind=origins.kind, date=date, block=block_for(date),
                           intent_ns=intents, minutes_since_london=(intents - lo) / (60 * NS), **x, **ctx))
    ev = day_events.set_index('match_id')[allowlist].copy()
    for name in allowlist:
        if ev[name].dtype == bool:
            ev[name] = ev[name].astype(float)  # NaN on clock origins, never a fabricated False
    joined = ev.reindex(origins.match_id.where(origins.kind == 'event'))
    for name in allowlist:
        xf['ev_' + name] = joined[name].to_numpy()
    yf = pd.DataFrame(dict(origin_id=origins.origin_id, **y))
    rows = []
    for i, intent in enumerate(intents):
        base = ExecutionReplay(tape, config)
        base.advance_until(int(intent))
        direction = xf['ev_direction'].iat[i] if origins.kind.iat[i] == 'event' else np.nan
        for side in (-1, 1):
            for plan_id in range(len(MENU_PLANS)):
                r = branch(base, tape, side, plan_id, int(intent), float(mid[i]))
                r.update(origin_id=origins.origin_id.iat[i],
                         relative=('continue' if side == direction else 'fade') if np.isfinite(direction) else
                         ('long' if side > 0 else 'short'))
                rows.append(r)
    bf = pd.DataFrame(rows)
    dest.mkdir(parents=True, exist_ok=True)
    xf.to_parquet(dest / 'x.parquet', index=False); yf.to_parquet(dest / 'y.parquet', index=False)
    bf.to_parquet(dest / 'branches.parquet', index=False)
    counts = bf.status.value_counts().to_dict()
    m = dict(status='COMPLETE', version=VERSION, date=date, block=block_for(date), symbol=day.meta['symbol'],
             identity=identity, event_bank_known=bool(bank_known), origins=len(origins), event_origins=int((origins.kind == 'event').sum()),
             branches=len(bf), branch_status={k: int(counts.get(k, 0)) for k in STATUSES},
             seconds=time.perf_counter() - started,
             files={n: sha(dest / n) for n in ('x.parquet', 'y.parquet', 'branches.parquet')})
    save_json(manifest, m); QUOTE_CACHE.clear()
    return m


def _job(args):
    return prepare_day(*args)


def day_paths(data, months):
    paths = []
    for month in months:
        m = json.loads((Path(data) / month / 'manifest.json').read_text(encoding='utf8'))
        if m['status'] != 'COMPLETE':
            raise ValueError('Unfinished source month ' + month)
        for row in m['days']:
            p = Path(data) / month / row['date']
            if sha(p / 'manifest.json') != row['manifest_sha256']:
                raise ValueError('Changed source day ' + str(p))
            paths.append(p)
    return sorted(paths)


def prepare(data=DEFAULT_DATA, events_dir=DEFAULT_EVENTS, output=DEFAULT_OUTPUT, months=TRAIN_MONTHS,
            workers=4, limit=None, market=DEFAULT_MARKET):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if any(m not in TRAIN_MONTHS for m in months):
        raise ValueError('This protocol reads train months only; later blocks stay untouched')
    code = {p.name: sha(p) for p in (Path(__file__), Path(op.__file__), Path(__file__).with_name('execution.py'))}
    protocol = dict(version=VERSION, months=list(months), blocks={k: list(v) for k, v in BLOCKS.items()},
        config=asdict(ExecutionConfig(plans=MENU_PLANS)), plans=[list(p) for p in MENU_PLANS],
        horizons=list(op.HORIZONS), sigma_windows=list(op.SIGMA_WINDOWS), baseline_window=op.BASELINE_WINDOW,
        clock_seconds=CLOCK_SECONDS, events_sha256=sha(Path(events_dir) / 'causal_events.parquet'),
        event_contract_sha256=sha(Path(events_dir) / 'feature_contract.json'),
        catalog_sha256=sha(Path(data) / 'catalog.json'), market_manifest_sha256=sha(Path(market) / 'manifest.json'),
        code=code,
        semantics=dict(
            origins='All recognised events in London..NY entry window at their decision time, plus a 300s clock grid; no outcome selection.',
            branches='From one flat state per origin: 2 sides x 4 plans, 1 oz, existing CAUSAL_HOLD_CLOSE_BBO_V1 execution; no re-entry. WAIT = 0.',
            event_bank='Known for a date iff every capture selected for it was in the event job; otherwise no event origins and masked event context.',
            exclusions='Plan duration + 5s must end before the liquidation deadline, else EXCLUDED_SESSION_TIME (explicit, not dropped).',
            opportunity='Y only: one-second as-of mid; mature iff every horizon step supported in one segment. Baseline sigma over trailing 600s.',
            economics='Independent fresh-account experiments; sums are not an equity curve. Train months only; validation/evaluation untouched.'))
    if (output / 'protocol.json').exists():
        old = json.loads((output / 'protocol.json').read_text(encoding='utf8'))
        if json.dumps(old, sort_keys=True) != json.dumps(protocol, sort_keys=True):
            raise ValueError('Protocol drift; new output directory required')
    else:
        save_json(output / 'protocol.json', protocol)
    protocol_sha = sha(output / 'protocol.json')
    allowlist = event_allowlist(events_dir)
    events = pd.read_parquet(Path(events_dir) / 'causal_events.parquet',
                             columns=['symbol', 'decision_ns', 'match_id'] + allowlist)
    paths = day_paths(data, months)
    if limit is not None:
        paths = paths[:limit]
    coverage = event_coverage(market, events_dir)
    jobs = []
    for p in paths:
        lo = int(json.loads((p / 'manifest.json').read_text(encoding='utf8'))['start_ns'])
        jobs.append((p, events[(events.decision_ns >= lo) & (events.decision_ns < lo + 2 * 86400 * NS)],
                     allowlist, output, protocol_sha, coverage.get(p.name, False)))
    started = time.perf_counter(); results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for m in pool.map(_job, jobs):
            results.append(m)
            print(json.dumps({k: m[k] for k in ('date', 'origins', 'event_origins', 'branches', 'seconds')}), flush=True)
            save_json(output / 'progress.json', dict(status='RUNNING', days=len(results), expected=len(jobs)))
    save_json(output / 'manifest.json', dict(status='COMPLETE' if limit is None else 'PILOT', version=VERSION,
        days=results, seconds=time.perf_counter() - started, origins=sum(d['origins'] for d in results),
        branches=sum(d['branches'] for d in results), code=code))
    save_json(output / 'progress.json', dict(status='COMPLETE', days=len(results), expected=len(jobs)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=DEFAULT_DATA); p.add_argument('--events', default=DEFAULT_EVENTS)
    p.add_argument('--output', default=DEFAULT_OUTPUT); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--months', nargs='+', default=list(TRAIN_MONTHS)); p.add_argument('--limit', type=int)
    a = p.parse_args(); prepare(a.data, a.events, a.output, a.months, a.workers, a.limit)
