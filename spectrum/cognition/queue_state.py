"""Queue state at the decision time of every signature origin.

All values use MBO records with ts_recv strictly before the intent t, i.e. the
information a decision at t can have; the 100 ms until the order rests are
not used. Prices are the decision-time best bid B and ask A.

  bid1/ask1, bid2/ask2   resting lots at B, A, B-1 tick, A+1 tick (kernel book)
  {bid,ask}_{add,cancel,fill}_{w}  lots added / cancelled / filled at the
                         address B (bid side) or A (ask side) in [t-w, t)
  tbuy_{w}, tsell_{w}    aggressor volume by side in [t-w, t)
Unknown decision quote -> every field masked (state_known = 0).
"""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.player.contract import NS
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from .execution import QuoteTape
from .signatures import DEFAULT_DATA, DEFAULT_OUTPUT as SIGNATURES, TRAIN_MONTHS
from .passive import Quotes, TICK, captures_by_date, day_tapes, MARKET
from . import queue as qu

VERSION = 'QUEUE_STATE_AT_DECISION_V1'
DEFAULT_OUTPUT = 'data/derived/queue_state_v1'
WINDOWS = (5, 30)
_A, _C, _F, _T, _B, _S = (ord(c) for c in 'ACFTBA')


def window_flows(tape, t, bid, ask, windows=WINDOWS):
    """Flows strictly before each t at the decision-time best-price addresses."""
    ts = tape['ts']; out = defaultdict(lambda: np.zeros(len(t)))
    hi = np.searchsorted(ts, t, side='left')
    for w in windows:
        lo = np.searchsorted(ts, t - w * NS, side='left')
        for i in range(len(t)):
            if hi[i] <= lo[i]:
                continue
            sl = slice(lo[i], hi[i]); a = tape['action'][sl]; s = tape['side'][sl]
            p = tape['price'][sl]; z = tape['size'][sl]
            for name, side, price in (('bid', _B, bid[i]), ('ask', _S, ask[i])):
                at = (s == side) & (p == price)
                out[f'{name}_add_{w}'][i] = z[at & (a == _A)].sum()
                out[f'{name}_cancel_{w}'][i] = z[at & (a == _C)].sum()
                out[f'{name}_fill_{w}'][i] = z[at & (a == _F)].sum()
            trade = a == _T
            out[f'tbuy_{w}'][i] = z[trade & (s == _B)].sum()
            out[f'tsell_{w}'][i] = z[trade & (s == _S)].sum()
    return dict(out)


def level_sizes(tape, t, bid, ask):
    """Resting lots at four addresses, from the validated kernel book at t."""
    n = len(t); prices = [bid, ask, bid - TICK, ask + TICK]; sides = [1, -1, 1, -1]
    place = np.concatenate([t] * 4); side = np.concatenate([np.full(n, s) for s in sides])
    price = np.concatenate(prices)
    status, _, _, ahead = qu.probe(tape, place, side, price, place)
    placed = status != qu.UNPLACED
    sizes = {name: np.where(placed[k * n:(k + 1) * n], ahead[k * n:(k + 1) * n], 0)
             for k, name in enumerate(('bid1', 'ask1', 'bid2', 'ask2'))}
    return sizes, placed[:n]


def process_day(date, tape, sig_dir, data):
    started = time.perf_counter()
    x = pd.read_parquet(Path(sig_dir) / date / 'x.parquet', columns=['origin_id', 'intent_ns'])
    day = MonthDay(Path(data) / date[:7] / date); q = Quotes(QuoteTape.from_day(day))
    t = x.intent_ns.to_numpy(np.int64)
    d = np.array([q.asof(v) for v in t])
    known = d >= 0
    bid = np.where(known, q.bid[np.clip(d, 0, None)], 0).astype(np.int64)
    ask = np.where(known, q.ask[np.clip(d, 0, None)], 0).astype(np.int64)
    sizes, placed = level_sizes(tape, t, bid, ask)
    flows = window_flows(tape, t, bid, ask)
    known &= placed
    frame = pd.DataFrame(dict(origin_id=x.origin_id, intent_ns=t, state_known=known.astype(np.int8),
                              spread_ticks=np.where(known, (ask - bid) / TICK, 0.),
                              **{k: np.where(known, v, 0) for k, v in {**sizes, **flows}.items()}))
    QUOTE_CACHE.clear()
    return frame, dict(date=date, origins=len(frame), known=int(known.sum()), seconds=time.perf_counter() - started)


def run_member(member_key, dates, sig_dir, data, output):
    receipts = []
    for d, tape, order in day_tapes(member_key, dates):
        frame, r = process_day(d, tape, sig_dir, data)
        dest = Path(output) / f'{d}.parquet'; frame.to_parquet(dest, index=False)
        r.update(sha256=sha(dest), captures=order); save_json(Path(output) / f'{d}.json', r); receipts.append(r)
        print(json.dumps({k: r[k] for k in ('date', 'origins', 'known', 'seconds')}), flush=True)
    return receipts


def _job(args):
    return run_member(*args)


def prepare(sig_dir=SIGNATURES, data=DEFAULT_DATA, output=DEFAULT_OUTPUT, workers=5, months=TRAIN_MONTHS):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(sig_dir) / 'manifest.json').read_text(encoding='utf8'))
    dates = [d['date'] for d in manifest['days'] if d['date'][:7] in months]
    by = captures_by_date(MARKET, months); groups = defaultdict(list)
    for d in dates:
        if not (out / f'{d}.json').exists() and by.get(d):
            groups[by[d][0]['member_key']].append(d)
    code = {p: sha(Path(__file__).with_name(p)) for p in ('queue_state.py', 'queue.py', 'passive.py')}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_job, [(m, ds, sig_dir, data, output) for m, ds in sorted(groups.items())]))
    done = sorted(p.stem for p in out.glob('*.json') if p.stem[:4].isdigit())
    save_json(out / 'manifest.json', dict(status='COMPLETE' if len(done) == len(dates) else 'PARTIAL', version=VERSION,
        days=done, missing=[d for d in dates if d not in done], windows=list(WINDOWS), code=code,
        seconds=time.perf_counter() - started, semantics=__doc__.strip()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=DEFAULT_OUTPUT)
    p.add_argument('--workers', type=int, default=5)
    a = p.parse_args(); prepare(output=a.output, workers=a.workers)
