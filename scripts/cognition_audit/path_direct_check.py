"""Independent check of the path ladder on real quotes: vectorised direct search per entry and threshold.

For a sample of decisions on one day per block, every ladder (T, P long, P short) is recomputed from the
raw quote arrays with numpy (no shared code with the scanner loop) and compared field by field.
"""
import json
import sys
from pathlib import Path
import numpy as np
from spectrum.player.contract import NS
from spectrum.playground.data import MonthDay
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition.passive import Quotes, day_tapes, captures_by_date, MARKET
from spectrum.cognition.signatures import DEFAULT_DATA
from spectrum.cognition import path_scan as ps
from spectrum.cognition.scalper import entry_points

DATES = ('2025-10-15', '2025-12-10', '2026-02-11')
ROWS = 300


def direct(q, j0, thr):
    t, bid, ask = q.t, q.bid, q.ask
    seg = slice(j0, min(len(t), j0 + 400_000))
    tt = t[seg]; bad = (~q.ok[seg]) | (q.epoch[seg] != q.epoch[j0]) | (tt >= q.deadline)
    span = np.flatnonzero(tt - t[j0] >= ps.SPAN * NS)
    cut = len(tt)
    if bad.any():
        cut = min(cut, int(np.argmax(bad)))
    if len(span):
        cut = min(cut, int(span[0]) + 1)
    mv = .5 * (bid[seg][:cut] + ask[seg][:cut]) - .5 * (bid[j0] + ask[j0])
    hi, lo = np.maximum.accumulate(mv), np.minimum.accumulate(mv)
    out = dict(up_j=[], dn_j=[], up_lo=[], dn_hi=[])
    for x in thr:
        u = np.flatnonzero(mv >= x); d = np.flatnonzero(mv <= -x)
        out['up_j'].append(j0 + u[0] if len(u) else -1); out['up_lo'].append(lo[u[0]] if len(u) else None)
        out['dn_j'].append(j0 + d[0] if len(d) else -1); out['dn_hi'].append(hi[d[0]] if len(d) else None)
    snaps = [int(np.argmax(tt[:cut] - t[j0] >= e * NS)) if np.any(tt[:cut] - t[j0] >= e * NS) else -1 for e in ps.EDGES]
    out['snap_j'] = [j0 + s if s >= 0 else -1 for s in snaps]
    return out


def check(date):
    by = captures_by_date(MARKET, (date[:7],)); mk = by[date][0]['member_key']
    for d, tape, order in day_tapes(mk, [date]):
        day = MonthDay(Path(DEFAULT_DATA) / d[:7] / d); qtape = QuoteTape.from_day(day); q = Quotes(qtape)
        t_dec, sigma, spread = ps.decisions(day, q, qtape)
        rng = np.random.default_rng(7); pick = np.sort(rng.choice(len(t_dec), ROWS, replace=False))
        t_dec, sigma = t_dec[pick], sigma[pick]
        entries = entry_points(q, tape, t_dec, np.ones(len(t_dec), bool)); thr = ps.thresholds(sigma)
        bad = 0; checked = 0
        for start in (entries[('T', 1)][0], entries[('P', 1)][0], entries[('P', -1)][0]):
            lad = ps.ladder(q, start, thr)
            for k in np.flatnonzero(start >= 0):
                ref = direct(q, int(start[k]), thr[k]); checked += 1
                ok = (np.array_equal(lad['up_j'][k], ref['up_j']) and np.array_equal(lad['dn_j'][k], ref['dn_j'])
                      and np.array_equal(lad['snap_j'][k], ref['snap_j'])
                      and all(v is None or lad['up_lo'][k, i] == v for i, v in enumerate(ref['up_lo']))
                      and all(v is None or lad['dn_hi'][k, i] == v for i, v in enumerate(ref['dn_hi'])))
                bad += not ok
        return dict(date=d, ladders=checked, levels=checked * thr.shape[1], mismatched_ladders=bad)


if __name__ == '__main__':
    res = [check(d) for d in (sys.argv[1:] or DATES)]
    print(json.dumps(res))
    out = Path('reports/scalper_mind_v1'); out.mkdir(parents=True, exist_ok=True)
    (out / 'path_direct_check.json').write_text(json.dumps(res, indent=2), encoding='utf8')
