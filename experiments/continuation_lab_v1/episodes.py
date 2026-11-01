"""Three explicit market phrases on native quotes, with causal confirmations.

    drive -> pullback -> resume
    frozen-place break -> first retest -> resume
    first bounce -> significant departure -> return -> repeated bounce

An earlier extreme is a remembered fact, never an earlier executable signal.
R is frozen before the relevant reaction. Gate failures and unfinished stories
are retained in the ledger; recognition success is not a profitability label.
"""
import numpy as np
import pandas as pd
from numba import njit

NS = 1_000_000_000
COLS = ['source_index', 'available_ns', 'side', 'family_code', 'episode_id',
        'phase_code', 'start_index', 'peak_index', 'trough_index', 'gate_index',
        'risk_nano', 'context_index', 'level_nano', 'block']
PHASES = {1: 'DRIVE', 2: 'PULLBACK', 3: 'GATE_START', 4: 'GATE_ABORT',
          5: 'CONFIRMED', 6: 'INVALIDATED', 7: 'EXPIRED', 8: 'SUPPORT_LOST',
          9: 'CONTACT', 10: 'FIRST_REACTION', 11: 'DEPARTED', 12: 'RETEST',
          13: 'BLOCK_END', 14: 'OPEN_AT_END'}
FAMILIES = {1: 'pullback', 2: 'break_retest', 3: 'repeat_bounce'}


@njit(cache=True)
def _pullbacks(t, p, good, epoch, ci, risk, start, end, side):
    rows = []
    stage = 0
    base = peak = trough = frozen_ci = 0
    gate = -1
    R = 0
    serial = 0
    seen = 0
    last = -1
    for i in range(len(t)):
        if t[i] < start:
            continue
        if t[i] >= end:
            break
        continuous = (last >= 0 and good[i] and good[last] and epoch[i] == epoch[last]
                      and t[i] - t[last] <= 30 * NS)
        if not good[i] or risk[i] <= 0 or not continuous:
            if stage:
                rows.append((i, t[i], side, 1, serial, 8, base, peak, trough, gate, R, frozen_ci, 0, -1))
            stage = 0; base = peak = trough = i; R = risk[i]; gate = -1; last = i
            continue
        last = i
        if stage == 0:
            if R <= 0 or side * (p[i] - p[base]) <= 0 or t[i] - t[base] > 300 * NS:
                base = peak = trough = i; R = risk[i]
            elif side * (p[i] - p[base]) >= R:
                serial += 1; stage = 1; peak = i; frozen_ci = ci[i]
                rows.append((i, t[i], side, 1, serial, 1, base, peak, peak, -1, R, frozen_ci, 0, -1))
        elif stage == 1:
            if side * (p[i] - p[peak]) > 0:
                peak = i
            if t[i] - t[base] > 300 * NS:
                rows.append((i, t[i], side, 1, serial, 7, base, peak, i, -1, R, frozen_ci, 0, -1))
                stage = 0; base = i; R = risk[i]
            elif side * (p[peak] - p[i]) * 2 >= R:
                trough = i; gate = -1; seen = t[i]; frozen_ci = ci[i]
                if side * (p[peak] - p[i]) * 4 > 5 * R:
                    rows.append((i, t[i], side, 1, serial, 6, base, peak, trough, -1, R, frozen_ci, 0, -1))
                    stage = 0; base = i; R = risk[i]
                else:
                    stage = 2
                    rows.append((i, t[i], side, 1, serial, 2, base, peak, trough, -1, R, frozen_ci, 0, -1))
        else:
            depth = side * (p[peak] - p[i])
            if depth * 4 > 5 * R or t[i] - seen > 180 * NS:
                code = 6 if depth * 4 > 5 * R else 7
                rows.append((i, t[i], side, 1, serial, code, base, peak, trough, gate, R, frozen_ci, 0, -1))
                stage = 0; base = i; R = risk[i]; gate = -1
                continue
            if side * (p[i] - p[trough]) < 0:
                trough = i
            resumed = side * (p[i] - p[trough]) * 10 >= 3 * R
            if not resumed and gate >= 0:
                rows.append((i, t[i], side, 1, serial, 4, base, peak, trough, gate, R, frozen_ci, 0, -1))
                gate = -1
            elif resumed and gate < 0:
                gate = i
                rows.append((i, t[i], side, 1, serial, 3, base, peak, trough, gate, R, frozen_ci, 0, -1))
            elif resumed and t[i] - t[gate] >= 2 * NS:
                rows.append((i, t[i], side, 1, serial, 5, base, peak, trough, gate, R, frozen_ci, 0, -1))
                stage = 0; base = peak = trough = i; R = risk[i]; gate = -1
    if stage and last >= 0:
        rows.append((last, end, side, 1, serial, 14, base, peak, trough, gate, R, frozen_ci, 0, -1))
    return rows


@njit(cache=True)
def _places(t, p, good, epoch, start, end, level, R, frozen_ci, frozen_epoch, block, side, family):
    rows = []
    stage = 0
    base = peak = trough = 0
    gate = -1
    serial = 0
    first = 0
    last = -1
    a = np.searchsorted(t, start, 'left')
    b = np.searchsorted(t, end, 'left')
    for i in range(a, b):
        continuous = (last >= 0 and good[i] and good[last] and epoch[i] == epoch[last]
                      and t[i] - t[last] <= 30 * NS)
        if not good[i] or not continuous or epoch[i] != frozen_epoch:
            if stage:
                rows.append((i, t[i], side, family, serial, 8, base, peak, trough, gate, R, frozen_ci, level, block))
            stage = 0; gate = -1; last = i
            continue
        z = side * (p[i] - level)
        old_z = side * (p[last] - level)
        last = i
        near = abs(z) * 4 <= R
        if stage == 0:
            # A repeated bounce must first approach from its eventual departure
            # side. A breakout may start from either side of the place.
            if near and (family == 2 or old_z * 4 > R):
                serial += 1; stage = 1; base = peak = trough = i; first = t[i]; gate = -1
                rows.append((i, t[i], side, family, serial, 9, base, peak, trough, gate, R, frozen_ci, level, block))
            continue
        expired = ((family == 2 and t[i] - t[base] > 180 * NS)
                   or (stage <= 1 and t[i] - first > 180 * NS))
        if z * 2 < -R or expired:
            code = 6 if z * 2 < -R else 7
            rows.append((i, t[i], side, family, serial, code, base, peak, trough, gate, R, frozen_ci, level, block))
            stage = 0; gate = -1
            continue
        if stage == 1:
            above = z * 2 >= R
            if not above and gate >= 0:
                rows.append((i, t[i], side, family, serial, 4, base, peak, trough, gate, R, frozen_ci, level, block))
                gate = -1
            elif above and gate < 0:
                gate = i
                rows.append((i, t[i], side, family, serial, 3, base, peak, trough, gate, R, frozen_ci, level, block))
            elif above and t[i] - t[gate] >= 2 * NS:
                stage = 2; peak = i; first = t[i]; gate = -1
                rows.append((i, t[i], side, family, serial, 10, base, peak, trough, gate, R, frozen_ci, level, block))
                if family == 3 and z >= R:
                    stage = 3
                    rows.append((i, t[i], side, family, serial, 11, base, peak, trough, gate, R, frozen_ci, level, block))
        elif stage == 2:
            if side * (p[i] - p[peak]) > 0:
                peak = i
            if family == 3 and z >= R:
                stage = 3
                rows.append((i, t[i], side, family, serial, 11, base, peak, trough, gate, R, frozen_ci, level, block))
            elif family == 2 and near:
                stage = 4; trough = i; first = t[i]
                rows.append((i, t[i], side, family, serial, 12, base, peak, trough, gate, R, frozen_ci, level, block))
            elif family == 2 and t[i] - first > 180 * NS:
                rows.append((i, t[i], side, family, serial, 7, base, peak, trough, gate, R, frozen_ci, level, block))
                stage = 0
        elif stage == 3:
            if side * (p[i] - p[peak]) > 0:
                peak = i
            if near:
                stage = 4; trough = i; first = t[i]
                rows.append((i, t[i], side, family, serial, 12, base, peak, trough, gate, R, frozen_ci, level, block))
        else:
            if t[i] - first > 180 * NS:
                rows.append((i, t[i], side, family, serial, 7, base, peak, trough, gate, R, frozen_ci, level, block))
                stage = 0; gate = -1
                continue
            if side * (p[i] - p[trough]) < 0:
                trough = i
            resumed = side * (p[i] - p[trough]) * 10 >= 3 * R and z >= 0
            if not resumed and gate >= 0:
                rows.append((i, t[i], side, family, serial, 4, base, peak, trough, gate, R, frozen_ci, level, block))
                gate = -1
            elif resumed and gate < 0:
                gate = i
                rows.append((i, t[i], side, family, serial, 3, base, peak, trough, gate, R, frozen_ci, level, block))
            elif resumed and t[i] - t[gate] >= 2 * NS:
                rows.append((i, t[i], side, family, serial, 5, base, peak, trough, gate, R, frozen_ci, level, block))
                stage = 0; gate = -1
    if stage and last >= 0:
        # End of a fixed level's lifetime is an administrative boundary, not a
        # statement that the market invalidated the setup.
        rows.append((last, end, side, family, serial, 13, base, peak, trough, gate, R, frozen_ci, level, block))
    return rows


def arrays(quotes):
    return (quotes.available_ns.to_numpy(np.int64),
            (quotes.bid_nano.to_numpy(np.int64) + quotes.ask_nano.to_numpy(np.int64)) // 2,
            (quotes.valid & quotes.epoch.ge(0) & quotes.bid_nano.gt(0)
             & quotes.ask_nano.gt(quotes.bid_nano)).to_numpy(bool),
            quotes.epoch.to_numpy(np.int64))


def generate(quotes, ci, risk, levels, start_ns, end_ns):
    t, p, good, epoch = arrays(quotes)
    frames = []
    for side in (1, -1):
        f = pd.DataFrame(_pullbacks(t, p, good, epoch, ci, risk, start_ns, end_ns, side), columns=COLS, dtype='int64')
        f['level_kind'] = 'none'; frames.append(f)
    for level in levels:
        if not level['known']:
            continue
        for side in (1, -1):
            for family in (2, 3):
                f = pd.DataFrame(_places(t, p, good, epoch, level['start_ns'], level['end_ns'],
                                        int(round(level['level'] * NS)), int(np.ceil(level['risk'] * NS)),
                                        level['context_index'], level['epoch'], level['block'], side, family), columns=COLS, dtype='int64')
                f['level_kind'] = level['kind']; frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    f['family'] = f.family_code.map(FAMILIES)
    f['phase'] = f.phase_code.map(PHASES)
    f['story_id'] = (f.family + ':' + f.level_kind + ':' + f.block.astype(str)
                     + ':' + f.side.astype(str) + ':' + f.episode_id.astype(str))
    return f.sort_values(['available_ns', 'source_index', 'family_code', 'side'], kind='stable').reset_index(drop=True)
