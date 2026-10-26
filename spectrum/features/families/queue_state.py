"""Reusable formulas extracted from spectrum/cognition/queue_state.py; see docs/MIGRATION.json."""
from collections import defaultdict
import numpy as np
from spectrum.core.schema import NS
from spectrum.market import queue as qu
TICK = 100_000_000

VERSION = 'QUEUE_STATE_AT_DECISION_V1'


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
