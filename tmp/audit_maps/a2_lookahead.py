"""Perturbation test: change everything after second T and check every feature at <= T is unchanged."""
import sys, numpy as np
sys.path.insert(0, 'scripts/hotpath')
import opportunity_sense as osn, tug_spectrum as ts, bands_compare as bc
from spectrum.cognition import bands as bd, race as rc
from spectrum.cognition.hotpath import base_day
from pathlib import Path
from spectrum.playground.data import MonthDay
from spectrum.cognition.signatures import DEFAULT_DATA
from spectrum.cognition.representation import load_seconds
date = '2025-11-18'
b, m = base_day(date)
rng = np.random.default_rng(0)
T = m['london'] + 4 * 3600
b2 = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in b.items()}
for kk in ('mid', 'spread', 'trade_buy', 'trade_sell', 'trade_unknown', 'add_bid', 'add_ask', 'remove_bid', 'remove_ask', 'sigma', 'sigma60', 'sigma3600'):
    x = b2[kk]; x[T + 1:] = x[T + 1:] * rng.uniform(.5, 2, len(x) - T - 1) + rng.normal(0, 5, len(x) - T - 1) * (kk == 'mid')
f1, f2 = osn.features(b, m), osn.features(b2, m)
bad = [c for c in osn.FEATS if not np.allclose(np.nan_to_num(np.asarray(f1[c], float)[:T + 1], nan=-99), np.nan_to_num(np.asarray(f2[c], float)[:T + 1], nan=-99))]
print('opportunity_sense.features changed at <=T:', bad)
v1 = bd.vwap_bands(b['mid'], b['trade_buy'] + b['trade_sell'] + b['trade_unknown']); v2 = bd.vwap_bands(b2['mid'], b2['trade_buy'] + b2['trade_sell'] + b2['trade_unknown'])
for W in bd.WINDOWS:
    for j in range(2):
        a, c = np.nan_to_num(v1[W][j][:T + 1], nan=-99), np.nan_to_num(v2[W][j][:T + 1], nan=-99)
        if not np.allclose(a, c, rtol=0, atol=1e-6): print('vwap_bands changed', W, j, np.max(np.abs(a - c)))
print('vwap_bands checked')
k = np.arange(m['london'], m['ny_close'])
e1, fr1, p1 = ts.stft_frames(b['mid'], k); e2, fr2, p2 = ts.stft_frames(b2['mid'], k)
kk = k <= T
print('stft per-second changed at <=T:', not np.allclose(np.nan_to_num(p1[kk]), np.nan_to_num(p2[kk])))
# availability lag of second rows
md = MonthDay(Path(DEFAULT_DATA) / date[:7] / date); s = load_seconds(md.meta['quote_parts'])
lag = (s.available_ns - s.t_ns) / 1e9
print('available_ns - t_ns (s): min %.3f median %.3f max %.3f' % (lag.min(), lag.median(), lag.max()))
print('t_ns mod 1s unique (first 5):', np.unique((s.t_ns % 1_000_000_000).to_numpy())[:5])
