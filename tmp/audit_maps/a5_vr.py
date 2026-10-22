"""Is 'price sticks to VWAP' real mean reversion? VR of the test week vs null; time-weighted vs volume-weighted |z|."""
import numpy as np
from spectrum.cognition import race as rc, bands as bd
from spectrum.cognition.hotpath import base_day
DAYS = ('2025-11-17', '2025-11-18', '2025-11-19', '2025-11-20', '2025-11-21')
acc = {}
for date in DAYS:
    b, m = base_day(date); mid, seg = b['mid'], b['seg']
    k = np.arange(m['london'], m['ny_close']); k = k[b['sigma_known'][k] & np.isfinite(mid[k]) & (b['sigma'][k] > 0)]
    null = rc.sign_flipped(mid, seg, np.random.default_rng(51 * 100000 + int(date.replace('-', '')) % 100000))
    vol = b['trade_buy'] + b['trade_sell'] + b['trade_unknown']
    for name, p in (('real', mid), ('null', null)):
        d = np.diff(p); d = d[np.isfinite(d)]; s2 = np.mean(d ** 2)
        for h in (60, 600, 1800):
            x = p[k[k + h < len(p)] + h] - p[k[k + h < len(p)]]; x = x[np.isfinite(x)]
            acc.setdefault((name, 'VR', h), []).append(np.mean(x ** 2) / (h * s2))
        for wname, vv in (('vol', vol), ('flat', np.ones_like(vol))):
            bb = bd.vwap_bands(p, vv)
            for W in (240, 2800):
                vw, half = bb[W]; z = (p[k] - vw[k]) / half[k]
                acc.setdefault((name, wname + '_absz', W), []).append(np.nanmean(np.abs(z)))
                acc.setdefault((name, wname + '_half', W), []).append(np.nanmedian(half[k]))
for key in sorted(set((a, b_) for a, b_, c in acc)):
    pass
for key, v in sorted(acc.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
    print(key, np.round(v, 3), 'mean', np.round(np.mean(v), 3))
