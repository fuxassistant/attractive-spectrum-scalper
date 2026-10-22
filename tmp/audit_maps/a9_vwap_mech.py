import numpy as np
from spectrum.cognition import race as rc, bands as bd
from spectrum.cognition.hotpath import base_day
for date in ('2025-11-18', '2025-11-20'):
    b, m = base_day(date); mid, seg = b['mid'], b['seg']
    k = np.arange(m['london'], m['ny_close']); k = k[b['sigma_known'][k] & np.isfinite(mid[k]) & (b['sigma'][k] > 0)]
    vol = b['trade_buy'] + b['trade_sell'] + b['trade_unknown']
    ad = np.abs(np.nan_to_num(np.diff(mid, prepend=np.nan)))
    out = []
    for s in range(3):
        null = rc.sign_flipped(mid, seg, np.random.default_rng(100 + s))
        for wn, vv in (('vol', vol), ('vol_lead1', np.r_[vol[1:], 0]), ('absd', ad), ('absd_lag1', np.r_[0, ad[:-1]])):
            for name, p in (('real', mid), ('null', null)):
                if name == 'real' and s > 0: continue
                vw, half = bd.vwap_bands(p, vv, windows=(600,))[600]
                out.append((wn, name, s, np.nanmean(np.abs((p[k] - vw[k]) / half[k]))))
    for wn in ('vol', 'vol_lead1', 'absd', 'absd_lag1'):
        r = [x[3] for x in out if x[0] == wn and x[1] == 'real']; n = [x[3] for x in out if x[0] == wn and x[1] == 'null']
        print(date, wn, 'real |z| %.3f  null %s' % (r[0], np.round(n, 3)))
