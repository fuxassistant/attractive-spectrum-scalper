import numpy as np
from spectrum.cognition import race as rc, bands as bd
from spectrum.cognition.hotpath import base_day
for date in ('2025-11-18', '2025-11-20'):
    b, m = base_day(date); mid, seg = b['mid'], b['seg']
    k = np.arange(m['london'], m['ny_close']); k = k[b['sigma_known'][k] & np.isfinite(mid[k]) & (b['sigma'][k] > 0)]
    null = rc.sign_flipped(mid, seg, np.random.default_rng(51 * 100000 + int(date.replace('-', '')) % 100000))
    vol = b['trade_buy'] + b['trade_sell'] + b['trade_unknown']
    # volume shifted by 1 hour (keeps its own clustering, breaks coupling with the path)
    vshift = np.roll(vol, 3600)
    for name, p in (('real', mid), ('null', null)):
        row = []
        for wn, vv in (('vol', vol), ('flat', np.ones_like(vol)), ('vol_roll1h', vshift)):
            bb = bd.vwap_bands(p, vv); vw, half = bb[600]; f = bd.future(p, seg, k, vw[k], half[k], 150)
            row.append(f'{wn}: exit {np.nanmean(f["E"] != 0):.3f} |z| {np.nanmean(np.abs((p[k]-vw[k])/half[k])):.3f}')
        print(date, name, ' | '.join(row))
    # correlation of volume with |d| and lagged |d|
    d = np.abs(np.diff(mid, prepend=np.nan)); ok = np.isfinite(d)
    print('  corr(vol_t, |d_t|)', np.corrcoef(vol[ok], d[ok])[0, 1].round(3), ' corr(vol_t, |d_t+1|)', np.corrcoef(vol[ok][:-1], d[ok][1:])[0, 1].round(3),
          ' corr(vol_t, |d_t-1|)', np.corrcoef(vol[ok][1:], d[ok][:-1])[0, 1].round(3))
