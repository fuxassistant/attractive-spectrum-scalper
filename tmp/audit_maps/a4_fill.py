import numpy as np
from spectrum.cognition import race as rc
from spectrum.cognition.hotpath import base_day
for date in ('2025-11-18', '2025-11-20'):
    b, m = base_day(date); mid, seg = b['mid'], b['seg']
    k = np.arange(m['london'], m['ny_close']); k = k[b['sigma_known'][k] & np.isfinite(mid[k]) & (b['sigma'][k] > 0)]
    sig = b['sigma'][k]
    null = rc.sign_flipped(mid, seg, np.random.default_rng(3))
    for G in rc.GRID_G:
        need = (G + rc.C_LOSS) / (2 * G - rc.C_WIN + rc.C_LOSS)
        out = []
        for path in (mid, null):
            r = rc.race(path, seg, k, sig, G); w = r['winner']; res = np.isin(w, (1, -1))
            pw = np.mean(w[res] == 1)
            filled_win = (w == 1) & (r['mae'] >= 1.5); loss = w == -1
            out.append((pw, filled_win.sum() / (filled_win.sum() + loss.sum())))
        print(date, f'G{G} need {need:.3f} | real: P(long|resolved) {out[0][0]:.3f}, with fill proxy {out[0][1]:.3f} | null: {out[1][0]:.3f}, {out[1][1]:.3f}')
