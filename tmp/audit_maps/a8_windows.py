import numpy as np, datetime as dt
from spectrum.cognition import race as rc, bands as bd
from spectrum.cognition.hotpath import base_day
date = '2025-11-18'
b, m = base_day(date); mid, seg = b['mid'], b['seg']
k = np.arange(m['london'], m['ny_close']); k = k[b['sigma_known'][k] & np.isfinite(mid[k]) & (b['sigma'][k] > 0)]
sig = b['sigma'][k]; st = m['start_ns'] / 1e9
T = lambda i: dt.datetime.fromtimestamp(st + i, dt.timezone.utc).strftime('%H:%M:%S')
mv = mid[k[:-600] + 600] - mid[k[:-600]]
for lab, i0 in (('max rise 10m', int(np.nanargmax(mv))), ('max fall 10m', int(np.nanargmin(mv)))):
    print(lab, T(k[i0]), 'move', mv[i0], 'ticks; sigma', sig[i0].round(2))
    sel = np.arange(max(0, i0 - 300), min(len(k), i0 + 900), 60)
    r18 = rc.race(mid, seg, k[sel], sig[sel], 18); r27 = rc.race(mid, seg, k[sel], sig[sel], 27)
    e18 = rc.excursion(mid, seg, k[sel], sig[sel], 18)
    for j, i in enumerate(sel):
        print(' ', T(k[i]), f'${mid[k[i]]/10:.1f}', 'sig', sig[i].round(2), 'G18 w', r18['winner'][j], 'tw', r18['t_win'][j], 'H', r18['H'][j], '| G27 w', r27['winner'][j],
              f'| R18 L {e18["R_long"][j]:.2f} S {e18["R_short"][j]:.2f}', 'mid+H18', f'{(mid[k[i]+r18["H"][j]]-mid[k[i]]) if k[i]+r18["H"][j] < len(mid) else np.nan:+.0f}')
# quiet window: lowest sigma hour
h = np.convolve(sig, np.ones(3600) / 3600, 'valid'); iq = int(np.argmin(h))
sel = np.arange(iq, iq + 3600)
for G in (18, 40):
    e = rc.excursion(mid, seg, k[sel], sig[sel], G); R = np.fmax(e['R_long'], e['R_short']); r = rc.race(mid, seg, k[sel], sig[sel], G)
    print(f'quiet hour from {T(k[iq])}: sigma {h[iq]:.2f}, range ${(np.nanmax(mid[k[sel]])-np.nanmin(mid[k[sel]]))/10:.1f}; G{G}: opp R>=2 {np.nanmean(R>=2):.3f}, race opp>0 {np.nanmean(r["opp"]>0):.3f}, H median {np.median(r["H"])}')
