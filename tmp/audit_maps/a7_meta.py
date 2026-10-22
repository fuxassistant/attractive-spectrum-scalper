import numpy as np
from spectrum.cognition import race as rc
from spectrum.cognition.hotpath import base_day
for date in ('2025-10-15', '2025-10-31', '2025-11-03', '2025-12-15'):
    b, m = base_day(date)
    st = m['start_ns'] / 1e9
    import datetime as dt
    f = lambda s: dt.datetime.utcfromtimestamp(st + s).strftime('%m-%d %H:%M')
    k = np.arange(m['london'], m['ny_close']); fin = np.isfinite(b['mid'])
    segs = np.flatnonzero(b['seg'][1:] != b['seg'][:-1]) + 1
    print(date, 'start', f(0), 'london', f(m['london']), 'ny_close', f(m['ny_close']), 'end', f(m['n']), 'seg changes', [f(s) for s in segs][:6],
          'session NaN mid', int((~fin[k]).sum()), 'sigma unknown in session', int((~b['sigma_known'][k]).sum()))
