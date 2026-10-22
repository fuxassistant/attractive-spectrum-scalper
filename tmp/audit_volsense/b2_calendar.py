import numpy as np, pandas as pd, json, glob
from spectrum.cognition.hotpath import base_day
from spectrum.cognition import volsense as vs
man = json.load(open('data/derived/volsense_v1/manifest.json'))['days']
rows = []
for d in man:
    p = pd.read_parquet(f'data/derived/volsense_v1/real/{d}.parquet', columns=['t_ns', 'sec_london', 'mid', 'sigma600', 'close_min', 'ev_us_data', 'ev_lbma_am', 'y' if False else 'RL12', 'RS12', 'RL40', 'RS40', 'H40'])
    t = pd.to_datetime(p.t_ns, utc=True)
    last = p.tail(720)  # last hour
    rows.append(dict(date=d, lon=t.iloc[0].strftime('%H:%M'), end=t.iloc[-1].strftime('%H:%M'), n=len(p),
                     usdata=(t.iloc[0] + pd.Timedelta(minutes=float(-p.ev_us_data.iloc[0]) if abs(p.ev_us_data.iloc[0]) < 120 else 0)).strftime('%H:%M'),
                     frozen_last_hour=float((np.diff(last.mid) == 0).mean()), sig_last=float(last.sigma600.median()),
                     y40_last=float(np.nanmean(np.fmax(last.RL40, last.RS40) >= 2)), nan40_last=float(np.mean(~np.isfinite(np.fmax(last.RL40, last.RS40))))))
r = pd.DataFrame(rows)
print(r.groupby(['lon', 'end']).size())
print(r[r.frozen_last_hour > .5].to_string())
print(r.sort_values('sig_last').head(8).to_string())
# DST window March 2026 and late Oct/early Nov
print(r[r.date.between('2026-03-06', '2026-03-31') | r.date.between('2025-10-22', '2025-11-05')][['date', 'lon', 'end', 'n']].to_string())
odd = r[~r.end.isin(['20:59', '21:59']) | (r.lon == '13:00')]
print(odd[['date', 'lon', 'end', 'n', 'frozen_last_hour', 'sig_last']].to_string())
print(r[r.date.isin(['2026-07-02', '2026-07-03', '2026-07-06', '2026-06-19', '2026-05-25', '2026-01-19', '2025-11-28', '2025-12-24', '2025-12-26', '2025-12-31', '2026-01-02', '2026-04-03'])][['date', 'lon', 'end', 'n', 'frozen_last_hour', 'sig_last']].to_string())
# frozen share over whole session per day, flag > 20%
fz = []
for d in man:
    p = pd.read_parquet(f'data/derived/volsense_v1/real/{d}.parquet', columns=['mid', 'sigma600', 'RL12', 'RS12'])
    run = (np.diff(p.mid.to_numpy()) == 0)
    # 5-second rows: share of rows where mid unchanged over 10 minutes
    m10 = p.mid.to_numpy(); same = np.r_[np.zeros(120, bool), m10[120:] == m10[:-120]]
    fz.append((d, float(same.mean()), float((p.sigma600 < .3).mean())))
fz = pd.DataFrame(fz, columns=['date', 'flat10min', 'sig_lt_0.3'])
print(fz[(fz.flat10min > .02) | (fz['sig_lt_0.3'] > .02)].to_string())
