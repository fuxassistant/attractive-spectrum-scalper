"""Descriptive support diagnostics, not execution or economic admission."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/telegram_labels_v1'
c = pd.read_csv(OUT / 'normalized_calls.csv')
c['ts'] = pd.to_datetime(c.archive_reference_utc, utc=True)
source = c[c.clock_role != 'public_repost'].copy()
source['sl_geometry'] = np.where(source.sl.isna() | ~source.price_ready, 'UNKNOWN',
    np.where(((source.side == 'BUY') & (source.sl < source.entry_low)) |
             ((source.side == 'SELL') & (source.sl > source.entry_high)), 'VALID', 'INVALID'))
summaries = []
for trader, group in source.groupby('trader'):
    t = group.ts
    # Reference clocks are hypotheses. Envelopes are only broad data ranges;
    # holidays, gaps, actual book support and ability to fill still need the gym.
    env = ((t >= '2026-05-03T22:00Z') & (t < '2026-05-08T21:00Z')) | ((t >= '2026-05-10T22:00Z') & (t < '2026-05-15T21:00Z'))
    hours = (t.dt.tz_convert('Europe/London').dt.hour >= 8) & (t.dt.tz_convert('America/New_York').dt.hour < 17)
    summaries.append(dict(trader=trader, calls=len(group), distinct_days=t.dt.date.nunique(),
        start=str(t.min()), end=str(t.max()), may_mbo_envelope=int(env.sum()),
        may_envelope_and_entry_hours=int((env & hours & group.price_ready).sum()),
        sl_geometry_counts={str(k):int(v) for k,v in group.sl_geometry.value_counts().items()}))
(OUT / 'baseline_support.json').write_text(json.dumps(summaries, indent=2), encoding='utf8')
source[['record_id','side','entry_low','entry_high','sl','sl_geometry']].to_csv(OUT / 'stop_geometry.csv', index=False)
a = c[(c.trader == 'T3') & (c.clock_role == 'forwarded_source')]
b = c[(c.trader == 'T3') & (c.clock_role == 'public_repost')]
x = a[['signal_id','ts']].merge(b[['signal_id','ts']], on='signal_id', suffixes=('_source','_public'))
x['repost_delay_minutes'] = (x.ts_public-x.ts_source).dt.total_seconds()/60
x.to_csv(OUT / 'T3_public_availability_delays.csv', index=False)
# This flags shared plans, not human/channel identity or plagiarism.
g = source[source.price_ready & source.ts.notna()].sort_values('ts').reset_index(drop=True)
coincidences = []
for i, r in g.iterrows():
    nearby = g[(g.trader != r.trader) & (g.ts > r.ts-pd.Timedelta(minutes=5)) & (g.ts <= r.ts) &
               (g.index < i) & (g.side == r.side) & (g.entry_low == r.entry_low) &
               (g.entry_high == r.entry_high) & (g.sl == r.sl)]
    for _, other in nearby.iterrows():
        coincidences.append(dict(first=other.record_id, second=r.record_id,
            delta_seconds=(r.ts-other.ts).total_seconds(), side=r.side, low=r.entry_low, high=r.entry_high, sl=r.sl))
pd.DataFrame(coincidences, columns=['first','second','delta_seconds','side','low','high','sl']).to_csv(OUT / 'cross_trader_plan_coincidences.csv', index=False)
print(json.dumps(dict(support=summaries, T3_delay_minutes=x.repost_delay_minutes.describe().to_dict(),
                     cross_trader_plan_coincidences=len(coincidences)), indent=2))
