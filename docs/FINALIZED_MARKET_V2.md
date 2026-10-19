# Finalized market V2

The offline compiler owns reconstruction and source diagnostics. Consumers own
feature formulas and execution policies. A day-level integrity certificate is
not a claim of profitable trading or complete exchange-status information.

## Published layers

Each immutable member directory contains:

* `quotes.parquet`: every completed F_LAST publication; integer prices, sizes,
  route/source coordinates, receive availability, independent quality fields,
  segment ID, T-volume and book mutation increments, 12 price steps per side.
* `l3.parquet`: original action facts, original clocks and source order, accepted
  application result and resulting order state. Every row becomes available at
  its completed publication, never at a partial message's timestamp.
* compressed L3 checkpoints after snapshots, at 30-minute observed intervals,
  and at the final complete publication. A checkpoint describes order state;
  restoring an arbitrary later time applies stored resulting mutations.
* `seconds.parquet`: committed midpoint/depth/flow samples. Quote age is retained
  without a hidden 5-second or 30-second invalidation rule. A second is published
  only when a raw receive watermark proves the preceding observations complete.
* `bars.parquet`: exact completed-publication midpoint OHLC, **not sampled-second
  OHLC**. Buckets are [start,end); the inherited opening book is included. Volume
  includes T only, including unknown side. Partial EOF buckets are not fabricated.
* `states.json`, `segments.json`, `gaps.json`, `rates.parquet`, `diagnostics.json`:
  reasons, source coordinates, measured gaps, wall-second counts including zeros,
  exact-count bounded-bin quantiles and time-weighted/second-sampled diagnostics.
* `acceptance.json`: all-publication comparison to the canonical capture plus a
  bounded independent dictionary-book prefix comparison, run before publication.
* `manifest.json`: source identities, implementation hashes, columnar block index,
  payload sizes/hashes, checkpoint index and certificate identity.

Prices are USD per troy ounce in integer nanounits. Sizes are native GC contracts,
not the simulated strategy's ounces. The first 12 depth entries are **price
steps**, including empty steps, not the first 12 occupied levels.

Missing spot reference, calendar convention, source quality and book support are
independent. New York 17:00–18:00 and weekends are regular schedule conventions,
not evidence of a particular historical exchange halt. Long observed silence is
not automatically packet loss. Selected-channel sequence jumps are not packet
loss counts. Day-level diagnostics contain hindsight and are excluded from X.

## Commands

Run the two pilots before a full release:

```powershell
.venv/Scripts/python.exe -m spectrum finalize --output data/derived/finalized_market_v2_r2 --members member_010 member_011 --workers 2
.venv/Scripts/python.exe -m spectrum finalize --output data/derived/finalized_market_v2_r2 --workers 6
.venv/Scripts/python.exe -m spectrum verify-finalized data/derived/finalized_market_v2_r2/member_011
.venv/Scripts/python.exe -m spectrum replay-finalized data/derived/finalized_market_v2_r2/member_011 --rows 3
.venv/Scripts/python.exe -m spectrum.finalized_audit --market data/derived/finalized_market_v2_r2 --output reports/finalized_market_v2/acceptance_r2
```

The full source inventory comprises 27 members dated 2026-08-02 through 2026-09-01;
it does not contain all September. The existing causal contract selection is
retained. Current members start with explicit resets. A member without a reset
requires an explicit predecessor engine checkpoint; the compiler refuses to
invent state or silently reset at a date/file boundary. `DayBuilder` accepts an
initial state for that case; the current daily-archive CLI does not infer one.

The scheduler uses at most six workers and reserves 2 GiB of available RAM per
worker. A whole-run disk budget uses measured pilot bytes per record, a 30%
margin, and an 8 GiB free-space reserve. Each writer also checks the reserve at
block flushes. Complete outputs are reused only after input/code identity and
explicit administrative verification. Failed `.building` directories retain
their error receipts and are not silently overwritten. No old dataset is deleted.

## Reader and laboratory

```python
from spectrum.finalized import FinalizedReplay, FinalizedSession, restore_l3

r = FinalizedReplay(day_directory, columns=["available_ns", "bid_nano", "ask_nano"])
for batch in r.batches():
    use_numpy(batch["bid_nano"].to_numpy())
cursor = r.checkpoint()
r = FinalizedReplay.restore(day_directory, cursor)
session = FinalizedSession(day_directory)
features = session.features()  # explicit offline feature computation
orders = restore_l3(day_directory, available_ns)
```

Sequential reads are O(N) and retain a bounded row group. Seeking uses the
precomputed group index; restoring a cursor never replays the prefix. `advance`
returns a last observation with an age, not a new observed quote or an implicit
execution guarantee. A projected time window can include only requested columns.
`read_window` materializes the requested range; large consumers should use batches.

Open checks publication status, contract version and payload size. It trusts the
immutable published content. Full hashing belongs to `verify-finalized`, after
copying or suspected corruption, not every training episode. Silent same-size
disk corruption is not detected by the fast path. Keep publications immutable.

## Gym adapter

```powershell
.venv/Scripts/python.exe -m spectrum.player_v3.finalized_data --market data/derived/finalized_market_v2_r2 --date 2026-08-03 --output data/derived/finalized_gym_v2/2026-08-03
```

Load the result with `FinalizedDay` from that module and pass it to `SurvivalGym`.
Preparation explicitly computes feature families and freezes arrays, joins,
session bounds, support segments and memory indices. Episode loading only maps
arrays; full array hash verification is optional and administrative.

The adapter retains 15-minute second memory, current/previous available session
M1 and opening anchors through the existing memory interfaces. Missing prior
data remains missing. The event bank is explicitly empty until a separately
certified causal event job is available for the new corpus; May events are never
relabelled as August events. Daily audit statistics do not enter model inputs.

For `FINALIZED_SEGMENTS_V2`, silence alone does not stop the gym. Unsupported
observations stop an open-position trajectory with the position and last
supported equity retained. They never create a recovery fill, realized PnL,
barrier death or special terminal reward. Legacy inputs retain legacy behavior
for reproducibility; no old checkpoint or training run is relabelled as repaired.

## Acceptance

Synthetic tests cover intra-second extremes, exact volume, atomic publication,
pending-group depth, invalid-book prices, unknown trade side, long pauses,
missing reference, DST, seek/cursor restoration and explicit corruption checks.
Real-data acceptance compares every publication and valid BBO against the
canonical source, checks an independent raw prefix and checkpoint restoration.
The latter checks have explicit scope: they are not a second full-history L3
implementation. Benchmarking uses the same nine quote columns, five runs and
half/full-size scaling. The first observed run is shown separately; no forced
cold OS cache is claimed. The target is at least 3x the legacy dictionary reader.
