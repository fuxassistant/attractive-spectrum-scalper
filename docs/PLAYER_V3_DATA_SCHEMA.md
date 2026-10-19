# Mixed-memory input contract

`spectrum.player_v3.data` prepares a separate, hash-verified adapter over the
completed full MBO publication. It does not alter reconstructed sources.

## Clocks and sessions

The nominal session runs from 18:00 America/New_York on the preceding calendar
date to 17:00 on the named date. Both boundaries follow timezone DST. The observed
OHLC minute calendar masks actual absent observations inside that interval;
absence is not asserted to prove an exchange closure. Trading is permitted only
after 08:00 Europe/London, before the configured New York close, after contract
selection becomes available, and on fresh, admitted completed-minute origins.

The full-history source consists of UTC daily captures. Their evening/current
parts are joined without filling absent seconds or minutes. Source epochs are
made distinct across captures; source ordinals remain in a separate audit array.
An adapter-local strictly increasing ordinal orders the replay arrays. Quotes,
integer nanodollar prices, publication timestamps and source timestamps are
otherwise retained for the existing exact BBO simulator.

Previous session means the preceding observed weekday with at least 60 supported
reference minutes during London–New York. It must itself be admitted and use the
same selected contract. Missing or excluded prior MBO data is masked; the adapter
does not silently replace it with an older day. A Monday normally receives
Friday. The April 13 pilot starts with previous April 10 unavailable.

## Stored branches

| Array | Fields | Interpretation |
|---|---:|---|
| `x` | 71 | Existing baseline: 11 candle/time fields, 30 selected features, 30 known masks |
| `fast_x` | 64 | 20 existing values and 12 VWAP-proxy values, followed by 32 known masks |
| `slow_x` | 71 | Previous and current completed-minute records, with separate `slow_session` 0/1 identity |
| `anchors` | 38 | 19 quantities and their 19 known masks |
| `event_x` | 25 | Explicit allowlist of causal event quantities and category indicators |

Fast VWAP values are z, return RMS and volume rate at 60/300/900/3600 seconds.
They use second midpoint weighted by complete T volume, including unknown side.
They are explicitly a proxy, not an exact trade-price VWAP. Legacy signed-flow
spectral masks remain as published; their unknown masks do not remove the
window or the unknown trade-volume field.

`Day.memory(i)` returns at most 900 actual seconds, 2,760 minutes and 256 events.
`memory_bounds(i)` returns `(fast_left, fast_right, slow_left, slow_right,
event_left, event_right)` for batching. The final endpoints are gated by
publication time as well as the nominal clock. A prefix maximum prevents late
proofs from being joined back into a previously visible recurrent sequence.

The fast prefix stops at the latest invalid second, missing second or epoch
change. `fast_segment_start`, `fast_valid` and `fast_epoch` expose this boundary.
No state should be propagated across it. The fast window is never materialized
900 times on disk; it is a slice of one stored tape.

The prior slow day includes its final completed minute if reconstructed. The
current slow branch contains only minute prefixes available at each decision.
Its elapsed-minute field preserves missing-minute gaps.

## Openings, landmarks and events

Opening anchors are the first supported **second midpoint** at or after the
session/London boundary, with a maximum 60-second delay. They become known only
when that second is published. They are not asserted to be the first individual
exchange print or a broker opening fill. Delay and known fields are explicit.

Previous close is the last supported second midpoint. Previous extrema come from
completed M1 bars and are masked unless at least 80% of the nominal prior-session
minutes are valid. Current extrema are expanding prefixes, never final-day
extrema. Price levels are converted to current-price differences in ticks.

The event source is `event_landmarks_v1/run_003/causal_events.parquet`, whose
recognition protocol currently covers London–New York only. It does not provide
an overnight event bank. The reader uses an explicit allowlist and never reads
the outcomes file, retrospective isolation, future recurrence or PnL as X.
`Day.memory` adds three dynamic quantities to the stored 25 event fields: age in
minutes, distance from current midpoint to event start, and distance to event end.
Thus the model receives 28 fields per event. Available and truncated counts are
explicit anchors. The bank contains only the current and admitted previous
same-contract session, with publication gating at each decision.

## Pilot and validation

The initial independent adapter is `data/research/player_v3/memory_001`: 30 dates
from April 13 to May 22, 2026. It occupies 6.107 GiB. Initial generation took
97.85 seconds; a subsequent small-array refinement included the terminal prior
minute, without changing exact BBO or fast tapes. Verification took 10.72 seconds
and checked every saved file hash, finite model arrays, action-time boundaries,
and sampled causal/continuity/memory-size constraints on every day. Six focused
data tests cover delayed publication, DST/weekends, event allowlisting, future
mutation, availability, openings and fast support resets.

This is a technical input contract. It does not establish learned memory use,
economic usefulness, or independent evaluation status; those require the player
experiment and separate behavioral audit.
