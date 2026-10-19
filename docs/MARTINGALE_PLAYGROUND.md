# Martingale playground: August 2026, $1500

The later basket implementation and profitable August search are documented in
[BASKET_MARTINGALE.md](BASKET_MARTINGALE.md). This document retains the original
single-position baseline and its losing results for comparison.

Run from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 -m spectrum.playground.server
```

Open http://127.0.0.1:8794/. Choose direction rule, lookback, signal threshold,
exit plan, size multiplier and volume cap. Run on development or validation;
the separate holdout button freezes configuration, data identity and code hashes.
Saved results include all trades, series, decision traces and fill checks.
The server accepts one experiment at a time and binds only to loopback.

## Data and replay

`data/derived/playground_august_1500` contains all 21 August weekday sessions,
with causal current/prior session context when available. August 3 has no July 31
source in this release; its previous-session inputs are explicitly unknown, not
filled with invented values. Development: August 3–14 (10 days),
validation: August 17–21 (5), holdout: August 24–31 (6). Each independent split
starts at $1500. Capital and active martingale series carry between days within
the split. New entries: 08:00 Europe/London to 17:00 America/New_York
(07:00–21:00 UTC in this month).

Canonical MBO-derived quotes are shared from `full_mbo_v1`, not duplicated.
Preparation verifies raw compressed member and derived file hashes and Parquet
footers against the source audit. Source files must remain at their recorded paths.
This release adds approximately 1.02 GiB of memory arrays and execution masks.
It does not claim to complete the separate full L3 finalized-market release.

Execution validity/segments are materialized once. Runtime reads prepared masks
and projected quote columns, with a bounded two-session quote cache. It does not
recompute market diagnostics. Quote silence alone does not terminate support;
real invalid books do. Source counter resets remain feature warmup boundaries.
Mixed memory provides 900 seconds, yesterday/today M1 and opening anchors.
The event bank is explicitly empty: August events have not been certified.
Existing `spectrum.player_v3.data.Day` accepts these prepared days.

## Trading contract

Initial capital $1500, base size 1 oz; $0.30 per oz roundtrip; bid/ask fills,
100 ms latency and 2-second pending-entry timeout. This is a quote-based ounce
proxy on GC prices, not a futures margin/queue/market-impact simulator.

Default rule follows the last five minutes, threshold two ticks. Default exit:
TP $1.20/oz, SL $0.80/oz, timeout 120 seconds. Default martingale multiplier 1.5,
size cap 4 oz. Size increases after a losing trade, but the series resets only
after the complete series recovers net of costs. The series dies when marked
equity loses **strictly more than 50% of capital at the start of that series**.
Exactly 50% is not death. There is no artificial daily equity reset.

An open position encountering invalid support is censored: no invented exit,
realized PnL, or death. Such a population cannot receive a complete economic score.

## Evidence and baseline

`reports/playground_august_1500/market_audit.json`: 77,444,788 publications in
London–NY; 208 sampled M1 OHLC/volume checks passed. August 28 contains 813
invalid publications around 14:00:01.540–14:00:06.538 UTC. Independent raw
dictionary reconstruction confirms crossed books; the injected-position gym test
confirms censoring without a fictitious fill. This is normalized-source evidence,
not proof of exchange status. See `aug28_independent_book.json`.

`baseline.json` records three controls across three splits. WAIT retains $1500.
The prespecified fixed-size rule and martingale both reach series death. Default
martingale final equity: development $749.625, validation $748.2625, holdout
$748.6625. These are completed rule evaluations, not neural training or profitable
strategy evidence. Active baselines die before the invalid episode; the separate
injected-position test covers exposure to that episode.

The holdout has been observed for this baseline. Repeated adaptation against it
makes it research data; a future profitability claim needs another untouched period.

```powershell
.venv/Scripts/python.exe -X utf8 -m spectrum.playground.runner --baseline
.venv/Scripts/python.exe -X utf8 -m spectrum.playground.audit
.venv/Scripts/python.exe -X utf8 -m pytest -q
```

Code entry points: `spectrum/playground/data.py`, `runner.py`, `audit.py`,
`server.py`, `index.html`. The runner accepts a policy through the existing
`SurvivalGym` evaluation interface; the UI initially exposes reproducible rule
controls. No optimizer is represented as an already trained model.
