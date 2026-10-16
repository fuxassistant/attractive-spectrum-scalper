# Full MBO generation v1

Run: `.venv/Scripts/python.exe -X utf8 -m spectrum.full_history`.
The immutable configuration is `data/derived/full_mbo_v1/run_config.json`.
Re-running the command resumes verified completed captures; a live owner blocks duplicate writers.

## Scope and parallelism

All 37 MBO members in the two source ZIPs are scanned, including August and September 1.
The first archive begins October 1, 2025. July 31, 2026 is absent from the supplied source ranges and remains an explicit gap.
The materialized trading representation uses one selected GC outright route per UTC date, not every illiquid contract or spread in the parent-symbol archive.
Six reconstruction workers share a single queue across months and weeks. Two feature workers consume completed captures concurrently. One source planner streams and hashes the archives, staging at most one member ahead of the reconstruction queue.

## Contract and availability rules

For each date, select the GC outright with greatest trade volume in complete F_LAST groups received before 08:00 Europe/London; complete record count breaks ties. The selection is actionable only after that cutoff. If no morning evidence exists, a full-day activity selection is stored for context only and is never admitted for trading.
Instrument identifiers are resolved using the DBN metadata validity intervals. Routes and price series remain separate at contract changes. Reconstruction starts only at explicit source resets; no synthetic midnight reset or price splice is allowed.

The reference calendar is the exact UTC minute index observed in the supplied cTrader OHLC. Missing minutes, weekends and daily pauses are not interpolated. Absence is recorded as unavailable reference support, not proof of exchange closure. Degraded or unknown source days have their admissible masks disabled. This conservative reference is spot XAUUSD and can differ from the GC exchange calendar.

Features: 182 motion, flow and spectral values with individual known masks, calculated by the existing bounded FeatureStream adapter. Entries additionally require routing availability, London opening and a time before 17:00 America/New_York. These are dataset admission masks, not a trained trading policy.

## Disk, integrity and recovery

The configured floor is 12 GB free (decimal), plus 6 GB reserved for in-flight output when admitting reconstruction or staging. Staging also estimates its selected raw size before writing. Workers check free space during writing. A reserve violation stops generation safely; persistent insufficient admission space appears as WAITING_FOR_SPACE. No original archive or previous dataset is removed.

Generated staging binaries are retired only after payload hashes, Parquet feature readback, sampled independent quote-to-OHLC reconstruction and a durable receipt are verified. Source member SHA, raw partition SHA, code identity, contract choice and output hashes remain recorded. Interrupted outputs are retained under an interrupted suffix; they may consume additional space on retries.

Live status: `data/derived/full_mbo_v1/status.json`.
Completed captures: `catalog.partial.json` and `receipts/`.
Source planner: `staging/member_NNN/progress.json`.
Book progress: `market/member_NNN/capture_NNNNNN/progress.json`.
Final catalog and week indexes are published only after all work completes.
Logs: `reports/full_mbo_v1/stdout.log` and `stderr.log`.

## Empty-capture repair and resume, October 16

The first run stopped on April 3, 2026: 574 records and one committed quote produced no finalized seconds. Such captures now receive an EXCLUDED manifest and receipt with NO_FINALIZED_SECONDS, source counts and hashes; no artificial second or empty feature schema is generated. A mismatch between a zero-row payload and a nonzero manifest remains a hard failure. Corruption and other exceptions are not silently skipped.

The explicitly recorded orchestration migration accepts the previous full_history.py identity only while every other recorded code hash remains identical. Original market/feature manifests and receipts remain unchanged. Resume verifies their payload hashes, reuses market captures and finished feature jobs, and only calculates unfinished work. The historical errors.json is appended to rather than reset. Migration evidence and restart logs are under `reports/full_mbo_v1/resume_20261016/`.

`training_catalog.json` records technical input eligibility and every exclusion reason separately from generation completion. It excludes captures without finalized seconds, usable entry rows or OHLC support, context-only contract choices, unavailable source days, unverified OHLC QA and less than 80% admitted seconds among observed-reference seconds. This last fraction does not prove full session coverage; timestamps and gaps must remain explicit. Consumers must still apply entry_allowed and the known masks of their required features, and split sequences at gaps, epochs and contract changes. The catalog is PARTIAL until the entire source queue finishes; it is not a claim that labels, chronological evaluation splits or a profitable policy are ready.

For a stop, identify the owner PID and creation time in owner.json and stop only that owned process tree. Do not terminate unrelated Python sessions. Resume with the same command and frozen implementation. Code changes require a separate versioned output, not mixed feature formulas.
