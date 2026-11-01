# Feature week V2

Bounded GCZ6 feature/representation readiness audit, 2026-08-23 22:00 through
2026-08-28 21:00 UTC. This is technical and descriptive admission, not a trading
or full-history training certificate. The original V1 receipts/data are frozen.

## Source and calculation contracts

- Canonical COMPLETE captures and exact T/interaction sidecars are checked
  against payload and partition hashes. The fixed route is `(1, 41271, 22)`.
- A UTC file snapshot is joined only after full order-ID/side/price/size identity
  and independent replay of the next 65-second prefix. Genuine support gaps and
  session closures are never joined. The next real publication releases the
  missing endpoint; no artificial EOF flush is used.
- Daily scalar jobs read 3606 seconds of left history. The seven families yield
  480 values and 480 masks. Original strict signed-flow spectra remain optional.
- Missing sparse-ledger tails and the collector's first two seconds of refill
  history are masked. Exact T volume and the canonical book have separate masks.
- Completed native BB: SMA20, k=2, sample SD (`ddof=1`). The crossing bank's
  developing quote geometry is a different, unchanged clock/contract.
- Prior context: explicit previous CME trading date, previous and current
  sessions, physical closed/missing rows preserved. Prior anchors admit 99%
  supported-minute coverage and expose quality and strict completeness. These
  are observed supported-minute levels, not guaranteed full-session extrema.
- All queries use `available_ns`; native invalid quotes are never skipped in
  favor of older valid quotes. Raw prices retain float64 before normalization.

## Reproduction

Use the pinned CPython 3.12.14 runtime and project lock. Paths below are relative
to the checkout; commands write only new directories. `prepare` is already
materialized locally; an existing destination deliberately fails.

```powershell
.tmp/baseline/venv/Scripts/python.exe experiments/feature_week_v2/context_source.py
.tmp/baseline/venv/Scripts/python.exe experiments/feature_week_v2/build.py prepare
.tmp/baseline/venv/Scripts/python.exe experiments/feature_week_v2/build.py run --name final_parallel2 --workers 2
.tmp/baseline/venv/Scripts/python.exe experiments/feature_week_v2/build.py run --name benchmark_serial --workers 1 --days-only
.tmp/baseline/venv/Scripts/python.exe experiments/feature_week_v2/audit.py
.venv/Scripts/python.exe experiments/feature_week_v2/plot.py
.tmp/baseline/venv/Scripts/python.exe experiments/feature_week_v2/report.py
```

`report.py` finalizes an audit only after `visual_review.json`, `validation.json`
and `installed_package.json` have been recorded from actual verification. Do
not fabricate those receipts when reproducing. The installed-package check is
reproducible with `verify_install.py --root ABS_REPO --site ABS_INSTALLED_TARGET
--wheel ABS_WHEEL`, invoked using the pinned Python `-I` from a directory outside
the package source. It rejects wheels with extra/missing runtime files.

The local plotting environment has Matplotlib; it is used only for rendering
verified payloads. Runtime formulas use the pinned environment. Raw source
archives are private local dependencies, not packaged or published.

The report and admission receipt are in
`data/derived/feature_week_v2/audit/`. The final COMPLETE session dataset is
`data/derived/feature_week_v2/final_parallel2/`. Initial `parallel2` and
`serial_final` attempts failed the minute dependency-clock check and are not
admitted. Their manifests remain FAILED. `benchmark_serial` is a deliberate
day-only benchmark, not an independently complete session-store dataset.

`audit.py` checks arithmetic against raw T prints, independent DFT windows,
completed BB against core, overlap/restart, future-prefix invariance, clocks,
physical coverage, native quote lookup, real invalid-book handling, and actual
model snapshots. `plot.py` emits eight standalone scientific figures. The
same-price examples use only historical activity and do not select on returns.

## Actual model observation

```python
import pandas as pd
from spectrum.representation import open_session_store

store = open_session_store(
    "data/derived/feature_week_v2/final_parallel2/stores/2026-08-24"
)
observation = store.observe(pd.Timestamp("2026-08-24 02:30", tz="UTC").value)
# observation.market: native BBO validity, publication, age, source ordinal.
# observation.fast/context: raw values, normalized values, masks, clocks.
# observation.context['quality']: day/combined/prior coverage and completeness.
# observation.second_features/minute_features: latest published values + masks.
```

Do not intersect every feature mask. Strict signed spectra require no unknown
aggressors, whereas a meaningful spectrum of the unknown-side channel requires
that channel to vary. Complete-case selection therefore admits zero rows in
this full diagnostic profile. The current native market-state gate remains
independent from the availability of historical observations.

## Measured result

414,000 expected seconds, none missing, 413,995 price-supported. Real unsupported
endpoints: Friday 14:00:02–14:00:06 UTC. Exact tape: 579,530 T records and 841,550
contracts over the complete venue week. Interaction support: 413,974 seconds.

Daily stage: one worker 82.574 s; two workers 52.369 s; 16 payloads bitwise equal.
Context/store assembly: 88.057 s. Complete two-worker materialization: 140.479 s.
These numbers start from available canonical/raw sidecars. Additional raw
Thursday extraction took 415.769 s and handoff preparation 13.075 s separately.
Do not call 140 s the cost of parsing a fresh week of MBO from scratch.

600 exact-T VWAP windows, 900 DFT windows, 661 independent completed BB windows,
13,833 core/representation BB pairs, 4.32 million restart/partition comparisons,
11 actual as-of snapshots, and one full L3 book snapshot were checked.
The full suite passed 337 tests. A clean 0.4.0 wheel was imported outside the
source tree and its actual session loader checked; its 110 runtime files
exactly match the source inventory. A stale `build/lib` initially introduced
deleted legacy modules; that failed wheel was rejected and the build directory
preserved under `.tmp/pre_v2_build_20261029` before a clean rebuild.

## Version evidence

`docs/EVOLUTION_0_4_0.json` binds the current runtime, prior runtime receipt and
historical certificate. The historical 0.3.0 pyproject bytes are preserved in
`baselines/0.4.0/parent-pyproject.toml`; old validation hashes still validate
their original version. No old feature-equivalence certificate is relabeled as
a certificate for the new formulas or data.
