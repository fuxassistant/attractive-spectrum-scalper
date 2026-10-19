# Basket martingale: reproducible August experiment

The basket implementation is `spectrum/playground/basket.py`. It replaces the
single-position sizing example for this experiment; the old baseline is retained
for comparison. Initial capital is $1500 and initial order size is 1 oz.

## Run

From the repository root, evaluate the frozen candidate into a new directory:

```powershell
.venv/Scripts/python.exe -X utf8 -m spectrum.playground.basket_run --output reports/basket_replay_new
```

Use `--split train`, `validation`, `holdout`, or `month`. `--recipe path.json`
accepts the saved frozen recipe or a JSON object with BasketConfig fields.
Changing this recipe defines a new experiment. It does not update old results.

Repeat the seeded population search into a new directory:

```powershell
.venv/Scripts/python.exe -X utf8 -m spectrum.playground.basket_search --n 320 --output reports/basket_search_new
.venv/Scripts/python.exe -X utf8 -m spectrum.playground.basket_audit --output reports/basket_search_new
.venv/Scripts/python.exe -X utf8 -m spectrum.playground.basket_report --output reports/basket_search_new
```

The portable interactive report is `reports/basket_search_august_1500/index.html`.
Serve it locally with:

```powershell
.venv/Scripts/python.exe -m http.server 8795 --bind 127.0.0.1 --directory reports/basket_search_august_1500
```

## Contract

Entries and adds are allowed only between 08:00 London and 17:00 New York.
Orders execute at the first supported quote strictly after 100 ms latency and
expire after 2 seconds. Entry pays ask, exit bid for longs; the reverse for shorts.
Commission is $0.30 per oz roundtrip, reserved on entry. No fills occur at an
ideal grid level. Adds triggered by entry-side prices avoid treating the spread
as a price decline. Every exposed quote is marked at the liquidation side.

A basket has one direction, accumulated quantity and weighted entry cost.
The basket target covers all reserved commissions. All positions close before
the daily data boundary; capital and unrecovered series persist between days.
Series capital is frozen before its first fill. Strictly more than 50% loss
relative to that base kills the population. A basket stop or partial recovery
does not reset the series. Unsupported observations censor an exposed population;
they never fabricate exits, profit, or death. Flat unknown intervals can be skipped.
This remains an ounce quote proxy, without exchange margin, queue or impact.

## Causal inputs and frozen candidate

Inputs use only available completed M1 candles. Price is represented by cumulative
close changes. The range scale is sqrt(5) times the mean M1 high-low range over
60 minutes, with a $0.50 floor; it is not Wilder ATR. There is no backfill.
Signals require at least 120 contiguous known minutes. ER and relative range
filters are available; the selected thresholds are broad (1 and 10).

Selected candidate #8 buys a decline below the 15-minute mean by 2 range scales.
The range scale freezes at the entry decision. First add distance is 1 scale;
subsequent distance doubles. Orders are 1, 1.5, 2.25 oz, at most 3 orders and
8 oz configured cap (the actual attainable maximum here is 4.75 oz).
Target net profit is 8 scales times basket quantity. A basket stops at 10% of
its initial capital, times out at 360 minutes, or closes at the session boundary.
After closure, a 30-minute cooldown applies. Stop/target/timeout intents also
have execution latency; the series death barrier uses immediate marked equity.

## Selection and measured outcome

Seed 61018 generated 320 configurations. The top 24 from August 3–14 were
evaluated on August 17–21. Selection maximized the worse of train/validation
scores, log growth minus 0.5 times maximum series loss, requiring at least three
closed baskets, survival and complete observations. Positive outcomes in both
parts took precedence. Candidate #8 was frozen before August 24–31 evaluation.
The whole August market had already been studied, so this is not a pristine
blind holdout. Monthly continuity was evaluated separately after selection.

| Run | Start | End | Net |
|---|---:|---:|---:|
| Development, 10 days | 1500 | 1729.05 | 229.05 |
| Validation, 5 days | 1500 | 1756.775 | 256.775 |
| Later 6 days | 1500 | 1560.475 | 60.475 |
| Continuous month | 1500 | 2046.30 | 546.30 |
| Same grid, equal 1 oz adds | 1500 | 1896.90 | 396.90 |
| Same signal, no adds | 1500 | 1656.60 | 156.60 |

Month: 14 closed baskets, 29 entry/add orders, $11.85 commissions, $244.025
maximum peak-to-trough dollar drawdown and 6.072% maximum own-series-base loss.
The final position is flat, but the final series has a $20.10 unrecovered loss
from its $2066.40 base. That debt is retained. This is a sparse long-only
pullback strategy with hours of holding, not the former seconds-long scalper.

## Verification and practical limits

Independent vector accounting reconstructed all fills and 15,232,044 exposed
quote valuations for the selected month; equity, fees, drawdown and series loss
matched. The real August 28 invalid-book interval was separately tested with an
injected position: censoring, no fabricated close or death. The selected strategy
was not exposed to that bad interval. Tests include future-candle invariance,
bid/ask add prices, expiration, short symmetry and the strict 50% series barrier.

Frozen-parameter stresses remain profitable: 500 ms latency gives +$508.375;
double commission +$536.75; both together +$502.025. Ten one-parameter neighbors
also survive and profit on this month (+$60.85 to +$732.80). They were diagnostic
only and did not replace the frozen candidate. Fourteen baskets on one studied
month do not establish a stable out-of-sample edge or live readiness.

Evidence: `protocol.json`, `frozen.json`, `leaderboard.json`, `summary.json`,
`audit.json`, `sensitivity.json`, `real_gap_test.json`, `tests.log`, and each
variant's `result.json`. `frozen_source/` retains the exact search/kernel text
used for the initial search. Subsequent search serialization converts undefined
censored PnL to JSON null; this does not alter the unchanged numerical kernel.
