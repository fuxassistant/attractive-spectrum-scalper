# Каталог причинных семейств

Каталог: `spectrum.features.list_families()`; CLI: `python -m spectrum families`.

| Семейство | Группа | Clock | Вход и точка вызова |
|---|---|---|---|
| `core.ohlc` | price | completed M1 | core context / Family('ohlc'); `spectrum.core.feature_families:default_families` |
| `core.quotes` | price | publication + completed M1 | core context; `spectrum.core.feature_families:quotes` |
| `core.trade` | flow | publication + completed M1 | core context; `spectrum.core.feature_families:trade` |
| `core.depth` | liquidity | publication + completed M1 | core context; `spectrum.core.feature_families:depth` |
| `motion` | price | 1s | Context / seconds; `spectrum.features.families.motion:motion` |
| `flow` | flow | 1s | Context / seconds; `spectrum.features.families.flow:flow` |
| `spectral` | spectrum | 1s | Context / seconds; `spectrum.features.families.spectral:spectral` |
| `depth_balance` | liquidity | 1s | Context / seconds; `spectrum.features.families.depth_balance:calculate` |
| `volatility_vwap_proxy` | volatility_geometry | 1s | Context / midpoint seconds; `spectrum.features.families.volatility_vwap:proxy` |
| `volatility_vwap_exact` | volatility_geometry | 1s | Context / exact trade moments; `spectrum.features.families.volatility_vwap:exact` |
| `opportunity_baseline` | volatility_geometry | as-of decision | supported second grid + origin indices; `spectrum.features.families.opportunity:causal` |
| `signature.price` | price | as-of decision | supported grid + origins + known sigma; `spectrum.features.families.signature:price_family` |
| `signature.flow` | flow | as-of decision | published MBO seconds + intents; `spectrum.features.families.signature:flow_family` |
| `signature.micro` | liquidity | completed M1 | prepared minute arrays + intents; `spectrum.features.families.signature:micro_family` |
| `episodes` | events | 1s | seconds + segment tokens; `spectrum.features.families.episodes:features` |
| `flow_context.local` | flow | 1s | seconds + segment tokens; `spectrum.features.families.flow_context:local_features` |
| `flow_context.day` | context | completed M1 / as-of decision | current/previous minute tapes + origins; `spectrum.features.families.flow_context:context_features` |
| `flow_context.spectrum` | spectrum | completed M1 | minute tape + supported windows; `spectrum.features.families.flow_context:trailing_spectra` |
| `volatility` | volatility_geometry | 1s | supported second arrays + session metadata; `spectrum.features.families.volatility:features` |
| `anchored_profile` | context | sampled 1s prefix | completed previous session + current arrays; `spectrum.features.families.anchored_profile:features_day` |
| `response` | spectrum | 512s trailing window | published seconds + window endpoint; `spectrum.features.families.response:window_features` |
| `landmarks` | events | recognition time | published window descriptors; `spectrum.features.families.landmarks:detect_bursts` |
| `event_context` | events | strictly before intent | intent times + recognized event times; `spectrum.features.families.event_context:event_context` |
| `market_history` | context | as-of decision | minute arrays + exact quote arrays; `spectrum.features.families.market_history:market_features` |
| `scalper_context` | context | as-of decision | supported grid + origins + published seconds; `spectrum.features.families.scalper_context:features` |
| `side_orientation` | context | as-of decision | causal scalper context + chosen side; `spectrum.features.families.scalper_context:side_features` |
| `observation_summary` | representation | publication watermark | fast/context tables + explicit origins; `spectrum.features.families.observation_summary:summary_at` |
| `queue.flow` | liquidity | strictly before intent | MBO tape + decision BBO addresses; `spectrum.features.families.queue_state:window_flows` |
| `queue.depth` | liquidity | strictly before intent | MBO tape + decision BBO addresses; `spectrum.features.families.queue_state:level_sizes` |
| `crossing_geometry` | volatility_geometry | as-of decision | origin scene with causal BB geometry; `spectrum.features.alignment:geometry_features` |
| `memory.minutes` | context | publication time | completed minute tape; `spectrum.representation.memory:minute_values` |
| `memory.anchors` | context | publication time | published seconds/minutes + previous session; `spectrum.representation.memory:anchor_values` |
| `memory.events` | events | event availability | recognized event table; `spectrum.representation.memory:encode_events` |
| `representation.sequences` | representation | physical continuity + cadence | origin frame + float32 matrix + family column groups; `spectrum.representation.windows:SequenceStore` |
| `representation.fast` | representation | 1s publication watermark | canonical seconds + interaction ledger + exact trades; `spectrum.representation.representation:make_fast` |
| `representation.minutes` | representation | completed M1 watermark | fast frame + exact trades + canonical bars; `spectrum.representation.representation:make_minutes` |
| `representation.context` | representation | two-session prefix | minutes + trades + explicit current/previous date; `spectrum.representation.representation:anchored_context` |
| `representation.zones` | representation | version availability | fast/context frames + per-price ledger; `spectrum.representation.zones:zone_and_event_tables` |

Секундные `motion`, `flow`, `spectral`, `depth_balance`, `volatility_vwap_proxy`, `volatility_vwap_exact` доступны в общем `compute_batch` и `FeatureStream.feed_batch`. Остальные функции принимают native contract из таблицы.

Будущие targets и path descriptors находятся в `outcomes`. Функция `spectrum.outcomes.volatility.recent` обрабатывает только уже созревшие outcomes и не регистрируется обычной X-семьёй.

Исторические сочетания семейств, схемы весов и выбранные модели сохранены в архивных конфигурациях.
