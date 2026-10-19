# Преднарегистрация проверки на март–апрель 2026

Создано 2026-10-19T18:52:29.581285+00:00, родительский коммит `55fda8a16fbe8fbdac65ed4191dff3490c06454a`. Процедура заморожена: `spectrum.cognition.validation` сверяет SHA-256 всех перечисленных модулей и отказывается работать при любом изменении.

Процедура: библиотека схем (ассимиляция, аккомодация с дневной t ≥ 4 на еженедельных контрольных точках, карантин 15 дней, роутер LCB κ = 2.5, полураспад 40 дней) с языком очереди, пассивное исполнение PQ_30 (вход в очередь на 30 с, take лимиткой), один счёт 1 oz, комиссия $0.30. Обучение — окт 2025 – фев 2026; блок V — март–апрель 2026.

На момент регистрации ни одна сигнатура, ни одно исполнение и ни одно состояние очереди блока V не сгенерированы и не прочитаны.

Март–апрель уже просматривались другими исследованиями проекта, поэтому это проверка процедуры, а не слепой holdout.

## Гипотезы и критерии

- **H1 — экономика схем.** Economic: block V net of the one-account 1 oz schema router (PQ_30 execution, queue language) > 0 with day-bootstrap 95% lower bound > 0. Zero trades = NO_TRADE (not a pass). Evaluated separately for prequential and frozen-at-V learning.
- **H2 — нулевой контроль.** Null control: shuffled_context promotes 0 schemas and makes 0 trades in block V.
- **H3 — стоимость исполнения.** Execution: for every plan, paired daily mean of PQ_30 attempt minus TT attempt on block V has day-bootstrap lower bound > 0.
- **H3 — неблагоприятный отбор.** Adverse selection replicates: daily mean taker net where PQ_30 did not fill minus where it filled has lower bound > 0.
- **H4 — ранжирование «выставляться ли».** Resting ranking (model fitted on Oct-Feb): for plan (12, 8, 120), R^2 on V vs the train mean > 0 AND daily mean attempt of the top prediction decile minus the bottom half (edges from train predictions) has lower bound > 0.
- **H5 — переход стадии 0 (opportunity).** Opportunity stage-0 gate: per horizon, the challenger fitted on Oct-Feb beats the train-scaled causal premium (MSE skill > 0) in BOTH March and April.
- **H6 — знание схем.** Schema knowledge: on V, attempts of short/40-20-600 in trend=down are worse than all short/40-20-600 attempts (daily paired residual, upper bound < 0).
- **R0 — воспроизводимость.** Reproducibility: prequential, assimilate_only and shuffled_context A/B block metrics equal the earlier train-only queue-language run.

## Замороженные модули

- `execution.py` `a3fb6adac03cc4e14ca68953f05717569058fb03236a175e1e0e4b565f0e2005`
- `opportunity.py` `8a2b856bda856ed81ee7cc96296c18aeb4f8e651a80bf1adfff5c6619936a4c9`
- `signatures.py` `a3862cb86dd18a8223d2a569839c17c5365800414f0c77dd8038bc60aaad8173`
- `signature_report.py` `6e4ae677c348b8bdd1c34e0cd545662db9dd3b878c939ebe00b8156aa3209dbf`
- `representation.py` `ef76d93d5141ee10acd80f6208b4e5c9d99f8ca8915d674c1a5d0b13c1e18620`
- `qmodel.py` `d4591438587fbc2d56af4ab5e1c24419db79d809c9ff0d2c1412f661cbd75466`
- `schemas.py` `3c966b7244ff02de8d9b78f258a52d25f6d44be68ba78daaa577390c012004b4`
- `schema_run.py` `f8ac82a5ac9ff6ed6be5b3f45e3fc12cd8f96a9635f79538dfa142919f0f055e`
- `queue.py` `cd3abfba4e86fb6a1c62040edce5e518edc7426d6098c2e458ebb6957ed0d45d`
- `passive.py` `e7392916c22ec3a5a2667c6ec2071f0cc2c00ae4e6efd404c101db5620aeed5e`
- `queue_state.py` `1baf88d636e6a0c251efede7b45e02d7912ab68429e6d90105e1b890d05f72e4`
- `passive_learn.py` `a18602a78e9eac2a3698d99dfa078e4085bff6d9591fd586153310110e708d8a`
- `passive_report.py` `5e2e6ce42626ca102a4b81bbc20c25b935b87d7d49249ffaf1f7db7e55f12a4b`
- `validation.py` `3dea7fecdc41976a36ca3eb9b66d4673de5f7e7fcdb485740e227bbe3610e86c`

## Команды

    python -m spectrum.cognition.signatures --months 2026-03 2026-04 --output data/derived/opportunity_signatures_validation_v1
    python -m spectrum.cognition.representation --signatures data/derived/opportunity_signatures_validation_v1 --output data/derived/representation_validation_v1
    python -m spectrum.cognition.passive --signatures data/derived/opportunity_signatures_validation_v1 --months 2026-03 2026-04 --output data/derived/passive_execution_validation_v1
    python -m spectrum.cognition.queue_state --signatures data/derived/opportunity_signatures_validation_v1 --months 2026-03 2026-04 --output data/derived/queue_state_validation_v1
    python -m spectrum.cognition.validation

## Поправка 1

`passive.py`: `e7392916c22ec3a5a2667c6ec2071f0cc2c00ae4e6efd404c101db5620aeed5e` → `8314ec4f37e5e6678bf0352bc75da2d393fb41ebcb0e703c1f0d9e864fa8f934`. В `day_tapes` поиск captures ограничен месяцами запрошенных дат вместо месяцев train по умолчанию. Это ошибка маршрутизации: для дат валидации captures не находились, и пассивное исполнение и состояние очереди дали 0 дней. Правила решений, исполнение, критерии и train-результаты не меняются. На момент поправки сгенерированы сигнатуры и представление блока V (выведены только счётчики завершения); ни один исход, исполнение или состояние очереди V не просматривались.

## Поправка 2 (после просмотра исходов)

Независимый аудит после первого прогона нашёл ошибки учёта очереди в ядре (исполнения, сообщённые до постановки; iceberg-пополнения; скрытый объём; устаревший уровень книги при агрессивной перестановке). Ядро V5 совпадает с явной FIFO-реализацией в 520 из 520 проб и проходит инвариант «нет исполнения без сделки по цене или сквозь неё» на всех 151 днях. Изменены: `schema_run.py`, `queue.py`, `passive.py`, `validation.py`. Пересчёт марта–апреля после этой поправки — исправленный пересчёт той же процедуры, а не чистая преднарегистрированная проверка.
