# Рабочее ядро и подключаемая лаборатория: две недели GCM6

Дата: 15 октября 2026. Репозиторий: `D:/scalper-mini/attractive-spectrum-scalper`.

Создан действующий маршрут DBN MBO → независимые source-reset captures → общий причинный MarketReplay → exact BBO, секундные измерения и midpoint M1 → подключаемые фичи. Две реальные недели реконструированы, datasets сохранены, результаты проверены арифметически и визуально. Торговая политика не добавлялась.

## Что доступно

- [Итоговый атлас](audit_v2/index.html): 12 случайных сцен, 2 контроля ежедневного перерыва, обзор обеих недель.
- [Техническая приёмка и покрытие](audit_v2/audit.json).
- [Независимый аудит Sol 6.1](INDEPENDENT_AUDIT_SOL61_RU.md).
- [Команды запуска и API](../README.md), [контракт](../contracts/foundation_v1.json), [решения и исправления](../docs/ENGINEERING_NOTES_RU.md).

Рыночные данные: `data/derived/market_v1`; фичи: `data/derived/features_v1`; проверенные исходные partitions: `data/derived/raw_v1`. Каждый каталог содержит manifests; рыночный и feature manifests ссылаются на SHA предыдущего слоя. Большие массивы исключены из Git.

## Источник и реконструкция

DBN header подтверждает **GCM6**, route `(1,19181,22)`; [mapping](instrument_mapping.json). Другие контракты не склеивались. Первичный майский subset содержит 45 103 379 записей разных инструментов. Все его 56-byte records в исходном порядке побайтово сверены с заявленным диапазоном оригинального monthly ZIP member (93 756 430 records), включая SHA compressed member. [Source lineage](source_lineage.json).

Для GCM6 реконструировано **35 697 456 MBO records**, опубликовано **33 527 506 полных F_LAST**, сохранено **927 547 секундных строк**, **15 450 midpoint M1 строк**. Общий объём market Parquet — 538 821 680 bytes. Секундные/минутные таблицы включают также неподдержанное время и закрытые часы; этот полный счёт строк не является числом допустимых model origins.

Параллелизм основан на 12 явных R/reset началах, включая воскресные префиксы. Нет разрезания рынка по произвольному времени с пустой книгой. Каждая часть имеет единственного MBO owner. Сохраняются source ordinal, F_LAST, epoch, integer BBO prices и отдельный available_ns. Snapshot rows не считаются добавленным рыночным потоком. T volume не дублируется F records.

Исходное ядро перенесено из `poi-scalper` с SHA каждого файла: [salvage manifest](salvage_manifest.json). 22/22 файла независимый аудитор повторно сверил с исходным checkout. Новая оболочка и лаборатория отделены от перенесённого ядра.

## Достаточность двух недель

Календарь подтверждён по OHLC и пользователем: **21:00–22:00 UTC** в будни, **Friday 21:00–Sunday 22:00 UTC** на уикенде. Он ограничен этим исследовательским периодом и не выдаётся за универсальный календарь DST/праздников. cTrader OHLC используется только для отдельной проверки часов; рыночные цены и свечи GC получены из MBO.

| Показатель | 3 мая 22:00 — 8 мая 21:00 UTC | 10 мая 22:00 — 15 мая 21:00 UTC |
|---|---:|---:|
| Ожидаемые открытые минуты | 6 900 | 6 900 |
| Наблюдаемые M1 строки | 6 895 | 6 895 |
| Валидные M1 | 6 885 | 6 885 |
| Валидное минутное покрытие | 99,7826% | 99,7826% |
| Ожидаемые открытые секунды | 414 000 | 414 000 |
| Наблюдаемые секундные строки | 413 994 | 413 995 |
| Допущенные свежие origins | 378 058 | 382 439 |
| Доля свежих origins | 91,3184% | 92,3766% |

В каждой неделе 5 отсутствующих M1 закрытий в 00:00 относятся к source-reset boundaries; 10 invalid M1 — к первым минутам после resets/открытий. Они не заполнены синтетическими свечами. Независимый пересчёт содержит каждый timestamp: [coverage receipt](independent_sol61_coverage.json).

Это проходит инженерный gate 95% valid minutes / 80% fresh seconds для тестового представления. Все feature families сохраняют отдельные masks; недостаточная поддержка signed-trade FFT не удаляет остальные каналы. Историческое подтверждение секунды и свежесть текущего origin — разные условия.

## Лаборатория и скорость

По умолчанию **182 признака** в семействах motion / flow / spectral. Сохранено 927 547 строк `values + known masks + exact clocks`, 811 845 350 bytes feature Parquet. Реальные единицы измерения сохранены в manifests. Для каждой части выполнено точное read-back сравнение Parquet. False-mask значения — технические нули, не наблюдённая нулевая активность.

FFT: 6 каналов, trailing 32/64/128s, periodic Hann, mean subtraction с отдельным сохранением среднего, one-sided PSD, полосы периодов 4–8 / 8–16 / 16–32s при ≥4 nominal cycles. Это первая компактная лаборатория, а не полный перенос старых 536 Pulse признаков. Direction не восстанавливается из PSD.

Новый plugin подключается через `--plugin module:attribute` и versioned `FeaturePlugin`. Есть рабочий пример `spectrum.plugins.depth_balance:PLUGIN`, протестированный в разных процессах. `FeatureStream` использует те же формулы с halo до 129 строк и checkpoint/restore, проверяющим источник, код, календарь и plugins. Batch/stream masks совпадают точно, values — с указанным численным допуском.

| Измерение | Результат | Область утверждения |
|---|---:|---|
| Полная реконструкция 35,7 млн records, 6 workers | 1 873,94 s = 31m 13,94s | Источник/декодирование partitions, replay, Parquet и checkpoints; текущая общая нагрузка машины |
| Наибольшая RSS sample одного raw worker | 476 155 904 bytes | Sample на границах batches; не доказательство OS peak всей группы |
| Scalar → columnar decoder | 3,20× по медиане | 30 000 одинаковых records, 5 чередующихся repeats; не speedup всего replay |
| 161 999 секундных строк, 1 feature worker | 27,70 s | Две реальные части, чтение + 182 features + запись + readback |
| Те же данные, 2 feature workers | 17,75 s, 1,56× | Побайтово одинаковый Parquet; single matched pair под concurrent raw load |
| Сумма compute времени 12 feature jobs | 140,33 worker-seconds | Полный набор; pipelined wall time включает ожидание raw и не является benchmark |

Машина: Windows 11, 12 logical CPUs, RAM 42 675 761 152 bytes. [Matched performance receipt](feature_performance.json), [decoder/restart verification](foundation_verification.json). Основное преимущество для дальнейших экспериментов — повторный расчёт фич без нового восстановления десятков миллионов MBO events. Сырые ядра и feature workers не делят mutable book.

## Проверки

- **50 tests PASS**, [финальная JUnit квитанция](tests_final_v2.xml): перенос, F_LAST, snapshot/recovery, clocks, checkpoints, source identity, future suffix, DFT, masks, batch/stream, serial/multiprocess parity, plugin import и artifact tampering.
- Independent plain-dictionary book oracle на 20 000 реальных records / 16 979 полных публикациях: BBO и L1 quantities совпали. Это scoped prefix check, не независимая вторая реконструкция всех недель.
- JSON checkpoint внутри F_LAST и иное batching дали идентичные quote/snapshot/bar outputs.
- В случайных и контрольных сценах **282 независимые OHLC сверки** по всем точным BBO publications; максимальная ошибка 0.0. Визуальное прореживание графика не участвует в oracle.
- Независимый Sol 6.1 checker проверил все 12 captures: SHA payloads, все 182 masks/finite/zero/ratio/power constraints, clocks, quote validity, source ordinals каждой F_LAST публикации. [Saved checks](independent_sol61_saved_checks.json).

Первый атлас `audit_v1` сохранён как superseded: визуальные line plots соединяли точки через перерывы. Renderer исправлен, данные не менялись; финальная визуальная приёмка относится только к `audit_v2`.

Здесь установлена работоспособность и инженерная достаточность причинного рыночного слоя на двух неделях. Предиктивная ценность фич, экономическое преимущество, broker fills и live SLA этим этапом не проверялись.
