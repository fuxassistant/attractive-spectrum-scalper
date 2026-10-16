# Независимый аудит Sol 6.1 — 15 октября 2026

**Вердикт: PASS в ограниченном инженерном scope causal market reconstruction / replay / feature laboratory. P0/P1 не обнаружены.** Обе недели завершены, все 12 source-reset captures представлены. Это не допуск торговой политики, не свидетельство прогнозной или экономической ценности 182 признаков и не измерение live latency.

Аудитор не менял implementation и не запускал вторую полную реконструкцию. Проверки ниже выполнены самостоятельно над сохранёнными массивами, а не выведены только из авторских PASS receipts. Полная вторая независимая raw-to-book реконструкция всех 35.7 млн строк не выполнялась; независимый dictionary oracle автора ограничен чистым префиксом 20,000 raw records / 16,979 publications.

## Что действительно проверено

- Прочитаны README, contracts/foundation_v1.json, docs/ENGINEERING_NOTES_RU.md; replay.py, tape.py, lab.py, feature_stream.py, ingest.py, build.py, calendar.py, audit.py; существенные части core/stream.py, aggregate.py, schema.py и тесты foundation/process adapters.
- Независимо пересчитаны SHA всех 22 salvaged core files против файлов исходного checkout и обоих наборов SHA salvage_manifest.json: 22/22 совпадают. Это проверка текущих байтов; source_commit зафиксирован manifest, отдельно Git object каждого исходного файла не извлекался.
- Собственный checker reports/independent_sol61_check.py проверил все 12 captures: SHA всех market payloads и feature Parquet, совпадение seconds/features clocks, строго возрастающий endpoint clock, монотонную availability и available_ns >= t_ns, отсутствие crossed valid BBO.
- Для каждой сохранённой quote проверено **точное совпадение source_ordinal со всем списком raw F_LAST records соответствующего partition**. Пропавших или придуманных committed publications по этому ключу не найдено. Ordinal относится к входному двухнедельному subset, сохранённому в original subset wire order; это не переименование в абсолютный ordinal monthly ZIP.
- Во всех 182 feature columns каждого capture проверены finite values, false-mask => technical zero, known => admissible origin, неотрицательная PSD power и диапазон [0,1] для fractions/efficiency с малой floating tolerance. На закрытой календарём секунде admissible не встречается. Receipt: independent_sol61_saved_checks.json.
- Самостоятельно повторены foundation + process/plugin tests после API fixes: **12 PASS, 5.71 s**, reports/independent_sol61_tests_final.xml. Авторский полный XML tests_final_v2.xml прочитан отдельно: 50 tests, failures/errors 0.
- Самостоятельно пересчитана полнота обеих недель через явную минутную UTC-сетку и исключение часа 21, без вызова MarketCalendar. Receipt independent_sol61_coverage.json сохраняет каждый missing и invalid close timestamp.

## Обе реальные недели и остаточные границы

| UTC-неделя | Ожидалось open M1 | Сохранено | Valid M1 | Open seconds сохранено / ожидалось | Admitted seconds |
|---|---:|---:|---:|---:|---:|
| 03 May 22:00 — 08 May 21:00 | 6900 | 6895 | 6885 (99.7826%) | 413994 / 414000 | 378058 (91.3184%) |
| 10 May 22:00 — 15 May 21:00 | 6900 | 6895 | 6885 (99.7826%) | 413995 / 414000 | 382439 (92.3766%) |

В каждой неделе отсутствуют пять close-labelled 00:00 M1 на переходах source reset. Десять invalid M1 находятся строго на 00:01 после reset и 22:01 после session open. Это реальная потеря/неполная поддержка, а не заполненные свечи. Остаточные 6 и 5 missing second rows также не спрятаны. Непрерывный полный календарный массив с 100% допустимостью не заявляется. Пороги 95% valid M1 и 80% admitted seconds — инженерная достаточность именно этой лаборатории.

Sunday preopen source snapshots сохраняются в raw/market, но время до 22:00 закрыто календарём и не допускается как origin. История не перескакивает через reset, epoch, разрыв или закрытый час. Недостаточная свежесть текущего endpoint запрещает текущую feature row, при этом уже известная поддержанная история остаётся доступной более позднему свежему окну. Отдельные provider/source availability и endpoint time сохранены. Запаздывание вычисления на текущей машине не выдаётся за provider latency.

## Визуальный осмотр

Лично через view_image просмотрены preview_scene.png, все 12 random scenes, оба break controls и overview. Финальный атлас — **reports/audit_v2/index.html**. Все 12 random PNG в audit_v2 побайтово совпали по SHA с лично просмотренными audit_v1 PNG. Три изменённых файла audit_v2 открыты и проверены повторно.

Просмотренные финальные изображения:

- random_capture_000.png — 03 May 23:35:26 UTC; спокойное открытое время, отдельные admission holes.
- random_capture_001.png — 04 May 06:28:58 UTC; падение около 06:34 согласуется с path/trade-sell/short PSD всплеском.
- random_capture_002.png — 05 May 09:19:07 UTC; умеренное снижение, path существенно больше net displacement.
- random_capture_003.png — 06 May 13:08:24 UTC; резкий спад около 13:05, рост gross activity/path/PSD, signed-trade FFT unknown при unknown-side trades.
- random_capture_004.png — 07 May 15:30:18 UTC; снижение и последующая стабилизация, нормальный разный масштаб 32/128 s PSD.
- random_capture_005.png — 08 May 09:57:39 UTC; восходящее/колебательное движение, OHLC close не заменяет intraminute extrema.
- random_capture_006.png — 10 May 22:21:11 UTC; спад/возврат и unknown trades около 22:18 корректно исключают signed FFT.
- random_capture_007.png — 11 May 09:19:22 UTC; спад около 09:25 сопровождается sell/path/gross-flow/PSD всплесками.
- random_capture_008.png — 12 May 15:42:58 UTC; движение/глубина и потоки меняются раздельно; unknown signed FFT не выключает midpoint PSD.
- random_capture_009.png — 13 May 00:05:39 UTC; начало capture с видимым прогревом long-window PSD.
- random_capture_010.png — 14 May 09:17:35 UTC; короткий спад слева, затем боковой режим, path остаётся положительным.
- random_capture_011.png — 15 May 15:56:11 UTC; колебательный участок, unsigned/midpoint features сохраняются при mostly-unknown signed FFT.
- break_capture_001.png и break_capture_007.png — 21:00–22:00 UTC: пустые price/features в закрытом интервале, false admission, новый прогрев после открытия.
- two_weeks.png — обе недели, ежедневные часы и weekend согласованы; соединения через weekend отсутствуют.

Цены bid/ask и midpoint M1 согласуются по масштабу и времени; path измеряет весь committed-event путь и поэтому может многократно превышать секундный net displacement. PSD не направленная величина: реакция на активность с разными окнами не доказывает предсказательность. Gross add/remove не равны net depth: большие взаимные обороты при сравнительно небольшой глубине допустимы. Пробелы видны, а не нарисованы нулями.

Авторский арифметический OHLC аудит имеет 282 проверки, каждая сцена >0, max error <1e-8. Проверен код его независимого пересчёта open/high/low/close по exact quote layer; это проверяет OHLC относительно BBO publications, а не независимо всю raw exchange semantics. Случайная выборка фиксирована seed 20261015, по одному admissible origin на capture, без фильтра по будущему движению. Визуальный осмотр всех случайных сцен не заменяет перебор каждого raw события; все численные masks проверены отдельно.

## Замечания и их закрытие

1. Изначальный audit PASS мог быть пустым (0 OHLC checks) и не проверял полноту. Исправлено до финального атласа: >=100 checks суммарно, >0 в каждой сцене, OHLC error gate и явные weekly engineering coverage gates. В финале 282 checks.
2. Fingerprint FeatureStream checkpoint первоначально не включал calendar/adapter implementation. Исправление прочитано и tests повторены: теперь hash lab.py, calendar.py, feature_stream.py и plugin identities проверяется на restore.
3. В audit_v1 top BBO диагонально соединял редкие публикации через scheduled break; overview coverage диагонально соединял Friday/Sunday. **P2 закрыт:** audit_v2 закрывает BBO календарём, разрывает длинный display gap, разрывает coverage при missing M1. Обе изменённые break-картинки и overview лично повторно открыты. Численные datasets не менялись; audit_v1 сохранён как superseded evidence.

Незакрытых P0/P1/P2 в проверенном scope не осталось. Широкий API произвольных сторонних plugins опирается на честно объявленный lookback, версию и causal compute; произвольный plugin может сам прочитать будущий suffix, и базовый интерфейс не является sandbox. Использованные встроенные формулы имеют prefix-invariance тест.

## Скорость и реальные пределы вывода

Полный сохранённый market build: **1873.936 s (31 min 14 s), 6 процессов, 35,697,456 selected records**, около 19,050 records/s в сумме. Это наблюдённая wall duration одной генерации; скейлинг относительно 1 процесса для полного replay не измерен. Один стакан остаётся последовательным, parallelism допускается только по фактическим source resets.

Decoder receipt показывает median 3.203x на 30k одинаковых records, 5 interleaved repeats; изучен код benchmark. Это только column extraction / MboRecord construction, не 3.2x ускорение всей реконструкции.

Matched real-feature workload: 161,999 seconds rows, serial 27.696 s против 2-process 17.749 s, **1.560x**. Сохранённые serial/parallel Parquet имеют совпавшие SHA; отдельный process/plugin тест аудитора также PASS. Измерение включает process startup, read, FFT, write/readback и проводилось при concurrent raw work. Это один matched pair, без доверительного интервала, isolated scaling sweep и peak-memory benchmark. rss_sample/peak_rss_sample — выборочные измерения, не доказанный OS high-water mark. Pipelined feature total elapsed включает ожидание raw и не является чистой стоимостью формул.

Рекомендуемое использование результата: версионированная лаборатория воспроизводимого MBO рынка и causal features с masks, epochs и lineage. Ready-to-trade, fill simulation, economic edge и universal calendar из выполненной проверки не следуют.

## SHA проверенных основных артефактов

Таблица ниже автоматически добавлена после завершения аудита. Полные payload SHA доступны в проверенных manifests; все они сверены независимым checker.

| Файл | SHA256 |
|---|---|
| data/derived/raw_v1/manifest.json | fe286914c81e361e665a6bd3cf8956079facd707f242c305180aa21e3728c38e |
| data/derived/market_v1/manifest.json | cd6f82b15d2914a23008901fb509b0ffa190ce758c9b6ab05cdf58cbe61ceae3 |
| data/derived/features_v1/manifest.json | 94cdfbfb80d78b72a57e4acc094b497a4b7072e01fcec81758bf587d08ef9efb |
| contracts/foundation_v1.json | 174e62ae2247f9e535d7a51b556c80fe4b4b8d2373c246ba92014604942102c1 |
| reports/audit_v2/audit.json | ba08389e47a20cfa0243dd0112f8181a69da634d4fb598f14e6238c851f57e24 |
| reports/audit_v2/two_weeks.png | f7d75d69a4e1a8b1ba75504526e4454c2d205513447031d4fe8ec7409d4b82f0 |
| reports/source_lineage.json | 797559f439c823a6dbf76e76a95f6fbae85396369dddae91e4a70f6cca529002 |
| reports/salvage_manifest.json | 41ac6a6f6264be7b487ac64ac80006471b75f863b5aac04e6c257d90ed5c378e |
| reports/feature_performance.json | b55ffbb342e4982c07b296511678562f6f56cdd7ab1ea70562c167ab63b32c73 |
| reports/foundation_verification.json | 808026396b819bef5d1d9fd107ecef674c41c7d19a161827b45c0bc7c78dcebe |
| reports/independent_sol61_saved_checks.json | 29d568f67c96f4a389ad92d0c8b0c71bd38ce959434d6aaf1895efd28c170c0a |
| reports/independent_sol61_coverage.json | f578aa96fc11e557e6240ee386edbb6e4a30895e417e71c2f84fecb1eea842aa |
| reports/independent_sol61_tests_final.xml | 4f4592b17ef14bca819c3cf4981b70108d3e3acfaa63fdc5283c47aa10507972 |
| reports/tests_final_v2.xml | bffca333e68653627c4760af32ae93172dc353ff193a6fb3b9573a62f1aead69 |
| spectrum/lab.py | 58fc768e26db6744b4fe105ab2aa53298b246c0d119f328c9bee701736dff2d5 |
| spectrum/replay.py | 00971fa10d15e8d1f9cb02830916ea27648890ab9444b91478f9bd181b7f7531 |
| spectrum/feature_stream.py | 965724a13894a5d5881d6bc8ef98bc6e61ae8cce6263beab2ce5c69229f22e9f |
| spectrum/tape.py | 9102338a670e96b19470d0ef9426d66bf68c8f9f375ed275961a3accb29cec1c |
| spectrum/audit.py | 7ec2bf28f6ff66416a4b6b1eb2ca35232dac67d8ad4e91880e6094cb7df8ca85 |
| reports/audit_v2/random_capture_000.png | 93c5d247be5fd8afccfdfe3165af35053d40d5ea66049073f1044a74b4c54585 |
| reports/audit_v2/random_capture_001.png | 1802f9bf7d44b7234ba840c072dd3cd7e9347626e5911bca7d642d57cec5e71e |
| reports/audit_v2/break_capture_001.png | 215550fed9d2fc77226231a8fa6df62b1486f29f0773139314dd49842d401d53 |
| reports/audit_v2/random_capture_002.png | 584c4fcf7a24ba5b54eab702550a32bd56b7f9b6109a7023401a3d8a00d8a5f6 |
| reports/audit_v2/random_capture_003.png | 5ba52fea73370f2f6c8e4828bb03f866f05fa1b4a39873e6325cdbd01eebe0ed |
| reports/audit_v2/random_capture_004.png | b2ae1c4d38006fc9b1cea8d97c69068657b72595ab247041d211b3e795180ba3 |
| reports/audit_v2/random_capture_005.png | c27b84917f1762ff5679151016dcce59d8341cb8a6606ad93550c37550a33716 |
| reports/audit_v2/random_capture_006.png | 59ca79581926502f8a2cdc191ffa775c76dbcfc578ac706ccfe6a4e96bf60e6b |
| reports/audit_v2/random_capture_007.png | 4e8b4ae2fd6da8bec8c141518a6706351471f03877f84a1d39a2ccd575c22f56 |
| reports/audit_v2/break_capture_007.png | 958cbf7e0605f7d7a7b15d037794a74c31b4b870e80e2530c2bea4dcf5bf525c |
| reports/audit_v2/random_capture_008.png | 7e5b8b04cc27b9ca612d22a035cfeff90322a088fe1e5f5aae27e9ca48f47fa5 |
| reports/audit_v2/random_capture_009.png | a4158afdc7570863d57f129752e83907e96f69d50c56bb67cba4d2d2ee30f899 |
| reports/audit_v2/random_capture_010.png | c8bb06d9e838015f39102c4d15e2927c562b77851634756e1b8d754bac4f648b |
| reports/audit_v2/random_capture_011.png | 36308d9682c06ffe0d85086f9ab0edd34b058ab05fc08470069604fd7183ec1d |
