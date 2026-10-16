# Attractive Spectrum Scalper: причинное ядро, лаборатория и LSTM gym

Исследовательский проект для причинной реконструкции рынка GC, лаборатории фич и обучения торговой политики. Экономическое преимущество модели пока не установлено.

## Текущее состояние — 16 октября 2026

- Полная генерация завершена: 37 исходных MBO-файлов, 291 фрагмент, 288 с фичами и 3 пустых исключённых; август включён.
- Сохранено 21 937 619 секундных строк с 182 фичами и масками. Данные занимают около 35 ГБ и не входят в Git.
- Аудит: 2 034 контрольные суммы совпали, временные метки и маски проверены на всех строках. Численная проверка фич и дополнительные проверки OHLC выполнены выборочно.
- Текущий каталог допускает 231 фрагмент; дополнительная проверка полноты сессии исключает 25 мая и оставляет 230. Источники не покрывают 31 июля.
- Загрузчик gym пока привязан к двум майским неделям. Интеграция полного каталога, границ контрактов и хронологических разбиений остаётся следующей задачей.

[Полная генерация и возобновление](docs/full_mbo_generation.md) · [Отчёт аудита](reports/full_mbo_v1/audit_20261016/audit.json) · [Замечания](reports/full_mbo_v1/audit_20261016/findings.json)

## Установка и проверки

Python 3.11+. Из корня репозитория:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-foundation.txt
.venv\Scripts\python.exe -m pip install -e ".[player,dev]"
.venv\Scripts\python.exe -m pytest -q --basetemp .tmp/pytest_local
.venv\Scripts\python.exe -m spectrum --help
```

На Linux/macOS используйте `.venv/bin/python`. Для ядра без торгового игрока достаточно `pip install -e ".[dev]"`. Тесты используют синтетические данные и не требуют частных архивов. GitHub Actions запускает Python-тесты с CPU PyTorch.

## Структура и правила Git

| Каталог | Содержание |
|---|---|
| `spectrum/core` | Перенесённое ядро реконструкции и причинных измерений |
| `spectrum/player`, `spectrum/player_v2` | Gym, LSTM и иерархическая политика |
| `spectrum/full_history.py` | Полная очередь MBO, фичи, контроль диска и продолжение |
| `scripts` | Генерация, измерения, аудит и локальные инспекторы |
| `tests` | Автоматические проверки без исходных рыночных архивов |
| `contracts`, `docs` | Контракты и инженерные решения |
| `reports` | В Git — выбранные небольшие отчёты; остальные результаты локальные |
| `data` | Только локальные исходники, Telegram-разметка и производные данные |

Сырые данные, сообщения трейдеров, веса моделей, подробные сделки и тяжёлые HTML/изображения не публикуются через Git. Команды воспроизведения требуют собственных исходников в указанных путях. Ссылки внутри исторических аудитов могут указывать на локальные артефакты, отсутствующие в клоне. Абсолютные пути и PID в сохранённых receipts относятся к машине исходного запуска.

Происхождение перенесённого кода и данных зафиксировано в `reports/salvage_manifest.json` и `reports/source_lineage.json`. Репозиторий не предоставляет лицензию на исходные рыночные данные или Telegram-контент.

## Предыдущие исследовательские этапы

Первый слой проекта — реконструкция GC из MBO, готовый market replay и подключаемые фичи. Поверх него реализован первый LSTM-игрок WAIT/LONG/SHORT с выбором плана выхода и PPO по результатам собственных действий.

**Новый этап v2:** отдельные участие / направление / план, графический инспектор цены и политики (локально: `reports/player_v2/run_001/index.html`), [контракт и управление](docs/PLAYER_V2_RU.md), [независимый Sol6.1 аудит](reports/player_v2/run_001/INDEPENDENT_AUDIT_SOL61_RU.md). Три seed×60эпох,243 визуальные траектории, исходные и ограниченные вероятности, чувствительность входов. Положительное ранжирование after-cost состояний пока не подтверждено: два selected WAIT, третий−$6.40/10сделок/1censored на ранее изученной второй неделе.

**Игрок:** [управление и контракт gym](docs/PLAYER_GYM_RU.md), интерактивный отчёт 30 эпох (локально: `reports/player_v1/run_001/index.html`), [независимый аудит Sol 6.1](reports/player_v1/run_001/INDEPENDENT_AUDIT_SOL61_RU.md). Обучение4–7 мая, выбор8 мая, frozen проверка11–15 мая. Прибыльное преимущество не установлено: selected test37 сделок, наблюдаемый proxy−$46.40,17censored; WAIT=$0. Все17 censor вызваны объявленным правилом межпубликационной паузы>5 s. Отчёт сохраняет эти ограничения.

Готовый результат: [отчёт двух недель](reports/REPORT_RU.md), итоговый атлас (локально: `reports/audit_v2/index.html`), [независимый аудит Sol 6.1](reports/INDEPENDENT_AUDIT_SOL61_RU.md).

```text
DBN/zstd → проверенные source-reset partitions → MarketReplay / один MboStream
                                             ├─ exact BBO publications
                                             ├─ секундный стакан и поток
                                             └─ midpoint M1 OHLC
                         сохранённый рынок → feature plugins → values + masks
```

## Простое управление

Из корня проекта в PowerShell:

```powershell
.venv\Scripts\python.exe -m spectrum --help
.venv\Scripts\python.exe -m spectrum status data/derived/market_v1
.venv\Scripts\python.exe -m spectrum replay data/derived/market_v1/capture_001 --limit 5

# Фичи можно менять без нового MBO replay. Каждый запуск получает новый output.
.venv\Scripts\python.exe -m spectrum features --market data/derived/market_v1 --output data/derived/features_motion_v2 --families motion flow --workers 4

# Подключить новую формулу, не изменяя ядро и не создавая вторую книгу.
.venv\Scripts\python.exe -m spectrum features --market data/derived/market_v1 --output data/derived/features_plugin_v2 --plugin spectrum.plugins.depth_balance:PLUGIN --families motion depth_balance --workers 4
```

Полный маршрут воспроизведения в новые каталоги:

```powershell
.venv\Scripts\python.exe -m spectrum ingest --source data/raw/GC_MBO_2026-05-03_to_2026-05-15.dbn.zst --route 1 19181 22 --output data/derived/raw_v2
.venv\Scripts\python.exe -m spectrum build --raw data/derived/raw_v2 --output data/derived/market_v2 --workers 6
.venv\Scripts\python.exe -m spectrum features --market data/derived/market_v2 --output data/derived/features_v2 --workers 4
.venv\Scripts\python.exe -m spectrum audit --market data/derived/market_v2 --features data/derived/features_v2 --output reports/audit_v2
.venv\Scripts\python.exe -m pytest -q --basetemp .tmp/pytest_local
```

`workers` задаёт независимые процессы, а не потоки одного стакана. Одна часть начинается с действительного R/snapshot источника; произвольное деление суток с пустой книгой не допускается. Выделенные в источнике resets сохраняются, поэтому история фич разогревается заново. Каталоги завершённых генераций не перезаписываются.

## Данные и перерывы

Две исследовательские недели: 3 мая 22:00 — 8 мая 21:00 UTC и 10 мая 22:00 — 15 мая 21:00 UTC 2026. По OHLC и подтверждению пользователя ежедневный перерыв в этом периоде **21:00–22:00 UTC**; уикенд — пятница 21:00 — воскресенье 22:00. Это зафиксированная настройка выбранного периода, не универсальный календарь праздников/DST.

Сырьё сохранено в `data/raw`, результаты — в `data/derived`; большие массивы исключены из Git. cTrader bid-OHLC служит отдельным свидетельством календаря. Цены GC и GC midpoint-OHLC реконструируются из MBO, не подставляются из cTrader. Пропуск cTrader 12 мая 20:15–20:24 и отсутствующая первая минута воскресенья 10 мая — отдельные source gaps.

Часы и ключи BBO — integer nanoseconds/prices. Публикация доступна после полного F_LAST. Минутные свечи маркированы временем закрытия и основаны на midpoint. Секундные измерения публикуются после подтверждающей записи; время endpoint и доступности различаются. EOF не дорисовывает последний бар.

## Подключение фич

`spectrum.lab.FeaturePlugin(name, version, compute, lookback_seconds)` описывает семейство. `compute(context)` читает проверенную секундную ленту, вызывает `context.add(name, values, known, unit)`. Пример — `spectrum/plugins/depth_balance.py`. `--plugin module:attribute` импортируется в каждом worker; источник и версия сохраняются в manifest. Для произвольных зависимостей пользовательского плагина требуется собственная версия/учёт зависимостей.

По умолчанию 182 признака: движение/путь/глубина, объёмы и изменения стакана, спектры шести каналов на 32/64/128 секундах. Это компактная первая лаборатория; она не объявляется переносом всех 536 признаков старого Pulse или всей библиотеки 2143 selectors.

Значение при false mask равно техническому нулю. Оно не означает наблюдённое отсутствие активности. Историческая поддержка отделена от допуска текущего origin: поздно подтверждённое прошлое может использоваться в следующем свежем окне. Unknown trade-side исключает signed-trade спектр, но не остальные каналы. PSD не задаёт направление; среднее и displacement остаются отдельно.

## Replay API

```python
from spectrum.core import StreamConfig
from spectrum.replay import MarketReplay

market = MarketReplay(StreamConfig(1, 19181, 22, snapshot_ns=1_000_000_000))
market.feed_batch(records)  # тот же переход пригоден для последовательной доставки
view = market.query()       # frozen committed BBO, не промежуточная книга
checkpoint = market.checkpoint()  # допустим внутри незавершённой F_LAST
restored = MarketReplay.restore(checkpoint)
```

`TapeReplay` читает уже построенные BBO-публикации с проверкой hash, поддерживает последовательную выдачу, `advance(until_ns)`, checkpoint/restore курсора. Это replay наблюдений; он не симулирует исполнения. Восстановление cursor перечитывает префикс; быстрого произвольного seek пока нет.

В `advance` исходный `valid` относится к исторической публикации; текущую свежесть показывает отдельный `admissible_at_query` с тем же budget 30 секунд и `quote_age_ns`. Чтение во время перерыва не превращает старую котировку в свежую.

`spectrum.feature_stream.FeatureStream` использует те же формулы для последовательных пакетов секундных наблюдений, удерживая только halo до 129 строк. Есть checkpoint/restore с проверкой source identity, кода и плагинов. Фичи batch/stream совпадают с численной точностью `rtol=1e-12, atol=1e-10`, маски — точно; микроскопическое отличие порядка floating-point суммирования не скрывается.

## Проверки и результаты

Контракт — `contracts/foundation_v1.json`. Итоговые receipts, визуальный атлас и независимый аудит размещаются в `reports`. Различайте воспроизводимость реконструкции, смысл фич и их будущую экономическую полезность.
