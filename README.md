# Spectrum — причинная библиотека рынка

Основа для новой работы с GC MBO: реконструкция, replay, семейства признаков,
представления рынка, банк пересечений и отдельные вычислители будущих исходов.
Обучающие опыты, подборы политик, старые игроки и отчёты сохранены в ветке
`codex/archive/research-2026-10-26`.

## Карта проекта

| Пакет | Ответственность |
|---|---|
| `spectrum.canon` | Канон вводных: единицы, часы UTC, правила MBO, часы площадок, сессии и оверлапы с учётом DST, издержки по умолчанию; у каждого значения единица, статус и источник (`resources/canon_v1.json`) |
| `spectrum.core` | Единственное исходное ядро L3/MBO, F_LAST, состояние, clocks, поддержка и потоковые метрики |
| `spectrum.market` | Чтение исходников, подготовка tapes, replay, календарь, quote arrays, общая FIFO-механика |
| `spectrum.features` | Каталог семейств, формулы, причинное выравнивание, пакетный и потоковый расчёт |
| `spectrum.representation` | Interaction ledger, fast/M1/context/zones/events, наблюдение as-of, память и lazy sequence windows |
| `spectrum.crossings` | Фактический банк пересечений (правило bid OR ask), native BB/VWAP, устойчивые адреса геометрии, сохранение и чтение |
| `spectrum.future` | Всё, что видно только вперёд от точки: `future_*` функции и поля (ближайшее пересечение каждой линии, движение цены). Вход модели отвергает любые поля со словом future |
| `spectrum.economics` | Экономический слой: комиссия и деньги поверх `future`; факты не меняет |
| `spectrum.execution` | Исполнение и последствия заданных действий; без обучения и выбора торговой политики |
| `spectrum.resources` | Версионированные определения, поставляемые вместе с установленным пакетом |
| `experiments` | Чистое пространство для новых опытов; зависимость направлена из опыта в библиотеку |

Данные доступны по времени публикации `available_ns`. Endpoint `t_ns` не даёт права
использовать ещё не опубликованное значение. Неизвестность сохраняется масками;
future target, техническая корректность и экономическая полезность — разные свойства.

## Установка и проверки

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m spectrum families
```

Библиотеке и её тестам не нужны PyTorch, scikit-learn или исходные рыночные архивы.
Для реального replay необходимы собственные данные. На Linux/macOS используйте
`.venv/bin/python`.

## Выбор семейств

```python
from spectrum.features import list_families, get_family, compute_batch, FeatureStream

for family in list_families("liquidity"):
    print(family.name, family.input_contract, family.entrypoint)

# seconds — уже опубликованные канонические секунды, а не будущие метки.
selected = ("motion", "flow", "spectral", "depth_balance", "volatility_vwap_proxy")
batch = compute_batch(seconds, selected)
stream = FeatureStream(selected, source_id="my-source-identity")
part = stream.feed_batch(seconds.iloc[:1000])
```

`FeatureBatch` содержит `values`, `known`, `names`, `units`, `t_ns`, `available_ns`.
Пакетный и потоковый маршруты используют одни формулы. Trade-exact VWAP требует
точных trade moments; midpoint proxy выбирается отдельно.

У MBO/M1, секундных функций и sequence observations разные входные контракты.
Каталог делает их явными и позволяет найти функции через `get_family(name).resolve()`.
Подробности: [семейства](docs/FAMILIES.md), [слои и инварианты](docs/ARCHITECTURE.md).

## Рынок, взаимодействия и пересечения

```python
from spectrum.market import MarketReplay, FinalizedReplay, QuoteColumns
from spectrum.representation import interaction_stream
from spectrum.crossings import build_bank, publish_bank, open_bank

# Поток взаимодействий подключён к тому же MboStream и его BookView.
stream = interaction_stream(config, emit_levels=levels.extend, emit_trades=trades.append)
stream.feed_many(records)

# Канонические целочисленные цены и publication-clock; книги здесь не создаются.
quotes = QuoteColumns.from_frame(finalized_quotes)
bank, tape = build_bank(quotes, source_id="verified-source-identity")
publish_bank(quotes, "data/derived/my_crossings", source_id="verified-source-identity")
saved_bank, tape, manifest = open_bank("data/derived/my_crossings")
```

Линия пересечена, когда bid или ask перешёл на другую её сторону. Банк хранит
только факты пересечений с ordinal/event ID. Полосы строятся по прошлому, поэтому каждое пересечение — уже состоявшийся
факт в свой `t_ns`. `future.future_crossings(bank, tape, ...)` из каждой точки
находит ближайшее будущее пересечение каждой линии или причину цензуры;
`economics.future_crossing_marks` считает по ним деньги с комиссией из канона.
Дребезг остаётся в банке, а маркеры отмечают новые возможности:
`line_visit` — первое пересечение линии после пересечения другой линии того же масштаба;
`opportunity` — `line_visit` или пауза на этой линии не короче кулдауна из канона (8 с);
`cluster_event_id`/`cluster_rank` связывают дребезг с его открывающим событием.
`future_crossings(bank[bank.opportunity], ...)` ищет ближайшую новую возможность.

`spectrum.leakage` отделяет будущие таблицы (`FutureFrame`/`FutureSeries`) и массивы
(`FutureArray`) от X. Срезы, нормализация, смена dtype и NumPy-операции над
`FutureArray` сохраняют метку; неявное преобразование через `np.asarray` запрещено.
Входы X проверяют значения и маски до преобразования типов. Имена `future_*` и
отпечатки значений служат дополнительными проверками.

`future_examples(x, y)` соединяет X и Y по времени: X опубликован к origin,
Y созрел после него, `cutoff_ns` ограничивает зрелость обучающих ответов.
При пустом X возвращаются нулевые значения, ложные маски и `usable=False`.
Для передачи Y внешней модели или сериализатору используйте
`export_future_values(y)` из `spectrum.leakage`. Экспорт намеренно снимает тип:
не возвращайте такой массив в построители X. Произвольные внешние преобразования
или восстановление обычных таблиц из чисел не сохраняют проверяемое происхождение;
отпечатки не заменяют проверку причинности источника и формулы.
Банки по дням: `python -m spectrum bank --days data/derived/basket_multimonth_v1 --output ...`;
`open_bank` возвращает `(bank, tape, manifest)`.

## Общий календарь

`MarketCalendar()` использует `gc_globex` из канона и поставляемую зависимость
`tzdata`. Это общий default для batch, stream, finalized schedule и представлений,
включая зимнее время и переходы DST. Переданный `scheduled_closed` проверяется
на соответствие выбранному календарю; старые несовместимые данные требуют пересчёта.
`reference_observed_open` — отдельная маска покрытия источника, не расписание биржи.
В каноне праздничное покрытие указано отдельно; полный календарь праздников не заявляется.

Checkpoint `FEATURE_STREAM_V3` сохраняет fingerprint канона, его реализации и
версию `tzdata`. Старые V2 checkpoints требуют пересоздания на исходном потоке.
Текущие результаты проверки пакета 0.3.0: [VALIDATION.md](docs/VALIDATION.md).

## Представления для новой модели

`make_fast`, `make_minutes`, `anchored_context` и `zone_and_event_tables` строят
таблицы представления из переданных данных. `MarketObservationStore.observe(t)`
возвращает только доступный к `t` префикс. `publish_observation_store` сохраняет
таблицы в новый каталог с manifest и хешами; `open_observation_store` читает их.

`SequenceStore` принимает матрицу признаков и описание физических часов без
обязательных targets. Он строит окна по требованию; пропущенный X остаётся
позицией на часах с нулевой маской. Нормализатор `fit_scaler` получает явно
выбранные строки обучения и один раз вычисляет статистики для всех семейств.

## Новая работа и архив

Начните с [правил нового эксперимента](experiments/README.md). Не добавляйте fit,
выбор модели или отчёты в библиотечные пакеты. Результаты складывайте в
`research_results/<experiment>/<run>/`, большие данные — в `data/`.

[Архив и восстановление](docs/ARCHIVE.md) содержит точный сохранённый commit.
[Карта миграции](docs/MIGRATION.json) связывает старые реализации с новыми.
Исторические контракты и результаты остаются привязаны к исходной версии кода;
новая организация не переобъявляет старые артефакты результатами нового запуска.
