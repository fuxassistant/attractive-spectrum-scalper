# Spectrum — причинная библиотека рынка

Основа для новой работы с GC MBO: реконструкция, replay, семейства признаков,
представления рынка, банк пересечений и отдельные вычислители будущих исходов.
Обучающие опыты, подборы политик, старые игроки и отчёты сохранены в ветке
`codex/archive/research-2026-10-26`.

## Карта проекта

| Пакет | Ответственность |
|---|---|
| `spectrum.core` | Единственное исходное ядро L3/MBO, F_LAST, состояние, clocks, поддержка и потоковые метрики |
| `spectrum.market` | Чтение исходников, подготовка tapes, replay, календарь, quote arrays, общая FIFO-механика |
| `spectrum.features` | Каталог семейств, формулы, причинное выравнивание, пакетный и потоковый расчёт |
| `spectrum.representation` | Interaction ledger, fast/M1/context/zones/events, наблюдение as-of, память и lazy sequence windows |
| `spectrum.crossings` | Фактический банк пересечений, native BB/VWAP, устойчивые адреса геометрии, сохранение и чтение |
| `spectrum.outcomes` | Будущие метки и запросы к путям: crossing, risk, MFE/MAE, время, favorable, maturity/censoring |
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
bank, geometry = build_bank(quotes, source_id="verified-source-identity")
publish_bank(quotes, "data/derived/my_crossings", source_id="verified-source-identity")
saved_bank, receipt = open_bank("data/derived/my_crossings")
```

Банк хранит факты пересечений с ordinal/event ID. Будущий запрос к банку должен
выполняться отдельно от формирования X. Правила entry/latency, риск, единицы и
цензурирование задаются явно в `outcomes`/`execution`.

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
