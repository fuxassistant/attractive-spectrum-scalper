# Воспроизводимость opportunity episodes v1

Запускать из корня репозитория в Python environment с `requirements-foundation.txt` и project extras `cognition,dev`. В выполненном опыте: Python 3.12.14, scikit-learn 1.9.1, NumPy 2.2.6, pandas 2.2.3, PyArrow 20.0.0, Numba 0.64.0, SciPy 1.18.1, Matplotlib 3.11.1; полный снимок в `environment.json`. Исходные приватные данные `scalper_v1`, `full_mbo_v1` и MBO/quote caches не входят в Git. Итоговые модели, дневные Parquet и вероятности лежат локально в `data/derived/opportunity_episodes_v1`; их хеши сохранены в receipts. Графики и компактные результаты доступны в Git.

## Проверки без приватных данных

```powershell
.venv\Scripts\python.exe -m pytest tests/test_cognition_episodes.py -q
```

Полный suite до календарной поправки: 305 passed in 80.84s. После неё целевые проверки: 9 passed in 3.30s, включая два новых теста календаря и цензуры. Изменения календаря не меняют X, Y, обучение, прогнозы, пороги или сделки.

## Новый независимый запуск

Не перезаписывать опубликованный опыт. Для новых вычислений выбрать новые каталоги:

```powershell
$env:OMP_NUM_THREADS='2'
.venv\Scripts\python.exe -m spectrum.cognition.episodes_lab build --data data/derived/opportunity_episodes_v1_repro --report reports/opportunity_episodes_v1_repro
.venv\Scripts\python.exe -m spectrum.cognition.episodes_lab run --data data/derived/opportunity_episodes_v1_repro --report reports/opportunity_episodes_v1_repro
```

Конфигурация намеренно одна — `contracts/opportunity_episodes_v1.json`. Смена конфигурации после чтения B является новым исследованием, не повтором заранее зафиксированного опыта. Для исходного опыта совместимость исторического source code доказана AST; исходные хеши не заменены текущими.

## Аудит и рисунки опубликованного опыта

Эти команды используют каталог v1 и обновляют его отчёты; они предназначены для восстановления тех же артефактов, а не для нового опыта:

```powershell
.venv\Scripts\python.exe -m scripts.opportunity_episodes_audit
.venv\Scripts\python.exe -m scripts.opportunity_episodes_report
```

Аудит проверил 668 сочетаний origin/сторона/режим на шести календарных днях: свежие MBO queue fills, тейкерские входы и независимый последовательный quote-scan выходов. Расхождений 0. Это проверка исполнения и меток на выборке, не независимый полный audit всех 106 дней и не проверка брокерских fills.

`rescore-calendar` — одноразовая миграция старого отчёта, уже выполненная и сохранённая в `AMENDMENTS_RU.md`. Повторно запускать её в исходном v1 не нужно: она откажется перезаписывать историю поправки.

`summary.json` — исправленный основной результат; `summary_pre_calendar_fix.json` — прежний результат для истории. `completion_receipt.json` связывает summary, данные, прогнозы и исходный/аналитический code identity; `visual_receipt.json` связывает PNG с summary. `posthoc_score_decomposition.json` явно помечает разложение общего выигрыша после чтения результата. `SOL61_VISUAL_AUDIT_RU.md` — отдельный визуальный вердикт.
