# Воспроизводимость flow/context v2

Ветка `codex/flow-response-context-v2` ответвлена от `codex/opportunity-episodes-v1`, commit `386e35d`. Протокол, feature/target code и проверки заморожены **до обучения** в `55f8b0e`. Месяцы A1: октябрь–ноябрь 2025, A2: декабрь 2025; refit A1+A2, B: январь–февраль 2026. Эти периоды уже исследованы и не объявлены blind OOS.

Сначала проведён coverage audit; его JSON входит в commit до fit. Затем материализация, обучение и независимая проверка исполнения. Все изменения после fit относятся к отчёту, визуализации, независимым аудитам и упаковке результатов; пять файлов научного identity не изменены. Изменения финальных PNG касались читаемости, явных разрывов поддержки, дат цензуры и обозначения тейкерского overshoot, а не значений прогнозов.

## Что требуется локально

`.venv/Scripts/python.exe` и библиотеки из `environment.json`. Приватные данные: `data/derived/scalper_v1`, подготовленные M1-дни `data/derived/basket_multimonth_v1`, канонические quotes/features/MBO captures из source manifests. `data/derived/flow_context_v2` содержит 106 дневных Parquet, 28 файлов пар прогнозов, 28 файлов пар обученных голов, receipts и replay facts. Тяжёлые данные и pickle-модели не входят в Git; готовые PNG/HTML и численные отчёты доступны без них.

Полный набор M1 проверялся по manifest hashes при materialization (`MonthDay(..., verify=True)`). Coverage audit сопоставил 16541 root с BBO на десяти календарных датах; build сопоставил спреды всех 173724 выбранных roots. Полный повтор свежего event-level MBO replay на всех днях не заявлен: независимый execution audit охватывает пять календарных дат, 824 сравнения.

## Последовательность исходного запуска

Из корня репозитория, PowerShell:

```powershell
$env:OMP_NUM_THREADS='2'
.venv/Scripts/python.exe -m scripts.flow_context_coverage
.venv/Scripts/python.exe -m spectrum.cognition.flow_context_lab build
.venv/Scripts/python.exe -m pytest -q --basetemp .tmp/pytest_flow_context_v2
.venv/Scripts/python.exe -m spectrum.cognition.flow_context_lab run
.venv/Scripts/python.exe -m scripts.flow_context_execution_audit
.venv/Scripts/python.exe -m scripts.flow_context_report
.venv/Scripts/python.exe -m scripts.flow_context_validate
```

Исходный полный pytest завершён: **315 passed in 48.46s**; лог `.tmp/flow_context_v2_full_tests.log`, его SHA-256 записан в `validation_receipt.json`. До полного набора отдельно прошли 17 тестов текущих feature/head contracts. `flow_context_validate` проверяет сохранённый лог, а не запускает pytest повторно. При новом запуске лог нужно сохранить отдельно и явно записать новое свидетельство.

`build` возобновляет уже совпадающий cache и проверяет scientific identity. `run`, coverage audit, execution audit и report используют фиксированные каталоги. **Для нового опыта создайте отдельную ветку, version/contract и каталоги**, сохранив нынешние receipts, вместо перезаписи этих результатов. Повтор полного запуска в изолированной копии может изменить служебные timestamp/runtime/Git поля, поэтому содержательная воспроизводимость не означает побайтовое совпадение всех JSON.

## Связи доказательств

- `completion_receipt.json` связывает scientific code, contract, daily manifest и 28 prediction Parquet.
- `data_receipt.json` содержит counts, доступность каждого поля и previous-day masks; полные дневные hashes находятся в локальном manifest.
- `execution_audit.json` связывает свежий MBO replay, независимый quote-scan выходов, исходный audit code и сохранённые execution facts.
- `visual_receipt.json` связывает все 14 PNG с summary, правилами выбора сцен и report code.
- `validation_receipt.json` проверяет scientific identity против байтов pre-fit Git, source/daily/prediction hashes и footer, изображения, execution receipts; записывает hashes моделей, финальных отчётов и окружения.
- `FEATURE_CONTEXT_AUDIT_RU.md` и `SOL61_VISUAL_AUDIT_RU.md` — независимый Sol 6.1: чтение фактических входов и saved models, воспроизведение 3334 прогнозов, визуальный просмотр всех 14 PNG и повтор исправленных сцен.

## Границы результата

Это проверка фиксированного плана 24/12 тиков, 300 секунд, 1 oz/$0.30, пассивного входа V5 и вторичного тейкерского входа; выходы тейкерские. Обе головы используют X на момент решения. Conditional outcome после fill не означает обучение новым сведениям после входа. HGB получает скаляры, а не полную дневную последовательность. EMPTY и неизвестный исход различаются; цензурированный счёт не получает полный net. Все допуски A2 отрицательны и выбранных E[net]>0 открытий нет. Технический PASS, lower log-loss и чтение деревами признаков не устанавливают прибыльность.
