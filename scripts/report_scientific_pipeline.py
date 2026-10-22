"""Publish the bounded scientific pipeline example and its measured limitations."""
import html
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spectrum.research.training import sha, save_json, account_projection, load_execution
from spectrum.research.data import load_market

RUNS = ('tcn_execution_bce_seed17', 'mlp_execution_bce_seed17', 'tcn_crossing_bce_seed17')
NAMES = ('TCN · execution', 'MLP · execution', 'TCN · crossing')
ROOT = Path('data/derived/scientific_pipeline_v1/runs')
OUT = Path('reports/scientific_pipeline_v1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(str(c))+'</th>' for c in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, configs, inputs = [], [], {}
    for run in RUNS:
        directory = ROOT / run
        receipt = read(directory / 'receipt.json')
        if receipt['status'] != 'COMPLETE':
            raise ValueError('Unfinished run')
        for name, expected in receipt['outputs'].items():
            if sha(directory / name) != expected:
                raise ValueError('Run output changed: ' + name)
        summaries.append(read(directory / 'summary.json'))
        configs.append(read(directory / 'config.json'))
        for name in ('summary.json', 'config.json', 'receipt.json'):
            shutil.copyfile(directory / name, OUT / (run + '_' + name))
            inputs[str(directory / name)] = sha(directory / name)
    control = read(ROOT / RUNS[0] / 'controls.json')
    shutil.copyfile(ROOT / RUNS[0] / 'controls.json', OUT / 'controls.json')
    execution_manifest = read(Path(configs[0]['data']['execution_dir']) / 'manifest.json')
    shutil.copyfile(Path(configs[0]['data']['execution_dir']) / 'manifest.json', OUT / 'execution_manifest.json')
    null = read(OUT / 'independent_null.json')
    diagnostics = read(OUT / 'data_diagnostics.json')
    # Post-fit stress diagnostic: deliberately ignore the probability threshold.
    # This is never fed back into threshold selection and never changes a run.
    loaded = load_market(configs[0])
    frame = loaded['frame']
    execution, _, _ = load_execution(configs[0], frame)
    stress = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
    colors = ('#2563eb', '#d97706', '#059669')
    for run, name, color, config in zip(RUNS, NAMES, colors, configs):
        pred = pd.read_parquet(ROOT / run / 'predictions.parquet')
        p = pred[['p_none', 'p_long', 'p_short', 'p_both']].to_numpy()
        stress[run] = {}
        for ax, role in zip(axes, ('calibration', 'test')):
            result, ledger = account_projection(frame, execution, p, config['splits'][role], 0., config)
            stress[run][role] = dict(summary=result, ledger=ledger)
            start = pd.Timestamp(config['splits'][role][0] + ' 08:00:00', tz='UTC').value
            elapsed = [0.] + [(int(r['exit_ns']) - start) / 3.6e12 for r in ledger]
            cash = [0.] + [r['cash'] for r in ledger]
            label = f"{name}: {len(ledger)} trades" + ('' if result['completed'] else ' (CENSORED PREFIX)')
            ax.step(elapsed, cash, where='post', color=color, label=label)
            if not result['completed']:
                ax.scatter(elapsed[-1], cash[-1], color=color, marker='x', s=70)
            ax.set(title=role.upper() + ' | threshold=0 stress only', xlabel='Hours after London open', ylabel='Closed cash, USD/reference oz')
            ax.grid(alpha=.15)
            ax.legend(fontsize=8)
    fig.suptitle('Forcing every eligible decision is not the selected strategy', fontsize=14)
    fig.savefig(OUT / '06_forced_account_stress.png', dpi=145)
    plt.close(fig)
    save_json(OUT / 'account_stress.json', dict(post_fit_diagnostic=True, selection_feedback=False,
        interpretation='Threshold 0 forces whichever side has the larger score. One position only. A censored selected branch stops the curve; prefix PnL is not a full-day return. All actual calibrated policies chose WAIT.', runs=stress))
    rows = []
    for name, s in zip(NAMES, summaries):
        m, prior = s['evaluation']['test']['model'], s['evaluation']['test']['prior']
        rows.append([name, m['n'], s['training']['best_epoch'], f"{m['event_auc']:.3f}",
                     f"{m['joint_nll']:.3f} / {prior['joint_nll']:.3f}",
                     f"{m['direction_nll_given_event']:.3f} / {prior['direction_nll_given_event']:.3f}", 'WAIT, 0 сделок'])
    results_table = table(['Модель / таргет', 'Test roots', 'Epoch', 'Event AUC', 'Joint NLL модель / prior ↓', 'Direction NLL модель / prior ↓', 'Выбранная политика'], rows)
    cohort_table = table(['Роль', 'Дни февраля', 'Все roots', 'Допущено execution', 'NONE / L / S / BOTH'],
        [[role, ', '.join(d[-2:] for d in c['dates']), c['roots'], c['admitted'], ' / '.join(map(str, c['state_counts']))] for role, c in summaries[0]['cohorts'].items()])
    hyper_table = table(['Параметр', 'Первый опыт', 'Обоснование и ограничение'], [
        ['Локальный вход', '64 × 30 s, 163 канала', '81 значение + 81 known-mask + padding. У TCN реально 61 root: 30 минут между крайними. 64,22% сделок короче одного шага.'],
        ['Архитектура', 'TCN, hidden 32, 4 блока, kernel 3; 30 212 параметров', 'Малая причинная residual TCN. На 3 205 train roots и всего двух днях заметное переобучение; лучший epoch=1.'],
        ['Оптимизатор', 'AdamW, lr=0.001, wd=0.0001, batch=128', 'Стартовые значения, не найденный оптимум. Positive control и memorization проходят; само уменьшение lr не подтверждено как решение отсутствия переноса.'],
        ['Регуляризация', 'dropout=0.05, clip L2=1', 'В первом epoch execution TCN clipped 26,9% batches. У crossing позднее clipping почти постоянный одновременно с ухудшением validation.'],
        ['Остановка', 'max 60, patience 12, min_delta=0.0001', 'Checkpoint по joint NLL только validation. Лимит эпох не мешал: остановка после 13 для TCN, лучший epoch=1.'],
        ['Loss', 'BCE(E) + 1[E]·CE(D), общий denominator', 'Proper joint NLL. Event в train ≈38%; доминирование миллиардов easy negatives не установлено. Focal gamma=0 — обоснованный baseline.'],
        ['Риск и барьеры', 'causal R; TAKE +2R; STOP −R', 'Медианный entry cost=0.5R: до TAKE нужно +2.5R котировкой ликвидации, до STOP остаётся −0.5R. 87,24% R на cost floor.'],
        ['Дедлайн / задержка', '300 s / 100 ms; timeout 2 s', 'Реальный таймер и первая поддержанная публикация строго после задержки. Ни fill, ни stop price не гарантируются номинальным барьером.'],
        ['Порог решения', '0.35 / 0.5 / 0.65 / 0.8 / WAIT', '0.35 близок к nominal break-even 1/3 для net 2:1, но реальные payoff отличаются. Выбор только на calibration при ≥20 сделках и положительном net.'],
        ['Источник R', 'Замороженный causal R исходных roots', 'Изменение fee не пересчитывает R молча. Новую формулу риска нужно материализовать отдельной версией входных roots.'],
        ['Контекст дня', 'Текущий и предыдущий London: summaries', 'Это не полная двухдневная история. STFT=128 завершённых M1; midpoint/volume proxies, не полный DOM/MBO и не exact trade VWAP.'],
    ])
    switch_table = table(['Слой', 'Переключатели', 'Предел текущей реализации'], [
        ['Архитектура', 'mlp / cnn / tcn / lstm; hidden, layers, kernel_size, dropout', 'Mamba-2 не реализована и отклоняется явно. Полные отдельные LONG/SHORT сети пока не реализованы.'],
        ['Таргет', 'execution_profit / crossing_amplitude / safe / raw; BB/VWAP; scales; deadline; gain', 'BB/VWAP и scales меняют crossing comparator/таргет, но не фиксированный execution recipe и не набор X.'],
        ['Экономика', 'gain_multiple, stop_multiple, deadline_seconds, fee, latency, timeout, minimum_net_R', 'Параметры replay требуют нового cache path; minimum_net_R меняет только классификацию сохранённого net.'],
        ['Loss', 'factorized BCE / focal / BCE+Brier / proper_confidence; flat CE', 'Выполненные market fits используют factorized BCE. Другие losses проверены на форму/градиент, их превосходство не тестировалось.'],
        ['Данные', 'feature_family rich/geometry; sequence_length; source paths; chronological dates', 'Готовые roots только 30 s. Другой cadence требует отдельного resampling/data adapter, а не смены числа в config.'],
        ['Эксперимент', 'seed; batch; настройки AdamW; split dates; threshold grid; immutable run_id', 'Каждый run хранит config, scaler, checkpoint, predictions, history, source/output SHA и COMPLETE receipt.'],
    ])
    html_text = r'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Научный pipeline: первый обученный пример</title><style>
body{margin:0;background:#edf2f5;color:#182e3b;font:17px/1.65 system-ui,sans-serif}main{max-width:1220px;margin:auto;padding:34px 24px 80px}h1{font-size:38px;line-height:1.2;max-width:1050px}h2{font-size:27px;margin-top:44px;line-height:1.3}p{max-width:1120px}a{color:#086b98}table{width:100%;border-collapse:collapse;background:#fff;font-size:15px}td,th{border:1px solid #d4dfe6;text-align:left;vertical-align:top;padding:11px}th{background:#e0eaf0}figure,.card{background:#fff;border:1px solid #d4dfe6;border-radius:10px;padding:20px;margin:22px 0}.card{border-left:5px solid #20799f}img{max-width:100%;height:auto}figcaption,.meta{font-size:15px;color:#4f6471}code,pre{font-family:ui-monospace,monospace;background:#e3ebf1;border-radius:5px}code{padding:2px 5px}pre{padding:16px;overflow:auto;font-size:14px}.scroll{overflow:auto}nav{display:flex;flex-wrap:wrap;gap:12px 22px}li{margin:7px 0}@media(max-width:750px){main{padding:22px 14px}h1{font-size:29px}table{min-width:740px}}
</style><main><p class="meta">22 октября 2026 · GCJ6, 2–6 февраля · codex/opportunity-pipeline-v1 · воспроизводимый инженерный опыт</p>
<h1>Модель умеет учиться. Исполнимая возможность пока остаётся сложной целью.</h1>
<div class="card"><b>Получен рабочий научный pipeline.</b> Один общий интерфейс данных, архитектуры, таргета и loss; отдельный causal replay; проверки градиентов и обучаемости; сохранённые конфигурации и результаты. Реальные fits показывают важное разделение: событие на карте пересечений частично распознаётся, но точный успех короткой сделки и направление не получили подтверждённого переноса. Все выбранные политики остались в WAIT.</div>
<nav><a href="#result">Результаты</a><a href="#contract">Контракт</a><a href="#learning">Обучаемость</a><a href="#target">Форма таргета</a><a href="#hyper">Гиперпараметры</a><a href="#switch">Переключение</a><a href="#next">Следующий опыт</a><a href="#evidence">Артефакты</a></nav>
<h2 id="result">Что получилось на реальных данных</h2>
__RESULTS__
<p>NLL сравнивается с собственным train-prior на той же когорте. Строку crossing нельзя напрямую ранжировать против execution по NLL: это разные события и немного разные наборы roots. На crossing event AUC=0,692, но conditional direction NLL хуже prior. Следовательно, полезность event-головы не переносится автоматически в выбор LONG/SHORT.</p>
<p>На calibration-дне crossing-модель при заранее заданном пороге 0,5 сделала 204 последовательные сделки с net −150,30 USD на референсную унцию. Порог 0,35 дал 834 сделки и −1145,60. Это диагностическое применение crossing score к фиксированному исполнению, а не согласованный profit target. Выбранный по правилам порог — WAIT. У execution-моделей допустимые scores не достигли минимального порога 0,35. Нулевая кривая выбранной политики означает отсутствие сделок.</p>
<p><b>Неделя уже исследовалась ранее.</b> Разделение по времени предотвращает использование test в fit/selection этой процедуры, но не превращает эти дни в новый слепой holdout. Один seed и пять дней не позволяют выбрать окончательную архитектуру или сделать статистический вывод о торговом преимуществе.</p>
__COHORTS__
<h2 id="contract">Один ясный контракт: X → outcome → loss → решение</h2>
<ol><li><b>X в момент t:</b> явный allowlist 81 признака; значения, finite masks и padding. Будущие crossing IDs, time-to-hit, support-until и метки исключены. Все 8395 исходных roots сохранены; 60 не имеют feature origin. Scaler обучается только на уникальных строках train windows.</li>
<li><b>Исполнимый target:</b> для LONG и SHORT по отдельности вход на первой поддержанной котировке строго после t+100 ms. На net контролируются TAKE +2R, STOP −R, таймер ровно через 300 s после entry. Каждый выход имеет 100 ms задержку и timeout 2 s. LONG покупает ask и продаёт bid; SHORT наоборот; комиссия 0,30 USD/oz roundtrip вычитается один раз.</li>
<li><b>Метка:</b> y<sub>L</sub>=1[actual net<sub>L</sub>&gt;0], y<sub>S</sub>=1[actual net<sub>S</sub>&gt;0]. Joint state NONE/L_ONLY/S_ONLY/BOTH известен только при COMPLETE обеих сторон. UNKNOWN и EMPTY не превращаются в отрицательный пример.</li>
<li><b>Факторизация:</b> E=1[y≠NONE], D∈{L_ONLY,S_ONLY,BOTH} при E=1. Loss = BCE(E) + 1[E]·CE(D), делённый на общее число известных roots. Предсказание E не выключает direction gradient. p(L)=p(E)[p(D=L_ONLY|E)+p(D=BOTH|E)]; аналогично SHORT.</li>
<li><b>Policy:</b> выбирается сторона с большим p(side), если score проходит заранее заданный threshold. Одновременно одна позиция. При выбранном CENSORED исходе replay останавливается как неполный; такие roots не исключаются задним числом.</li></ol>
<p>Исправлены реальные дефекты: wall-clock TIME вместо ожидания новой котировки, проверка support внутри pending перед timeout, обход validation прямым SimulationConfig, silent-ignore source paths и невалидные NaN probabilities/outcomes. Старый replay сохранён отдельно; используется <code>execution_clock_v2</code>. 100 независимых scalar-проверок на полных котировках совпали по времени, статусам и деньгам, максимальная денежная ошибка 0. Это taker BBO-модель без гарантии глубины, очереди и market impact.</p>
<h2 id="learning">Градиент и обучаемость проверены отдельно от рынка</h2>
<p>Контроль с наблюдаемым прошлым дрейфом и независимыми будущими инновациями размечен тем же симулятором: на 384 новых эпизодах accuracy 100%. Намеренная подгонка 64 real train roots: 100%. Дополнительный null с независимым перемешиванием labels в train, validation и test: accuracy __NULLACC__%, event AUC __NULLAUC__. Это проверки механики, не торговые результаты.</p>
<p>Изначальный TRAIN-only shuffle дал accuracy 34,38%, но event AUC 0,702: в нём checkpoint выбирался по настоящим validation labels. Поэтому он сохранён как слабый контраст. Строгий относительно этой ошибки дополнительный null находится в <a href="independent_null.json">independent_null.json</a>; один случайный null всё равно не является доказательством отсутствия утечки или market p-value.</p>
<figure><img src="01_learning_and_controls.png" alt="Learning curves and controls"><figcaption>TCN продолжает уменьшать train NLL, когда validation уже ухудшается. Увеличение числа эпох не исправляет этот разрыв.</figcaption></figure>
<figure><img src="05_gradients_and_clipping.png" alt="Gradient scales and clipping"><figcaption>Измерены конечные ненулевые градиенты event и direction, их нормы и clipping. У execution TCN cosine на фиксированном train probe меняется +0,128 → −0,145. Одна отрицательная проекция не доказывает пользу двух отдельных моделей.</figcaption></figure>
<h2 id="target">Как выглядит цель: частые переключения или устойчивая волна</h2>
<p>Исполнимое четырёхклассовое состояние меняется в 44,27% соседних известных 30-секундных пар; бинарный event — в 41,88%. Медианная наблюдаемая серия — один root. Для crossing amplitude четырёхклассовая смена ≈22,11% среди roots с доступными X. На 7237 общих roots совпадает лишь 53,20% состояний и 56,25% event. В 1952 точках crossing положителен, но обе фиксированные исполнительные ветви не дали положительного net.</p>
<p>Более плавный прогноз не доказывает, что модель выучила плавный процесс. На графике оставлены исходные дискретные labels без сглаживания, а вероятности показаны отдельно. Иллюстрация фиксирована: первые два часа 6 февраля, независимо от качества прогноза. Crossing может описывать достижимое движение после временной просадки; жёсткий STOP может закрыть ту же ветвь раньше.</p>
<figure><img src="02_cohorts_and_target_relation.png" alt="Cohort distribution and relationship between targets"><figcaption>Примеров BOTH в fixed-plan execution нет; в crossing они существуют. Общая четырёхсостоянийная схема сохраняет эту разницу вместо искусственного p(SHORT)=1−p(LONG).</figcaption></figure>
<figure><img src="03_fixed_test_timeline.png" alt="Raw labels and probabilities in a fixed two-hour test slice"><figcaption>Детальная последовательность охватывает около 30 минут, текущий и предыдущий день представлены сводками. Гладкость вероятностей и точность распознавания события — разные измерения.</figcaption></figure>
<figure><img src="04_calibration_and_distributions.png" alt="Calibration and predicted probability distributions"><figcaption>Calibration и test подписаны отдельно. AUC измеряет ранжирование; threshold требует соответствия score экономическому событию и величинам выигрыша/потери.</figcaption></figure>
<h2 id="hyper">Гиперпараметры и измеренные ограничения</h2>
__HYPER__
<p><b>Исполнение существенно меняет барьеры.</b> Медианное удержание 15,69 s; 64,22% COMPLETE trials закрываются быстрее 30 s. 36,66% TAKE exits заканчиваются ниже +2R, 59,83% STOP exits — глубже −R. Эти отклонения объединяют перескок барьера на первой публикации и движение за задержку; это не отдельная оценка slippage от latency.</p>
<p><b>Богатство признаков подключено, но его полезность не доказана.</b> Все 81 allowlisted значения поступают в модель, однако внутридневной DOM/MBO не представлен как подробный поток. В первый день STFT отсутствует примерно в 30,7% roots, BB M9 — в 41,7%; в последний день pressure-change пропущен примерно в трети. Known masks предотвращают ложные нули, но сами пропуски и их распределение между днями остаются источником сдвига. Нужен отдельный grouped-ablation опыт, чтобы утверждать, какие семьи использованы моделью с пользой.</p>
<figure><img src="06_forced_account_stress.png" alt="Post-fit forced account stress diagnostic"><figcaption>Дополнительная проверка после fit: threshold=0 принудительно использует большую из двух сторон. Она не участвует в выборе модели/порога. Крест обозначает цензурированное продолжение, поэтому такой prefix нельзя называть полным дневным результатом. Настоящие выбранные политики — WAIT.</figcaption></figure>
<h2 id="switch">Что уже можно менять</h2>__SWITCH__
<p>Одинаковые hidden/layers не делают архитектуры равными: при 163 каналах MLP имеет 8548 параметров и RF=1, CNN 25124 и RF=9 roots (4 min), TCN 30212 и RF=61 roots (30 min), LSTM 50692 и всё переданное окно, без памяти между окнами. Эти параметры записываются в receipt.</p>
<pre>.venv\Scripts\python.exe -m scripts.prepare_research_execution --config configs/research/opportunity_pipeline_v1.json --output data/derived/scientific_pipeline_v1/execution_clock_v2

.venv\Scripts\python.exe -m scripts.run_opportunity_pipeline --config configs/research/opportunity_pipeline_v1.json --run-id my_new_tcn

.venv\Scripts\python.exe -m scripts.run_opportunity_pipeline --config configs/research/opportunity_pipeline_v1.json --run-id my_lstm --set model.architecture=lstm

.venv\Scripts\python.exe -m scripts.run_opportunity_pipeline --config configs/research/opportunity_pipeline_v1.json --run-id my_focal --set loss.event_loss=focal --set loss.gamma=1.0

.venv\Scripts\python.exe -m scripts.run_opportunity_pipeline --config configs/research/opportunity_pipeline_v1.json --run-id my_vwap --set target.kind=crossing_amplitude --set target.family=VWAP

.venv\Scripts\python.exe -m pytest tests/test_research_models.py tests/test_research_data.py tests/test_research_simulation.py tests/test_research_controls.py tests/test_research_training.py tests/test_crossing_map.py tests/test_flow_context.py -q</pre>
<p>Существующие run directories неизменяемы: повтор требует нового run_id. Новый replay-рецепт требует нового execution_dir и подготовки cache. Scales BB/VWAP берутся из существующего банка M1/M3/M5/M6/M9; смена scales сама по себе не создаёт новые causal признаки.</p>
<h2 id="next">Какой следующий опыт следует из результата</h2>
<ol><li><b>Согласовать скорость X с экономическим горизонтом.</b> Ввести короткий поток 1–5 s с объёмом/агрессией/спредом и оставить отдельный медленный контекст London/VWAP/STFT. Сравнить на одинаковых origins и новых временных блоках. В первую очередь проверить распределение holding/cost, затем выбирать cadence.</li>
<li><b>Сохранить структурный event как отдельную обучаемую цель.</b> Crossing E уже показывает ранжирование. Добавить conditional time/risk/payoff, чтобы модель различала «движение будет» и «его можно пережить и исполнить». Net-only бинарная голова теряет размер выигрыша и потерь; threshold по win probability не равен максимизации expected net.</li>
<li><b>Провести ограниченный опыт риска.</b> Заранее задать сетку R/стоимость и барьеров, контролируя median holding и censored долю. Цель — убрать навязанную микро-лотерею от слишком близкого net stop; изменение правила не является найденной прибылью.</li>
<li><b>Focal пока не усиливать.</b> BCE остаётся контрольным proper loss. Перед gamma&gt;0 измерить вклад easy/hard примеров и градиенты по классам; разумный будущий небольшой тест gamma 0 / 0,5 / 1, с отдельной калибровкой. Здесь evidence указывает прежде всего на temporal resolution, cost geometry, нехватку независимых дней и direction ambiguity.</li>
<li><b>Разделять полные LONG/SHORT модели после проверки negative transfer.</b> Сейчас измерен E/D cosine одного probe, а не конфликт LONG/SHORT задач. Нужны несколько seed и временных блоков, сопоставимые parameter/history budgets, сравнение shared encoder и отдельных side encoders. Mamba-2 пока не реализована; её нельзя объявить решением по этим графикам.</li></ol>
<h2 id="evidence">Проверяемые артефакты</h2>
<p>49 новых unit tests +13 regression tests старого crossing/flow интерфейса прошли; дополнительно 100 независимых raw-quote сравнений. На source level проверены causal prefixes, реальные receptive fields, finite differences loss, нулевой unknown gradient, train-only scaler, timeouts/support, one-position accounting и отсутствие выбора по будущей полноте.</p>
<p><a href="../../configs/research/opportunity_pipeline_v1.json">Базовый config</a> · <a href="../../spectrum/research/training.py">Trainer и account</a> · <a href="../../spectrum/research/simulation.py">Симулятор</a> · <a href="../../scripts/run_opportunity_pipeline.py">CLI</a> · <a href="data_diagnostics.json">Данные и bottlenecks</a> · <a href="controls.json">Контроли</a> · <a href="independent_null.json">Независимый null</a> · <a href="execution_manifest.json">Execution manifest</a> · <a href="visual_receipt.json">Графики и hashes</a> · <a href="sol61_visual_audit.json">Визуальный аудит Sol 6.1</a> · <a href="completion_receipt.json">Итоговый receipt</a></p>
<p class="meta">Веса, predictions, scaler и полные training histories сохранены локально в data/derived/scientific_pipeline_v1/runs. Экономические единицы — USD на референсную унцию; account sizing, контрактный множитель, queue/depth/impact, несколько месяцев и live-проверка в этот опыт не входят.</p></main></html>'''
    html_text = html_text.replace('__RESULTS__', results_table).replace('__COHORTS__', cohort_table).replace('__HYPER__', hyper_table).replace('__SWITCH__', switch_table)
    html_text = html_text.replace('__NULLACC__', f"{100*null['test']['accuracy']:.2f}").replace('__NULLAUC__', f"{null['test']['event_auc']:.3f}")
    (OUT / 'index.html').write_text(html_text, encoding='utf-8', newline='\n')
    save_json(OUT / 'report_receipt.json', dict(status='COMPLETE_REPORT_PENDING_FINAL_AUDIT', inputs=inputs,
        report_script_sha256=sha(__file__), html_sha256=sha(OUT / 'index.html'), stress_figure_sha256=sha(OUT / '06_forced_account_stress.png'),
        tests=dict(new_unit_passed=49, regression_passed=13, independent_raw_checks=100),
        limitations='Previously inspected week, one seed, only two training days; no economic admission'))
    print(json.dumps(dict(report=str(OUT / 'index.html'), runs=RUNS, stress={k:{s:v['summary']['completed'] for s,v in r.items()} for k,r in stress.items()})))


if __name__ == '__main__':
    main()
