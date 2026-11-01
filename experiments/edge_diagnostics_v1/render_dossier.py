"""Render a source-bound project diagnosis; no market or model calculations."""
from pathlib import Path
import hashlib
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads((HERE/'review_config.json').read_text(encoding='utf-8'))
    src = ROOT/config['cost_run']; out = ROOT/config['output']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        raise FileExistsError('Finished dossier exists; choose a new output in the review config')
    manifest = json.loads((src/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['artifacts'].items():
        assert sha(src/name) == expected, name
    summary = json.loads((src/'summary.json').read_text(encoding='utf-8'))
    accounts = pd.read_parquet(src/'account_cost_summary.parquet')
    accounts = accounts[accounts.condition.eq(config['condition'])]
    names = {'base':'Геометрия', 'memory':'Память', 'memory_spectrum':'+ STFT',
             'memory_mbo':'+ глобальный MBO', 'all':'Все признаки', 'past_placebo':'Чужая прошлая зона'}
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), sharex=True)
    fig.patch.set_facecolor('#fafbf8')
    for ax, block, title in zip(axes, config['blocks'], ['24–28 августа · 5 дней', '31 августа – 1 сентября · 2 дня']):
        a = accounts[accounts.block.eq(block)].set_index('arm').loc[config['arms']]
        y = np.arange(len(a)); gross = a.mean_mid_move_per_oz; net = a.mean_net_per_oz
        ax.barh(y-.16, gross, height=.27, color='#327f87', label='Движение mid между fills')
        ax.barh(y+.16, net, height=.27, color='#af4f53', label='Net после издержек')
        for i, r in enumerate(a.itertuples()):
            ax.text(r.mean_net_per_oz-.025, i+.16, f'{r.mean_net_per_oz:+.2f}',
                    ha='right', va='center', fontsize=9)
            ax.text(r.mean_mid_move_per_oz+(.025 if r.mean_mid_move_per_oz>=0 else -.025), i-.16,
                    f'{r.mean_mid_move_per_oz:+.2f}', ha='left' if r.mean_mid_move_per_oz>=0 else 'right',
                    va='center', fontsize=9)
        ax.axvline(0, color='#45535a', lw=.8)
        ax.set_yticks(y, [names[x]+(' *' if not a.loc[x,'capital_curve_complete'] else '') for x in a.index])
        ax.invert_yaxis(); ax.set_xlim(-1.1,.42)
        ax.set_title(title, loc='left', pad=18, fontweight='bold')
        ax.set_xlabel('USD/oz на закрытую сделку'); ax.grid(axis='x', alpha=.16)
        ax.set_axisbelow(True); ax.set_facecolor('#fafbf8')
    axes[0].legend(loc='lower left', bbox_to_anchor=(-.4, -0.29), frameon=False, ncol=2)
    fig.suptitle('Издержки велики; знак движения также не устойчив', x=.03, ha='left', fontsize=16, fontweight='bold')
    fig.text(.03,.025,'Фиксированные исполнения уже выбранных сделок. Не zero-cost replay. * Неполный счёт: только закрытая часть.', fontsize=10)
    fig.tight_layout(rect=[0,.09,1,.94]); fig.savefig(out/'cost_attribution.png', dpi=165)
    plt.close(fig)

    table_rows=[]
    for block in config['blocks']:
        for arm in config['arms']:
            r=accounts[accounts.block.eq(block)&accounts.arm.eq(arm)].iloc[0]
            table_rows.append(f'<tr><td>{"24–28 августа" if block=="evaluation" else "31 августа – 1 сентября"}</td>'
                f'<td>{names[arm]}</td><td>{r.trades}</td><td>{r.mean_mid_move_per_oz:+.4f}</td>'
                f'<td>−{r.mean_spread_crossing_per_oz:.4f}</td><td>−{r.mean_fee_per_oz:.4f}</td>'
                f'<td>{r.mean_net_per_oz:+.4f}</td><td>{"полный" if r.capital_curve_complete else "НЕПОЛНЫЙ"}</td></tr>')
    contract=ROOT/config['research_contract']; patch=ROOT/config['execution_fix_receipt']
    independent=out/'independent_cost_review.json'
    assert contract.exists() and patch.exists() and independent.exists(), 'Wait for signed-off input artifacts'
    patch_info=json.loads(patch.read_text(encoding='utf-8'))
    assert patch_info['validation']['status']=='PASS'
    assert json.loads(independent.read_text(encoding='utf-8'))['status']=='PASS_WITH_SCOPE_LIMITATIONS'
    dist=summary['candidate_clock_distributions']['decision_cost_proxy_over_r']
    fee=summary['candidate_clock_distributions']['fee_over_r']
    body=f'''<header><span class="eyebrow">Исследование проекта · 30 октября 2026</span>
<h1>Как перейти от описания рынка к оценке действия</h1>
<p class="lead">Проект умеет воспроизводимо описывать события и считать отрицательные результаты.
Сейчас нужно проверить величину доступного преимущества, стоимость действия и чувствительность обучающего тракта.</p>
<div class="badges"><span>22 уже раскрытых дня</span><span>Новая политика не обучалась</span><span>Экономического допуска нет</span></div></header>
<section><h2>Что проверено сейчас</h2>
<div class="cards"><article><b>{summary['candidate_clocks']:,}</b><p>общих моментов решения для измерения известной тогда цены входа</p></article>
<article><b>{dist['p50']:.3f}R</b><p>медиана «текущий спред + комиссия»; p5–p95 {dist['p5']:.3f}–{dist['p95']:.3f}R</p></article>
<article><b>{summary['saved_trade_rows']:,}</b><p>сохранённых строк сделок; {summary['unique_fill_paths']:,} комбинаций исполнения и условий, не независимые наблюдения</p></article></div>
<p>Проверены 119 189 566 native BBO publications за 22 дня. Средний спред по времени в окне поддержки исполнения
07:00–20:30 UTC — {summary['time_weighted_policy_session']['mean_spread_per_oz']:.4f} USD/oz.
Комиссия в замороженном эксперименте — 0,30 USD/oz за круг, а её медиана относительно риска — {fee['p50']:.3f}R.
Это заданная комиссия симуляции; соответствие конкретному брокерскому тарифу не проверялось.</p>
<p class="note">Текущий спред + комиссия — доступная в момент решения оценка порядка издержек. Она не знает будущий спред,
потери задержки или вероятность пассивного исполнения.</p></section>
<section><h2>Два ограничения видны одновременно</h2>
<img src="cost_attribution.png" alt="Для всех шести моделей показаны движение mid между сохранёнными fills и net на двух поздних блоках">
<p>Например, «все признаки» на пяти днях дали +0,180 USD/oz движения mid, но заплатили 0,331 спреда и 0,300 комиссии:
net −0,451. На следующих двух днях движение само стало −0,246, а net −0,874. Слабый положительный компонент первой недели
не покрывает издержки и не сохраняет знак на следующем блоке.</p>
<p><strong>Это реализованное движение по уже выбранным сделкам, а не доказанный условный дрейф.</strong>
Выбор входов, выходов и допуска следующих сделок уже зависел от исходного контракта с издержками.
При нулевой комиссии пришлось бы переиграть барьеры и последовательный счёт. Не складываем разные модели в один портфель.</p>
<details><summary>Все шесть вариантов: точные компоненты в USD/oz</summary><div class="scroll"><table>
<thead><tr><th>Период</th><th>Вариант</th><th>Сделки</th><th>Mid</th><th>Спред</th><th>Комиссия</th><th>Net</th><th>Счёт</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div></details></section>
<section><h2>Чего не хватает</h2><ol>
<li><strong>Цель должна оценивать деньги и риск действия.</strong> Вероятность RETURN_FIRST описывает геометрию.
Порог верхней пятой части калибровочных оценок не требует положительного ожидаемого результата. Позднее это фиксированный порог,
а не точная дневная квота. Нужны сопоставимые LONG/SHORT/WAIT с одинаковыми часами и известными выплатами.</li>
<li><strong>Нужен измеренный предел чувствительности.</strong> Старые positive/shuffled controls уже были.
Новая постановка должна восстановить заранее заданный слабый сигнал по всей цепочке X → target → fit → action → account
и показать, при какой его величине сигнал теряется. Нулевые отдельные корреляции не исключают совместный сигнал.</li>
<li><strong>Зона пока слишком грубо описана.</strong> Полоса $1, три визита и последние восемь эпизодов, сжатые в статистики,
ещё не равны устойчивому интересу участников. Локальные MBO-наблюдения есть только для семи дней без обучающего префикса.
Необходимы причинные последовательности «подход → воздействие потока → ответ цены → восстановление/проход».</li>
<li><strong>Полная непрерывная политика ещё не допущена.</strong> Session loader и недельный профиль 480 признаков существуют.
Остаются согласование исполнения, batch/online parity выбранных X и модели, состояние позиции, восстановление после границы сессии,
а затем отдельный последовательный капитал. Старые сертификаты 0.3 не доказывают историческую эквивалентность 0.4.</li>
<li><strong>Нужен источник будущей честной проверки.</strong> Известные исторические периоды многократно исследовались.
Не найдено подтверждения untouched-статуса доступного блока; каталог full_mbo_v1 заканчивается 1 сентября 2026.
Нужны реестр раскрытия и новый либо документированно нераскрытый материал.</li></ol></section>
<section><h2>Как учитывать техническую интуицию</h2>
<p>Базовый объект — конкретная зона и упорядоченные попытки её пройти. Для каждого посещения нужны сторона и скорость подхода,
время внутри, глубина пробоя, исполненный агрессивный поток, снятие/добавление ликвидности, восстановление книги и ответ цены.
Падение при продажах и падение при исчезновении bid — разные наблюдаемые конфигурации; MBO позволяет проверять согласованность,
но не читать намерения и личности участников.</p>
<p>Характеристику «вязкая/проходимая зона» надо сравнивать при одинаковых расстоянии, волатильности, времени суток и поддержке.
STFT остаётся проверкой периодичности посещений. Она сама не объясняет удержание цены. Для SMA–VWAP сначала выравниваются окна,
источник цены и доступность: различие временного и объёмного веса может описывать концентрацию оборота, но не задаёт автоматически
направление будущего возврата.</p></section>
<section><h2>Следующий ограниченный опыт</h2>
<p><strong>Зафиксировать ожидаемую стоимость действия на общей популяции, прежде чем расширять конструктор стратегий.</strong></p>
<ol><li>На синтетических причинных путях пройти ноль, известный дрейф разной силы, перевёрнутый знак и совместный нелинейный сигнал.
Пересчитать признаки и геометрию на каждом пути; отдельные seed/session-блоки для подтверждения.</li>
<li>На тех же 6 881 адресах рынка получить fixed-horizon markouts обеих сторон: основной горизонт 120 с,
диагностические 30 и 300 с. Отдельно движение mid, исполнимый BBO, комиссия, support и неизвестные исходы.</li>
<li>После проверки сигнала — малое заранее заданное меню выходов с общим deadline, модель ожидаемой net-ценности
и законный WAIT. Не выбирать выход задним числом для каждого примера.</li>
<li>Управление HOLD/CLOSE добавить отдельным опытом с новой информацией после входа и ветвями из одного состояния.
Пассивность проверять с вероятностью fill, adverse selection и стоимостью пропуска, используя уже проверенную очередь V5.</li></ol>
<p>Полный <a href="../../../{config['research_contract']}">контракт следующего исследования</a> отделяет критерии принятия
от направлений дальнейшего поиска. Он предложен, но ещё не исполнен. Новый рынок и новый обученный контроллер в текущем аудите не заявляются.</p></section>
<section><h2>Поправки к внешнему рассуждению</h2>
<ul><li>Optional stopping имеет условия на мартингал и остановку; она не делает любые net-выходы одинаковыми при меняющихся издержках.
<a href="https://www.stats.ox.ac.uk/~etheridg/martingales.pdf">Etheridge, теорема 6.4</a>.</li>
<li>Преимущество может находиться и в исполнении относительно mid: исходная модель
<a href="https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf">Avellaneda–Stoikov</a>
рассматривает предоставление ликвидности при mid без дрейфа. Она не доказывает пригодность пассива к нашим данным.</li>
<li>Уже оплаченный вход исключается из парного HOLD/CLOSE, но будущая ликвидация, риск и занятый капитал остаются.</li>
<li>В таблице корреляций смешаны односторонний отбор и торговля по знаку. Универсальной границы «ρ≥0,1» нет.
Числа «corr≤0,03» и «17% против 31%» не получили найденного воспроизводимого источника именно для этих зон.</li>
<li>Пассивное исполнение и HOLD/CLOSE уже исследовались. Старый парный контроллер на августе прибавил к early CLOSE всего
0,00102 USD/oz на попытку, дневной интервал [−0,00403; +0,00686]. Повтор того же target под новым названием не решит проблему.</li></ul></section>
<section><h2>Исправление исполнения и границы проверки</h2>
<p>Предварительный аудит воспроизвёл пропуск точного TAKE/STOP в библиотечном float-сравнении. Исправление оформлено отдельно
как {html.escape(str(patch_info.get('version','0.4.1')))}; <a href="../../../{config['execution_fix_receipt']}">паспорт версии</a>
сохраняет связь с 0.4.0. Данные cost_audit_v1 сохраняют состояние до исправления. Последний зональный опыт использовал другой,
исправленный mechanical engine; этот дефект не отменяет его отрицательный результат.</p>
<p>Стоимость проверена тождеством на всех 3 660 строках; максимальный остаток {summary['max_net_identity_residual_per_oz']:.2g} USD/oz.
Два исходных счёта неполны и сохраняют этот статус. Raw lineage отделён от заново проверенных canonical quote payloads.</p>
<p>0.4.1: полный набор tests + checks в исследовательской среде — 678 PASS; в закреплённой библиотечной среде — 664 PASS
без двух модулей, которым нужен отсутствующий там sklearn. Эти два модуля входят в полные 678. Новая wheel и remote CI не запускались.
Независимый cost review оставил небольшое замечание к следующему producer: проверять saved config до его использования.
Значения текущего запуска независимо сверены с исходным config и совпадают; численный результат допущен с указанными ограничениями.</p>
<p><a href="../cost_audit_v1/report.html">Полный отчёт стоимости</a> ·
<a href="independent_cost_review.json">Независимая проверка стоимости</a> ·
<a href="manifest.json">Источники и SHA-256 досье</a></p></section>
<footer>Исследовательский результат: обнаружены проверяемые пробелы и величина издержек. Прибыльная политика не продемонстрирована.</footer>'''
    style='''*{box-sizing:border-box}body{margin:0;background:#f5f6f2;color:#23363d;font:17px/1.65 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:36px 26px 70px}header{padding:30px 0 22px}h1{font-size:clamp(32px,5vw,53px);line-height:1.12;max-width:900px;letter-spacing:-1.5px;margin:20px 0}h2{font-size:27px;line-height:1.3;margin:0 0 18px}.eyebrow{font-size:13px;letter-spacing:1px;text-transform:uppercase;color:#52736d}.lead{font-size:21px;max-width:900px}.badges{display:flex;gap:8px;flex-wrap:wrap}.badges span{font-size:13px;background:#e6ece7;border-radius:18px;padding:5px 12px}section{padding:30px 0;border-top:1px solid #d7dfd7;margin-top:18px}p{max-width:950px}li{padding:5px 0 9px}a{color:#116b73;text-underline-offset:3px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cards article{padding:20px;background:white;border:1px solid #dde4dc;border-radius:10px}.cards b{font-size:30px}.cards p{font-size:14px;line-height:1.5;margin-bottom:0}.note{border-left:3px solid #9eaa83;padding:12px 20px;background:#edf0e6;font-size:15px}img{width:100%;height:auto;border:1px solid #dce1d8;border-radius:8px}details{background:#fff;padding:15px;border:1px solid #dae0d8;border-radius:8px}summary{cursor:pointer;font-weight:600}.scroll{overflow-x:auto}table{border-collapse:collapse;white-space:nowrap;font-size:13px;margin-top:14px}td,th{text-align:right;padding:9px 12px;border-bottom:1px solid #e4e9e0}td:first-child,td:nth-child(2){text-align:left}footer{font-size:14px;border-top:1px solid #cad4ca;padding-top:24px;color:#5a6968}@media(max-width:650px){main{padding:22px 16px 44px}.cards{grid-template-columns:1fr}.cards article{padding:15px}.cards b{font-size:27px}body{font-size:16px}h2{font-size:24px}.lead{font-size:19px}ol,ul{padding-left:22px}}'''
    doc='<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Состояние проекта и следующий опыт</title><style>'+style+'</style><main>'+body+'</main></html>'
    (out/'report.html').write_text(doc,encoding='utf-8',newline='\n')
    paths=[src/'manifest.json',src/'summary.json',src/'account_cost_summary.parquet',contract,patch,independent,HERE/'review_config.json',Path(__file__)]
    result={'status':'COMPLETE_DIAGNOSTIC_ONLY','untouched_holdout':False,'new_policy_fit':False,
            'inputs':{str(p.relative_to(ROOT)):sha(p) for p in paths},
            'artifacts':{p.name:sha(p) for p in [out/'report.html',out/'cost_attribution.png']},
            'scope':config['scope']}
    (out/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
