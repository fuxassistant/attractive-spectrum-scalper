"""Build a standalone Russian audit report with selectable actual observations."""
import base64
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from spectrum.io import save_json,sha

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/derived/feature_week_v2';A=OUT/'audit'
read=lambda p:json.loads(p.read_text())
def js_safe(value):
    if isinstance(value,dict):return {k:js_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [js_safe(v) for v in value]
    if isinstance(value,int) and abs(value)>2**53:return str(value)
    return value
audit=read(A/'audit.json');parity=read(A/'parallel_parity.json');run=read(OUT/'final_parallel2/manifest.json')
validation=read(A/'validation.json')
coverage=audit['coverage'];rows=audit['days'];cases=read(A/'same_price_cases.json')
cadence=pd.read_parquet(A/'cadence.parquet')
summary=cadence.groupby('name').agg(rows=('rows','sum'),known=('known','sum'),pairs=('pairs','sum'),changes=('changes','sum'),max_distinct_in_day=('distinct','max')).reset_index()
summary['known_fraction']=summary.known/summary.rows
summary['change_fraction']=summary.changes/summary.pairs.replace(0,np.nan)
units=read(OUT/'final_parallel2/days/2026-08-24/manifest.json')['units'];summary['unit']=summary.name.map(units)
summary.to_parquet(A/'feature_review.parquet',index=False)
summary.to_json(A/'feature_review.json',orient='records',indent=2)
admitted=sum(r['admitted'] for r in rows);total_known=sum(r['total128_known'] for r in rows);strict=sum(r['strict_signed128_known'] for r in rows)
unchanged=sum(r['unchanged_mid'] for r in rows);churn=sum(r['unchanged_with_flow'] for r in rows)
checkpoints=[]
for path in sorted((A/'snapshots').glob('*.json')):
    r=read(path);r['display_time']=r['query_utc']+' UTC';checkpoints.append(r)
figures=[
('01_week.png','Неделя целиком','Разрывы торгового расписания остаются разрывами. Красная маска строгого спектра направления отбрасывает большинство активных окон; зелёная маска полного объёма не требует угадывать сторону сделки.'),
('02_midnight.png','Полночь файла не обнуляет рынок','Старый профиль терял историю VWAP после UTC-полуночи. Новый перенос разрешён только при идентичности всех ордеров книги и проверенном продолжении. Небольшие провалы масок вдали от границы — реальное ограничение свежести, а не сброс памяти.'),
('03_real_gap.png','Реальная потеря поддержки','Пять секунд в пятницу 14:00:02–14:00:06 UTC остаются неизвестными. Столбики T показывают записанные факты; допустимость их использования отмечена отдельной маской. Окна 30/128/3600 секунд восстанавливаются в разное время.'),
('04_same_price.png','Цена и короткое смещение не описывают всё состояние',f'Обе точки имеют midpoint 4653,10 USD и нулевой сдвиг за 30 секунд. Объём за эти 30 секунд — {cases["cases"][0]["trade_quantity30"]:.0f} и {cases["cases"][1]["trade_quantity30"]:.0f} контрактов. Диапазоны ограничены 8 тиками. Это описательный подбор по прошлым данным без выбора по будущему результату. Оси Y у двух примеров имеют разные пределы.'),
('05_spectral_controls.png','Спектральный фон и положительный контроль','В широкой полосе есть энергия даже у белого шума. Новый ориентир учитывает удаление среднего, окно Hann, односторонний спектр и Nyquist. Отношение агрегированных мощностей шума к ориентиру ≈1,002; синусоида с периодом 6 секунд даёт ≈3,725. Это контроль формулы, а не вероятность прибыльного движения.'),
('06_cadence.png','Частота обновления и содержательность','Измерены все 480 каналов. Спектр меняется почти каждый шаг из-за скользящего окна; это не делает соседние строки независимыми. Таблица всех каналов доступна ниже.'),
('07_l3_snapshot.png','Проверяемый снимок исходной книги','Показаны ближайшие 12 уровней с каждой стороны. На границе файла совпали все 2024 ордера, а не только лучший bid/ask. Это настоящий снимок L3; компактные признаки depth1/4/12 не заменяют полную очередь.'),
('08_model_masks.png','Что фактически доступно модели','Маски сняты с реального as-of загрузчика в 11 моментах. Цена текущей книги и исторические окна имеют отдельные условия готовности. При INVALID_BOOK нельзя использовать старую корректную котировку как текущую.')]

style='''body{margin:0;background:#f1f5f9;color:#172033;font:16px/1.55 system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:32px}h1{font-size:34px;line-height:1.2}h2{font-size:25px;margin-top:35px}section,.card{background:white;border:1px solid #d9e2ec;border-radius:12px;padding:24px;margin:20px 0}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #dce4ed}th{background:#f8fafc}code,pre{font-family:Consolas,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:14px}.ok{color:#047857}.muted{color:#64748b}.bad{color:#b91c1c}select,input{padding:9px;font:inherit;max-width:100%}.scroll{max-height:520px;overflow:auto}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.metric{padding:16px;background:#ecfdf5;border-radius:8px}.metric strong{display:block;font-size:26px}a{color:#075985}figcaption{color:#475569}nav a{margin-right:16px}small{color:#475569}'''
content=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GC — аудит признаков V2</title><style>{style}</style><main>
<h1>GC: исправления и аудит представления рынка</h1><p>Неделя 23 августа 22:00 — 28 августа 21:00 UTC, 2026 · GCZ6 · библиотека 0.4.0</p>
<p><strong>Техническая готовность ограниченной недели подтверждена с явными масками.</strong> Прибыльность, экономический таргет и готовность всей истории этим аудитом не установлены.</p>
<nav><a href="#changes">Исправления</a><a href="#observations">Входы модели</a><a href="#figures">Графики</a><a href="#limits">Ограничения</a><a href="#channels">480 каналов</a></nav>
<div class="metrics"><div class="metric"><strong>414 000 / 414 000</strong>секунд на физической сетке</div><div class="metric"><strong>413 995</strong>секунд с поддержанной ценой</div><div class="metric"><strong>480 + маски</strong>секундных признаков</div><div class="metric"><strong>{parity['full_parallel_seconds']:.1f} с</strong>расчёт и сборка из готовых источников</div></div>
<section id="changes"><h2>Что изменено</h2><ul>
<li>Шесть переносов через UTC-файлы подтверждены идентичностью полной книги; пять из них закрывают границы внутри целевой недели. Проверены оригинальные DBN-члены ZIP, исходные partitions, checkpoint и конечная книга после продолжения.</li>
<li>Дневной расчёт получает 3606 секунд левой истории. Разрыв торговой сессии, смена поддержки или несовпадающая книга прекращают непрерывность.</li>
<li>Факт последней секунды сессии отделён от возможности принять новое решение после закрытия. Поздние сделки и все зависимости двигают available_ns, включая минутный слой.</li>
<li>Завершённые BB используют SMA20 ± 2 sample SD, ddof=1. Геометрия банка пересечений с текущей развивающейся ценой остаётся отдельной и не переписана.</li>
<li>Добавлены полный объём T, классифицированный B−A, неизвестный объём и границы возможного дисбаланса. Строгий спектр направления сохранён как отдельный необязательный канал.</li>
<li>Загрузчик доступен всю сессию CME, возвращает текущий native BBO, 51 быстрый канал, 39 контекстных каналов, показатели покрытия, 480 секундных и 166 минутных признаков. Цена центрируется в float64 до float32-нормализации.</li>
<li>Цикл rolling по множеству одноэлементных закрытых сегментов заменён одним вычислением с маской границ; общая семантика сохранена.</li></ul></section>
<section><h2>Проверки и скорость</h2><table><tr><th>Проверка</th><th>Результат</th></tr>
<tr><td>Исходная лента за полную целевую неделю</td><td>{coverage['exact_trade_rows']:,} T-записей, {coverage['exact_trade_quantity']:,} контрактов; объём совпадает с секундными корзинами</td></tr>
<tr><td>VWAP / дисперсия по исходным ценам и объёмам T</td><td>600 окон, включая полночь; цена ≤{max(r['vwap_center_error'] for r in rows):.2g} USD, sigma ≤{max(r['vwap_sigma_ticks_error'] for r in rows):.2g} тика</td></tr>
<tr><td>Независимый DFT</td><td>900 окон; максимум ошибки {max(r['dft_max_error'] for r in rows):.2g}</td></tr>
<tr><td>Завершённые BB</td><td>661 независимое окно; 13 833 пары core/representation, максимум {audit['minutes']['core_representation_max_error']:.2g} USD с учётом float32 core</td></tr>
<tr><td>Перезапуск, overlap, будущий суффикс</td><td>4 320 000 сравнений; маски совпали, максимум абсолютной численной разницы {audit['partitions']['max_error']:.2g}</td></tr>
<tr><td>Один / два процесса, одинаковый профиль</td><td>{parity['serial_day_seconds']:.2f} / {parity['parallel_day_seconds']:.2f} с; ускорение {parity['speedup']:.2f}×; 16 Parquet побайтово идентичны</td></tr>
<tr><td>Последовательная сборка контекста и пяти stores</td><td>{parity['context_seconds']:.2f} с; полный параллельный прогон {parity['full_parallel_seconds']:.2f} с</td></tr>
<tr><td>Подготовка переносов / дополнительный raw-день</td><td>13,08 с / 415,77 с. Полный повторный разбор всей raw-недели в 140,48 с не включён.</td></tr>
<tr><td>RAM процесса генерации</td><td>Пиковый working set до {max(d['working_set_peak'] for d in run['days'])/1024**3:.2f} GiB; общее потребление двух процессов отдельно не измерено</td></tr>
<tr><td>Тесты / установленный пакет</td><td>{validation['tests']} passed, {validation['seconds']:.2f} с; чистый wheel, 100 модулей импортированы вне дерева исходников, старый V1 store и новый session loader проверены</td></tr></table></section>
<section><h2>Что установлено об информативности</h2><p>У {churn:,} из {unchanged:,} поддержанных секунд с неизменной mid-price ({100*churn/unchanged:.2f}%) меняется поток сделок или ближняя книга. Это дополнительное описание состояния, а не доказательство прогнозирования прибыли.</p>
<p>Среди допущенных секунд будних UTC-файлов спектр полного объёма 128s известен в {100*total_known/admitted:.2f}%, строгий спектр направления — в {100*strict/admitted:.2f}%. Улучшение означает сохранение наблюдаемого потока при честной неопределённости стороны.</p>
<p><strong>Строк с известными всеми 480 каналами: 0.</strong> Условия некоторых каналов несовместимы. Общий dropna/all-known запрещён контрактом загрузчика: используются значения и маски по отдельным каналам.</p>
<p>Для следующего опыта разумен небольшой вход из групп: цена и спред; движение и путь 30/128s; исполненный объём и границы стороны; depth1/4/12; исполнения, снятия и пополнения; положение относительно причинных VWAP/BB; качество источника и возраст наблюдений. Полный банк 480 каналов служит для абляций. Экономический эффект каждой группы проверяется на следующих хронологических неделях.</p></section>
<section id="observations"><h2>Реальные входы модели</h2><p>Выбор показывает сохранённый вход по available_ns. Серый unknown — технический ноль с ложной маской. Он не означает нулевую активность.</p><select id="snapshot"></select><pre id="state"></pre><input id="featureFilter" placeholder="Фильтр по имени признака"><div class="scroll"><table><thead><tr><th>Канал</th><th>Известен</th><th>Значение</th></tr></thead><tbody id="features"></tbody></table></div></section>
<div id="figures">'''
for name,title,caption in figures:
    encoded=base64.b64encode((A/'figures'/name).read_bytes()).decode()
    content+=f'<section><h2>{html.escape(title)}</h2><figure style="margin:0"><img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure></section>'
content+='''</div><section id="limits"><h2>Границы готовности</h2><ul>
<li>Одна неделя одного контракта. Другие месяцы, ролловер, праздничные и сокращённые сессии не проверены; текущий календарь не является верифицированным календарём всех праздников.</li>
<li>Пять секунд реальной потери поддержки книги сохранены. Для interaction ledger известны 413 974 секунды: хвосты старых sparse-файлов и двухсекундная память refill после сброса отдельно замаскированы.</li>
<li>Открывающая M1 некоторых сессий имеет неполное внутриминутное покрытие. Предыдущие уровни допускаются при явно объявленном пороге 99%, фактически 99,9275–100%; quality.previous_session_complete отличает полную сессию. Это уровни по поддержанным минутам, а не гарантированные экстремумы всего прошедшего рынка.</li>
<li>Медианные задержки секундных публикаций в будних файлах — примерно 63–81 мс, p99 — 1,91–2,48 с. Endpoint не равен моменту принятия решения. Поздние строки не становятся свежими решениями задним числом.</li>
<li>Скалярная секундная сводка не хранит весь внутрисекундный порядок событий или полную историю очередей. Native BBO и исходный MBO остаются отдельными источниками для исполнения.</li>
<li>Спектральная энергия, контакты с уровнями, withdrawal/refill — описания, не установленные причины будущего движения. Контроль на шуме не доказывает рыночную предсказуемость.</li>
<li>Новые признаки и BB требуют нового versioned dataset/checkpoint. Исторические receipts 0.3.0 и банк пересечений сохранены; положительная экономическая или policy-admission не присвоена.</li>
<li>Визуальную проверку выполнял текущий агент по восьми PNG и фактическим снимкам. Независимое второе экспертное заключение не получено.</li></ul></section>
<section id="channels"><h2>Измерения всех каналов</h2><p><a href="feature_review.json">Полная таблица JSON</a> · <a href="feature_review.parquet">Parquet</a> · <a href="audit.json">Численный аудит</a> · <a href="parallel_parity.json">Скорость и паритет</a> · <a href="admission.json">Граница допуска</a> · <a href="installed_package.json">Проверка установленного пакета</a></p><div class="scroll">'''
content+=summary[['name','unit','known_fraction','change_fraction','max_distinct_in_day']].to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)
content+='</div></section><p class="muted">Исходные, канонические, X, будущие исходы и экономика остаются разными слоями. В этом отчёте нет подбора по будущему PnL.</p></main>'
content+='<script>const observations='+json.dumps(js_safe(checkpoints),ensure_ascii=False).replace('</',r'<\/')+''';
const select=document.getElementById('snapshot'); observations.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.display_time;select.append(o)});
function render(){let r=observations[Number(select.value)],filter=document.getElementById('featureFilter').value.toLowerCase();
document.getElementById('state').textContent=JSON.stringify({time:r.display_time,current_market:r.market,context_quality:r.context_quality,tensor_shapes:r.shape,normalization:r.normalization},null,2);
let body=document.getElementById('features');body.replaceChildren();r.names.second.forEach((n,i)=>{if(!n.toLowerCase().includes(filter))return;let row=document.createElement('tr');let known=r.second_known[i];[n,known?'known':'unknown',known?Number(r.second_values[i]).toPrecision(8):'—'].forEach(v=>{let td=document.createElement('td');td.textContent=v;row.append(td)});row.className=known?'ok':'muted';body.append(row)})}
select.onchange=render;document.getElementById('featureFilter').oninput=render;render();</script></html>'''
(A/'REPORT_RU.html').write_text(content,encoding='utf8')
admission=dict(status='TECHNICAL_READY_WITH_LIMITATIONS',scope='GCZ6 ordinary venue week 2026-08-23T22:00Z..2026-08-28T21:00Z',
    source_integrity=True,physical_grid=True,causal_loader=True,per_feature_masks=True,complete_case_admission=False,
    full_history_training_admission=False,predictive_information_proven=False,economic_admission=False,policy_admission=False,
    numerical_audit_sha256=sha(A/'audit.json'),parallel_parity_sha256=sha(A/'parallel_parity.json'),installed_package_sha256=sha(A/'installed_package.json'),
    evolution_receipt_sha256=sha(ROOT/'docs/EVOLUTION_0_4_0.json'),
    visual_review_sha256=sha(A/'visual_review.json'),validation_sha256=sha(A/'validation.json'),
    experiment_code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
    limitations=['5 real unsupported seconds','UTC sparse-ledger tails plus 2s refill memory masked','Prior anchors admit >=99% supported minute coverage with quality flags',
                 'Holiday calendar and rollover not validated','Full queue histories not compressed into 480 scalar features','No independent second visual reviewer','No predictive or economic experiment in this task'])
save_json(A/'admission.json',admission)
print(json.dumps(dict(report=str(A/'REPORT_RU.html'),bytes=(A/'REPORT_RU.html').stat().st_size,status=admission['status'],
                     total_activity_admitted_pct=100*total_known/admitted,strict_direction_admitted_pct=100*strict/admitted,unchanged_price_churn_pct=100*churn/unchanged)))
