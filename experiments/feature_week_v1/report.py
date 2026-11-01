"""Build the reviewable Russian report only from completed audit receipts."""
from pathlib import Path
import sys,json,html
sys.path.insert(0,str(Path(__file__).resolve().parent))
from profile import OUTPUT, WEEK, DATES
import pandas as pd


def table(headers, rows):
    esc=lambda v:html.escape(str(v))
    return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    audit=json.loads((OUTPUT/'audit/audit.json').read_text())
    serial=json.loads((OUTPUT/'serial_v2/manifest.json').read_text())
    parallel=json.loads((OUTPUT/'parallel2/manifest.json').read_text())
    if any(r['status'] not in ('PASS','COMPLETE') for r in (audit,serial,parallel)):
        raise ValueError('Incomplete run/audit')
    cadence=pd.read_parquet(OUTPUT/'audit/cadence.parquet').groupby('name')[['rows','known','pairs','changes']].sum()
    c=audit['coverage']; days=audit['day_audits']
    unchanged=sum(d['unchanged_mid_seconds'] for d in days)
    informative=sum(d['unchanged_mid_with_flow'] for d in days)
    total_rows=sum(d['rows'] for d in days)
    pubs=sum(d['complete_publications'] for d in days)
    changes=sum(d['bbo_changes'] for d in days)
    chosen=['spread_ticks','imbalance_l1','depth_balance_12','trade_buy_1s','trade_buy_30s',
            'displacement_30s','fft_mid_change_128s_power','fft_signed_trade_128s_power',
            'vwap_exact_60s_center','vwap_exact_900s_center','vwap_exact_3600s_center']
    speed=serial['total_elapsed_seconds']/parallel['total_elapsed_seconds']
    size=sum(p.stat().st_size for p in (OUTPUT/'serial_v2').rglob('*') if p.is_file())/1e9
    parts=['''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Готовность недельных признаков GCZ6</title>
<style>body{font:17px/1.55 system-ui,sans-serif;color:#203048;background:#f5f7fb;max-width:1180px;margin:45px auto;padding:0 26px}h1{font-size:34px;line-height:1.2}h2{margin-top:42px;color:#174c77}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{border-bottom:1px solid #dce3ec;padding:10px;text-align:left}th{background:#e7eef7}img{width:100%;background:white;margin:14px 0}code{font-size:14px}a{color:#1768ac}.lead{font-size:21px}.box{background:white;border-left:5px solid #1768ac;padding:18px 24px}small{color:#556478}</style>
<h1>Готовность данных: проверенная неделя GCZ6</h1><p>29 октября 2026 · торговая неделя 24–28 августа · исходный маршрут (1, 41271, 22)</p>
<p class="lead">Недельный профиль сгенерирован, проверен численно и сравнен между одним и двумя процессами.
Это материализация наблюдений с явными ограничениями. Допуск к обучению игрока и экономическая полезность требуют следующих отдельных проверок.</p>''']
    parts.append(f'''<div class="box"><b>Объём:</b> {total_rows:,} секундных строк пяти UTC-дней;
409 признаков с 409 масками; 166 минутных признаков с 166 масками;
51 канал быстрого observation и 36 каналов дневного контекста; зоны и события.
Отдельно сохранены пятница 21 августа и воскресенье 23 августа.
Один итоговый run занимает {size:.2f} GB. Значения не являются независимыми 409 источниками информации.</div>''')
    parts.append('''<h2>Что означает цена в каждый момент</h2><p>На исходной шкале есть поток завершённых MBO-публикаций с bid/ask, размерами, source ordinal и epoch.
Между публикациями известна последняя допустимая котировка и её возраст. Это не интерполированная цена и не гарантия исполнения по midpoint.
Секундная сетка — слой наблюдений; она не заменяет исходный порядок движений внутри секунды. Для first-hit и исполнения нужен исходный quote tape.</p>''')
    parts.append(f'''<p>В пяти UTC-днях учтено {pubs:,} завершённых публикаций в торговые часы и {changes:,} изменений BBO.
Все проверенные поддержанные секундные BBO совпали с последней строго более ранней завершённой котировкой.</p>''')
    parts.append('<h2>Скорость и параллелизация</h2>')
    parts.append(table(['Режим','Дневные jobs','Контекст + loader','Весь профиль'],[
        [str(r['workers'])+' процесс(а)',f"{r['days_elapsed_seconds']:.2f} s",f"{r['context']['elapsed_seconds']:.2f} s",f"{r['total_elapsed_seconds']:.2f} s"] for r in (serial,parallel)]))
    parts.append(f'''<p>Ускорение всего профиля: <b>{speed:.2f}×</b>. Два процесса дают одинаковые данные:
{audit['parity']['files']} Parquet-файла сравнили значения; {audit['parity']['byte_identical']} совпали побайтно.
Пиковый working set дневного worker: {max(d['working_set_peak'] for d in parallel['day_results'])/2**30:.2f} GiB.
Это по одному замеру на режим с прогретым файловым кэшем; интервал статистической уверенности не оценивался.</p>
<p>Время включает проверку SHA, подготовку точных моментов, формулы, запись, чтение результата и context loader.
Оно <b>не включает новое восстановление всей недели из DBN</b>: пять будних MBO-sidecars повторно использованы после проверки SHA и partition identity.
Пятница 21 августа восстановлена заново: 7 016 290 записей за 522,53 s; воскресенье: 371 227 за 33,17 s.
Оба новых raw-run дали нулевой остаток stock reconciliation.</p>
<p>Главный измеренный узкий участок — rolling по многочисленным отдельным закрытым секундам воскресенья.
Общая сборка контекста сейчас выполняется последовательно. Ежедневные jobs независимы только при явном source reset и корректной книге;
история предыдущего дня/недели передаётся отдельной зависимостью. Межнедельный запуск возможен по той же схеме.
Часовой перерыв и выходные сами по себе не дают права забыть дневные уровни и накопленные профили.</p>''')
    parts.append('<h2>Покрытие и смысл масок</h2>')
    parts.append(table(['Проверка','Результат'],[
        ['Ожидаемые открытые секунды от Sun 22:00 до Fri 21:00 UTC',c['expected_open_seconds']],
        ['Присутствуют в секундных таблицах',c['present_open_seconds']],
        ['Поддержанная цена по текущему контракту',c['known_open_seconds']],
        ['Отсутствуют физические строки',c['missing_open_seconds']],
        ['Строка есть, но состояние unknown',c['present_but_unknown_open']],
        ['Все 409 масок одновременно true, пять UTC-дней',sum(d['all_features_known'] for d in days)],
    ]))
    parts.append('<p>Пропущенные endpoints: '+html.escape(', '.join(c['missing_endpoints_utc']))+'''. Это границы UTC captures: EOF не доказывает ненаблюдаемый хвост.
Нужно либо продолжить replay через проверенный переход, либо оставить gap и честный прогрев. Не заполнять пропуск последней ценой как наблюдённым фактом.</p>''')
    parts.append('<p>В raw T-tape пяти дней 834 770 контрактов; секундный профиль содержит 834 762. Восемь контрактов в завершающих неполных секундах не превращены в завершённые buckets и сохранены в исходной ленте сделок.</p>')
    parts.append('''<p>Существующий observation loader публикует решения начиная с London 08:00; его fast-окно покрывает этот decision day.
Полные круглосуточные секундные таблицы существуют отдельно. Для игрока на всей CME-сессии нужен общий session-aware loader.
Предыдущая полная ISO-неделя в fixture отсутствует, поэтому три соответствующих core-поля и их маски обнулены.
Календарь обычных часов поддерживается, покрытие праздников общего календаря помечено NOT_COVERED.</p>''')
    parts.append('''<h2>Как часто меняются признаки</h2><p>Номинальный clock — 1 Hz, но одинаковые строки не превращаются в независимые учебные примеры.
Изменение измерялось между соседними поддержанными секундами, в одном epoch; допуск сравнения 1e-9 + 1e-10 относительно величины.</p>''')
    parts.append(table(['Признак','Known / все строки','Изменился / пары known'],[
        [n,f"{100*cadence.loc[n,'known']/cadence.loc[n,'rows']:.2f}%",f"{100*cadence.loc[n,'changes']/cadence.loc[n,'pairs']:.2f}%"] for n in chosen]))
    parts.append('''<p>Core и OHLC публикуются по завершённым M1. Native BB обновляются после завершённой M1/M3/M6/M9;
между обновлениями применим только уже опубликованный вариант. M1-спектр использует 128 прошлых минут.
События публикуются в момент распознавания; состояния зон также имеют собственную версию и available_ns.</p>''')
    parts.append('<h2>Проверки формул и причинности</h2>')
    parts.append(table(['Проверка','Результат'],[
        ['VWAP и σ напрямую из T-принтов',f"{sum(d['exact_checks'] for d in days)} окон; max error центра {max(d['exact_center_max_error'] for d in days):.2e} USD"],
        ['σ по T-принтам',f"max error {max(d['exact_sigma_ticks_max_error'] for d in days):.2e} ticks"],
        ['Оконный DFT цены',f"{sum(d['fft_checks'] for d in days)} окон; max error {max(d['fft_max_error'] for d in days):.2e}"],
        ['BB representation, ddof=0',f"{audit['minute']['bb_checks']} окон; max error {audit['minute']['bb_max_error']:.2e} USD"],
        ['BB core, ddof=1',f"{audit['minute']['core_bb_checks']} окон; max error {audit['minute']['core_bb_max_error']:.2e} USD"],
        ['Chunk + checkpoint/restore',f"409 каналов; маски совпали; max error {audit['partition']['chunk_restart_max_error']:.2e}"],
        ['Добавление будущих строк',str(audit['partition']['future_prefix_equal'])+' — прежний prefix не изменился'],
        ['Сброс состояния в середине сессии',str(audit['partition']['split_without_state_lost_known_rows']['vwap_exact_3600s_center'])+' потерянных ранее known строк VWAP 3600s'],
    ]))
    parts.append('''<p>Сдвиги цены, длины пути, глубина и 30-секундные объёмы сверены на всех допускаемых строках.
Проверены конечные значения, нули при false-mask, временной порядок и загрузка as-of. Prefix/checkpoint тест ограничен реальным трёхчасовым фрагментом;
это не формальное доказательство всего стека.</p><p><b>Геометрия BB требует явного выбора:</b> sample SD при 20 значениях шире population SD в √(20/19) ≈ 1,026 раза.
Обе реализации численно корректны в своих контрактах. Core хранится в float32; его независимая сверка учитывает округление (rtol=1e-7, atol=1e-8 USD).
Для labels и execution следует закрепить одну версию.</p>''')
    parts.append('<h2>Визуальный аудит содержательности</h2>')
    parts.append(f'''<p>Из {unchanged:,} секунд с неизменившимся midpoint в {informative:,} ({100*informative/unchanged:.1f}%)
есть сделки или изменение ближней книги. Это прямой пример информации сверх одной цены.
При этом большой объём снятий не доказывает направленный спрос: книга может интенсивно обновляться без сдвига.</p>
<p>Смотреть следует на согласованность масштабов и масок: часовой VWAP не должен мгновенно следовать за тиками,
спектр не должен означать направление, ночная тишина не должна становиться вымышленной активностью,
прогрев не должен выглядеть известным нулём. Число признаков и визуальная правдоподобность ещё не показывают предиктивный edge.</p>''')
    signed=cadence.loc['fft_signed_trade_128s_power']
    parts.append(f'''<div class="box"><b>Наблюдения при просмотре всех четырёх графиков:</b><ul>
<li>В пятницу около 14:00 UTC растут и сделки, и изменения книги; высокая частота снятий намного превышает связанные исполнения.
Баланс глубины быстро колеблется и сам по себе не задаёт надёжного направления.</li>
<li>В 14:00:02–14:00:06 пятницы пять секунд имеют неподдержанную книгу. Есть 813 invalid complete publications в окружающем 30-секундном фрагменте.
Возврат valid в 14:00:07 запускает новый прогрев. Это важный режим, а не основание незаметно убрать весь участок.</li>
<li>Остальные пять unknown открытых interval-end строк — окончания сессий в 21:00 UTC:
интервал до них торговый, но endpoint уже закрыт. Следует разделять известность завершённого исторического bucket и разрешение нового решения.</li>
<li>Signed-flow FFT 128s имеет только {100*signed.known/signed.rows:.2f}% покрытия.
Любая неизвестная сторона T в окне исключает весь направленный спектр. Все 409 одновременно known: {100*sum(d['all_features_known'] for d in days)/total_rows:.2f}%.
Требование общего complete-case набора создаст сильную селекцию; маски должны оставаться по семействам.</li>
<li>Полоса 2–8 минут охватывает 48 из 64 ненулевых FFT-бинов. Доля около 75% во многом соответствует её ширине.
Такая картинка не доказывает наличие циклов. Пунктир показывает ориентир равномерной мощности; для смысла нужны превышение ориентира и null-сравнения.</li>
<li>Exact и proxy VWAP 300s визуально близки. Это контроль правильности, а не основание считать их независимыми сигналами.</li>
</ul></div>''')
    for name,caption in [('01_week.png','Неделя: цена, объём, минутный спектр и покрытие.'),
                         ('02_market_detail.png','Фиксированное окно пятницы: цена, точные сделки, глубина, снятия и задержки.'),
                         ('03_boundaries.png','Разные причины прогрева на UTC-границе и часовом перерыве.'),
                         ('04_cadence_speed.png','Частота реальных изменений и измеренная стоимость генерации.')]:
        parts.append(f'<p><b>{caption}</b></p><img src="figures/{name}" alt="{caption}">')
    parts.append('''<h2>Что требуется до обучения игрока</h2><ol>
<li>Закрепить одну геометрию BB/crossings, единицы и clock решений, включая поведение при late/unknown.</li>
<li>Собрать session-aware loader для объявленного торгового интервала; передавать весь нужный прошлый контекст и masks.</li>
<li>Разрешить UTC-переходы через replay/checkpoint либо сделать явный admission для прогрева; дополнить праздничный календарь при масштабировании.</li>
<li>Сохранить event-level котировки для first-hit и исполнения. При необходимости очереди добавить адресуемое состояние уровней, не подменять его агрегатным depth.</li>
<li>На хронологических разделах проверить добавочную информацию: цена+время → точные сделки → глубина → изменения уровней → спектр.
Цель и последовательная политика проверяются отдельно; штраф за редкие неудачи не должен превращать всю задачу в постоянный WAIT.</li>
</ol><p>Успех этого опыта — проверяемое наблюдение рынка. Следующий эксперимент должен выяснить, улучшает ли оно выбор действия и итоговую стоимость портфеля после расходов.</p>
<p><small>Исходные receipts: audit.json, ../serial_v2/manifest.json, ../parallel2/manifest.json.
Формулы среды закреплены baseline; графики используют отдельную plotting-среду.
Официальные ссылки: <a href="https://www.cmegroup.com/markets/metals/precious/gold.timeAndSales.html">CME GC hours/specs</a>;
<a href="https://numpy.org/doc/stable/reference/generated/numpy.std.html">NumPy ddof</a>.</small></p></html>''')
    out=OUTPUT/'audit/REPORT_RU.html'
    out.write_text('\n'.join(parts),encoding='utf8')
    print(str(out))


if __name__=='__main__':main()
