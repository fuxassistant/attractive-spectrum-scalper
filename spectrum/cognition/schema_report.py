"""Readable readout: representation inventory, Q ablation, schema library, controls."""
from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd
from .representation import DEFAULT_OUTPUT as REPRESENTATION, family_columns
from .schema_run import REPORT


def day(ns):
    return str(datetime.fromtimestamp(int(ns) / 1e9, timezone.utc).date())


def known_rates(representation=REPRESENTATION):
    rep = pd.concat([pd.read_parquet(p) for p in sorted(Path(representation).glob('*.parquet'))], ignore_index=True)
    fam = family_columns(rep.columns); rows = []
    for name in ('price', 'flow', 'micro'):
        masks = [c for c in fam[name] if c.endswith('_known')]
        values = [c for c in fam[name] if not c.endswith('_known')]
        rows.append((name, len(values), float(rep[masks].mean().mean()), float(rep[masks].mean().min()),
                     min(masks, key=lambda c: rep[c].mean())))
    return rows, len(rep)


def markdown(report=REPORT):
    out = Path(report)
    q = json.loads((out / 'qmodel_ablation.json').read_text(encoding='utf8'))
    s = json.loads((out / 'schema_summary.json').read_text(encoding='utf8'))
    lib = json.loads((out / 'library_full.json').read_text(encoding='utf8'))
    rates, origins = known_rates()
    v = s['variants']; L = []
    L += ['# Модель, схемы и представление рынка — train, A→B', '',
          'A = окт–дек 2025, B = янв–фев 2026, обе части внутри train. Validation/evaluation не читались. '
          'Всё ниже — development evidence на ранее изученной истории.', '',
          '## 1. Какое представление рынка доступно', '',
          '| Семейство | Источник | Полей | Средняя известность | Худшее поле |', '|---|---|---:|---:|---|',
          '| vol | quote grid: σ 60/600/3600 с, спред, часы сессии | 5 | ≈1.00 | sigma3600 в начале сессии |']
    src = dict(price='quote grid: доходности в σ, позиция в диапазоне 5–60 мин, якоря сессии',
               flow='MBO seconds: поток buy/sell/unknown 5–60 мин с границами, объём, add/remove, глубина L1/L4/L12',
               micro='компактный M1: 30 секундных полей стакана, потока и FFT')
    for name, n, mean, worst, col in rates:
        L.append(f'| {name} | {src[name]} | {n} | {mean:.3f} | {col} = {worst:.3f} |')
    L += ['| event | event_landmarks run_003: 50 полей события, контекст прошлых событий | 54 | 6 дат без покрытия маскируются | — |', '',
          f'Точек входа: {origins}. Всё причинно на момент намерения: сетка — граница ≤ t, секунды — по available_ns, '
          'M1 — по времени решения, окна счётчиков только внутри одной эпохи.', '',
          'Не доступно в текущих данных: точный trade-price VWAP (есть только proxy), адресная память уровней L3 '
          '(нужна отдельная версионная работа над finalized L3), опционы/IV, события на непокрытых датах.', '',
          '## 2. Q-модель: какое представление включать', '',
          'Цель разложена на m=(L+S)/2 (стоимость/волатильность) и d=(L−S)/2 (направление); Q(long)=m+d, Q(short)=m−d, WAIT=0. '
          'HistGradientBoosting, гиперпараметры зафиксированы до чтения B. R² направления против нуля, попадание взвешено |d|.', '',
          '| Набор | полей | R² d на A (30 мин) | R² d на B (5/10/30 мин) | попадание B (30 мин) | политика B: сделок, $ | та же со случайной стороной $ |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in q.items():
        if 'skill_A' not in r:
            continue
        a = r['skill_A']['(80, 40, 1800)']['r2_vs_zero']; b = r['skill_B']
        L.append(f"| {name} | {r['features']} | {a:+.4f} | " + ' / '.join(f"{b[k]['r2_vs_zero']:+.4f}" for k in b) +
                 f" | {b['(80, 40, 1800)']['dollar_weighted_hit']:.3f} | {r['policy_B']['trades']}, {r['policy_B']['net_sum']:+.0f} | "
                 f"{r['policy_B_random_side']['net_sum']:+.0f} |")
    L += ['', 'На A все наборы «находят» направление (до R² 0.08 и +$8.5k in-sample), на B — ни один: R² ≤ 0, попадание ≈ 0.49–0.53. '
          'Ridge (α 10²/10⁴, цель в единицах σ) даёт то же. Вывод: в доступных семействах нет переносимого A→B сигнала направления; '
          'для величины движения достаточно vol. Включать в схемы стоит малый набор интерпретируемых координат из price/flow, '
          'а не 190 полей — большая сеть только переобучится.', '',
          '## 3. Что даёт скрипт: свобода и адаптивность', '',
          '**signatures.py / signature_report.py** — без адаптивности: фиксированное меню 2×4, фиксированные кластеры, одна регрессия. '
          'Это таблица знаний, не агент.', '',
          '**schemas.py** — библиотека схем. Свобода агента:',
          f"- 16 корней: {len(s['programs'])} программ × 2 плана ({', '.join(s['programs'])}); планы 40/20/600 и 80/40/1800.",
          f"- Контекст — конъюнкция до 2 координат из {len(s['coordinates'])}: {', '.join(s['coordinates'])} (по 3 значения).",
          '- Не может: менять планы выхода, размер (1 oz), изобретать новые программы, ADD/REDUCE, управлять открытой позицией.', '',
          '**Ассимиляция:** каждый созревший (после exit) исход каждой ветви обновляет все применимые схемы off-policy; '
          'среднее с полураспадом 40 дней, история не удаляется; убытки входят всегда.',
          '**Аккомодация:** таблица маржинальной атрибуции Дрешера — остатки y−μ по каждому значению координаты вне контекста. '
          'Раз в 5 дней ячейка с дневной кластеризованной |t| ≥ 4.0 (≥80 исходов, ≥10 дней) рождает ребёнка.',
          '**Карантин:** ребёнок учится только на точках после рождения; ровно через 15 свежих дней один тест '
          '(t ≥ 2 в сторону триггера) — подтверждение или опровержение.',
          '**Перераспределение заслуги:** у родителя с подтверждёнными детьми решения опираются на остаток вне их контекстов.',
          '**Быстрое опровержение:** схема не действует, пока среднее последних 5 дней ≤ 0, но продолжает учиться.',
          '**Роутер:** один счёт 1 oz; действует лучшая схема с LCB = μ − 2.5·SE(дни) > 0; иначе WAIT.', '',
          '## 4. Контроли и что они показали', '',
          '| Вариант | рождено | подтверждено | опровергнуто | сделок A | сделок B | net B $ |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in v.items():
        L.append(f"| {name} | {r['spawned']} | {r['promoted']} | {r['refuted']} | {r['A']['trades']} | {r['B']['trades']} | {r['B']['net']:+.1f} |")
    pc = s['planted_control']
    L += ['', f"planted_* — синтетический положительный контроль на реальном потоке: в {pc['start']}…{pc['end']} при "
          f"{pc['coordinate'][0]}={pc['coordinate'][1]} лонг плана 80/40/1800 получает +${pc['usd']}, шорт −${pc['usd']} "
          f"({pc['origins']} точек); в феврале преимущества нет (REGIME_CHANGE_AND_RETURN).", '']
    pf = v['planted_full']
    act = [e for e in pf['events'] if e['kind'] == 'active']
    spawn = [e for e in pf['events'] if e['kind'] == 'spawn' and 'range' in e['describe'] and 'long' in e['describe']]
    L += [f"- Обнаружение: первая верная гипотеза {spawn[0]['describe']} родилась {day(spawn[0]['ns'])}; "
          f"подтверждена {day(act[0]['ns'])} (режим начался {pc['start']})." if spawn and act else '- Обнаружение: нет.',
          f"- Помесячно full: {json.dumps({m: round(x['net'], 1) for m, x in pf['monthly'].items()})}; "
          f"no_forgetting: {json.dumps({m: round(x['net'], 1) for m, x in v['planted_no_forgetting']['monthly'].items()})}.",
          '- assimilate_only и frozen_after_A не торгуют: корень усредняет контекст с преимуществом и без него; '
          'без аккомодации знание не выделяется. Перемешанный контекст: ложные рождения опровергнуты, подтверждений 0.', '',
          'Настройки бдительности/роутера найдены по синтетическим мирам (20 нулевых зёрен — 0 ложных сделок, '
          '5 подсаженных — все находят преимущество), не по реальной прибыли.', '',
          '## 5. Что схемы выучили на реальных данных', '']
    roots = sorted([k for k in lib['schemas'] if k['parent'] < 0], key=lambda k: -(k['mean_all'] or -9))
    L += ['| Корневая схема | исходов | дней | среднее $ | среднее с забыванием $ |', '|---|---:|---:|---:|---:|']
    for k in roots:
        L.append(f"| {k['describe']} | {k['n']} | {k['days']} | {k['mean_all']:+.3f} | {k['mean_decayed']:+.3f} |")
    L += ['', 'Гипотезы аккомодации (full):', '']
    for e in lib['events']:
        extra = f" t={e['t']:+.2f}, остаток {e['mean_residual']:+.2f}$" if e['kind'] == 'spawn' else \
            f" t={e.get('confirm_t', 0):+.2f}, {e.get('reason', '')}"
        L.append(f"- {day(e['ns'])} {e['kind']}: {e['describe']}{extra}")
    err = v['full']['expectation_error']
    L += ['', 'Навык ожиданий против бесконтекстного среднего по месяцам: ' +
          ', '.join(f"{m}: {x['skill']:+.4f}" for m, x in err.items()) + ' (перемешанный контроль — тот же порядок).', '',
          '## 6. Вывод', '',
          '- Механизм работает: в подсаженном мире находит контекст, подтверждает его, торгует и останавливается после ухода режима; '
          'в нулевом мире молчит.',
          '- На реальном GC (окт 2025 – фев 2026) ни одна схема не дошла до LCB > 0. Наименее убыточны контртрендовые программы '
          '(против потока/тренда, возврат в диапазон) — слабый возврат к среднему, в разы меньше издержек круга.',
          '- Найденные в феврале гипотезы — знание «где хуже», а не новое преимущество; они в карантине.',
          '- Измеренная адаптация: ≈5 недель от смены режима до действия, ≈2.5 недели до остановки после возврата режима.', '',
          'Независимые от прибыли выводы — проверка механизма; экономического допуска нет.']
    (out / 'SCHEMAS_RU.md').write_text('\n'.join(L) + '\n', encoding='utf8')
    return out / 'SCHEMAS_RU.md'


if __name__ == '__main__':
    print(markdown())
