"""Readout of the sequential scalper with passive take-profit (reads live_summary.json)."""
from pathlib import Path
import json
from .scalper_live import REPORT


def ci(d):
    return '—' if not d or d.get('lower') is None else f"{d['mean']:+.2f} [{d['lower']:+.2f}; {d['upper']:+.2f}]"


def row(title, v):
    if not v or not v.get('trades'):
        return f'| {title} | 0 | 0 | 0 | — | — | 0 | — | 0 | — |'
    pt = '—' if v['per_trade'] is None else f"{v['per_trade']:+.3f}"
    wr = '—' if v['win_rate'] is None else f"{v['win_rate']:.3f}"
    hold = '—' if v['median_hold_s'] is None else f"{v['median_hold_s']:.0f}"
    return (f"| {title} | {v['trades']} | {v['filled']} | {v['net']:+.1f} | {pt} | {wr} | {v['per_day']:.1f} | "
            f"{ci(v['daily'])} | {v['max_drawdown']:.1f} | {hold} |")


def markdown(s):
    head = '| Вариант | сделок | исполнено | net $ | $ на сделку | доля прибыльных | сделок в день | дневной net [95% дни] | макс. просадка $ | медиана удержания, с |'
    sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    L = ['# Скальпер на одном счёте: пассивный take-profit и порог, выбранный без B', '',
         'Каждая реально выбранная сделка симулируется по ленте MBO. Вход — тейкером или в очередь, take-profit — лимитка в очереди '
         '(ядро V5, инвариант исполнений). Стоп, лимит времени и дедлайн — тейкером по сдвигу mid. Реальный выход определяет, '
         'когда счёт снова свободен.', '',
         f"Разбиение: модели на {', '.join(s['split']['A1'])}; порог по {', '.join(s['split']['A2'])} (A2); затем модели на всём A, "
         f"проверка на B ({', '.join(s['split']['B'])}). B, validation и evaluation при выборе не использовались.", '',
         '## 1. Выбор порога на A2 (модели A1, пассивный take-profit)', '', head, sep]
    for th in s['thresholds']:
        L.append(row(f'порог {th:.2f}$' + (' ← выбран' if th == s['chosen_threshold'] else ''), s['selection_A2'].get(f'A2_th{th}')))
    b = s['B']
    L += ['', f"## 2. Проверка на B (модели A, порог {s['chosen_threshold']:.2f}$)", '', head, sep,
          row('политика, пассивный take-profit', b.get('policy_passive_tp')),
          row('та же политика, take-profit тейкером', b.get('policy_taker_tp')),
          row('случайные входы, то же число сделок по дням, пассивный take-profit', b.get('random_passive_tp'))]
    pp, pt, rr = b.get('policy_passive_tp', {}), b.get('policy_taker_tp', {}), b.get('random_passive_tp', {})
    sel = s['selection_A2']; best = max(v['net'] for v in sel.values())
    L += ['', '## Оговорки выбора', '',
          f"- Все пороги на A2 убыточны (лучший {best:+.1f}$). WAIT ($0) лучше любого из них: при правиле, включающем WAIT, "
          'решение на A — не торговать. Проверка на B ниже — проверка варианта, который A уже отвергает.',
          f"- Выбран порог {s['chosen_threshold']:.2f}$ — верхняя граница сетки; результат на A2 монотонно улучшался с ростом порога."]
    L += ['', '## Вывод', '']
    if pp.get('trades'):
        L.append(f"- Политика на B: {pp['trades']} сделок, {pp['net']:+.1f}$, дневной результат {ci(pp['daily'])}.")
        if pt.get('per_trade') is not None and pp.get('per_trade') is not None:
            L.append(f"- Пассивный take-profit против тейкерского на тех же правилах: {pp['per_trade']:+.3f}$ против {pt['per_trade']:+.3f}$ на сделку.")
        if rr.get('per_trade') is not None:
            L.append(f"- Случайные входы с тем же числом сделок: {rr['net']:+.1f}$ ({rr['per_trade']:+.3f}$ на сделку).")
        ok = pp['daily'] and pp['daily'].get('lower') is not None and pp['daily']['lower'] > 0
        L.append('- Дневной интервал политики ' + ('выше нуля.' if ok else 'не выше нуля: экономического преимущества не установлено.'))
    else:
        L.append('- На B при выбранном пороге политика не совершила сделок.')
    L += ['', 'Development evidence на train-месяцах; экономического допуска нет.']
    return '\n'.join(L) + '\n'


def report(output=REPORT):
    s = json.loads((Path(output) / 'live_summary.json').read_text(encoding='utf8'))
    (Path(output) / 'LIVE_RU.md').write_text(markdown(s), encoding='utf8')
    return Path(output) / 'LIVE_RU.md'


if __name__ == '__main__':
    print(report())
