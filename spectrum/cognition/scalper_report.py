"""Readout of the continuous scalper redesign (reads scalper_summary.json only)."""
from pathlib import Path
import json
import pandas as pd
from .scalper_policy import REPORT, THRESHOLD, GATE_SPREAD_TICKS, COST_MULT


def ci(d):
    if not d:
        return '—'
    return f"{d['mean']:+.2f} [{d['lower']:+.2f}; {d['upper']:+.2f}]"


def markdown(s):
    d = pd.DataFrame(s['diagnostics'])
    L = ['# Скальпер: непрерывные решения, выходы в единицах волатильности, фильтр спреда', '',
         f"{s['days']} train-дней, {s['rows']:,} секундных точек решения. A = окт–дек 2025 (обучение), B = янв–фев 2026 (проверка). "
         'Validation/evaluation не затронуты.', '',
         '## Что изменено', '',
         '- **Точки решения:** каждая секунда окна Лондон–Нью-Йорк вместо событий и сетки 5 минут.',
         '- **Выходы:** стоп и тейк — сдвиг mid на m·u, где u = σ600·√H тиков (H = 30/120/600 с; формы 1.0/1.0 и 1.5/0.75). '
         'Расширение спреда само по себе стоп не выбивает. Исполнение выхода — тейкером по исполнимой цене.',
         f'- **Фильтры входа:** спред ≤ {GATE_SPREAD_TICKS:.0f} тиков и ожидаемый тейк ≥ {COST_MULT:.0f}×(спред + комиссия 3 тика).',
         '- **Входы:** тейкер (T) или очередь на 30 с по ядру V5 (P; неисполненная попытка = $0).', '',
         '## 1. Без модели: стоимость попытки по всем секундам', '',
         '| План | Вход | Блок | без фильтра $ | фильтр спреда $ | спред+издержки $ | доля секунд с обоими фильтрами |',
         '|---|---|---|---:|---:|---:|---:|']
    for (plan, mode, blk), g in d.groupby(['plan', 'mode', 'block'], sort=False):
        v = {r.gate: r for r in g.itertuples()}
        L.append(f"| {plan} | {mode} | {blk} | {v['none'].attempt:+.3f} | {v['spread'].attempt:+.3f} | "
                 f"{v['spread+cost'].attempt:+.3f} | {v['spread+cost'].share:.3f} |")
    L += ['', 'Адаптивные планы (a…) и старые фиксированные (f…) при входе в случайную секунду стоят почти одинаково: без выбора '
          'момента форма выхода не создаёт преимущества. Фильтр спреда удешевляет тейкерскую попытку; на пассив почти не влияет.', '',
          '## 2. Непрерывный счёт 1 oz (модель стоимости попытки на A)', '',
          '| Вариант | сделок | исполнено | net $ | $ на сделку | сделок в день | дневной net [95% дни] | макс. просадка $ |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k, title in (('policy_A', 'политика, A (в выборке)'), ('policy_B', 'политика, B'), ('random_B', 'случайные входы той же частоты, B'),
                     ('permuted_B', 'модель на перемешанной цели, B')):
        v = s[k]
        if not v.get('trades'):
            L.append(f'| {title} | 0 | 0 | 0 | — | 0 | — | 0 |'); continue
        L.append(f"| {title} | {v['trades']} | {v['filled']} | {v['net']:+.1f} | {v['per_trade']:+.3f} | {v['per_day_trades']:.1f} | "
                 f"{ci(v['daily'])} | {v['max_drawdown']:.1f} |")
    L += ['', f'Порог входа — прогноз > {THRESHOLD:+.2f}$. Чувствительность к порогу на B:', '',
          '| Порог $ | сделок | net $ | $ на сделку |', '|---:|---:|---:|---:|']
    for th in ('0.05', '0.1', '0.2'):
        v = s[f'policy_B_threshold_{th}']
        L.append(f"| {th} | {v['trades']} | {v['net']:+.1f} | " + (f"{v['per_trade']:+.3f}" if v.get('per_trade') is not None else '—') + ' |')
    pb, rb = s['policy_B'], s['random_B']
    def att(plan, mode, blk, gate):
        return float(d[(d.plan == plan) & (d['mode'] == mode) & (d.block == blk) & (d.gate == gate)].attempt.iloc[0])
    L += ['', 'Пороги выше нуля выбраны после просмотра B — это чувствительность, не результат.', '',
          '## Вывод', '',
          f"- **Фильтр спреда — главный выигрыш редизайна.** Тейкерская попытка a600s в B: {att('a600s', 'T', 'B', 'none'):+.2f}$ → "
          f"{att('a600s', 'T', 'B', 'spread'):+.2f}$; пассивная: {att('a600s', 'P', 'B', 'none'):+.2f}$ → {att('a600s', 'P', 'B', 'spread'):+.2f}$. "
          'Адаптивные и фиксированные выходы при случайном входе стоят почти одинаково.',
          f"- **Политика на B:** {pb['trades']} сделок, {pb['net']:+.1f}$, {pb['per_trade']:+.3f}$ на сделку против {rb['per_trade']:+.3f}$ "
          f"у случайных входов той же частоты ({rb['net']:+.1f}$). Отбор моментов частично переносится, но дневной результат "
          f"{ci(pb['daily'])} включает ноль и в плюс не выходит.",
          '- **Предел текущих признаков.** Улучшение не превышает издержек круга: сигнала направления, перекрывающего комиссию $0.30 и спред, '
          'в нынешних признаках нет.', '',
          'Development evidence на train-месяцах; экономического допуска нет.']
    return '\n'.join(L) + '\n'


def report(output=REPORT):
    s = json.loads((Path(output) / 'scalper_summary.json').read_text(encoding='utf8'))
    (Path(output) / 'SCALPER_RU.md').write_text(markdown(s), encoding='utf8')
    return Path(output) / 'SCALPER_RU.md'


if __name__ == '__main__':
    print(report())
