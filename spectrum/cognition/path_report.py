"""Readout of scalper mind stage 1 (reads stage1_summary.json and null_grid.json) and the null-grid calibration.

null_grid: for symmetric barriers +-d around the entry mid (d = c * sigma600 or k ticks), the share of
entries whose first barrier is hit by t against the finite-horizon Brownian null, and which barrier comes
first. After a taker entry the side is irrelevant (up vs down); after a queue fill the adverse barrier is
the one against the filled side: an excess over 1/2 is the counterparty's information, seen in the path.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from spectrum.ingest import save_json
from .path_scan import DEFAULT_OUTPUT as PATH_DATA, EDGES, VOL_LEVELS, ABS_LEVELS, END_SPAN
from .path_model import REPORT, SPLIT, dates_of
from . import path_null as pn

LADDER_NAMES = ('T', 'P_long', 'P_short')


def null_grid(dates, data=PATH_DATA):
    acc = {}
    for d in dates:
        z = np.load(Path(data) / f'{d}.npz'); sig = pd.read_parquet(Path(data) / f'{d}.parquet', columns=['sigma600']).sigma600.to_numpy(float)
        up, dn, end = z['up_t'], z['dn_t'], z['end_t']
        for li in range(3):
            for col in range(up.shape[2]):
                kind, lvl = ('vol', VOL_LEVELS[col]) if col < len(VOL_LEVELS) else ('abs', ABS_LEVELS[col - len(VOL_LEVELS)])
                dist = pn.lattice(lvl * sig if kind == 'vol' else np.full(len(sig), float(lvl)))
                u, w, e = up[:, li, col], dn[:, li, col], end[:, li]
                live = np.isfinite(e)
                tau = np.fmin(u, w); first_up = np.where(np.isfinite(u) & (~np.isfinite(w) | (u < w)), 1., 0.)
                key = (LADDER_NAMES[li], kind, lvl); a = acc.setdefault(key, dict(n=0, by_t={t: [0, 0., 0.] for t in EDGES}, up=0., ev=0))
                a['n'] += int(live.sum())
                for t in EDGES:
                    seen = live & ((tau <= t) | (e >= t))
                    hit = seen & (tau <= t)
                    p = 1 - pn.two_barrier(dist[seen], dist[seen], sig[seen], float(t))[2]
                    a['by_t'][t][0] += int(seen.sum()); a['by_t'][t][1] += float(hit.sum()); a['by_t'][t][2] += float(p.sum())
                ev = live & (tau <= EDGES[-1])
                a['ev'] += int(ev.sum()); a['up'] += float(first_up[ev].sum())
    rows = []
    for (lad, kind, lvl), a in sorted(acc.items()):
        r = dict(ladder=lad, kind=kind, level=lvl, n=a['n'], up_first=a['up'] / max(a['ev'], 1), events=a['ev'])
        for t, (n, h, p) in a['by_t'].items():
            r[f'hit_{t}'] = h / max(n, 1); r[f'null_{t}'] = p / max(n, 1)
        rows.append(r)
    return rows


def pct(x):
    return '—' if x is None else f'{100 * x:.1f}%'


def tstat(d):
    if not d or d.get('t') is None:
        return '—'
    return f"{d['mean']:+.4f} (t = {d['t']:+.1f}, {d['days']} дн.)"


def ci(d):
    return '—' if not d or d.get('lower') is None else f"{d['mean']:+.2f} [{d['lower']:+.2f}; {d['upper']:+.2f}]"


def verdict(d, need=2.):
    return d is not None and d.get('t') is not None and d['t'] >= need


def markdown(s, grid):
    kA, kB = s['knowledge_A2'], s['knowledge_B']
    L = ['# Интеллект скальпера, стадия 1: измерение пути, различение, риск, адаптация', '',
         f"Преррегистрация: `PREREGISTRATION_RU.md`. Разбиение: обучение A1 ({len(s['split']['A1'])} дн.) → разработка A2 "
         f"({len(s['split']['A2'])} дн.); переобучение на A → проверка B ({len(s['split']['B'])} дн.). Validation и evaluation не читались.", '']
    # verdict table
    rows = []
    def add(code, claim, a2, b, ok):
        rows.append(f'| {code} | {claim} | {a2} | {b} | {ok} |')
    add('M1', 'измеритель = scalper_v1', '108/108 дней точно', '', '✅')
    add('M3', 'лестница = прямой поиск на реальных котировках', '0 расхождений / 2 275 лестниц', '', '✅')
    for mode in ('T', 'P'):
        a, b = kA[f'{mode}_all'], kB[f'{mode}_all']
        add(f'K0 {mode}', 'эмпирический ноль сильнее броуновского («когда»)', tstat(a['when_null_vs_brownian']), tstat(b['when_null_vs_brownian']),
            '✅' if verdict(a['when_null_vs_brownian']) else '❌')
        for code, comp, claim in (('K1', 'when', 'full предсказывает **когда** сверх нуля'), ('K2', 'what', 'full предсказывает **что** сверх нуля')):
            da, db = a[f'{comp}_full_vs_null'], b[f'{comp}_full_vs_null']
            ok = '✅ подтверждено на B' if verdict(da) and verdict(db) else ('⚠️ только A2' if verdict(da) else '❌')
            add(f'{code} {mode}', claim, tstat(da), tstat(db), ok)
    da, db = kA['P_all']['fill_full_vs_null'], kB['P_all']['fill_full_vs_null']
    add('K3 P', 'full предсказывает **выбор контрагента** (исполнение)', tstat(da), tstat(db),
        '✅ подтверждено на B' if verdict(da) and verdict(db) else ('⚠️ только A2' if verdict(da) else '❌'))
    sa, sb = s['selection_A2'], s['selection_B']
    add('D1', 'разложение отбирает лучше скаляра (лучший 1% дня)', tstat(sa['top_full_vs_scalar']), tstat(sb['top_full_vs_scalar']),
        '✅' if verdict(sb['top_full_vs_scalar']) else '❌')
    ag = s['agent']; b = ag['b']
    d2 = b.get('daily') if isinstance(b, dict) else None
    d2ok = d2 is not None and d2.get('lower') is not None and d2['lower'] > 0
    add('D2', f"экономическое преимущество агента (выбран на A2: `{ag['choice']}`)", f"{ag['a2_net']:+.1f} $",
        ('0 сделок' if not b.get('trades') else f"{b['net']:+.1f} $, {ci(d2)}"), '✅' if d2ok else ('воздержался' if ag['choice'] == 'WAIT' else '❌'))
    h = s['hedge']
    add('G1', 'Hedge в пределах своей гарантии', f"сожаление {h['regret']:+.1f} $ ≤ {h['bound']:.1f} $", f"B: {h['B_net']:+.1f} $",
        '✅' if h['regret'] <= h['bound'] else '❌')
    lg = s['ledger_B']
    add('L1', 'денежная ошибка раскладывается точно', '', f"макс. расхождение {lg['identity_max_gap']:.1e}", '✅' if lg['identity_max_gap'] < 1e-9 else '❌')
    L += ['## Вердикты', '', '| Код | Утверждение | A2 (модели A1) | B (модели A) | Итог |', '|---|---|---|---|---|'] + rows
    # knowledge table
    L += ['', '## 1. Различение: что предсказуемо сверх нуля', '',
          'Средняя потеря на попытку (log-loss «когда» — по корзинам времени, «что» — цель/стоп при событии; Brier — три исхода к горизонту). '
          'Меньше — лучше.', '',
          '| Блок | Режим | Компонент | броуновский | эмп. ноль | full |', '|---|---|---|---:|---:|---:|']
    for blk, k in (('A2', kA), ('B', kB)):
        for mode in ('T', 'P'):
            r = k[f'{mode}_all']
            for comp, nm in (('when', 'когда'), ('what', 'что'), ('brier', 'Brier')):
                L.append(f"| {blk} | {mode} | {nm} | {r[comp]['brownian']:.4f} | {r[comp]['null']:.4f} | {r[comp]['full']:.4f} |")
            if mode == 'P':
                L.append(f"| {blk} | P | исполнение (доля {pct(r['fill_rate'])}) | — | {r['fill']['null']:.4f} | {r['fill']['full']:.4f} |")
    L += ['', 'Те же сравнения только на допустимых попытках (фильтры спреда и издержек):', '',
          '| Блок | Режим | «когда»: full − ноль | «что»: full − ноль | исполнение: full − ноль |', '|---|---|---|---|---|']
    for blk, k in (('A2', kA), ('B', kB)):
        for mode in ('T', 'P'):
            r = k[f'{mode}_legal']
            L.append(f"| {blk} | {mode} | {tstat(r['when_full_vs_null'])} | {tstat(r['what_full_vs_null'])} | "
                     f"{tstat(r.get('fill_full_vs_null')) if mode == 'P' else '—'} |")
    # calibration
    L += ['', '## 2. Калибровка исходов по планам (B)', '', '| Режим | План | n | цель факт / бр. / ноль / full | стоп факт / бр. / ноль / full | время факт / full |',
          '|---|---|---:|---|---|---|']
    for r in s['calibration_B']:
        L.append(f"| {r['mode']} | {r['plan']} | {r['n']} | {pct(r['take'])} / {pct(r['take_brownian'])} / {pct(r['take_null'])} / {pct(r['take_full'])} | "
                 f"{pct(r['stop'])} / {pct(r['stop_brownian'])} / {pct(r['stop_null'])} / {pct(r['stop_full'])} | {pct(r['time'])} / {pct(r['time_full'])} |")
    # null grid
    if grid:
        g = pd.DataFrame(grid)
        L += ['', '## 3. Броуновский ноль на сетке симметричных барьеров (A и B, без обучения)', '',
              'Доля входов, у которых первый из барьеров ±d достигнут к моменту t, против конечногоризонтного броуновского нуля с σ600 на момент решения.', '',
              '| Вход | d | 5 с факт / ноль | 30 с | 120 с | 600 с | первым вверх* |', '|---|---|---|---|---|---|---:|']
        for blk in ('A', 'B'):
            for _, r in g[(g.block == blk) & (g.kind == 'vol') & (g.ladder == 'T') & g.level.isin([1, 3, 8, 22])].iterrows():
                L.append(f"| {blk}: тейкер | {r.level}σ | " + ' | '.join(f"{pct(r[f'hit_{t}'])} / {pct(r[f'null_{t}'])}" for t in (5, 30, 120, 600))
                         + f" | {pct(r.up_first)} |")
        L += ['', '\\* Для тейкера — доля «вверх первым» (ноль: 50%).', '',
              '**След контрагента после пассивного исполнения.** Доля входов, у которых первым достигнут барьер **против** исполненной стороны (ноль: 50%):', '',
              '| Блок | d | после покупки в очереди | после продажи в очереди | после тейкера (вниз первым) |', '|---|---|---:|---:|---:|']
        for blk in ('A', 'B'):
            for lvl in (1, 2, 3, 4, 6):
                q = g[(g.block == blk) & (g.kind == 'abs') & (g.level == lvl)].set_index('ladder')
                if len(q) == 3:
                    L.append(f"| {blk} | {lvl} тик. | {pct(1 - q.loc['P_long', 'up_first'])} | {pct(q.loc['P_short', 'up_first'])} | "
                             f"{pct(1 - q.loc['T', 'up_first'])} |")
    # decisions
    L += ['', '## 4. Риск с результативностью: решения на одном счёте', '',
          f"Качество отбора (допустимые попытки с известным результатом). Лучший {TOPPCT}% дня по E[G] и ранговая корреляция E[G] с фактическим net:", '',
          '| Блок | лучший 1%: scalar | null | full | Спирмен: scalar | null | full |', '|---|---:|---:|---:|---:|---:|---:|']
    for blk, x in (('A2', sa), ('B', sb)):
        L.append(f"| {blk} | {x['top_scalar']:+.3f} | {x['top_null']:+.3f} | {x['top_full']:+.3f} | {x['rho_scalar']:+.4f} | {x['rho_null']:+.4f} | {x['rho_full']:+.4f} |")
    L += ['', 'Эксперты на счёте 1 oz (сетка решений 5 с):', '', '| Эксперт | A2: сделок | A2 net $ | B: сделок | B net $ | B $ на сделку | B дневной [95%] | B просадка $ |',
          '|---|---:|---:|---:|---:|---:|---|---:|']
    for k in s['experts_B']:
        a, bb = s['experts_A2'][k], s['experts_B'][k]
        pt = '—' if bb.get('per_trade') is None else f"{bb['per_trade']:+.3f}"
        L.append(f"| {k}{' ← агент' if k == ag['choice'] else ''} | {a['trades']} | {a['net']:+.1f} | {bb['trades']} | {bb['net']:+.1f} | {pt} | "
                 f"{ci(bb.get('daily'))} | {bb.get('max_drawdown', 0):.1f} |")
    rb = s['random_B']
    if rb.get('trades'):
        L.append(f"| случайные входы с частотой full@0 | — | — | {rb['trades']} | {rb['net']:+.1f} | {rb['per_trade']:+.3f} | {ci(rb.get('daily'))} | {rb['max_drawdown']:.1f} |")
    L += ['', f"full@0 минус scalar@0 на B по дням: {tstat(ag['full_minus_scalar_theta0_B'])}.", '',
          '**Сожаление о несмелости** (net при каждом θ; учиться можно только на A2):', '',
          '| θ | ' + ' | '.join(k.split('@')[1] for k in s['daring_A2']) + ' |', '|---|' + '---:|' * len(s['daring_A2']),
          '| A2 | ' + ' | '.join(f'{v:+.1f}' for v in s['daring_A2'].values()) + ' |',
          '| B | ' + ' | '.join(f'{v:+.1f}' for v in s['daring_B'].values()) + ' |']
    # adversarial
    L += ['', '## 5. Теория игр', '', '### 5.1 Контрагент очереди: проклятие победителя', '']
    for blk in ('A2', 'B'):
        a = s[f'adverse_{blk}']
        L.append(f"- {blk}: тейкер в тех же точках и планах — {a['taker_where_queue_filled']:+.3f} $ там, где очередь исполнилась, и "
                 f"{a['taker_where_queue_empty']:+.3f} $ там, где нет ({a['pairs']:,} пар).")
    L += ['', '| Квинтиль P(fill), B | P(fill) | факт исполнения | прогноз E[G \\| fill] | факт net после fill | доля стопов факт / прогноз |', '|---|---:|---:|---:|---:|---|']
    for r in s['adverse_B']['quintiles']:
        L.append(f"| {r['quintile'] + 1} | {pct(r['p_fill'])} | {pct(r['fill_rate'])} | {r['ef_pred']:+.3f} | {r['net_after_fill']:+.3f} | "
                 f"{pct(r['stop_share'])} / {pct(r['p_stop_pred'])} |")
    L += ['', '### 5.2 Рынок как противник: Hedge по экспертам', '',
          f"{len(h['experts'])} экспертов, {h['days']} дней (A2 + B), η = {h['eta']:.3f}. Hedge: A2 {h['A2_net']:+.1f} $, B {h['B_net']:+.1f} $. "
          f"Лучший эксперт задним числом на всём прогоне — `{h['best_expert']}` ({h['best_net']:+.1f} $), на B — `{h['B_best_expert']}` ({h['B_best_net']:+.1f} $). "
          f"Сожаление {h['regret']:+.1f} $ при гарантии ≤ {h['bound']:.1f} $.", '',
          'Крупные финальные веса: ' + ', '.join(f'`{k}` {v:.2f}' for k, v in sorted(h['final_weights'].items(), key=lambda x: -x[1])[:4]) + '.']
    # ledger
    L += ['', '## 6. Журнал сожаления: какая составляющая ошиблась (B)', '',
          f"Политика журнала: `{lg['policy']}`, {lg['trades']} сделок. Прогноз {lg['forecast']:+.3f} $ на сделку, факт {lg['realised']:+.3f} $.", '',
          'net − E[G] = (исполнение − P(fill))·E[G|fill] + [исход − P(исход)]·V + (net − V(факт. исход)):', '',
          '| Компонент | средняя ошибка $ | доля квадрата ошибки | доля сделок, где компонент главный |', '|---|---:|---:|---:|']
    for c, nm in (('err_fill', 'исполнение (выбор контрагента)'), ('err_outcome', 'исход: когда и что'),
                  ('err_value', 'цена исхода: у цели/стопа — спред и проскок, у выхода по времени — прогноз дрейфа')):
        L.append(f"| {nm} | {lg['bias'][c]:+.3f} | {pct(lg['square_share'][c])} | {pct(lg['dominant'].get(c, 0))} |")
    L += ['', 'Сделки по режимам: ' + '; '.join(f"{m}: {v['trades']} сделок, прогноз {v['forecast']:+.3f}, факт {v['realised']:+.3f}" for m, v in lg['by_mode'].items()) + '.',
          'Планы выбранных сделок: ' + ', '.join(f'{k} — {v}' for k, v in lg.get('plans', {}).items()) + '.']
    if lg.get('by_outcome'):
        L += ['', 'Цена исхода по фактическому исходу (добавлено после результата, описательно):', '',
              '| Режим | Исход | сделок | факт net $ | прогноз V $ | ошибка цены $ |', '|---|---|---:|---:|---:|---:|']
        names = {0: 'не исполнено', 1: 'цель', 2: 'стоп', 3: 'время'}
        for r in lg['by_outcome']:
            L.append(f"| {r['mode']} | {names[r['cat']]} | {r['trades']} | {r['net']:+.2f} | "
                     + ('—' if r['v_pred'] is None else f"{r['v_pred']:+.2f}") + f" | {r['err_value']:+.2f} |")
    L += conclusion(s, grid)
    return '\n'.join(L) + '\n'


def conclusion(s, grid):
    kA, kB = s['knowledge_A2'], s['knowledge_B']; ag = s['agent']; h = s['hedge']; lg = s['ledger_B']
    ex = s['experts_B']; exa = s['experts_A2']
    full_b = [ex[k]['net'] for k in ex if k.startswith('full@')]; full_a = [exa[k]['net'] for k in exa if k.startswith('full@')]
    adv = s['adverse_B']['quintiles']; sel_b, sel_a = s['selection_B'], s['selection_A2']
    t_out = {r['cat']: r for r in lg.get('by_outcome', []) if r['mode'] == 'P'}
    g = pd.DataFrame(grid) if grid else None

    def adverse(blk, lvl):
        q = g[(g.block == blk) & (g.kind == 'abs') & (g.level == lvl)].set_index('ladder')
        return 1 - q.loc['P_long', 'up_first']
    L = ['', '## Вывод', '',
         '**Измерение и нули готовы.** Сканер пути воспроизводит набор скальпера без единого расхождения (108 дней, 25 млн попыток) '
         'и совпадает с прямым поиском на реальных котировках. Эмпирический ноль сильнее броуновского в 15–25 стандартных ошибок: '
         'на малых масштабах рынок медленнее броуновского (±1σ за 5 с — 81–86% против 98–99%), на больших у него тяжёлые хвосты.', '',
         '**Различать.**',
         f"- «Когда» — знание сверх сильного нуля есть и подтверждено на B (T: t = {kB['T_all']['when_full_vs_null']['t']:+.1f}, "
         f"P: t = {kB['P_all']['when_full_vs_null']['t']:+.1f}). Эффект мал: около {1000 * kB['T_all']['when_full_vs_null']['mean']:.1f}·10⁻³ нат на попытку.",
         f"- «Выбор контрагента» (исполнится ли заявка в очереди) — подтверждено (B: t = {kB['P_all']['fill_full_vs_null']['t']:+.1f}).",
         f"- «Что» (какой барьер первым — сторона) — на A2 порог развития не пройден (t = {kA['T_all']['what_full_vs_null']['t']:+.1f} и "
         f"{kA['P_all']['what_full_vs_null']['t']:+.1f}); на B t = {kB['T_all']['what_full_vs_null']['t']:+.1f} и {kB['P_all']['what_full_vs_null']['t']:+.1f}. "
         'По протоколу не засчитано: это кандидат на проверку на новых данных, а не установленный факт.', '',
         '**Теория игр.**']
    if g is not None:
        L.append(f"- **След информированного контрагента виден в пути.** После исполнения в очереди первым приходит барьер против нас: "
                 f"±1 тик — {pct(adverse('A', 1))} (A) и {pct(adverse('B', 1))} (B), ±4 тика — {pct(adverse('A', 4))} и {pct(adverse('B', 4))}; "
                 'после тейкерского входа — 50%. Часть эффекта механическая (исполнение = снятие уровня), но он держится до 6 тиков.')
    L += [f"- Модель предсказывает, **исполнят ли** заявку, но не **насколько вредно** исполнение: прогноз E[G | fill] "
          f"{min(r['ef_pred'] for r in adv):+.2f}…{max(r['ef_pred'] for r in adv):+.2f} $ во всех квинтилях P(fill), факт "
          f"{min(r['net_after_fill'] for r in adv):+.2f}…{max(r['net_after_fill'] for r in adv):+.2f} $. "
          'Отличать токсичные исполнения от безвредных агент пока не умеет — это главная задача для пассива.',
          f"- **Hedge** по 19 экспертам на B: {h['B_net']:+.1f} $ против {ex[ag['choice']]['net']:+.1f} $ у точечного выбора на A2 и 0 у WAIT. "
          f"Гарантия выполнена ({h['regret']:+.1f} ≤ {h['bound']:.0f} $), но на 63 днях граница пустая: это защита от выбора по шуму, а не преимущество.", '',
          '**Рисковать с результативностью.**',
          f"- Агент по правилу «лучший net на A2» выбрал `{ag['choice']}` — {exa[ag['choice']]['trades']} сделок, {ag['a2_net']:+.1f} $ — и на B проиграл "
          f"{ex[ag['choice']]['net']:+.1f} $, дневной интервал ниже нуля. Слабое звено — правило выбора по малой выборке.",
          f"- Разложенная модель на B положительна при всех θ ({min(full_b):+.1f}…{max(full_b):+.1f} $), но на A2 отрицательна при всех θ "
          f"({min(full_a):+.1f}…{max(full_a):+.1f} $). Дневные интервалы на B включают ноль. **Экономического преимущества нет.**",
          f"- Качество отбора (D1) не доказано. Ранговая корреляция E[G] с net на B: full {sel_b['rho_full']:+.3f} против scalar {sel_b['rho_scalar']:+.3f} "
          f"(t = {sel_b['rho_full_vs_scalar']['t']:+.1f}); на A2 наоборот (t = {sel_a['rho_full_vs_scalar']['t']:+.1f}).", '',
          '**Журнал показывает, какая составляющая ошиблась.**',
          f"- У выбранных сделок {pct(lg['square_share']['err_outcome'])} квадрата денежной ошибки дал исход и {pct(lg['square_share']['err_value'])} — цена исхода.",
          (f"- Почти вся ошибка цены приходится на выходы по времени, где V — прогноз дрейфа. Пример P: прогноз {t_out[3]['v_pred']:+.2f} $, факт {t_out[3]['net']:+.2f} $."
           if 3 in t_out else '- Разбивки по исходам нет.'),
          '- Это проклятие оптимизатора: выбирая максимум E[G], агент выбирает и максимум ошибки головы дрейфа.',
          '- У цели и стопа в очереди ошибка цены — 4–10% их величины; у тейкера стопов всего два, выводов по ним нет.',
          f"- Все выбранные сделки — 10-минутные планы ({', '.join(lg.get('plans', {}))}): при постоянных издержках круга требуемое отношение сигнал/шум c/(σ√T) там наименьшее.", '',
          '**Что следует для стадии 2** (новая преррегистрация, без пересмотра этой):',
          '1. **Рисковать только на подтверждённом знании.** Брать α и fill из модели full, а β и дрейф выхода по времени — из нуля, пока «что» не подтверждено. Добавить нижнюю доверительную границу E[G] против проклятия оптимизатора.',
          '2. **Выбор эксперта** через SPA/StepM со штрафом за число сделок или через Hedge — вместо максимума net на A2.',
          '3. **Модель токсичности исполнения:** состояние очереди и поток в момент постановки → β после исполнения.',
          '4. **Проверка «что»** на validation-месяцах (уже видены) и окончательно на C.', '',
          'Development evidence на train-месяцах; экономического допуска нет.']
    return L


TOPPCT = 1


def report(output=REPORT, grid=True):
    out = Path(output); s = json.loads((out / 'stage1_summary.json').read_text(encoding='utf8'))
    g = []
    if grid:
        if not (out / 'null_grid.json').exists():
            for blk, months in (('A', SPLIT['A1'] + SPLIT['A2']), ('B', SPLIT['B'])):
                g += [dict(block=blk, **r) for r in null_grid(dates_of(months))]
            save_json(out / 'null_grid.json', g)
        g = json.loads((out / 'null_grid.json').read_text(encoding='utf8'))
    (out / 'STAGE1_RU.md').write_text(markdown(s, g), encoding='utf8')
    return out / 'STAGE1_RU.md'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=REPORT); p.add_argument('--grid-only', action='store_true')
    a = p.parse_args()
    if a.grid_only:
        g = []
        for blk, months in (('A', SPLIT['A1'] + SPLIT['A2']), ('B', SPLIT['B'])):
            g += [dict(block=blk, **r) for r in null_grid(dates_of(months))]
        save_json(Path(a.output) / 'null_grid.json', g)
    else:
        print(report(a.output))
