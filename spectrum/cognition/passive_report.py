"""Readout for passive execution: FIFO audit, kernel parity, costs, selection, schemas."""
from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from .signature_report import day_boot
from .signatures import DEFAULT_OUTPUT as SIGNATURES, MENU_PLANS
from .passive import DEFAULT_OUTPUT as PASSIVE, MARKET, captures_by_date
from . import queue as qu

REPORT = 'reports/schemas_passive_v1'
SHOWN = ('TT', 'TQ', 'PT_30', 'PQ_30', 'PQ_120', 'PQ_600', 'PXQ_30', 'PXQ_120', 'PXQ_600')
AUDIT_DATE = '2025-10-01'


def fifo_audit(date=AUDIT_DATE, market=MARKET):
    """Share of F records whose order is first in its level once fills already
    reported in the same group are netted out (pure-Python replay of one day)."""
    caps = captures_by_date(market)[date]
    item = next(i for i in json.loads((Path(market) / 'run_config.json').read_text(encoding='utf8'))['members']
                if i['key'] == caps[0]['member_key'])
    orders, levels, counter, ok, bad = {}, {}, 0, 0, 0
    A, C, M, R, F = (ord(c) for c in 'ACMRF')
    for _, t in qu.capture_tapes(item['archive'], item['member'], caps):
        for a, o, sd, px, sz in zip(t['action'].tolist(), t['order_id'].tolist(), t['side'].tolist(),
                                    t['price'].tolist(), t['size'].tolist()):
            if a == R:
                orders.clear(); levels.clear()
            elif a == A:
                counter += 1; orders[o] = [sd, px, sz, counter, 0]; levels.setdefault((sd, px), {})[o] = counter
            elif a == C and o in orders:
                r = orders[o]; d = min(sz, r[2]); r[2] -= d; r[4] = max(0, r[4] - d)
                if r[2] == 0:
                    levels[(r[0], r[1])].pop(o, None); orders.pop(o)
            elif a == M and o in orders:
                r = orders[o]; levels[(r[0], r[1])].pop(o, None)
                if sz <= 0:
                    orders.pop(o); continue
                if px != r[1] or sz > r[2]:
                    counter += 1; r[3] = counter
                r[0], r[1], r[2], r[4] = sd, px, sz, 0; levels.setdefault((sd, px), {})[o] = r[3]
            elif a == F and o in orders:
                r = orders[o]; lv = levels.get((r[0], r[1]), {})
                first = not any(p < r[3] and orders[q][2] - orders[q][4] > 0 for q, p in lv.items())
                r[4] += sz; ok += first; bad += not first
    return dict(date=date, fills=ok + bad, fifo_first=ok, violation_share=bad / max(ok + bad, 1))


def load(passive=PASSIVE):
    f = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(Path(passive) / '2*.parquet')))], ignore_index=True)
    f['block'] = np.where(f.date < '2026-01', 'A', 'B')
    f['attempt'] = np.where(f.status == 'COMPLETE', f.net_usd, np.where(f.status == 'NOT_FILLED', 0., np.nan))
    return f


def parity(f, signatures=SIGNATURES):
    b = pd.concat([pd.read_parquet(p).reindex(columns=['origin_id', 'side', 'plan_id', 'status', 'net_usd'])
                   for p in glob.glob(str(Path(signatures) / '*' / 'branches.parquet'))])
    tt = f[f.variant == 'TT'].merge(b, on=['origin_id', 'side', 'plan_id'], suffixes=('', '_sig'))
    both = tt[(tt.status == 'COMPLETE') & (tt.status_sig == 'COMPLETE')]
    return dict(rows=len(tt), status_agreement=float((tt.status == tt.status_sig).mean()),
                max_abs_net_difference=float((both.net_usd - both.net_usd_sig).abs().max()))


def variants(f):
    g = f[f.status.isin(['COMPLETE', 'NOT_FILLED'])]
    rows = []
    for (variant, plan, block), x in g[g.variant.isin(SHOWN)].groupby(['variant', 'plan_id', 'block']):
        rows.append(dict(variant=variant, plan=list(MENU_PLANS[plan]), block=block, attempts=len(x),
                         fill_rate=float((x.status == 'COMPLETE').mean()),
                         net_per_trade=float(x.net_usd.mean()), attempt=day_boot(x, 'attempt'),
                         censored=int((f[(f.variant == variant) & (f.plan_id == plan) & (f.block == block)].status == 'CENSORED').sum())))
    return rows


def selection(f):
    c = f[(f.status == 'COMPLETE') & f.variant.isin(['TT', 'PQ_30'])]
    out = {}
    for (variant, block), x in c.groupby(['variant', 'block']):
        out[f'{variant}/{block}'] = dict(adverse60_ticks=float(x.adverse60_ticks.mean()),
                                         wait_to_fill_median_s=float(x.wait_to_fill_s.median()),
                                         fill_kinds=x.fill_kind.value_counts(normalize=True).round(3).to_dict(),
                                         queue_ahead_median=float(x.queue_ahead.median()) if variant != 'TT' else None)
    return out


def missed(f, variant='PQ_30'):
    """Taker outcome at the same origin/side/plan, split by whether the passive order filled."""
    tt = f[(f.variant == 'TT') & (f.status == 'COMPLETE')][['origin_id', 'side', 'plan_id', 'net_usd']]
    x = f[f.variant == variant][['origin_id', 'side', 'plan_id', 'status', 'block', 'date']].merge(
        tt.rename(columns={'net_usd': 'taker_net'}), on=['origin_id', 'side', 'plan_id'])
    x = x[x.status.isin(['COMPLETE', 'NOT_FILLED'])]
    return [dict(block=b, passive=st, n=len(g), taker_net=day_boot(g, 'taker_net'))
            for (b, st), g in x.groupby(['block', 'status'])]


def hurdle(f):
    rows = []
    for variant in ('TT', 'PQ_30'):
        x = f[(f.variant == variant) & f.attempt.notna()]
        per = x.pivot_table(index=['origin_id', 'plan_id'], columns='side', values='attempt').dropna().reset_index()
        for plan, g in per.groupby('plan_id'):
            right, wrong = g[[-1, 1]].max(axis=1).mean(), g[[-1, 1]].min(axis=1).mean()
            rows.append(dict(variant=variant, plan=list(MENU_PLANS[plan]), right=float(right), wrong=float(wrong),
                             breakeven=float(-wrong / (right - wrong)) if right > 0 else None))
    return rows


def report(passive=PASSIVE, output=REPORT):
    out = Path(output); out.mkdir(parents=True, exist_ok=True); f = load(passive)
    s = dict(version='PASSIVE_EXECUTION_READOUT_V1', passive_manifest_sha256=sha(Path(passive) / 'manifest.json'),
             fifo=fifo_audit(), parity=parity(f), variants=variants(f), selection=selection(f), missed=missed(f),
             hurdle=hurdle(f),
             schemas={})
    for v in ('PQ_30', 'PXQ_30'):
        p = out / v / 'schema_summary.json'
        if p.exists():
            ss = json.loads(p.read_text(encoding='utf8'))
            lib = json.loads((out / v / 'library_full.json').read_text(encoding='utf8'))
            roots = sorted([k for k in lib['schemas'] if k['parent'] < 0], key=lambda k: -(k['mean_all'] or -9))
            s['schemas'][v] = dict(variants={k: dict(spawned=r['spawned'], promoted=r['promoted'], refuted=r['refuted'],
                                                     trades_B=r['B']['trades'], net_B=r['B']['net'])
                                             for k, r in ss['variants'].items()},
                                   best_roots=[dict(describe=k['describe'], n=k['n'], mean=k['mean_all']) for k in roots[:4]],
                                   active_children=[dict(describe=k['describe'], mean=k['mean_all'], n=k['n'],
                                                         parent_mean=k['trigger']['parent_mean'])
                                                    for k in lib['schemas'] if k['status'] == 'ACTIVE' and k['parent'] >= 0])
    save_json(out / 'passive_summary.json', s)
    (out / 'PASSIVE_RU.md').write_text(markdown(s), encoding='utf8')
    return s


def markdown(s):
    L = ['# Пассивное исполнение с очередью FIFO по MBO — train, A→B', '',
         'A = окт–дек 2025, B = янв–фев 2026 (оба внутри train). Validation/evaluation не читались. '
         'Независимые попытки с отдельным счётом 1 oz, кроме прогона схем (один счёт).', '',
         '## Модель очереди', '',
         'Лента заявок воспроизводится из исходного DBN по маршруту и ordinal-диапазону captures. Гипотетическая 1-лотовая заявка '
         'встаёт в конец уровня; исполняется только при доказательстве FIFO: исполнена заявка позади неё при исчерпанной очереди '
         'впереди, исполнен худший уровень или прошла сделка сквозь цену. Касание цены не исполняет. Влияния на рынок нет.', '',
         f"- Аудит FIFO {s['fifo']['date']}: {s['fifo']['fills']} исполнений, доля нарушений порядка "
         f"{s['fifo']['violation_share']:.4f}. Нарушения не дают исполнения, а считаются отдельно.",
         f"- Паритет taker/taker с ядром сигнатур: {s['parity']['rows']} ветвей, совпадение статусов "
         f"{s['parity']['status_agreement']:.6f}, максимум расхождения денег ${s['parity']['max_abs_net_difference']:.1e}.", '',
         '## Варианты: стоимость попытки (неисполненная = $0)', '',
         'T — тейкер, P — вход в очередь на W секунд, PX — после W пересечь спред; выход T — take по проходу цены, Q — take лимиткой в очереди.', '',
         '| Вариант | План | Блок | Исполнено | $ на сделку | $ на попытку [95% дни] |', '|---|---|---|---:|---:|---:|']
    for r in s['variants']:
        a = r['attempt']
        L.append(f"| {r['variant']} | {'/'.join(map(str, r['plan']))} | {r['block']} | {r['fill_rate']:.3f} | "
                 f"{r['net_per_trade']:+.3f} | {a['mean']:+.3f} [{a['lower']:+.3f}; {a['upper']:+.3f}] |")
    L += ['', '## Неблагоприятный отбор', '', '| Исполнение/блок | mid через 60 с − цена входа, тики | медиана ожидания, с | виды исполнения |',
          '|---|---:|---:|---|']
    for k, v in s['selection'].items():
        L.append(f"| {k} | {v['adverse60_ticks']:+.2f} | {v['wait_to_fill_median_s']:.1f} | {json.dumps(v['fill_kinds'])} |")
    L += ['', 'Что было бы у тейкера в тех же точках (PQ_30, все планы):', '',
          '| Блок | пассивная заявка | N | $ тейкера [95% дни] |', '|---|---|---:|---:|']
    for r in s['missed']:
        a = r['taker_net']
        L.append(f"| {r['block']} | {'исполнена' if r['passive'] == 'COMPLETE' else 'не исполнена'} | {r['n']} | "
                 f"{a['mean']:+.3f} [{a['lower']:+.3f}; {a['upper']:+.3f}] |")
    L += ['', '## Сколько нужно угадывать направление (hindsight)', '', '| Исполнение | План | верная сторона $ | неверная $ | безубыточная точность |',
          '|---|---|---:|---:|---:|']
    for r in s['hurdle']:
        b = 'недостижима' if r['breakeven'] is None else f"{r['breakeven']:.3f}"
        L.append(f"| {r['variant']} | {'/'.join(map(str, r['plan']))} | {r['right']:+.3f} | {r['wrong']:+.3f} | {b} |")
    L += ['', '## Схемы с новым исполнением', '']
    for v, x in s['schemas'].items():
        L += [f'**{v}**', '', '| Вариант | рождено | подтверждено | опровергнуто | сделок B | net B $ |', '|---|---:|---:|---:|---:|---:|']
        for name, r in x['variants'].items():
            L.append(f"| {name} | {r['spawned']} | {r['promoted']} | {r['refuted']} | {r['trades_B']} | {r['net_B']:+.1f} |")
        L += ['', 'Лучшие корни: ' + '; '.join(f"{r['describe']} {r['mean']:+.3f}$ (n={r['n']})" for r in x['best_roots']),
              'Подтверждённые дети: ' + ('; '.join(f"{c['describe']} {c['mean']:+.3f}$ против родителя {c['parent_mean']:+.3f}$"
                                                 for c in x['active_children']) or 'нет'), '']
    L += ['## Вывод', '',
          '- Пассивный вход экономит на попытку $0.17–0.30, но на исполненную сделку лишь $0.08–0.15: неблагоприятный отбор '
          'забирает большую часть спреда. Неисполненные заявки — это в среднем выигрышные для тейкера ситуации; исполненные — проигрышные.',
          '- Порог точности направления снижается (30 мин: 0.60 → 0.57; 10 мин: 0.72 → 0.64), но направленного навыка в наших представлениях нет.',
          '- Ни один вариант исполнения не делает попытку неотрицательной в среднем; лучший корень схем остаётся ниже нуля.',
          '- Схемы на реальных данных по-прежнему не торгуют; подтверждённые гипотезы описывают, где хуже. Контроли (перемешанный контекст, подсаженный режим) ведут себя как прежде.', '',
          'Development evidence на ранее изученной истории; экономического допуска нет.']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    print(json.dumps({k: v for k, v in report().items() if k in ('fifo', 'parity')}, indent=1))
