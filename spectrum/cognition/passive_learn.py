"""When to rest a passive order: learn the value of a PQ attempt from queue state.

Each (origin, side, plan) attempt of a passive variant has value = net if it
filled, $0 if it did not. Features are side-relative: 'own' is the queue the
order joins (bid for a long), 'opp' the opposite best; tape and representation
signs are multiplied by the side. Everything is causal at the decision time.
Fit on block A, read block B; hyperparameters fixed before any B readout.

Policy: rest the order iff the predicted attempt value is > 0; per origin and
plan at most one side (the higher prediction). Controls: always rest, random
resting at the same rate, the same model fitted on permuted targets.
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from spectrum.ingest import save_json
from .passive_report import load as load_passive
from .signature_report import day_boot
from .signatures import MENU_PLANS
from .representation import DEFAULT_OUTPUT as REPRESENTATION
from .queue_state import DEFAULT_OUTPUT as QUEUE_STATE

SEED = 61061
REPORT = 'reports/passive_learn_v1'


def side_features(x):
    """Queue and representation features in the order's own frame."""
    s = x.side.to_numpy(float); long = s > 0
    pick = lambda a, b: np.where(long, x[a].to_numpy(float), x[b].to_numpy(float))
    own1, opp1, own2, opp2 = pick('bid1', 'ask1'), pick('ask1', 'bid1'), pick('bid2', 'ask2'), pick('ask2', 'bid2')
    q = dict(log_own1=np.log1p(own1), log_opp1=np.log1p(opp1), log_own2=np.log1p(own2), log_opp2=np.log1p(opp2),
             imb1=(own1 - opp1) / np.maximum(own1 + opp1, 1),
             imb12=(own1 + own2 - opp1 - opp2) / np.maximum(own1 + own2 + opp1 + opp2, 1),
             spread=x.spread_ticks.to_numpy(float))
    for w in (5, 30):
        for kind in ('add', 'cancel', 'fill'):
            q[f'log_own_{kind}_{w}'] = np.log1p(pick(f'bid_{kind}_{w}', f'ask_{kind}_{w}'))
            q[f'log_opp_{kind}_{w}'] = np.log1p(pick(f'ask_{kind}_{w}', f'bid_{kind}_{w}'))
        tb, ts = x[f'tbuy_{w}'].to_numpy(float), x[f'tsell_{w}'].to_numpy(float)
        q[f'tape_{w}'] = s * (tb - ts) / np.maximum(tb + ts, 1); q[f'log_trades_{w}'] = np.log1p(tb + ts)
    q['own_depletion_30'] = (pick('bid_fill_30', 'ask_fill_30') + pick('bid_cancel_30', 'ask_cancel_30')) / np.maximum(own1, 1)
    r = dict(sigma600=x.sigma600_ticks_per_sqrt_s.to_numpy(float),
             sigma_ratio=x.sigma600_ticks_per_sqrt_s.to_numpy(float) / np.maximum(x.sigma3600_ticks_per_sqrt_s.to_numpy(float), 1e-6),
             minutes=x.minutes_since_london.to_numpy(float))
    for w in (300, 900, 1800, 3600):
        r[f'ret{w}'] = s * x[f'price_ret{w}_sigma'].to_numpy(float)
        pos = x[f'price_range{w}_position'].to_numpy(float)
        r[f'range{w}'] = np.where(long, pos, 1 - pos)
    for name in ('flow_pressure900_low', 'flow_pressure900_high', 'flow_book900_pressure', 'flow_depth1_imbalance',
                 'flow_depth12_imbalance', 'flow_pressure300_low', 'flow_pressure300_high'):
        r[name] = s * x[name].to_numpy(float)
    return pd.DataFrame(q, index=x.index), pd.DataFrame(r, index=x.index)


def dataset(variant='PQ_30'):
    f = load_passive()
    f = f[(f.variant == variant) & f.status.isin(['COMPLETE', 'NOT_FILLED'])]
    q = pd.concat([pd.read_parquet(p) for p in glob.glob(str(Path(QUEUE_STATE) / '2*.parquet'))])
    rep = pd.concat([pd.read_parquet(p) for p in glob.glob(str(Path(REPRESENTATION) / '*.parquet'))])
    sig = pd.concat([pd.read_parquet(p, columns=['origin_id', 'sigma600_ticks_per_sqrt_s', 'sigma3600_ticks_per_sqrt_s',
                                                 'minutes_since_london'])
                     for p in glob.glob('data/derived/opportunity_signatures_v1/*/x.parquet')])
    x = f.merge(q, on='origin_id').merge(rep, on='origin_id').merge(sig, on='origin_id')
    x = x[x.state_known == 1].reset_index(drop=True)
    qf, rf = side_features(x)
    return x, qf, rf


def model():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=200, min_samples_leaf=300,
                                         l2_regularization=1., random_state=SEED)


def weights(dates):
    s = pd.Series(dates); return (1 / s.map(s.value_counts())).to_numpy(float)


def policy(x, pred, block, rng=None, rate=None):
    """At most one side per origin and plan; rest iff its prediction > 0."""
    b = x[x.block == block].assign(pred=pred[(x.block == block).to_numpy()])
    if rate is not None:  # random resting at a given rate, one side chosen at random
        b = b.assign(pred=rng.random(len(b)))
        best = b.loc[b.groupby(['origin_id', 'plan_id']).pred.idxmax()]
        chosen = best[rng.random(len(best)) < rate * 2]
    else:
        best = b.loc[b.groupby(['origin_id', 'plan_id']).pred.idxmax()]
        chosen = best[best.pred > 0]
    origins = b.groupby('plan_id').origin_id.nunique()
    out = {}
    for plan, g in b.groupby('plan_id'):
        c = chosen[chosen.plan_id == plan]
        daily = pd.DataFrame(dict(date=sorted(b[b.plan_id == plan].date.unique()))).merge(
            c.groupby('date', as_index=False).attempt.sum(), on='date', how='left').fillna({'attempt': 0.})
        out[str(MENU_PLANS[plan])] = dict(origins=int(origins[plan]), rested=len(c),
                                          rest_rate=len(c) / max(int(origins[plan]), 1),
                                          per_attempt=float(c.attempt.mean()) if len(c) else None,
                                          fill_rate=float((c.status == 'COMPLETE').mean()) if len(c) else None,
                                          daily=day_boot(daily, 'attempt'))
    return out


def experiment(variant='PQ_30', output=REPORT):
    x, qf, rf = dataset(variant)
    a = (x.block == 'A').to_numpy(); y = x.attempt.to_numpy(float); w = weights(x.date[a])
    sets = dict(queue=qf, representation=rf, both=pd.concat([qf, rf], axis=1))
    rng = np.random.default_rng(SEED); res = {}
    base_A = {}
    for plan in sorted(x.plan_id.unique()):
        m = a & (x.plan_id == plan).to_numpy(); base_A[int(plan)] = float(np.average(y[m], weights=weights(x.date[m])))
    for name, X in sets.items():
        pred = np.zeros(len(x)); pred_perm = np.zeros(len(x)); skill = {}
        for plan in sorted(x.plan_id.unique()):
            p = (x.plan_id == plan).to_numpy(); tr = a & p
            mdl = model().fit(X[tr], y[tr], sample_weight=weights(x.date[tr]))
            perm = model().fit(X[tr], rng.permutation(y[tr]), sample_weight=weights(x.date[tr]))
            pred[p] = mdl.predict(X[p]); pred_perm[p] = perm.predict(X[p])
            te = (~a) & p; wt = weights(x.date[te]); c = base_A[int(plan)]
            mse = lambda h: float(np.average((y[te] - h) ** 2, weights=wt))
            skill[str(MENU_PLANS[plan])] = dict(r2_vs_A_mean=1 - mse(pred[te]) / mse(np.full(te.sum(), c)),
                                                corr=float(np.corrcoef(pred[te], y[te])[0, 1]))
        always = policy(x, np.ones(len(x)), 'B')
        learned = policy(x, pred, 'B')
        rate = np.mean([v['rest_rate'] for v in learned.values()])
        res[name] = dict(features=X.shape[1], skill_B=skill, policy_A=policy(x, pred, 'A'), policy_B=learned,
                         permuted_B=policy(x, pred_perm, 'B'),
                         random_same_rate_B=policy(x, pred, 'B', rng=np.random.default_rng(SEED + 1), rate=rate),
                         always_B=always)
        print(name, {k: (round(v['r2_vs_A_mean'], 4)) for k, v in skill.items()},
              {k: (v['rested'], v['per_attempt'] and round(v['per_attempt'], 3)) for k, v in learned.items()}, flush=True)
    Path(output).mkdir(parents=True, exist_ok=True)
    save_json(Path(output) / f'passive_learn_{variant}.json', dict(variant=variant, results=res, base_A=base_A,
                                                                  semantics=__doc__.strip()))
    return res


def diagnostics(x):
    """Attempt value, fill rate and adverse selection by decision-time queue state."""
    s = x.side.to_numpy(); long = s > 0
    own = np.where(long, x.bid1, x.ask1); opp = np.where(long, x.ask1, x.bid1)
    imb = (own - opp) / np.maximum(own + opp, 1)
    tape = s * (x.tbuy_30 - x.tsell_30) / np.maximum(x.tbuy_30 + x.tsell_30, 1)
    buckets = dict(own_queue=pd.cut(own, [0, 1, 2, 4, 10**6], labels=['1', '2', '3-4', '5+']),
                   l1_imbalance=pd.cut(imb, [-1.01, -.5, -.01, .01, .5, 1.01], labels=['<=-.5', '(-.5,0)', '0', '(0,.5)', '>=.5']),
                   tape_30s=pd.cut(tape, [-1.01, -.3, -.1, .1, .3, 1.01], labels=['<=-.3', '(-.3,-.1]', '(-.1,.1)', '[.1,.3)', '>=.3']))
    out = {}
    for name, b in buckets.items():
        g = x.assign(bucket=b, filled=x.status == 'COMPLETE').groupby(['block', 'bucket'], observed=True)
        out[name] = [dict(block=k[0], bucket=str(k[1]), n=len(v), fill=float(v.filled.mean()),
                          attempt=float(v.attempt.mean()), adverse60=float(v.adverse60_ticks.mean()))
                     for k, v in g]
    return out


def ranking(x, F):
    """Prediction buckets on B with edges from A predictions (no B threshold fitting)."""
    a = (x.block == 'A').to_numpy(); y = x.attempt.to_numpy(float); rows = []
    for plan in sorted(x.plan_id.unique()):
        p = (x.plan_id == plan).to_numpy(); tr = a & p
        m = model().fit(F[tr], y[tr], sample_weight=weights(x.date[tr]))
        edges = np.quantile(m.predict(F[tr]), [.5, .8, .9])
        b = x[(~a) & p].assign(bucket=np.searchsorted(edges, m.predict(F[(~a) & p])))
        for k, g in b.groupby('bucket'):
            fill = float((g.status == 'COMPLETE').mean())
            rows.append(dict(plan=list(MENU_PLANS[plan]), bucket=['<p50', 'p50-80', 'p80-90', 'top10'][k], n=len(g),
                             attempt=day_boot(g, 'attempt'), fill=fill,
                             per_fill=float(g.attempt.mean() / fill) if fill else None))
    return rows


def report(variant='PQ_30', output=REPORT, schemas='reports/schemas_queue_v1/PQ_30'):
    import json
    x, qf, rf = dataset(variant)
    learned = json.loads((Path(output) / f'passive_learn_{variant}.json').read_text(encoding='utf8'))['results']
    s = dict(variant=variant, diagnostics=diagnostics(x), ranking_both=ranking(x, pd.concat([qf, rf], axis=1)),
             ranking_queue=ranking(x, qf),
             skill={k: v['skill_B'] for k, v in learned.items()},
             rested_B={k: {p: r['rested'] for p, r in v['policy_B'].items()} for k, v in learned.items()},
             always_B={p: r['per_attempt'] for p, r in learned['both']['always_B'].items()})
    sm_path = Path(schemas) / 'schema_summary.json'
    if sm_path.exists():
        ss = json.loads(sm_path.read_text(encoding='utf8'))
        lib = json.loads((Path(schemas) / 'library_full.json').read_text(encoding='utf8'))
        s['schemas'] = dict(variants={k: dict(spawned=v['spawned'], promoted=v['promoted'], refuted=v['refuted'],
                                              trades_B=v['B']['trades'], net_B=v['B']['net']) for k, v in ss['variants'].items()},
                            queue_roots=[dict(describe=k['describe'], mean=k['mean_all'], n=k['n']) for k in lib['schemas']
                                         if k['parent'] < 0 and any(q in k['program'] for q in ('l1', 'tape'))],
                            events=[e['describe'] + f" ({e['kind']})" for e in lib['events']])
    save_json(Path(output) / 'queue_summary.json', s)
    (Path(output) / 'QUEUE_RU.md').write_text(markdown(s), encoding='utf8')
    return s


def markdown(s):
    L = ['# Состояние очереди и решение «выставляться ли» — train, A→B', '',
         f"Исполнение {s['variant']}: вход в очередь на 30 с, take-profit лимиткой. Попытка без исполнения = $0. "
         'A = окт–дек 2025, B = янв–фев 2026, обе части внутри train; validation/evaluation не читались.', '',
         'Признаки берутся строго из записей до момента решения: размеры L1/L2 считаются тем же ядром очереди; '
         'за 5/30 с — добавления, снятия и исполнения на адресах лучших цен и объём агрессоров. Всё переведено в '
         'систему стороны заявки: «своя» очередь — та, в которую встаёт заявка.', '',
         '## 1. Связь с исходом попытки', '']
    names = dict(own_queue='Своя очередь L1, лоты', l1_imbalance='Дисбаланс L1 в свою сторону',
                 tape_30s='Лента агрессоров 30 с в свою сторону')
    for key, rows in s['diagnostics'].items():
        L += [f'**{names[key]}**', '', '| Блок | Корзина | N | Исполнено | $ на попытку | mid через 60 с − вход, тики |',
              '|---|---|---:|---:|---:|---:|']
        for r in rows:
            L.append(f"| {r['block']} | {r['bucket']} | {r['n']} | {r['fill']:.3f} | {r['attempt']:+.3f} | {r['adverse60']:+.2f} |")
        L.append('')
    L += ['## 2. Модель стоимости попытки (HistGradientBoosting, A → B)', '',
          '| Признаки | 12/8/120 | 24/12/300 | 40/20/600 | 80/40/1800 |', '|---|---:|---:|---:|---:|']
    for k, v in s['skill'].items():
        L.append(f'| R² на B: {k} | ' + ' | '.join(f"{v[p]['r2_vs_A_mean']:+.4f}" for p in v) + ' |')
    L += ['', 'Правило «выставляться при прогнозе > 0» на B: ' +
          '; '.join(f"{k}: {sum(v.values())} заявок" for k, v in s['rested_B'].items()) +
          '. Всегда выставляться: ' + ', '.join(f"{p} {v:+.3f}$" for p, v in s['always_B'].items()) + ' на попытку.', '',
          '## 3. Ранжирование на B (границы децилей — по прогнозам на A)', '',
          '| Признаки | План | Корзина | N | $ на попытку [95% дни] | Исполнено | $ на исполненную |',
          '|---|---|---|---:|---:|---:|---:|']
    for label, rows in (('очередь+представление', s['ranking_both']), ('только очередь', s['ranking_queue'])):
        for r in rows:
            a = r['attempt']
            L.append(f"| {label} | {'/'.join(map(str, r['plan']))} | {r['bucket']} | {r['n']} | {a['mean']:+.3f} "
                     f"[{a['lower']:+.3f}; {a['upper']:+.3f}] | {r['fill']:.3f} | {r['per_fill']:+.3f} |")
    if 'schemas' in s:
        x = s['schemas']
        L += ['', '## 4. Схемы с языком очереди (координаты l1, thickness, tape; программы с/против L1 и ленты)', '',
              '| Вариант | рождено | подтверждено | опровергнуто | сделок B | net B $ |', '|---|---:|---:|---:|---:|---:|']
        for k, v in x['variants'].items():
            L.append(f"| {k} | {v['spawned']} | {v['promoted']} | {v['refuted']} | {v['trades_B']} | {v['net_B']:+.1f} |")
        L += ['', 'Корни очереди: ' + '; '.join(f"{r['describe']} {r['mean']:+.3f}$" for r in x['queue_roots']),
              'События full: ' + ('; '.join(x['events']) or 'нет')]
    L += ['', '## Вывод', '',
          '- Состояние очереди несёт переносимую A→B информацию о стоимости попытки только на коротком горизонте; к 30 минутам она исчезает.',
          '- Ранжирование снижает неблагоприятный отбор: верхние корзины прогноза теряют меньше и на попытку, и на исполненную сделку. '
          'Часть выигрыша — более низкая доля исполнений: модель учится выставляться там, где исполняют реже.',
          '- Ни одна корзина и ни одна схема не выходит в плюс после комиссии $0.30: пассивный скальпинг 1 oz на этих данных остаётся убыточным.', '',
          'Development evidence на ранее изученной истории; экономического допуска нет.']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--variant', default='PQ_30'); p.add_argument('--output', default=REPORT)
    a = p.parse_args(); experiment(a.variant, a.output); report(a.variant, a.output)
