"""Stage 0/1 readout: opportunity calibration and outcome signatures on train.

Block A (Oct-Dec 2025) fits; block B (Jan-Feb 2026) checks. Both lie inside the
train months: this is an internal chronological check, not the stage gate on
later validation blocks. The unit of uncertainty is a day, never a quote/trade.
Hindsight quantities (oracle direction) are labelled diagnostics, not targets.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from . import opportunity as op
from .signatures import DEFAULT_OUTPUT, MENU_PLANS

FEE_TICKS = 3.  # $0.30 per oz round trip
REPORT = 'reports/opportunity_signatures_v1'


def load(root):
    root = Path(root); manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    if manifest['status'] != 'COMPLETE':
        raise ValueError('Readout requires the complete declared dataset')
    xs, ys, bs = [], [], []
    for d in manifest['days']:
        for name, digest in d['files'].items():
            if sha(root / d['date'] / name) != digest:
                raise ValueError('Signature payload integrity failure ' + d['date'])
        xs.append(pd.read_parquet(root / d['date'] / 'x.parquet'))
        ys.append(pd.read_parquet(root / d['date'] / 'y.parquet'))
        bs.append(pd.read_parquet(root / d['date'] / 'branches.parquet'))
    x = pd.concat(xs, ignore_index=True); y = pd.concat(ys, ignore_index=True)
    if not x.origin_id.equals(y.origin_id):
        raise ValueError('X/Y origin alignment')
    b = pd.concat(bs, ignore_index=True).merge(x[['origin_id', 'date', 'block', 'kind']], on='origin_id', validate='m:1')
    return manifest, x, y, b


def day_boot(frame, column, seed=61031, draws=2000):
    """Equal-weight days; iid day resampling (no correction for project selection)."""
    daily = frame.groupby('date')[column].mean().dropna().to_numpy()
    if len(daily) < 2:
        return dict(mean=float(daily.mean()) if len(daily) else None, lower=None, upper=None, days=len(daily))
    rng = np.random.default_rng(seed)
    means = daily[rng.integers(0, len(daily), (draws, len(daily)))].mean(axis=1)
    return dict(mean=float(daily.mean()), lower=float(np.quantile(means, .025)),
                upper=float(np.quantile(means, .975)), days=len(daily))


def costs(b):
    c = b[b.status == 'COMPLETE']
    rows = []
    for plan_id, g in c.groupby('plan_id'):
        rows.append(dict(plan=list(MENU_PLANS[plan_id]), n=len(g), mean_net_usd=float(g.net_usd.mean()),
                         mean_gross_usd=float(g.gross_usd.mean()), fee_usd=float(g.fees_usd.mean()),
                         entry_cost_ticks=float(g.entry_cost_ticks.mean()),
                         median_latency_ms=float(g.entry_latency_ms.median()), median_hold_s=float(g.hold_s.median()),
                         exit_reasons=g.exit_reason.value_counts().to_dict()))
    return dict(by_plan=rows, statuses=b.status.value_counts().to_dict())


def design(x, premium):
    """|dP| ~ premium * (beta . z): all z causal at the origin, masks respected."""
    # A known zero sigma (frozen mid) is real; the floor only keeps log ratios finite.
    s60, s600, s3600 = (np.maximum(x[f'sigma{w}_ticks_per_sqrt_s'].to_numpy(), 1e-2) for w in op.SIGMA_WINDOWS)
    hour = np.clip((x.minutes_since_london.to_numpy() // 60).astype(int), 0, 8)
    z = [np.ones(len(x)), np.log(s60 / s600), np.log(s3600 / s600),
         (x.kind == 'event').to_numpy(float), x.spread_ticks.to_numpy(float),
         np.minimum(x.prior_events_300s.to_numpy(float), 6), x.event_bank_known.to_numpy(float)] + [
         (hour == k).astype(float) for k in range(1, 9)]
    return np.column_stack(z) * premium[:, None]


def opportunity(x, y):
    known = (x.sigma60_known & x.sigma600_known & x.sigma3600_known & x.mid_known).to_numpy()
    cost = x.spread_ticks.to_numpy() + FEE_TICKS
    out = []
    for h in op.HORIZONS:
        ok = known & y[f'mature_{h}'].to_numpy()
        a = ok & (x.block == 'A').to_numpy(); bb = ok & (x.block == 'B').to_numpy()
        target = y[f'abs_move_ticks_{h}'].to_numpy(); premium = y[f'premium_ticks_{h}'].to_numpy()
        # Equal days, as in every other readout.
        w = 1 / x.groupby('date').origin_id.transform('size').to_numpy(float)
        scale = np.average(target[a], weights=w[a]) / np.average(premium[a], weights=w[a])
        za = design(x[a], premium[a]); sw = np.sqrt(w[a])
        beta = np.linalg.lstsq(za * sw[:, None], target[a] * sw, rcond=None)[0]
        row = dict(horizon_s=h, scale_fit_on_A=float(scale), coefficients=beta.tolist(),
                   coefficient_names=['premium', 'log_s60_s600', 'log_s3600_s600', 'is_event', 'spread',
                                      'prior_events_300s', 'event_bank_known'] + [f'hour_{k}' for k in range(1, 9)])
        for name, m in (('A', a), ('B', bb)):
            pred = design(x[m], premium[m]) @ beta
            mse = lambda p: float(np.average((target[m] - p) ** 2, weights=w[m]))
            base, scaled, model = mse(premium[m]), mse(scale * premium[m]), mse(pred)
            row[name] = dict(n=int(m.sum()), days=int(x.date[m].nunique()),
                mean_abs_move_ticks=float(np.average(target[m], weights=w[m])),
                mean_premium_ticks=float(np.average(premium[m], weights=w[m])),
                mean_cost_ticks=float(np.average(cost[m], weights=w[m])),
                cost_over_move=float(np.average(cost[m], weights=w[m]) / np.average(target[m], weights=w[m])),
                straddle_mean_ticks=float(np.average(target[m] - premium[m], weights=w[m])),
                mse_raw_premium=base, mse_scaled_premium=scaled, mse_challenger=model,
                skill_vs_scaled=1 - model / scaled,
                calibration_slope=float(np.polyfit(pred, target[m], 1)[0]))
            for kind in ('event', 'clock'):
                k = (x.kind[m] == kind).to_numpy()
                row[name][f'{kind}_abs_over_premium'] = float(target[m][k].mean() / premium[m][k].mean())
        out.append(row)
    return out


def clusters(x):
    """Causal cluster labels. Volatility terciles are cut on block A clock origins only."""
    s = x.sigma600_ticks_per_sqrt_s.where(x.sigma600_known)
    edges = s[(x.block == 'A') & (x.kind == 'clock')].quantile([1 / 3, 2 / 3]).to_numpy()
    vol = np.where(s.isna(), 'vol_unknown', np.where(s <= edges[0], 'vol_low', np.where(s <= edges[1], 'vol_mid', 'vol_high')))
    ev = x.ev_event_class.astype(str) + '/' + x.ev_activity.astype(str)
    return pd.Series(np.where(x.kind == 'event', 'event:' + ev, 'clock:' + vol), index=x.index), edges.tolist()


def signatures(x, b):
    label, edges = clusters(x)
    c = b[b.status == 'COMPLETE'].merge(pd.DataFrame(dict(origin_id=x.origin_id, cluster=label)), on='origin_id')
    c['action'] = c.relative + '/P' + c.plan_id.astype(str)
    c['win'] = (c.net_usd > 0).astype(float)
    origins_per_cluster = label.groupby([label, x.block]).size()
    cells = []
    for (cluster, action), g in c.groupby(['cluster', 'action']):
        cell = dict(cluster=cluster, action=action)
        for block in ('A', 'B'):
            gb = g[g.block == block]
            cell[block] = dict(n=len(gb), days=int(gb.date.nunique()), net=day_boot(gb, 'net_usd'),
                               win_rate=float(gb.win.mean()) if len(gb) else None,
                               mfe_usd=float(gb.mfe_usd.mean()) if len(gb) else None,
                               mae_usd=float(gb.mae_usd.mean()) if len(gb) else None)
        cells.append(cell)
    # Reproducibility of the RELATIVE preference inside a cluster: continue minus fade
    # (or long minus short) per plan, block A versus block B, equal days.
    pairs = []
    for (cluster, plan_id), g in c.groupby(['cluster', 'plan_id']):
        rel = sorted(g.relative.unique())
        if len(rel) != 2:
            continue
        first = 'continue' if 'continue' in rel else 'long'
        means = {}
        for block in ('A', 'B'):
            gb = g[g.block == block]
            per = gb.pivot_table(index='origin_id', columns='relative', values='net_usd').dropna()
            per = per.join(gb.groupby('origin_id').date.first())
            diff = (per[first] - per[[r for r in rel if r != first][0]]).groupby(per.date).mean()
            means[block] = (float(diff.mean()), len(diff)) if len(diff) else (None, 0)
        pairs.append(dict(cluster=cluster, plan_id=int(plan_id), contrast=f'{first}_minus_other',
                          A=means['A'][0], B=means['B'][0], days_A=means['A'][1], days_B=means['B'][1]))
    pf = pd.DataFrame(pairs).dropna(subset=['A', 'B'])
    pf = pf[(pf.days_A >= 10) & (pf.days_B >= 10)]
    reproducibility = dict(cells=len(pf), sign_agreement=float((np.sign(pf.A) == np.sign(pf.B)).mean()) if len(pf) else None,
                           spearman=float(pf.A.rank().corr(pf.B.rank())) if len(pf) > 2 else None,
                           meaning='Relative side preference within a causal cluster; >0.5 agreement alone is not significance')
    # PDF priority value(s) = P(s) * [max_a Q_A(s,a)]_+ * 1[same action positive on B]
    value = []
    for cluster in label.unique():
        cs = [cell for cell in cells if cell['cluster'] == cluster and cell['A']['net']['mean'] is not None]
        if not cs:
            continue
        best = max(cs, key=lambda cell: cell['A']['net']['mean'])
        p = origins_per_cluster.get((cluster, 'A'), 0) / max(1, (x.block == 'A').sum())
        b_mean = best['B']['net']['mean']
        value.append(dict(cluster=cluster, frequency_A=float(p), best_action_A=best['action'],
                          best_net_A=best['A']['net']['mean'], same_action_net_B=b_mean,
                          value=float(p * max(0., best['A']['net']['mean']) * (b_mean is not None and b_mean > 0))))
    return dict(volatility_edges_ticks_per_sqrt_s=edges, cells=cells, relative_pairs=pairs,
                reproducibility=reproducibility, value=sorted(value, key=lambda v: -v['best_net_A']))


def direction_hurdle(x, y, b):
    """Hindsight diagnostic: what a perfect side call is worth, and the side
    accuracy needed to break even, per plan and per causal premium/cost bucket."""
    c = b[b.status == 'COMPLETE']
    per = c.pivot_table(index=['origin_id', 'plan_id'], columns='side', values='net_usd').dropna().reset_index()
    per = per.merge(x[['origin_id', 'date', 'block', 'spread_ticks', 'sigma600_known']], on='origin_id')
    per = per.merge(y[['origin_id', 'premium_ticks_600']], on='origin_id')
    per['right'] = per[[-1, 1]].max(axis=1); per['wrong'] = per[[-1, 1]].min(axis=1)
    ratio = (per.premium_ticks_600 / (per.spread_ticks + FEE_TICKS)).where(per.sigma600_known)
    edges = ratio[per.block == 'A'].quantile([.2, .4, .6, .8]).to_numpy()
    per['bucket'] = np.where(ratio.isna(), -1, np.searchsorted(edges, ratio.to_numpy()))
    rows = []
    for (plan_id, bucket), g in per.groupby(['plan_id', 'bucket']):
        right, wrong = g.groupby('date').right.mean().mean(), g.groupby('date').wrong.mean().mean()
        rows.append(dict(plan=list(MENU_PLANS[plan_id]), premium600_over_cost_bucket=int(bucket), n=len(g),
                         days=int(g.date.nunique()), right_side_net_usd=float(right), wrong_side_net_usd=float(wrong),
                         breakeven_side_accuracy=breakeven(right, wrong)))
    return dict(bucket_edges=edges.tolist(), rows=rows)


def breakeven(right, wrong):
    """p* with p*right + (1-p)*wrong = 0; None when even a perfect side call loses."""
    if right <= 0:
        return None
    return float(-wrong / (right - wrong)) if wrong < 0 else 0.


def markdown(s):
    lines = ['# Карта возможностей и сигнатуры исходов — train, блоки A→B', '',
             f"Дней: {s['coverage']['days']}, origins: {s['coverage']['origins']} (событий {s['coverage']['event_origins']}). "
             'A = окт–дек 2025, B = янв–фев 2026; обе части внутри train. Validation/evaluation не читались.', '',
             '## Издержки (1 oz, все ветви COMPLETE)', '', '| План take/stop/сек | N | net $ | gross $ | вход, тики | удержание, с |', '|---|---:|---:|---:|---:|---:|']
    for r in s['costs']['by_plan']:
        lines.append(f"| {'/'.join(map(str, r['plan']))} | {r['n']} | {r['mean_net_usd']:.3f} | {r['mean_gross_usd']:.3f} | "
                     f"{r['entry_cost_ticks']:.2f} | {r['median_hold_s']:.0f} |")
    lines += ['', '## Opportunity: движение против причинной премии', '',
              '| h, с | блок | |ΔP|, тики | премия | издержки/движение | skill vs scaled | наклон калибровки |', '|---:|---|---:|---:|---:|---:|---:|']
    for r in s['opportunity']:
        for blk in ('A', 'B'):
            v = r[blk]
            lines.append(f"| {r['horizon_s']} | {blk} | {v['mean_abs_move_ticks']:.2f} | {v['mean_premium_ticks']:.2f} | "
                         f"{v['cost_over_move']:.3f} | {v['skill_vs_scaled']:+.4f} | {v['calibration_slope']:.3f} |")
    rep = s['signatures']['reproducibility']
    lines += ['', '## Сигнатуры', '',
              f"Воспроизводимость относительного выбора стороны внутри кластера (A→B): ячеек {rep['cells']}, "
              f"совпадение знака {rep['sign_agreement']}, Spearman {rep['spearman']}.", '',
              '| Кластер | лучшее действие A | net A $ | то же на B $ | частота A |', '|---|---|---:|---:|---:|']
    for v in s['signatures']['value'][:12]:
        nb = 'n/a' if v['same_action_net_B'] is None else f"{v['same_action_net_B']:.3f}"
        lines.append(f"| {v['cluster']} | {v['best_action_A']} | {v['best_net_A']:.3f} | {nb} | {v['frequency_A']:.3f} |")
    lines += ['', '## Цена знания направления (hindsight-диагностика)', '',
              '| План | корзина премия/издержки | верная сторона $ | неверная $ | безубыточная точность |', '|---|---:|---:|---:|---:|']
    for r in s['direction_hurdle']['rows']:
        acc = 'недостижима' if r['breakeven_side_accuracy'] is None else f"{r['breakeven_side_accuracy']:.3f}"
        bucket = 'σ неизв.' if r['premium600_over_cost_bucket'] < 0 else r['premium600_over_cost_bucket']
        lines.append(f"| {'/'.join(map(str, r['plan']))} | {bucket} (n={r['n']}) | {r['right_side_net_usd']:.3f} | "
                     f"{r['wrong_side_net_usd']:.3f} | {acc} |")
    lines += ['', 'Независимые старты с отдельным счётом; суммы не являются кривой капитала. Это development evidence на ранее изученной истории.']
    return '\n'.join(lines) + '\n'


def report(data=DEFAULT_OUTPUT, output=REPORT):
    manifest, x, y, b = load(data)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    s = dict(version='OPPORTUNITY_SIGNATURES_READOUT_V1', source_manifest_sha256=sha(Path(data) / 'manifest.json'),
             code_sha256=sha(Path(__file__)),
             coverage=dict(days=int(x.date.nunique()), origins=len(x), event_origins=int((x.kind == 'event').sum()),
                           days_by_block=x.groupby('block').date.nunique().to_dict()),
             costs=costs(b), opportunity=opportunity(x, y), signatures=signatures(x, b),
             direction_hurdle=direction_hurdle(x, y, b))
    save_json(out / 'summary.json', s)
    (out / 'SUMMARY_RU.md').write_text(markdown(s), encoding='utf8')
    return s


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', default=DEFAULT_OUTPUT); p.add_argument('--output', default=REPORT)
    a = p.parse_args(); report(a.data, a.output)
