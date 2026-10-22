"""Frozen flow ablation, explicit fill/path heads and compressed day context."""
from pathlib import Path
import argparse
import json
import pickle
import subprocess
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from spectrum.cognition.execution import QuoteTape
from spectrum.cognition.episodes_lab import policy_stats
from scripts.flow_context_coverage import save, sha
from . import flow_context as fc
from . import episodes as ep

DATA = Path('data/derived/flow_context_v2')
REPORT = Path('reports/flow_context_v2')
MARKET = Path('data/derived/basket_multimonth_v1')
CONTRACT = Path('contracts/flow_context_v2.json')


def identity():
    paths = [CONTRACT, Path(fc.__file__), Path(ep.__file__), Path(__file__), Path('contracts/opportunity_episodes_v1.json')]
    return {str(p): sha(p) for p in paths}


def build():
    c = fc.contract(); ids = identity(); DATA.mkdir(parents=True, exist_ok=True)
    coverage = json.loads((REPORT / 'coverage_audit.json').read_text())
    if coverage['status'] != 'PASS':
        raise ValueError('Coverage audit required')
    source_proof = Path('reports/opportunity_episodes_v1/source_code_compatibility.json')
    for name, proof in json.loads(source_proof.read_text()).items():
        if sha(Path(ep.__file__).with_name(name)) != proof['current']:
            raise ValueError('Source execution code changed: ' + name)
    receipts = []; excluded = []; previous_cache = {}
    for date, source_hash in coverage['source_sha256'].items():
        begin = time.perf_counter(); source = Path(c['source']) / (date + '.parquet')
        if sha(source) != source_hash:
            raise ValueError('Source changed after coverage audit')
        rp = DATA / (date + '.json'); dest = DATA / (date + '.parquet')
        if rp.exists():
            r = json.loads(rp.read_text())
            if r['identity'] != ids or r['source_sha256'] != source_hash:
                raise ValueError('Cached identity mismatch')
            if r.get('excluded'):
                excluded.append(r); continue
            if sha(dest) != r['sha256']:
                raise ValueError('Cached rows changed')
            receipts.append(r); continue
        names = pq.ParquetFile(source).schema_arrow.names
        cols = [n for n in names if not n.startswith(('P_', 'T_'))] + [f'{mode}_{side}_f24_12_{field}' for mode in ('P', 'T') for side in ('L', 'S') for field in ('st', 'net', 'busy_s')]
        d = pd.read_parquet(source, columns=cols)
        if d.empty:
            r = dict(date=date, excluded='NO_SOURCE_DECISIONS', source_sha256=source_hash, identity=ids)
            save(rp, r); excluded.append(r); continue
        day = MonthDay(MARKET / date[:7] / date, verify=True); q = QuoteTape.from_day(day)
        times = d.t_ns.to_numpy(np.int64); j = np.searchsorted(q.quote_t, times, side='right') - 1
        if (j < 0).any():
            raise ValueError('Source precedes quote support')
        x, _ = fc.local_features(d, q.epoch[j])
        gate = ((times // ep.NS) % 30 == 0) & (d.sigma_known.to_numpy() > 0) & (d.sigma600.to_numpy() >= .3)
        gate &= (d.spread_ticks.to_numpy() > 0) & (times + 335 * ep.NS < q.deadline_ns) & q.valid[j] & (q.epoch[j] >= 0)
        k = np.flatnonzero(gate); intents = times[k]; x = x.iloc[k].reset_index(drop=True)
        spread = (q.ask[j[k]] - q.bid[j[k]]) / 1e8
        if not np.allclose(spread, d.spread_ticks.to_numpy()[k], atol=1e-6, rtol=0):
            raise ValueError('Spread/BBO mismatch')
        current = fc.minute_tape(day)
        previous_date = str((pd.Timestamp(date) - pd.offsets.BDay()).date()); previous_path = MARKET / previous_date[:7] / previous_date
        previous = previous_cache.get(previous_date)
        previous_manifest = None
        if previous is None and (previous_path / 'manifest.json').exists():
            prior = MonthDay(previous_path, verify=True); previous = fc.minute_tape(prior)
        if previous is not None:
            previous_manifest = sha(previous_path / 'manifest.json')
        mid = (q.ask[j[k]].astype(float) + q.bid[j[k]]) / 2e8
        ctx, clocks = fc.context_features(current, previous, intents, mid, d.sigma600.to_numpy()[k])
        if np.any(clocks['latest_available_ns'] > intents) or np.any(clocks['latest_endpoint_ns'] > intents):
            raise ValueError('Context not yet published')
        x = pd.concat([x, ctx], axis=1); frames = []
        for mode in ('P', 'T'):
            for side, sn in ((1, 'L'), (-1, 'S')):
                key = f'{mode}_{sn}_f24_12'; f = fc.orient(x, side)
                f['date'] = date; f['t_ns'] = intents; f['mode'] = mode
                for name in ('st', 'net', 'busy_s'):
                    f[{'st': 'status'}.get(name, name)] = d[key + '_' + name].to_numpy()[k]
                f['category'] = ep.categories(f.status, f.net)
                frames.append(f)
        out = pd.concat(frames, ignore_index=True)
        for col in set(sum(fc.FAMILIES.values(), [])):
            out[col] = out[col].astype(np.float32)
        if any(n.startswith(('P_', 'T_', 'target', 'future')) for n in sum(fc.FAMILIES.values(), [])):
            raise ValueError('Target in feature allowlist')
        out.to_parquet(dest, index=False)
        r = dict(date=date, roots=len(k), rows=len(out), identity=ids, source_sha256=source_hash, sha256=sha(dest),
                 current_manifest_sha256=sha(day.path / 'manifest.json'), previous_date=previous_date, previous_manifest_sha256=previous_manifest,
                 previous_known_roots=int((ctx.previous_known > .5).sum()),
                 known_fraction={col: float(np.isfinite(x[col]).mean()) if len(x) else None for col in fc.FAMILIES['rich_spectrum'] if col != 'side'},
                 max_context_lookahead_ns=int(max(0, np.max(clocks['latest_available_ns'] - intents))),
                 spread_bbo_max_error=float(np.max(np.abs(spread - d.spread_ticks.to_numpy()[k]))), seconds=round(time.perf_counter() - begin, 2))
        save(rp, r); receipts.append(r); previous_cache = {date: current}; QUOTE_CACHE.clear()
        print(json.dumps({key: r[key] for key in ('date', 'roots', 'previous_known_roots', 'seconds')}), flush=True)
    manifest = dict(status='COMPLETE', identity=ids, days=receipts, excluded=excluded, families=fc.FAMILIES,
                    coverage_sha256=sha(REPORT / 'coverage_audit.json'), source_execution_proof_sha256=sha(source_proof),
                    quote_hash_scope='10 calendar dates checked in coverage audit; all source label hashes and prepared M1 arrays checked; canonical quote file sizes and all selected spreads checked in build')
    save(DATA / 'manifest.json', manifest)
    save(REPORT / 'data_receipt.json', dict(status='COMPLETE', days=len(receipts), roots=sum(r['roots'] for r in receipts), rows=sum(r['rows'] for r in receipts),
        manifest_sha256=sha(DATA / 'manifest.json'), identity=ids, families=fc.FAMILIES,
        daily=[{k: r[k] for k in ('date', 'roots', 'previous_known_roots', 'known_fraction', 'previous_date')} for r in receipts]))


def fit_head(x, y):
    unique = np.unique(y)
    if len(unique) == 1:
        return int(unique[0])
    model = HistGradientBoostingClassifier(random_state=fc.contract()['seed'], **fc.contract()['model'])
    return model.fit(x, y)


def head_probability(model, x, classes):
    out = np.zeros((len(x), classes))
    if isinstance(model, int):
        out[:, model] = 1
    else:
        out[:, model.classes_.astype(int)] = model.predict_proba(x)
    return out


def fit_predict(train, test, family):
    columns = fc.FAMILIES[family]; known = train.category >= 0; tr = train.loc[known]
    x = tr[columns].to_numpy(np.float32); y = tr.category.to_numpy(int)
    values = np.array([tr.loc[tr.category == k, 'net'].mean() for k in range(4)])
    if not np.isfinite(values).all():
        values = np.where(np.isfinite(values), values, [2.1, -1.5, -.30, 0.])
    if family == 'shuffled':
        y = np.random.default_rng(fc.contract()['seed']).permutation(y)
    filled = y < 3
    fill_model = fit_head(x, filled.astype(int)); path_model = fit_head(x[filled], y[filled])
    xx = test[columns].to_numpy(np.float32)
    pf = head_probability(fill_model, xx, 2)[:, 1]; pc = head_probability(path_model, xx, 3)
    joint = fc.factor_probabilities(pf, pc)
    fields = ['date', 't_ns', 'mode', 'side', 'status', 'category', 'net', 'busy_s', 'spread_ticks', 'log_sigma', 'previous_known']
    out = test[fields].copy(); out['p_fill'] = pf
    for k, name in enumerate(ep.CLASSES):
        out['p_' + name] = joint[:, k]
        if k < 3:
            out['c_' + name] = pc[:, k]
    out['expected_net'] = joint @ values
    fit = dict(n_known=len(tr), n_filled=int(filled.sum()), columns=columns, class_values=values.tolist(),
               fill_head='constant' if isinstance(fill_model, int) else 'HGB', weighting='uniform')
    return out, dict(fill=fill_model, path=path_model, class_values=values, columns=columns), fit


def binary_metrics(y, p):
    p = np.clip(np.asarray(p), 1e-9, 1 - 1e-9); y = np.asarray(y, bool)
    return dict(log_loss=float(-np.mean(np.where(y, np.log(p), np.log1p(-p)))), brier=float(np.mean((p - y) ** 2)),
                auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, observed=float(y.mean()), predicted=float(p.mean()))


def scores(frame):
    known = frame.category >= 0; f = frame.loc[known]; y = f.category.to_numpy(int); filled = y < 3
    pf = np.clip(f.p_fill.to_numpy(), 1e-9, 1 - 1e-9); pc = f[['c_' + c for c in ep.CLASSES[:3]]].to_numpy()
    fill = np.where(filled, np.log(pf), np.log1p(-pf)); path = np.zeros(len(f)); take = path.copy()
    path[filled] = np.log(np.clip(pc[filled, y[filled]], 1e-9, 1))
    pt = np.clip(pc[filled, 0], 1e-9, 1 - 1e-9)
    take[filled] = np.where(y[filled] == 0, np.log(pt), np.log1p(-pt))
    return f, filled, dict(fill=fill, path=path, take=take, joint=fill + path)


def metrics(frame):
    f, filled, ss = scores(frame)
    if not len(f):
        return dict(n_known=0, n_unknown=len(frame))
    out = dict(n_known=len(f), n_unknown=len(frame) - len(f), n_filled=int(filled.sum()),
               fill=binary_metrics(filled, f.p_fill), joint_log_loss=float(-ss['joint'].mean()))
    if filled.any():
        ff = f.loc[filled]
        out['path_log_loss'] = float(-ss['path'][filled].mean())
        out['take_given_fill'] = binary_metrics(ff.category == 0, ff.c_TAKE)
        out['mean_net_given_fill'] = float(ff.net.mean())
    return out


def compare(a, b, days):
    fa, filled, sa = scores(a); fb, filled_b, sb = scores(b)
    pd.testing.assert_frame_equal(fa[['date', 't_ns', 'side', 'category']], fb[['date', 't_ns', 'side', 'category']])
    result = {}
    for name in sa:
        gain = sb[name] - sa[name]
        daily = pd.DataFrame(dict(date=fa.date, gain=gain)).groupby('date').gain.sum().reindex(days, fill_value=0.)
        result[name] = dict(mean_per_known_attempt=float(gain.mean()), mean_per_fill=float(gain[filled].mean()) if name in ('path', 'take') else None,
            daily_mean=float(daily.mean()), daily_ci=ep.block_interval(daily.to_numpy()), daily={d: float(v) for d, v in daily.items()})
    return result


def run():
    c = fc.contract(); man = json.loads((DATA / 'manifest.json').read_text())
    if man['status'] != 'COMPLETE' or man['identity'] != identity():
        raise ValueError('Dataset not complete for this frozen code')
    frames = []
    for day in man['days']:
        path = DATA / (day['date'] + '.parquet')
        if sha(path) != day['sha256']:
            raise ValueError('Dataset hash changed')
        frames.append(pd.read_parquet(path))
    all_rows = pd.concat(frames, ignore_index=True); all_rows['month'] = all_rows.date.str[:7]
    result = dict(version=c['version'], identity=identity(), stages={}, comparisons={}, coverage={}, admissions={},
                  families=fc.FAMILIES, days=len(man['days']), roots=sum(r['roots'] for r in man['days']), evaluation_status=c['evaluation_status'])
    thresholds = {}; admission = {}
    for stage, fit_months, test_months in (('A2', c['train_months'], c['development_months']), ('B', c['train_months'] + c['development_months'], c['evaluation_months'])):
        days = [r['date'] for r in man['days'] if r['date'][:7] in test_months]; result['stages'][stage] = {}
        for mode in ('P', 'T'):
            train = all_rows.loc[all_rows.month.isin(fit_months) & (all_rows['mode'] == mode)].reset_index(drop=True)
            test = all_rows.loc[all_rows.month.isin(test_months) & (all_rows['mode'] == mode)].reset_index(drop=True)
            result['stages'][stage][mode] = {}; predictions = {}
            for family in fc.FAMILIES:
                begin = time.perf_counter(); f, models, fit = fit_predict(train, test, family); predictions[family] = f
                pol, trades = policy_stats(f, calendar_days=days); key = mode + '/' + family
                roots = f.groupby('t_ns').expected_net.max()
                if stage == 'A2':
                    thresholds[key] = float(roots.quantile(.9))
                    admission[key] = bool(pol['censor'] is None and pol['fills'] >= 30 and pol['daily_ci'][0] > 0)
                forced, _ = policy_stats(f, threshold=thresholds[key], calendar_days=days)
                forced['actual_score_coverage'] = float((roots > thresholds[key]).mean())
                spread = f.spread_ticks
                res = dict(fit=fit, all=metrics(f), months={m: metrics(f.loc[f.date.str[:7] == m]) for m in test_months},
                    strata={name: metrics(f.loc[mask]) for name, mask in {'0-2': spread <= 2, '2-4': (spread > 2) & (spread <= 4), '4-8': (spread > 4) & (spread <= 8), '8+': spread > 8}.items()},
                    context={str(k): metrics(f.loc[(f.previous_known > .5) == bool(k)]) for k in (0, 1)},
                    policy=pol, forced_top10=forced, operational_admitted=admission[key], calendar_days=days)
                result['stages'][stage][mode][family] = res
                f.to_parquet(DATA / f'predictions_{stage}_{mode}_{family}.parquet', index=False)
                trades.to_parquet(DATA / f'trades_{stage}_{mode}_{family}.parquet', index=False)
                with (DATA / f'model_{stage}_{mode}_{family}.pkl').open('wb') as dst:
                    pickle.dump(models, dst)
                print(json.dumps(dict(stage=stage, mode=mode, family=family, known=res['all']['n_known'], fill_loss=res['all']['fill']['log_loss'], path_loss=res['all'].get('path_log_loss'), attempts=pol['attempts'], net=pol['net'], seconds=round(time.perf_counter() - begin, 2))), flush=True)
            for pair in c['primary_comparisons'] + ['shuffled-baseline']:
                newer, older = pair.split('-')
                result['comparisons'][stage + '/' + mode + '/' + pair] = compare(predictions[older], predictions[newer], days)
            save(REPORT / 'summary.running.json', result)
    result['admissions'] = admission; result['coverage'] = thresholds
    result['git_head_at_run'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(REPORT / 'summary.json', result)
    save(REPORT / 'completion_receipt.json', dict(status='COMPLETE', identity=identity(), summary_sha256=sha(REPORT / 'summary.json'),
         manifest_sha256=sha(DATA / 'manifest.json'), prediction_sha256={p.name: sha(p) for p in DATA.glob('predictions_*.parquet')},
         completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['build', 'run'])
    args = parser.parse_args(); {'build': build, 'run': run}[args.command]()
