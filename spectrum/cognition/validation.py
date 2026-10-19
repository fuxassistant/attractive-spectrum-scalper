"""Frozen, preregistered validation of the schema + passive execution + queue
procedure on March-April 2026 (block V).

The procedure, its parameters and every pass criterion are fixed in
reports/validation_v1/preregistration.json before any block-V outcome is
produced. This module refuses to run if any frozen source file changed.
Block V was viewed by other project studies (it is not a blind holdout); it is
the first contact of THIS procedure with these months.
"""
from dataclasses import replace
from pathlib import Path
import glob
import json
import time
import numpy as np
import pandas as pd
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from . import schemas as sm
from . import schema_run as sr
from . import passive_learn as pl
from . import signature_report as srep
from . import opportunity as op
from .qmodel import dataset as qdataset
from .signatures import MENU_PLANS

REPORT = Path('reports/validation_v1')
PREREGISTRATION = REPORT / 'preregistration.json'
EXECUTION = 'PQ_30'
TRAIN = dict(signatures='data/derived/opportunity_signatures_v1', representation='data/derived/representation_v1',
             passive='data/derived/passive_execution_v1', queue='data/derived/queue_state_v1')
VALID = dict(signatures='data/derived/opportunity_signatures_validation_v1',
             representation='data/derived/representation_validation_v1',
             passive='data/derived/passive_execution_validation_v1', queue='data/derived/queue_state_validation_v1')
FROZEN_FILES = ('execution.py', 'opportunity.py', 'signatures.py', 'signature_report.py', 'representation.py',
                'qmodel.py', 'schemas.py', 'schema_run.py', 'queue.py', 'passive.py', 'queue_state.py',
                'passive_learn.py', 'passive_report.py', 'validation.py')
REFERENCE = Path('reports/schemas_queue_v1/PQ_30/schema_summary.json')
SEED = 61071


def code_hashes():
    here = Path(__file__).parent
    return {name: sha(here / name) for name in FROZEN_FILES}


def check_frozen():
    pre = json.loads(PREREGISTRATION.read_text(encoding='utf8'))
    now = code_hashes()
    changed = [k for k, v in pre['code_sha256'].items() if now.get(k) != v]
    if changed:
        raise RuntimeError(f'Frozen procedure changed after preregistration: {changed}')
    return pre


def paired_daily(frame, a, b):
    """Day-bootstrap of the daily mean difference a - b (days having both)."""
    d = frame.groupby('date')[[a, b]].mean().dropna()
    return srep.day_boot(pd.DataFrame(dict(date=d.index, diff=(d[a] - d[b]).to_numpy())), 'diff')


# ---- schemas (H1, H2, H6, R0) ----------------------------------------------
def schema_frame():
    frames = []
    for paths in (TRAIN, VALID):
        f, _, _ = qdataset(paths['signatures'], paths['representation'])
        qs = pd.concat([pd.read_parquet(p) for p in glob.glob(str(Path(paths['queue']) / '2*.parquet'))])
        f = f.join(qs.set_index('origin_id').drop(columns=['intent_ns', 'spread_ticks']), how='left')
        frames.append(f)
    frame = pd.concat(frames).sort_values('intent_ns', kind='stable')
    frame['state_known'] = frame.state_known.fillna(0)
    queue_cols = [c for c in frames[0].columns if c.startswith(('bid', 'ask', 'tbuy', 'tsell'))]
    frame[queue_cols] = frame[queue_cols].fillna(0)
    frame['day_index'] = frame.date.rank(method='dense').astype(int)
    return frame


def schema_outcomes(frame, execution):
    out = {}
    for paths in (TRAIN, VALID):
        out.update(sr.outcome_table(paths['signatures']) if execution == 'taker'
                   else sr.passive_outcomes(execution, paths['passive']))
    return sr.fix_unknown_times(out, frame)


def run_schemas(frame):
    programs = sm.PROGRAMS + sm.QUEUE_PROGRAMS; coords = sm.COORDINATES + sm.QUEUE_COORDINATES
    passive = schema_outcomes(frame, EXECUTION); taker = schema_outcomes(frame, 'taker')
    v_start = int(frame.intent_ns[frame.block == 'V'].min())
    runs = dict(prequential=(dict(), passive, None), frozen=(dict(freeze_at_block='V'), passive, None),
                assimilate_only=(dict(accommodation=False), passive, None),
                shuffled_context=(dict(), passive, 61051), taker_prequential=(dict(), taker, None))
    results = {}
    for name, (overrides, stream, seed) in runs.items():
        lib, trades = sm.run(frame, stream, replace(sm.Params(), **overrides), shuffle_seed=seed,
                             programs=programs, coordinate_names=coords)
        events_v = [e for e in lib.events if int(e['ns']) >= v_start]
        results[name] = dict(A=sr.block_metrics(trades, frame, 'A'), B=sr.block_metrics(trades, frame, 'B'),
                             V=sr.block_metrics(trades, frame, 'V'),
                             events_train=sum(int(e['ns']) < v_start for e in lib.events), events_V=events_v,
                             promoted_V=sum(e['kind'] == 'active' for e in events_v),
                             statuses=pd.Series([k.status for k in lib.schemas]).value_counts().to_dict(),
                             monthly=sr.monthly(trades))
        save_json(REPORT / f'library_{name}.json', dict(schemas=sr.library_view(lib), events=lib.events))
        print(name, {b: results[name][b]['trades'] for b in 'ABV'}, results[name]['promoted_V'], flush=True)
    return results


def reproducibility(results):
    """R0: the prequential library through February equals the earlier train-only run."""
    ref = json.loads(REFERENCE.read_text(encoding='utf8'))['variants']
    checks = {}
    for name, ref_name in (('prequential', 'full'), ('assimilate_only', 'assimilate_only'), ('shuffled_context', 'shuffled_context')):
        r, q = results[name], ref[ref_name]
        checks[name] = dict(A_equal=r['A'] == q['A'], B_equal=r['B'] == q['B'],
                            train_events=r['events_train'], reference_events=q['spawned'] + q['promoted'] + q['refuted'])
    return checks


def knowledge(frame):
    """H6: '[trend=down] -> short/P2' (40/20/600) is worse than its root on V."""
    stream = schema_outcomes(frame, EXECUTION)
    coords = sm.coordinates(frame); v = (frame.block == 'V').to_numpy()
    rows = []
    for i in np.flatnonzero(v):
        o = stream.get((frame.index[i], 2, -1))
        if o and o[0] in ('COMPLETE', 'EMPTY'):
            rows.append(dict(date=frame.date.iat[i], y=o[1] if o[0] == 'COMPLETE' else 0., child=coords['trend'][i] == 'down'))
    d = pd.DataFrame(rows)
    daily = d.groupby('date').apply(lambda g: pd.Series(dict(child=g.y[g.child].mean(), parent=g.y.mean())), include_groups=False)
    daily = daily.dropna().reset_index()
    return dict(n_child=int(d.child.sum()), n_parent=len(d), child_mean=float(d.y[d.child].mean()), parent_mean=float(d.y.mean()),
                residual=paired_daily(daily, 'child', 'parent'))


# ---- execution (H3) ------------------------------------------------------
def execution():
    f = sr_load_passive(VALID['passive'])
    tt = f[f.variant == 'TT'][['origin_id', 'side', 'plan_id', 'date', 'attempt', 'net_usd', 'status']]
    pq = f[f.variant == EXECUTION][['origin_id', 'side', 'plan_id', 'attempt', 'status']]
    m = tt.merge(pq, on=['origin_id', 'side', 'plan_id'], suffixes=('_tt', '_pq'))
    m = m[m.attempt_tt.notna() & m.attempt_pq.notna()]
    per_plan = {str(MENU_PLANS[p]): dict(tt=float(g.attempt_tt.mean()), pq=float(g.attempt_pq.mean()),
                                         improvement=paired_daily(g, 'attempt_pq', 'attempt_tt'))
                for p, g in m.groupby('plan_id')}
    m = m[m.status_tt == 'COMPLETE']
    m['unfilled'] = np.where(m.status_pq == 'NOT_FILLED', m.net_usd, np.nan)
    m['filled'] = np.where(m.status_pq == 'COMPLETE', m.net_usd, np.nan)
    return dict(per_plan=per_plan, taker_where_unfilled=float(np.nanmean(m.unfilled)),
                taker_where_filled=float(np.nanmean(m.filled)), selection_gap=paired_daily(m, 'unfilled', 'filled'))


def sr_load_passive(path):
    from .passive_report import load
    f = load(path)
    f['block'] = 'V'
    return f


# ---- resting ranking (H4) -------------------------------------------------
def resting():
    xt, qt, rt = pl.dataset(EXECUTION, TRAIN['passive'], TRAIN['queue'], TRAIN['representation'], TRAIN['signatures'])
    xv, qv, rv = pl.dataset(EXECUTION, VALID['passive'], VALID['queue'], VALID['representation'], VALID['signatures'])
    Ft, Fv = pd.concat([qt, rt], axis=1), pd.concat([qv, rv], axis=1)
    out = {}
    for plan in sorted(xt.plan_id.unique()):
        t = (xt.plan_id == plan).to_numpy(); v = (xv.plan_id == plan).to_numpy()
        yt, yv = xt.attempt.to_numpy(float)[t], xv.attempt.to_numpy(float)[v]
        w = pl.weights(xt.date[t]); m = pl.model().fit(Ft[t], yt, sample_weight=w)
        c = float(np.average(yt, weights=w)); pv = m.predict(Fv[v]); wv = pl.weights(xv.date[v])
        r2 = 1 - np.average((yv - pv) ** 2, weights=wv) / np.average((yv - c) ** 2, weights=wv)
        edges = np.quantile(m.predict(Ft[t]), [.5, .9])
        b = xv[v].assign(bucket=np.searchsorted(edges, pv))
        b['top'] = np.where(b.bucket == 2, b.attempt, np.nan); b['bottom'] = np.where(b.bucket == 0, b.attempt, np.nan)
        out[str(MENU_PLANS[plan])] = dict(r2_vs_train_mean=float(r2), top10=float(np.nanmean(b.top)),
                                          bottom50=float(np.nanmean(b.bottom)),
                                          top_fill=float((b[b.bucket == 2].status == 'COMPLETE').mean()),
                                          gap=paired_daily(b, 'top', 'bottom'))
    return out


# ---- opportunity (H5) -----------------------------------------------------
def opportunity():
    _, xt, yt, _ = srep.load(TRAIN['signatures']); _, xv, yv, _ = srep.load(VALID['signatures'])
    out = {}
    def known(x, y, h):
        return (x.sigma60_known & x.sigma600_known & x.sigma3600_known & x.mid_known).to_numpy() & y[f'mature_{h}'].to_numpy()
    for h in op.HORIZONS:
        kt = known(xt, yt, h); target = yt[f'abs_move_ticks_{h}'].to_numpy(); prem = yt[f'premium_ticks_{h}'].to_numpy()
        w = 1 / xt.groupby('date').origin_id.transform('size').to_numpy(float)
        scale = np.average(target[kt], weights=w[kt]) / np.average(prem[kt], weights=w[kt])
        z = srep.design(xt[kt], prem[kt]); sw = np.sqrt(w[kt])
        beta = np.linalg.lstsq(z * sw[:, None], target[kt] * sw, rcond=None)[0]
        row = {}
        for month in sorted(xv.date.str[:7].unique()):
            m = known(xv, yv, h) & (xv.date.str[:7] == month).to_numpy()
            tv = yv[f'abs_move_ticks_{h}'].to_numpy()[m]; pv = yv[f'premium_ticks_{h}'].to_numpy()[m]
            wv = 1 / xv[m].groupby('date').origin_id.transform('size').to_numpy(float)
            pred = srep.design(xv[m], pv) @ beta
            mse = lambda p: float(np.average((tv - p) ** 2, weights=wv))
            row[month] = dict(n=int(m.sum()), skill_vs_scaled=1 - mse(pred) / mse(scale * pv),
                              slope=float(np.polyfit(pred, tv, 1)[0]), cost_over_move=float(
                                  np.average(xv.spread_ticks.to_numpy()[m] + 3, weights=wv) / np.average(tv, weights=wv)))
        out[h] = row
    return out


def verdicts(s, pre):
    c = pre['criteria']; v = {}
    pq = s['schemas']['prequential']['V']; fz = s['schemas']['frozen']['V']
    def econ(m):
        if m['trades'] == 0:
            return 'NO_TRADE'
        return 'PASS' if m['net'] > 0 and (m['daily']['lower'] or -1) > 0 else 'FAIL'
    v['H1_prequential'] = econ(pq); v['H1_frozen'] = econ(fz)
    sh = s['schemas']['shuffled_context']
    v['H2'] = 'PASS' if sh['promoted_V'] == 0 and sh['V']['trades'] == 0 else 'FAIL'
    v['H3_cost'] = 'PASS' if all(x['improvement']['lower'] > 0 for x in s['execution']['per_plan'].values()) else 'FAIL'
    v['H3_selection'] = 'PASS' if s['execution']['selection_gap']['lower'] > 0 else 'FAIL'
    r = s['resting'][c['H4_plan']]
    v['H4'] = 'PASS' if r['r2_vs_train_mean'] > 0 and r['gap']['lower'] > 0 else 'FAIL'
    v['H5'] = {str(h): 'PASS' if all(x['skill_vs_scaled'] > 0 for x in row.values()) else 'FAIL'
               for h, row in s['opportunity'].items()}
    v['H6'] = 'PASS' if s['knowledge']['residual']['upper'] < 0 else 'FAIL'
    v['R0'] = 'PASS' if all(x['A_equal'] and x['B_equal'] for x in s['reproducibility'].values()) else 'FAIL'
    return v


def validate():
    started = time.perf_counter(); pre = check_frozen(); REPORT.mkdir(parents=True, exist_ok=True)
    frame = schema_frame()
    s = dict(version='FROZEN_VALIDATION_V1', preregistration_sha256=sha(PREREGISTRATION),
             days_V=int(frame.date[frame.block == 'V'].nunique()), origins_V=int((frame.block == 'V').sum()))
    s['schemas'] = run_schemas(frame); s['reproducibility'] = reproducibility(s['schemas'])
    s['knowledge'] = knowledge(frame); s['execution'] = execution(); s['resting'] = resting()
    s['opportunity'] = opportunity(); s['verdicts'] = verdicts(s, pre); s['seconds'] = time.perf_counter() - started
    save_json(REPORT / 'validation_summary.json', s)
    print(json.dumps(s['verdicts'], indent=1))
    return s


if __name__ == '__main__':
    validate()
