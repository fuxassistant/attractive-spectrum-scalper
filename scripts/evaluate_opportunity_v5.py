"""CAL-frozen filter diagnostics and censor-safe one-position economics.

Future labels are only attached after the alert/replay decisions are fixed.
TEST exact-count rankings are explicitly descriptive and never feed replay.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from scripts.run_opportunity_v5 import read, save, PNAMES, QNAMES, validate_recipe
from spectrum.research_v3.learning import load_dataset, sha
from scripts.audit_economics_v4 import prepare_account, project_account, safe_projection
from scripts.audit_opportunity_v4 import identity_coverage, empirical_none_threshold, root_metrics

ARMS = ('base', 'semantic', 'risk', 'hash_control')
PLANS = ('cost2', 'cost4', 'cost6')


def ratio(a, b):
    return float(a/b) if b else None


def hash_scores(t_ns, seed):
    """SplitMix64; timestamp and fixed seed only, no rank/full-stream context."""
    with np.errstate(over='ignore'):
        z = np.asarray(t_ns, np.uint64) ^ np.uint64(seed)
        z = z + np.uint64(0x9e3779b97f4a7c15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        z ^= z >> np.uint64(31)
    return ((z >> np.uint64(11)).astype(np.float64)+.5)/(2.**53)


def score_channels(pred, seed):
    p, q = pred[PNAMES].to_numpy(float), pred[QNAMES].to_numpy(float)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any((q<0)|(q>1)):
        raise ValueError('Finite probabilities required')
    if not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6):
        raise ValueError('Normalized frozen base required')
    side = np.where(p[:,2]+p[:,3] > p[:,1]+p[:,3], -1, 1)
    if not np.array_equal(side, pred.side):
        raise ValueError('Filter changed frozen chosen side')
    base = np.maximum(p[:,1]+p[:,3], p[:,2]+p[:,3])
    semantic = q[:,0]*q[:,1]
    h = hash_scores(pred.t_ns.to_numpy(np.int64), seed)
    return {plan:dict(base=base, semantic=semantic, risk=semantic*q[:,2+j], hash_control=h)
            for j,plan in enumerate(PLANS)}, side


def retention_threshold(score, calibration_candidates, fraction):
    score, use = np.asarray(score, float), np.asarray(calibration_candidates, bool)
    if score.shape != use.shape or not np.isfinite(score).all() or not 0 < fraction <= 1:
        raise ValueError('Finite scores and positive retention required')
    values = np.sort(score[use])[::-1]
    count = int(np.floor(fraction*len(values)))
    threshold = (0. if fraction == 1 and len(values) else
                 float(np.nextafter(values[count], np.inf)) if len(values) else 1.01)
    actual = int((values>=threshold).sum())
    if actual > count:
        raise AssertionError('Retention budget exceeded')
    return dict(fraction=float(fraction), calibration_candidates=len(values), allowed=count,
                actual=actual, threshold=threshold, conservative_ties=count-actual)


def economic_probabilities(score, side):
    score = np.asarray(score, float)
    if not np.isfinite(score).all() or np.any((score<0)|(score>1)):
        raise ValueError('Score outside [0,1]')
    out = np.zeros((len(score),4), float)
    out[:,0] = 1-score
    out[np.arange(len(score)), np.where(np.asarray(side)==1,1,2)] = score
    return out


def alarm_metrics(frame, side, safe, candidate, alert, cohort):
    y = frame.y_crossing.to_numpy(int)
    admitted = frame.feature_available.to_numpy(bool) & np.asarray(cohort,bool)
    known, unknown = admitted & (y>=0), admitted & (y<0)
    bit = np.where(np.asarray(side)==1,1,2)
    correct = known & (y>0) & ((y & bit)!=0)
    none = known & (y==0)
    wrong = known & (y>0) & ~correct
    alert = np.asarray(alert,bool) & np.asarray(candidate,bool)
    n, tp, fp, ws = [int((alert&m).sum()) for m in (known,correct,none,wrong)]
    if n != tp+fp+ws:
        raise AssertionError('Known alarm partition mismatch')
    risksafe = correct & (safe==1)
    riskbad = correct & (safe==0)
    riskunknown = correct & (safe<0)
    return dict(known_roots=int(known.sum()), unknown_roots=int(unknown.sum()),
        causal_candidates=int((candidate&admitted).sum()), alerts=n, unknown_alerts=int((alert&unknown).sum()),
        total_causal_alerts=int((alert&admitted).sum()), true_positives=tp, none_false_positives=fp,
        wrong_side_alerts=ws, false_signals=fp+ws, precision=ratio(tp,n),
        event_opportunity_recall=ratio(tp,int((known&(y>0)).sum())),
        none_fpr=ratio(fp,int(none.sum())), wrong_side_fraction=ratio(ws,n),
        safe_correct_alerts=int((alert&risksafe).sum()), unsafe_correct_alerts=int((alert&riskbad).sum()),
        unknown_risk_correct_alerts=int((alert&riskunknown).sum()),
        safe_precision=ratio(int((alert&risksafe).sum()),n),
        unsafe_rate_among_risk_known_correct=ratio(int((alert&riskbad).sum()),int((alert&(riskbad|risksafe)).sum())),
        safe_recall=ratio(int((alert&risksafe).sum()),int(risksafe.sum())),
        positive_identity=identity_coverage(frame,known,alert,bit,'selected_side'))


def head_metrics(y, probability):
    y, p = np.asarray(y), np.asarray(probability, float)
    good = y>=0
    yy, pp = y[good], np.clip(p[good],1e-12,1-1e-12)
    if not len(yy):
        return dict(count=0, positives=0, nll=None, auc=None, ap=None, reliability=[])
    bins = []
    for lower,upper in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        m=(pp>=lower)&((pp<upper) if upper<1 else (pp<=upper))
        bins.append(dict(lower=float(lower),upper=float(upper),n=int(m.sum()),
            predicted=float(pp[m].mean()) if m.any() else None, observed=float(yy[m].mean()) if m.any() else None))
    return dict(count=len(yy), positives=int(yy.sum()), positive_fraction=float(yy.mean()),
        nll=float(-(yy*np.log(pp)+(1-yy)*np.log1p(-pp)).mean()),
        brier=float(np.square(pp-yy).mean()),
        auc=float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None,
        ap=float(average_precision_score(yy,pp)) if yy.any() else None, reliability=bins)


def replay(book, frame, score, side, plan, threshold, days, pending):
    # Feed the audited account an explicit Boolean causal gate and a one-hot
    # side. Encoding a zero score as pure NONE would silently turn SHORT into
    # LONG at threshold zero. This adapter preserves even that boundary case.
    score=np.asarray(score,float)
    if not np.isfinite(score).all() or np.any((score<0)|(score>1)):
        raise ValueError('Finite [0,1] alert score required')
    gate=score>=threshold
    gated_book={**book,'available':book['available'] & gate}
    onehot=economic_probabilities(np.ones(len(score)),side)
    summary,ledger=project_account(gated_book,onehot,plan,0.5,days,pending)
    summary['threshold']=float(threshold)
    summary['causal_feature_origins']=int((book['available'] & np.isin(book['date'],days)).sum())
    score_by_time=dict(zip(book['t'],score))
    for row in ledger:
        row['probability']=float(score_by_time[int(row['t_ns'])])
    return safe_projection(frame,summary,ledger),ledger


def evaluate():
    config_path='configs/research/opportunity_pipeline_v5.json'
    c=read(config_path); validate_recipe(c)
    report=Path('reports/scientific_pipeline_v5')
    suite=read(report/'suite_summary.json')
    if suite['status']!='COMPLETE' or suite['config_sha256']!=sha(config_path):
        raise ValueError('Complete bound V5 suite required')
    store=load_dataset(c['data']['root']); f=store.frame
    root=Path(c['data']['root'])
    execution=pd.concat([pd.read_parquet(root/(d+'_execution.parquet')) for d in c['data']['dates']],ignore_index=True)
    book0=prepare_account(f,execution,c)
    results=[]
    for run in suite['runs']:
        seed=run['seed']; out=Path(suite['runs_root'])/run['run_id']
        receipt=read(out/'receipt.json')
        if receipt['status']!='COMPLETE':
            raise ValueError('Incomplete filter fit')
        for name,digest in receipt['outputs'].items():
            if sha(out/name)!=digest:
                raise ValueError('Changed filter artifact')
        pred=pd.read_parquet(out/'predictions.parquet')
        if not np.array_equal(pred.t_ns,f.t_ns) or not np.array_equal(pred.y_crossing,f.y_crossing):
            raise ValueError('Root clock/teacher mismatch')
        channels,side=score_channels(pred,seed)
        base=channels['cost2']['base']
        candidate=base>=c['experiment']['candidate_floor']
        book={**book0,'available':book0['available'] & candidate}
        roles={role:f.date.isin(days).to_numpy() for role,days in c['splits'].items()}
        cal_candidates=roles['calibration'] & book['available']
        safe=pred[[f'target_q_safe_{p}' for p in PLANS]].to_numpy(float)
        labels=pred[['target_'+q for q in QNAMES]].to_numpy(float)
        q=pred[QNAMES].to_numpy(float)
        diagnostics={}
        for role,mask in roles.items():
            known=mask & f.feature_available.to_numpy() & f.y_crossing.ge(0).to_numpy()
            use=mask & f.feature_available.to_numpy()
            diagnostics[role]=dict(available=int(use.sum()),known=int(known.sum()),unknown=int((use&~known).sum()),
                base=root_metrics(f.y_crossing.to_numpy()[known],pred[PNAMES].to_numpy()[known]),
                filter_heads={name:head_metrics(labels[use,j],q[use,j]) for j,name in enumerate(QNAMES)})
        grids=[]; selected={}; primary={}; none_budget=[]; exact_count=[]
        ledgerroot=out/'ledgers'; ledgerroot.mkdir(exist_ok=True)
        for arm in ARMS:
            armgrid=[]
            for j,plan in enumerate(PLANS):
                score=channels[plan][arm]
                for fraction in c['experiment']['retention_fractions']:
                    rule=retention_threshold(score,cal_candidates,fraction)
                    alerts=score>=rule['threshold']
                    entry=dict(arm=arm,plan=plan,fraction=fraction,threshold_rule=rule)
                    for role in ('calibration','test'):
                        entry[role+'_signal']=alarm_metrics(f,side,safe[:,j],candidate,alerts,roles[role])
                        econ,ledger=replay(book,f,score,side,plan,rule['threshold'],c['splits'][role],c['execution']['pending_timeout_seconds'])
                        entry[role+'_economics']=econ
                        if plan==c['experiment']['primary_plan'] and fraction==c['experiment']['primary_retention']:
                            filename=f'{arm}_{role}_primary.json'
                            save(ledgerroot/filename,ledger)
                            entry[role+'_ledger']=str(ledgerroot/filename)
                    if plan==c['experiment']['primary_plan'] and fraction==c['experiment']['primary_retention']:
                        primary[arm]=entry
                    armgrid.append(entry); grids.append(entry)
                none=roles['calibration'] & f.feature_available.to_numpy() & f.y_crossing.eq(0).to_numpy()
                gated=np.where(candidate,score,-1.)
                rule=empirical_none_threshold(gated,none,c['experiment']['calibration_none_fpr'])
                alerts=gated>=rule['threshold']
                none_budget.append(dict(arm=arm,plan=plan,rule=rule,
                    calibration=alarm_metrics(f,side,safe[:,j],candidate,alerts,roles['calibration']),
                    test=alarm_metrics(f,side,safe[:,j],candidate,alerts,roles['test'])))
            if arm=='hash_control':
                continue
            eligible=[r for r in armgrid if r['calibration_economics']['completed']
                      and r['calibration_economics']['trades']>=c['experiment']['minimum_calibration_trades']
                      and r['calibration_economics']['total_net_if_complete']>0]
            if eligible:
                chosen=sorted(eligible,key=lambda r:(-r['calibration_economics']['total_net_if_complete'],
                    -r['threshold_rule']['threshold'],float(r['plan'][4:])))[0]
                selected[arm]=dict(reason='positive_complete_CAL',plan=chosen['plan'],fraction=chosen['fraction'],
                    threshold=chosen['threshold_rule']['threshold'],calibration=chosen['calibration_economics'],test=chosen['test_economics'])
            else:
                selected[arm]=dict(reason='WAIT_no_positive_complete_CAL_with_50_trades',plan='cost2',fraction=0.,threshold=1.01)
                for role in ('calibration','test'):
                    selected[arm][role],_=replay(book,f,channels['cost2'][arm],side,'cost2',1.01,c['splits'][role],2.)
        # Exact TEST root-count comparison uses future score ranks; never replay
        # this diagnostic, and never use it to choose a threshold or checkpoint.
        candidates=np.flatnonzero(roles['test'] & book['available'])
        reference_count=primary['base']['test_signal']['total_causal_alerts']
        for arm in ARMS:
            score=channels['cost2'][arm]
            order=np.lexsort((f.t_ns.to_numpy()[candidates],-score[candidates]))
            selected_ids=candidates[order[:reference_count]]
            alerts=np.zeros(len(f),bool); alerts[selected_ids]=True
            exact_count.append(dict(arm=arm,root_count=reference_count,
                role='DESCRIPTIVE_NONCAUSAL_TEST_RANKING_NOT_A_POLICY',
                metrics=alarm_metrics(f,side,safe[:,0],candidate,alerts,roles['test'])))
        legacy=[]
        for plan in PLANS:
            for threshold in (.35,.5):
                econ,_=replay(book,f,base,side,plan,threshold,c['splits']['test'],2.)
                legacy.append(dict(plan=plan,threshold=threshold,test=econ))
        item=dict(seed=seed,diagnostics=diagnostics,selected=selected,primary=primary,grid=grids,
            calibrated_none_budget=none_budget,exact_test_count_diagnostic=exact_count,legacy_base_stress=legacy)
        results.append(item)
    report_data=dict(status='COMPLETE',config_sha256=sha(config_path),suite_sha256=sha(report/'suite_summary.json'),
        dataset_manifest_sha256=sha(root/'manifest.json'),runs=results,
        semantics=dict(base_frozen=True,original_target_loss_unchanged=True,
            primary='cost2, retain 50% of CAL causal base candidates; frozen numeric threshold on TEST',
            economics='USD/reference ounce, quote taker + 0.3 fee and 100ms latency; one position; censor halts account',
            none_wrong='Geometric NONE/wrong-side labels, not synonymous with losing execution',
            risk='Path safety before first favorable crossing; no claim of positive exit net',
            matching='CAL root counts matched conservatively; TEST counts can differ; hash thinning is not matched account state',
            exact_count='TEST exact-count roots are retrospective diagnostics, never executable thresholds or economic replay'))
    save(report/'evaluation.json',report_data)
    print(json.dumps(dict(status='COMPLETE',selected={str(r['seed']):{a:v['reason'] for a,v in r['selected'].items()} for r in results})),flush=True)
    return report_data


if __name__=='__main__':
    evaluate()
