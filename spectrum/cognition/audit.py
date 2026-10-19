"""Audit numerical counterfactuals, learned response and meaningful controls."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import balanced_accuracy_score
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.playground.data import MonthDay, QUOTE_CACHE
from .execution import ExecutionReplay
from .dataset import features
from .learn import load_dataset, load_predictor, sample_weights, metrics
from .evaluate import bootstrap_days


def audit(data='data/derived/hold_close_v1',output='reports/hold_close_v1'):
    data=Path(data);out=Path(output)
    protocol=json.loads((data/'protocol.json').read_text(encoding='utf8'))
    for path,digest in protocol['code'].items():
        if sha(Path(path))!=digest:raise ValueError('Generation code changed: '+path)
    x,f,names,md=load_dataset(data);predictor,checkpoint=load_predictor(out/'full_61021.pt')
    tr=f.target_known.to_numpy()&(f.split.to_numpy()=='train');te=f.target_known.to_numpy()&(f.split.to_numpy()=='test')
    weights=sample_weights(f.loc[te]);y=f.loc[te,'target_advantage'].to_numpy(float)
    p=predictor(x[te]);base=metrics(y,p,weights)
    rng=np.random.default_rng(61021);permutations=[]
    # Swap whole feature families within same-day same side/plan/age buckets.
    # Association stress only: not a causal intervention in a coherent market.
    ft=f.loc[te].reset_index(drop=True)
    groups=list(ft.groupby(['date','side','plan_id','age_seconds']).indices.values())
    for label,cols in [('market_context',np.arange(md)),('position_account',np.arange(md,len(names)))]:
        xx=x[te].copy()
        for ix in groups:
            perm=rng.permutation(ix);xx[np.ix_(ix,cols)]=xx[np.ix_(perm,cols)]
        pp=predictor(xx);m=metrics(y,pp,weights)
        permutations.append(dict(family=label,mse_increase=m['weighted_mse']-base['weighted_mse'],
            decision_disagreement=float(np.average((pp<0)!=(p<0),weights=weights)),**m))
    slices=[]
    sample=ft.copy();sample['prediction']=p;sample['close']=p<0;sample['weight']=weights
    pnl=x[te,names.index('unrealized_usd')]
    sample['pnl_bucket']=pd.cut(pnl,[-np.inf,-1,-.25,.25,1,np.inf]).astype(str)
    for grouping in ('side','plan_id','age_seconds','pnl_bucket'):
        for value,g in sample.groupby(grouping,observed=True):
            w=g.weight.to_numpy();yy=g.target_advantage.to_numpy();pp=g.prediction.to_numpy()
            slices.append(dict(group=grouping,value=str(value),rows=len(g),roots=g.root_id.nunique(),
                mean_prediction=float(np.average(pp,weights=w)),mean_actual=float(np.average(yy,weights=w)),
                close_fraction=float(np.average(pp<0,weights=w)),
                mean_gain_vs_hold=float(np.average(np.where(pp<0,-yy,0),weights=w))))
    tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=200,random_state=61021)
    train_p=predictor(x[tr]);tree.fit(x[tr],train_p<0,sample_weight=sample_weights(f.loc[tr]))
    tc=tree.predict(x[te]);actual=p<0
    surrogate=dict(description='Depth3 train-only imitation of network decisions; description of outputs, not a discovered market law.',
        fidelity=float(np.average(tc==actual,weights=weights)),always_close_fidelity=float(np.average(actual,weights=weights)),
        balanced_accuracy=float(balanced_accuracy_score(actual,tc,sample_weight=weights)),
        rule_text=export_text(tree,feature_names=names,decimals=3))
    # Exact saved-state replay on 24 deterministic rows spanning all split dates.
    candidates=np.flatnonzero(f.target_known.to_numpy())
    selected=candidates[np.linspace(0,len(candidates)-1,24,dtype=int)]
    reconciliation=[]
    for idx in selected:
        row=f.iloc[idx];month=row.date[:7];day=MonthDay(Path('data/derived/basket_multimonth_v1')/month/row.date)
        s=json.loads(row.snapshot_json);r=ExecutionReplay(day).restore(s)
        observed,_=features(day,r)
        np.testing.assert_allclose(observed,x[idx],rtol=0,atol=0)
        h=r.clone();h.reference_exit(int(row.horizon_ns));c=r.clone();c.request_close(int(row.decision_ns));c.reference_exit(int(row.horizon_ns))
        if h.state.censored or c.state.censored:raise ValueError('Known sample became censored')
        delta=h.state.cash-c.state.cash
        if abs(delta-row.target_advantage)>1e-8:raise ValueError('Paired target replay mismatch')
        reconciliation.append(dict(root_id=row.root_id,decision_ns=str(int(row.decision_ns)),advantage=float(delta),pass_=True))
        QUOTE_CACHE.clear()
    sequential=pd.read_parquet(out/'sequential_options.parquet');primary=sequential.loc[sequential.controller=='full_61021']
    comparisons=[]
    for controller in ('fixed_hold','close_first','age60','account_only_61021','permuted_targets_61021'):
        other=sequential.loc[sequential.controller==controller,['root_id','net','censored']].rename(columns={'net':'other_net','censored':'other_censored'})
        paired=primary.merge(other,on='root_id',validate='one_to_one')
        good=paired.loc[~paired.censored&~paired.other_censored&paired.net.notna()&paired.other_net.notna()].copy()
        good['delta']=good.net-good.other_net
        comparisons.append(dict(primary='full_61021',control=controller,roots=len(good),root_mean_delta=float(good.delta.mean()),
            interval=bootstrap_days(good,'delta'),interpretation='Same-origin paired cash difference, equal observed-day block estimate; all options reset1500.'))
    save_json(out/'sequential_comparisons.json',comparisons)
    constant=[names[i] for i in range(md,len(names)) if np.ptp(x[tr,i])<1e-8]
    risk_coverage=dict(constant_account_features=constant,
        initial_capital=1500,initial_oz=1,series_debt_scope='Only current fresh root; no prior losing trades. Risk-rule mechanics tested, debt adaptation not learned.',
        supported_learning='Position/plan-conditioned stopping under deterministic externally assigned entries; no entry selection, ADD/REDUCE or episode retrieval learned.')
    summary=dict(status='PASS_TECHNICAL_NOT_PROFIT_ADMISSION',source_sha256=sha(Path(__file__)),
        target_reconciliation=reconciliation,code_identity='PASS',paired_test_primary=base,
        family_permutations=permutations,behavior_slices=slices,surrogate=surrogate,risk_coverage=risk_coverage,
        limitations=['Permutation changes correlations and is not proof of economic causality.',
                     'Realized branch maximum is hindsight; not a feasible oracle target.',
                     'All previously explored dates are development history.',
                     'Mechanical protections immediate, policy exits delayed100ms; live execution robustness not established.',
                     'No independent samples claimed from multiple ages/sides/plans.'],
        counts=dict(rows=len(f),known=int(f.target_known.sum()),unknown=int((~f.target_known).sum()),
            dates=f.date.nunique(),feature_dim=len(names),market_dim=md,position_dim=len(names)-md))
    save_json(out/'behavior_audit.json',summary)
    print(json.dumps(dict(status=summary['status'],replayed_samples=len(reconciliation),comparisons=comparisons,permutations=permutations),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/derived/hold_close_v1');p.add_argument('--output',default='reports/hold_close_v1')
    a=p.parse_args();audit(a.data,a.output)
