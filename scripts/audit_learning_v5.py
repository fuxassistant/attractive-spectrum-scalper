"""Independent saved-fit audit: no V5 learner, target builder or runner imports.

Inference-only functional Torch replay uses saved weights, not training code.
All decisions and reported aggregates require a COMPLETE two-seed suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


HEADS = ('event', 'chosen_correct_given_event', 'safe_cost2_given_correct',
         'safe_cost4_given_correct', 'safe_cost6_given_correct')
QNAMES = ('q_event', 'q_correct', 'q_safe_cost2', 'q_safe_cost4', 'q_safe_cost6')
PNAMES = ('p_none', 'p_long', 'p_short', 'p_both')
LABELS = ('y_event', 'y_side_correct', 'y_risk_safe_cost2', 'y_risk_safe_cost4', 'y_risk_safe_cost6')
PLANS = ('cost2', 'cost4', 'cost6')
COEFFICIENTS = np.array([1., 1., 1./3, 1./3, 1./3])


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(array, dtype='<i8'):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def close(actual, expected, name, atol=1e-10, rtol=0.):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), name)
    return float(np.nanmax(np.abs(a-b))) if a.size else 0.


def sigmoid(z):
    return np.exp(-np.logaddexp(0., -np.asarray(z, np.float64)))


def verify_hierarchy(y):
    y = np.asarray(y)
    require(y.ndim == 2 and y.shape[1] == 5 and np.isin(y, [-1,0,1]).all(), 'Malformed conditional labels')
    require(not ((y[:,1] >= 0) & (y[:,0] != 1)).any(), 'C label outside E-positive support')
    require(not ((y[:,2:] >= 0) & ((y[:,0] != 1) | (y[:,1] != 1))[:,None]).any(), 'Risk outside correct-side support')


def head_metrics(y, p, z=None):
    verify_hierarchy(y)
    y, p = np.asarray(y), np.asarray(p, np.float64)
    require(p.shape == y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'Invalid probabilities')
    if z is None:
        pc = np.clip(p, 1e-12, 1-1e-12)
        loss = -np.maximum(y,0)*np.log(pc) - (1-np.maximum(y,0))*np.log1p(-pc)
    else:
        z = np.asarray(z, np.float64)
        loss = np.logaddexp(0.,z)-np.maximum(y,0)*z
    known = y >= 0
    n = int(known[:,0].sum())
    out = dict(n=n, composite_nll=float((loss*known*COEFFICIENTS).sum()/n) if n else None, heads={})
    for j, name in enumerate(HEADS):
        mask = known[:,j]
        yy, pp = y[mask,j], p[mask,j]
        out['heads'][name] = dict(n=int(mask.sum()), positive=int(yy.sum()), negative=int(len(yy)-yy.sum()),
            nll=float(loss[mask,j].mean()) if len(yy) else None,
            brier=float(np.square(pp-yy).mean()) if len(yy) else None,
            auc=float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None)
    return out


def compare_metrics(actual, expected, name):
    require(actual['n'] == expected['n'], name+' count')
    close(actual['composite_nll'], expected['composite_nll'], name+' composite NLL')
    for head in HEADS:
        for key, value in expected['heads'][head].items():
            if value is None:
                require(actual['heads'][head][key] is None, name+' empty '+head+'/'+key)
            else:
                close(actual['heads'][head][key], value, name+' '+head+'/'+key)


def verify_filter_gradients(prediction):
    y = prediction[['y_'+h for h in HEADS]].to_numpy()
    p = prediction[['p_'+h for h in HEADS]].to_numpy()
    z = prediction[['z_'+h for h in HEADS]].to_numpy()
    g = prediction[['g_'+h for h in HEADS]].to_numpy()
    verify_hierarchy(y)
    require(np.isfinite(z).all() and np.isfinite(g).all(), 'Nonfinite saved logits/gradient')
    pe = close(p, sigmoid(z), 'Saved sigmoid vs logits', atol=1e-14)
    expected = (p-np.maximum(y,0))*(y>=0)*COEFFICIENTS
    ge = close(g, expected, 'Actual BCE gradient vs independent formula', atol=1e-14)
    require(np.all(g[y<0] == 0), 'Unknown/masked head has nonzero gradient')
    return dict(sigmoid_max_error=pe, logit_gradient_max_error=ge, unknown_gradient_exact_zero=True), y, p, z


def verify_selection(history, training, config, train_ids, seed):
    tr = config['filter']
    require(history[0]['update']==0, 'Initial checkpoint candidate missing')
    best, selected, stale = history[0]['validation_nll'], 0, 0
    for row in history[1:]:
        require(stale < tr['patience_checks'], 'Training continued after early-stop decision')
        if row['validation_nll'] < best-tr['min_delta']:
            best, selected, stale = row['validation_nll'], row['update'], 0
        else:
            stale += 1
    require(selected == training['best_update'], 'Wrong validation-only selected checkpoint')
    close(best, training['best_validation_nll'], 'Selected VAL score')
    require(history[-1]['update']==training['updates_completed'], 'Update count mismatch')
    require(history[-1]['examples_seen']==training['examples_seen'], 'Examples-seen mismatch')
    require(history[-1]['update']==tr['max_updates'] or stale==tr['patience_checks'], 'Unexplained early training stop')
    expected_checks = [0]+list(range(tr['validate_every'], history[-1]['update']+1, tr['validate_every']))
    if history[-1]['update'] not in expected_checks:
        expected_checks.append(history[-1]['update'])
    require([r['update'] for r in history] == expected_checks, 'Validation schedule mismatch')
    rng = np.random.default_rng(seed)
    order, offset, examples = rng.permutation(train_ids), 0, 0
    h = hashlib.sha256()
    by_update = {r['update']:r for r in history}
    for update in range(1, training['updates_completed']+1):
        if offset == len(order):
            order, offset = rng.permutation(train_ids), 0
        ids = order[offset:offset+tr['batch_size']]
        offset += len(ids)
        examples += len(ids)
        h.update(np.asarray(ids,dtype='<i8').tobytes())
        if update in by_update:
            require(by_update[update]['sampled_root_stream_sha256']==h.hexdigest(), 'Uniform root stream mismatch')
            require(by_update[update]['examples_seen']==examples, 'Sampled-root accounting mismatch')
    require(training['sampled_root_stream_sha256']==h.hexdigest(), 'Final root stream SHA mismatch')
    return dict(selected_update=selected, best_validation_nll=best, uniform_stream_verified=True,
                updates=training['updates_completed'], checks=len(history)-1, min_delta=tr['min_delta'])


def expected_feature_names(micro):
    return (['current_value__'+v for v in micro]+['current_known__'+v for v in micro]
            +['current_supported_clock']+['frozen_hidden_'+str(i) for i in range(32)]
            +list(PNAMES)+['chosen_side','R_cost2_over_G','R_cost4_over_G','R_cost6_over_G','cost_over_G'])


def verify_feature_schema(schema, micro):
    names = expected_feature_names(micro)
    require(schema['names']==names and schema['features']==len(names)==182, 'Exact causal feature whitelist mismatch')
    require(schema['base_frozen'] is True and schema['long_day_context'] is False, 'Incorrect feature scope claim')
    return dict(features=len(names), exact_allowlist=True, future_outcome_id_columns=0,
                hidden_representation=32, raw_micro_features=len(micro), long_day_context=False)


def segment_starts(frame):
    t = frame.t_ns.to_numpy(np.int64)
    dates = frame.date.to_numpy()
    support = frame.physical_support_available.to_numpy(bool)
    token = frame.physical_segment_token.to_numpy(np.int64)
    reset = np.ones(len(frame),bool)
    reset[1:] = ((t[1:]-t[:-1] != 5_000_000_000) | (dates[1:]!=dates[:-1]) |
                 (token[1:]!=token[:-1]) | ~support[1:] | ~support[:-1])
    return np.maximum.accumulate(np.where(reset,np.arange(len(frame)),0))


def restored_batch(frame, raw_micro, scaler, micro_columns, ids, starts=None, length=64):
    """Independent dense-clock reconstruction; no future or teacher field reads."""
    ids = np.asarray(ids,np.int64)
    starts = segment_starts(frame) if starts is None else starts
    positions = ids[:,None]-np.arange(length-1,-1,-1)
    safe = np.maximum(positions,0)
    support = frame.physical_support_available.to_numpy(bool)
    clock = ((positions>=starts[ids,None]) & (positions>=0) & support[safe] & support[ids,None])
    values = raw_micro[safe]
    observed = np.isfinite(values) & frame.feature_available.to_numpy(bool)[safe,None] & clock[...,None]
    mean = np.asarray(scaler['mean'],np.float32)[micro_columns]
    std = np.asarray(scaler['scale'],np.float32)[micro_columns]
    v = np.where(observed,(values-mean)/std,0.)
    v = np.clip(v,-scaler['cap'],scaler['cap']).astype(np.float32)
    return np.concatenate([v,observed.astype(np.float32),clock[...,None].astype(np.float32)],-1)


def tcn_forward(batch, state, model_config):
    """Functional frozen TCN implementation independent of build_model."""
    h = F.linear(batch,state['projection.weight'],state['projection.bias']).transpose(1,2)
    for layer in range(model_config['layers']):
        residual = h
        dilation = 2**layer
        for convolution in (0,3):
            key = f'encoder.{layer}.path.{convolution}.conv.'
            weight, bias = state[key+'weight'], state[key+'bias']
            h = F.gelu(F.conv1d(F.pad(h,((weight.shape[2]-1)*dilation,0)),weight,bias,dilation=dilation))
        h = residual+h
    hidden = h[:,:,-1]
    z = F.linear(hidden,state['output.weight'],state['output.bias'])
    event = torch.sigmoid(z[:,:1])
    conditional = torch.softmax(z[:,1:],dim=1)
    p = torch.cat((1-event,event*conditional),dim=1)
    return hidden, z, p


def filter_forward(x, scaler, state, layers):
    mean, std = np.asarray(scaler['mean']), np.asarray(scaler['std'])
    h = np.clip(np.nan_to_num((x.astype(np.float64)-mean)/std,nan=0.),-10,10).astype(np.float32)
    h = torch.as_tensor(h,device=next(iter(state.values())).device)
    for layer in range(layers):
        key = f'encoder.{3*layer}.'
        h = F.gelu(F.linear(h,state[key+'weight'],state[key+'bias']))
    return F.linear(h,state['head.weight'],state['head.bias'])


@torch.no_grad()
def reconstruct_features(frame, raw_micro, feature_names, micro, base, device):
    checkpoint = torch.load(Path(base['root'])/'checkpoint.pt',map_location=device,weights_only=True)
    scaler = read(Path(base['root'])/'scaler.json')
    require(scaler['feature_names']==feature_names, 'Base scaler schema mismatch')
    columns = [feature_names.index(name) for name in micro]
    starts = segment_starts(frame)
    x, base_z, base_p = [], [], []
    for start in range(0,len(frame),512):
        ids = np.arange(start,min(start+512,len(frame)))
        batch = restored_batch(frame,raw_micro,scaler,columns,ids,starts)
        h,z,p = tcn_forward(torch.as_tensor(batch,device=device),checkpoint['state_dict'],checkpoint['model_config'])
        h,z,p = h.cpu().numpy(),z.cpu().numpy(),p.cpu().numpy()
        side = np.where(p[:,2]+p[:,3]>p[:,1]+p[:,3],-1,1)
        f = frame.iloc[ids]
        causal = np.column_stack([f['R_'+plan].to_numpy() for plan in PLANS]+[f.decision_spread.to_numpy()+.3])
        gain = f.G_event.to_numpy()
        ratios = np.divide(causal,gain[:,None],out=np.zeros_like(causal),where=np.isfinite(gain[:,None])&(gain[:,None]>0))
        ratios = np.where(np.isfinite(ratios),ratios,0).astype(np.float32)
        x.append(np.column_stack([batch[:,-1],h,p,side,ratios]).astype(np.float32))
        base_z.append(z); base_p.append(p)
    return np.concatenate(x),np.concatenate(base_z),np.concatenate(base_p)


def independently_reconstruct_targets(frame, mapped, side, duration=300):
    """Verify the chosen earliest first-line witness and strict prefix risk."""
    n = len(frame)
    m = mapped.sort_values(['t_ns','scale','role']).reset_index(drop=True)
    require(len(m)==9*n and np.array_equal(m.t_ns,np.repeat(frame.t_ns,9)), 'Map/root identity mismatch')
    require(np.array_equal(m.scale,np.tile(np.repeat([1,3,6],3),n)) and
            np.array_equal(m.role,np.tile([0,1,2],3*n)), 'Invalid nine-line map geometry')
    def a(name):
        return m[name].to_numpy().reshape(n,9)
    due = frame.entry_ns.to_numpy()+int(duration*1e9)
    observed = a('ready').astype(bool)&(a('cross_ordinal')>=0)&(a('cross_ns')<=due[:,None])
    positive = []
    for suffix in ('L','S'):
        positive.append((observed & (a('mark_'+suffix)>=frame.G_event.to_numpy()[:,None])).any(1))
    known = frame.y_crossing.to_numpy()>=0
    y = frame.y_crossing.to_numpy()
    close((positive[0].astype(int)+2*positive[1])[known],y[known],'Original known crossing labels from factual bank',atol=0)
    bit = np.where(side==1,1,2)
    correct = known & ((y&bit)!=0)
    marks = np.where(side[:,None]==1,a('mark_L'),a('mark_S'))
    mae = np.where(side[:,None]==1,a('mae_L'),a('mae_S'))
    qualifying = observed & np.isfinite(marks) & (marks>=frame.G_event.to_numpy()[:,None])
    index = np.argmin(np.where(qualifying,a('cross_ordinal'),np.iinfo(np.int64).max),axis=1)
    rows = np.arange(n)
    found = qualifying.any(1)
    witness_ns = np.where(found,a('cross_ns')[rows,index],0)
    witness_ordinal = np.where(found,a('cross_ordinal')[rows,index],-1)
    witness_id = np.where(found,a('event_id')[rows,index],'')
    prefix_mae = np.where(found,mae[rows,index],np.nan)
    risk_known = correct & (witness_ns<due)
    labels = np.full((n,5),-1,np.float32)
    labels[known,0] = y[known]>0
    e = known & (y>0)
    labels[e,1] = correct[e]
    for j,plan in enumerate(PLANS):
        labels[risk_known,j+2] = prefix_mae[risk_known]>-frame['R_'+plan].to_numpy()[risk_known]
    verify_hierarchy(labels)
    require(not (np.diff(labels[risk_known,2:],axis=1)<0).any(), 'Nonmonotonic risk teacher')
    for suffix,value in [('ns',witness_ns),('ordinal',witness_ordinal),('event_id',witness_id)]:
        original = np.where(side==1,frame['first_qualifying_'+suffix+'_L'],frame['first_qualifying_'+suffix+'_S'])
        require(np.array_equal(original[correct],value[correct]), 'Earliest witness '+suffix+' mismatch')
    return labels,dict(witness_ns=witness_ns,witness_ordinal=witness_ordinal,witness_id=witness_id,
        prefix_mae=prefix_mae,risk_known=risk_known,at_deadline=correct&(witness_ns==due))


def verify_receipt(directory, base):
    directory = Path(directory)
    receipt = read(directory/'receipt.json')
    require(receipt['status']=='COMPLETE', 'Run is not COMPLETE')
    require(receipt['base']==base, 'Run base differs from prespecified frozen base')
    for path,digest in receipt['sources'].items():
        require(sha(path)==digest,'Bound source changed: '+path)
    for name,digest in receipt['outputs'].items():
        require(sha(directory/name)==digest,'Run output changed: '+name)
    for name,digest in base['files'].items():
        require(sha(Path(base['root'])/name)==digest,'Frozen base changed: '+name)
    return dict(bound_sources=len(receipt['sources']),run_outputs=len(receipt['outputs']),base_files=len(base['files']))


def verify_scaler(x, ids, scaler, atol=3e-6):
    values = np.asarray(x[ids],np.float64)
    finite = np.isfinite(values)
    count = finite.sum(0)
    mean = np.where(finite,values,0).sum(0)/np.maximum(count,1)
    variance = np.where(finite,np.square(values-mean),0).sum(0)/np.maximum(count,1)
    std = np.sqrt(variance)
    std[(std<1e-8)|(count==0)] = 1
    require(scaler['fitted_rows']==len(ids) and scaler['fitted_root_indices_sha256']==array_sha(ids),'Scaler TRAIN rows mismatch')
    close(scaler['finite_count'],count,'Scaler finite observation counts',atol=0)
    return dict(mean_max_error=close(scaler['mean'],mean,'TRAIN-only scaler mean',atol=atol,rtol=1e-6),
                std_max_error=close(scaler['std'],std,'TRAIN-only scaler std',atol=atol,rtol=1e-6),
                train_rows=len(ids),fitted_root_indices_sha256=array_sha(ids))


def quantiles(values):
    a = np.asarray(values,np.float64)
    levels = (0.,.01,.1,.25,.5,.75,.9,.95,.99,1.)
    return dict(n=len(a),quantiles={str(level):float(np.quantile(a,level)) for level in levels} if len(a) else {})


def candidate_diagnostics(labels, base_p, filter_p, filter_z, split, floor=.35):
    """Fixed prespecified candidate cohort, never fit a threshold on TEST."""
    bp = np.asarray(base_p,np.float64)
    event, side = 1-bp[:,0],np.maximum(bp[:,1]+bp[:,3],bp[:,2]+bp[:,3])
    answer = {}
    for role,ids in split.items():
        candidates = ids[side[ids]>=floor]
        known = candidates[labels[candidates,0]>=0]
        positive = labels[ids,0]==1
        answer[role] = dict(admitted_roots=len(ids),candidate_roots=len(candidates),
            candidate_fraction=float(len(candidates)/len(ids)) if len(ids) else None,
            known_candidates=len(known),unknown_candidates=len(candidates)-len(known),
            known_candidate_NONE=int((labels[known,0]==0).sum()),
            known_candidate_event=int((labels[known,0]==1).sum()),
            known_candidate_correct_side=int((labels[known,1]==1).sum()),
            all_admitted_known_event=int(positive.sum()),
            event_coverage=float((labels[known,0]==1).sum()/positive.sum()) if positive.any() else None,
            candidate_root_indices_sha256=array_sha(candidates),
            base_event_score=quantiles(event[ids]),base_selected_side_score=quantiles(side[ids]),
            candidate_head_metrics=head_metrics(labels[known],filter_p[known],filter_z[known]))
    return dict(candidate_floor=floor,rule='float64 max(p_long+p_both,p_short+p_both)>=floor on all feature-admitted roots',
        comparison='Candidate metrics are separate from global metrics. Conditional head denominators remain conditional; unknown is excluded from scores, retained in coverage.',
        by_role=answer)


def clipping_diagnostics(raw_micro,x,split,base_scaler,micro_columns,filter_scaler,feature_names):
    """Pre-cap numerical saturation, feature-only and independent of labels.

    Base observations exclude missing cells. Filter nonconstant channels are
    identified on ALL admitted meta-TRAIN rows, including unknown teachers.
    """
    base_mean=np.asarray(base_scaler['mean'],np.float32)[micro_columns]
    base_std=np.asarray(base_scaler['scale'],np.float32)[micro_columns]
    base_z=(raw_micro-base_mean)/base_std
    base_variable=np.array([i not in base_scaler.get('constant_columns',[]) and
                            i not in base_scaler.get('all_missing_columns',[]) for i in micro_columns])
    filter_mean,filter_std=np.asarray(filter_scaler['mean']),np.asarray(filter_scaler['std'])
    filter_z=(x.astype(np.float64)-filter_mean)/filter_std
    values=x[split['train']].astype(np.float64)
    actual_std=np.nanstd(values,axis=0)
    filter_variable=np.isfinite(actual_std)&(actual_std>=1e-8)
    groups={'current_values':np.arange(70),'current_masks':np.arange(70,140),'supported_clock':np.array([140]),
            'frozen_hidden':np.arange(141,173),'base_probabilities':np.arange(173,177),'chosen_side':np.array([177]),
            'causal_ratios':np.arange(178,182)} if x.shape[1]==182 else {'all':np.arange(x.shape[1])}
    def counts(z,variable):
        finite=np.isfinite(z); clipped=(np.abs(z)>10)&finite
        used=finite&variable[None,:]
        denom=int(used.sum()); count=int((clipped&used).sum())
        return dict(rows=len(z),channels=int(variable.sum()),observed_cells=denom,clipped_cells=count,
            clipped_cell_fraction=count/denom if denom else None,
            rows_with_any_clipped_fraction=float((clipped&used).any(1).mean()) if len(z) else None)
    by_role={}
    for role,ids in split.items():
        z=filter_z[ids]
        by_role[role]=dict(base_observed_nonconstant=counts(base_z[ids],base_variable),
            filter_all=counts(z,np.ones(x.shape[1],bool)),filter_nonconstant=counts(z,filter_variable),
            filter_groups={group:counts(z[:,cols],filter_variable[cols]) for group,cols in groups.items()})
    test_ids=split.get('test',np.empty(0,np.int64))
    finite=np.isfinite(filter_z[test_ids]);clip=(np.abs(filter_z[test_ids])>10)&finite
    rates=np.divide(clip.sum(0),finite.sum(0),out=np.zeros(x.shape[1]),where=finite.sum(0)>0)
    top=np.argsort(-rates,kind='stable')[:15]
    return dict(threshold=10.,definition='Strict pre-cap abs(z)>10; excludes exact boundary values. Numerical saturation is not evidence of signal quality.',
        filter_constant_channels=int((~filter_variable).sum()),filter_nonconstant_channels=int(filter_variable.sum()),
        filter_constant_names=[feature_names[j] for j in np.flatnonzero(~filter_variable)],
        by_role=by_role,top_test_filter_clipping=[dict(name=feature_names[j],fraction=float(rates[j]),
            variable_on_meta_train=bool(filter_variable[j])) for j in top if rates[j]>0])


def risk_monotonicity(probabilities,labels,split):
    """Report unprojected sigmoid violations; this never modifies predictions."""
    q=np.asarray(probabilities,np.float64)
    deltas=q[:,2:4]-q[:,3:5]
    def measure(ids):
        d=deltas[ids]
        result=dict(roots=len(ids),any_strict_violation_fraction=float((d>0).any(1).mean()) if len(ids) else None,
            any_above_1e_6_fraction=float((d>1e-6).any(1).mean()) if len(ids) else None)
        for j,name in enumerate(('cost2_above_cost4','cost4_above_cost6')):
            positive=np.maximum(d[:,j],0)
            result[name]=dict(strict_violation_count=int((d[:,j]>0).sum()),
                fraction=float((d[:,j]>0).mean()) if len(ids) else None,
                maximum=float(positive.max()) if len(ids) else None,
                mean_positive_difference_over_all_roots=float(positive.mean()) if len(ids) else None)
        return result
    return dict(expected='q_safe_cost2 <= q_safe_cost4 <= q_safe_cost6; independent sigmoid heads impose no monotonic constraint',
        posthoc_projection_applied=False,by_role={role:dict(all_admitted=measure(ids),
            risk_known_conditional=measure(ids[(labels[ids,2:]>=0).all(1)])) for role,ids in split.items()})


def prefix_check(frame, raw_micro, feature_names, micro, base_scaler):
    """Observed values strictly after fixed cut cannot change an earlier window.

    Also compare every selected root with a prefix-truncated dataset and
    perturb teacher metadata, which must never enter a causal input window.
    """
    columns = [feature_names.index(n) for n in micro]
    cut = min(len(frame)-2,max(0,len(frame)//2))
    ids = np.unique([0,min(80,cut),cut])
    original = restored_batch(frame,raw_micro,base_scaler,columns,ids)
    changed = raw_micro.copy(); changed[cut+1:] = 999
    ff = frame.copy()
    for name in ('y_crossing','available_ns','first_qualifying_ns_L','first_qualifying_ns_S'):
        if name in ff:
            ff[name] = -777
    other = restored_batch(ff,changed,base_scaler,columns,ids)
    short = restored_batch(frame.iloc[:cut+1],raw_micro[:cut+1],base_scaler,columns,ids)
    require(np.array_equal(original,other) and np.array_equal(original,short),'Causal prefix/future-label invariance failed')
    return dict(probes=ids.tolist(),future_values_and_all_teacher_metadata_mutated=True,prefix_equal=True,
                scope='Saved feature/window boundary only; raw-source feature causality is checked by dataset builders')


def audit(config_path='configs/research/opportunity_pipeline_v5.json', report_root='reports/scientific_pipeline_v5',device='cpu'):
    config, report_root = read(config_path),Path(report_root)
    suite = read(report_root/'suite_summary.json')
    require(suite['status']=='COMPLETE' and len(suite['runs'])==2,'Only COMPLETE two-fit suite may be audited')
    require(suite['config_sha256']==sha(config_path),'Suite config changed')
    require(sorted(r['seed'] for r in suite['runs'])==[17,29],'Two fixed seeds required')
    root = Path(suite['dataset_root']); manifest = read(root/'manifest.json')
    require(manifest['status']=='COMPLETE' and sha(root/'manifest.json')==suite['dataset_manifest_sha256'],'Dataset incomplete or changed')
    dates = sum([config['splits'][role] for role in ('train','validation','calibration','test')],[])
    require(dates==sorted(set(dates))==config['data']['dates']==manifest['dates'],'Chronological split identity')
    require(manifest['splits']==config['splits'],'Dataset/experiment split mismatch')
    for name in ('event_query','execution','risk'):
        require(manifest['identity'][name]==config[name],'Dataset target/execution identity mismatch')
    frame_parts, raw_parts, map_parts = [],[],[]
    checked = 0
    names, micro = manifest['feature_names'],manifest['groups']['micro']
    columns = [names.index(name) for name in micro]
    for day in manifest['days']:
        require(day['status']=='COMPLETE','Partial day')
        for filename,digest in day['files'].items():
            require(sha(root/filename)==digest,'Dataset payload changed: '+filename); checked += 1
        f = pd.read_parquet(root/day['roots_file'])
        raw = np.load(root/day['x_file'])
        require(len(f)==len(raw) and (f.date==day['date']).all(),'Day/root X alignment')
        frame_parts.append(f); raw_parts.append(raw[:,columns]); map_parts.append(pd.read_parquet(root/(day['date']+'_map.parquet')))
    frame, raw, mapped = pd.concat(frame_parts,ignore_index=True),np.concatenate(raw_parts),pd.concat(map_parts,ignore_index=True)
    require((np.diff(frame.t_ns)>0).all(),'Nonchronological roots')
    admitted = frame.feature_available.to_numpy(bool)
    require((~admitted|frame.physical_support_available.to_numpy(bool)).all(),'Admitted X without physical support')
    require((frame.source_endpoint_ns[admitted]<=frame.source_available_ns[admitted]).all() and
            (frame.source_available_ns[admitted]<=frame.t_ns[admitted]).all(),'Future source publication in admitted X')
    split = {role:np.flatnonzero(admitted&frame.date.isin(ds).to_numpy()) for role,ds in config['splits'].items()}
    for role,next_role in (('train','validation'),('validation','calibration'),('calibration','test')):
        rows = frame.date.isin(config['splits'][role])&(frame.y_crossing>=0)
        require((frame.loc[rows,'available_ns']<frame.loc[frame.date.isin(config['splits'][next_role]),'t_ns'].min()).all(),
                'Teacher maturity crosses a split boundary')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    runs = []
    for run in suite['runs']:
        seed = run['seed']; directory = Path(suite['runs_root'])/run['run_id']
        base = config['base_models'][str(seed)]
        receipts = verify_receipt(directory,base)
        summary, history = read(directory/'summary.json'),read(directory/'history.json')
        require(summary==run['summary'],'Suite summary differs from saved run')
        require(read(directory/'config.json')==config,'Saved training config changed')
        require(summary['learner_source_sha256']==sha('spectrum/research_v5/learning.py'),'Learner source mismatch')
        base_c = read(Path(base['root'])/'config.json')
        require(max(base_c['splits']['validation'])<min(config['splits']['train']),'Meta TRAIN overlaps frozen base fit/selection')
        for key in ('event_query','execution','risk','model','loss'):
            require(config[key]==base_c[key],'Original base recipe changed: '+key)
        pframe, fp = pd.read_parquet(directory/'predictions.parquet'),pd.read_parquet(directory/'filter_predictions.parquet')
        target = pd.read_parquet(directory/'targets.parquet')
        require(len(frame)==len(pframe)==len(fp)==len(target),'Full clock rows lost')
        require(np.array_equal(pframe.root_index,np.arange(len(frame))) and np.array_equal(fp.root_index,pframe.root_index),'Root-index alignment')
        for col in ('date','t_ns','y_crossing','feature_available','physical_segment_token','physical_support_available'):
            require(np.array_equal(pframe[col],frame[col]),'Full clock metadata mismatch: '+col)
        for col in ('date','t_ns','y_crossing'):
            require(np.array_equal(target[col],frame[col]),'Target clock metadata mismatch: '+col)
        gcheck,y,q,z = verify_filter_gradients(fp)
        close(q,pframe[list(QNAMES)],'Runner lost filter probabilities',atol=0)
        close(y,pframe[['target_'+n for n in QNAMES]],'Runner lost filter masks',atol=0)
        close(y,target[list(LABELS)],'Target builder/filter label mismatch',atol=0)
        bp = pframe[list(PNAMES)].to_numpy(); side = np.where(bp[:,2]+bp[:,3]>bp[:,1]+bp[:,3],-1,1)
        require(np.array_equal(side,pframe.side) and np.array_equal(side,target.chosen_side),'Causal base side/tie handling mismatch')
        independent_y,witness = independently_reconstruct_targets(frame,mapped,side,config['event_query']['deadline_seconds'])
        close(y,independent_y,'Independent chosen-side and pre-cross risk labels',atol=0)
        for field,expected in [('witness_ns',witness['witness_ns']),('witness_ordinal',witness['witness_ordinal']),('witness_event_id',witness['witness_id'])]:
            require(np.array_equal(target[field],expected),'Saved witness mismatch: '+field)
        close(target.witness_prefix_mae_net,witness['prefix_mae'],'Saved pre-cross MAE',atol=1e-10)
        expected_available = np.where(witness['risk_known'],np.maximum(frame.available_ns,witness['witness_ns']),0)
        require(np.array_equal(target.risk_available_ns,expected_available),'Conditional risk maturity differs')
        receipt = read(directory/'target_receipt.json')
        require(receipt['known_per_head']==(y>=0).sum(0).tolist() and receipt['positives_per_head']==(y==1).sum(0).tolist(),'Target receipt counts')
        schema = verify_feature_schema(read(directory/'feature_schema.json'),micro)
        known_train = split['train'][y[split['train'],0]>=0]
        known_val = split['validation'][y[split['validation'],0]>=0]
        selection = verify_selection(history,summary['training'],config,known_train,seed)
        prior = np.array([(y[known_train,j][y[known_train,j]>=0].sum()+.5)/((y[known_train,j]>=0).sum()+1) for j in range(5)])
        close(summary['train_prior'],prior,'TRAIN-only smoothed conditional priors',atol=0)
        by_role = {}
        for role,ids in split.items():
            known_ids = ids[y[ids,0]>=0]
            model_metric = head_metrics(y[known_ids],q[known_ids],z[known_ids])
            prior_metric = head_metrics(y[known_ids],np.tile(prior,(len(known_ids),1)))
            by_role[role] = dict(admitted=len(ids),known_event=len(known_ids),unknown_event=len(ids)-len(known_ids),
                root_indices_sha256=array_sha(ids),model=model_metric,prior=prior_metric)
            if role in ('train','validation'):
                compare_metrics(summary['evaluation'][role]['model'],model_metric,role+' model')
                compare_metrics(summary['evaluation'][role]['prior'],prior_metric,role+' prior')
                cohort = summary['cohorts'][role]
                require(cohort['admitted_root_indices_sha256']==array_sha(ids) and cohort['known_root_indices_sha256']==array_sha(known_ids),'Saved cohort hash')
                require(cohort['known_per_head']==(y[known_ids]>=0).sum(0).tolist() and
                        cohort['positive_per_head']==(y[known_ids]==1).sum(0).tolist(),'Saved cohort head counts')
        close(by_role['validation']['model']['composite_nll'],selection['best_validation_nll'],'Saved checkpoint predictions vs selected validation score')
        print(json.dumps(dict(stage='INDEPENDENT_INFERENCE',seed=seed,device=device)),flush=True)
        x,bz,replayed_bp = reconstruct_features(frame,raw,names,micro,base,device)
        base_error = close(replayed_bp,bp,'Functional frozen base probabilities',atol=3e-6,rtol=2e-5)
        base_logit_error = close(bz,pframe[['base_logit_'+str(j) for j in range(4)]],'Functional frozen base logits',atol=2e-5,rtol=2e-5)
        filter_scaler = read(directory/'scaler.json')
        scaler_check = verify_scaler(x,split['train'],filter_scaler)
        require(summary['scaler_sha256']==sha(directory/'scaler.json'),'Saved filter scaler hash')
        checkpoint = torch.load(directory/'checkpoint.pt',map_location=device,weights_only=False)
        require(state_sha(checkpoint['state_dict'])==summary['training']['selected_state_sha256'],'Selected parameter hash')
        require(checkpoint['best_update']==selection['selected_update'] and checkpoint['scaler_sha256']==summary['scaler_sha256'],'Checkpoint identity')
        require(checkpoint['learner_source_sha256']==summary['learner_source_sha256'],'Checkpoint learner hash')
        filter_logits = []
        with torch.no_grad():
            for start in range(0,len(x),2048):
                filter_logits.append(filter_forward(x[start:start+2048],filter_scaler,checkpoint['state_dict'],config['filter']['layers']).cpu().numpy())
        filter_error = close(np.concatenate(filter_logits),z,'Functional selected filter logits',atol=3e-5,rtol=3e-5)
        base_scaler=read(Path(base['root'])/'scaler.json')
        prefix = prefix_check(frame,raw,names,micro,base_scaler)
        candidate=candidate_diagnostics(y,bp,q,z,split,config['experiment']['candidate_floor'])
        clipping=clipping_diagnostics(raw,x,split,base_scaler,columns,filter_scaler,expected_feature_names(micro))
        monotonicity=risk_monotonicity(q,y,split)
        # Priors/cohorts/probes must derive from admitted known TRAIN only.
        probe_counts = (('train_metric_probe',2048,410),('gradient_probe',256,411))
        for key,default,offset in probe_counts:
            expected = np.sort(np.random.default_rng(seed+offset).choice(known_train,min(config['filter'].get(key,default),len(known_train)),replace=False))
            require(array_sha(expected)==summary['training'][key+'_sha256'],'TRAIN probe sampling/hash mismatch')
        require(summary['training']['base_updates']==summary['base_updates']==0 and summary['base_frozen'],'Base mutation claim')
        receipts_after = verify_receipt(directory,base)
        runs.append(dict(seed=seed,run_id=run['run_id'],receipts=receipts_after,gradient=gcheck,
            feature_schema=schema,selection=selection,scaler=scaler_check,prefix=prefix,
            inference=dict(base_probability_max_error=base_error,base_logit_max_error=base_logit_error,filter_logit_max_error=filter_error),
            targets=dict(all_labels_independently_equal=True,known_per_head=(y>=0).sum(0).tolist(),
                positive_per_head=(y==1).sum(0).tolist(),exact_deadline_risk_unknown=int(witness['at_deadline'].sum())),
            candidate_diagnostics=candidate,clipping_diagnostics=clipping,risk_monotonicity=monotonicity,by_role=by_role))
        del x
    return dict(status='PASS',version='INDEPENDENT_V5_SAVED_FILTER_AUDIT_1',config_sha256=sha(config_path),
        suite_sha256=sha(report_root/'suite_summary.json'),dataset_manifest_sha256=sha(root/'manifest.json'),
        auditor_source_sha256=sha(__file__),dataset_files_verified=checked,roots=len(frame),runs=runs,
        checks=dict(frozen_base_unchanged=True,exact_gradient_and_hierarchy=True,full_clock_retained=True,
                    independent_target_labels=True,independent_functional_forward=True,train_only_scaler=True,
                    validation_only_checkpoint=True,train_only_priors=True,causal_feature_allowlist=True,
                    uniform_sampling_prefixes=True,teacher_maturity_precedes_next_split=True),
        limitations=['This is an independent implementation audit by the same agent that implemented the learner, not an independent-person review.',
                     'Inference replay uses already saved causal feature payloads and saved weights; raw MBO lineage is additionally checked by dataset/source audits.',
                     'Risk labels describe STOP before the favorable crossing, not delayed-exit profit. No economics conclusion or policy selection is made here.',
                     'Five TEST days and a GCQ6-to-GCZ6 transfer limit statistical generalization; these metrics are descriptive.',
                     'Checkpoint selection follows min_delta sequential early stopping; it need not equal the absolute smallest recorded NLL when differences are below min_delta.'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',default='configs/research/opportunity_pipeline_v5.json')
    parser.add_argument('--report-root',default='reports/scientific_pipeline_v5')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args = parser.parse_args()
    result = audit(args.config,args.report_root,args.device)
    path = Path(args.report_root)/'learning_audit.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=result['status'],roots=result['roots'],runs=len(result['runs']),output=str(path))),flush=True)


if __name__=='__main__':
    main()
