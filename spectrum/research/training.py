"""Common trainer, diagnostics and causal one-position account projection.

Model selection is on a validation day. Threshold selection uses a separate
calibration day. Test observations never affect preprocessing or selection.
The simulator's paired trials are labels, not simultaneous live positions.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from .losses import classification_loss, joint_probabilities
from .models import build_model
from .simulation import SimulationConfig


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def validate_config(c):
    """Fail closed on unsupported scientific switches instead of ignoring them."""
    if c.get('schema') != 'scientific_opportunity_pipeline_v1':
        raise ValueError('Unsupported configuration schema')
    if c['data']['cadence_seconds'] != 30:
        raise ValueError('Saved roots have 30-second cadence; changing cadence requires an explicit resampling adapter')
    if set(c['splits']) != {'train', 'validation', 'calibration', 'test'}:
        raise ValueError('Four explicit chronological split roles required')
    dates = [d for role in ('train', 'validation', 'calibration', 'test') for d in c['splits'][role]]
    if any(not c['splits'][k] for k in c['splits']) or dates != sorted(set(dates)) or set(dates) != set(c['data']['dates']):
        raise ValueError('Split dates must be nonempty, disjoint, chronological and exhaustive')
    tr, pol = c['training'], c['policy']
    for k in ('batch_size', 'max_epochs', 'patience', 'cpu_threads'):
        if not isinstance(tr[k], int) or isinstance(tr[k], bool) or tr[k] <= 0:
            raise ValueError('Positive training integer required: ' + k)
    for k in ('learning_rate', 'gradient_clip'):
        if not np.isfinite(tr[k]) or tr[k] <= 0:
            raise ValueError('Positive finite training scalar required: ' + k)
    if not np.isfinite([tr['weight_decay'], tr['min_delta']]).all() or tr['weight_decay'] < 0 or tr['min_delta'] < 0 or tr['weighting'] != 'uniform_roots':
        raise ValueError('Only uniform-root baseline weighting is implemented')
    if c['target']['risk_source'] != 'frozen_crossing_roots_causal_R':
        raise ValueError('Recompute causal risk in a separately versioned dataset')
    if not np.isfinite(c['target'].get('minimum_net_R', 0.)):
        raise ValueError('Finite minimum_net_R required')
    if not pol['one_position'] or pol['selection'] != 'calibration_after_cost_net_else_wait':
        raise ValueError('Unsupported account policy')
    if pol['temperature'] != 1.0:
        raise ValueError('This baseline uses uncalibrated T=1; fit calibration in a separate experiment')
    thresholds = pol['threshold_candidates']
    if not thresholds or any(not np.isfinite(t) or t < 0 for t in thresholds) or not any(t > 1 for t in thresholds):
        raise ValueError('Finite nonnegative thresholds including WAIT (>1) required')
    SimulationConfig.from_dict(c)
    probe = build_model(3, c['model'])
    classification_loss(probe(torch.zeros(2, 3, 3)), torch.tensor([0, 1]), c['loss'])


def set_seed(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_execution(config, frame):
    """Verify the completed cache and align it without deleting unknown roots.

    Raw quote hashes were directly checked by the cache builder. Here its
    receipt and output hashes are checked; that distinction is saved in runs.
    """
    directory = Path(config['data']['execution_dir'])
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'COMPLETE' or manifest['identity']['execution_config'] != asdict(SimulationConfig.from_dict(config)):
        raise ValueError('Execution cache missing or effective recipe mismatch')
    if manifest['identity']['dates'] != config['data']['dates']:
        raise ValueError('Execution cache date mismatch')
    for path, expected in manifest['identity']['code'].items():
        if sha(path) != expected:
            raise ValueError('Execution source code changed: ' + path)
    roots_dir = Path(config['data']['roots_dir'])
    receipts = {str(directory / 'manifest.json'): sha(directory / 'manifest.json')}
    frames = []
    for r in manifest['days']:
        path = directory / r['file']
        if sha(path) != r['sha256'] or sha(roots_dir / (r['date'] + '_roots.parquet')) != r['root_source']['root_sha256']:
            raise ValueError('Execution output or decision roots changed')
        if json.loads((directory / (r['date'] + '.json')).read_text(encoding='utf-8')) != r:
            raise ValueError('Execution daily receipt mismatch')
        receipts[str(path)] = r['sha256']
        frames.append(pd.read_parquet(path))
    paired = pd.concat(frames, ignore_index=True)
    if paired.duplicated(['date', 't_ns', 'side']).any():
        raise ValueError('Duplicate execution trial')
    complete_rows = paired.loc[paired.status == 'COMPLETE']
    if (not np.isfinite(complete_rows[['net', 'mae', 'mfe', 'risk_R']].to_numpy()).all()
            or not (complete_rows.risk_R > 0).all()
            or not (complete_rows.entry_ns > complete_rows.t_ns).all()
            or not (complete_rows.exit_ns > complete_rows.entry_ns).all()
            or not (complete_rows.mae <= complete_rows.net + 1e-9).all()
            or not (complete_rows.mfe >= complete_rows.net - 1e-9).all()):
        raise ValueError('Malformed COMPLETE execution outcome')
    sides = []
    for side, suffix in ((1, 'L'), (-1, 'S')):
        part = paired.loc[paired.side == side].drop(columns=['side'])
        aligned = frame[['date', 't_ns']].merge(part, on=['date', 't_ns'], how='left', validate='one_to_one', sort=False)
        if aligned.status.isna().any() or not np.array_equal(aligned.t_ns, frame.t_ns):
            raise ValueError('Execution root alignment mismatch')
        sides.append(aligned.rename(columns={k: k + '_' + suffix for k in aligned if k not in ('date', 't_ns')}))
    out = sides[0].merge(sides[1], on=['date', 't_ns'], validate='one_to_one', sort=False)
    complete = (out.status_L == 'COMPLETE') & (out.status_S == 'COMPLETE')
    floor = config['target'].get('minimum_net_R', 0.)
    y = ((out.net_L > floor * out.risk_R_L).astype(int) + 2 * (out.net_S > floor * out.risk_R_S).astype(int)).to_numpy(np.int64)
    y[~complete] = -1
    return out, y, dict(source_sha256=receipts, manifest=manifest,
                        raw_verification='Builder checked raw payload hashes; trainer checks saved output, receipt, root and code hashes')


def split_indices(frame, y, config):
    eligible = frame.feature_available.to_numpy(bool) & (y >= 0)
    return {role: np.flatnonzero(frame.date.isin(days).to_numpy() & eligible) for role, days in config['splits'].items()}


def metrics(y, probability):
    y, p = np.asarray(y), np.asarray(probability, float)
    known = y >= 0
    y, p = y[known], p[known]
    if not len(y):
        return {'n': 0}
    p = np.clip(p, 1e-12, 1.)
    e, ep = (y > 0).astype(float), 1 - p[:, 0]
    ep = np.clip(ep, 1e-12, 1 - 1e-12)
    event_nll = -(e * np.log(ep) + (1 - e) * np.log1p(-ep))
    positive = y > 0
    direction = float(-np.log(p[positive, y[positive]] / ep[positive]).mean()) if positive.any() else None
    bins = []
    for lo in np.arange(0, 1, .1):
        selected = (ep >= lo) & (ep < lo + .1)
        bins.append(dict(lower=float(lo), count=int(selected.sum()),
                         probability=float(ep[selected].mean()) if selected.any() else None,
                         frequency=float(e[selected].mean()) if selected.any() else None))
    ece = sum(b['count'] / len(y) * abs(b['probability'] - b['frequency']) for b in bins if b['count'])
    return dict(n=len(y), joint_nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
                accuracy=float((p.argmax(axis=1) == y).mean()), event_nll=float(event_nll.mean()),
                event_brier=float(np.square(ep - e).mean()), event_auc=float(roc_auc_score(e, ep)) if len(np.unique(e)) > 1 else None,
                event_ece=float(ece), direction_nll_given_event=direction,
                state_counts=np.bincount(y, minlength=4).tolist(), mean_joint_probability=p.mean(axis=0).tolist(),
                reliability=bins)


@torch.no_grad()
def predict(model, x, config, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    out = []
    for a in range(0, len(x), batch_size):
        logits = model(torch.as_tensor(x[a:a + batch_size], device=device))
        out.append(joint_probabilities(logits, config['loss'].get('formulation', 'factorized')).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 4), np.float32)


def gradient_diagnostic(model, x, y, config):
    """Unclipped event/direction gradient scale and shared encoder alignment."""
    model.eval()
    device = next(model.parameters()).device
    xx, yy = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    logits = model(xx)
    total = classification_loss(logits, yy, config['loss'])
    parameters = [(n, p) for n, p in model.named_parameters() if not n.startswith('output.')]
    params = [p for _, p in parameters]
    event = F.binary_cross_entropy_with_logits(logits[:, 0], (yy > 0).float())
    direction = (F.cross_entropy(logits[:, 1:], (yy - 1).clamp_min(0), reduction='none') * (yy > 0)).mean()
    def vector(loss):
        gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(params, gs)])
    ge, gd = vector(event), vector(direction)
    outgrad = torch.autograd.grad(total, logits)[0]
    en, dn = float(ge.norm()), float(gd.norm())
    return dict(event_bce_encoder_norm=en, direction_encoder_norm=dn,
                cosine_event_direction=float(torch.dot(ge, gd) / (ge.norm() * gd.norm()).clamp_min(1e-20)),
                output_gradient_abs_mean=outgrad.abs().mean(dim=0).detach().cpu().tolist(),
                loss=float(total.detach()), n=len(y), event_n=int((yy > 0).sum()),
                diagnostic_formulation='BCE event versus conditional CE, same root denominator; only meaningful for factorized output',
                all_finite=bool(torch.isfinite(ge).all() & torch.isfinite(gd).all() & torch.isfinite(outgrad).all()))


def fit_model(x_train, y_train, x_validation, y_validation, config, directory=None):
    tr = config['training']
    set_seed(config['seed'], tr.get('cpu_threads', 4))
    device = tr.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(x_train.shape[-1], config['model']).to(device)
    if not len(y_train) or not len(y_validation) or np.any(y_train < 0) or np.any(y_validation < 0):
        raise ValueError('Nonempty fully known train/validation cohorts required')
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr['learning_rate'], weight_decay=tr['weight_decay'])
    rng = np.random.default_rng(config['seed'])
    history, best, best_epoch, stale = [], np.inf, -1, 0
    best_state = None
    probe_ids = np.linspace(0, len(y_train) - 1, min(256, len(y_train)), dtype=int)
    before = gradient_diagnostic(model, x_train[probe_ids], y_train[probe_ids], config) if config['loss'].get('formulation') == 'factorized' else None
    started = time.perf_counter()
    for epoch in range(tr['max_epochs']):
        model.train()
        losses, norms, clipped = [], [], 0
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), tr['batch_size']):
            ids = order[start:start + tr['batch_size']]
            xb = torch.as_tensor(x_train[ids], device=device)
            yb = torch.as_tensor(y_train[ids], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = classification_loss(model(xb), yb, config['loss'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tr['gradient_clip'], error_if_nonfinite=True))
            optimizer.step()
            losses.append((float(loss.detach()), len(ids)))
            norms.append(norm)
            clipped += norm > tr['gradient_clip']
        pv = predict(model, x_validation, config)
        pt = predict(model, x_train, config)
        mv, mt = metrics(y_validation, pv), metrics(y_train, pt)
        row = dict(epoch=epoch + 1, optimization_loss=sum(v*n for v, n in losses) / len(y_train),
                   train_nll=mt['joint_nll'], validation_nll=mv['joint_nll'], train_accuracy=mt['accuracy'],
                   validation_accuracy=mv['accuracy'], gradient_norm_mean=float(np.mean(norms)),
                   gradient_norm_max=float(np.max(norms)), gradient_clip_fraction=clipped / len(norms))
        history.append(row)
        # All losses use ordinary joint NLL for checkpoint selection. Focal's
        # training objective is deliberately not compared as a likelihood.
        if mv['joint_nll'] < best - tr.get('min_delta', 0.):
            best, best_epoch, stale = mv['joint_nll'], epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps(dict(stage='fit', run=config.get('run_id'), **row)), flush=True)
        if stale >= tr['patience']:
            break
    model.load_state_dict(best_state)
    after = gradient_diagnostic(model, x_train[probe_ids], y_train[probe_ids], config) if before is not None else None
    receipt = dict(best_epoch=best_epoch, best_validation_nll=best, epochs=len(history),
                   seconds=time.perf_counter() - started, parameter_count=sum(p.numel() for p in model.parameters()),
                   device=str(device), device_name=torch.cuda.get_device_name() if str(device).startswith('cuda') else 'CPU',
                   architecture_receptive_field=model.receptive_field, supplied_tokens=x_train.shape[1],
                   effective_tokens=min(x_train.shape[1], model.receptive_field or x_train.shape[1]),
                   gradients_before=before, gradients_selected=after, selection='minimum validation joint NLL, min_delta, earliest retained tie')
    if directory is not None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=best_state, input_dim=x_train.shape[-1], model_config=config['model'],
                        loss_config=config['loss'], seed=config['seed'], best_epoch=best_epoch), directory / 'checkpoint.pt')
        save_json(directory / 'history.json', history)
        save_json(directory / 'training.json', receipt)
    return model, history, receipt


def account_projection(frame, execution, probability, days, threshold, config):
    """Replay decisions without selecting on future completeness or labels.

    The first censored selected position halts this projection. Missing/unknown
    trial outcomes cannot be silently skipped, including for losing trades.
    """
    probability = np.asarray(probability)
    if (probability.shape != (len(frame), 4) or len(execution) != len(frame)
            or not np.isfinite(probability).all() or np.any(probability < 0)
            or not np.allclose(probability.sum(axis=1), 1., atol=1e-6)
            or not np.isfinite(threshold) or threshold < 0):
        raise ValueError('Finite aligned normalized joint probabilities required')
    selected = frame.date.isin(days).to_numpy() & frame.feature_available.to_numpy(bool)
    pl, ps = probability[:, 1] + probability[:, 3], probability[:, 2] + probability[:, 3]
    busy_until, cash, peak, closed_dd = -1, 0., 0., 0.
    ledger, rejected, empty = [], 0, 0
    censored = None
    for i in np.flatnonzero(selected):
        t = int(frame.t_ns.iloc[i])
        if t <= busy_until or max(pl[i], ps[i]) < threshold:
            continue
        side = 'L' if pl[i] >= ps[i] else 'S'
        r = execution.iloc[i]
        status = r['status_' + side]
        if status == 'INELIGIBLE':  # determined from origin state/known clock only
            rejected += 1
            continue
        if status == 'EMPTY':
            empty += 1
            busy_until = t + int(config['execution']['pending_timeout_seconds'] * 1e9)
            continue
        if status == 'CENSORED':
            censored = dict(date=str(frame.date.iloc[i]), t_ns=str(t), side=side, reason=r['reason_' + side])
            break
        if status != 'COMPLETE':
            raise ValueError('Unknown execution status')
        busy_until = int(r['exit_ns_' + side])
        net = float(r['net_' + side])
        if not np.isfinite([net, r['mae_' + side], r['mfe_' + side]]).all() or busy_until <= t:
            raise ValueError('Malformed COMPLETE account outcome')
        cash += net
        peak = max(peak, cash)
        closed_dd = max(closed_dd, peak - cash)
        ledger.append(dict(date=str(frame.date.iloc[i]), t_ns=str(t), side=side,
                           score=float(max(pl[i], ps[i])), entry_ns=str(int(r['entry_ns_' + side])),
                           exit_ns=str(busy_until), net=net, mae=float(r['mae_' + side]),
                           mfe=float(r['mfe_' + side]), reason=r['reason_' + side], cash=cash))
    result = dict(threshold=float(threshold), completed=censored is None, censored=censored,
                  trades=len(ledger), closed_net=cash, closed_cash_max_drawdown=closed_dd,
                  win_rate=sum(r['net'] > 0 for r in ledger) / len(ledger) if ledger else None,
                  entry_timeouts=empty, causally_ineligible=rejected,
                  maximum_adverse_trade_mark=min((r['mae'] for r in ledger), default=None),
                  interpretation='One reference ounce, fixed taker recipe, realized cash only; closed drawdown excludes intra-trade drawdown; incomplete projection is not a full-period return')
    return result, ledger


def select_threshold(frame, execution, probability, config):
    rows = [account_projection(frame, execution, probability, config['splits']['calibration'], t, config)[0]
            for t in config['policy']['threshold_candidates']]
    valid = [r for r in rows if r['completed'] and r['trades'] >= config['policy']['minimum_calibration_trades'] and r['closed_net'] > 0]
    chosen = max(valid, key=lambda r: (r['closed_net'], r['threshold'])) if valid else next(r for r in rows if r['threshold'] > 1.)
    return chosen['threshold'], rows
