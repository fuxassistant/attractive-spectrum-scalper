import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from spectrum.research.training import account_projection, select_threshold, metrics, validate_config, split_indices
from scripts.run_opportunity_pipeline import override


def config():
    return json.loads(Path('configs/research/opportunity_pipeline_v1.json').read_text())


def fixture():
    frame = pd.DataFrame(dict(date=['2026-02-05']*4, t_ns=np.arange(4)*10**9, feature_available=[True]*4))
    execution = pd.DataFrame({**{f'status_{s}': ['COMPLETE', 'CENSORED', 'COMPLETE', 'COMPLETE'] for s in ['L', 'S']},
                             **{f'reason_{s}': ['TAKE', 'SUPPORT', 'TAKE', 'TAKE'] for s in ['L', 'S']},
                             **{f'{field}_{s}': [value]*4 for field, value in [('net', 2.), ('mae', -1.), ('mfe', 2.)] for s in ['L', 'S']},
                             **{f'{field}_{s}': np.arange(4)*10**9 + value for field, value in [('entry_ns', 200000000), ('exit_ns', 500000000)] for s in ['L', 'S']}})
    p = np.tile([.1, .8, .1, 0.], (4, 1))
    return frame, execution, p


def test_account_does_not_drop_future_censored_trade():
    f, e, p = fixture()
    result, ledger = account_projection(f, e, p, ['2026-02-05'], .5, config())
    assert result['trades'] == 1 and not result['completed']
    assert result['censored']['t_ns'] == str(10**9)
    assert result['closed_net'] == 2.  # partial, not a full-period result
    p[1] = [.9, .05, .05, 0.]
    result, ledger = account_projection(f, e, p, ['2026-02-05'], .5, config())
    assert result['trades'] == 3 and result['completed']  # future censor is unselected


def test_one_position_and_wait_policy():
    f, e, p = fixture()
    e.loc[:, ['status_L', 'status_S']] = 'COMPLETE'
    e.loc[0, 'exit_ns_L'] = 2500000000
    result, ledger = account_projection(f, e, p, ['2026-02-05'], .5, config())
    assert len(ledger) == 2 and ledger[-1]['t_ns'] == str(3*10**9)
    result, ledger = account_projection(f, e, p, ['2026-02-05'], 1.01, config())
    assert result['trades'] == 0 and result['closed_net'] == 0


def test_threshold_selection_never_uses_test_and_rejects_partial_profit():
    f, e, p = fixture()
    c = config()
    c['policy']['minimum_calibration_trades'] = 1
    threshold, grid = select_threshold(f, e, p, c)
    assert threshold > 1  # profitable partial prefix cannot win
    e.loc[:, ['status_L', 'status_S']] = 'COMPLETE'
    threshold, grid = select_threshold(f, e, p, c)
    assert threshold <= .8
    extra = f.iloc[[0]].copy()
    extra['date'] = '2026-02-06'
    f2 = pd.concat([f, extra], ignore_index=True)
    e2 = pd.concat([e, e.iloc[[0]]], ignore_index=True)
    e2.loc[len(e), 'net_L'] = -1000000.
    assert select_threshold(f2, e2, np.vstack([p, p[0]]), c)[0] == threshold


def test_prior_metrics_and_joint_nll():
    m = metrics(np.array([0, 1, 2, 3, -1]), np.full((5, 4), .25))
    assert m['n'] == 4
    assert m['joint_nll'] == pytest.approx(np.log(4))
    assert m['event_auc'] == .5


def test_config_and_switches_fail_closed():
    c = config()
    validate_config(c)
    override(c, 'model.architecture=lstm')
    override(c, 'loss.event_loss=proper_confidence')
    override(c, 'loss.epsilon=0.3333333333333333')
    validate_config(c)
    with pytest.raises(ValueError):
        override(c, 'model.architecure=tcn')
    c['splits']['validation'] = c['splits']['test']
    with pytest.raises(ValueError):
        validate_config(c)


def test_joint_mask_keeps_unknown_out_of_fit_only():
    f = pd.DataFrame(dict(date=['2026-02-02']*3, feature_available=[True, True, False]))
    ids = split_indices(f, np.array([1, -1, 0]), config())
    assert ids['train'].tolist() == [0]
    assert len(f) == 3


def test_fail_closed_nonfinite_predictions_and_hyperparameters():
    f, e, p = fixture()
    p[0, 1] = np.nan
    with pytest.raises(ValueError):
        account_projection(f, e, p, ['2026-02-05'], 1.01, config())
    for field in ('min_delta', 'weight_decay'):
        c = config()
        c['training'][field] = float('nan')
        with pytest.raises(ValueError):
            validate_config(c)
    c = config()
    c['data']['cadence_seconds'] = 60
    with pytest.raises(ValueError):
        validate_config(c)
