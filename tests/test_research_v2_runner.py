import copy

import pytest

from scripts.run_opportunity_v2 import choose_validation, validate_dataset_identity, read


def test_development_selection_cannot_use_test_or_economic_winners():
    runs = [
        dict(run_id='a', evaluation=dict(validation=dict(model=dict(joint_nll=.8)), test=dict(joint_nll=100)), economics=-1000),
        dict(run_id='b', evaluation=dict(validation=dict(model=dict(joint_nll=.9)), test=dict(joint_nll=0)), economics=1000),
    ]
    assert choose_validation(runs) == 'a'
    changed = copy.deepcopy(runs)
    for row in changed:
        row['evaluation']['test'] = {'arbitrary_future': object()}
        row['economics'] = object()
    assert choose_validation(changed) == 'a'


def test_selection_ties_and_invalid_scores_are_explicit():
    assert choose_validation([
        dict(run_id=n, evaluation=dict(validation=dict(model=dict(joint_nll=.8))))
        for n in ('b', 'a')
    ]) == 'a'
    with pytest.raises(ValueError, match='Nonfinite'):
        choose_validation([dict(run_id='a', evaluation=dict(validation=dict(model=dict(joint_nll=float('nan')))))])


def test_dataset_reuse_rejects_target_cost_and_split_changes():
    from scripts.prepare_opportunity_v2 import identity
    c = read('configs/research/opportunity_pipeline_v2.json')
    manifest = {'identity': identity(c)}
    validate_dataset_identity(manifest, c)
    for section, key, value in [('event_query', 'deadline_seconds', 600.),
                                ('execution', 'fee_roundtrip', .6),
                                ('risk', 'cost_multipliers', [3., 5.]),
                                ('splits', 'validation', c['splits']['test'])]:
        changed = copy.deepcopy(c)
        changed[section][key] = value
        with pytest.raises(ValueError, match='Dataset recipe'):
            validate_dataset_identity(manifest, changed)
    # Selecting a feature view or architecture does not alter the common data.
    c['data']['family'] = 'geometry'
    c['model']['architecture'] = 'mlp'
    validate_dataset_identity(manifest, c)
