"""Freeze the paired design and inherited target/execution contract."""
import copy
import json
from pathlib import Path

import pytest

from scripts.run_opportunity_v3 import experiment_configs, validate_recipe
from scripts.prepare_opportunity_v3 import identity
from spectrum.research.training import sha


def config():
    return json.loads(Path('configs/research/opportunity_pipeline_v3.json').read_text())


def test_complete_factorial_and_new_chronological_dates():
    c = config()
    validate_recipe(c)
    arms = list(experiment_configs(c))
    assert len(arms) == len({a['run_id'] for a in arms}) == 8
    assert {(a['seed'], a['data']['context_mode'], a['training']['weighting']) for a in arms} == {
        (s, x, w) for s in (17, 29) for x in ('legacy', 'restored') for w in ('uniform', 'crossing_id')}
    old = json.loads(Path('configs/research/opportunity_pipeline_v2.json').read_text())
    assert min(c['data']['dates']) > max(old['data']['dates'])
    assert [len(v) for v in c['splits'].values()] == [16, 5, 5, 10]
    assert sum(list(c['splits'].values()), []) == sorted(set(c['data']['dates']))
    inv = json.loads(Path(c['data']['source_inventory']).read_text())
    assert inv['dates'] == c['data']['dates'] and inv['splits'] == c['splits']
    assert sha(c['data']['source_inventory']) == c['data']['source_inventory_sha256']
    # Context/weights change lazy inputs/training, never the saved target bank.
    assert all(identity(a) == identity(c) for a in arms)


@pytest.mark.parametrize('section,key,value', [
    ('event_query', 'gain_multiple', 3.), ('loss', 'gamma', 1.),
    ('execution', 'latency_ms', 0.), ('training', 'learning_rate', .01),
    ('model', 'hidden', 64), ('data', 'family', 'rich')])
def test_unregistered_target_loss_architecture_changes_fail(section, key, value):
    c = copy.deepcopy(config())
    c[section][key] = value
    with pytest.raises(ValueError):
        validate_recipe(c)
