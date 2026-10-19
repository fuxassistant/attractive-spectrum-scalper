"""Missing event banks are absent evidence, not synthetic training samples."""
import copy
import json

import numpy as np
import pytest
import torch

from spectrum.player_v3.policy import MixedPlayer, PreparedDay, Standardizer


class MemoryDay:
    def __init__(self, events=None):
        self.x = np.asarray([[1, 1], [3, 1]], np.float32)
        self.fast_x = self.x.copy()
        self.slow_x = self.x.copy()
        self.anchors = self.x.copy()
        self.slow_session = np.ones(2, np.float32)
        self.slow_cuts = np.asarray([1, 2])
        self.events = (np.empty((0, 3), np.float32) if events is None
                       else np.asarray(events, np.float32))
        self.meta = {key: ['value', 'value__known'] for key in
                     ('feature_names', 'fast_names', 'slow_names', 'anchor_names')}
        self.meta['event_names'] = ['amplitude', 'amplitude__known', 'fresh_impulse']

    def memory(self, i):
        return {'events': self.events}

    def memory_bounds(self, i):
        return 0, i+1, 0, i+1, 0, len(self.events)


def test_empty_event_fit_has_identity_stats_and_zero_support(tmp_path):
    scaler = Standardizer().fit([MemoryDay(), MemoryDay()])
    stats = scaler.stats['events']
    np.testing.assert_array_equal(stats['center'], np.zeros(3))
    np.testing.assert_array_equal(stats['scale'], np.ones(3))
    assert stats['status'] == 'empty'
    assert stats['sample_count'] == 0
    assert stats['finite_counts'] == [0, 0, 0]
    assert stats['names'] == MemoryDay().meta['event_names']
    assert scaler.transform('events', np.empty((0, 3))).shape == (0, 3)

    path = tmp_path/'normalization.json'
    scaler.save(path)
    loaded = Standardizer.load(path)
    assert loaded.stats['events']['status'] == 'empty'
    assert loaded.stats['events']['sample_count'] == 0
    assert loaded.stats['events']['finite_counts'] == [0, 0, 0]
    np.testing.assert_array_equal(loaded.stats['events']['center'], stats['center'])
    np.testing.assert_array_equal(loaded.stats['events']['scale'], stats['scale'])


def test_nonempty_statistics_preserve_existing_moments_and_masks(tmp_path):
    events = np.asarray([[2, 1, 1], [6, 1, 0], [0, 0, 1]], np.float32)
    scaler = Standardizer().fit([MemoryDay(), MemoryDay(events)])
    stats = scaler.stats['events']
    expected_center = np.nanmean(events.astype(np.float64), axis=0).astype(np.float32)
    expected_scale = np.maximum(np.nanstd(events.astype(np.float64), axis=0), 1e-4).astype(np.float32)
    expected_center[1:] = 0
    expected_scale[1:] = 1
    np.testing.assert_array_equal(stats['center'], expected_center)
    np.testing.assert_array_equal(stats['scale'], expected_scale)
    assert stats['sample_count'] == 3
    assert stats['finite_counts'] == [3, 3, 3]
    assert stats['status'] == 'observed'
    transformed = scaler.transform('events', events)
    np.testing.assert_array_equal(transformed[:, 1:], events[:, 1:])
    assert transformed[2, 0] == 0  # Unknown amplitude remains masked.

    path = tmp_path/'normalization.json'
    scaler.save(path)
    loaded = Standardizer.load(path)
    np.testing.assert_array_equal(loaded.transform('events', events), transformed)


def test_empty_bank_batches_without_fake_events_or_router_mass():
    torch.set_num_threads(1)
    torch.manual_seed(10)
    day = MemoryDay()
    prepared = PreparedDay(day, Standardizer().fit([day]))
    _, _, events, known, counts = prepared.batch([0, 1], torch.device('cpu'))
    assert events.shape == (2, 256, 3)
    assert not known.any()
    assert not events.any()
    np.testing.assert_array_equal(counts[:, 2], [0, 0])

    model = MixedPlayer(fast_dim=2, slow_dim=2, anchor_dim=2, event_dim=3).eval()
    changed = copy.deepcopy(model)
    with torch.no_grad():
        changed.event_value.bias.fill_(100)
        changed.branch_project[3].bias.fill_(100)
        output, diag = model.encode_day(prepared, diagnostics=True)
        other = changed.encode_day(prepared)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, other, rtol=0, atol=0)
    np.testing.assert_array_equal(diag['event_attention'], np.zeros((2, 256)))
    np.testing.assert_array_equal(diag['branch_weights'][:, 3], np.zeros(2))


def test_legacy_normalization_loads_with_explicit_unknown_support(tmp_path):
    path = tmp_path/'legacy.json'
    path.write_text(json.dumps({'events': {'center': [2, 0, 0], 'scale': [2, 1, 1],
        'names': ['amplitude', 'amplitude__known', 'fresh_impulse']}}))
    scaler = Standardizer.load(path)
    assert scaler.stats['events']['status'] == 'legacy_unknown_support'
    assert scaler.stats['events']['sample_count'] is None
    assert scaler.stats['events']['finite_counts'] is None
    np.testing.assert_array_equal(scaler.transform('events', [[4, 1, 0], [0, 0, 1]]),
                                  [[1, 1, 0], [0, 0, 1]])
    scaler.save(tmp_path/'resaved.json')
    assert Standardizer.load(tmp_path/'resaved.json').stats['events']['sample_count'] is None


def test_event_single_underscore_mask_preserves_unknown_zero():
    scaler = Standardizer()
    scaler.stats = {'events': {'center': np.asarray([4, 0]), 'scale': np.asarray([2, 1]),
                              'names': ['interval_cv', 'interval_cv_known']}}
    np.testing.assert_array_equal(scaler.transform('events', [[0, 0], [6, 1]]),
                                  [[0, 0], [1, 1]])


@pytest.mark.parametrize('events', [np.empty((0, 2)), np.empty(0),
                                   [[1, 1]], [[np.nan, 1, 0]], [[np.inf, 1, 0]]])
def test_malformed_event_bank_fails_instead_of_becoming_empty(events):
    with pytest.raises(ValueError, match='events:'):
        Standardizer().fit([MemoryDay(events)])


def test_schema_mismatch_and_absent_training_data_fail():
    with pytest.raises(ValueError, match='at least one'):
        Standardizer().fit([])
    changed = MemoryDay()
    changed.meta['event_names'] = ['other', 'amplitude__known', 'fresh_impulse']
    with pytest.raises(ValueError, match='feature schemas differ'):
        Standardizer().fit([MemoryDay(), changed])
    changed = MemoryDay()
    changed.x = np.empty((0, 2), np.float32)
    with pytest.raises(ValueError, match='base: no training observations'):
        Standardizer().fit([changed])


@pytest.mark.parametrize('value', [np.empty((0, 2)), np.empty(0), [[np.nan, 1, 0]]])
def test_transform_checks_event_schema(value):
    scaler = Standardizer().fit([MemoryDay()])
    with pytest.raises(ValueError, match='events:'):
        scaler.transform('events', value)


def test_invalid_saved_normalization_rejected(tmp_path):
    path = tmp_path/'invalid.json'
    path.write_text(json.dumps({'events': {'center': [0], 'scale': [0], 'names': ['amplitude']}}))
    with pytest.raises(ValueError, match='invalid saved normalization'):
        Standardizer.load(path)
