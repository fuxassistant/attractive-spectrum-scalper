from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from spectrum.research_v2 import features as v2
from spectrum.research_v3 import features as v3

NS = 1_000_000_000


def config():
    return dict(data=dict(cadence_seconds=5, max_feature_age_seconds=2.),
                execution=dict(max_quote_gap_seconds=60.))


def quotes(seconds, valid=None, epoch=None):
    t = np.rint(np.asarray(seconds)*NS).astype(np.int64)
    return SimpleNamespace(quote_t=t, bid=np.full(len(t), 100_000_000_000, np.int64),
        ask=np.full(len(t), 100_200_000_000, np.int64),
        valid=np.ones(len(t), bool) if valid is None else np.asarray(valid, bool),
        epoch=np.zeros(len(t), np.int64) if epoch is None else np.asarray(epoch, np.int64))


def source(endpoints, published=None, admissible=None):
    t = np.rint(np.asarray(endpoints)*NS).astype(np.int64)
    return pd.DataFrame(dict(t_ns=t,
        available_ns=t if published is None else np.rint(np.asarray(published)*NS).astype(np.int64),
        epoch=0, source_part=0, source_day_degraded=False,
        admissible=np.ones(len(t), bool) if admissible is None else admissible,
        sample=np.arange(len(t), dtype=float), sample__known=True))


def make_base(q, origins, f):
    t = np.rint(np.asarray(origins)*NS).astype(np.int64)
    x, proof = v2.asof_source(f, t, ['sample'], 2.)
    available, token = v2.supported_source_at_origin(q, t, proof)
    legacy = v2.common_segments(t, available, proof['source_segment'], token, 5)
    # V2 may retain finite geometry/context values on inadmissible rows. The
    # adapter must preserve these raw facts while masking their observation.
    raw = np.c_[x, np.full(len(t), 100.)].astype(np.float32)
    base = dict(x_raw=raw, feature_names=['sample', 'geometry'],
        groups={'micro': ['sample', 'geometry']}, feature_available=available,
        segment_token=legacy, receipts={'implementation_sha256': {}, 'original': 'frozen'})
    return base, proof, t


def test_delayed_ordinary_second_changes_legacy_but_not_quote_continuity():
    f = source(np.arange(20), np.arange(20)+.1)
    f.loc[6, 'available_ns'] = int(7.215*NS)
    f.loc[6, 'admissible'] = False
    # A short quiet period does not violate the 60-second support contract.
    q = quotes([0, 1, 2, 3, 4, 5, 5.962, 7.215, 8, 9, 10, 11, 15, 20])
    base, proof, t = make_base(q, [5, 10], f)
    original = deepcopy(base)
    out = v3.extend_history_metadata(base, q, t, proof, config())
    assert base['feature_available'].tolist() == [True, True]
    assert out['legacy_segment_token'][0] != out['legacy_segment_token'][1]
    assert out['physical_segment_token'][0] == out['physical_segment_token'][1]
    np.testing.assert_array_equal(out['x_raw'], original['x_raw'])
    np.testing.assert_array_equal(out['feature_available'], original['feature_available'])
    assert base['receipts'] == original['receipts']
    assert out['receipts']['v3_history']['adjacent_legacy_changes_with_same_supported_quote_token'] == 1


def test_missing_current_row_retains_clock_position_and_masks_finite_raw_values():
    f = source(np.arange(21), np.arange(21)+.1)
    f.loc[9, 'admissible'] = False
    q = quotes(np.arange(22))
    base, proof, t = make_base(q, [5, 10, 15], f)
    out = v3.extend_history_metadata(base, q, t, proof, config())
    assert out['feature_available'].tolist() == [True, False, True]
    assert len(set(out['physical_segment_token'])) == 1
    assert out['clock_row_present'].tolist() == [True, True, True]
    assert not out['feature_known_mask'][1].any()
    assert out['x_raw'][1, 1] == 100.  # unchanged raw geometry, never observed
    assert out['legacy_segment_token'][0] != out['legacy_segment_token'][2]
    assert out['source_age_seconds'].tolist() == [1., 1., 1.]


@pytest.mark.parametrize('kind', ['invalid', 'epoch', 'gap'])
def test_real_support_change_between_available_origins_always_changes_physical_token(kind):
    if kind == 'gap':
        q = quotes([0, 1, 70, 71, 72])
        f = source([0, 1, 70, 71, 72])
        origins = [1, 71]
    else:
        q = quotes([0, 1, 2, 3, 4, 5, 6],
            valid=[1, 1, 0, 1, 1, 1, 1] if kind == 'invalid' else None,
            epoch=[0, 0, 0, 1, 1, 1, 1] if kind == 'epoch' else None)
        f = source(np.arange(7))
        origins = [1, 6]
    base, proof, t = make_base(q, origins, f)
    out = v3.extend_history_metadata(base, q, t, proof, config())
    assert out['feature_available'].all()
    assert out['physical_support_available'].all()
    assert out['physical_segment_token'][0] != out['physical_segment_token'][1]


def test_fresh_by_age_prereset_source_is_not_observed_after_quote_recovery():
    q = quotes([0, .25, .5, 1, 1.5], valid=[1, 0, 1, 1, 1])
    f = source([0, 1])
    base, proof, t = make_base(q, [.1, .6, 1.5], f)
    out = v3.extend_history_metadata(base, q, t, proof, config())
    assert out['feature_available'].tolist() == [True, False, True]
    assert out['physical_support_available'].all()
    assert out['source_age_seconds'][1] == .6
    assert not out['source_same_quote_support'][1]
    assert not out['feature_known_mask'][1].any()
    assert out['physical_segment_token'][0] != out['physical_segment_token'][1]
    assert out['physical_segment_token'][1] == out['physical_segment_token'][2]


def test_unavailable_physical_support_is_distinct_from_feature_missingness():
    q = quotes([0, 1, 2, 3, 4], valid=[1, 1, 0, 1, 1])
    f = source([0, 1, 2, 3, 4])
    base, proof, t = make_base(q, [1, 2, 3], f)
    out = v3.extend_history_metadata(base, q, t, proof, config())
    assert out['physical_support_available'].tolist() == [True, False, True]
    assert out['physical_segment_token'][1] == -1
    assert out['clock_row_present'][1]  # a real grid row, not left padding
    assert not out['feature_known_mask'][1].any()


def test_future_mutation_cannot_change_prefix_values_admission_or_physical_tokens():
    q = quotes(np.arange(21))
    f = source(np.arange(21), np.arange(21)+.1)
    a, proof, t = make_base(q, [5, 10], f)
    before = v3.extend_history_metadata(a, q, t, proof, config())
    altered = quotes(np.arange(21))
    altered.valid[12] = False
    altered.epoch[13:] = 10
    altered.bid[14:] = 500_000_000_000
    altered.ask[14:] = 500_200_000_000
    f.loc[12:, 'sample'] = 99999
    f.loc[12:, 'epoch'] = 77
    b, proof2, t2 = make_base(altered, [5, 10], f)
    after = v3.extend_history_metadata(b, altered, t2, proof2, config())
    for name in ('x_raw', 'feature_available', 'feature_known_mask', 'legacy_segment_token',
                 'physical_segment_token', 'source_endpoint_ns', 'source_available_ns', 'source_age_seconds'):
        np.testing.assert_array_equal(before[name], after[name])


@pytest.mark.parametrize('mutation', ['cohort', 'publication', 'age', 'legacy'])
def test_changed_admission_or_invalid_clock_proof_fails_closed(mutation):
    q = quotes(np.arange(21))
    base, proof, t = make_base(q, [5, 10], source(np.arange(21), np.arange(21)+.1))
    if mutation == 'cohort':
        base['feature_available'][0] = False
    elif mutation == 'publication':
        proof['publication_ns'][0] = t[0]+1
    elif mutation == 'age':
        proof['age_seconds'][0] = 999
    else:
        base['segment_token'][0] = 999
    with pytest.raises(ValueError):
        v3.extend_history_metadata(base, q, t, proof, config())


def test_wrapper_preserves_v2_family_schema_and_array_identity(monkeypatch):
    q = quotes(np.arange(21))
    base, proof, t = make_base(q, [5, 10], source(np.arange(21), np.arange(21)+.1))
    monkeypatch.setattr(v2, 'build_features', lambda *args, **kwargs: base)
    monkeypatch.setattr(v3, '_metadata_proof', lambda *args, **kwargs: proof)
    out = v3.build_features(None, q, t, None, config())
    assert out['x_raw'] is base['x_raw']
    assert out['feature_names'] is base['feature_names']
    assert out['groups'] is base['groups']
    assert out['feature_available'] is base['feature_available']
    assert out['segment_token'] is base['segment_token']
    assert out['receipts']['v3_history']['extra_feature_channels'] == 0
