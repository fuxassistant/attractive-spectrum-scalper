from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spectrum.research_v2.features import (
    asof_source, common_segments, geometry_features, quote_state,
    transform_kind, transform_values, _selected_capture,
    supported_source_at_origin,
)

NS = 10**9


def source():
    return pd.DataFrame(dict(t_ns=np.arange(5) * NS,
                             available_ns=np.array([0, 1, 4, 3, 4]) * NS,
                             epoch=[0, 0, 0, 0, 0], source_part=0,
                             admissible=True, source_day_degraded=False,
                             depth_bid_1=[1., 2., 3., 4., 5.],
                             depth_bid_1__known=[1, 1, 1, 1, 1],
                             imbalance_l1=[.1, .2, .3, .4, .5],
                             imbalance_l1__known=[1, 0, 1, 1, 1]))


def test_asof_requires_endpoint_and_full_publication_prefix_and_age():
    f = source()
    origins = np.array([1, 2, 3, 4, 7]) * NS
    x, proof = asof_source(f, origins, ["depth_bid_1", "imbalance_l1"], 2.)
    np.testing.assert_array_equal(proof["row_index"], [1, 1, 1, 4, 4])
    assert proof["ready"].tolist() == [True, True, True, True, False]
    assert np.isnan(x[0, 1]) and proof["ready"][0]
    assert np.isnan(x[-1]).all()
    assert np.all(proof["publication_ns"][:4] <= origins[:4])


def test_future_mutation_does_not_change_earlier_asof_values_or_segment():
    f = source()
    t = np.array([0, 1, 2, 3]) * NS
    a, pa = asof_source(f, t, ["depth_bid_1", "imbalance_l1"])
    f.loc[4, ["depth_bid_1", "imbalance_l1"]] = 99999
    f.loc[4, "epoch"] = 99
    b, pb = asof_source(f, t, ["depth_bid_1", "imbalance_l1"])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(pa["source_segment"], pb["source_segment"])


def test_source_admission_and_capture_boundary_are_distinct_from_feature_masks():
    f = source()
    f.available_ns = f.t_ns
    f.loc[2, "admissible"] = False
    f.loc[4, "source_part"] = 1
    x, proof = asof_source(f, f.t_ns, ["depth_bid_1", "imbalance_l1"])
    assert proof["ready"].tolist() == [True, True, False, True, True]
    assert np.isnan(x[2]).all()
    assert proof["source_segment"][3] != proof["source_segment"][1]
    assert proof["source_segment"][4] != proof["source_segment"][3]


def test_quote_segment_detects_intervening_invalid_publications_and_long_silence():
    q = SimpleNamespace(quote_t=np.array([0, 1, 2, 3, 70]) * NS,
                        bid=np.ones(5) * 100, ask=np.ones(5) * 101,
                        valid=np.array([1, 1, 0, 1, 1], bool), epoch=np.zeros(5, np.int64))
    proof = quote_state(q, np.array([1, 3, 70]) * NS, 60.)
    assert proof["ready"].all()
    assert len(set(proof["segment"])) == 3
    source_segment = np.ones(3, np.int64)
    tokens = common_segments(np.array([1, 3, 70]) * NS, proof["ready"], source_segment, proof["segment"], 2.)
    assert len(set(tokens)) == 3


def test_source_row_before_quote_reset_is_not_admitted_after_recovery():
    q = SimpleNamespace(quote_t=np.rint(np.array([0, .25, .5, 1, 1.5]) * NS).astype(np.int64),
                        bid=np.ones(5) * 100, ask=np.ones(5) * 101,
                        valid=np.array([1, 0, 1, 1, 1], bool), epoch=np.zeros(5, np.int64))
    origins = np.rint(np.array([.1, .6, 1.5]) * NS).astype(np.int64)
    proof = dict(ready=np.ones(3, bool), publication_ns=np.array([0, 0, 1]) * NS)
    ready, tokens = supported_source_at_origin(q, origins, proof)
    assert ready.tolist() == [True, False, True]
    assert tokens[0] != tokens[1] and tokens[1] == tokens[2]


def test_transforms_are_named_monotone_and_do_not_change_ratios_or_masks():
    names = ["depth_bid_1", "event_count_5s", "path_30s", "fft_signed_trade_32s_power",
             "fft_mid_change_32s_period4_8_fraction", "fft_mid_change_32s_entropy",
             "efficiency_5s", "imbalance_l1"]
    values = np.array([[-10., 0., 10., np.nan, .2, .3, .4, -.5]])
    x, mapping = transform_values(values, names)
    np.testing.assert_allclose(x[0, :3], [-np.log(11), 0, np.log(11)])
    np.testing.assert_allclose(x[0, 4:], values[0, 4:])
    assert np.isnan(x[0, 3])
    assert mapping[0]["output"].endswith("__signed_log1p")
    assert all(transform_kind(n) == "identity" for n in names[4:])


def test_geometry_ignores_future_columns_and_preserves_support_unknown():
    t = np.array([0, 5]) * NS
    f = pd.DataFrame(dict(t_ns=t, mid=[101., 102.], decision_spread=[.2, .3], R_event=[1., 1.]))
    for s in (1, 3, 6):
        f[f"BB{s}_middle"] = 100.
        f[f"BB{s}_half"] = 2.
        f[f"BB{s}_support"] = [0, -1] if s == 6 else [0, 0]
    f["future_event_id"] = ["a", "b"]
    x, names = geometry_features(f, t, 0)
    assert len(names) == 14
    assert x[0, names.index("bb1_position")] == .5
    assert np.isnan(x[1, names.index("bb6_position")])
    f.future_event_id = "changed"
    other, _ = geometry_features(f, t, 0)
    np.testing.assert_array_equal(x, other)
    with pytest.raises(ValueError, match="clocks"):
        geometry_features(f, t + 1, 0)


def test_previous_source_is_not_admitted_by_complete_status_alone():
    selection = dict(date="2026-04-10", symbol="GCM6", source_condition="degraded", context_only=False)
    catalog = dict(captures=[dict(status="COMPLETE", key="k", entry=dict(selection=selection))])
    overlay = dict(k=dict(audit_eligible=False))
    assert _selected_capture(catalog, overlay, "2026-04-10", "GCM6") is None
    selection["source_condition"] = "available"
    overlay["k"]["audit_eligible"] = True
    assert _selected_capture(catalog, overlay, "2026-04-10", "GCM6")["key"] == "k"
    assert _selected_capture(catalog, overlay, "2026-04-10", "GCQ6") is None
