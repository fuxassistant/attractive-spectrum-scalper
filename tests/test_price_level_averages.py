"""Independent identities and availability/coverage failures for averages."""
import numpy as np
import pandas as pd
import pytest

from experiments.price_level_memory_v1.averages import compare_windows, weighted_decomposition

NS = 1_000_000_000


def frames(mid=(100, 102, 104, 106, 108), times=(0, 1, 2, 3, 4),
           trade_times=(.5, 1.5, 2.5, 3.5), prices=(100, 102, 104, 106), weights=(1, 1, 1, 1)):
    q = pd.DataFrame(dict(available_ns=np.rint(np.asarray(times)*NS).astype(np.int64),
        bid_nano=np.rint((np.asarray(mid)-.1)*NS).astype(np.int64),
        ask_nano=np.rint((np.asarray(mid)+.1)*NS).astype(np.int64), valid=True, epoch=7))
    t = pd.DataFrame(dict(available_ns=np.rint(np.asarray(trade_times)*NS).astype(np.int64),
        price_nano=np.rint(np.asarray(prices)*NS).astype(np.int64),quantity=weights,supported=True,epoch=0))
    return q, t


def query(q, t, right=4, width=4, **kwargs):
    return compare_windows(q, t, np.array([round(right*NS)], np.int64), (width,),
        coverage_start_ns=0, coverage_end_ns=10*NS, **kwargs).iloc[0]


@pytest.mark.parametrize("weights,sign", [([1, 2, 4], 1), ([4, 2, 1], -1), ([1, 1, 1], 0)])
def test_covariance_predicts_center_direction_for_identical_samples(weights, sign):
    r = weighted_decomposition([100, 101, 102], weights)
    assert np.sign(r["center_gap"]) == sign
    assert r["center_gap"] == pytest.approx(r["covariance_center_gap"], abs=1e-14)
    assert r["variance_gap"] == pytest.approx(r["covariance_variance_gap"], abs=1e-14)


def test_zero_center_covariance_does_not_imply_identical_band_width():
    r = weighted_decomposition([100, 101, 102], [4, 1, 4])
    assert r["center_gap"] == 0
    assert r["weighted_population_variance"] > r["population_variance"]
    r = weighted_decomposition([100, 101, 102], [1, 4, 1])
    assert r["center_gap"] == 0
    assert r["weighted_population_variance"] < r["population_variance"]


def test_equal_volume_and_uniform_print_timing_converge_to_exact_time_mean_and_population_sigma():
    q, t = frames()
    r = query(q, t)
    assert r.known
    assert r.twa_mid == r.vwap_trade == r.equal_trade_price_mean == 103
    assert r.twa_mid_sigma_population == pytest.approx(np.sqrt(5))
    assert r.vwap_trade_sigma_population == pytest.approx(np.sqrt(5))
    assert r.trade_size_weighting_gap == r.trade_price_basis_gap == r.trade_timing_gap == 0


def test_trade_timing_basis_and_size_are_distinct_terms_with_exact_sum():
    q, t = frames(trade_times=(.5, 1.5, 3.1, 3.5), prices=(100.1, 102.1, 106.1, 106.1), weights=(1, 1, 2, 4))
    r = query(q, t)
    assert r.known
    assert r.trade_timing_gap == pytest.approx(.5)
    assert r.trade_price_basis_gap == pytest.approx(.1)
    assert r.trade_size_weighting_gap > 0
    assert abs(r.decomposition_residual) < 1e-12
    assert r.trade_price_size_covariance/r.mean_trade_size == pytest.approx(r.trade_size_weighting_gap)
    assert r.vwap_trade-r.twa_mid == pytest.approx(r.weighted_mid_gap+r.volume_weighted_trade_basis_gap)


def test_time_integral_is_not_equal_quote_update_average():
    q, t = frames(mid=(100, 104, 104), times=(0, 3, 4), trade_times=(.5, 3.5), prices=(100, 104), weights=(1, 1))
    r = query(q, t)
    assert r.twa_mid == 101
    assert r.vwap_trade == 102
    assert r.trade_timing_gap == 1
    assert r.twa_mid_sigma_population == pytest.approx(np.sqrt(3))


@pytest.mark.parametrize("bad", ["quote_invalid", "quote_epoch", "trade_unsupported", "trade_epoch", "stale"])
def test_unsupported_windows_are_not_imputed(bad):
    q, t = frames()
    kwargs = {}
    if bad == "quote_invalid": q.loc[2, "valid"] = False
    if bad == "quote_epoch": q.loc[2:, "epoch"] = 8
    if bad == "trade_unsupported": t.loc[2, "supported"] = False
    if bad == "trade_epoch": t.loc[2:, "epoch"] = 1
    if bad == "stale": kwargs["max_gap_seconds"] = .5
    r = query(q, t, **kwargs)
    assert not r.known
    assert np.isnan(r.decomposition_residual)


def test_half_open_boundaries_no_trades_no_fake_vwap_and_future_prefix_invariance():
    q, t = frames(trade_times=(0, 1, 2, 3), prices=(100, 102, 104, 106))
    first = query(q, t, right=3, width=3)
    assert first.trade_count == 3  # print exactly at R is excluded
    prefix = query(q.loc[q.available_ns<=3*NS], t.loc[t.available_ns<3*NS], right=3, width=3)
    pd.testing.assert_series_equal(first, prefix)
    zero = query(q, t.iloc[:0])
    assert zero.quote_known and not zero.trade_known and not zero.known
    assert np.isnan(zero.vwap_trade) and zero.trade_count == 0


def test_same_publication_quote_reference_is_final_published_state_not_pretrade():
    q, t = frames(mid=(100, 102, 108, 104, 106, 108), times=(0, 1, 1, 2, 3, 4),
                  trade_times=(1,), prices=(102,), weights=(2,))
    r = query(q, t)
    assert r.equal_trade_publication_mid == 108
    assert r.trade_price_basis_gap == -6
    assert r.decomposition_residual == pytest.approx(0)


def test_band_ddof_difference_remains_even_when_weighted_and_equal_price_distribution_match():
    p = np.arange(20, dtype=float)
    r = weighted_decomposition(p, np.ones(20))
    population = np.sqrt(r["weighted_population_variance"])
    assert np.std(p, ddof=1)/population == pytest.approx(np.sqrt(20/19))


def test_invalid_clocks_and_weights_are_rejected():
    q, t = frames()
    q.available_ns = q.available_ns.astype(float)
    with pytest.raises(ValueError, match="integer"):
        query(q, t)
    with pytest.raises(ValueError, match="positive total"):
        weighted_decomposition([100, 101], [0, 0])

def test_sparse_prints_cannot_establish_source_coverage_from_future_last_print():
    q,t=frames()
    for prefix in (t,t[t.available_ns<3*NS]):
        with pytest.raises(ValueError,match='Explicit verified coverage'):
            compare_windows(q,prefix,np.array([3*NS]),(2,))
