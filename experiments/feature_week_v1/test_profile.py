"""Admission tests for the new trade-to-feature adapter, independent T arithmetic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from profile import exact_moments


def test_moments_keep_positional_alignment_and_zero_bucket():
    # Deliberately non-default input index and nanosecond group index.
    ns = 1_000_000_000
    start = 1787529600 * ns
    seconds = pd.DataFrame(dict(t_ns=[start+ns, start+2*ns, start+3*ns],
                                available_ns=[start+ns+5, start+2*ns+5, start+3*ns+5]), index=[8, 9, 10])
    trades = pd.DataFrame(dict(t_ns=[start+2, start+3, start+2*ns+7],
        price_nano=[4700*ns, 4702*ns, 4703*ns], quantity=[1, 3, 2],
        supported=[True]*3, available_ns=[start+2, start+3, start+2*ns+7]))
    release = seconds.available_ns.to_numpy()
    out = exact_moments(seconds, trades, release)
    np.testing.assert_allclose(out.trade_volume, [4, 0, 2])
    np.testing.assert_allclose(out.trade_price_mean, [4701.5, 0, 4703])
    np.testing.assert_allclose(out.trade_price_m2, [3, 0, 0])
    assert out.trade_moments_valid.all()
    assert out.index.equals(seconds.index)
    assert (out.available_ns == release).all()


def test_unsupported_trade_is_not_certified_as_exact_moment():
    seconds = pd.DataFrame(dict(t_ns=[2_000_000_000], available_ns=[2_000_000_001]))
    trades = pd.DataFrame(dict(t_ns=[1_500_000_000], price_nano=[4700_000_000_000],
                              quantity=[1], supported=[False], available_ns=[1_500_000_000]))
    assert not exact_moments(seconds, trades, seconds.available_ns.to_numpy()).trade_moments_valid.iloc[0]
