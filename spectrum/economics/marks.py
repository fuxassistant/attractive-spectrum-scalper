"""Money of one reference ounce held from an origin to its future crossing."""
import numpy as np
from spectrum.canon import load_canon
from spectrum.future.crossings import ORIGIN_KEYS
from spectrum.leakage import mark_future, export_future_values


def future_crossing_marks(future, fee_usd_per_oz=None):
    """Long buys the origin ask and sells the crossing bid; short the reverse.

    ``future`` are rows of future.future_crossings. Marks are USD/oz net of the
    round-trip fee (canon fee_roundtrip_usd_per_oz unless given), NaN where
    the line was not crossed. They are future too, hence future_* names.
    Latency and fills are execution rules and are not modelled here.
    """
    fee = load_canon().value('fee_roundtrip_usd_per_oz') if fee_usd_per_oz is None else float(fee_usd_per_oz)
    crossed = future.future_status.to_numpy() == 'CROSSED'
    long = np.where(crossed, future.future_bid.to_numpy() - future.origin_ask.to_numpy() - fee, np.nan)
    short = np.where(crossed, future.origin_bid.to_numpy() - future.future_ask.to_numpy() - fee, np.nan)
    marked = future.assign(future_long_usd=export_future_values(long),
                           future_short_usd=export_future_values(short), fee_usd=fee)
    return mark_future(marked, __name__ + '.future_crossing_marks', ORIGIN_KEYS + ('fee_usd',))
