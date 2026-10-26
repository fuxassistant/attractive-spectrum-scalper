"""Future midpoint move from a start to an end price; extracted from spectrum/market_repr/probes.py."""
import numpy as np
import pandas as pd
from spectrum.units import NS, GC_TICK_USD as TICK
from spectrum.leakage import future_output

@future_output()
def future_endpoint_move(start, end, deadband_ticks=2):
    # Midpoints live on half ticks. Avoid changing the FLAT label because
    # binary floating point represented an exact 2-tick move as 2+epsilon.
    delta = (np.rint(np.asarray(end) / TICK * 2) - np.rint(np.asarray(start) / TICK * 2)) / 2
    outcome = np.where(delta > deadband_ticks, 2, np.where(delta < -deadband_ticks, 0, 1))
    return delta, outcome
