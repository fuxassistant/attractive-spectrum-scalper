"""Causal, descriptive recurrence diagnostics for one fixed absolute price band.

Input values are *completed* uniform-bin occupancy fractions, not returns and
not an event-indexed arrival series. A value stamped t describes [t-step,t).
Every bin in the trailing window must be observed and belong to one epoch.
Unknown bins are never filled, interpolated or treated as zero occupancy.

One right-aligned periodic-Hann FFT is one causal STFT slice. Repeating this
function at successive query clocks supplies the time axis; no centred future
window or edge padding is used. The Hann/constant-detrend PSD matches the
definition in scipy.signal.periodogram, implemented here with NumPy only.

The supported-period flag means only sufficient resolved cycles and observed
arrival intervals. Concentration is not a significance test, a forecast, or an
economic edge. A transient/random walk can have a concentrated spectrum.

Primary references:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html
"""
from __future__ import annotations

import math
import numpy as np

NS = 1_000_000_000


def _clock_array(value, name):
    a = np.asarray(value)
    if a.ndim == 1 and len(a) == 0:
        return np.empty(0, dtype=np.int64)
    if a.ndim != 1 or a.dtype.kind not in "iu" or a.dtype.kind == "b":
        raise ValueError(f"{name} must be a one-dimensional integer nanosecond array")
    if a.dtype.kind == "u" and len(a) and int(a.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds int64 nanosecond range")
    return a.astype(np.int64, copy=False)


def _seconds_ns(value, name):
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    result = round(float(value) * NS)
    if result <= 0 or not math.isclose(result / NS, value, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{name} must resolve to integer nanoseconds")
    return result


def hann_periodogram(values, sample_seconds):
    """Return frequency Hz and one-sided PSD of an observed finite segment.

    The segment mean is removed before multiplication by the periodic Hann.
    DC is returned here for independent numerical verification; zone_spectrum
    excludes DC when reporting concentration. There is no FFT zero-padding.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("values must be a finite one-dimensional segment of length >=4")
    _seconds_ns(sample_seconds, "sample_seconds")
    n = len(x)
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(n) / n)
    transformed = np.fft.rfft((x - x.mean()) * window)
    power = abs(transformed) ** 2 * float(sample_seconds) / np.dot(window, window)
    power[1:-1 if n % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(n, float(sample_seconds)), power


def zone_spectrum(times_ns, occupancy, query_ns, *, window_seconds=3600,
                  sample_seconds=5, known=None, epoch=None, arrival_ns=None,
                  arrival_available_ns=None, min_cycles=3, min_period_seconds=30,
                  include_spectrum=False):
    """Describe one frozen band's trailing occupancy and exact visit recurrence.

    ``times_ns`` contains strictly increasing completed-bin end clocks on the
    UTC/Unix grid. ``occupancy`` is dwell/step in [0,1]. Query is floored to the
    last complete grid boundary. Input may contain later data; only bins in
    (window_start, window_end] contribute. ``known=False`` marks any bin with
    incomplete support, including gaps inside that bin. Pass ``epoch`` to reject
    windows across continuity boundaries; absence means caller asserts one epoch.

    Exact confirmed ``arrival_ns`` is a distinct irregular event series. Pass
    ``arrival_available_ns`` when contacts are known only after debounce; absent
    availability means event time is availability (or the caller has already
    selected its confirmed prefix). Confirmations must preserve contact order
    (nondecreasing availability), otherwise skipping a pending intervening visit
    could invent a complete return interval. No restart-present event is allowed.
    Only contacts in [window_start,window_end), available by window_end, count.
    Three *complete* return intervals require at least four arrivals.

    ``dominant_period_seconds`` is withheld unless the strongest non-DC peak
    spans >=min_cycles and is >=min_period_seconds, with enough exact arrivals.
    It is not replaced by a weaker eligible high-frequency peak. This avoids
    relabelling a single long passage as a shorter recurrence. ``raw_peak_*``
    remain descriptive even when the period is unsupported. Unknown outputs
    are None; observed zero occupancy remains known with NO_OCCUPANCY status.

    Autocorrelation uses global segment demeaning and the biased finite-window
    denominator sum(z*z), without circular wrap. The best eligible lag can be
    persistence; it is not automatically a recurrence peak.
    """
    t = _clock_array(times_ns, "times_ns")
    if len(t) > 1 and np.any(t[1:] <= t[:-1]):
        raise ValueError("times_ns must be strictly increasing")
    if isinstance(query_ns, (bool, np.bool_)) or not isinstance(query_ns, (int, np.integer)):
        raise ValueError("query_ns must be integer nanoseconds")
    step = _seconds_ns(sample_seconds, "sample_seconds")
    width = _seconds_ns(window_seconds, "window_seconds")
    minimum = _seconds_ns(min_period_seconds, "min_period_seconds")
    if width % step:
        raise ValueError("window_seconds must be an integer multiple of sample_seconds")
    if not isinstance(min_cycles, (int, np.integer)) or isinstance(min_cycles, (bool, np.bool_)) or min_cycles < 3:
        raise ValueError("min_cycles must be an integer >=3")
    n = width // step
    if n < 8 or minimum < 2 * step or minimum * min_cycles > width:
        raise ValueError("window must support >=3 cycles and periods >=2 samples")
    x = np.asarray(occupancy, dtype=float)
    if x.shape != t.shape:
        raise ValueError("occupancy and times_ns shapes differ")
    mask = np.ones(len(t), dtype=bool) if known is None else np.asarray(known)
    if mask.shape != t.shape or mask.dtype.kind != "b":
        raise ValueError("known must be a boolean array matching times_ns")
    epochs = None if epoch is None else np.asarray(epoch)
    if epochs is not None and (epochs.shape != t.shape or epochs.dtype.kind not in "iu"):
        raise ValueError("epoch must be an integer array matching times_ns")
    right = int(query_ns) // step * step
    left = right - width
    if not np.iinfo(np.int64).min <= left <= right <= np.iinfo(np.int64).max:
        raise ValueError("window clock exceeds int64 range")
    lo, hi = np.searchsorted(t, [left, right], side="right")
    local_t, segment = t[lo:hi], x[lo:hi]
    if len(local_t) and np.any(local_t % step):
        raise ValueError("completed bins must use the UTC/Unix sampling grid")
    valid = mask[lo:hi] & np.isfinite(segment) & (segment >= 0) & (segment <= 1)
    if epochs is not None:
        valid &= epochs[lo:hi] >= 0
    found = len(segment)
    observed = int(valid.sum())
    result = dict(query_ns=int(query_ns), window_start_ns=left, window_end_ns=right,
                  available_ns=right, window_seconds=width / NS, sample_seconds=step / NS,
                  frequency_resolution_hz=NS / width, min_period_seconds=minimum / NS,
                  max_period_seconds=width / NS / min_cycles, min_cycles=int(min_cycles),
                  n_expected=int(n), n_present=found, n_known=observed,
                  coverage_fraction=observed / n, known=False, spectrum_known=False,
                  status="INCOMPLETE_WINDOW", occupancy_mean=None, occupancy_variance=None,
                  raw_peak_period_seconds=None, raw_peak_cycles=None,
                  peak_power_fraction=None, main_lobe_power_fraction=None,
                  spectral_entropy=None, dominant_period_seconds=None,
                  period_candidate_supported=False, autocorr_at_peak=None,
                  autocorr_best_lag_seconds=None, autocorr_best_value=None,
                  arrival_count=None, return_interval_count=None,
                  return_interval_mean_seconds=None, return_interval_cv=None,
                  return_interval_cv_known=False, recurrence_evidence_known=False,
                  descriptive_only=True, periodicity_significance=None,
                  economic_admission="NOT_ESTABLISHED")
    arrivals = availability = None
    if arrival_available_ns is not None and arrival_ns is None:
        raise ValueError("arrival_available_ns requires arrival_ns")
    if arrival_ns is not None:
        arrivals = _clock_array(arrival_ns, "arrival_ns")
        if len(arrivals) > 1 and np.any(arrivals[1:] <= arrivals[:-1]):
            raise ValueError("arrival_ns must be strictly increasing; do not duplicate visits")
        availability = arrivals if arrival_available_ns is None else _clock_array(arrival_available_ns, "arrival_available_ns")
        if availability.shape != arrivals.shape or np.any(availability < arrivals):
            raise ValueError("arrival availability must match and cannot precede contact")
        if len(availability) > 1 and np.any(availability[1:] < availability[:-1]):
            raise ValueError("arrival confirmations must preserve contact order (FIFO availability)")
    if found != n or observed != n:
        return result
    if epochs is not None and np.any(epochs[lo:hi] != epochs[lo]):
        result["status"] = "EPOCH_BOUNDARY"
        return result
    result.update(known=True, occupancy_mean=float(segment.mean()),
                  occupancy_variance=float(segment.var()))
    if arrivals is not None:
        selected = arrivals[(arrivals >= left) & (arrivals < right) & (availability <= right)]
        intervals = np.diff(selected).astype(float) / NS
        result.update(recurrence_evidence_known=True, arrival_count=len(selected),
                      return_interval_count=len(intervals))
        if len(intervals):
            result["return_interval_mean_seconds"] = float(intervals.mean())
        if len(intervals) >= min_cycles:
            result.update(return_interval_cv=float(intervals.std() / intervals.mean()),
                          return_interval_cv_known=True)
    if result["occupancy_mean"] == 0:
        result["status"] = "NO_OCCUPANCY"
        return result
    if result["occupancy_variance"] <= np.finfo(float).eps ** 2:
        result["status"] = "CONSTANT_OCCUPANCY"
        return result
    frequency, power = hann_periodogram(segment, step / NS)
    power[0] = 0  # Report non-DC shape, not weighted occupancy mean.
    total = float(power.sum())
    if total <= np.finfo(float).tiny:
        result["status"] = "NO_RESOLVED_VARIATION"
        return result
    fraction = power / total
    peak = int(np.argmax(power))
    period = 1 / frequency[peak]
    cycles = width / NS / period
    positive = fraction[1:]
    entropy = -float(np.dot(positive, np.log(np.maximum(positive, np.finfo(float).tiny)))) / math.log(len(positive))
    result.update(spectrum_known=True, raw_peak_period_seconds=float(period), raw_peak_cycles=float(cycles),
                  peak_power_fraction=float(fraction[peak]),
                  main_lobe_power_fraction=float(fraction[max(1, peak-1):peak+2].sum()),
                  spectral_entropy=entropy)
    z = segment - segment.mean()
    denominator = float(np.dot(z, z))
    # Linear correlation: zero-padding here prevents circular wrap, unlike
    # fabricating unknown occupancy samples in the analysed STFT window.
    fft_size = 1 << (2 * n - 1).bit_length()
    correlation = np.fft.irfft(abs(np.fft.rfft(z, n=fft_size)) ** 2, n=fft_size)[:n] / denominator
    def autocorr(lag):
        return float(correlation[lag])
    peak_lag = round(period / (step / NS))
    if 0 < peak_lag < n:
        result["autocorr_at_peak"] = autocorr(peak_lag)
    lags = np.arange(math.ceil(minimum / step), n // min_cycles + 1)
    correlations = correlation[lags]
    best = int(np.argmax(correlations))
    result.update(autocorr_best_lag_seconds=float(lags[best] * step / NS),
                  autocorr_best_value=float(correlations[best]))
    if include_spectrum:
        result.update(frequency_hz=frequency[1:].tolist(), power_fraction=fraction[1:].tolist())
    if cycles + 1e-12 < min_cycles:
        result["status"] = "FEWER_THAN_MIN_CYCLES"
    elif period + 1e-12 < minimum / NS:
        result["status"] = "PEAK_BELOW_MIN_PERIOD"
    elif arrivals is None:
        result["status"] = "ARRIVALS_NOT_SUPPLIED"
    elif result["return_interval_count"] < min_cycles:
        result["status"] = "FEWER_THAN_MIN_RETURN_INTERVALS"
    else:
        result.update(status="DESCRIPTIVE_PERIOD_SUPPORTED", period_candidate_supported=True,
                      dominant_period_seconds=float(period))
    return result
