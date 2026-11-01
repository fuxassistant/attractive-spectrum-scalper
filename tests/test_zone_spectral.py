"""Synthetic controls and clock/support checks for absolute-band spectra."""
import numpy as np
import pytest

from experiments.zone_episode_lab.spectral import hann_periodogram, zone_spectrum

NS = 1_000_000_000


def series(length=3600, step=5):
    return np.arange(step, length + step, step, dtype=np.int64) * NS


def periodic(length=3600, period=300, step=5):
    times = series(length, step)
    occupancy = ((times // NS - step) % period < period / 4).astype(float)
    arrivals = np.arange(0, length, period, dtype=np.int64) * NS
    return times, occupancy, arrivals


def run(times, x, arrivals=None, **kwargs):
    return zone_spectrum(times, x, int(times[-1]), arrival_ns=arrivals, **kwargs)


def test_periodic_occupancy_and_exact_arrivals_agree():
    t, x, a = periodic()
    r = run(t, x, a, include_spectrum=True)
    assert r["known"] and r["spectrum_known"] and r["period_candidate_supported"]
    assert r["dominant_period_seconds"] == pytest.approx(300)
    assert r["raw_peak_cycles"] == pytest.approx(12)
    assert r["main_lobe_power_fraction"] > .5
    assert r["return_interval_count"] == 11
    assert r["return_interval_mean_seconds"] == 300
    assert r["return_interval_cv"] == 0
    assert r["autocorr_at_peak"] == pytest.approx(11 / 12)
    assert sum(r["power_fraction"]) == pytest.approx(1)
    assert r["periodicity_significance"] is None
    assert r["economic_admission"] == "NOT_ESTABLISHED"


def test_one_pass_cannot_be_named_cycle_even_when_spectrum_is_concentrated():
    t = series()
    x = np.zeros(len(t))
    x[180:540] = 1
    r = run(t, x, np.array([900 * NS], dtype=np.int64))
    assert r["known"] and r["spectrum_known"]
    assert r["main_lobe_power_fraction"] > .7
    assert r["arrival_count"] == 1
    assert r["return_interval_count"] == 0
    assert not r["period_candidate_supported"]
    assert r["dominant_period_seconds"] is None
    assert r["return_interval_cv"] is None


def test_random_walk_control_is_descriptive_and_not_a_significance_test():
    rng = np.random.default_rng(811)
    t = series()
    price = np.cumsum(rng.normal(size=len(t)))
    x = (abs(price) < 2).astype(float)
    a = (np.flatnonzero((x[1:] > 0) & (x[:-1] == 0)) + 1) * 5 * NS
    r = run(t, x, a)
    assert r["known"] and r["spectrum_known"]
    assert r["return_interval_cv"] > .5
    assert r["descriptive_only"] and r["periodicity_significance"] is None
    if r["period_candidate_supported"]:
        assert r["raw_peak_cycles"] >= 3 and r["return_interval_count"] >= 3


@pytest.mark.parametrize("damage", ["missing", "unknown", "nan", "epoch"])
def test_interrupted_support_is_not_zero_filled(damage):
    t, x, a = periodic()
    known = np.ones(len(t), dtype=bool)
    epoch = np.zeros(len(t), dtype=np.int64)
    if damage == "missing":
        t, x, known, epoch = [np.delete(v, 333) for v in (t, x, known, epoch)]
    elif damage == "unknown":
        known[333] = False
    elif damage == "nan":
        x[333] = np.nan
    else:
        epoch[333:] = 1
    r = run(t, x, a, known=known, epoch=epoch)
    assert not r["known"] and not r["spectrum_known"]
    assert r["dominant_period_seconds"] is None
    assert r["arrival_count"] is None
    assert r["status"] == ("EPOCH_BOUNDARY" if damage == "epoch" else "INCOMPLETE_WINDOW")


def test_prefix_causality_including_later_arrivals_and_unknown_future_bins():
    t, x, a = periodic(length=7200)
    q = 3600 * NS
    known = np.ones(len(t), bool)
    known[720:] = False
    x[720:] = np.nan
    epoch = np.zeros(len(t), np.int64)
    epoch[720:] = 99
    full = zone_spectrum(t, x, q, known=known, epoch=epoch, arrival_ns=a, include_spectrum=True)
    prefix = zone_spectrum(t[:720], x[:720], q, known=known[:720], epoch=epoch[:720],
                           arrival_ns=a[a < q], include_spectrum=True)
    assert full == prefix


def test_query_between_grid_boundaries_does_not_consume_partial_bin():
    t, x, a = periodic(length=3610)
    q = 3602 * NS
    r = zone_spectrum(t, x, q, arrival_ns=a)
    assert r["window_end_ns"] == 3600 * NS
    assert r["available_ns"] <= q
    expected = run(t[:720], x[:720], a[a < 3600 * NS])
    assert {k: v for k, v in r.items() if k != "query_ns"} == {k: v for k, v in expected.items() if k != "query_ns"}


def test_confirmed_later_arrival_cannot_backdate_cycle_evidence():
    t, x, _ = periodic()
    arrivals = np.array([100, 1000, 2000, 3599], np.int64) * NS
    availability = arrivals + 3 * NS
    r = run(t, x, arrivals, arrival_available_ns=availability)
    assert r["arrival_count"] == 3
    assert r["return_interval_count"] == 2
    assert not r["period_candidate_supported"]
    assert r["return_interval_cv"] is None


def test_out_of_order_confirmation_cannot_stitch_across_pending_contact():
    t, x, _ = periodic()
    arrivals = np.array([100, 500, 1000, 1500, 2000], np.int64) * NS
    availability = np.array([100, 4000, 1000, 1500, 2000], np.int64) * NS
    with pytest.raises(ValueError, match="FIFO"):
        run(t, x, arrivals, arrival_available_ns=availability)


def test_unknown_negative_epoch_never_counts_as_continuous_support():
    t, x, a = periodic()
    r = run(t, x, a, epoch=np.full(len(t), -1, dtype=np.int64))
    assert not r["known"] and r["coverage_fraction"] == 0
    assert r["return_interval_count"] is None


def test_three_complete_return_intervals_required_and_lomb_of_ones_not_invented():
    t, x, a = periodic()
    no_events = run(t, x)
    two_intervals = run(t, x, a[:3])
    three_intervals = run(t, x, a[:4])
    assert no_events["status"] == "ARRIVALS_NOT_SUPPLIED"
    assert no_events["arrival_count"] is None
    assert two_intervals["status"] == "FEWER_THAN_MIN_RETURN_INTERVALS"
    assert three_intervals["period_candidate_supported"]


@pytest.mark.parametrize("value,status", [(0., "NO_OCCUPANCY"), (1., "CONSTANT_OCCUPANCY")])
def test_observed_constant_and_unknown_are_distinct(value, status):
    t = series()
    r = run(t, np.full(len(t), value), np.array([], np.int64))
    assert r["known"] and not r["spectrum_known"]
    assert r["status"] == status and r["occupancy_mean"] == value
    assert r["arrival_count"] == 0 and r["return_interval_cv"] is None


@pytest.mark.parametrize("n", [119, 120])
def test_periodogram_matches_independent_direct_dft_and_one_sided_scaling(n):
    sample_seconds = 5
    x = np.random.default_rng(n).uniform(size=n)
    frequency, psd = hann_periodogram(x, sample_seconds)
    w = .5 - .5 * np.cos(2 * np.pi * np.arange(n) / n)
    dft = np.array([np.dot((x-x.mean())*w, np.exp(-2j*np.pi*k*np.arange(n)/n)) for k in range(n//2+1)])
    expected = abs(dft)**2 * sample_seconds / np.dot(w, w)
    expected[1:-1 if n % 2 == 0 else None] *= 2
    assert psd == pytest.approx(expected, rel=1e-10, abs=1e-12)
    assert frequency == pytest.approx(np.arange(n//2+1) / (n*sample_seconds))
    assert psd.sum() / (n*sample_seconds) == pytest.approx(np.dot((x-x.mean())*w, (x-x.mean())*w) / np.dot(w,w))


def test_biased_autocorrelation_has_no_circular_wrap():
    t, x, a = periodic()
    r = run(t, x, a)
    z = x - x.mean()
    lag = round(r["raw_peak_period_seconds"] / 5)
    assert r["autocorr_at_peak"] == pytest.approx(np.dot(z[:-lag], z[lag:]) / np.dot(z,z))
    assert r["autocorr_at_peak"] < 1


@pytest.mark.parametrize("bad", ["float_clock", "off_grid", "duplicate", "out_of_range", "too_fast"])
def test_invalid_input_clock_and_support_are_explicit(bad):
    t, x, a = periodic()
    if bad == "float_clock": t = t.astype(float)
    if bad == "off_grid": t = t + 1
    if bad == "duplicate": t[10] = t[9]
    if bad == "out_of_range": x[10] = 1.1
    if bad == "too_fast":
        x = (np.arange(len(t)) % 4 < 2).astype(float)
        r = run(t, x, a)
        assert r["status"] == "PEAK_BELOW_MIN_PERIOD"
    elif bad == "out_of_range":
        r = run(t, x, a)
        assert not r["known"] and r["coverage_fraction"] < 1
    else:
        with pytest.raises(ValueError): run(t, x, a)


def native_tape(clocks, midpoint):
    import pandas as pd
    p = np.rint(np.asarray(midpoint)*NS).astype(np.int64)
    return pd.DataFrame(dict(available_ns=np.asarray(clocks, np.int64),
                             bid_nano=p-50_000_000, ask_nano=p+50_000_000,
                             valid=True, epoch=1,
                             source_ordinal=np.arange(len(p), dtype=np.int64)))


def test_native_dense_full_occupancy_stays_exactly_one_and_spectral_known():
    from experiments.zone_episode_lab.episodes import occupancy_grid
    # The old per-update seconds accumulator overshot 1 by ~7e-16 on this tape.
    subdivisions = np.arange(201, dtype=np.int64)*(5*NS)//200
    assert np.cumsum(np.diff(subdivisions)/NS)[-1]/5 > 1
    t = np.arange(60*200//5+1, dtype=np.int64)*(5*NS)//200
    q = native_tape(t, np.full(len(t), 100.5))
    g = occupancy_grid(q, start_ns=0, end_ns=60*NS)
    assert np.array_equal(g["occupancy"], np.ones((12, 1)))
    assert np.array_equal(g["support_seconds"], np.full(12, 5.))
    assert g["known"].all()
    r = zone_spectrum(g["times_ns"], g["occupancy"][:, 0], 60*NS,
                      window_seconds=60, min_period_seconds=10,
                      known=g["known"], epoch=g["epoch"], arrival_ns=[])
    assert r["known"] and r["n_known"] == 12
    assert r["status"] == "CONSTANT_OCCUPANCY"


def test_exact_native_occupancy_period_survives_grid_and_future_extension():
    from experiments.zone_episode_lab.episodes import occupancy_grid
    # Dense updates in first10s, ordinary updates thereafter, absolute nanoclock.
    base = 1_787_529_600*NS
    offsets = np.unique(np.r_[np.arange(401, dtype=np.int64)*(5*NS)//200,
                              np.arange(0, 7205, 5, dtype=np.int64)*NS])
    price = np.where((offsets//NS) % 300 < 75, 100.5, 102.5)
    q = native_tape(base+offsets, price)
    early = occupancy_grid(q, start_ns=base, end_ns=base+3600*NS)
    q.loc[q.available_ns.gt(base+3600*NS), "epoch"] = 5
    later = occupancy_grid(q, start_ns=base, end_ns=base+7200*NS)
    arrivals = base+np.arange(0, 7200, 300, dtype=np.int64)*NS
    def evaluate(grid):
        col = int(np.flatnonzero(grid["price_bin_nano"] == 100*NS)[0])
        assert np.all((grid["occupancy"] >= 0) & (grid["occupancy"] <= 1))
        return zone_spectrum(grid["times_ns"], grid["occupancy"][:, col], base+3600*NS,
                             known=grid["known"], epoch=grid["epoch"], arrival_ns=arrivals)
    a, b = evaluate(early), evaluate(later)
    assert a == b
    assert a["known"] and a["period_candidate_supported"]
    assert a["dominant_period_seconds"] == pytest.approx(300)
    assert a["return_interval_cv"] == 0


def test_exact_integration_does_not_hide_a_true_native_coverage_gap():
    from experiments.zone_episode_lab.episodes import occupancy_grid
    clocks = np.arange(0, 3605, 5, dtype=np.int64)*NS
    clocks = clocks[(clocks < 1200*NS) | (clocks >= 1270*NS)]
    q = native_tape(clocks, np.where((clocks//NS)%300 < 75, 100.5, 102.5))
    g = occupancy_grid(q, start_ns=0, end_ns=3600*NS)
    col = int(np.flatnonzero(g["price_bin_nano"] == 100*NS)[0])
    r = zone_spectrum(g["times_ns"], g["occupancy"][:, col], 3600*NS,
                      known=g["known"], epoch=g["epoch"],
                      arrival_ns=np.arange(0, 3600, 300, dtype=np.int64)*NS)
    assert not r["known"] and r["status"] == "INCOMPLETE_WINDOW"
    assert 0 < r["coverage_fraction"] < 1
