"""A profitable leg is not necessarily a recovered martingale series."""
import pytest
from spectrum.player_v3.martingale import MartingaleConfig, MartingalePolicy, cycle_rows, parameter_grid


def leg(net, status='active', series=1):
    return {'net': net, 'series_status': status, 'series_id': series}


def test_partial_win_keeps_escalated_size_until_whole_series_recovers():
    policy = MartingalePolicy(MartingaleConfig(multiplier=2., cap_oz=8.))
    policy.on_trade(leg(-10.))
    assert policy.next_size == 2.
    policy.on_trade(leg(4.))
    assert policy.next_size == 2.  # Series is still down $6.
    policy.on_trade(leg(-5.))
    assert policy.next_size == 4.
    policy.on_trade(leg(11., 'recovered'))
    assert policy.next_size == 1.
    assert policy.loss_streak == 0


def test_zero_leg_and_cap_do_not_imply_series_recovery():
    policy = MartingalePolicy(MartingaleConfig(multiplier=2., cap_oz=2.))
    for net in (-1., -1., 0., .1):
        policy.on_trade(leg(net))
        assert policy.next_size == 2.
    policy.on_trade(leg(1.9, 'recovered'))
    assert policy.next_size == 1.


def test_series_metadata_required_instead_of_old_trade_sign_heuristic():
    policy = MartingalePolicy()
    with pytest.raises(KeyError, match='series_status'):
        policy.on_trade({'net': 1.})
    with pytest.raises(ValueError, match='Unknown martingale series status'):
        policy.on_trade(leg(1., 'won'))


def test_new_population_reset_is_explicit_and_base_size_can_be_configured():
    policy = MartingalePolicy(MartingaleConfig(base_oz=2., multiplier=2., cap_oz=8.))
    policy.on_trade(leg(-2.))
    policy.on_trade(leg(1.))
    assert policy.next_size == 4.
    policy.reset()
    assert policy.next_size == 2.
    assert policy.series_id is None


def test_grid_is_same_180_fixed_policies():
    configs = parameter_grid()
    assert len(configs) == 180
    assert len({tuple(c.dict().values()) for c in configs}) == 180
    assert all(c.base_oz == 1. for c in configs)


def test_cycle_export_keeps_partial_wins_in_same_series_across_dates():
    common = dict(series_id=1, series_start_equity=200., series_floor=100.,
                  series_start_ns=1_000_000_000)
    trades = [
        dict(common, exit_ns=2_000_000_000, net=-10., series_cumulative_net=-10.,
             series_min_equity=189., series_loss_fraction=.055, series_status='active', date='2026-05-18'),
        dict(common, exit_ns=86_402_000_000_000, net=4., series_cumulative_net=-6.,
             series_min_equity=189., series_loss_fraction=.055, series_status='active', date='2026-05-19'),
        dict(common, exit_ns=86_403_000_000_000, net=7., series_cumulative_net=1.,
             series_min_equity=189., series_loss_fraction=.055, series_status='recovered', date='2026-05-19'),
        dict(series_id=2, series_start_equity=201., series_floor=100.5,
             series_start_ns=86_404_000_000_000, exit_ns=86_405_000_000_000,
             net=-101., series_cumulative_net=-101., series_min_equity=100.,
             series_loss_fraction=101/201, series_status='ruined', date='2026-05-19')]
    rows = cycle_rows(trades)
    assert len(rows) == 2
    assert rows[0]['trades'] == 3 and rows[0]['wins'] == 2
    assert rows[0]['last_flat_equity'] == 201.
    assert rows[0]['status'] == 'recovered'
    assert rows[0]['first_date'] != rows[0]['last_date']
    assert rows[1]['start_equity'] == 201. and rows[1]['floor'] == 100.5
    assert rows[1]['status'] == 'ruined'


def test_cycle_export_rejects_moving_base_inside_same_series():
    common = dict(series_id=1, series_floor=100., series_start_ns=1_000_000_000,
                  exit_ns=2_000_000_000, net=-1., series_cumulative_net=-1.,
                  series_min_equity=199., series_loss_fraction=.005, series_status='active')
    with pytest.raises(ValueError, match='Series base changed'):
        cycle_rows([dict(common, series_start_equity=200.), dict(common, series_start_equity=199.)])
