from types import SimpleNamespace
import numpy as np
import pytest
from spectrum.player.contract import Contract, NS
from spectrum.player_v3.gym import CapitalState, SurvivalGym, evaluate_policy, population_metrics, MONEY_TOLERANCE, risk_contract


def day(times, bids, asks=None, decision=(0, 60), valid=None, epoch=None, allowed=None, deadline=1000):
    t = np.asarray(times, dtype=np.int64) * NS
    b = np.rint(np.asarray(bids) * 1e9).astype(np.int64)
    a = b + 100_000_000 if asks is None else np.rint(np.asarray(asks) * 1e9).astype(np.int64)
    dt = np.asarray(decision, dtype=np.int64) * NS
    return SimpleNamespace(quote_t=t, source_t=t.copy(), bid=b, ask=a,
                           valid=np.ones(len(t), bool) if valid is None else np.asarray(valid),
                           epoch=np.zeros(len(t), np.int64) if epoch is None else np.asarray(epoch, np.int64),
                           decision_t=dt, cuts=np.searchsorted(t, dt, side='right'), x=np.zeros((len(dt), 71), np.float32),
                           can_enter=np.ones(len(dt), bool) if allowed is None else np.asarray(allowed),
                           deadline=deadline * NS, bar_t=dt.copy(), meta={'date': 'test', 'ny_close_ns': str(deadline * NS)})


def contract(**kw):
    return Contract(initial_capital=200., support_gap_ns=2000 * NS, **kw)


def test_finalized_support_does_not_censor_silence():
    d=day([0,1,10,60,61,120],[100,100,100.1,100.2,100.2,100.2],decision=(0,60,120))
    d.meta['support_contract']='FINALIZED_SEGMENTS_V2'
    e=SurvivalGym(d,Contract(initial_capital=200.,support_gap_ns=5*NS))
    e.step(3)
    assert not e.support_halted


def test_finalized_support_does_not_invent_recovery_fill():
    d=day([0,1,2,60],[100,100,1,200],valid=[True,True,False,True])
    d.meta['support_contract']='FINALIZED_SEGMENTS_V2'
    e=SurvivalGym(d,contract())
    e.step(3)
    assert e.support_halted and e.done
    assert not e.trades and not e.s[12]
    assert e.s[0]!=0
    assert e.capital_state().censored


def finish(env):
    while not env.done:
        env.step(0)


def test_capital_barrier_first_quote_and_permanent_halt():
    d = day([0, 1, 2, 3, 60, 61, 120], [100, 100, 99, 200, 200, 200, 200], decision=(0, 60, 120))
    e = SurvivalGym(d, contract(), True, CapitalState(cash=105., peak=200., series_id=1,
                     series_active=True, series_start_equity=200., series_min_equity=105., series_trade_count=10))
    e.step(3, size_oz=4.)
    assert e.trades[0]['exit_ns'] == 2 * NS
    assert e.trades[0]['reason'] == 'series_barrier'
    assert e.trades[0]['size_oz'] == 4
    assert e.f[1] == pytest.approx(99.4)
    assert e.s[12] and not e.mask()[1:].any()
    finish(e)
    np.testing.assert_allclose(e.trace[2:], 99.4)
    e2 = SurvivalGym(d, contract(), capital_state=e.capital_state())
    assert not e2.mask()[1:].any()


def test_fees_and_size_scaled_once_and_reward_telescopes():
    d = day([0, 1, 2, 60], [100, 100, 102, 102])
    e = SurvivalGym(d, contract())
    r = e.step(1, size_oz=2.)[1]
    r += e.step(0)[1]
    assert e.trades[0]['gross'] == pytest.approx(3.8)
    assert e.trades[0]['fee'] == pytest.approx(.6)
    assert e.trades[0]['net'] == pytest.approx(3.2)
    assert r == pytest.approx(e.metrics()['net'])
    assert e.f[0] == pytest.approx(203.2)


def test_daily_drawdown_rule_is_not_reused():
    d = day([0, 1, 2, 60, 61, 62, 120], [100, 100, 95, 95, 95, 97, 97], decision=(0, 60, 120))
    e = SurvivalGym(d, contract(daily_drawdown_stop=1.))
    e.step(1)
    assert e.trades[0]['reason'] == 'stop'
    assert e.mask()[1]
    e.step(1)
    finish(e)
    assert not e.metrics()['ruined']


def test_cross_day_cash_peak_and_loss_streak_carry():
    d = day([0, 1, 2, 60], [100, 100, 98, 98])
    e = SurvivalGym(d, contract())
    e.step(1)
    finish(e)
    state = e.capital_state()
    assert state.cash == pytest.approx(197.6) and state.loss_streak == 1
    assert state.series_active and state.series_start_equity == 200.
    assert state.series_trade_count == 1 and state.series_id == 1
    e2 = SurvivalGym(d, contract(), capital_state=state)
    assert e2.f[0] == pytest.approx(197.6) and e2.f[2] == 200.
    e2.step(1)
    finish(e2)
    assert e2.capital_state().cash == pytest.approx(195.2)
    assert e2.capital_state().loss_streak == 2
    assert e2.capital_state().series_trade_count == 2
    assert e2.capital_state().series_id == 1
    assert e2.capital_state().series_start_ns == state.series_start_ns


def test_support_censor_blocks_continuation_and_score():
    d = day([0, 1, 2, 3, 60], [100] * 5, valid=[True, True, False, True, True])
    e = SurvivalGym(d, contract())
    e.step(1)
    finish(e)
    state = e.capital_state()
    assert state.censored
    assert e.trades[0]['reason'] == 'support_loss'
    with pytest.raises(ValueError, match='support-censored'):
        SurvivalGym(d, contract(), capital_state=state)
    assert population_metrics([d], state, [e.metrics()])['score'] is None
    assert e.done and e.support_halted and e.i == 1
    assert state.censor_ns == 3 * NS
    assert not e.mask()[1:].any()


def test_market_terminal_is_censored_and_not_given_fake_flat_fill():
    d = day([0, 1, 60], [100] * 3)
    e = SurvivalGym(d, contract())
    e.step(1)
    finish(e)
    assert e.metrics()['open_terminal']
    assert not e.metrics()['complete']
    assert e.capital_state().censored
    assert e.capital_state().series_history[0]['series_status'] == 'censored'
    assert e.capital_state().series_history[0]['end_ns'] == str(60 * NS)


def test_market_deadline_closes_at_observed_bid_and_carries():
    d = day([0, 1, 60, 130], [100, 100, 100, 100], deadline=130)
    e = SurvivalGym(d, contract())
    e.step(1)
    finish(e)
    assert e.trades[0]['reason'] == 'session_close'
    assert e.capital_state().cash == pytest.approx(199.6)
    assert e.metrics()['complete']


def test_no_illegal_entry_and_no_stale_origin():
    d = day([0, 1, 60], [100] * 3, allowed=[False, True])
    e = SurvivalGym(d, contract())
    with pytest.raises(ValueError, match='Illegal'):
        e.step(1)
    d2 = day([0, 1, 60], [100] * 3, decision=(30, 60))
    e2 = SurvivalGym(d2, Contract(initial_capital=200., support_gap_ns=5 * NS))
    assert not e2.mask()[1:].any()


def test_wait_score_zero_and_population_loss_below_floor_not_clamped():
    d = day([0, 1, 2, 60], [100, 100, 1, 1])
    wait = evaluate_policy([d], lambda e: 0, contract())
    assert wait['score'] == 0. and wait['survival_fraction'] == 1.
    active = evaluate_policy([d], lambda e: (1, 4.) if e.mask()[1] else 0, contract())
    assert active['final_equity'] < 0
    assert active['ruined'] and active['complete']
    assert active['survival_fraction'] == pytest.approx(2 / 120)
    assert active['score'] < -10


def test_fee_alone_can_cross_barrier_at_entry():
    d = day([0, 1, 2, 60], [100] * 4, asks=[100] * 4)
    e = SurvivalGym(d, contract(), capital_state=CapitalState(cash=100.2, series_id=1,
                   series_active=True, series_start_equity=200., series_min_equity=100.2))
    e.step(1)
    assert e.trades[0]['exit_index'] == e.trades[0]['entry_index'] == 1
    assert e.trades[0]['net'] == pytest.approx(-.3)
    assert e.f[1] == pytest.approx(99.9)


def test_censored_recovery_below_barrier_does_not_certify_ruin():
    d = day([0, 1, 2, 3, 60], [100, 100, 100, 1, 1], valid=[True, True, False, True, True])
    e = SurvivalGym(d, contract(), record=True)
    e.step(3, size_oz=4.)
    assert e.done and e.support_halted
    assert e.metrics()['observed_below_barrier']
    assert not e.metrics()['ruined']
    assert e.trades[0]['reason'] == 'support_loss'
    assert np.isnan(e.trace[-1])
    m = population_metrics([d], e.capital_state(), [e.metrics()])
    assert m['score'] is None and m['lifetime_right_censored']
    assert m['survival_fraction'] == pytest.approx(3 / 120)


def test_martingale_sizes_only_after_closed_results_and_caps():
    from spectrum.player_v3.martingale import MartingaleConfig, MartingalePolicy, parameter_grid
    p = MartingalePolicy(MartingaleConfig(multiplier=2., cap_oz=4.))
    assert p.next_size == 1.
    p.on_trade({'net': -1., 'series_status': 'active'})
    assert p.next_size == 2.
    p.on_trade({'net': -1., 'series_status': 'active'})
    p.on_trade({'net': -1., 'series_status': 'active'})
    assert p.next_size == 4.
    p.on_trade({'net': 0., 'series_status': 'active'})
    assert p.next_size == 4.
    p.on_trade({'net': 1., 'series_status': 'active'})
    assert p.next_size == 4.
    p.on_trade({'net': 1., 'series_status': 'recovered'})
    assert p.next_size == 1.
    grid = parameter_grid()
    assert len(grid) == 180
    assert len({tuple(c.dict().values()) for c in grid}) == 180


def test_martingale_signal_requires_past_contiguous_candles():
    from spectrum.player_v3.martingale import MartingalePolicy
    d = day(np.arange(10) * 60, [100.] * 10, decision=np.arange(10) * 60)
    d.x[:, 8] = 1.
    d.x[:, 3] = 1.
    p = MartingalePolicy()
    assert p.signal(d)[4] == 0. and p.signal(d)[5] == 5.
    d2 = day(np.arange(10) * 60, [100.] * 10, decision=np.arange(10) * 60)
    d2.x[:, 8] = 1.
    d2.x[:, 3] = 1.
    d2.x[4, 8] = 0.
    assert np.all(p.signal(d2)[5:10] == 0.)


def test_exact_half_alive_then_next_loss_dies_first_supported_quote():
    d = day([0, 1, 2, 60, 61, 62, 63, 120],
            [100, 100, 50, 50, 50, 49.99, 100, 100],
            asks=[100, 100, 50, 50, 50, 49.99, 100, 100], decision=(0, 60, 120))
    e = SurvivalGym(d, contract(fee_roundtrip=0.), record=True)
    e.step(1, size_oz=2.)
    assert e.f[1] == 100.
    assert not e.s[12] and e.s[16] and e.mask()[1]
    assert e.trades[0]['series_status'] == 'active'
    e.step(1, size_oz=2.)
    assert e.trades[1]['reason'] == 'series_barrier'
    assert e.trades[1]['exit_ns'] == 62 * NS
    assert e.f[1] == pytest.approx(99.98)
    assert e.trades[1]['series_cumulative_net'] == pytest.approx(-100.02)
    assert e.s[12] and not e.mask()[1:].any()
    assert e.trace[2] == 100. and e.trace[5] == pytest.approx(99.98)
    assert e.capital_state().max_series_loss_fraction == pytest.approx(.5001)


def test_exact_half_and_strict_breach_are_distinguished_inside_one_position():
    prices = [100, 100, 50, 49.99, 200, 200]
    d = day([0, 1, 2, 3, 4, 60], prices, asks=prices)
    e = SurvivalGym(d, contract(fee_roundtrip=0.), record=True)
    e.plans[0] = [100000, 100000, 600]  # isolate capital first-hit from the price stop.
    e.step(1, size_oz=2.)
    assert e.trades[0]['exit_index'] == 3
    assert e.trades[0]['reason'] == 'series_barrier'
    assert e.trace[2] == 100.
    assert e.trace[3] == pytest.approx(99.98)
    assert e.trace[4] == pytest.approx(99.98)  # later recovery cannot revive a dead population.


def test_series_base_updates_after_profit_and_floor_is_not_initial_100():
    d = day([0, 1, 2, 60, 61, 62, 120], [100, 100, 200, 200, 200, 49, 49],
            asks=[100, 100, 200, 200, 200, 49, 49], decision=(0, 60, 120))
    e = SurvivalGym(d, contract(fee_roundtrip=0.), record=True)
    e.step(1)
    assert e.f[0] == 300. and not e.s[16] and e.s[19] == 1
    assert e.trades[0]['series_status'] == 'recovered'
    assert e.trades[0]['series_base'] == 200.
    assert e.floor_trace[2] == 100. and np.isnan(e.floor_trace[3])
    e.step(1)
    assert e.trades[1]['series_id'] == 2
    assert e.trades[1]['series_base'] == 300. and e.trades[1]['series_floor'] == 150.
    assert e.trades[1]['series_status'] == 'ruined'
    assert e.f[1] == 149. and e.s[12]
    assert e.floor_trace[4] == 150. and e.trace_series_id[4] == 2
    assert len(e.capital_state().series_history) == 2
    assert e.capital_state().series_history[0]['series_status'] == 'recovered'
    assert e.capital_state().series_history[1]['series_status'] == 'ruined'


def test_partial_win_does_not_reset_series_and_recovery_requires_all_fees():
    prices = [100, 100, 90.3, 90.3, 90.3, 94.6, 94.6, 94.6, 100.9, 100.9, 100.9, 103, 103]
    d = day([0, 1, 2, 60, 61, 62, 120, 121, 122, 180, 181, 182, 240], prices,
            asks=prices, decision=(0, 60, 120, 180, 240))
    e = SurvivalGym(d, contract(), record=True)
    e.step(1)
    assert e.f[0] == pytest.approx(190.)
    e.step(1)
    assert e.f[0] == pytest.approx(194.) and e.s[16]
    assert e.s[17] == 1 and e.s[19] == 0
    assert e.trades[1]['net'] == pytest.approx(4.)
    assert e.trades[1]['series_net'] == pytest.approx(-6.)
    assert not e.trades[1]['series_recovered']
    assert e.trades[1]['series_base'] == 200.
    e.step(1)
    assert e.f[0] == pytest.approx(200.) and not e.s[16] and e.s[19] == 1
    assert e.trades[2]['series_recovered']
    assert e.trades[2]['series_trade_count'] == 3
    assert e.trades[2]['series_start_ns'] == NS
    e.step(1)
    assert e.trades[3]['series_id'] == 2 and e.trades[3]['series_base'] == pytest.approx(200.)
    assert e.trades[3]['series_start_ns'] == 181 * NS
    assert e.trades[3]['equity_before'] == pytest.approx(200.)


def test_first_actual_fill_starts_series_cancelled_decision_does_not():
    d = day([0, 10, 60], [100, 100, 100])
    e = SurvivalGym(d, contract(), record=True)
    assert e.risk_snapshot()['series_status'] == 'none'
    e.step(1)
    assert e.cancelled == 1 and e.s[17] == 0
    assert e.capital_state().series_history == []
    assert np.isnan(e.floor_trace).all()
    d2 = day([0, 1, 2, 60], [100, 100, 102, 102])
    e2 = SurvivalGym(d2, contract(), capital_state=CapitalState(cash=300., peak=300.))
    e2.step(1)
    assert e2.trades[0]['series_start_ns'] == NS
    assert e2.trades[0]['series_base'] == 300.


def test_censor_cannot_certify_recovery_even_after_large_diagnostic_win():
    d = day([0, 1, 2, 3, 60], [100, 100, 100, 200, 200], valid=[True, True, False, True, True])
    e = SurvivalGym(d, contract())
    e.step(1)
    assert e.done and e.support_halted and not e.s[12]
    assert e.trades[0]['net'] > 0
    assert e.trades[0]['series_status'] == 'censored'
    assert e.s[16] and e.s[19] == 0
    assert e.capital_state().series_history[0]['series_status'] == 'censored'
    assert e.capital_state().max_series_loss_fraction < .01


@pytest.mark.parametrize('cash,dead', [(100., False), (100. - 2e-12, False), (100. - 1e-5, True)])
def test_strict_half_boundary_has_only_explicit_tiny_money_tolerance(cash, dead):
    d = day([0, 1, 60], [100, 100, 100], asks=[100, 100, 100])
    state = CapitalState(cash=cash, series_id=7, series_active=True,
                         series_start_equity=200., series_min_equity=cash)
    e = SurvivalGym(d, contract(fee_roundtrip=0.), capital_state=state)
    e.step(1)
    assert bool(e.s[12]) is dead
    assert risk_contract()['money_tolerance_usd'] == MONEY_TOLERANCE == 1e-8


def test_new_series_below_initial_100_is_allowed_and_has_own_floor():
    d = day([0, 1, 2, 60], [100, 100, 99, 99], asks=[100, 100, 99, 99])
    e = SurvivalGym(d, contract(fee_roundtrip=0.), capital_state=CapitalState(cash=90., peak=90.))
    assert e.mask()[1]
    e.step(1)
    assert e.trades[0]['series_base'] == 90. and e.trades[0]['series_floor'] == 45.
    assert e.f[1] == 89. and not e.s[12]
    assert len(e.account()) == e.account_dim == 11
    np.testing.assert_allclose(e.account()[7:], [1., -.55, 1 / 90, .01], rtol=1e-6)


def test_objective_uses_supported_series_fraction_not_lifetime_dollar_drawdown():
    d = day([0, 60], [100, 100])
    state = CapitalState(cash=300., peak=500., max_drawdown=200., max_series_loss_fraction=.1)
    m = population_metrics([d], state, [{'complete': True}])
    assert m['score'] == pytest.approx(np.log(1.5) - .25 * .1)
    assert m['max_drawdown'] == 200.
