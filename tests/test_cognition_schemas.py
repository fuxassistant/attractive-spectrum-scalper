from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from spectrum.cognition import schemas as sm
from spectrum.cognition.representation import flow_family

NS = 10**9
DAY = 86400 * NS


def stream(days=80, per_day=12, seed=1, edge=None, link=True):
    """Synthetic origins with random coordinates; outcomes for plan 3 only.

    edge(range_value, side, day) -> extra USD; default noise around -0.7.
    Outcomes mature 30 minutes after the origin, origins are 40 minutes apart.
    """
    rng = np.random.default_rng(seed); rows = []; outcomes = {}
    for d in range(days):
        for j in range(per_day):
            oid = f'{d}:{j}'; t = d * DAY + j * 2400 * NS
            pos = rng.choice([.1, .5, .9])
            rows.append(dict(origin_id=oid, date=f'D{d:03d}', block='A' if d < days // 2 else 'B', intent_ns=t,
                             day_index=d + 1, price_ret900_sigma=rng.normal(), price_ret900_known=1,
                             flow_pressure900_low=rng.normal(0, .1), flow_pressure900_high=rng.normal(0, .1),
                             flow_pressure900_known=1, price_range1800_position=pos, price_range1800_known=1,
                             sigma600_ticks_per_sqrt_s=1., sigma3600_ticks_per_sqrt_s=1., sigma600_known=True,
                             sigma3600_known=True, kind='clock', prior_events_300s=0., event_bank_known=True,
                             minutes_since_london=j * 40., flow_depth12_imbalance=0., flow_depth12_known=1))
            label = 'low' if pos < .25 else 'high' if pos > .75 else 'mid'
            if not link:
                label = rng.choice(['low', 'mid', 'high'])
            for side in (-1, 1):
                y = rng.normal(-.7, 2.) + (edge(label, side, d) if edge else 0.)
                outcomes[(oid, 3, side)] = ('COMPLETE', float(y), t + 1800 * NS)
                outcomes[(oid, 2, side)] = ('EXCLUDED', None, None)
    frame = pd.DataFrame(rows).set_index('origin_id')
    return frame, outcomes


def low_long(label, side, day):
    return (3. if side > 0 else -3.) if label == 'low' else 0.


PARAMS = replace(sm.Params(), spawn_n=40, spawn_days=6, min_n=20, min_days=5, quarantine_days=8, spawn_t=4.)


def test_stat_day_clustered_t_uses_days_not_outcome_count():
    s = sm.Stat()
    for d in range(5):
        for _ in range(100):
            s.add(1. if d % 2 else -.5, d, float('inf'))
    assert s.n == 500 and len(s.days) == 5
    # 500 identical-within-day outcomes must not look like 500 independent ones.
    assert abs(s.t_day()) < 2


def test_decay_forgets_mean_but_keeps_history():
    s = sm.Stat(); s.add(10., 0, 1.); s.add(0., 10, 1.)
    assert s.mean < 1. and s.n == 2 and s.total == 10.


def test_program_sides():
    c = dict(trend='up', flow='sell', range='low')
    assert sm.program_side('with_trend', c) == 1 and sm.program_side('against_trend', c) == -1
    assert sm.program_side('with_flow', c) == -1 and sm.program_side('range_revert', c) == 1
    assert sm.program_side('range_break', c) == -1 and sm.program_side('short', c) == -1
    assert sm.program_side('with_trend', dict(trend='flat', flow='', range='mid')) is None


def test_planted_context_edge_is_accommodated_confirmed_and_traded():
    frame, outcomes = stream(edge=low_long)
    lib, trades = sm.run(frame, outcomes, PARAMS)
    active = [k for k in lib.schemas if k.status == 'ACTIVE' and k.parent >= 0]
    assert any(('range', 'low') in k.context and sm.program_side(k.program, dict(trend='up', flow='buy', range='low')) == 1
               for k in active)
    assert len(trades) and trades.net.sum() > 0
    # credit re-attribution: a parent with an active child decides on its remainder only
    for k in lib.schemas:
        if k.active_children:
            assert k.remainder.n < k.stat.n


def symmetric(label, side, day):
    """Zero mean for each side when the label is unobservable."""
    return {'low': 3., 'high': -3.}.get(label, 0.) * side


def test_no_link_null_world_promotes_nothing_and_never_trades():
    frame, outcomes = stream(edge=symmetric, link=False, seed=3)
    lib, trades = sm.run(frame, outcomes, PARAMS)
    assert not [k for k in lib.schemas if k.parent >= 0 and k.status == 'ACTIVE']
    assert len(trades) == 0


def test_immature_outcomes_are_invisible_to_decisions():
    frame, outcomes = stream(days=40, edge=low_long)
    # Make every outcome mature only after the stream ends: nothing can be learned.
    late = {k: (s, n, None if t is None else 10**6 * DAY) for k, (s, n, t) in outcomes.items()}
    lib, trades = sm.run(frame, late, PARAMS)
    assert len(trades) == 0 and all(k.stat.n == 0 for k in lib.schemas)


def test_frozen_learning_is_not_updated_in_later_block():
    frame, outcomes = stream(edge=low_long)
    lib, _ = sm.run(frame, outcomes, replace(PARAMS, freeze_at_block='B'))
    last_a = frame[frame.block == 'A'].day_index.max()
    assert max(max(k.stat.days, default=0) for k in lib.schemas) <= last_a


def test_flow_window_requires_same_epoch_and_available_rows():
    t = np.arange(0, 4000, dtype=np.int64) * NS
    s = pd.DataFrame(dict(t_ns=t, available_ns=t + NS // 5, epoch=np.where(t < 2000 * NS, 1, 2), part=0,
                          book_valid=True, depth_bid_1=1., depth_bid_4=1., depth_bid_12=2., depth_ask_1=1.,
                          depth_ask_4=1., depth_ask_12=1.,
                          **{c: np.arange(4000, dtype=float) for c in ('c_trade_buy', 'c_add_bid', 'c_add_ask',
                                                                      'c_remove_bid', 'c_remove_ask')},
                          c_trade_sell=np.zeros(4000), c_trade_unknown=np.zeros(4000)))
    out = flow_family(s, np.array([1500 * NS, 2100 * NS, 3999 * NS + NS // 10], np.int64))
    assert out['flow_pressure300_known'].tolist() == [True, False, True]   # 2100-300 crosses the epoch change
    assert out['flow_pressure300_low'][0] == pytest.approx(1.)
    assert out['flow_volume3600_known'].tolist() == [False, False, False]
    # the row stamped 3999s is not yet available 0.1s later: the 3998s row is used
    assert out['flow_log_volume300'][2] == pytest.approx(np.log1p(300.))
    assert out['flow_depth12_imbalance'][0] == pytest.approx(1 / 3)
