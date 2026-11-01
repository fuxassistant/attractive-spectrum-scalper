import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from experiments.scalper_adapter_v1.tokens import motion_tokens, zone_tokens, NS, TICK
from experiments.scalper_adapter_v1.adapter import build_session, ScalperAdapter, live_states
from experiments.scalper_adapter_v1.measures import path_measures, enrich, encode, FlowView, annotate_zone_history
from experiments.scalper_adapter_v1.similarity import ordered_distance, descriptors

CFG = json.loads((Path(__file__).parents[1]/'experiments/scalper_adapter_v1/config.json').read_text())
SCALE = dict(CFG['scales'][0], amplitude_ticks=10, time_limit_seconds=20,
             update_seconds=2, zone_ticks=10, departure_ticks=10, confirmation_seconds=3)


def tape(times, prices, valid=None, epoch=None):
    p = np.rint(np.array(prices)*NS).astype('int64')
    return pd.DataFrame(dict(available_ns=np.array(times, dtype='int64')*NS,
        bid_nano=p-50_000_000, ask_nano=p+50_000_000,
        valid=np.ones(len(p), bool) if valid is None else np.array(valid, bool),
        epoch=np.ones(len(p), dtype='int64') if epoch is None else np.array(epoch, dtype='int64'),
        source_ordinal=np.arange(len(p), dtype='int64')))


def session(q, date='today', previous='yesterday', start=0, end=200, through=None, scales=None):
    return build_session(q, session_id=date, previous_session_id=previous, symbol='TEST',
        start_ns=start*NS, end_ns=end*NS, observed_through_ns=(through*NS if through is not None else None),
        scales=[SCALE] if scales is None else scales)


def test_turn_is_published_at_reversal_not_at_extreme():
    q = tape([0,1,2,3,4,5], [100,101,102,101.7,101.3,101])
    m = motion_tokens(q, 's', SCALE, end_ns=5*NS, session_end_ns=200*NS)
    turn = m[m.kind.eq('TURN_CONFIRMED')].iloc[0]
    assert turn.available_ns == 5*NS
    assert turn.extreme_ns == 2*NS and turn.extreme_nano == 102*NS
    assert not m[(m.available_ns < 5*NS) & m.kind.eq('TURN_CONFIRMED')].shape[0]


def test_motion_prefix_is_immutable_and_has_no_artificial_eof_censor():
    q = tape(range(10), [100,101,102,101.7,101.3,101,100,101,102,99])
    a = motion_tokens(q, 's', SCALE, end_ns=9*NS, session_end_ns=200*NS)
    b = motion_tokens(q.iloc[:6], 's', SCALE, end_ns=5*NS, session_end_ns=200*NS)
    assert_frame_equal(a[a.available_ns <= 5*NS].reset_index(drop=True), b)
    assert not b.kind.eq('CENSORED').any()


def test_zone_prefix_retains_forming_and_removes_only_terminal_censor():
    q = tape([0,1,2,3,5,6], [99.5,100.5,102.2,102.3,102.3,100.5])
    a = zone_tokens(q, 's', SCALE, end_ns=6*NS, session_end_ns=200*NS)
    b = zone_tokens(q.iloc[:4], 's', SCALE, end_ns=3*NS, session_end_ns=200*NS)
    assert_frame_equal(a[a.available_ns <= 3*NS].reset_index(drop=True), b)
    assert b.kind.eq('GATE_START').any() and not b.kind.eq('DEPARTURE_CONFIRMED').any()


def test_same_quote_side_switch_aborts_then_starts_gate():
    q = tape([0,1,2,3,4], [100.5,102.2,98.8,98.8,98.8])
    z = annotate_zone_history(zone_tokens(q, 's', SCALE, end_ns=4*NS, session_end_ns=200*NS))
    gate = z[(z.zone_low_nano == 100*NS) & (z.available_ns == 2*NS) & z.kind.str.startswith('GATE')]
    assert list(gate.kind)[-2:] == ['GATE_ABORT','GATE_START']
    assert gate.pending_start_ns.iloc[-1] == 2*NS


@pytest.mark.parametrize('mode', ['invalid','epoch','gap'])
def test_unknown_intervals_break_stories(mode):
    times = [0,1,2,3] if mode != 'gap' else [0,1,100,101]
    q = tape(times, [100,101,110,111], valid=[True,True,False,True] if mode == 'invalid' else None,
             epoch=[1,1,2,2] if mode == 'epoch' else None)
    s = session(q, through=times[-1])
    censored = s.motion[s.motion.kind.eq('CENSORED')]
    assert len(censored) >= 1
    assert int(censored.available_ns.iloc[0]) == (31 if mode == 'gap' else 2)*NS
    assert s.motion.segment.nunique() == 2
    assert s.zones.kind.eq('PRESENT_ON_RESTART').sum() >= 2


def test_timer_precedes_quote_at_same_clock():
    q = tape([0,2], [100,105])
    m = motion_tokens(q, 's', SCALE, end_ns=2*NS, session_end_ns=200*NS)
    progress = m[m.kind.eq('PROGRESS')].iloc[0]
    direction = m[m.kind.eq('DIRECTION_SET')].iloc[0]
    assert progress.mid_nano == 100*NS and progress.source_index == 0
    assert direction.mid_nano == 105*NS and direction.source_index == 1
    enriched = enrich(m, q)
    assert enriched.loc[m.kind.eq('PROGRESS'), 'net_ticks'].iloc[0] == 0


def test_duration_timeout_is_distinct_from_turn():
    q = tape([0,10,20,25], [100,100,100,100])
    m = motion_tokens(q, 's', SCALE, end_ns=25*NS, session_end_ns=200*NS)
    assert m.kind.eq('TIME_LIMIT').sum() == 1
    assert not m.kind.eq('TURN_CONFIRMED').any()
    assert m[m.kind.eq('TIME_LIMIT')].available_ns.iloc[0] == 20*NS


def test_path_time_measures_on_known_hand_example():
    q = tape([0,1,2,3,4], [100,100.5,100,99.5,100])
    s = session(q, through=4)
    state = live_states(s, 4*NS)
    m = enrich(state[state.family.eq('motion')], q).iloc[0]
    assert m.net_ticks == 0 and m.path_ticks == 20 and m.range_ticks == 10
    assert m.up_ticks == 10 and m.down_ticks == 10 and m.direction_changes == 2
    assert m.above_start_fraction == .25 and m.below_start_fraction == .25
    assert m.shape_q1 == .5 and m.shape_q3 == -.5
    assert m.is_friction == 1 and m.path_known


def test_equal_aggregate_opposite_order_has_different_shape_and_similarity():
    vectors = []
    for prices in ([100,100.5,100,99.5,100], [100,99.5,100,100.5,100]):
        q = tape(range(5), prices)
        s = session(q, through=4)
        active = live_states(s, 4*NS)
        m = active[active.family.eq('motion')]
        vectors.append(descriptors(m, q)[0])
        measured = enrich(m, q).iloc[0]
        assert measured.path_ticks == 20 and measured.net_ticks == 0 and measured.range_ticks == 10
    assert not np.array_equal(*vectors)
    a = np.stack([vectors[0], vectors[1], vectors[0]])
    b = np.stack([vectors[1], vectors[0], vectors[0]])
    d = ordered_distance(a, np.stack([a,b]), [True]*3)
    assert d[0] == 0 and d[1] > 0


def test_separate_speed_scales_produce_distinct_segmentation():
    q = tape(range(301), 100+np.sin(np.arange(301)/20)*2+np.sin(np.arange(301)/2)*.6)
    s = session(q, through=300, end=400, scales=CFG['scales'])
    counts = s.motion.groupby('scale').story_id.nunique()
    assert counts['fast'] > counts['middle'] > counts['slow']


def test_previous_session_retained_and_active_priority_and_causal_links():
    old = session(tape(range(41), 100+np.sin(np.arange(41)/3)), date='yesterday', previous='older', end=40)
    new = session(tape(range(60,161), 100+np.sin(np.arange(101)/3)), start=60, through=160)
    adapter = ScalperAdapter([old,new], CFG)
    o = adapter.observe('today', 160*NS)
    assert o['receipt']['previous_retained'] > 0 and o['receipt']['active_retained'] > 0
    assert o['tokens'].available_ns.max() <= 160*NS
    assert o['tokens'].token_id.is_unique
    ids = set(o['tokens'].token_id)
    assert set(o['tokens'].next_selected_observed_update_id)-{''} <= ids
    assert not set(old.motion.story_id) & set(new.motion.story_id)
    assert set(o['price_memory'].session_id) == {'yesterday','today'}
    for m in o['similarity']:
        assert m['available_ns'] < m['query_context_start_ns'] <= m['query_available_ns']


def test_whole_observation_prefix_invariance():
    q = tape(range(101), 100+np.sin(np.arange(101)/4)*2)
    full = session(q, through=100)
    prefix = session(q[q.available_ns<=55*NS], through=55)
    a = ScalperAdapter([full], CFG).observe('today', 55*NS)
    b = ScalperAdapter([prefix], CFG).observe('today', 55*NS)
    assert_frame_equal(a['tokens'], b['tokens'])
    assert_frame_equal(a['price_memory'], b['price_memory'])
    assert a['similarity'] == b['similarity']
    assert np.array_equal(a['model']['values'], b['model']['values'])


def test_no_previous_session_claim_without_loaded_data():
    s = session(tape([0,1,2], [100,100.5,100]), through=2)
    o = ScalperAdapter([s], CFG).observe('today', 2*NS)
    assert not o['receipt']['previous_session_loaded']
    assert o['receipt']['previous_retained'] == 0


def test_flow_publication_and_unknown_side_bounds():
    f = pd.DataFrame(dict(t_ns=[NS,2*NS], available_ns=[2*NS,4*NS], epoch=[1,1],
                         trade_known=[True,True], ledger_known=[True,False]))
    from experiments.scalper_adapter_v1.measures import FLOW_FIELDS
    for field in FLOW_FIELDS: f[field] = [0.,0.]
    f['trade_buy'] = [6.,1000.]; f['trade_sell'] = [2.,0.]; f['trade_unknown'] = [2.,0.]
    view = FlowView(f)
    result = view.measure(pd.Series(dict(available_ns=2*NS, flow_seconds=1)))
    assert result['trade_buy'] == 6
    assert result['flow_imbalance_low'] == .2 and result['flow_imbalance_high'] == .6
    result = view.measure(pd.Series(dict(available_ns=4*NS, flow_seconds=1)))
    assert np.isnan(result['trade_buy'])  # bucket too old; do not carry a stale activity total


def test_zero_duration_invalid_publication_is_not_hidden():
    q = tape([0,1,1,2], [100,100,100,101], [True,False,True,True])
    s = session(q, through=2)
    assert s.motion.kind.eq('CENSORED').sum() == 1 and s.motion.segment.nunique() == 2


def test_clock_and_cross_contract_validation():
    s = session(tape([0,1], [100,100]), through=1)
    with pytest.raises(ValueError, match='Integer'):
        ScalperAdapter([s], CFG).observe('today', float(NS))
    other = session(tape([0,1], [100,100]), date='other', through=1)
    other.symbol = 'OTHER'
    with pytest.raises(ValueError, match='cross-contract'):
        ScalperAdapter([s,other], CFG)


def test_missing_flow_is_masked_not_taught_as_zero():
    q = tape([0,1,2], [100,100.5,100])
    s = session(q, through=2)
    e = enrich(live_states(s, 2*NS), q)
    model = encode(e)
    j = model['names'].index('trade_buy')
    assert not model['known'][:,j].any() and not model['values'][:,j].any()


def test_translation_preserves_motion_similarity_but_not_absolute_address():
    q = tape(range(31), 100+np.sin(np.arange(31)/3)*2)
    shifted = q.copy()
    shifted['bid_nano'] += 1000*NS; shifted['ask_nano'] += 1000*NS
    a = session(q, through=30); b = session(shifted, through=30)
    assert np.array_equal(descriptors(a.motion, q), descriptors(b.motion, shifted))
    assert (b.zones.zone_low_nano-a.zones.zone_low_nano).eq(1000*NS).all()


def test_large_nanosecond_clocks_and_simultaneous_native_publications():
    base = 1787659200000000000
    q = tape([0,1,1,2,3], [100,100.5,101.5,102,101])
    q['available_ns'] += base+123
    m = motion_tokens(q, 's', SCALE, end_ns=base+3*NS+123, session_end_ns=base+100*NS)
    assert m.available_ns.dtype.kind in 'iu'
    assert (m.available_ns.to_numpy()%NS == 123).all()
    direction = m[m.kind.eq('DIRECTION_SET')].iloc[0]
    assert direction.source_index == 2 and direction.available_ns == base+NS+123


def test_completed_history_reserve_survives_tiny_recent_budget():
    q = tape(range(301), 100+np.sin(np.arange(301)/5)*2)
    s = session(q, through=300, end=400)
    cfg = dict(CFG, retention=dict(CFG['retention'], current_updates_per_family_scale=1))
    o = ScalperAdapter([s], cfg).observe('today', 300*NS)
    assert o['receipt']['current_completed_reserved'] == 16
    for _, f in o['tokens'].groupby('family'):
        assert f.phase.isin(['CONFIRMED','TIME_LIMIT']).sum() >= 8


def test_returned_historical_neighbors_are_actual_model_input():
    q = tape(range(301), 100+np.sin(np.arange(301)/5)*2)
    s = session(q, through=300, end=400)
    o = ScalperAdapter([s], CFG).observe('today', 300*NS)
    assert len(o['similarity']) > 0
    for match in o['similarity']:
        assert set(match['context_token_ids']).issubset(set(o['tokens'].token_id))


def test_live_state_updates_extreme_between_periodic_publications():
    q = tape([0,1,2,3], [100,101,101.1,101.8])
    s = session(q, through=3)
    r = live_states(s, 3*NS)
    r = r[r.family.eq('motion')].iloc[0]
    assert r.extreme_nano == int(101.8*NS) and r.extreme_ns == 3*NS
    assert r.up_nano == 18*TICK and r.path_nano == 18*TICK


def test_live_zone_state_uses_current_side_after_small_cross_zone_jump():
    q = tape([0,1,2], [100.5,99.5,101.5])
    s = session(q, through=2)
    r = live_states(s, 2*NS)
    r = r[(r.family=='zone') & (r.zone_low_nano==100*NS)].iloc[0]
    assert r.direction == 1


def test_combined_channels_preserve_source_order_at_identical_nanoseconds():
    q = tape([0,1,1,2,2,3], [100.5,101.5,103.5,100.5,98.5,100.5])
    s = session(q, through=3)
    o = ScalperAdapter([s], CFG).observe('today', 3*NS)
    for _, same_time in o['tokens'].groupby('available_ns'):
        assert same_time.source_index.is_monotonic_increasing


def test_nullable_event_clocks_never_transit_through_float64():
    base = 1787659200000000123
    q = tape([0,1,2,3,4,5], [99.5,100.5,102.5,102.2,102.4,100.5])
    q['available_ns'] += base
    z = annotate_zone_history(zone_tokens(q,'s',SCALE,end_ns=base+5*NS,session_end_ns=base+100*NS))
    assert z.kind.eq('JUMP_OVER').any()
    assert (z.start_ns.to_numpy()%NS == 123).all()
    pending = z[z.pending_start_ns>=0].pending_start_ns.to_numpy(np.int64)
    assert len(pending) and (pending%NS == 123).all()
    s = build_session(q,session_id='s',previous_session_id='p',symbol='TEST',start_ns=base,
                      end_ns=base+100*NS,observed_through_ns=base+5*NS,scales=[SCALE])
    states = live_states(s,base+5*NS)
    for frame in (s.tokens, states):
        for name in ('start_ns','available_ns','extreme_ns','pending_start_ns'):
            x = frame[name].dropna().to_numpy(np.int64)
            assert ((x[x>=0]%NS)==123).all(), name
