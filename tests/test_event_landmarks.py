import numpy as np
from spectrum.event_landmarks import detect_bursts, recurrence_labels


def tape(n=240):
    return np.zeros(n),np.ones(n),np.ones(n,dtype=bool)


def test_recurrence_is_past_only_and_regular_requires_three_gaps():
    m,v,s=tape();m[[130,136,142,148]]=10
    rows,*_=detect_bursts(m,v,s)
    assert [r['event_class'] for r in rows]==['fresh_impulse','second_in_cluster','recurrent_irregular','recurrent_regular']
    assert rows[-1]['last_gap_s']==30
    prefix=detect_bursts(m[:149],v[:149],s[:149])[0]
    assert prefix==rows
    m[149:]=1000;v[149:]=1e6
    assert detect_bursts(m,v,s)[0][:4]==rows


def test_plateau_rearm_and_gap_reset():
    m,v,s=tape();m[130:136]=10;m[137]=10;m[140]=30
    rows=detect_bursts(m,v,s)[0]
    assert [r['bin'] for r in rows]==[130,140]
    s[138]=False
    assert [r['bin'] for r in detect_bursts(m,v,s)[0]]==[130]


def test_unknown_volume_does_not_require_direction():
    m,v,s=tape();v[130]=100
    r=detect_bursts(m,v,s)[0][0]
    assert r['activity']=='volume_led' and r['move_threshold']==3
    assert r['volume_threshold']==10


def test_isolation_censored_at_gap_and_eof():
    s=np.ones(200,bool)
    r=recurrence_labels([130,136,190],[130,136,190],s)
    assert r[0]['future_events_180']==1 and not r[0]['retrospectively_no_repeat']
    assert r[1]['retrospectively_no_repeat']
    assert not r[2]['recurrence_mature'] and not r[2]['retrospectively_no_repeat']
    s[150]=False
    assert not recurrence_labels([130],[130],s)[0]['recurrence_mature']
