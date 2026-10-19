import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import pyarrow.parquet as pq

from spectrum.finalized import FinalizedReplay, FinalizedSession, restore_l3, verify
from spectrum.finalize_market import DayBuilder, Intervals, session_bounds, flags_for_times
from spectrum.core.vendor.l3 import MboRecord


def record(i,t,action='A',side='B',price=100_000_000_000,size=2,oid=None,flags=128):
    return MboRecord(i,1,7,1,t,t, i, i if oid is None else oid,price,size,flags,action,side)


def build(tmp_path, extra=()):
    start=pd.Timestamp('2026-08-03 07:00',tz='UTC').value
    choice=dict(date='2026-08-03',route=[1,7,1],symbol='GCZ6',context_only=False,source_condition='available')
    b=DayBuilder(tmp_path/'day',choice,np.array([],np.int64),0,{'test':True})
    rows=[record(0,start,'R','N',flags=32),record(1,start,'A','B',oid=1,flags=32),
          record(2,start,'A','A',price=101_000_000_000,oid=2,flags=160),
          record(3,start+1_100_000_000,'T','N',size=3,oid=3),
          record(4,start+8_000_000_000,'A','B',price=99_000_000_000,oid=4),*extra]
    for r in rows:b.feed(r)
    b.raw_records=len(rows)
    b.finish([dict(records=len(rows)),dict(records=0)],'test',{})
    return b,start


def test_index_cursor_l3_and_reference_absence(tmp_path,monkeypatch):
    b,start=build(tmp_path)
    assert verify(b.root)['status']=='PASS'
    r=FinalizedReplay(b.root)
    table=r.read_batch(2);cursor=r.checkpoint()
    tail=r.read_batch();restored=FinalizedReplay.restore(b.root,cursor).read_batch()
    assert tail.equals(restored)
    assert table['valid'].to_pylist()==[True,True]
    assert not any(table['reference_present'].to_pylist())
    assert table['trade_unknown'].to_pylist()==[0,3]
    assert table['segment_id'].to_pylist()==[0,0]
    assert restore_l3(b.root,start+2_000_000_000)=={1:('B',100_000_000_000,2),2:('A',101_000_000_000,2)}
    # Reader must not hash any payload or reconstruct raw state.
    import spectrum.core.storage
    monkeypatch.setattr(spectrum.core.storage,'sha',lambda *_:pytest.fail('reader hashed payload'))
    assert sum(len(t) for t in FinalizedReplay(b.root).batches())==3
    assert len(FinalizedSession(b.root).segments())==1


def test_crossed_real_prices_split_not_silence(tmp_path):
    start=pd.Timestamp('2026-08-03 07:00',tz='UTC').value
    b,_=build(tmp_path,[record(5,start+9_000_000_000,price=102_000_000_000,oid=5),
                       record(6,start+10_000_000_000,'C',price=102_000_000_000,oid=5)])
    q=pq.read_table(b.root/'quotes.parquet').to_pandas()
    assert q.iloc[3].crossed and q.iloc[3].bid_nano==102_000_000_000
    assert q.segment_id.tolist()==[0,0,0,-1,1]
    segments=json.loads((b.root/'segments.json').read_text())
    assert len(segments)==2
    assert segments[-1]['end_reason']=='OBSERVATION_END'


def test_pending_group_cannot_leak_depth(tmp_path):
    start=pd.Timestamp('2026-08-03 07:00',tz='UTC').value
    b,_=build(tmp_path,[record(5,start+8_100_000_000,'A','B',size=100,oid=5,flags=0),
                       record(6,start+10_100_000_000,'T','N')])
    s=pq.read_table(b.root/'seconds.parquet').to_pandas()
    assert s[s.t_ns==start+9_000_000_000].depth_bid_1.iloc[0]==2


def test_hours_dst_reference_independence():
    summer=session_bounds('2026-08-03');winter=session_bounds('2026-12-03')
    assert pd.Timestamp(summer['london_ns'],tz='UTC').hour==7
    assert pd.Timestamp(winter['london_ns'],tz='UTC').hour==8
    t=pd.to_datetime(['2026-08-03 20:00Z','2026-08-03 21:00Z','2026-08-03 22:00Z','2026-08-08 12:00Z']).asi8
    closed,ref=flags_for_times(t,np.array([],np.int64))
    assert closed.tolist()==[False,True,False,True]
    assert not ref.any()


def test_histograms_match_scalar():
    a=Intervals();b=Intervals();values=[0,1,2,100,10**9,10**12,-1]
    a.add(values)
    for v in values:b.add_one(v)
    assert a.result()==b.result()


def test_corruption_detected_explicitly(tmp_path):
    b,_=build(tmp_path)
    p=b.root/'states.json';p.write_text('[]')
    with pytest.raises(ValueError,match='Payload mismatch'):verify(b.root)


def test_incomplete_publication_not_published(tmp_path):
    choice=dict(date='2026-08-03',route=[1,7,1],symbol='GCZ6',context_only=False,source_condition='available')
    b=DayBuilder(tmp_path/'bad',choice,np.array([],np.int64),0,{})
    b.feed(record(0,session_bounds('2026-08-03')['london_ns'],'R','N',flags=0))
    with pytest.raises(ValueError,match='UNFINISHED'):b.finish([],None,{})
    assert not (b.root/'manifest.json').exists()
    for w in (b.quotes,b.l3,b.seconds):w.writer.close()


def test_exact_candles_keep_intrasecond_extreme_and_volume(tmp_path):
    start=pd.Timestamp('2026-08-03 07:00',tz='UTC').value
    b,_=build(tmp_path,[record(5,start+8_100_000_000,'A','B',price=100_500_000_000,oid=5),
                       record(6,start+8_200_000_000,'C','B',price=100_500_000_000,oid=5),
                       record(7,start+60_100_000_000,'T','N',size=7)])
    bars=pq.read_table(b.root/'bars.parquet').to_pandas()
    assert bars.h.iloc[0]==100.75
    assert bars.volume.iloc[0]==3
    assert bars.available_ns.iloc[0]==start+60_100_000_000


def test_advance_cursor_restore_age_and_projected_window(tmp_path):
    b,start=build(tmp_path)
    r=FinalizedReplay(b.root)
    assert r.advance(start)['source_ordinal']==2
    assert r.advance(start+6_000_000_000)['quote_age_ns']==4_900_000_000
    cursor=r.checkpoint()
    restored=FinalizedReplay.restore(b.root,cursor)
    assert restored.advance(start+7_000_000_000)==r.advance(start+7_000_000_000)
    with pytest.raises(ValueError,match='rewind'):r.advance(start)
    assert FinalizedReplay(b.root,columns=['bid_nano']).read_window(start,start+1).column_names==['bid_nano']


def test_multi_group_seek_and_restore(tmp_path,monkeypatch):
    import spectrum.finalize_market as fm
    monkeypatch.setattr(fm,'BLOCK',2)
    b,start=build(tmp_path)
    r=FinalizedReplay(b.root);r.seek(start+8_000_000_000)
    cursor=r.checkpoint()
    assert cursor['group']==1
    assert FinalizedReplay.restore(b.root,cursor).read_batch()['source_ordinal'].to_pylist()==[4]


def test_existing_experiment_loader_dispatches_finalized(tmp_path,monkeypatch):
    from spectrum.player_v3.data import Day
    from spectrum.player_v3.finalized_data import FinalizedDay
    (tmp_path/'manifest.json').write_text(json.dumps(dict(support_contract='FINALIZED_SEGMENTS_V2')))
    def load(self,path,verify=False):self.dispatch_verified=verify
    monkeypatch.setattr(FinalizedDay,'__init__',load)
    assert Day(tmp_path).dispatch_verified is False
    assert Day(tmp_path,verify=True).dispatch_verified is True
