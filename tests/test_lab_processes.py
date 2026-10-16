import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from spectrum.lab import generate
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from spectrum.tape import TapeReplay
from test_foundation import frame


def test_parallel_feature_jobs_and_external_plugin_match_serial(tmp_path):
    market=tmp_path/'market';market.mkdir();captures=[]
    for i in range(2):
        root=market/f'capture_{i:03d}';root.mkdir()
        f=frame(250);f.t_ns+=i*86400*10**9;f.available_ns+=i*86400*10**9
        pq.write_table(pa.Table.from_pandas(f,preserve_index=False),root/'seconds.parquet')
        save_json(root/'manifest.json',dict(status='COMPLETE',payloads={'seconds.parquet':sha(root/'seconds.parquet')}))
        captures.append(dict(input={'name':f'capture_{i:03d}.bin'}))
    save_json(market/'manifest.json',dict(captures=captures))
    plugins=['spectrum.plugins.depth_balance:PLUGIN'];families=['motion','flow','spectral','depth_balance']
    generate(market,tmp_path/'serial',families,1,plugins)
    generate(market,tmp_path/'parallel',families,2,plugins)
    for i in range(2):
        a=tmp_path/'serial'/f'capture_{i:03d}'/'features.parquet';b=tmp_path/'parallel'/f'capture_{i:03d}'/'features.parquet'
        assert pq.read_table(a).equals(pq.read_table(b))
        assert sha(a)==sha(b)
        assert 'depth_balance_12' in pq.read_schema(a).names
    with pytest.raises(FileExistsError):generate(market,tmp_path/'serial',families,1,plugins)


def test_tape_clock_cursor_equal_time_and_tamper(tmp_path):
    table=pa.table(dict(available_ns=[1,2,2,4],t_ns=[1,2,2,4],source_ordinal=[5,10,11,12],bid_nano=[100,101,102,103],valid=[True]*4))
    pq.write_table(table,tmp_path/'quotes.parquet')
    save_json(tmp_path/'manifest.json',dict(status='COMPLETE',payloads={'quotes.parquet':sha(tmp_path/'quotes.parquet')}))
    r=TapeReplay(tmp_path);assert r.advance(2)['source_ordinal']==11
    state=r.checkpoint();assert state['cursor']==3
    restored=TapeReplay.restore(tmp_path,state);assert next(r)==next(restored)
    with pytest.raises(ValueError):r.advance(1)
    stale=r.advance(40*10**9)
    assert stale['valid'] and not stale['admissible_at_query']
    pq.write_table(pa.table(dict(available_ns=[1])),tmp_path/'quotes.parquet')
    with pytest.raises(ValueError):TapeReplay(tmp_path)
