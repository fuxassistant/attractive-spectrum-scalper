import numpy as np
import pandas as pd
import pytest
from spectrum.features.full_history import ObservedCalendar
from spectrum.features.full_history import MINUTE
from spectrum.features.full_history import DAY
from spectrum.features.full_history import london_ns
from spectrum.features.full_history import prefix_score
from spectrum.features.full_history import child_path
from spectrum.core.vendor.glbx_dbn import MBO_DTYPE


def test_only_explicit_orchestration_migration_is_compatible():
    from spectrum.features.full_history import compatible_identity
    old={'full_history.py':'old','lab.py':'fixed','core':'fixed'}
    new={**old,'full_history.py':'new'}
    config={'code_identity':new,'compatible_code_identities':[old]}
    assert compatible_identity(config,old)
    assert not compatible_identity(config,{**old,'lab.py':'changed'})
    assert not compatible_identity({'code_identity':new},old)


def test_empty_capture_is_excluded_without_inventing_features(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from spectrum.features.full_history import code_identity
    from spectrum.features.full_history import generate_features
    from spectrum.core.storage import sha
    from spectrum.io import save_json
    capture=tmp_path/'market';capture.mkdir()
    pq.write_table(pa.table({'t_ns':pa.array([],type=pa.int64())}),capture/'seconds.parquet')
    ref=tmp_path/'minutes.npy';np.save(ref,np.array([1,2]))
    identity=code_identity()
    config={'code_identity':identity,'reference':{'path':str(ref),'sha256':sha(ref)}}
    entry={'sha256':'raw','route':[1,2,3],'selection':{'source_condition':'available','context_only':False}}
    market={'source_partition_sha256':'raw','route':[1,2,3],'implementation':identity['core'],
            'payloads':{'seconds.parquet':sha(capture/'seconds.parquet')},'seconds_rows':0,'records':574,'quotes':1,'bars':0}
    save_json(capture/'manifest.json',market)
    out=tmp_path/'features'
    result=generate_features(config,entry,capture,out)
    assert result['status']=='EXCLUDED' and result['rows']==0
    assert result['exclusion_reasons']==['NO_FINALIZED_SECONDS']
    assert result['feature_sha256'] is None and not (out/'features.parquet').exists()
    assert json.loads((out/'manifest.json').read_text())['evidence']['records']==574
    market['seconds_rows']=1;save_json(capture/'manifest.json',market)
    with pytest.raises(ValueError,match='contradicts'):
        generate_features(config,entry,capture,tmp_path/'bad_features')


def test_training_catalog_preserves_reasons_and_threshold():
    from spectrum.features.full_history import training_assessment
    r={'rows':100,'entry':{'selection':{'source_condition':'available','context_only':False}},
       'entry_allowed_rows':70,'reference_open_rows':100,'admitted_rows':80,'independent_ohlc':{'status':'PASS'}}
    assert training_assessment(r)['eligible']
    r['admitted_rows']=79
    assert 'ADMISSIBLE_REFERENCE_COVERAGE_BELOW_80_PERCENT' in training_assessment(r)['reasons']
    r.update(rows=0,entry_allowed_rows=0,reference_open_rows=0,admitted_rows=0)
    assert 'NO_FINALIZED_SECONDS' in training_assessment(r)['reasons']


def test_observed_breaks_and_missing_minutes_remain_closed(tmp_path):
    path=tmp_path/'minutes.npy'
    start=pd.Timestamp('2026-05-04T20:59Z').value//MINUTE
    np.save(path,np.array([start,start+61,start+62]))
    c=ObservedCalendar(path)
    assert c.closed(np.array([start-1,start,start+1,start+60,start+61,start+63])*MINUTE).tolist()==[True,False,True,True,False,True]
    with pytest.raises(ValueError,match='drift'):ObservedCalendar(path,'wrong')


def test_materialization_separates_venue_closure_from_reference_coverage(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import spectrum.features.full_history as history
    from spectrum.core.storage import sha
    from spectrum.io import save_json
    from test_foundation import frame
    capture = tmp_path / 'market'; capture.mkdir()
    data = frame(180)
    data['t_ns'] = pd.Timestamp('2026-01-14 21:59Z').value + np.arange(len(data)) * 10**9
    data['available_ns'] = data.t_ns + 100
    pq.write_table(pa.Table.from_pandas(data), capture / 'seconds.parquet')
    reference = tmp_path / 'reference.npy'
    # Missing reference during an OPEN minute; present reference during CLOSED minutes.
    np.save(reference, np.unique(data.t_ns.to_numpy()[60:] // MINUTE))
    identity = history.code_identity()
    config = dict(code_identity=identity, reference=dict(path=str(reference), sha256=sha(reference)), reserve_bytes=0)
    entry = dict(name='capture_000', sha256='raw', route=[1, 2, 3],
                 selection=dict(source_condition='available', context_only=False, selection_available_ns=0))
    save_json(capture / 'manifest.json', dict(source_partition_sha256='raw', route=entry['route'],
              implementation=identity['core'], payloads={'seconds.parquet': sha(capture / 'seconds.parquet')}))
    monkeypatch.setattr(history, 'verify_ohlc', lambda *_: dict(status='NOT_RUN_FIXTURE'))
    output = tmp_path / 'features'
    receipt = history.generate_features(config, entry, capture, output)
    result = pq.read_table(output / 'features.parquet').to_pandas()
    assert not result.scheduled_closed.iloc[:60].any()
    assert result.scheduled_closed.iloc[60:].all()
    assert not result.reference_observed_open.iloc[:60].any()
    assert result.reference_observed_open.iloc[60:].all()
    assert not result.admissible.any()
    assert receipt['calendar']['version'] == 'CANON_CALENDAR_V1'
    assert 'reference_calendar' in receipt


def test_london_cutoff_tracks_dst():
    for date,hour in [('2026-01-05',8),('2026-05-04',7)]:
        day=pd.Timestamp(date,tz='UTC').value//DAY
        assert pd.Timestamp(london_ns(day),tz='UTC').hour==hour


def test_prefix_excludes_incomplete_group_across_batches():
    r=np.zeros(4,dtype=MBO_DTYPE)
    r['action']=b'T';r['size']=[2,3,5,7];r['flags']=[128,0,128,0]
    ready,pending=prefix_score(r[:2],(0,0))
    assert ready==(2,1) and pending==(3,1)
    ready,pending=prefix_score(r[2:],pending)
    assert ready==(8,2) and pending==(7,1)
    ready,pending=prefix_score(r[3:],pending)
    assert ready==(0,0) and pending==(14,2)


def test_staging_paths_cannot_escape(tmp_path):
    assert child_path(tmp_path,'staging','one.bin').is_relative_to(tmp_path)
    with pytest.raises(ValueError):child_path(tmp_path,'..','outside.bin')
    with pytest.raises(ValueError):child_path(tmp_path)
