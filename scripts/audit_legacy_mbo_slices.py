"""Ensure old May extracts are overlapping copies, not additional observations."""
from pathlib import Path
import hashlib,json
import pandas as pd
from spectrum.source import DbnSource
from spectrum.ingest import save_json


def main():
    specs=[]
    for path in sorted(Path('data/raw').glob('GC_MBO_*.dbn.manifest.json')):
        meta=json.loads(path.read_text());file=path.with_name(path.name.replace('.manifest.json','.zst'))
        digest=hashlib.sha256();count=0;first=last=None
        for block in DbnSource(file):
            digest.update(block.tobytes());count+=len(block)
            if len(block):
                if first is None:first=int(block['ts_recv'][0])
                last=int(block['ts_recv'][-1])
        specs.append(dict(file=str(file),start=pd.Timestamp(meta['interval_utc'][0]).value,
            end=pd.Timestamp(meta['interval_utc'][1]).value,expected_count=meta['selected_records'],
            records=count,first_ns=str(first),last_ns=str(last),record_sha256=digest.hexdigest(),
            original=hashlib.sha256(),original_count=0))
    config=json.loads(Path('data/derived/full_mbo_v1/run_config.json').read_text())
    member=next(m for m in config['members'] if m['key']=='member_007')
    for block in DbnSource(member['archive'],member['member']):
        for spec in specs:
            rows=block[(block['ts_recv']>=spec['start'])&(block['ts_recv']<spec['end'])]
            spec['original'].update(rows.tobytes());spec['original_count']+=len(rows)
        if int(block['ts_recv'][-1])>=max(s['end'] for s in specs):break
    for spec in specs:
        spec['original_record_sha256']=spec.pop('original').hexdigest()
        spec['matches_original_interval']=spec['record_sha256']==spec['original_record_sha256'] and spec['records']==spec['original_count']==spec['expected_count']
    result=dict(status='PASS' if all(s['matches_original_interval'] for s in specs) else 'FAIL',files=specs,
        interpretation='These legacy overlapping May slices are not added to canonical population counts or treated as independent train/test observations')
    save_json('reports/data_properties_full_001/legacy_slices.json',result);print(json.dumps(result))


if __name__=='__main__':main()
