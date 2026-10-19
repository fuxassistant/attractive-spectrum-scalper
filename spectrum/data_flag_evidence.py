"""Extract every selected non-snapshot BAD_TS_RECV record; no guessed cause."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json
import numpy as np
from spectrum.source import DbnSource
from spectrum.ingest import save_json


def extract(job):
    member,captures=job;offset=0;rows=[]
    for block in DbnSource(member['archive'],member['member'],block_records=524288):
        indices=np.flatnonzero(((block['flags']&8)!=0)&((block['flags']&32)==0))
        for i in indices:
            record=block[i];ordinal=int(offset+i)
            for c in captures:
                e=c['entry']
                if int(e['first_ordinal'])<=ordinal<=int(e['last_ordinal']) and list(e['route'])==[int(record[k]) for k in ('publisher_id','instrument_id','channel_id')]:
                    row={name:(record[name].decode() if block.dtype[name].kind=='S' else str(int(record[name]))) for name in block.dtype.names}
                    rows.append(dict(key=c['key'],date=e['selection']['date'],source_ordinal=str(ordinal),**row));break
        offset+=len(block)
    return dict(member=member['key'],selected_flagged_records=rows)


def main(output):
    config=json.loads(Path('data/derived/full_mbo_v1/run_config.json').read_text());master=json.loads(Path('data/derived/full_mbo_v1/manifest.json').read_text())
    root=Path(output);root.mkdir(exist_ok=False,parents=True);results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(extract,(m,[r for r in master['captures'] if r['entry'].get('member_key')==m['key']])) for m in config['members']]
        for f in as_completed(fs):
            r=f.result();results.append(r);save_json(root/(r['member']+'.json'),r)
    rows=[r for m in results for r in m['selected_flagged_records']]
    result=dict(status='COMPLETE',count=len(rows),rows=rows)
    save_json(root/'summary.json',result);print(json.dumps(dict(status='COMPLETE',count=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
