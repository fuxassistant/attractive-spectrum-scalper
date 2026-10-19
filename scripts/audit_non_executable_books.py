"""Independently reconstruct three in-session invalid-book episodes from reset."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from spectrum.source import DbnSource
from spectrum.data_semantics_audit import ReferenceBook
from spectrum.ingest import save_json


def probe(job):
    r,member,reference=job
    q=pq.read_table(Path(r['market'])/'quotes.parquet').to_pandas()
    t=q.t_ns.to_numpy();v=q.valid.to_numpy(bool);ref=np.load(reference)
    ix=np.searchsorted(ref,t//60000000000);known=(ix<len(ref))&(ref[np.minimum(ix,len(ref)-1)]==t//60000000000)
    date=r['entry']['selection']['date'];lo=pd.Timestamp(date+' 08:00',tz='Europe/London').tz_convert('UTC').value;hi=pd.Timestamp(date+' 17:00',tz='America/New_York').tz_convert('UTC').value
    bad=np.flatnonzero(~v&known&(t>=lo)&(t<hi));first=int(bad[0]);last=first+int(np.flatnonzero(v[first:])[0])-1
    indices=sorted({first-1,first,(first+last)//2,last,last+1})
    expected={int(q.source_ordinal.iloc[i]):q.iloc[i].to_dict() for i in indices}
    book=ReferenceBook(expected);end=int(q.source_ordinal.iloc[last+1]);offset=0;states=[]
    for block in DbnSource(member['archive'],member['member'],block_records=262144):
        start=max(0,int(r['entry']['first_ordinal'])-offset);stop=min(len(block),end-offset+1)
        if stop>start:
            part=block[start:stop];route=r['entry']['route'];ids=np.flatnonzero((part['publisher_id']==route[0])&(part['instrument_id']==route[1])&(part['channel_id']==route[2]))
            for i in ids:
                ordinal=int(offset+start+i);record=part[i];book.feed(record,ordinal)
                if ordinal in expected:
                    bid=max(book.levels[b'B'],default=0);ask=min(book.levels[b'A'],default=0)
                    states.append(dict(ordinal=str(ordinal),t_ns=str(int(record['ts_recv'])),
                        stored_valid=bool(expected[ordinal]['valid']),stored_bid=int(expected[ordinal]['bid_nano']),stored_ask=int(expected[ordinal]['ask_nano']),
                        independent_bid=bid,independent_ask=ask,locked=bool(bid==ask and bid>0),crossed=bool(bid>ask and ask>0)))
        offset+=len(block)
        if offset>end:break
    return dict(key=r['key'],date=date,selected_records_replayed=book.rows,
        valid_anchor_checks=book.checks,valid_anchor_errors=book.errors,
        unknown_cancels=book.unknown_cancels,unknown_modifies=book.unknown_modifies,
        episode_first_ns=str(int(t[first])),episode_recovery_ns=str(int(t[last+1])),
        invalid_publications=last-first+1,states=states,
        limitation='Reproducing a locked/crossed raw book does not determine exchange trading status; status messages are absent from this MBO-only dataset.')


def main():
    master=json.loads(Path('data/derived/full_mbo_v1/manifest.json').read_text());config=json.loads(Path('data/derived/full_mbo_v1/run_config.json').read_text())
    members={m['key']:m for m in config['members']};dates=['2025-10-17','2026-01-30','2026-05-22']
    jobs=[]
    for date in dates:
        r=next(r for r in master['captures'] if r['status']=='COMPLETE' and r['entry']['selection']['date']==date)
        jobs.append((r,members[r['entry']['member_key']],config['reference']['path']))
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(probe,jobs))
    result=dict(status='COMPLETE',probes=rows)
    save_json('reports/data_properties_full_001/invalid_book_probes.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
