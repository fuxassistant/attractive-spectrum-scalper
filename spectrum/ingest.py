"""Partition only at explicit source resets; never time-slice an unknown book."""
from pathlib import Path
import json, time
import numpy as np
from .source import DbnSource
from .core.vendor.glbx_dbn import MBO_DTYPE
from .core.storage import sha

PACKED = np.dtype([('source_ordinal','<i8'),('record',MBO_DTYPE)])

def save_json(path, value):
    path=Path(path); temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
    temp.replace(path)


def partition(path, output, route=(1,19181,22), member=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); src=DbnSource(path,member)
    offset=0; stream=None; entries=[]; current=None
    try:
        for b in src:
            selected=np.flatnonzero((b['publisher_id']==route[0])&(b['instrument_id']==route[1])&(b['channel_id']==route[2]))
            a=b[selected]; packed=np.empty(len(a),PACKED)
            packed['source_ordinal']=selected+offset; packed['record']=a
            cuts=sorted(set([0,len(a),*np.flatnonzero(a['action']==b'R').tolist()]))
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                if lo==hi:continue
                if a[lo]['action']==b'R':
                    if stream:stream.close()
                    name=f'capture_{len(entries):03d}.bin'
                    current=dict(name=name,records=0,start_ns=int(a[lo]['ts_recv']),first_ordinal=int(packed[lo]['source_ordinal']),reset_flags=int(a[lo]['flags']))
                    entries.append(current);stream=(output/name).open('wb')
                if stream is None:raise ValueError('Selected route lacks initial explicit reset; cannot reconstruct')
                stream.write(packed[lo:hi].tobytes())
                current['records']+=hi-lo
                current['end_ns']=int(a[hi-1]['ts_recv']);current['last_ordinal']=int(packed[hi-1]['source_ordinal'])
            offset+=len(b)
        if stream:stream.close();stream=None
        if not entries or not src.sha256:raise ValueError('Empty/incomplete source')
        for entry in entries:entry['sha256']=sha(output/entry['name'])
        manifest=dict(schema='RESET_PARTITIONS_V1',source=str(Path(path).resolve()),member=member,source_sha256=src.sha256,
                      raw_records=offset,selected_records=sum(e['records'] for e in entries),route=list(route),
                      partitions=entries,seconds=time.perf_counter()-started,
                      order='original source ordinal; no reset invented; overlap-free selected raw rows')
        save_json(output/'manifest.json',manifest)
        return manifest
    finally:
        if stream:stream.close()


def packed_records(block,route):
    # Column extraction in native NumPy replaces repeated structured-scalar access.
    from .core.vendor.l3 import MboRecord
    b=block['record']
    fields=['ts_event','ts_recv','sequence','order_id','price','size','flags','action','side']
    columns=[block['source_ordinal'].tolist()]+[b[f].tolist() for f in fields]
    for ordinal,te,tr,seq,oid,price,size,flags,action,side in zip(*columns):
        yield MboRecord(ordinal,*route,te,tr,seq,oid,price,size,flags,action.decode(),side.decode())
