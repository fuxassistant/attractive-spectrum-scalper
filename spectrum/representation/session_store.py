"""Immutable session dataset including the actual model-facing as-of loader."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from spectrum.io import save_json,sha
from spectrum.features.contract import reject_future
from spectrum.representation.session import SessionObservationStore,session_bounds


def publish_session_store(root,*,date,previous_date,symbol,tables,quote_sources,source_identity):
    required={'fast','context','zones','events','features','core_features'}
    if set(tables)!=required or not quote_sources or not source_identity:
        raise ValueError('Complete session profile and source lineage required')
    for name,frame in tables.items():
        reject_future(frame.columns,frame)
        if not frame.available_ns.is_monotonic_increasing:
            raise ValueError('Unordered publication clock: '+name)
        if 't_ns' in frame and (frame.available_ns<frame.t_ns).any():
            raise ValueError('Premature publication: '+name)
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    save_json(root/'manifest.json',dict(status='RUNNING'))
    files={}
    try:
        for name,frame in tables.items():
            frame.to_parquet(root/(name+'.parquet'),index=False)
            files[name+'.parquet']=sha(root/(name+'.parquet'))
        result=dict(version='CME_SESSION_STORE_V1',status='COMPLETE',date=date,previous_date=previous_date,
            symbol=symbol,files=files,quote_sources=quote_sources,source_identity=source_identity,
            geometry=dict(version='COMPLETED_NATIVE_BB_V2',clock='completed_native',period=20,k=2,ddof=1),
            crossing_geometry='Developing quote-native BB is a separately named contract, not these completed bands')
        save_json(root/'manifest.json',result)
        return result
    except Exception as exc:
        save_json(root/'manifest.json',dict(status='FAILED',error=repr(exc)))
        raise


def open_session_store(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if m.get('status')!='COMPLETE' or m.get('version')!='CME_SESSION_STORE_V1':
        raise ValueError('Incomplete or unsupported session store')
    tables={}
    for name,digest in m['files'].items():
        if sha(root/name)!=digest:
            raise ValueError('Session payload drift: '+name)
        tables[name.removesuffix('.parquet')]=pd.read_parquet(root/name)
    start,end=session_bounds(m['date']);parts=[]
    for source in m['quote_sources']:
        path=Path(source['path'])
        if sha(path)!=source['sha256']:
            raise ValueError('Native quote source drift')
        q=pd.read_parquet(path,filters=[('t_ns','>=',start-30*10**9),('t_ns','<=',end)])
        cutoff=source.get('skip_bootstrap_through_ordinal')
        if cutoff is not None:
            # Allowed only for a complete, identical order-book proof.
            if source.get('handoff_version')!='IDENTICAL_BOOK_HANDOFF_V1' or not source.get('book_sha256'):
                raise ValueError('Unproven snapshot suppression')
            q=q[q.source_ordinal>cutoff].copy()
        q['source_epoch']=q.epoch
        mapping={int(k):int(v) for k,v in source.get('epoch_map',{}).items()}
        q['epoch']=q.epoch.replace(mapping).astype(np.int64)
        parts.append(q)
    quotes=pd.concat(parts,ignore_index=True).sort_values('available_ns',kind='stable').reset_index(drop=True)
    return SessionObservationStore(**tables,quotes=quotes,symbol=m['symbol'],date=m['date'],previous_date=m['previous_date'])
