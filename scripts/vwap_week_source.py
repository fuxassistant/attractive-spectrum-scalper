"""Extract real T prints; bind them to verified canonical publication ordinals."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from spectrum.source import DbnSource
from spectrum.ingest import PACKED
from scripts.flow_context_coverage import save, sha

OUT=Path('data/derived/vwap_bands_week_v1')
REPORT=Path('reports/vwap_bands_week_v1')
ROOT=Path('data/derived/full_mbo_v1')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'run_config.json').read_text())
    member=next(m for m in config['members'] if m['key']=='member_004')
    captures=[]
    for i in range(6):
        root=ROOT/'market/member_004'/f'capture_{14000+i:06}'
        m=json.loads((root/'manifest.json').read_text());e=m['input']
        captures.append(dict(root=root,entry=e,hash=hashlib.sha256(),pieces=[]))
    source=DbnSource(member['archive'],member['member'],block_records=262144)
    offset=0;last=time.perf_counter()
    for block in source:
        if offset<=captures[-1]['entry']['last_ordinal']:
            ix=np.flatnonzero((block['publisher_id']==1)&(block['instrument_id']==42000890)&(block['channel_id']==22))
            ordinals=offset+ix
            for cap in captures:
                e=cap['entry'];use=(ordinals>=e['first_ordinal'])&(ordinals<=e['last_ordinal'])
                if not use.any():continue
                selected=block[ix[use]];packed=np.empty(len(selected),PACKED)
                packed['source_ordinal']=ordinals[use];packed['record']=selected
                cap['hash'].update(packed.tobytes())
                take=(selected['action']==b'T')&((selected['flags']&32)==0)&(selected['size']>0)
                trades=selected[take]
                if len(trades):
                    if np.any((trades['price']<=0)|(trades['price']>=2**62)):raise ValueError('Invalid T price')
                    cap['pieces'].append(pd.DataFrame(dict(source_ordinal=ordinals[use][take],recv_ns=trades['ts_recv'].astype(np.int64),
                        price_nano=trades['price'],quantity=trades['size'],side=trades['side'])))
        offset+=len(block)
        if time.perf_counter()-last>25:
            print(json.dumps(dict(raw_records=offset,source_day=str(pd.Timestamp(int(block['ts_recv'][-1]),tz='UTC')))),flush=True);last=time.perf_counter()
    if source.sha256!=member['sha256']:raise ValueError('Original source member hash mismatch')
    receipts=[]
    for cap in captures:
        e=cap['entry'];root=cap['root']
        if cap['hash'].hexdigest()!=e['sha256']:raise ValueError('Partition identity mismatch')
        f=pd.concat(cap['pieces'],ignore_index=True)
        q=pd.read_parquet(root/'quotes.parquet',columns=['source_ordinal','available_ns','valid','epoch','bid_nano','ask_nano'])
        m=json.loads((root/'manifest.json').read_text())
        qsha=sha(root/'quotes.parquet')
        if qsha!=m['payloads']['quotes.parquet']:raise ValueError('Quote hash mismatch')
        j=np.searchsorted(q.source_ordinal.to_numpy(),f.source_ordinal.to_numpy(),side='left')
        if (j>=len(q)).any():raise ValueError('Unpublished trade')
        f['available_ns']=q.available_ns.to_numpy()[j]
        f['quote_ordinal']=j;f['valid']=q.valid.to_numpy()[j]
        f['epoch']=q.epoch.to_numpy()[j]
        f['mid']=(q.bid_nano.to_numpy()[j].astype(float)+q.ask_nano.to_numpy()[j])/2e9
        if (f.available_ns<f.recv_ns).any():raise ValueError('Trade available before receive')
        # Independent volume reconciliation against every uninterrupted canonical second.
        s=pd.read_parquet(root/'seconds.parquet')
        c=s.c_trade_buy.to_numpy()+s.c_trade_sell.to_numpy()+s.c_trade_unknown.to_numpy()
        prefix=np.r_[0,np.cumsum(f.quantity.to_numpy(np.int64))]
        cuts=np.searchsorted(f.available_ns.to_numpy(),s.t_ns.to_numpy(),side='right')
        exact=np.diff(prefix[cuts]);observed=np.diff(c)
        same=(np.diff(s.epoch.to_numpy())==0)&(np.diff(s.t_ns.to_numpy())==10**9)
        err=float(np.max(abs(exact[same]-observed[same]))) if same.any() else 0.
        if err:raise ValueError(f'Canonical T counter mismatch {e["selection"]["date"]}: {err}')
        path=OUT/(e['selection']['date']+'_trades.parquet');f.to_parquet(path,index=False)
        receipts.append(dict(date=e['selection']['date'],file=str(path),sha256=sha(path),rows=len(f),quantity=int(f.quantity.sum()),
            partition_sha256=e['sha256'],quote_sha256=qsha,canonical_volume_checks=int(same.sum()),max_volume_error=err))
    save(REPORT/'source_receipt.json',dict(status='PASS',source_member_sha256=source.sha256,records=offset,route=[1,42000890,22],
        symbol='GCJ6',captures=receipts,script_sha256=sha(__file__),scope='T only, no F/C duplication; publication clock from canonical last source ordinal; full raw member and six partitions rehashed'))
    print(json.dumps(dict(status='PASS',trades=sum(x['rows'] for x in receipts))),flush=True)


if __name__=='__main__':main()
