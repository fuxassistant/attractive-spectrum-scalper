"""Market-only causal observations; no forward returns or alternative labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from spectrum.core.storage import sha
from spectrum.ingest import save_json
from .contract import NS

FEATURES=('spread_ticks','imbalance_l1','depth_bid_12','depth_ask_12',
          'displacement_5s','displacement_30s','displacement_60s','path_30s','efficiency_30s',
          'trade_buy_30s','trade_sell_30s','trade_unknown_30s','add_bid_30s','add_ask_30s',
          'remove_bid_30s','remove_ask_30s','matched_fill_remove_bid_30s','matched_fill_remove_ask_30s',
          'fft_mid_change_32s_power','fft_mid_change_64s_power','fft_mid_change_128s_power',
          'fft_mid_change_32s_mean','fft_mid_change_128s_mean','fft_mid_change_128s_entropy',
          'fft_signed_trade_32s_power','fft_signed_trade_128s_power',
          'fft_bid_add_32s_power','fft_ask_add_32s_power','fft_bid_remove_32s_power','fft_ask_remove_32s_power')

def session_bounds(date,contract):
    day=pd.Timestamp(date,tz='UTC')
    start=day.value-2*3600*NS
    end=day.value+21*3600*NS
    london=pd.Timestamp(date+f' {contract.london_open_hour:02d}:00',tz='Europe/London').value
    ny=pd.Timestamp(date+f' {contract.new_york_close_hour:02d}:00',tz='America/New_York').value
    return start,end,london,ny


def prepare_week(market,features,output,week,contract):
    market=Path(market);features=Path(features);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    master=json.loads((market/'manifest.json').read_text());fm=json.loads((features/'manifest.json').read_text())
    receipts={x['input']['name'].replace('.bin',''):x for x in master['captures']}
    feature_receipts={x['capture']:x for x in fm['captures']}
    dates=[f'2026-05-{d:02d}' for d in (range(4,9) if week==1 else range(11,16))]
    offset=0 if week==1 else 6;results=[]
    for k,date in enumerate(dates):
        start,end,london,ny=session_bounds(date,contract);parts=[];bars=[];fs=[];lineage=[]
        for number in (offset+k,offset+k+1):
            name=f'capture_{number:03d}';root=market/name;feat=features/name
            for file in ('quotes.parquet','bars.parquet'):
                if sha(root/file)!=receipts[name]['payloads'][file]:raise ValueError('Market SHA drift')
            if sha(feat/'features.parquet')!=feature_receipts[name]['feature_sha256']:raise ValueError('Feature SHA drift')
            filters=[('t_ns','>=',start),('t_ns','<',end)]
            parts.append(pq.read_table(root/'quotes.parquet',filters=filters).to_pandas())
            bars.append(pq.read_table(root/'bars.parquet',filters=filters).to_pandas())
            cols=['t_ns','available_ns','admissible']+[n for f in FEATURES for n in (f,f+'__known')]
            fs.append(pq.read_table(feat/'features.parquet',columns=cols,filters=filters).to_pandas())
            lineage.append(dict(capture=name,market_manifest=sha(root/'manifest.json'),feature_manifest=sha(feat/'manifest.json')))
        q=pd.concat(parts,ignore_index=True).sort_values(['available_ns','source_ordinal'],kind='stable').reset_index(drop=True)
        b=pd.concat(bars,ignore_index=True).sort_values('t_ns',kind='stable').reset_index(drop=True)
        f=pd.concat(fs,ignore_index=True).sort_values(['available_ns','t_ns'],kind='stable').reset_index(drop=True)
        # Feature endpoint no later than candle close AND publication no later
        # than its proof. A later quote in the same second is never joined back.
        idx=np.searchsorted(f.t_ns.to_numpy(),b.t_ns.to_numpy(),side='right')-1
        idx=np.maximum(idx,0)
        supported=(f.available_ns.to_numpy()[idx]<=b.available_ns.to_numpy())&(f.t_ns.to_numpy()[idx]<=b.t_ns.to_numpy())
        values=np.column_stack([f[n].to_numpy()[idx] for n in FEATURES]);known=np.column_stack([f[n+'__known'].to_numpy()[idx] for n in FEATURES])
        known&=supported[:,None];values=np.where(known,values,0.)
        raw=b[['o','h','l','c']].to_numpy();valid=b.valid.to_numpy()&np.isfinite(raw).all(axis=1)
        anchor=0.;previous=0.;running_high=-np.inf;running_low=np.inf;last_t=start;candles=[]
        for i,row in b.iterrows():
            ok=valid[i]
            if ok and not anchor:anchor=float(row.o);previous=anchor
            if ok:
                running_high=max(running_high,row.h);running_low=min(running_low,row.l)
                candle=[(row.o-previous)/.1,(row.h-row.l)/.1,(row.c-row.o)/.1,(row.c-previous)/.1,
                        (row.c-anchor)/.1,(running_high-running_low)/.1,np.log1p(max(0,row.volume)),row.coverage,1.]
                previous=float(row.c)
            else:candle=[0.]*7+[float(row.coverage),0.]
            candles.append(candle+[(int(row.t_ns)-last_t)/(60*NS),(int(row.t_ns)-start)/(23*3600*NS)])
            last_t=int(row.t_ns)
        x=np.c_[np.asarray(candles),values,known.astype(float)].astype(np.float32)
        decision_t=b.available_ns.to_numpy(np.int64)
        admission=valid&supported&f.admissible.to_numpy()[idx]&((decision_t-b.t_ns.to_numpy())<=NS)
        legal=admission&(decision_t>=london)&(decision_t<ny)
        dest=output/date;dest.mkdir()
        arrays=dict(x=x,decision_t=decision_t,bar_t=b.t_ns.to_numpy(np.int64),can_enter=legal,
                    quote_t=q.available_ns.to_numpy(np.int64),source_t=q.t_ns.to_numpy(np.int64),
                    bid=q.bid_nano.to_numpy(np.int64),ask=q.ask_nano.to_numpy(np.int64),valid=q.valid.to_numpy(bool),epoch=q.epoch.to_numpy(np.int64),
                    ordinal=q.source_ordinal.to_numpy(np.int64))
        for name,a in arrays.items():np.save(dest/(name+'.npy'),a)
        r=dict(date=date,week=week,rows=len(b),quotes=len(q),entry_origins=int(legal.sum()),context_origins=int((decision_t<london).sum()),
               start_ns=str(start),end_ns=str(end),london_ns=str(london),ny_close_ns=str(ny),
               liquidation_deadline_ns=str(end-contract.terminal_reserve_seconds*NS),lineage=lineage,
               feature_names=['open_minus_previous','range','body','close_change','from_session_open','session_range','log_volume','coverage','candle_known','elapsed_minutes','session_fraction']+list(FEATURES)+[n+'__known' for n in FEATURES],
               files={name+'.npy':sha(dest/(name+'.npy')) for name in arrays})
        save_json(dest/'manifest.json',r);results.append(r)
        print(json.dumps(dict(prepared=date,bars=len(b),quotes=len(q))),flush=True)
    save_json(output/'manifest.json',dict(week=week,days=results,market_sha256=sha(market/'manifest.json'),feature_sha256=sha(features/'manifest.json'),
                                        session_contract={k:contract.dict()[k] for k in ('london_open_hour','new_york_close_hour','terminal_reserve_seconds')}))
    return results


class Day:
    def __init__(self,path,verify=True):
        self.path=Path(path);self.meta=json.loads((self.path/'manifest.json').read_text())
        for file,h in self.meta['files'].items():
            if verify and sha(self.path/file)!=h:raise ValueError('Gym input drift')
            setattr(self,file[:-4],np.load(self.path/file,mmap_mode='r'))
        if np.any(np.diff(self.quote_t)<0) or np.any(np.diff(self.ordinal)<=0):raise ValueError('Quote order')
        self.cuts=np.searchsorted(self.quote_t,self.decision_t,side='right')
        self.deadline=int(self.meta['liquidation_deadline_ns'])
