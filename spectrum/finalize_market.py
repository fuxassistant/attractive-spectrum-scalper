"""One-time raw -> immutable market preparation; bounded memory and disk."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
import time
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from .finalized import VERSION, read_json, verify
from .source import DbnSource
from .ingest import PACKED, packed_records, save_json
from .core.vendor.l3_v2 import L3EventEngineV2
from .core.aggregate import FlowCounters, depth_vector
from .core.schema import COUNTERS, DEPTH, SNAP_SCHEMA, digest
from .core.storage import sha

NS = 10**9
DAY = 86400*NS
RESERVE = 8*1024**3
BLOCK = 65536


def implementation():
    files = [Path(__file__), Path(__file__).with_name('finalized.py'),
             Path(__file__).parent/'core/vendor/l3.py', Path(__file__).parent/'core/vendor/l3_v2.py',
             Path(__file__).parent/'core/aggregate.py', Path(__file__).parent/'core/event_facts.py',
             Path(__file__).with_name('source.py'), Path(__file__).with_name('ingest.py')]
    return {str(p.relative_to(Path(__file__).parent)):sha(p) for p in files}


def session_bounds(date):
    d = pd.Timestamp(date)
    return dict(date=date, london_ns=pd.Timestamp(date+' 08:00', tz='Europe/London').value,
                ny_close_ns=pd.Timestamp(date+' 17:00', tz='America/New_York').value,
                start_ns=pd.Timestamp(str((d-pd.Timedelta(days=1)).date())+' 18:00', tz='America/New_York').value,
                timezone_version='IANA Europe/London America/New_York')


def flags_for_times(times, reference_minutes):
    """Schedule convention and reference availability remain independent."""
    stamps = pd.to_datetime(np.asarray(times, np.int64), utc=True)
    ny = stamps.tz_convert('America/New_York')
    dow, hour = ny.dayofweek.to_numpy(), ny.hour.to_numpy()
    closed = ((dow == 5) | ((dow == 4)&(hour >= 17)) | ((dow == 6)&(hour < 18)) |
              ((dow < 4)&(hour >= 17)&(hour < 18)))
    minutes = np.asarray(times, np.int64)//(60*NS)
    idx = np.searchsorted(reference_minutes, minutes)
    if len(reference_minutes):
        present = (idx < len(reference_minutes)) & (reference_minutes[np.minimum(idx,len(reference_minutes)-1)] == minutes)
    else:
        present = np.zeros(len(times), bool)
    return closed, present


class Writer:
    def __init__(self, path, schema, reserve=RESERVE):
        self.path = Path(path)
        self.dtype = np.dtype(schema)
        self.buffer = np.empty(BLOCK+100001, dtype=self.dtype)
        self.n = self.rows = 0
        self.groups = []
        self.reserve = reserve
        self.writer = pq.ParquetWriter(self.path, pa.schema([(n,pa.from_numpy_dtype(self.dtype[n])) for n in self.dtype.names]), compression='zstd')

    def append(self, row):
        self.buffer[self.n] = tuple(row)
        self.n += 1

    def boundary(self):
        if self.n >= BLOCK:
            self.flush()

    def flush(self):
        if not self.n:
            return
        if psutil.disk_usage(str(self.path.parent)).free < self.reserve:
            raise RuntimeError('DISK_RESERVE_REACHED')
        table = pa.table({n:pa.array(self.buffer[n][:self.n]) for n in self.dtype.names})
        times = self.buffer['available_ns'][:self.n]
        self.groups.append(dict(rows=self.n, row_start=self.rows, first_available_ns=int(times[0]),last_available_ns=int(times[-1])))
        self.writer.write_table(table, row_group_size=self.n)
        self.rows += self.n
        self.n = 0

    def close(self):
        self.flush()
        self.writer.close()
        return dict(file=self.path.name, rows=self.rows, bytes=self.path.stat().st_size, groups=self.groups)


class Intervals:
    """Fixed logarithmic ns histogram, bounded memory, exact counts in bins."""
    def __init__(self):
        self.counts = Counter()
        self.count = self.total = self.maximum = self.zero = self.negative = 0

    def add(self, values):
        a = np.asarray(values, np.int64)
        self.negative += int((a<0).sum())
        a = a[a>=0]
        self.count += len(a)
        self.zero += int((a==0).sum())
        if not len(a): return
        self.total += int(a.sum())
        self.maximum = max(self.maximum, int(a.max()))
        p = a[a>0]
        keys, counts = np.unique(np.floor(np.log2(p)*128).astype(np.int32),return_counts=True)
        self.counts.update({int(k):int(v) for k,v in zip(keys,counts)})

    def add_one(self, value):
        value=int(value)
        if value<0:
            self.negative+=1;return
        self.count+=1;self.total+=value;self.maximum=max(self.maximum,value)
        if value==0:self.zero+=1
        else:self.counts[int(math.log2(value)*128)]+=1

    def result(self):
        def q(p):
            rank = int(np.ceil(self.count*p))
            if rank <= self.zero: return [0,0]
            c = self.zero
            for k,n in sorted(self.counts.items()):
                c += n
                if c >= rank:return [int(2**(k/128)),int(np.ceil(2**((k+1)/128)))]
            return None
        return dict(count=self.count,zero=self.zero,negative=self.negative,max_ns=self.maximum,
                    sum_ns=self.total,quantiles_ns={str(p):q(p) for p in (.5,.9,.95,.99,.999)},
                    method='full-population ranks, 1/128-octave bin bounds; no tail exclusion')


def rate_stats(counts):
    a = np.asarray(counts)
    return dict(seconds=len(a), zero_seconds=int((a==0).sum()),
                mean=float(a.mean()) if len(a) else None,
                quantiles={str(p):float(np.quantile(a,p,method='inverted_cdf')) if len(a) else None for p in (.5,.9,.95,.99,.999)},
                maximum=int(a.max()) if len(a) else None)


BASE_QUOTE = [(n,'int64') for n in ('t_ns','available_ns','source_ordinal','epoch','segment_id','bid_nano','ask_nano','bid_size','ask_size')]
BASE_QUOTE += [('valid','bool'),('reason_code','int16'),('source_flags','uint8'),('snapshot','bool'),('locked','bool'),('crossed','bool'),('clock_bad','bool'),('scheduled_closed','bool'),('reference_present','bool'),('activity_window','bool'),('routing_admissible','bool')]
BASE_QUOTE += [(n,'int64') for n in ('trade_buy','trade_sell','trade_unknown','trade_count','add_bid','add_ask','remove_bid','remove_ask')]
BASE_QUOTE += [(f'{side}_depth_{i}','int64') for side in ('bid','ask') for i in range(12)]
L3_SCHEMA = [(n,'int64') for n in ('available_ns','source_ordinal','ts_event_ns','ts_recv_ns','publication_ordinal','record_price_nano','record_size','resulting_price_nano','resulting_size')]
L3_SCHEMA += [('order_id','uint64'),('sequence','uint32'),('action','uint8'),('side','uint8'),('resulting_side','uint8'),('flags','uint8'),('accepted','bool'),('reason_code','int16')]
SECOND_SCHEMA = SNAP_SCHEMA + [('segment_id','int64'),('scheduled_closed','bool'),('reference_present','bool'),('activity_window','bool'),('routing_admissible','bool')]


class Candles:
    """Exact publication-price OHLC, not OHLC of a one-second subsample.

    Buckets are [start,end); inherited book at start is the opening mark.
    No bucket beyond the observed watermark is emitted at EOF.
    """
    columns=['t_ns','o','h','l','c','volume','coverage','epoch','valid','available_ns','segment_id']
    def __init__(self):
        self.cursor=None;self.end=None;self.mark=np.nan;self.support=False;self.segment=-1
        self.ohlc=[np.nan]*4;self.volume=0;self.coverage=0;self.segment_ids=set();self.rows=[]

    def advance(self,t):
        if self.cursor is None:return
        while self.cursor<t:
            until=min(t,self.end)
            if self.support:
                self.coverage+=until-self.cursor;self.segment_ids.add(self.segment)
            self.cursor=until
            if until==self.end:
                valid=self.coverage==60*NS and len(self.segment_ids)==1 and self.segment>=0
                self.rows.append([self.end,*self.ohlc,float(self.volume),self.coverage/(60*NS),self.segment,valid,t,self.segment if valid else -1])
                self.end+=60*NS
                self.ohlc=[self.mark]*4 if self.support else [np.nan]*4
                self.coverage=0;self.volume=0;self.segment_ids=set()

    def update(self,t,mid,supported,segment,volume):
        if self.cursor is None:
            self.cursor=t;self.end=(t//(60*NS)+1)*60*NS
        else:self.advance(t)
        self.support=bool(supported);self.mark=mid;self.segment=segment
        self.volume+=volume
        if supported:
            if not np.isfinite(self.ohlc[0]):self.ohlc=[mid]*4
            else:self.ohlc=[self.ohlc[0],max(self.ohlc[1],mid),min(self.ohlc[2],mid),mid]
            self.segment_ids.add(segment)

    def frame(self):
        if self.rows:return pd.DataFrame(self.rows,columns=self.columns)
        return pd.DataFrame({n:pd.Series(dtype='bool' if n=='valid' else 'float64' if n in ('o','h','l','c','volume','coverage') else 'int64') for n in self.columns})


class DayBuilder:
    def __init__(self, output, choice, reference, reserve, source_identity, initial_state=None):
        self.root = Path(output)
        self.root.mkdir(parents=True,exist_ok=False)
        self.choice, self.bounds = choice, session_bounds(choice['date'])
        self.day_start = pd.Timestamp(choice['date'],tz='UTC').value
        self.reference = reference
        self.reserve = reserve
        self.source_identity = source_identity
        self.engine = L3EventEngineV2.restore(initial_state) if initial_state else L3EventEngineV2(publisher_id=choice['route'][0],instrument_id=choice['route'][1],channel_id=choice['route'][2])
        self.initialized = initial_state is not None
        self.flow = FlowCounters()
        self.candles = Candles()
        self.quotes = Writer(self.root/'quotes.parquet',BASE_QUOTE,reserve)
        self.l3 = Writer(self.root/'l3.parquet',L3_SCHEMA,reserve)
        self.seconds = Writer(self.root/'seconds.parquet',SECOND_SCHEMA,reserve)
        self.reasons = {}; self.delta_reasons = {}
        self.checkpoints=[]; self.last_checkpoint=-1
        self.previous = None; self.previous_raw_time = None
        self.last_event = None; self.last_quote = None
        self.committed_depth = np.full(len(DEPTH),np.nan)
        self.next_second = None
        self.segment = -1; self.segments=[]; self.current_segment=None
        self.states=[]; self.current_state=None
        self.rate = np.zeros(86400,np.int64)
        self.raw_rate = np.zeros(86400,np.int64)
        self.bbo_rate = np.zeros(86400,np.int64)
        self.gaps=[]; self.raw_intervals=Intervals(); self.pub_intervals=Intervals(); self.active_intervals=Intervals()
        self.raw_counts=Counter(); self.raw_volume=Counter(); self.flags=Counter()
        self.latency=Intervals(); self.negative_latency=0
        self.live_raw_intervals=Intervals();self.live_previous=None
        self.clock_properties=Counter()
        self.publication_flags=Counter(); self.valid_count=self.invalid_count=0
        self.raw_records=0; self.raw_hash=hashlib.sha256(); self.received_last=-1
        self.started=time.perf_counter(); self.peak_rss=0
        times = self.day_start+np.arange(86400,dtype=np.int64)*NS
        self.closed_seconds,self.reference_seconds=flags_for_times(times,reference)
        self.active_seconds=(times>=self.bounds['london_ns'])&(times<self.bounds['ny_close_ns'])

    @staticmethod
    def code(mapping, reason):
        if reason not in mapping:mapping[reason]=len(mapping)
        return mapping[reason]

    def census(self, packed):
        b=packed['record']; times=b['ts_recv'].astype(np.int64)
        self.raw_hash.update(packed.tobytes())
        self.raw_records+=len(b)
        if len(b):
            if self.previous_raw_time is not None:self.raw_intervals.add([int(times[0])-self.previous_raw_time])
            self.raw_intervals.add(np.diff(times));self.previous_raw_time=int(times[-1])
        idx=(times-self.day_start)//NS
        np.add.at(self.raw_rate,idx[(idx>=0)&(idx<86400)],1)
        for action in np.unique(b['action']):
            for side in np.unique(b['side']):
                selected=(b['action']==action)&(b['side']==side)
                key=action.decode()+':'+side.decode()
                self.raw_counts[key]+=int(selected.sum());self.raw_volume[key]+=int(b['size'][selected].sum())
        for flag in (4,8,32,64,128):self.flags[str(flag)]+=int(((b['flags']&flag)!=0).sum())
        latency=times-b['ts_event'].astype(np.int64)
        self.negative_latency+=int((latency<0).sum());self.latency.add(latency[latency>=0])
        live=b[(b['flags']&32)==0]
        if len(live):
            for column,name,index in [('ts_recv','recv_regressions',0),('ts_event','event_regressions',1),('sequence','sequence_regressions',2)]:
                values=live[column].astype(np.int64)
                delta=np.diff(values,prepend=self.live_previous[index]) if self.live_previous else np.diff(values)
                self.clock_properties[name]+=int((delta<0).sum())
                if column=='ts_recv':self.live_raw_intervals.add(delta)
                if column=='sequence':self.clock_properties['sequence_forward_jumps_not_packet_loss']+=int((delta>1).sum())
            self.live_previous=tuple(int(live[-1][n]) for n in ('ts_recv','ts_event','sequence'))
            self.clock_properties['nonsnapshot_bad_recv']+=int(((live['flags']&8)!=0).sum())

    def _second_rows(self, before_ns):
        while self.next_second is not None and self.next_second < before_ns and self.next_second < self.day_start+DAY:
            t=self.next_second
            if self.last_event is not None:
                e=self.last_event; q=self.last_quote
                idx=max(0,min(86399,int((t-self.day_start)//NS)))
                b=e.book
                depth=self.committed_depth
                # No quote-age censorship. Age is a separate observed quantity.
                valid=bool(b.valid)
                bp,ap,bs,ass=q
                values=[bp/1e9,ap/1e9,(bp+ap)/2e9 if bp and ap else np.nan,bs,ass,*depth,*self.flow.values]
                self.seconds.append([t,b.continuity_epoch,valid,valid,t-e.available_at_ns,*values,
                    max(t,before_ns),self.segment,bool(self.closed_seconds[idx]),bool(self.reference_seconds[idx]),bool(self.active_seconds[idx]),not self.choice['context_only']])
                self.seconds.boundary()
            self.next_second+=NS

    def feed(self, r):
        if not self.initialized:
            if r.action!='R':raise ValueError('INITIAL_STATE_REQUIRED: no reset or predecessor checkpoint')
            self.initialized=True
        if r.ts_recv_ns < self.received_last:raise ValueError('Receive clock regression')
        if len(self.engine.pending)>=100000:raise ValueError('Unbounded unfinished F_LAST group')
        # Flush using the last COMMITTED book, before applying even partial events.
        # Pending groups may have mutated engine: use frozen depth below.
        self._second_rows(r.ts_recv_ns)
        self.candles.advance(r.ts_recv_ns)
        self.received_last=r.ts_recv_ns
        e=self.engine.append(r)
        if e is None:return
        previous=self.last_event
        b=e.book; bp,ap,bs,ass=self.engine._bbo_key()
        epoch_changed=previous is not None and b.continuity_epoch!=previous.book.continuity_epoch
        if epoch_changed:self.flow.reset()
        old=self.flow.values.copy()
        self.flow.update(e, previous.book if previous else e.book_before, previous is not None and not epoch_changed)
        increments=self.flow.values-old
        idx=max(0,min(86399,int((e.available_at_ns-self.day_start)//NS)))
        active=bool(self.active_seconds[idx])
        if previous is not None:
            gap=e.available_at_ns-previous.available_at_ns
            self.pub_intervals.add_one(gap)
            if active and self.bounds['london_ns']<=previous.available_at_ns<self.bounds['ny_close_ns']:self.active_intervals.add_one(gap)
            if gap>5*NS:self.gaps.append(dict(start_ns=previous.available_at_ns,end_ns=e.available_at_ns,duration_ns=gap,
                previous_ordinal=previous.last_source_ordinal,next_ordinal=e.last_source_ordinal,activity_window=active,cause='UNCLASSIFIED_OBSERVATION_PAUSE'))
        self.rate[idx]+=1
        if self.last_quote is not None and (bp,ap)!=self.last_quote[:2]:self.bbo_rate[idx]+=1
        flags=0
        for d in e.deltas:flags|=d.flags
        snapshot=any(d.flags&32 for d in e.deltas)
        locked=bool(bp and ap and bp==ap);crossed=bool(bp and ap and bp>ap)
        self.publication_flags[b.reason]+=1
        self.valid_count+=int(b.valid);self.invalid_count+=int(not b.valid)
        # Segment membership depends on observable support, never OHLC or silence.
        if self.current_segment is not None and (not b.valid or epoch_changed):
            self.current_segment['end_ns']=e.available_at_ns
            self.current_segment['end_reason']=b.reason if not b.valid else 'EPOCH_CHANGE'
            self.segments.append(self.current_segment);self.current_segment=None
        if b.valid and self.current_segment is None:
            self.segment+=1
            self.current_segment=dict(segment_id=self.segment,start_ns=e.available_at_ns,epoch=b.continuity_epoch,first_ordinal=e.last_source_ordinal)
        self.candles.update(e.available_at_ns,(bp+ap)/2e9 if bp and ap else np.nan,b.valid,self.segment if b.valid else -1,int(sum(increments[3:6])))
        state=(b.reason,locked,crossed,bool(flags&4),bool(flags&8),snapshot)
        if self.current_state is None or self.current_state['_key']!=state:
            if self.current_state:
                self.current_state['end_ns']=e.available_at_ns;self.states.append(self.current_state)
            self.current_state=dict(_key=state,start_ns=e.available_at_ns,reason=b.reason,locked=locked,crossed=crossed,source_bad=bool(flags&4),clock_bad=bool(flags&8),snapshot=snapshot,first_ordinal=e.last_source_ordinal)
        depth=[]
        for levels,best,sign in ((self.engine.bids,bp,-1),(self.engine.asks,ap,1)):
            for k in range(12):
                level=levels.get(best+sign*k*100000000) if best else None
                depth.append(level.size if level else 0)
        flow_indices=(3,4,5,6,7,8,9,10)
        self.quotes.append([e.event_ts_ns,e.available_at_ns,e.last_source_ordinal,b.continuity_epoch,self.segment if b.valid else -1,
            bp,ap,bs,ass,b.valid,self.code(self.reasons,b.reason),flags,snapshot,locked,crossed,bool(flags&8),
            bool(self.closed_seconds[idx]),bool(self.reference_seconds[idx]),active,not self.choice['context_only'],
            *(int(increments[k]) for k in flow_indices),*depth])
        for d in e.deltas:
            self.l3.append([e.available_at_ns,d.source_ordinal,d.ts_event_ns,d.ts_recv_ns,e.last_source_ordinal,
                d.record_price_nano,d.record_size,d.resulting_price_nano,d.resulting_size,d.order_id,d.sequence,
                ord(d.action),ord(d.side) if d.side else 0,ord(d.resulting_side) if d.resulting_side else 0,d.flags,d.accepted,self.code(self.delta_reasons,d.reason)])
        self.quotes.boundary();self.l3.boundary()
        self.last_event=e;self.last_quote=(bp,ap,bs,ass)
        # Frozen committed depth is needed if the next group spans a clock bucket.
        # Build from the already gathered ladder, avoiding another book traversal.
        if b.valid:
            quantities=[]
            for ladder in (depth[:12],depth[12:]):
                quantities.extend(sum(ladder[:n]) for n in (1,4,12))
            for ladder in (depth[12:],depth[:12]):
                for size in (1,5,10):
                    remaining=size;cost=0.
                    for k,amount in enumerate(ladder):
                        filled=min(remaining,amount);cost+=filled*k*.1;remaining-=filled
                        if not remaining:break
                    quantities.append(cost/size if not remaining else np.nan)
            self.committed_depth=np.asarray(quantities)
        else:self.committed_depth=np.full(len(DEPTH),np.nan)
        if self.next_second is None:self.next_second=(e.available_at_ns//NS+1)*NS
        if self.last_checkpoint<0 or e.available_at_ns-self.last_checkpoint>=1800*NS or snapshot:
            # Snapshot rows only become a checkpoint at the completed F_LAST.
            self.checkpoint(e)
        if self.engine.completed_events%10000==0:
            # Remove stale heap entries without changing order state.
            import heapq
            self.engine.bid_heap=[-p for p in self.engine.bids];heapq.heapify(self.engine.bid_heap)
            self.engine.ask_heap=list(self.engine.asks);heapq.heapify(self.engine.ask_heap)

    def checkpoint(self,e):
        name=f'checkpoint_{len(self.checkpoints):04d}.json.zst'
        state=self.engine.checkpoint()
        payload=json.dumps(state,separators=(',',':')).encode()
        (self.root/name).write_bytes(zstandard.ZstdCompressor(level=3).compress(payload))
        self.checkpoints.append(dict(file=name,available_ns=e.available_at_ns,source_ordinal=e.last_source_ordinal))
        self.last_checkpoint=e.available_at_ns

    def finish(self, expected_entries, raw_source_sha, code):
        if self.engine.pending:raise ValueError('UNFINISHED_F_LAST_AT_SOURCE_END')
        if not self.last_event:raise ValueError('NO_COMPLETED_PUBLICATIONS')
        end=self.last_event.available_at_ns
        if self.current_segment:
            self.current_segment.update(end_ns=end,end_reason='OBSERVATION_END',last_ordinal=self.last_event.last_source_ordinal)
            self.segments.append(self.current_segment)
        if self.current_state:self.current_state['end_ns']=end;self.states.append(self.current_state)
        for s in self.states:s.pop('_key')
        self.checkpoint(self.last_event)
        layers={n:w.close() for n,w in [('quotes',self.quotes),('l3',self.l3),('seconds',self.seconds)]}
        expected=sum(e['records'] for e in expected_entries)
        if self.raw_records!=expected:raise ValueError('Selected record count mismatch')
        if len(expected_entries)==1 and self.raw_hash.hexdigest()!=expected_entries[0]['sha256']:raise ValueError('Selected source hash mismatch')
        save_json(self.root/'segments.json',self.segments)
        save_json(self.root/'states.json',self.states)
        save_json(self.root/'gaps.json',self.gaps)
        rates=pa.table(dict(t_ns=self.day_start+np.arange(86400,dtype=np.int64)*NS,
            publications=self.rate,raw_records=self.raw_rate,bbo_price_changes=self.bbo_rate,
            activity_window=self.active_seconds,scheduled_closed=self.closed_seconds,reference_present=self.reference_seconds))
        pq.write_table(rates,self.root/'rates.parquet',compression='zstd')
        frame=pq.read_table(self.root/'seconds.parquet').to_pandas()
        bars=self.candles.frame()
        layers['bars']=write_frame(self.root/'bars.parquet',bars,self.reserve)
        diagnostics=dict(date=self.choice['date'],symbol=self.choice['symbol'],raw_records=self.raw_records,
            publications=layers['quotes']['rows'],valid_publications=self.valid_count,invalid_publications=self.invalid_count,
            raw_counts=dict(self.raw_counts),raw_quantities=dict(self.raw_volume),raw_flags=dict(self.flags),publication_reasons=dict(self.publication_flags),
            raw_intervals=self.raw_intervals.result(),publication_intervals=self.pub_intervals.result(),activity_publication_intervals=self.active_intervals.result(),
            nonsnapshot_raw_intervals=self.live_raw_intervals.result(),clock_properties=dict(self.clock_properties),
            recv_minus_event_nonnegative=self.latency.result(),negative_recv_minus_event=self.negative_latency,
            rates={label:dict(publications=rate_stats(self.rate[mask]),raw_records=rate_stats(self.raw_rate[mask]),bbo_changes=rate_stats(self.bbo_rate[mask])) for label,mask in
                [('utc_day',np.ones(86400,bool)),('london_ny',self.active_seconds),('london_ny_reference',self.active_seconds&self.reference_seconds)]},
            seconds_metrics=second_metrics(frame),long_pauses=len(self.gaps),segments=len(self.segments),
            calendar_semantics='NY 17-18 regular convention; not verified historical exchange status; absent OHLC is reference absence only',
            observation_start_ns=self.checkpoints[0]['available_ns'],observation_end_ns=end,
            clock='provider receive/publication availability, not measured live latency',
            source_quality=self.choice['source_condition'],elapsed_seconds=time.perf_counter()-self.started,
            peak_rss_bytes=max(self.peak_rss,psutil.Process().memory_info().rss))
        save_json(self.root/'diagnostics.json',diagnostics)
        payloads={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in self.root.iterdir() if p.is_file()}
        m=dict(contract=VERSION,status='COMPLETE',layers=layers,payloads=payloads,checkpoints=self.checkpoints,
               reasons=self.reasons,delta_reasons=self.delta_reasons,selection=self.choice,session=self.bounds,
               source_identity=self.source_identity,raw_source_sha256=raw_source_sha,selected_sha256=self.raw_hash.hexdigest(),
               implementation=code,segment_end_semantics='censored observation, never a forced exit or loss',
               diagnostics_are_future=True,depth_unit='native GC contracts; 12 price steps of $0.10 per oz',
               bar_basis='Exact completed-publication midpoint; [start,end), inherited opening book; no partial EOF bar',
               legacy_quote_age_threshold=None)
        m['identity']=digest(m);save_json(self.root/'manifest.json',m)
        verify(self.root)
        return m


def write_frame(path, frame, reserve):
    writer=Writer(path,[(n,str(frame[n].dtype)) for n in frame.columns],reserve)
    for row in frame.itertuples(index=False,name=None):writer.append(row);writer.boundary()
    return writer.close()


def second_metrics(frame):
    result={}
    for label,g in [('utc_day',frame),('london_ny',frame[frame.activity_window])]:
        v=g[g.valid]
        def quant(a):
            a=np.asarray(a,float);a=a[np.isfinite(a)]
            return {str(p):float(np.quantile(a,p)) if len(a) else None for p in (.5,.95,.99)}
        changes=v.mid.diff().to_numpy()
        adjacent=(v.t_ns.diff().to_numpy()==NS)&(v.segment_id.diff().to_numpy()==0)
        changes=changes[adjacent]
        result[label]=dict(seconds=len(g),supported_seconds=len(v),reference_seconds=int(g.reference_present.sum()),
            scheduled_closed_seconds=int(g.scheduled_closed.sum()),spread_usd_per_oz=quant(v.ask-v.bid),
            depth_bid12_contracts=quant(v.depth_bid_12),depth_ask12_contracts=quant(v.depth_ask_12),
            range_usd_per_oz=float(v.mid.max()-v.mid.min()) if len(v) else None,
            realized_price_variation=float(np.square(changes).sum()),quote_age_ns=quant(v.age_ns))
    return result


def prepare_member(config_path, member_key, output, reserve=RESERVE):
    config=read_json(config_path);origin=Path(config['root'])
    stage=read_json(origin/'staging'/member_key/'manifest.json')
    item=stage['item'];code=implementation();output=Path(output)
    expected_identity=dict(member=item['member'],sha256=item['sha256'],key=member_key)
    if (output/'manifest.json').exists():
        m=read_json(output/'manifest.json')
        if m.get('source_identity')!=expected_identity or m.get('implementation')!=code:raise ValueError('Existing publication has different inputs or implementation')
        verify(output);return dict(root=str(output),reused=True,identity=m['identity'])
    # Never overwrite unfinished work. Its identity and error remain inspectable.
    temp=output.with_name(output.name+'.building')
    if temp.exists():raise ValueError('Existing unfinished build: '+str(temp))
    choices=stage['choices']
    if len(choices)!=1:raise ValueError('Daily source member required; no invented day reset')
    day,choice=next(iter(choices.items()));route=choice['route']
    reference=np.load(config['reference']['path'],mmap_mode='r')
    builder=DayBuilder(temp,choice,reference,reserve,expected_identity)
    src=DbnSource(item['archive'],item['member']);offset=0
    try:
        for block in src:
            indices=np.flatnonzero((block['publisher_id']==route[0])&(block['instrument_id']==route[1])&(block['channel_id']==route[2]))
            selected=np.empty(len(indices),PACKED);selected['source_ordinal']=indices+offset;selected['record']=block[indices]
            builder.census(selected)
            for r in packed_records(selected,route):builder.feed(r)
            offset+=len(block)
            builder.peak_rss=max(builder.peak_rss,psutil.Process().memory_info().rss)
            save_json(temp/'progress.json',dict(pid=os.getpid(),records=builder.raw_records,expected=stage['selected_records'],seconds=time.perf_counter()-builder.started))
        if src.sha256!=item['sha256']:raise ValueError('Raw source member hash mismatch')
        (temp/'progress.json').unlink(missing_ok=True)
        m=builder.finish(stage['entries'],src.sha256,code)
        from .finalized_audit import compare_canonical, independent_prefix
        master=read_json(origin/'manifest.json')
        captures=[c for c in master['captures'] if c['entry']['member_key']==member_key and c['status']=='COMPLETE']
        if len(captures)!=1:raise ValueError('Expected one canonical daily capture for admission')
        acceptance=dict(canonical=compare_canonical(temp,captures[0]['market']),independent=independent_prefix(temp,config_path))
        save_json(temp/'acceptance.json',acceptance)
        p=temp/'acceptance.json';m['payloads'][p.name]=dict(bytes=p.stat().st_size,sha256=sha(p))
        m.pop('identity');m['identity']=digest(m);save_json(temp/'manifest.json',m)
        verify(temp)
        temp.rename(output)
        return dict(root=str(output),identity=m['identity'],reused=False)
    except BaseException as exc:
        save_json(temp/'failure.json',dict(status='FAILED',error=repr(exc),source=expected_identity))
        for w in (builder.quotes,builder.l3,builder.seconds):
            try:w.writer.close()
            except Exception:pass
        raise


def catalog(root, config_path):
    root=Path(root);config=read_json(config_path)
    expected=[m for m in config['members'] if 'Aug-1-Sep-2' in m['archive']]
    entries=[]
    for member in expected:
        folder=root/member['key']
        if (folder/'manifest.json').exists():
            m=read_json(folder/'manifest.json');d=read_json(folder/'diagnostics.json')
            entries.append(dict(key=member['key'],status='COMPLETE',date=m['selection']['date'],symbol=m['selection']['symbol'],identity=m['identity'],diagnostics=d,bytes=sum(x['bytes'] for x in m['payloads'].values())))
        else:entries.append(dict(key=member['key'],status='NOT_PUBLISHED',member=member['member']))
    result=dict(contract=VERSION,status='COMPLETE' if all(e['status']=='COMPLETE' for e in entries) else 'PARTIAL',days=entries,
        expected_members=len(expected),training_admission=False,source_scope='2026-08-02 through 2026-09-01; not all September',
        calendar_days=[dict(date=str(d.date()),source_members=[m['key'] for m in expected if str(d.date()).replace('-','') in m['member']],
            status='MEMBER_PRESENT' if any(str(d.date()).replace('-','') in m['member'] for m in expected) else 'NO_FILE_WEEKEND' if d.weekday()>=5 else 'NO_FILE_UNCLASSIFIED') for d in pd.date_range('2026-08-01','2026-09-01')])
    save_json(root/'catalog.json',result)
    rows=[]
    for e in entries:
        d=e.get('diagnostics');r=d['rates']['london_ny']['publications'] if d else None
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [e.get('date',e['key']),e['status'],e.get('symbol',''),d['publications'] if d else '',r['quantiles']['0.5'] if r else '',r['zero_seconds'] if r else '',d['invalid_publications'] if d else '',d['long_pauses'] if d else '',round(e.get('bytes',0)/1024**2,1)])+'</tr>')
    page='''<!doctype html><meta charset="utf-8"><title>Finalized market V2</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;background:#101825;color:#e7edf6}td,th{padding:8px;border-bottom:1px solid #354456;text-align:right}th:first-child,td:first-child{text-align:left}a{color:#8bcaff}</style><h1>Finalized market V2 — август / сентябрь</h1><p>Диагностика вычисляется при подготовке. Replay читает готовые блоки. Дневные метрики содержат будущее и не являются входами модели.</p><p>Окно активности: 08:00 Europe/London — 17:00 America/New_York. Отсутствие OHLC не означает невалидность MBO; пауза не означает потерю потока.</p><table><tr><th>Дата</th><th>Статус</th><th>Контракт</th><th>Публикации</th><th>Медиана/с L–NY</th><th>Нулевые с L–NY</th><th>Невалидные</th><th>Паузы &gt;5с</th><th>MiB</th></tr>'''+''.join(rows)+'</table><p><a href="catalog.json">Полная сводка JSON</a></p>'
    (root/'index.html').write_text(page,encoding='utf8')
    return result


def run(config_path, output, members=None, workers=6, reserve=RESERVE):
    if not 1<=workers<=6:raise ValueError('workers must be 1..6')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    config=read_json(config_path)
    selected=[m for m in config['members'] if 'Aug-1-Sep-2' in m['archive'] and (members is None or m['key'] in members)]
    if members and set(members)!={m['key'] for m in selected}:raise ValueError('Unknown member selection')
    # Pilot before full scheduling; estimate from largest measured output/record.
    measured=[]
    for p in root.glob('member_*/manifest.json'):
        m=read_json(p);d=read_json(p.parent/'diagnostics.json')
        measured.append(sum(x['bytes'] for x in m['payloads'].values())/d['raw_records'])
    needed=[]
    for m in selected:
        st=read_json(Path(config['root'])/'staging'/m['key']/'manifest.json')
        needed.append((m,st['selected_records']))
    pending_records=sum(n for m,n in needed if not (root/m['key']/'manifest.json').exists())
    if len(selected)>2:
        if not measured:raise RuntimeError('PILOT_REQUIRED: prepare a weekday and boundary capture first')
        estimate=int(pending_records*max(measured)*1.3)
        free=psutil.disk_usage(str(root)).free
        save_json(root/'space_budget.json',dict(estimated_remaining_bytes=estimate,reserve_bytes=reserve,free_bytes=free,bytes_per_record=max(measured),safety_factor=1.3))
        if free<estimate+reserve:
            save_json(root/'run.json',dict(status='BLOCKED_DISK_BUDGET',required_bytes=estimate+reserve,free_bytes=free))
            catalog(root,config_path)
            return read_json(root/'run.json')
    start=time.perf_counter();jobs={};errors=[]
    # One pool; conservative per-worker RAM reservation, and global output budget.
    workers=min(workers,max(1,int(psutil.virtual_memory().available//(2*1024**3))))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for m,n in needed:
            jobs[pool.submit(prepare_member,str(config_path),m['key'],str(root/m['key']),reserve)]=m['key']
        while jobs:
            done,_=wait(jobs,timeout=5,return_when=FIRST_COMPLETED)
            for f in done:
                key=jobs.pop(f)
                try:print(json.dumps(f.result()),flush=True)
                except Exception as exc:errors.append(dict(member=key,error=repr(exc)));print(json.dumps(errors[-1]),flush=True)
            save_json(root/'run.json',dict(status='RUNNING',pid=os.getpid(),workers=workers,remaining=len(jobs),errors=errors,elapsed_seconds=time.perf_counter()-start))
    c=catalog(root,config_path)
    result=dict(status='FAILED' if errors else c['status'],errors=errors,elapsed_seconds=time.perf_counter()-start,workers=workers)
    save_json(root/'run.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='data/derived/full_mbo_v1/run_config.json')
    p.add_argument('--output',default='data/derived/finalized_market_v2')
    p.add_argument('--members',nargs='+');p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();print(json.dumps(run(a.config,a.output,a.members,a.workers),indent=2))


if __name__=='__main__':main()
