"""Full-source, bounded-memory data audit. Never modifies market/training data.

Quantiles are deterministic population histogram bounds, not sampled estimates.
Raw DBN records, completed publications and actual BBO changes stay distinct.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from spectrum.source import DbnSource
from spectrum.ingest import PACKED, save_json
from spectrum.core.storage import sha

NS = 10**9
MINUTE = 60*NS
PAIR = np.dtype([('ordinal', '<i8'), ('recv', '<i8')])


class Distribution:
    """Exact counts in 1/128-octave bins; zeros and negatives are explicit."""
    def __init__(self):
        self.hist = np.zeros(8194, np.int64)
        self.negative = 0
        self.maximum = 0
        self.thresholds = {n: 0 for n in (1_000_000, NS, 5*NS, 30*NS, 60*NS)}
        self.tail_ns = {n: 0 for n in self.thresholds}
        self.total_ns = 0

    def add(self, values):
        x = np.asarray(values, np.int64)
        self.negative += int((x < 0).sum())
        x = x[x >= 0]
        if not len(x):
            return
        self.maximum = max(self.maximum, int(x.max()))
        # Per-block sums are below int64 limits for this timestamp domain.
        self.total_ns += int(x.sum())
        bins = np.zeros(len(x), np.int64)
        pos = x > 0
        bins[pos] = 1 + np.floor(np.log2(x[pos])*128).astype(np.int64)
        self.hist += np.bincount(bins, minlength=len(self.hist))
        for n in self.thresholds:
            tail = x[x > n]
            self.thresholds[n] += len(tail)
            self.tail_ns[n] += int(tail.sum())

    def result(self):
        count = int(self.hist.sum())
        cumulative = np.cumsum(self.hist)
        quantiles = {}
        for q in (.01, .1, .5, .9, .95, .99, .999):
            if not count:
                quantiles[str(q)] = None
                continue
            # Inverse empirical CDF: exact rank, bounded integer-ns value.
            j = int(np.searchsorted(cumulative, max(1, int(np.ceil(q*count)))))
            bounds = [0, 0] if j == 0 else [int(np.ceil(2**((j-1)/128))), int(np.ceil(2**(j/128)))-1]
            quantiles[str(q)] = bounds
        return dict(count=count, zeros=int(self.hist[0]), negative=self.negative,
                    max_ns=self.maximum, total_interval_seconds=self.total_ns/NS,
                    quantile_ns_bounds=quantiles,
                    tail_counts_gt_ns={str(k): int(v) for k,v in self.thresholds.items()},
                    tail_duration_seconds={str(k): v/NS for k,v in self.tail_ns.items()})


def interarrival(stats, times, previous=None):
    times = np.asarray(times, np.int64)
    if not len(times):
        return previous
    stats.add(np.diff(times, prepend=previous) if previous is not None else np.diff(times))
    return int(times[-1])


def publication_hash(ordinals, times):
    pairs = np.empty(len(ordinals), PAIR)
    pairs['ordinal'], pairs['recv'] = ordinals, times
    return pairs.tobytes()


def raw_member(job):
    member, captures, output = job
    started = time.perf_counter()
    all_dt, normal_dt, latency = Distribution(), Distribution(), Distribution()
    last_all = last_normal = None
    action_counts, flag_counts, side_counts = Counter(), Counter(), Counter()
    states = {}
    for capture in captures:
        states[capture['key']] = dict(entry=capture['entry'], capture=capture,
            digest=hashlib.sha256(), pub_digest=hashlib.sha256(), records=0, publications=0,
            dt=Distribution(), pub_dt=Distribution(), last=None, pub_last=None,
            flags=Counter(), actions=Counter(), sides=Counter(), negative_event_latency=0)
    source = DbnSource(member['archive'], member['member'], block_records=262144)
    offset = 0
    for block in source:
        recv = block['ts_recv'].astype(np.int64)
        last_all = interarrival(all_dt, recv, last_all)
        normal = (block['flags'] & 32) == 0
        last_normal = interarrival(normal_dt, recv[normal], last_normal)
        latency.add(recv[normal]-block['ts_event'][normal].astype(np.int64))
        for name, counter in (('action', action_counts), ('side', side_counts), ('flags', flag_counts)):
            keys, counts = np.unique(block[name], return_counts=True)
            counter.update({str(k.decode() if isinstance(k, bytes) else int(k)): int(v) for k,v in zip(keys,counts)})
        end = offset + len(block)
        for state in states.values():
            e = state['entry']
            if int(e['last_ordinal']) < offset or int(e['first_ordinal']) >= end:
                continue
            lo, hi = max(0, int(e['first_ordinal'])-offset), min(len(block), int(e['last_ordinal'])-offset+1)
            segment = block[lo:hi]
            route = e['route']
            indices = np.flatnonzero((segment['publisher_id']==route[0]) &
                (segment['instrument_id']==route[1]) & (segment['channel_id']==route[2]))
            selected = segment[indices]
            ordinals = indices + lo + offset
            packed = np.empty(len(selected), PACKED)
            packed['source_ordinal'], packed['record'] = ordinals, selected
            state['digest'].update(packed.tobytes())
            state['records'] += len(selected)
            regular = (selected['flags'] & 32) == 0
            state['last'] = interarrival(state['dt'], selected['ts_recv'][regular], state['last'])
            terminal = (selected['flags'] & 128) != 0
            state['publications'] += int(terminal.sum())
            state['pub_digest'].update(publication_hash(ordinals[terminal], selected['ts_recv'][terminal]))
            live_terminal = terminal & regular
            state['pub_last'] = interarrival(state['pub_dt'], selected['ts_recv'][live_terminal], state['pub_last'])
            for name in ('flags', 'action', 'side'):
                counter = state[{'flags':'flags', 'action':'actions', 'side':'sides'}[name]]
                keys, counts = np.unique(selected[name], return_counts=True)
                counter.update({str(k.decode() if isinstance(k, bytes) else int(k)): int(v) for k,v in zip(keys,counts)})
        offset = end
    results = []
    histograms = {'all_raw':all_dt.hist, 'non_snapshot_raw':normal_dt.hist, 'latency':latency.hist}
    for key, s in states.items():
        e = s['entry']
        results.append(dict(key=key, date=e['selection']['date'], symbol=e['selection']['symbol'],
            records=s['records'], expected_records=e['records'], raw_partition_sha256=s['digest'].hexdigest(),
            raw_partition_matches=s['digest'].hexdigest()==e['sha256'] and s['records']==e['records'],
            publications=s['publications'], publication_identity_sha256=s['pub_digest'].hexdigest(),
            flags=dict(s['flags']), actions=dict(s['actions']), sides=dict(s['sides']),
            non_snapshot_record_interarrival=s['dt'].result(), non_snapshot_publication_interarrival=s['pub_dt'].result()))
        histograms[key.replace('/','_')+'_records'] = s['dt'].hist
        histograms[key.replace('/','_')+'_publications'] = s['pub_dt'].hist
    out = Path(output)
    np.savez_compressed(out/(member['key']+'_hist.npz'), **histograms)
    result = dict(member=member['key'], source_member=member['member'], records=offset,
        compressed_sha256=source.sha256, source_hash_matches=source.sha256==member['sha256'],
        all_record_interarrival=all_dt.result(), non_snapshot_record_interarrival=normal_dt.result(),
        non_snapshot_recv_minus_event=latency.result(), actions=dict(action_counts),
        flags=dict(flag_counts), sides=dict(side_counts), captures=results, seconds=time.perf_counter()-started)
    save_json(out/(member['key']+'.json'), result)
    return dict(member=member['key'], records=offset, hash_ok=result['source_hash_matches'], seconds=result['seconds'])


def market_capture(job):
    capture, reference_path, output = job
    root = Path(capture['market'])
    q = pq.read_table(root/'quotes.parquet').to_pandas()
    t = q.t_ns.to_numpy(np.int64)
    av = q.available_ns.to_numpy(np.int64)
    epoch = q.epoch.to_numpy(np.int64)
    valid = q.valid.to_numpy(bool)
    dt = np.diff(av)
    ref = np.load(reference_path, mmap_mode='r')
    minute = t//MINUTE
    idx = np.searchsorted(ref, minute)
    reference_known = (idx<len(ref)) & (ref[np.minimum(idx,len(ref)-1)]==minute)
    ref_gap_prefix = np.r_[0, np.cumsum(np.diff(ref)>1)]
    adjacent_reference = reference_known[1:] & reference_known[:-1] & (ref_gap_prefix[np.minimum(idx[1:],len(ref)-1)]==ref_gap_prefix[np.minimum(idx[:-1],len(ref)-1)])
    continuous = valid[1:] & valid[:-1] & (epoch[1:]==epoch[:-1])
    date = capture['entry']['selection']['date']
    london = pd.Timestamp(date+' 08:00', tz='Europe/London').tz_convert('UTC').value
    ny = pd.Timestamp(date+' 17:00', tz='America/New_York').tz_convert('UTC').value
    trade_hours = (av>=london) & (av<ny)
    distributions = {}
    for name, x in (
        ('all_publications', dt), ('valid_same_epoch',dt[continuous]),
        ('reference_supported',dt[continuous & adjacent_reference]),
        ('london_ny',dt[continuous & adjacent_reference & trade_hours[1:] & trade_hours[:-1]])):
        stat = Distribution(); stat.add(x); distributions[name] = stat
    changed_price = np.r_[True,(np.diff(q.bid_nano)!=0)|(np.diff(q.ask_nano)!=0)]
    changed_top = changed_price | np.r_[False,(np.diff(q.bid_size)!=0)|(np.diff(q.ask_size)!=0)]
    for name, changed in (('bbo_price_changes',changed_price),('bbo_price_or_size_changes',changed_top)):
        ix = np.flatnonzero(changed & valid & reference_known & trade_hours)
        x = np.diff(av[ix])
        bad_prefix = np.r_[0,np.cumsum(~(valid & reference_known))]
        good = (epoch[ix[1:]]==epoch[ix[:-1]]) & (bad_prefix[ix[1:]+1]==bad_prefix[ix[:-1]+1])
        stat = Distribution(); stat.add(x[good]); distributions[name] = stat
    # Updates per wall-clock second, including zero-activity reference-supported seconds.
    start = pd.Timestamp(date, tz='UTC').value
    sec = ((av-start)//NS).astype(np.int64)
    counts = np.bincount(sec[(sec>=0)&(sec<86400)], minlength=86400)
    grid = start+np.arange(86400,dtype=np.int64)*NS
    mi = np.searchsorted(ref,grid//MINUTE)
    supported_seconds = (mi<len(ref)) & (ref[np.minimum(mi,len(ref)-1)]==grid//MINUTE) & (grid>=london)&(grid<ny)
    rate = counts[supported_seconds]
    gap_ix = np.flatnonzero((dt>5*NS) & continuous & adjacent_reference & trade_hours[1:] & trade_hours[:-1])+1
    gaps = [dict(left_ns=str(int(av[i-1])),right_ns=str(int(av[i])),seconds=float(dt[i-1]/NS),
                 left_ordinal=str(int(q.source_ordinal.iloc[i-1])),right_ordinal=str(int(q.source_ordinal.iloc[i])),
                 bbo_price_changed=bool(changed_price[i])) for i in gap_ix]
    checkpoint=json.loads((root/'checkpoint.json').read_text())
    # The checkpoint is sealed; read the persisted configuration, not current defaults.
    support_ns=checkpoint['state']['core']['state']['config']['max_quote_age_ns']
    hashes={name:sha(root/name)==digest for name,digest in capture['market_payloads'].items()}
    seconds=pq.read_table(root/'seconds.parquet',columns=['t_ns','valid','available_ns']).to_pandas()
    bars=pq.read_table(root/'bars.parquet',columns=['t_ns','valid','coverage']).to_pandas()
    result=dict(key=capture['key'],date=date,symbol=capture['entry']['selection']['symbol'],
        payload_hashes_match=hashes, quote_rows=len(q), source_ordinal_nonincreasing=int((np.diff(q.source_ordinal)<=0).sum()),
        recv_regressions=int((np.diff(t)<0).sum()), publication_regressions=int((dt<0).sum()),
        publication_before_source=int((av<t).sum()), invalid_quotes=int((~valid).sum()),
        valid_crossed_or_empty=int((valid & ((q.bid_nano<=0)|(q.ask_nano<q.bid_nano))).sum()),
        epoch_changes=int((np.diff(epoch)!=0).sum()), persisted_support_ns=support_ns,
        valid_quotes_without_spot_reference=int((valid & ~reference_known).sum()),
        valid_london_ny_quotes_without_spot_reference=int((valid & ~reference_known & trade_hours).sum()),
        publication_identity_sha256=hashlib.sha256(publication_hash(q.source_ordinal.to_numpy(),t)).hexdigest(),
        distributions={name:stat.result() for name,stat in distributions.items()},
        london_ny_per_second=dict(seconds=len(rate),zero_seconds=int((rate==0).sum()),
            median=float(np.median(rate)) if len(rate) else None,
            mean=float(np.mean(rate)) if len(rate) else None,
            p99=float(np.quantile(rate,.99)) if len(rate) else None,max=int(rate.max()) if len(rate) else None),
        gaps_over_5s_london_ny=gaps, seconds_rows=len(seconds), valid_seconds=int(seconds.valid.sum()),
        delayed_seconds_over_1s=int(((seconds.available_ns-seconds.t_ns)>NS).sum()),
        bars_rows=len(bars), valid_bars=int(bars.valid.sum()))
    name=capture['key'].replace('/','_')
    save_json(Path(output)/(name+'.json'),result)
    np.savez_compressed(Path(output)/(name+'_hist.npz'),**{name:stat.hist for name,stat in distributions.items()})
    return dict(key=capture['key'],rows=len(q),hash_ok=all(hashes.values()))


def run(root, workers=4):
    root=Path(root);root.mkdir(exist_ok=False,parents=True)
    config=json.loads(Path('data/derived/full_mbo_v1/run_config.json').read_text())
    master=json.loads(Path('data/derived/full_mbo_v1/manifest.json').read_text())
    for name in ('raw','market'):(root/name).mkdir()
    started=time.perf_counter()
    save_json(root/'protocol.json',dict(status='RUNNING',scope='All configured raw members including August and all COMPLETE captures',
        quantiles='Exact population rank bounded in integer nanoseconds by 1/128-octave histogram bins; no sampling or tail trimming',
        clocks='Receive/publication intervals, distinct from exchange event timestamp and BBO changes',
        reference='Spot OHLC absence is unavailable reference, NOT proof of exchange closure',
        input_manifest_sha256=sha('data/derived/full_mbo_v1/manifest.json'),workers=workers))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[(m,[r for r in master['captures'] if r['entry'].get('member_key')==m['key']],str(root/'raw')) for m in config['members']]
        futures=[pool.submit(raw_member,j) for j in jobs]
        for future in as_completed(futures):print(json.dumps(dict(stage='raw',**future.result())),flush=True)
        jobs=[(r,config['reference']['path'],str(root/'market')) for r in master['captures'] if r['status']=='COMPLETE']
        futures=[pool.submit(market_capture,j) for j in jobs]
        for future in as_completed(futures):print(json.dumps(dict(stage='market',**future.result())),flush=True)
    save_json(root/'scan_complete.json',dict(status='COMPLETE',seconds=time.perf_counter()-started,
        raw_members=len(config['members']),market_captures=len(jobs)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.output,a.workers)
