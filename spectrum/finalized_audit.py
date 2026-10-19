"""Administrative acceptance/benchmark, never imported by the hot reader."""
from pathlib import Path
import argparse
import json
import time
import statistics
import numpy as np
import pyarrow.parquet as pq
from .finalized import FinalizedReplay, restore_l3, verify, read_json
from .ingest import save_json


def benchmark(root, legacy, rows=200000, repeats=5):
    from .tape import TapeReplay
    rows=min(rows,FinalizedReplay(root).spec['rows'])
    def fast(n):
        r=FinalizedReplay(root,columns=['t_ns','available_ns','source_ordinal','epoch','bid_nano','ask_nano','bid_size','ask_size','valid']);count=0
        while count<n:
            table=r.read_batch(min(65536,n-count))
            if table is None:break
            # Materialize numeric arrays, not just lazy metadata.
            for name in table.column_names:table[name].to_numpy()
            count+=len(table)
        return count
    def slow(n):
        r=TapeReplay(legacy,verify=False);count=0
        for row in r:
            count+=1
            if count==n:break
        return count
    results={}
    for name,fn in [('columnar',fast),('legacy_dict',slow)]:
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter();assert fn(rows)==rows;samples.append(time.perf_counter()-start)
        results[name]=dict(first_observed_seconds=samples[0],seconds=samples,median_seconds=statistics.median(samples))
    sizes=[]
    for n in (rows//2,rows):
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter();fast(n);samples.append(time.perf_counter()-start)
        sizes.append(dict(rows=n,median_seconds=statistics.median(samples)))
    results.update(rows=rows,repeats=repeats,speedup=results['legacy_dict']['median_seconds']/results['columnar']['median_seconds'],
                   scaling=sizes,scaling_ratio=sizes[1]['median_seconds']/sizes[0]['median_seconds'],
                   cache_note='First observed read reported separately; OS cache was not forcibly flushed, so no cold-cache claim',
                   legacy_verification=False)
    return results


def compare_canonical(root, legacy):
    """All row identities and valid BBO values, bounded row-group batches."""
    new=FinalizedReplay(root,columns=['source_ordinal','available_ns','valid','bid_nano','ask_nano','bid_size','ask_size'])
    old=pq.ParquetFile(Path(legacy)/'quotes.parquet')
    old_rows=old.metadata.num_rows
    if old_rows!=new.spec['rows']:raise ValueError('Publication count mismatch')
    pending=None;offset=0;checked=valid_checked=0
    for ob in old.iter_batches(batch_size=32768,columns=new.columns):
        at=0
        while at<len(ob):
            if pending is None or offset==len(pending):pending=new.read_batch();offset=0
            n=min(len(ob)-at,len(pending)-offset)
            a=ob.slice(at,n);b=pending.slice(offset,n)
            for col in ('source_ordinal','available_ns','valid'):
                if not np.array_equal(a.column(col).to_numpy(zero_copy_only=False),b[col].to_numpy()):raise ValueError('Canonical mismatch '+col)
            mask=b['valid'].to_numpy()
            for col in ('bid_nano','ask_nano','bid_size','ask_size'):
                if not np.array_equal(a.column(col).to_numpy()[mask],b[col].to_numpy()[mask]):raise ValueError('Canonical BBO mismatch '+col)
            checked+=n;valid_checked+=int(mask.sum());at+=n;offset+=n
    return dict(status='PASS',publications=checked,valid_bbo_checked=valid_checked,
                limitation='Canonical comparison shares production L3 logic; independent prefix probe is separate')


def independent_prefix(root, config_path, limit=20000):
    from .data_semantics_audit import ReferenceBook
    from .source import DbnSource
    m=read_json(Path(root)/'manifest.json');config=read_json(config_path)
    member=next(x for x in config['members'] if x['key']==m['source_identity']['key'])
    q=FinalizedReplay(root,columns=['source_ordinal','bid_nano','ask_nano','bid_size','ask_size','valid']).read_batch(limit).to_pandas()
    book=ReferenceBook({int(row['source_ordinal']):row for row in q.to_dict('records')})
    route=m['selection']['route'];offset=0;last=None
    for block in DbnSource(member['archive'],member['member']):
        idx=np.flatnonzero((block['publisher_id']==route[0])&(block['instrument_id']==route[1])&(block['channel_id']==route[2]))
        for i in idx:
            book.feed(block[i],int(i+offset));last=block[i]
            if book.rows>=limit and int(block[i]['flags'])&128:break
        if book.rows>=limit and last is not None and int(last['flags'])&128:break
        offset+=len(block)
    # Rebuild the prevalidated L3 tape at exactly this publication boundary.
    # At equal timestamps restore_l3 intentionally includes ALL completed groups;
    # compare at a checkpoint instead for unambiguous order-state identity below.
    if book.errors:raise ValueError('Independent prefix mismatch '+repr(book.errors[:3]))
    return dict(status='PASS',raw_records=book.rows,valid_publications_checked=book.checks,
        unknown_modifies=book.unknown_modifies,unknown_cancels=book.unknown_cancels,
        limitation='Bounded prefix from explicit reset, not full independent reconstruction')


def checkpoint_parity(root):
    import zstandard
    m=read_json(Path(root)/'manifest.json');checks=0
    for c in m['checkpoints'][::max(1,len(m['checkpoints'])//4)]:
        # Verify checkpoint restoration; delta-tail reconstruction has its own
        # synthetic test. This check does not pretend to be independent L3 QA.
        payload=zstandard.ZstdDecompressor().decompress((Path(root)/c['file']).read_bytes())
        state=json.loads(payload);expected={int(oid):(side,price,size) for oid,side,price,size in state['orders']}
        actual=restore_l3(root,c['available_ns'])
        # Equal timestamp groups may legitimately follow a checkpoint. Count only
        # timestamps with no later same-time publication.
        q=FinalizedReplay(root,columns=['available_ns','source_ordinal']).read_window(c['available_ns'],c['available_ns']+1)
        if len(q) and int(q['source_ordinal'][-1].as_py())!=c['source_ordinal']:continue
        if actual!=expected:raise ValueError('Checkpoint order-state mismatch')
        checks+=1
    return dict(status='PASS',checkpoints_checked=checks,scope='checkpoint restore identity; arbitrary delta application covered by synthetic tests')


def exact_bar_checks(root, count=8):
    bars=pq.read_table(Path(root)/'bars.parquet').to_pandas()
    eligible=bars[bars.valid]
    checks=0
    columns=['available_ns','bid_nano','ask_nano','valid','trade_buy','trade_sell','trade_unknown']
    for row in eligible.sample(min(count,len(eligible)),random_state=6101).itertuples():
        start=int(row.t_ns)-60*10**9;end=int(row.t_ns)
        r=FinalizedReplay(root,columns=columns);r.seek(start)
        before=None
        g,o=r.group,r.offset
        if g or o:
            group=g if o else g-1
            block=r.file.read_row_group(group,columns=columns)
            before=block.slice(o-1 if o else len(block)-1,1).to_pydict()
        q=r.read_window(start,end).to_pandas()
        v=((q.bid_nano+q.ask_nano)/2e9).to_numpy()[q.valid.to_numpy()]
        if before and before['valid'][0]:v=np.r_[(before['bid_nano'][0]+before['ask_nano'][0])/2e9,v]
        if not len(v):raise ValueError('Valid bar without known price')
        expected=[v[0],v.max(),v.min(),v[-1]]
        if not np.allclose(expected,[row.o,row.h,row.l,row.c],rtol=0,atol=1e-9):raise ValueError('Exact midpoint OHLC mismatch')
        volume=int(q[['trade_buy','trade_sell','trade_unknown']].to_numpy().sum())
        if volume!=row.volume:raise ValueError('Exact T volume mismatch')
        checks+=1
    return dict(status='PASS' if checks else 'NO_VALID_BARS',bars_checked=checks,seed=6101,
                basis='All completed publications in [start,end), inherited supported opening book; T volume only')


def plot_day(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    s=FinalizedReplay(root,'seconds').read_window(0,2**63-1).to_pandas()
    rates=pq.read_table(Path(root)/'rates.parquet').to_pandas()
    m=read_json(Path(root)/'manifest.json');bounds=m['session']
    fig,axes=plt.subplots(7,1,figsize=(15,15),sharex=True,layout='constrained')
    t=pd.to_datetime(s.t_ns,utc=True);valid=s.valid.to_numpy()
    axes[0].plot(t,np.where(valid,s.mid,np.nan),lw=.6,label='GC midpoint USD/oz');axes[0].legend()
    axes[1].plot(pd.to_datetime(rates.t_ns,utc=True),rates.publications,lw=.4,label='Publications / wall second');axes[1].legend()
    axes[2].plot(t,np.where(valid,s.ask-s.bid,np.nan),lw=.5,label='Spread USD/oz');axes[2].legend()
    axes[3].plot(t,s.depth_bid_12,lw=.5,label='Bid depth: 12 price steps');axes[3].plot(t,s.depth_ask_12,lw=.5,label='Ask depth: 12 price steps');axes[3].legend()
    continuous=(s.epoch.diff()==0)&(s.segment_id.diff()==0)&(s.segment_id>=0)
    volume=(s.c_trade_buy+s.c_trade_sell+s.c_trade_unknown).diff().where(continuous)
    axes[4].plot(t,volume,lw=.5,label='All T volume / second');axes[4].plot(t,s.c_trade_unknown.diff().where(continuous),lw=.5,label='Unknown-side T volume');axes[4].legend()
    axes[5].plot(t,s.age_ns/1e9,lw=.5,label='Age of last observed publication (seconds)');axes[5].legend()
    axes[6].plot(t,s.valid.astype(int),label='Book supported',lw=.7);axes[6].plot(t,s.reference_present.astype(int)+1.2,label='OHLC reference (+1.2)',lw=.7);axes[6].legend()
    for ax in axes:
        ax.axvspan(pd.Timestamp(bounds['london_ns'],tz='UTC'),pd.Timestamp(bounds['ny_close_ns'],tz='UTC'),color='gold',alpha=.12)
        ax.grid(alpha=.2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=dt_timezone()))
    axes[-1].set_xlabel('UTC — yellow: London–NY clock window; separate calendar/routing permissions apply')
    fig.suptitle(m['selection']['date']+' '+m['selection']['symbol']+' — finalized source diagnostics'+(' (CONTEXT ONLY)' if m['selection']['context_only'] else ''))
    fig.savefig(output,dpi=125);plt.close(fig)


def dt_timezone():
    import datetime
    return datetime.timezone.utc


def run(root, output, config_path='data/derived/full_mbo_v1/run_config.json'):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config=read_json(config_path);rows=[]
    for folder in sorted(root.glob('member_*')):
        if not (folder/'manifest.json').exists():continue
        m=read_json(folder/'manifest.json');stage=read_json(Path(config['root'])/'staging'/folder.name/'manifest.json')
        if len(stage['entries'])!=1:raise ValueError('Audit requires matching per-capture canonical mapping')
        legacy=Path(config['root'])/'market'/Path(stage['entries'][0]['name']).stem
        if not legacy.exists():
            master=read_json(Path(config['root'])/'manifest.json')
            legacy=Path(next(c['market'] for c in master['captures'] if c['entry']['member_key']==folder.name))
        result=dict(member=folder.name,integrity=verify(folder),canonical=compare_canonical(folder,legacy),
                    independent=independent_prefix(folder,config_path),l3=checkpoint_parity(folder),bars=exact_bar_checks(folder),benchmark=benchmark(folder,legacy))
        result['performance_pass']=result['benchmark']['speedup']>=3 and result['benchmark']['scaling_ratio']<=2.5
        save_json(out/(folder.name+'.json'),result);plot_day(folder,out/(folder.name+'.png'));rows.append(result)
    summary=dict(status='PASS' if rows and all(r['performance_pass'] for r in rows) else 'NOT_ACCEPTED',members=rows)
    save_json(out/'summary.json',summary)
    sections=[]
    for r in rows:
        name=r['member'];b=r['benchmark']
        sections.append(f'<h2>{name}</h2><p>Full publication parity: {r["canonical"]["publications"]:,}. Independent prefix: {r["independent"]["raw_records"]:,} raw records. Reader speedup: {b["speedup"]:.1f}×. Double-row scaling: {b["scaling_ratio"]:.2f}×.</p><img src="{name}.png" style="width:100%"><p><a href="{name}.json">Evidence JSON</a></p>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Replay acceptance</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}p{line-height:1.5}</style><h1>Finalized replay — acceptance</h1><p>Full payload verification and diagnostics run here, never in the training hot path. First observed read is not a forced cold-cache benchmark.</p>'+''.join(sections),encoding='utf8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--market',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.market,a.output),indent=2))
