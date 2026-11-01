"""Frozen metric comparison on exposed sessions, with no future outcome labels."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from spectrum.io import sha
from spectrum.representation.session_store import open_session_store
from experiments.scalper_adapter_v1.adapter import SessionStories,live_states
from experiments.scalper_adapter_v1.tokens import NS
from experiments.scalper_adapter_v1.measures import FlowView
from experiments.scalper_view_audit_v1.audit import dump,clean
from .features import BOOK_NAMES
from .search import SimilaritySearch,sensitivity_methods,FEATURE_KEYS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_corpus(config):
    parent=ROOT/config['adapter_run'];m=read(parent/'manifest.json');v=read(parent/'validation.json')
    if m['status']!='COMPLETE_DESCRIPTIVE_ADAPTER' or v['status']!='PASS_BOUNDED_ADAPTER_CHECKS':
        raise ValueError('Verified adapter required')
    if v['manifest_sha256']!=sha(parent/'manifest.json'):raise ValueError('Parent validation mismatch')
    inputs={str(parent/'manifest.json'):sha(parent/'manifest.json'),str(parent/'validation.json'):sha(parent/'validation.json')}
    # Current dependencies must still match the generating code, including clock fixes.
    for path,digest in m['source_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Parent source drift: '+path)
        inputs[str(ROOT/path)]=digest
    artifacts={k.replace('\\','/'):v for k,v in m['artifacts'].items()}
    sessions=[]
    for date in config['sessions']:
        store_path=ROOT/config['source_root']/date
        sm=read(store_path/'manifest.json')
        if sha(store_path/'manifest.json')!=m['input_sha256'][str(store_path/'manifest.json')]:
            raise ValueError('Session identity drift')
        store=open_session_store(store_path)
        inputs[str(store_path/'manifest.json')]=sha(store_path/'manifest.json')
        for name,digest in sm['files'].items(): inputs[str(store_path/name)]=digest
        for source in sm['quote_sources']:inputs[source['path']]=source['sha256']
        files=[]
        for name in ('motion.parquet','zone_updates.parquet'):
            relative=date+'/'+name;p=parent/relative
            if sha(p)!=artifacts[relative]:raise ValueError('Adapter token drift')
            inputs[str(p)]=sha(p);files.append(pd.read_parquet(p))
        q=store.quotes[store.quotes.available_ns.between(store.start,store.end)].reset_index(drop=True)
        sessions.append(SessionStories(date,sm['previous_date'],sm['symbol'],store.start,store.end,store.end,
                                       q,store.fast,store.context,files[0],files[1]))
    return sessions,inputs


def query_clocks(config):
    date=config['query_session']
    anchors=[int(pd.Timestamp(date+'T'+t+'Z').value) for t in config['anchor_times_utc']]
    pool=[int(pd.Timestamp(date+f'T{h:02}:{m:02}:00Z').value) for h in range(7,20) for m in (0,15,30,45)]
    extra=sorted([t for t in pool if t not in anchors],key=lambda t:hashlib.sha256(str(t).encode()).hexdigest())[:config['extra_queries']]
    return sorted(anchors+extra),set(anchors)


def save_scene(folder,search,prepared,matches,case_id):
    scene=dict(case_id=case_id,query_ns=int(prepared['token'].available_ns),scale=prepared['token'].scale,methods={})
    contexts=[('query',prepared['query_frame'])]
    for name,items in matches.items():
        if items:
            match=items[0];ids=[search.base.lookup[x] for x in match['context_token_ids']]
            contexts.append((name,search.base.frame.iloc[ids]))
            scene['methods'][name]=match
        else:scene['methods'][name]=None
    for name,frame in contexts:
        date=frame.session_id.iloc[0];s=search.sessions[date]
        frame.to_parquet(folder/(case_id+'_'+name+'_tokens.parquet'),index=False)
        q=s.quotes.iloc[int(frame.start_index.min()):int(frame.source_index.max())+1]
        q[['available_ns','bid_nano','ask_nano','source_ordinal']].to_parquet(folder/(case_id+'_'+name+'_quotes.parquet'),index=False)
        from .features import extract
        pack=extract(frame,s.quotes,search.books[date],samples=search.config['shape_samples'])
        np.savez_compressed(folder/(case_id+'_'+name+'_features.npz'),**pack)
    return scene


def summarize(statuses,matches):
    rows=[]
    for (scope,method),g in statuses.groupby(['scope','method'],sort=False):
        top=matches[(matches.scope==scope)&(matches.method==method)&(matches['rank']==1)]
        rows.append(dict(scope=scope,method=method,queries=len(g),with_neighbor=int(g.returned.gt(0).sum()),
            no_analogue=int(g.returned.eq(0).sum()),
            median_total_duration_factor=float(top.total_duration_factor.median()) if len(top) else None,
            max_total_duration_factor=float(top.total_duration_factor.max()) if len(top) else None,
            median_shape_rmse=float(top.shape_rmse.median()) if len(top) else None,
            median_mechanism_rmse=float(top.mechanism_rmse.median()) if len(top) else None,
            median_book_common_fraction=float(top.book_common_fraction.median()) if len(top) else None))
    return rows


def main(out=None):
    config=read(HERE/'config.json');folder=ROOT/(out or config['output'])
    folder.mkdir(parents=True,exist_ok=False)
    snapshot=folder/'source_snapshot';snapshot.mkdir()
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+[ROOT/'tests/test_story_similarity_v2.py']
    source={str(p.relative_to(ROOT)):sha(p) for p in files}
    for p in files:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    dump(folder/'config.json',config);dump(folder/'manifest.json',dict(status='RUNNING',source_sha256=source))
    begun=time.perf_counter()
    try:
        sessions,inputs=load_corpus(config);loaded=time.perf_counter()
        search=SimilaritySearch(sessions,config);indexed=time.perf_counter()
        current=search.sessions[config['query_session']]
        clocks,anchors=query_clocks(config)
        # The first extra query is also rendered, irrespective of its outcomes.
        display_clocks=anchors|{next(t for t in clocks if t not in anchors)}
        statuses=[];matches=[];sweep_status=[];sweep_matches=[];scenes=[];query_rows=[]
        checks=dict(legacy_anchor_matches=0,book_scalar_checks=0,native_shape_samples=0,prefix_equal=False)
        regression=read(ROOT/config['adapter_run']/'atlas_index.json')
        for clock in clocks:
            start=time.perf_counter();states=live_states(current,clock)
            for _,token in states[states.family.eq('motion')].iterrows():
                prepared=search.prepare(token,'previous')
                query_rows.append(prepared['query_frame'].assign(query_id=token.token_id))
                by_method={}
                for scope in ('previous','current'):
                    task=prepared if scope=='previous' else search.prepare(token,'current')
                    for method in config['methods']:
                        status,neighbors=search.rank(task,method)
                        statuses.append(status);matches.extend(neighbors)
                        if scope=='previous':by_method[method['name']]=neighbors
                for method in sensitivity_methods(config):
                    status,neighbors=search.rank(prepared,method,neighbors=1)
                    sweep_status.append(status);sweep_matches.extend(neighbors)
                if clock in anchors:
                    old=next(r for r in regression if r['query_token_id']==token.token_id)
                    now=by_method['legacy'][0]
                    assert now['token_id']==old['token_id'] and abs(now['score']-old['distance'])<1e-10
                    checks['legacy_anchor_matches']+=1
                if clock in display_clocks:
                    case_id=pd.Timestamp(clock,tz='UTC').strftime('%H%M%S')+'_'+str(token.scale)
                    scene=save_scene(folder,search,prepared,{n:by_method[n] for n in ('legacy','rhythm_gate','book_gate')},case_id)
                    scene['statuses']={s['method']:s for s in statuses[-10:] if s['scope']=='previous'}
                    scenes.append(scene)
                # Original scalar context routine at clock-1 observes exactly the
                # same strict-before boundary as the new vectorized implementation.
                frame=prepared['query_frame'];b=search.books[current.session_id].measure(frame)
                for k,row in enumerate(frame.itertuples()):
                    probe=pd.Series(dict(available_ns=int(row.available_ns)-1,flow_seconds=int(row.flow_seconds)))
                    ref=FlowView(current.fast).measure(probe)
                    for col,name in ((0,'flow_imbalance_low'),(1,'flow_imbalance_high'),(2,'unknown_volume_fraction')):
                        if b['book_known'][k,col]:assert np.isclose(b['book'][k,col],ref[name],atol=1e-7)
                    if b['book_known'][k,4]:assert np.isclose(b['book'][k,4],np.log1p(ref['bid_near_fill']/row.flow_seconds)/np.log(11),atol=1e-7)
                    assert b['book_release_ns'][k]<row.available_ns
                    checks['book_scalar_checks']+=1
                # Last-held-price samples independently looked up in native rows.
                price=(current.quotes.bid_nano.to_numpy(np.int64)+current.quotes.ask_nano.to_numpy(np.int64))//2
                qt=current.quotes.available_ns.to_numpy(np.int64)
                from .features import extract
                pack=extract(frame,current.quotes,samples=config['shape_samples'])
                for k,row in enumerate(frame.itertuples()):
                    for n in range(1,17):
                        at=int(row.start_ns)+(int(row.available_ns)-int(row.start_ns))*n//16
                        j=min(int(row.source_index),max(int(row.start_index),int(np.searchsorted(qt,at,side='right'))-1))
                        expected=(int(price[j])-int(row.start_mid_nano))/(int(row.amplitude_ticks)*100_000_000)
                        assert np.isclose(pack['raw_shape'][k,n-1],expected,atol=1e-6)
                        checks['native_shape_samples']+=1
            print(json.dumps(dict(stage='query',time=pd.Timestamp(clock,tz='UTC').isoformat(),seconds=time.perf_counter()-start)),flush=True)
        statuses=pd.DataFrame(statuses);matches=pd.DataFrame(matches)
        sweep_status=pd.DataFrame(sweep_status);sweep_matches=pd.DataFrame(sweep_matches)
        # Strictly truncate input frames and native quotes; retrieve the same noon queries.
        noon=int(pd.Timestamp(config['query_session']+'T12:00:00Z').value)
        prefix=copy.copy(current)
        prefix.quotes=current.quotes[current.quotes.available_ns<=noon]
        prefix.motion=current.motion[current.motion.available_ns<=noon]
        prefix.fast=current.fast[current.fast.available_ns<noon]
        short=SimilaritySearch([sessions[0],prefix],config)
        for _,row in live_states(current,noon).query("family == 'motion'").iterrows():
            for scope in ('previous','current'):
                a=search.prepare(row,scope);b=short.prepare(row,scope)
                for name in a['parts']:assert np.allclose(a['parts'][name],b['parts'][name],equal_nan=True)
                for method in config['methods']:assert search.rank(a,method)==short.rank(b,method)
        checks['prefix_equal']=True
        statuses.to_parquet(folder/'statuses.parquet',index=False)
        matches.drop(columns=['weights','context_token_ids','query_context_token_ids']).to_parquet(folder/'matches.parquet',index=False)
        dump(folder/'matches.json',matches.to_dict('records'))
        sweep_status.to_parquet(folder/'sweep_statuses.parquet',index=False)
        sweep_matches.drop(columns=['weights','context_token_ids','query_context_token_ids']).to_parquet(folder/'sweep_matches.parquet',index=False)
        pd.concat(query_rows,ignore_index=True).to_parquet(folder/'query_contexts.parquet',index=False)
        dump(folder/'atlas_index.json',scenes)
        summary=dict(status='COMPLETE_DESCRIPTIVE_SIMILARITY',queries=len(clocks)*3,clocks=clocks,
            query_selection=config['query_selection'],methods=summarize(statuses,matches),
            parameter_sensitivity=summarize(sweep_status,sweep_matches),checks=checks,
            runtime=dict(load_seconds=loaded-begun,index_seconds=indexed-loaded,total_seconds=time.perf_counter()-begun),
            book_names=BOOK_NAMES,parameter_selection='All predefined variants retained; no winner selected by these reused observations',
            limitations=['Two exposed sessions and 36 query states; no untouched confirmation',
                'Tolerances are user-visible measuring definitions, not calibrated semantic truth',
                'Metrics used in the ranking cannot serve as independent proof of improved meaning',
                'Book windows are endpoint context, not the exact flow inside each subsecond story',
                'Shape/time/context similarity is not equality of participant intention or future money'])
        dump(folder/'summary.json',summary)
        artifacts={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'}
        dump(folder/'manifest.json',dict(status=summary['status'],version=config['version'],source_sha256=source,
            input_sha256=inputs,artifacts=artifacts,elapsed_seconds=summary['runtime']['total_seconds']))
        print(json.dumps(clean(summary)),flush=True)
    except Exception as exc:
        dump(folder/'manifest.json',dict(status='FAILED_SIMILARITY_DIAGNOSTIC',error=repr(exc),
            traceback=traceback.format_exc(),source_sha256=source));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out');main(p.parse_args().out)
