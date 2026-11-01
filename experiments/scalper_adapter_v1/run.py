"""Bounded two-session generation, causal-prefix checks and reviewable artifacts."""
import argparse
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from spectrum.io import sha
from experiments.scalper_view_audit_v1.audit import dump
from .adapter import ROOT, HERE, load_session, build_session, ScalperAdapter
from .measures import PATH_NAMES, MODEL_NAMES
from .tokens import NS, TICK


def scalar_path_check(row, quotes):
    """Separately expressed finite-interval arithmetic on original native BBO."""
    q = quotes.iloc[int(row.start_index):int(row.source_index)+1]
    prices = [(int(b)+int(a))//2 for b,a in zip(q.bid_nano,q.ask_nano)]
    times = [int(t) for t in q.available_ns]
    net = (prices[-1]-prices[0])/TICK
    path = sum(abs(b-a) for a,b in zip(prices,prices[1:]))/TICK
    span = (max(prices)-min(prices))/TICK
    dwell = 0
    for j in range(len(times)):
        if not bool(q.valid.iloc[j]): continue
        left = max(int(row.start_ns), times[j])
        right = min(int(row.available_ns), times[j]+30*NS,
                    times[j+1] if j+1<len(times) else int(row.available_ns))
        if int(row.zone_low_nano) <= prices[j] < int(row.zone_high_nano):
            dwell += max(0, right-left)
    assert abs(row.net_ticks-net)<1e-9
    assert abs(row.path_ticks-path)<1e-9
    assert abs(row.range_ticks-span)<1e-9
    assert abs(row.zone_dwell_seconds-dwell/NS)<1e-9


def make_plot_cases(adapter, observations, out):
    records = []
    for query_id, observation in observations.items():
        table = observation['tokens'].set_index('token_id', drop=False)
        for scale in ('fast','middle','slow'):
            matches = [m for m in observation['similarity'] if m['scale']==scale
                       and m['session_id'] != observation['receipt']['session_id']]
            if not matches: continue
            match = matches[0]
            case_id = query_id+'_'+scale
            for role, ids in [('query',match['query_context_token_ids']),('past',match['context_token_ids'])]:
                rows = table.loc[ids].copy()
                session = adapter.sessions[rows.session_id.iloc[0]]
                first = int(rows.start_index.min()); last = int(rows.source_index.max())
                q = session.quotes.iloc[first:last+1][['available_ns','bid_nano','ask_nano','source_ordinal']].copy()
                q.to_parquet(out/(case_id+'_'+role+'_quotes.parquet'), index=False)
                rows.to_parquet(out/(case_id+'_'+role+'_tokens.parquet'), index=False)
            records.append(dict(case_id=case_id, **match))
    dump(out/'atlas_index.json', records)


def main(out=None):
    cfg = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    destination = ROOT/(out or cfg['output'])
    destination.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    source_files = sorted(HERE.glob('*.py'))+sorted(HERE.glob('*.json'))+[ROOT/'tests/test_scalper_adapter.py']
    source_files += [ROOT/p for p in ('experiments/zone_episode_lab/episodes.py',
        'experiments/scalper_view_audit_v1/audit.py', 'experiments/price_level_memory_v1/memory.py',
        'spectrum/representation/session.py', 'spectrum/representation/session_store.py',
        'spectrum/representation/representation.py', 'spectrum/market/calendar.py', 'spectrum/units.py', 'spectrum/io.py')]
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in source_files}
    snapshot = destination/'source_snapshot'; snapshot.mkdir()
    for p in source_files:
        target = snapshot/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    dump(destination/'config.json', cfg)
    dump(destination/'manifest.json', dict(status='RUNNING', source_sha256=source_hashes))
    try:
        sessions, inputs, builds = [], {}, []
        for date in cfg['sessions']:
            start = time.perf_counter()
            session, hashes = load_session(ROOT/cfg['source_root']/date, cfg['scales'])
            elapsed = time.perf_counter()-start
            inputs.update(hashes); sessions.append(session)
            dest = destination/date; dest.mkdir()
            session.motion.to_parquet(dest/'motion.parquet', index=False)
            session.zones.to_parquet(dest/'zone_updates.parquet', index=False)
            stats = dict(session_id=date, previous_session_id=session.previous_session_id,
                quote_rows=len(session.quotes), seconds_including_source_verification=elapsed,
                native_quotes_per_second=len(session.quotes)/elapsed,
                motion_updates=len(session.motion), zone_updates=len(session.zones),
                story_counts=session.motion.groupby('scale').story_id.nunique().to_dict(),
                phase_counts=session.tokens.groupby(['family','scale','phase']).size().rename('count').reset_index().to_dict('records'))
            builds.append(stats)
            print(json.dumps(dict(stage='built', **{k:v for k,v in stats.items() if k!='phase_counts'})), flush=True)
        start = time.perf_counter()
        adapter = ScalperAdapter(sessions, cfg)
        indexing = time.perf_counter()-start
        observations, receipts = {}, []
        scalar_checks = 0
        for k, timestamp in enumerate(cfg['query_times_utc']):
            clock = int(pd.Timestamp(timestamp).value); date = cfg['sessions'][-1]
            start = time.perf_counter(); observation = adapter.observe(date, clock)
            elapsed = time.perf_counter()-start
            query_id = 'query_'+str(k+1)
            observations[query_id] = observation
            observation['tokens'].to_parquet(destination/(query_id+'_tokens.parquet'), index=False)
            observation['price_memory'].to_parquet(destination/(query_id+'_price_memory.parquet'), index=False)
            observation['context'].to_parquet(destination/(query_id+'_context.parquet'), index=False)
            dump(destination/(query_id+'_similarity.json'), observation['similarity'])
            model = observation['model']
            np.savez_compressed(destination/(query_id+'_model.npz'), values=model['values'], known=model['known'],
                categorical_values=model['categorical_values'], categorical_known=model['categorical_known'])
            dump(destination/(query_id+'_receipt.json'), dict(observation['receipt'], elapsed_seconds=elapsed,
                feature_names=model['names'], categorical_names=model['categorical_names'], categorical_vocab=model['categorical_vocab']))
            # Selection depends only on token identity, not on shape or any future outcome.
            sample = observation['tokens'].copy()
            sample['_selection'] = pd.util.hash_pandas_object(sample.token_id, index=False)
            for row in sample.sort_values('_selection').head(24).itertuples():
                if row.path_known:
                    scalar_path_check(row, adapter.sessions[row.session_id].quotes); scalar_checks += 1
            for match in observation['similarity']:
                assert match['available_ns'] < match['query_context_start_ns'] <= clock
                assert set(match['context_token_ids']).issubset(set(observation['tokens'].token_id))
            receipts.append(dict(query_id=query_id, timestamp=timestamp, elapsed_seconds=elapsed,
                matches=len(observation['similarity']), **observation['receipt']))
            print(json.dumps(dict(stage='observation', **receipts[-1])), flush=True)
        # Rebuild the entire current-session prefix, not merely filter a finished table.
        clock = int(pd.Timestamp(cfg['query_times_utc'][1]).value)
        current = sessions[-1]
        start = time.perf_counter()
        prefix = build_session(current.quotes[current.quotes.available_ns<=clock],
            session_id=current.session_id, previous_session_id=current.previous_session_id,
            symbol=current.symbol, start_ns=current.start_ns, end_ns=current.end_ns,
            scales=cfg['scales'], observed_through_ns=clock,
            fast=current.fast[current.fast.available_ns<=clock],
            context=current.context[current.context.available_ns<=clock])
        p = ScalperAdapter([sessions[0],prefix], cfg).observe(current.session_id, clock)
        full = observations['query_2']
        assert_frame_equal(full['tokens'], p['tokens'])
        assert_frame_equal(full['price_memory'], p['price_memory'])
        assert_frame_equal(full['context'], p['context'])
        assert full['similarity'] == p['similarity']
        assert np.array_equal(full['model']['values'], p['model']['values'])
        assert np.array_equal(full['model']['known'], p['model']['known'])
        prefix_seconds = time.perf_counter()-start
        make_plot_cases(adapter, observations, destination)
        summary = dict(status='COMPLETE_DESCRIPTIVE_ADAPTER', builds=builds, indexing_seconds=indexing,
            observations=receipts, scalar_native_interval_checks=scalar_checks,
            full_observation_prefix_invariance=True, prefix_rebuild_seconds=prefix_seconds,
            feature_count=len(MODEL_NAMES), path_measure_count=len(PATH_NAMES),
            elapsed_seconds=time.perf_counter()-begun,
            limitations=['Two exposed venue sessions; not full-history or live-stream certification',
                'Reference as-of implementation; measured observation latency is not a live scalper SLA',
                'Preset geometry and equal-weight similarity, no learned similarity or economic evidence',
                'Native BBO path; book and T context uses published 1s buckets',
                'No learned encoder, target, policy or capital replay in this adapter run',
                'Model context is bounded and selected; full event ledgers remain available',
                'The Aug24 previous session Aug21 is not loaded; Aug25 has the complete loaded Aug24 session'])
        dump(destination/'summary.json', summary)
        files = {str(p.relative_to(destination)):sha(p) for p in destination.rglob('*')
                 if p.is_file() and p.name!='manifest.json'}
        dump(destination/'manifest.json', dict(status=summary['status'], version=cfg['version'],
            source_sha256=source_hashes, input_sha256=inputs, artifacts=files,
            python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
            elapsed_seconds=summary['elapsed_seconds']))
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        dump(destination/'manifest.json', dict(status='FAILED_DESCRIPTIVE_ADAPTER', error=repr(exc),
            traceback=traceback.format_exc(), source_sha256=source_hashes))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out')
    main(p.parse_args().out)
