"""Cross-package guarantees for the extracted library, independent of old jobs."""
import ast
import importlib
import json
from pathlib import Path
import pkgutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import spectrum
from spectrum.core import StreamConfig
from spectrum.features import FeatureBatch, FeatureStream, compute_batch, list_families
from spectrum.market import QuoteColumns
from spectrum.crossings import build_bank, publish_bank, open_bank
from spectrum.representation import interaction_stream, publish_observation_store, open_observation_store
from test_foundation import frame
from test_observations import raw_sequence, collect, synthetic_fast
from spectrum.representation.representation import make_fast, DAY_VALUES, NS, MINUTE, anchor


def test_observation_summary_ignores_future_and_orientation_preserves_bounds():
    from spectrum.features.families.observation_summary import summary_at, FAMILIES
    from spectrum.features.families.scalper_context import side_features, SYMMETRIC, DIRECTIONAL
    seconds, levels, trades = synthetic_fast(1000)
    fast = make_fast(seconds, levels, trades)
    t = fast.t_ns.to_numpy()[::60]
    context = pd.DataFrame({n:np.ones(len(t)) for n in DAY_VALUES})
    context['available_ns'], context['known'] = t, True
    origins = fast.available_ns.to_numpy()[[500,600]]
    expected = summary_at(fast, context, origins, date='2026-08-25')
    assert set(FAMILIES['all']) <= set(expected)
    changed = fast.copy()
    changed.loc[changed.available_ns > origins[-1], 'mid'] = 1e9
    changed['future_outcome'] = 1e12
    pd.testing.assert_frame_equal(expected,summary_at(changed,context,origins,date='2026-08-25'))
    f = pd.DataFrame({n:[1.] for n in SYMMETRIC + DIRECTIONAL})
    for window in (5,30,60,300):
        f[f'flow_pressure{window}_low'], f[f'flow_pressure{window}_high'] = -.2,.7
    f['range60_pos'], f['range300_pos'] = .25,.75
    long, short = side_features(f,1), side_features(f,-1)
    assert long.pressure5_low.iloc[0] == pytest.approx(-.2)
    assert short.pressure5_low.iloc[0] == pytest.approx(-.7)
    assert short.pressure5_high.iloc[0] == pytest.approx(.2)
    assert short.range60.iloc[0] == .75


def test_sequence_store_needs_no_labels_and_keeps_missing_clock_positions():
    from spectrum.representation import SequenceStore, fit_scaler
    count = 20
    clock = np.arange(count, dtype=np.int64)*5*NS
    frame = pd.DataFrame(dict(date='2026-01-01', t_ns=clock, feature_available=True,
                             legacy_segment_token=0, physical_segment_token=0,
                             physical_support_available=True))
    frame.loc[8, 'feature_available'] = False
    raw = np.arange(count, dtype=np.float32)[:, None]
    store = SequenceStore(frame, raw, ['price'], {'price':['price']})
    scaler = fit_scaler(store, np.arange(7))
    result = store.batch(np.array([10]), family='price', length=4, cadence_seconds=5, scaler=scaler)
    assert result.shape == (1, 4, 3)
    assert result[0, 1].tolist() == [0., 0., 1.]
    changed = raw.copy()
    changed[11:] = 1e9
    other = SequenceStore(frame, changed, ['price'], {'price':['price']})
    np.testing.assert_array_equal(result,other.batch(np.array([10]),family='price',length=4,cadence_seconds=5,scaler=scaler))
    frame.loc[9:, 'physical_segment_token'] = 1
    broken = SequenceStore(frame, raw, ['price'], {'price':['price']})
    batch = broken.batch(np.array([10]),family='price',length=4,cadence_seconds=5,scaler=scaler)
    assert not batch[0,:2].any()


def test_catalog_entrypoints_resolve_and_library_imports_without_experiments():
    catalog = list_families()
    assert len({f.name for f in catalog}) == len(catalog)
    assert {'price', 'flow', 'liquidity', 'spectrum', 'events', 'context', 'representation'} <= {f.group for f in catalog}
    for family in catalog:
        assert callable(family.resolve()), family.name
    for item in pkgutil.walk_packages(spectrum.__path__, 'spectrum.'):
        if not item.name.endswith('__main__'):
            importlib.import_module(item.name)


def test_library_has_no_training_or_archive_import_and_core_stays_independent():
    root = Path(spectrum.__file__).parent
    forbidden = ('scripts', 'torch', 'sklearn', 'spectrum.cognition', 'spectrum.player',
                 'spectrum.playground', 'spectrum.research', 'archive')
    for path in root.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or '']
            assert not any(name == bad or name.startswith(bad + '.') for name in imports for bad in forbidden), path
            if path.is_relative_to(root/'core'):
                assert not any(name.startswith('spectrum.') and not name.startswith('spectrum.core') for name in imports), path
            if path.is_relative_to(root/'features') or path.is_relative_to(root/'representation'):
                assert not any(name.startswith('spectrum.outcomes') for name in imports), path


def test_batch_stream_and_restart_match_for_all_frame_families():
    data = frame(3850)
    data['trade_volume'] = 2.
    data['trade_price_mean'] = data.mid
    data['trade_price_m2'] = .5
    data['trade_moments_valid'] = True
    data['trade_moments_available_ns'] = data.available_ns
    # The exact family uses its declared trade-moment columns; a midpoint
    # proxy is independently selected and never substituted for exact data.
    selected = ('motion', 'flow', 'spectral', 'depth_balance', 'volatility_vwap_proxy', 'volatility_vwap_exact')
    expected = compute_batch(data, selected)
    assert expected.known[-1, expected.names.index('vwap_exact_3600s_sigma_ticks')]
    stream = FeatureStream(selected, source_id='synthetic')
    chunks = [stream.feed_batch(data.iloc[:1300])]
    stream = FeatureStream.restore(json.loads(json.dumps(stream.checkpoint())), expected_source_id='synthetic')
    chunks.extend(stream.feed_batch(data.iloc[a:b]) for a,b in ((1300,2700),(2700,len(data))))
    assert all(c.names == expected.names and c.units == expected.units for c in chunks)
    np.testing.assert_allclose(np.concatenate([c.values for c in chunks]), expected.values, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(np.concatenate([c.known for c in chunks]), expected.known)
    np.testing.assert_array_equal(np.concatenate([c.available_ns for c in chunks]), expected.available_ns)


def test_feature_batch_rejects_backdated_or_unmasked_unknown():
    with pytest.raises(ValueError, match='publication'):
        FeatureBatch(np.array([2],np.int64),np.array([1],np.int64),np.ones((1,1)),np.ones((1,1),bool),('x',),('tick',))
    with pytest.raises(ValueError, match='Unknown'):
        FeatureBatch(np.array([1],np.int64),np.array([1],np.int64),np.ones((1,1)),np.zeros((1,1),bool),('x',),('tick',))


def test_interaction_family_shares_core_book_and_survives_json_checkpoint():
    rows = raw_sequence()
    expected_levels, expected_trades, expected_stats = collect(rows)
    levels, trades = [], []
    config = StreamConfig(rows[0].publisher_id, rows[0].instrument_id, rows[0].channel_id)
    stream = interaction_stream(config, emit_levels=levels.extend, emit_trades=trades.append)
    cut = 5
    stream.feed_many(rows[:cut])
    checkpoint = json.loads(json.dumps(stream.checkpoint()))
    stream = interaction_stream(config, emit_levels=levels.extend, emit_trades=trades.append, checkpoint=checkpoint)
    stream.feed_many(rows[cut:])
    stream.finish()
    # The same core engine drives both the BBO stream and the interaction ledger.
    pd.testing.assert_frame_equal(pd.DataFrame(levels), pd.DataFrame(expected_levels))
    pd.testing.assert_frame_equal(pd.DataFrame(trades), pd.DataFrame(expected_trades))
    assert stream.metrics[0].collector.stats['max_stock_residual'] == 0
    assert stream.engine.completed_events == stream.metrics[0].collector.stats['events']


def test_interaction_delivery_delay_never_backdates_publication():
    rows = raw_sequence()
    levels, trades = [], []
    config = StreamConfig(rows[0].publisher_id, rows[0].instrument_id, rows[0].channel_id)
    stream = interaction_stream(config, emit_levels=levels.extend, emit_trades=trades.append)
    for row in rows:
        stream.feed(row, received_at_ns=row.ts_recv_ns + 5*NS)
    assert trades and levels
    assert all(row['available_ns'] >= row['t_ns'] + 5*NS for row in trades)
    assert all(row['available_ns'] >= row['t_ns'] + 5*NS for row in levels)


def test_crossing_bank_roundtrip_and_tamper_are_independent_of_reports(tmp_path):
    n = 20000
    t = np.arange(n, dtype=np.int64)*NS
    mid = (40000 + np.rint(10*np.sin(np.arange(n)/80))).astype(np.int64)*100_000_000
    quotes = QuoteColumns(t, mid-100_000_000, mid+100_000_000, np.ones(n,bool), np.ones(n,np.int64))
    expected, _ = build_bank(quotes, 'test-source')
    assert len(expected)
    root = tmp_path/'bank'
    publish_bank(quotes,root,source_id='test-source')
    bank, manifest = open_bank(root)
    pd.testing.assert_frame_equal(bank,expected)
    assert manifest['source_id'] == 'test-source'
    with pytest.raises(FileExistsError):
        publish_bank(quotes,root,source_id='test-source')
    with (root/'events.parquet').open('ab') as f:
        f.write(b'tamper')
    with pytest.raises(ValueError,match='payload'):
        open_bank(root)


def test_observation_store_roundtrip_masks_and_tamper(tmp_path):
    seconds, levels, trades = synthetic_fast(1000)
    fast = make_fast(seconds,levels,trades)
    stamps = anchor('2026-08-24') + np.arange(1,1801)*MINUTE
    context = pd.DataFrame({name:np.ones(len(stamps)) for name in DAY_VALUES})
    context['t_ns'],context['available_ns'] = stamps,stamps+NS
    context['known'],context['scheduled_open'] = True,True
    context.loc[10,'known'] = False
    zones = pd.DataFrame(dict(zone_id=['a'],available_ns=[stamps[0]],price=[1.]))
    events = pd.DataFrame(dict(available_ns=stamps[-300:],kind='CONTACT'))
    root = tmp_path/'observations'
    publish_observation_store(root,date='2026-08-25',previous_date='2026-08-24',symbol='GCZ6',
                              fast=fast,context=context,zones=zones,events=events,source_identity={'synthetic':True})
    store = open_observation_store(root,'2026-08-25')
    now = int(pd.Timestamp('2026-08-25 08:10',tz='UTC').value)
    observation = store.observe(now)
    assert observation.day['available_ns'].max() <= now
    assert not observation.day['known'][10].any()
    assert not observation.receipt['future_outcomes_present']
    path=root/'observations/2026-08-25_fast.parquet'
    with path.open('ab') as f:
        f.write(b'tamper')
    with pytest.raises(ValueError,match='payload'):
        open_observation_store(root,'2026-08-25')


def test_cli_catalog_runs_from_an_unrelated_directory(tmp_path):
    import os
    env = dict(os.environ, PYTHONPATH=str(Path(spectrum.__file__).parent.parent))
    result = subprocess.run([sys.executable,'-m','spectrum','families','--group','liquidity'],
                            cwd=tmp_path,env=env,text=True,capture_output=True,check=True)
    rows = json.loads(result.stdout)
    assert rows and all(r['group']=='liquidity' for r in rows)
