"""Typed future data cannot enter X through supported array/table operations."""
import importlib
import inspect
import pkgutil
import numpy as np
import pandas as pd
import pytest
import spectrum.future as future_package
from spectrum.features import FeatureBatch, compute_batch
from spectrum.features.contract import Context
from spectrum.features.pipeline import REGISTRY
from spectrum.future import future_crossings, future_examples
from spectrum.leakage import FutureFrame, FutureLeak, FutureSeries, check_model_input, is_future_name
from spectrum.leakage import FutureArray, export_future_values, clear_fingerprints, mark_future
from spectrum.market.calendar import MarketCalendar
from spectrum.representation import to_feature_batch
from spectrum.representation._windows import BaseSequenceStore
from spectrum.units import NS
from test_foundation import frame


def crossings():
    t = np.arange(40, dtype=np.int64) * 10 * NS
    support = np.zeros(40, np.int64)
    tape = dict(t=t, bid=10. + np.arange(40) / 10, ask=10.2 + np.arange(40) / 10,
                support={s: support for s in (1, 3, 6, 9)})
    bank = pd.DataFrame([dict(scale=1, role=2, t_ns=t[i], ordinal=i, direction=1, event_id=str(i), support=0,
                              line=11. + i / 7, bid=10. + i / 10, ask=10.2 + i / 10) for i in range(3, 40, 3)])
    return future_crossings(bank, tape, t[:30] + 1, 600 * NS, 400 * NS)


def batch(name, values):
    t = np.arange(len(values), dtype=np.int64) * NS
    v = np.asarray(values, float)
    known = np.isfinite(v)
    return FeatureBatch(t, t, np.where(known, v, 0.)[:, None], known[:, None], (name,), ('usd',))


@pytest.mark.parametrize('transform', [
    lambda y: y[:3], lambda y: y[[0, 3, 7]], lambda y: y[::-1],
    lambda y: (y - y.mean()) / y.std(), lambda y: np.clip(y, 2, 5),
    lambda y: np.where(y > 3, y, 0), lambda y: y.astype(np.float32),
    lambda y: np.concatenate([y[:4], y[4:]]), lambda y: np.zeros(8) + y,
    lambda y: np.stack([y, y], axis=1)[:, 0],
])
def test_numpy_derivatives_retain_future_provenance_without_fingerprints(transform):
    from spectrum.future.endpoint import future_endpoint_move
    y, _ = future_endpoint_move(np.full(8, 5000.), 5000. + np.arange(8) * .1)
    hidden = transform(y)
    assert isinstance(hidden, FutureArray)
    clear_fingerprints()  # admission must depend on type, not values or process history
    t = np.arange(len(hidden), dtype=np.int64)
    with pytest.raises(FutureLeak):
        FeatureBatch(t, t, hidden[:, None], np.ones((len(t), 1), bool), ('hidden',), ('unit',))
    with pytest.raises(FutureLeak):
        FeatureBatch.from_columns(t, t, {'hidden': hidden}, {'hidden': np.ones(len(t), bool)}, {'hidden': 'unit'})
    with pytest.raises(FutureLeak):
        np.asarray(hidden)


def test_constant_short_future_masks_and_implicit_exports_are_rejected():
    y = mark_future(np.ones(3), 'constant future')
    clear_fingerprints()
    t = np.arange(3, dtype=np.int64)
    with pytest.raises(FutureLeak):
        FeatureBatch(t, t, np.zeros((3, 1)), (y > 0)[:, None], ('x',), ('unit',))
    with pytest.raises(FutureLeak):
        pd.DataFrame({'renamed': y})
    with pytest.raises(FutureLeak):
        np.add(y, 1, out=np.empty(3))
    with pytest.raises(FutureLeak):
        np.concatenate([y, y], 0, np.empty(6))
    with pytest.raises(FutureLeak):
        y.sum(None, None, np.empty(()))
    assert export_future_values(y).tolist() == [1., 1., 1.]


def test_future_array_provenance_survives_pickle_without_global_fingerprints():
    import pickle
    original = mark_future(np.arange(20.), 'fixture future')
    restored = pickle.loads(pickle.dumps(original))
    clear_fingerprints()
    with pytest.raises(FutureLeak):
        check_model_input(['x'], {'x': (restored[:8] - restored.mean())})


def test_future_reduction_cannot_be_broadcast_as_an_unmarked_constant():
    y = mark_future(np.arange(10.), 'fixture future')
    summary = y.mean()
    clear_fingerprints()
    assert isinstance(summary, FutureArray)
    with pytest.raises(FutureLeak):
        check_model_input(['x'], {'x': np.broadcast_to(summary, (20,))})


def test_renamed_frame_columns_and_mixed_pandas_arithmetic_remain_future():
    y = crossings().rename(columns={'future_bid': 'hidden'})
    values = y.hidden.to_numpy()
    clear_fingerprints()
    with pytest.raises(FutureLeak):
        check_model_input(['x'], {'x': values})
    y = crossings()
    mixed = y.origin_ask + y.future_bid
    assert mixed.future_tainted
    with pytest.raises(FutureLeak):
        check_model_input(['x'], {'x': mixed.to_numpy()})
    disguised = y[['future_bid']].rename(columns={'future_bid': 'origin_bid'})
    assert disguised.origin_bid.future_tainted


def test_representation_boundary_rejects_a_future_mask_even_with_past_values():
    t = np.arange(3, dtype=np.int64)
    table = pd.DataFrame(dict(t_ns=t, available_ns=t, price=100., future_known=True))
    with pytest.raises(FutureLeak):
        to_feature_batch(table, {'price': 'usd'}, known={'price': 'future_known'})


@pytest.mark.parametrize('operation', [
    lambda y: (y.origin_ask > y.future_bid).to_numpy(),
    lambda y: np.add(y.origin_ask, y.future_bid),
    lambda y: np.concatenate([y.origin_ask, y.future_bid]),
    lambda y: ((y.origin_ask > 10) & (y.future_bid > 10)).to_numpy(),
])
def test_mixed_pandas_operations_track_future_from_either_operand(operation):
    hidden = operation(crossings())
    clear_fingerprints()
    with pytest.raises(FutureLeak):
        check_model_input(['hidden'], {'hidden': hidden})


@pytest.mark.parametrize('x_rows,y_rows', [(0, 2), (0, 0), (3, 0)])
def test_empty_examples_preserve_shapes_and_unusable_masks(x_rows, y_rows):
    t = np.arange(x_rows, dtype=np.int64)
    x = FeatureBatch(t, t, np.zeros((x_rows, 2)), np.ones((x_rows, 2), bool), ('a', 'b'), ('u', 'u'))
    y = mark_future(pd.DataFrame({'t_ns': np.arange(y_rows, dtype=np.int64),
                                 'future_maturity_ns': np.arange(y_rows, dtype=np.int64) + 10}),
                    'fixture', keys=('t_ns',))
    result = future_examples(x, y, cutoff_ns=5, max_x_age_ns=1)
    assert result['x'].shape == result['known'].shape == (y_rows, 2)
    assert result['names'] == ('a', 'b') and isinstance(result['y'], FutureFrame)
    assert not result['usable'].any() and not result['known'].any()
    assert np.all(result['x'] == 0) and np.all(result['x_available_ns'] == -1)


def test_name_mark_spares_the_contract_kind():
    assert is_future_name('future_bid') and is_future_name('Future_E') and is_future_name('nofuture')
    assert not is_future_name('futures_basis_ticks') and not is_future_name('roll_futures')
    with pytest.raises(FutureLeak):
        check_model_input(['spread', 'y_future'])


def test_type_mark_survives_pandas_arithmetic_and_renaming():
    y = crossings()
    assert isinstance(y, FutureFrame) and y.future_source.endswith('future_crossings')
    upper = y[(y.scale == 1) & (y.role == 'upper')].reset_index(drop=True)
    hidden = (upper.future_bid.rename('mid_change') * 2 - 1).rename('harmless')
    assert isinstance(hidden, FutureSeries)
    table = pd.DataFrame(dict(t_ns=upper.t_ns.to_numpy(), available_ns=upper.t_ns.to_numpy()))
    with pytest.raises(FutureLeak, match='future'):       # type lost in a plain frame, values remembered
        to_feature_batch(table.assign(harmless=hidden), {'harmless': 'usd'})
    with pytest.raises(FutureLeak, match='future'):
        to_feature_batch(table.assign(harmless=hidden.to_numpy()), {'harmless': 'usd'})


def test_value_mark_finds_a_renamed_plain_copy_also_masked_and_float32():
    y = crossings()
    copy = y.future_bid.to_numpy().copy()          # NaN where not crossed
    with pytest.raises(FutureLeak, match='future_bid'):
        batch('harmless', copy)                    # stored as masked zero
    with pytest.raises(FutureLeak, match='future_cross_ns'):
        check_model_input(['x'], {'x': y.future_cross_ns.to_numpy()})
    with pytest.raises(FutureLeak):
        check_model_input(['x'], copy.astype(np.float32)[:, None])
    # past facts and ordinary features pass
    batch('origin_ask', y.origin_ask.to_numpy())
    batch('noise', np.random.default_rng(0).normal(size=len(copy)))


def test_feature_context_and_sequence_store_reject_future_values():
    y = crossings()
    context = Context(frame(len(y)), MarketCalendar())
    with pytest.raises(FutureLeak):
        context.add('harmless', y.future_bid.to_numpy(), np.ones(len(y), bool), 'usd')
    rows = pd.DataFrame(dict(date='2026-05-04', t_ns=np.arange(len(y), dtype=np.int64), feature_available=True,
                             segment_token=np.zeros(len(y), np.int64)))
    raw = y[['future_bid']].to_numpy(np.float32)
    with pytest.raises(FutureLeak):
        BaseSequenceStore(rows, raw, ['harmless'], {'all': ['harmless']})


def test_every_public_future_function_returns_marked_results():
    for info in pkgutil.iter_modules(future_package.__path__):
        module = importlib.import_module('spectrum.future.' + info.name)
        for name, value in vars(module).items():
            if inspect.isfunction(value) and value.__module__ == module.__name__ and not name.startswith('_'):
                assert getattr(value, 'future_output', False) or getattr(value, 'future_join', False), (info.name, name)


def test_examples_join_on_publication_and_drop_late_or_unanswerable_labels():
    y = crossings()
    upper = y[(y.scale == 1) & (y.role == 'upper')].reset_index(drop=True)
    t = np.arange(0, 300, 7, dtype=np.int64) * NS
    x = FeatureBatch(t, t + 5 * NS, (t / NS)[:, None].astype(float), np.ones((len(t), 1), bool), ('age',), ('s',))
    e = future_examples(x, upper)
    origin = upper.t_ns.to_numpy()
    seen = e['x_available_ns'] >= 0
    assert np.all(e['x_available_ns'][seen] <= origin[seen])
    assert not e['usable'][0]                      # nothing published yet at the first origin
    assert np.all(upper.future_maturity_ns.to_numpy()[e['usable']] > origin[e['usable']])
    cut = int(origin[10])
    late = future_examples(x, upper, cutoff_ns=cut)
    assert np.all(upper.future_maturity_ns.to_numpy()[late['usable']] <= cut)
    with pytest.raises(TypeError):
        future_examples(x, pd.DataFrame(upper))   # an unmarked copy is not accepted as Y


def test_every_registered_frame_family_is_causal_under_future_mutation():
    data = frame(3000)
    data['trade_volume'] = 2.
    data['trade_price_mean'] = data.mid
    data['trade_price_m2'] = .5
    data['trade_moments_valid'] = True
    data['trade_moments_available_ns'] = data.available_ns
    families = tuple(REGISTRY)
    cut = 2000
    expected = compute_batch(data, families)
    mutated = data.copy()
    rng = np.random.default_rng(1)
    for column in mutated.columns:
        if column in ('t_ns', 'available_ns', 'trade_moments_available_ns', 'epoch') or mutated[column].dtype.kind != 'f':
            continue
        values = mutated[column].to_numpy(float).copy()
        if column.startswith('c_'):     # cumulative counters stay cumulative
            values[cut:] += rng.integers(0, 3, len(values) - cut).cumsum()
        else:
            values[cut:] = values[cut:] * rng.uniform(.9, 1.1, len(values) - cut) + rng.normal(size=len(values) - cut) / 10
        mutated[column] = values
    # exact trade moments agree with the mutated counters, as the data contract requires
    traded = mutated.c_trade_buy + mutated.c_trade_sell + mutated.c_trade_unknown
    mutated['trade_volume'] = np.r_[2., np.diff(traded.to_numpy())]
    mutated['trade_price_mean'] = mutated.mid
    changed = compute_batch(mutated, families)
    assert changed.names == expected.names and len(families) >= 6
    np.testing.assert_array_equal(changed.values[:cut], expected.values[:cut])
    np.testing.assert_array_equal(changed.known[:cut], expected.known[:cut])
    assert not np.array_equal(changed.values[cut:], expected.values[cut:])


def test_key_columns_and_their_derivatives_stay_past():
    y = crossings()
    spread = (y.origin_ask - y.origin_bid).to_numpy()
    mid = ((y.origin_ask + y.origin_bid) / 2).to_numpy()
    origin = y.t_ns.rename('origin').to_numpy()
    check_model_input(['spread', 'mid', 'origin'], {'spread': spread, 'mid': mid, 'origin': origin})
    joined = pd.concat([y, y], ignore_index=True)           # metadata lost, taint by name kept
    with pytest.raises(FutureLeak):
        check_model_input(['x'], {'x': (joined.future_bid + 1).to_numpy()})
    check_model_input(['ask'], {'ask': joined.origin_ask.to_numpy()})


def test_binary_future_labels_are_fingerprinted():
    y = crossings()
    label = (y.future_status == 'CROSSED').to_numpy()
    with pytest.raises(FutureLeak):
        check_model_input(['crossed_soon'], {'crossed_soon': label.astype(float)})


def test_examples_accept_late_publication_and_bound_staleness():
    y = crossings()
    upper = y[(y.scale == 1) & (y.role == 'upper')].reset_index(drop=True)
    t = np.arange(0, 300, 7, dtype=np.int64) * NS
    available = t + 5 * NS
    available[3] = t[3] + 100 * NS                # one late publication
    x = FeatureBatch(t, available, (t / NS)[:, None].astype(float), np.ones((len(t), 1), bool), ('age',), ('s',))
    e = future_examples(x, upper)
    seen = e['x_available_ns'] >= 0
    assert np.all(e['x_available_ns'][seen] <= upper.t_ns.to_numpy()[seen])
    fresh = future_examples(x, upper, max_x_age_ns=0)
    assert not fresh['usable'].any()               # every X endpoint is older than its origin
