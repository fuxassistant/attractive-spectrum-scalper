"""Layer direction and declared measures across the library.

Any upward import or an unregistered unit constant fails.
"""
import ast
import importlib
from pathlib import Path
import spectrum
from spectrum import units

ROOT = Path(spectrum.__file__).parent
BASE = {'units', 'canon', 'resources', 'io', 'leakage'}
ALLOWED = {
    'core': set(),
    'market': {'core'},
    'features': {'core', 'market'},
    'representation': {'core', 'market', 'features'},
    'crossings': {'core', 'market'},
    'execution': {'core', 'market'},
    'future': {'core', 'market', 'crossings', 'features'},
    'economics': {'core', 'market', 'crossings', 'future', 'execution'},
    'units': {'resources'},
    'canon': {'resources'},
    'resources': set(),
    'io': {'core'},
    'leakage': set(),
}
TICK_UNITS = {
    'core/schema.py': units.GC_TICK_USD,
    'features/families/volatility_vwap.py': units.GC_TICK_USD,
    'features/families/queue_state.py': units.GC_TICK_NANO,
}
CLOCK_UNITS = {'NS': units.NS, 'NANO': units.NS, 'SECOND': units.NS, 'MINUTE': units.MINUTE,
               'TICK_NANO': units.GC_TICK_NANO}


def sources():
    return sorted(p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts)


def package(path):
    parts = path.relative_to(ROOT).parts
    return parts[0] if len(parts) > 1 else path.stem


def imported(path):
    """Absolute spectrum modules imported anywhere in a file, lazy imports included."""
    here = ('spectrum',) + path.relative_to(ROOT).with_suffix('').parts
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent = here[:len(here) - node.level]
                names.append('.'.join(parent + ((node.module,) if node.module else ())))
            else:
                names.append(node.module or '')
    return [n for n in names if n.startswith('spectrum.')]


def test_layers_import_only_downward():
    found = {}
    for path in sources():
        source = package(path)
        if source in {'cli', '__main__'}:
            continue
        for name in imported(path):
            target = name.split('.')[1]
            if target == source or target in BASE and source not in {'units', 'canon', 'resources', 'leakage'}:
                continue
            if target not in ALLOWED.get(source, set()):
                found.setdefault((source, target), set()).add(path.relative_to(ROOT).as_posix())
    assert found == {}


def test_unit_constants_match_the_canon_and_tick_homonyms_are_registered():
    """Every module-level clock/tick name, defined or imported, carries its canon value.

    TICK is a homonym (USD or nano-USD), so each module DEFINING it is registered.
    """
    ticks = {}
    for path in sources():
        defined, names = set(), set()
        for node in ast.parse(path.read_text(encoding='utf8')).body:
            if isinstance(node, ast.Assign):
                defined |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            elif isinstance(node, ast.ImportFrom):
                names |= {a.asname or a.name for a in node.names}
        names = (names | defined) & (set(CLOCK_UNITS) | {'TICK'})
        if not names:
            continue
        relative = path.relative_to(ROOT).as_posix()
        module = importlib.import_module('spectrum.' + relative[:-3].replace('/', '.'))
        for name in names:
            if name != 'TICK':
                assert getattr(module, name) == CLOCK_UNITS[name], (relative, name)
            elif name in defined:
                ticks[relative] = getattr(module, name)
            else:
                assert getattr(module, name) in TICK_UNITS.values(), relative
    assert ticks == TICK_UNITS


def test_representation_boundary_turns_nan_into_masked_zero_and_keeps_measured_zero():
    import numpy as np
    import pandas as pd
    import pytest
    from spectrum.representation import to_feature_batch
    frame = pd.DataFrame(dict(t_ns=np.array([1, 2, 3], np.int64) * units.NS,
                              available_ns=np.array([1, 2, 3], np.int64) * units.NS + 5,
                              aggression30=[0.0, np.nan, 0.4], mechanism_known=[True, True, False]))
    batch = to_feature_batch(frame, {'aggression30': 'ratio'}, known={'aggression30': 'mechanism_known'})
    assert batch.values[:, 0].tolist() == [0.0, 0.0, 0.0]
    assert batch.known[:, 0].tolist() == [True, False, False]
    frame.loc[0, 'aggression30'] = np.inf
    with pytest.raises(ValueError):
        to_feature_batch(frame, {'aggression30': 'ratio'})


def test_tape_replay_names_fact_and_delivery_age(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from spectrum.core.storage import sha
    from spectrum.market import TapeReplay
    table = pa.table(dict(t_ns=[10 * units.NS, 20 * units.NS], available_ns=[11 * units.NS, 22 * units.NS],
                          bid=[1.0, 1.0], ask=[1.1, 1.1], valid=[True, True]))
    pq.write_table(table, tmp_path / 'quotes.parquet')
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        status='COMPLETE', payloads={'quotes.parquet': sha(tmp_path / 'quotes.parquet')})))
    row = TapeReplay(tmp_path).advance(25 * units.NS)
    assert row['fact_age_ns'] == row['quote_age_ns'] == 5 * units.NS
    assert row['delivery_age_ns'] == 3 * units.NS


def test_future_fields_cannot_enter_model_input():
    import numpy as np
    import pytest
    from spectrum.features.contract import FeatureBatch
    t = np.array([1, 2], np.int64)
    with pytest.raises(ValueError, match='Future fields'):
        FeatureBatch(t, t, np.zeros((2, 1)), np.ones((2, 1), bool), ('future_move_ticks_30',), ('tick',))
    with pytest.raises(ValueError, match='Future fields'):
        FeatureBatch(t, t, np.zeros((2, 1)), np.ones((2, 1), bool), ('Future_E',), ('band',))


def test_every_public_future_name_says_future():
    import inspect
    import spectrum.future as future
    import pkgutil, importlib
    for info in pkgutil.iter_modules(future.__path__):
        module = importlib.import_module('spectrum.future.' + info.name)
        for name, value in vars(module).items():
            if inspect.isfunction(value) and value.__module__ == module.__name__ and not name.startswith('_'):
                assert 'future' in name, (info.name, name)
