"""Marks that make future information detectable before it reaches model input.

Future is what is known only after an origin. It carries four marks:
1. name: every future field contains the word 'future' ('futures', the
   contract kind, is not the word);
2. type: FutureFrame/Series and FutureArray preserve provenance through
   pandas/NumPy slicing and arithmetic. Implicit NumPy coercion is rejected;
3. content: value fingerprints are a supplementary check for explicit exports;
4. time: future.future_examples joins X and Y only where X was published by
   the origin and Y matured after it.
Every public function of spectrum.future returns marked tables
(future_output). Model input (FeatureBatch, feature Context, SequenceStore,
observation store, to_feature_batch) calls check_model_input.
"""
import functools
import hashlib
import inspect
import re
import numpy as np
import pandas as pd

WORD = re.compile(r'future(?!s)', re.IGNORECASE)
MIN_LENGTH = 8      # shorter columns would fingerprint by coincidence
MAX_PRINTS = 1_000_000  # oldest fingerprints are forgotten first
_PRINTS = {}


class FutureLeak(ValueError):
    """Future information found where only the past is allowed."""


def _unwrap(value):
    if isinstance(value, FutureArray):
        return value._values
    if isinstance(value, FutureSeries):
        return pd.Series.to_numpy(value)
    if isinstance(value, FutureFrame):
        return pd.DataFrame.to_numpy(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_unwrap(v) for v in value)
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    return value


def _has_future(value):
    if is_future(value):
        return True
    if isinstance(value, (tuple, list)):
        return any(_has_future(v) for v in value)
    if isinstance(value, dict):
        return any(_has_future(v) for v in value.values())
    return False


def _wrap(value, source):
    if isinstance(value, (np.ndarray, np.generic)) or np.isscalar(value):
        return FutureArray(value, source)
    if isinstance(value, tuple):
        return tuple(_wrap(v, source) for v in value)
    if isinstance(value, list):
        return [_wrap(v, source) for v in value]
    return value


def _array_function(function, args, kwargs, source):
    # NumPy ufuncs already parse positional out before dispatch; array functions
    # do not. Check both spellings before any external buffer can be written.
    try:
        out = inspect.signature(function).bind_partial(*args, **kwargs).arguments.get('out')
    except (ValueError, TypeError):
        positions = {np.concatenate: 2, np.stack: 2, np.dot: 2}
        position = positions.get(function, len(args))
        out = args[position] if position < len(args) else kwargs.get('out')
    if out is not None:
        raise FutureLeak('Cannot write future results into an unmarked output')
    if function in (np.save, np.savez, np.savez_compressed, np.savetxt, np.copyto, np.putmask, np.place):
        raise FutureLeak('Export future values explicitly before serialization or mutation')
    return _wrap(function(*_unwrap(args), **_unwrap(kwargs)), source)


class FutureArray(np.lib.mixins.NDArrayOperatorsMixin):
    """Read-only future values, with provenance retained by NumPy operations.

    This is deliberately not an ndarray subclass: np.asarray silently strips
    ndarray subclasses. Export to an external Y consumer is an explicit act
    (export_future_values); it is never an input-admission operation.
    """
    __array_priority__ = 1000

    def __init__(self, values, source='spectrum.future'):
        self._values = np.array(_unwrap(values), copy=True)
        self._values.setflags(write=False)
        self.future_source = source
        if self._values.ndim == 1:
            _remember(self._values, source)

    def __array__(self, dtype=None, copy=None):
        raise FutureLeak(f'Implicit conversion of future values from {self.future_source} is forbidden; '
                         'use export_future_values only for an external Y consumer')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == 'at' or kwargs.get('out') is not None:
            raise FutureLeak('Cannot write future results into an unmarked output')
        return _wrap(getattr(ufunc, method)(*_unwrap(inputs), **_unwrap(kwargs)), self.future_source)

    def __array_function__(self, function, types, args, kwargs):
        return _array_function(function, args, kwargs, self.future_source)

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        raise FutureLeak('Export future values explicitly before iterating into an unmarked container')

    def __getitem__(self, key):
        return _wrap(self._values[_unwrap(key)], self.future_source)

    def __bool__(self):
        return bool(self._values)

    def __repr__(self):
        return f'FutureArray(source={self.future_source!r}, values={self._values!r})'

    @property
    def shape(self):
        return self._values.shape

    @property
    def ndim(self):
        return self._values.ndim

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def size(self):
        return self._values.size

    @property
    def T(self):
        return _wrap(self._values.T, self.future_source)

    def __getattr__(self, name):
        if name not in {'astype', 'copy', 'reshape', 'ravel', 'flatten', 'transpose',
                        'mean', 'std', 'var', 'sum', 'min', 'max', 'all', 'any', 'clip'}:
            raise AttributeError(name)
        def operation(*args, **kwargs):
            out_position = {'mean': 2, 'std': 2, 'var': 2, 'sum': 2, 'min': 1,
                            'max': 1, 'all': 1, 'any': 1, 'clip': 2}.get(name, len(args))
            positional_out = args[out_position] if out_position < len(args) else None
            if kwargs.get('out') is not None or positional_out is not None:
                raise FutureLeak('Cannot write future results into an unmarked output')
            return _wrap(getattr(self._values, name)(*_unwrap(args), **_unwrap(kwargs)), self.future_source)
        return operation


def export_future_values(value):
    """Detached unmarked Y for an external learner/serializer, never evidence of causal X.

    Outside this typed boundary arbitrary reconstruction, serialization or
    third-party transformations cannot be tracked. Fingerprints in this process
    remain a diagnostic only. Keep exported Y separate from feature builders.
    """
    if isinstance(value, FutureArray):
        raw = value._values
    elif isinstance(value, FutureFrame):
        raw = pd.DataFrame.to_numpy(value)
    elif isinstance(value, FutureSeries) and value.future_tainted:
        raw = pd.Series.to_numpy(value)
    else:
        raise TypeError('Expected marked future values')
    return np.array(raw, copy=True)


def clear_fingerprints():
    """Forget remembered future values (tests, or a fresh experiment process)."""
    _PRINTS.clear()


def is_future_name(name):
    return bool(WORD.search(str(name)))


class FutureSeries(pd.Series):
    """A column of a future table; what pandas derives from a future column is fingerprinted too.

    Columns except explicitly declared origin keys are tainted, including
    renamed columns. Arithmetic inherits taint from either operand. Placing a series into a plain
    DataFrame drops the type, so derived values are remembered at creation.
    """
    _metadata = ['future_source', 'future_tainted']
    future_source = 'spectrum.future'
    future_tainted = True

    @property
    def _constructor(self):
        return FutureSeries

    def __finalize__(self, other, method=None, **kwargs):
        result = super().__finalize__(other, method=method, **kwargs)
        if isinstance(other, pd.DataFrame):
            result.future_tainted = is_future_name(result.name) or result.name not in getattr(other, 'future_keys', ())
        if result.future_tainted:
            name = result.name if is_future_name(result.name) else f'{result.name} derived from future'
            _remember(pd.Series.to_numpy(result), f'{result.future_source}:{name}')
        return result

    def to_numpy(self, *args, **kwargs):
        values = super().to_numpy(*args, **kwargs)
        return FutureArray(values, self.future_source + ':' + str(self.name)) if self.future_tainted else values

    @property
    def values(self):
        return self.to_numpy()

    def __array__(self, dtype=None, copy=None):
        if self.future_tainted:
            raise FutureLeak('Implicit conversion of future Series is forbidden; use export_future_values for Y')
        return np.asarray(pd.Series.to_numpy(self), dtype=dtype)

    def _arith_method(self, other, op):
        return self._mark_result(super()._arith_method(other, op), other)

    def _cmp_method(self, other, op):
        return self._mark_result(super()._cmp_method(other, op), other)

    def _logical_method(self, other, op):
        return self._mark_result(super()._logical_method(other, op), other)

    def _mark_result(self, result, other):
        result.future_tainted = self.future_tainted or _has_future(other)
        if result.future_tainted:
            _remember(pd.Series.to_numpy(result), self.future_source + ':derived future')
        return result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if _has_future(inputs) or _has_future(kwargs):
            if method == 'at' or kwargs.get('out') is not None:
                raise FutureLeak('Cannot write future results into an unmarked output')
            return _wrap(getattr(ufunc, method)(*_unwrap(inputs), **_unwrap(kwargs)), self.future_source)
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def __array_function__(self, function, types, args, kwargs):
        if not _has_future(args) and not _has_future(kwargs):
            return function(*_unwrap(args), **_unwrap(kwargs))
        return _array_function(function, args, kwargs, self.future_source)

    @property
    def _constructor_expanddim(self):
        return FutureFrame


class FutureFrame(pd.DataFrame):
    """A table of future facts; columns sliced from it stay FutureSeries."""
    _metadata = ['future_source', 'future_keys']
    future_source = 'spectrum.future'
    future_keys = ()

    @property
    def _constructor(self):
        return FutureFrame

    @property
    def _constructor_sliced(self):
        return FutureSeries

    def __finalize__(self, other, method=None, **kwargs):
        result = super().__finalize__(other, method=method, **kwargs)
        if method == 'concat' and other.objs:
            result.future_keys = tuple(set.intersection(*(set(getattr(f, 'future_keys', ())) for f in other.objs)))
        return result

    def to_numpy(self, *args, **kwargs):
        return FutureArray(super().to_numpy(*args, **kwargs), self.future_source)

    @property
    def values(self):
        return self.to_numpy()

    def __array__(self, dtype=None, copy=None):
        raise FutureLeak('Implicit conversion of future Frame is forbidden; use export_future_values for Y')

    def rename(self, mapper=None, *, index=None, columns=None, axis=None, copy=None, inplace=False, level=None, errors='ignore'):
        old_columns = list(self.columns)
        old_keys = set(self.future_keys)
        result = super().rename(mapper=mapper, index=index, columns=columns, axis=axis,
                                copy=copy, inplace=inplace, level=level, errors=errors)
        target = self if inplace else result
        # Track column origins by position, including names that collide with a key.
        sources = {}
        for old, new in zip(old_columns, target.columns):
            sources.setdefault(new, []).append(old in old_keys)
        target.future_keys = tuple(name for name, past in sources.items() if all(past))
        return result


def is_future(value):
    return isinstance(value, (FutureFrame, FutureArray)) or isinstance(value, FutureSeries) and value.future_tainted


def _canonical(values):
    v = np.asarray(values)
    if v.ndim != 1 or v.dtype.kind not in 'fiub' or len(v) < MIN_LENGTH:
        return None
    v = v.astype(float) + 0.  # -0.0 and 0.0 are one value
    return v


def _prints(v):
    """Raw values (one NaN) and the masked storage form (unknown -> 0)."""
    finite = np.isfinite(v)
    return (hashlib.sha256(np.where(finite, v, np.nan).tobytes()).hexdigest(),
            hashlib.sha256(np.where(finite, v, 0.).tobytes()).hexdigest())


def _remember(values, label, replace=False):
    v = _canonical(values)
    if v is None:
        return
    finite = v[np.isfinite(v)]
    if not len(finite) or finite.min() == finite.max():   # constant columns would match by coincidence
        return
    # float32 storage (SequenceStore) is the same value rounded once
    for p in _prints(v) + _prints(v.astype(np.float32).astype(float)):
        if replace or p not in _PRINTS:
            _PRINTS.pop(p, None)
            _PRINTS[p] = label
    while len(_PRINTS) > MAX_PRINTS:
        del _PRINTS[next(iter(_PRINTS))]


def mark_future(result, source, keys=()):
    """Mark a future result: dicts and frames become FutureFrame, arrays are fingerprinted.

    Non-future columns are allowed only as keys (origin time, line, origin
    quote); every other column must be named future.
    """
    if isinstance(result, tuple):
        return tuple(mark_future(r, source, keys) for r in result)
    if isinstance(result, dict) and result and all(np.ndim(v) == 1 for v in result.values()):
        result = pd.DataFrame(result)
    if isinstance(result, pd.DataFrame):
        unnamed = [c for c in result.columns if not is_future_name(c) and c not in keys]
        if unnamed:
            raise FutureLeak(f'{source}: future table columns must be named future: {unnamed[:5]}')
        frame = FutureFrame(result)
        frame.future_source = source
        frame.future_keys = tuple(keys)
        for column in frame.columns:
            if is_future_name(column):
                _remember(pd.Series.to_numpy(frame[column]), f'{source}:{column}', replace=True)
        return frame
    if isinstance(result, np.ndarray):
        return FutureArray(result, source)
    return result


def future_output(keys=()):
    """Decorator for public functions of spectrum.future: their results are marked."""
    def wrap(function):
        source = function.__module__ + '.' + function.__qualname__

        @functools.wraps(function)
        def marked(*args, **kwargs):
            return mark_future(function(*args, **kwargs), source, keys)
        marked.future_output = True
        return marked
    return wrap


def future_join(function):
    """Marks a spectrum.future function that returns causal X and marked future Y apart."""
    function.future_join = True
    return function


def check_model_input(names, columns=None):
    """Raise FutureLeak if a name, a type or a value carries future information.

    ``columns`` maps names to arrays, or is a 2-D matrix aligned with names.
    """
    names = [str(n) for n in names]
    bad = [n for n in names if is_future_name(n)]
    if bad:
        raise FutureLeak('Future fields cannot enter model input: ' + ', '.join(bad[:5]))
    if columns is None:
        return
    if is_future(columns):
        raise FutureLeak('Model input is a future table from ' + str(getattr(columns, 'future_source', '?')))
    if isinstance(columns, np.ndarray) and columns.ndim == 2:
        items = ((n, columns[:, i]) for i, n in enumerate(names))
    else:
        items = ((n, columns[n]) for n in names)
    for name, values in items:
        if is_future(values):
            raise FutureLeak(f'Field {name} comes from a future table ({getattr(values, "future_source", "?")})')
        if getattr(values, 'dtype', None) == object and any(is_future(v) for v in np.asarray(values).flat):
            raise FutureLeak(f'Field {name} contains nested future values')
        if not _PRINTS:
            continue
        v = _canonical(values)
        if v is None:
            continue
        for p in _prints(v):
            if p in _PRINTS:
                raise FutureLeak(f'Field {name} has the values of future column {_PRINTS[p]}')
