"""Shared causal-window storage and explicit training-row normalization."""
import hashlib
import numpy as np
from spectrum.features.contract import reject_future

def _ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


class BaseSequenceStore:
    """In-memory N x F values with lazy family projection and causal windows."""

    def __init__(self, frame, raw, feature_names, groups, *, source_receipt=None):
        reject_future(feature_names, raw)
        self.frame = frame.reset_index(drop=True).copy()
        self.raw = np.asarray(raw)
        reject_future(feature_names, self.raw)
        self.feature_names = list(feature_names)
        self.groups = {k: list(v) for k, v in groups.items()}
        self.source_receipt = source_receipt or {}
        required = {"date", "t_ns", "feature_available", "segment_token"}
        if not required <= set(self.frame):
            raise ValueError("Missing origin columns: " + str(sorted(required - set(self.frame))))
        if self.raw.dtype != np.float32 or self.raw.ndim != 2 or self.raw.shape != (len(frame), len(feature_names)):
            raise ValueError("Expected aligned raw float32[N,F] feature matrix")
        if not len(frame) or len(set(feature_names)) != len(feature_names):
            raise ValueError("Nonempty rows and unique feature names required")
        if self.frame.feature_available.dtype != bool:
            raise ValueError("feature_available must be boolean")
        if not np.issubdtype(self.frame.t_ns.dtype, np.integer) or not np.issubdtype(self.frame.segment_token.dtype, np.integer):
            raise ValueError("Integer nanosecond clock and support segment required")
        if not self.frame.t_ns.is_monotonic_increasing or self.frame.t_ns.duplicated().any():
            raise ValueError("Strict chronological unique origins required")
        for family, names in self.groups.items():
            if not names or len(names) != len(set(names)) or not set(names) <= set(feature_names):
                raise ValueError("Invalid feature family: " + family)
        self._columns = {k: np.array([self.feature_names.index(n) for n in names], dtype=np.int64)
                         for k, names in self.groups.items()}
        self._segments = {}

    def segment_starts(self, cadence_seconds):
        """O(N) starts; support/date/missing-X resets, never unknown labels."""
        step = int(round(float(cadence_seconds) * 1e9))
        if not np.isfinite(cadence_seconds) or step <= 0:
            raise ValueError("Positive finite cadence required")
        if step not in self._segments:
            f = self.frame
            t, date = f.t_ns.to_numpy(), f.date.to_numpy()
            ok, token = f.feature_available.to_numpy(), f.segment_token.to_numpy()
            reset = np.ones(len(f), dtype=bool)
            reset[1:] = ((np.diff(t) != step) | (date[1:] != date[:-1]) |
                         (token[1:] != token[:-1]) | ~ok[1:] | ~ok[:-1])
            self._segments[step] = np.maximum.accumulate(np.where(reset, np.arange(len(f)), 0))
        return self._segments[step]

    def batch(self, root_indices, *, family, length, cadence_seconds, scaler):
        """Build only [B,T,2F+1], including observed-feature and real-row flags."""
        ids = np.asarray(root_indices)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= len(self.frame)):
            raise ValueError("Valid integer origin indices required")
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError("Positive integer window length required")
        if family not in self._columns or scaler["feature_names"] != self.feature_names:
            raise ValueError("Family or shared scaler schema mismatch")
        columns = self._columns[family]
        idx = ids[:, None] - np.arange(length - 1, -1, -1)
        starts = self.segment_starts(cadence_seconds)
        real = (idx >= starts[ids, None]) & (idx >= 0) & self.frame.feature_available.to_numpy()[ids, None]
        safe = np.maximum(idx, 0)
        values = self.raw[safe[..., None], columns]
        finite = np.isfinite(values)
        mean = np.asarray(scaler["mean"], np.float32)[columns]
        scale = np.asarray(scaler["scale"], np.float32)[columns]
        scaled = np.where(finite, (values - mean) / scale, 0.)
        scaled = np.clip(scaled, -scaler["cap"], scaler["cap"]).astype(np.float32)
        result = np.concatenate((scaled, finite.astype(np.float32), real[..., None].astype(np.float32)), axis=-1)
        result[~real] = 0.
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite lazy batch after normalization")
        return result


def fit_scaler(store, train_rows, chunk_size=8192):
    """One shared full-schema scaler on causal TRAIN rows, independent of Y.

    Stable merged population moments avoid retaining a float64 copy of N x F.
    All feature families slice these exact same means, scales and cap.
    """
    ids = np.unique(np.asarray(train_rows, dtype=np.int64))
    if not len(ids) or np.any(ids < 0) or np.any(ids >= len(store.frame)):
        raise ValueError("Nonempty valid scaler training roots required")
    ids = ids[store.frame.feature_available.to_numpy()[ids]]
    if not len(ids):
        raise ValueError("No causal TRAIN feature rows")
    size = len(store.feature_names)
    count, mean, m2 = np.zeros(size, np.int64), np.zeros(size), np.zeros(size)
    for start in range(0, len(ids), chunk_size):
        x = store.raw[ids[start:start + chunk_size]].astype(np.float64)
        good = np.isfinite(x)
        n = good.sum(axis=0)
        batch_mean = np.divide(np.where(good, x, 0.).sum(axis=0), n, out=np.zeros(size), where=n > 0)
        batch_m2 = np.where(good, x - batch_mean, 0.) ** 2
        batch_m2 = batch_m2.sum(axis=0)
        total, delta = count + n, batch_mean - mean
        mean += np.divide(delta * n, total, out=np.zeros(size), where=total > 0)
        m2 += batch_m2 + np.divide(delta ** 2 * count * n, total, out=np.zeros(size), where=total > 0)
        count = total
    std = np.sqrt(np.divide(m2, count, out=np.zeros(size), where=count > 0))
    scale = np.where(std > 1e-8, std, 1.)
    return dict(feature_names=store.feature_names, mean=mean.tolist(), scale=scale.tolist(), cap=10.,
                finite_count=count.tolist(), all_missing_columns=np.flatnonzero(count == 0).tolist(),
                constant_columns=np.flatnonzero((count > 0) & (std <= 1e-8)).tolist(),
                train_feature_row_count=len(ids), train_feature_indices_sha256=_ids_sha(ids),
                method="all unique causal TRAIN rows; label-independent; merged population moments; cap10")
