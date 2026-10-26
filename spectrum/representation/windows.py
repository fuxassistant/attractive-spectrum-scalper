"""Lazy causal sequence windows, independent of targets and learners."""
import numpy as np
from ._windows import BaseSequenceStore, fit_scaler

class SequenceStore(BaseSequenceStore):
    """One NxF matrix, two causal window definitions, identical input schema.

    The last channel denotes a supported clock position, including positions
    with no admitted X. Those positions have zero values AND zero feature
    masks. No forward fill, target-based reset, or new age channel is used.
    """

    def __init__(self, frame, raw, feature_names, groups, *, source_receipt=None):
        required = {"legacy_segment_token", "physical_segment_token", "physical_support_available"}
        if not required <= set(frame):
            raise ValueError("Missing physical context columns: " + str(sorted(required - set(frame))))
        f = frame.copy()
        if "segment_token" in f and not np.array_equal(f.segment_token, f.legacy_segment_token):
            raise ValueError("Compatibility segment_token must equal legacy_segment_token")
        f["segment_token"] = f.legacy_segment_token
        super().__init__(f, raw, feature_names, groups, source_receipt=source_receipt)
        if f.physical_support_available.dtype != bool:
            raise ValueError("physical_support_available must be boolean")
        for column in ("legacy_segment_token", "physical_segment_token"):
            if not np.issubdtype(f[column].dtype, np.integer):
                raise ValueError("Integer support tokens required")
        if (f.feature_available & ~f.physical_support_available).any():
            raise ValueError("Admitted current X requires physical support")
        causal = ("source_endpoint_ns", "source_available_ns")
        if all(column in f for column in causal):
            admitted = f.feature_available.to_numpy()
            endpoint, available, origin = (f[column].to_numpy()[admitted] for column in (*causal, "t_ns"))
            if np.any(endpoint > available) or np.any(available > origin):
                raise ValueError("Admitted X cannot precede its source endpoint/publication")

    def segment_starts(self, cadence_seconds, context_mode="restored"):
        if context_mode not in {"legacy", "restored"}:
            raise ValueError("Unsupported context mode")
        if not np.isfinite(cadence_seconds) or cadence_seconds <= 0:
            raise ValueError("Positive finite cadence required")
        step = int(round(float(cadence_seconds) * 1e9))
        if step <= 0:
            raise ValueError("Positive nanosecond cadence required")
        key = (context_mode, step)
        if key not in self._segments:
            f = self.frame
            t, date = f.t_ns.to_numpy(), f.date.to_numpy()
            if context_mode == "legacy":
                ok, token = f.feature_available.to_numpy(), f.legacy_segment_token.to_numpy()
            else:
                ok, token = f.physical_support_available.to_numpy(), f.physical_segment_token.to_numpy()
            reset = np.ones(len(f), bool)
            reset[1:] = ((np.diff(t) != step) | (date[1:] != date[:-1]) |
                         (token[1:] != token[:-1]) | ~ok[1:] | ~ok[:-1])
            self._segments[key] = np.maximum.accumulate(np.where(reset, np.arange(len(f)), 0))
        return self._segments[key]

    def batch(self, root_indices, *, family, length, cadence_seconds, scaler, context_mode="restored"):
        ids = np.asarray(root_indices)
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids >= len(self.frame))):
            raise ValueError("Valid integer origin indices required")
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError("Positive integer window length required")
        if family not in self._columns or scaler["feature_names"] != self.feature_names:
            raise ValueError("Family or shared scaler schema mismatch")
        columns = self._columns[family]
        idx = ids[:, None] - np.arange(length - 1, -1, -1)
        starts = self.segment_starts(cadence_seconds, context_mode)
        safe = np.maximum(idx, 0)
        support = self.frame.physical_support_available.to_numpy()
        clock = (idx >= starts[ids, None]) & (idx >= 0) & support[safe] & support[ids, None]
        values = self.raw[safe[..., None], columns]
        observed = np.isfinite(values) & self.frame.feature_available.to_numpy()[safe, None] & clock[..., None]
        mean = np.asarray(scaler["mean"], np.float32)[columns]
        scale = np.asarray(scaler["scale"], np.float32)[columns]
        scaled = np.where(observed, (values - mean) / scale, 0.)
        scaled = np.clip(scaled, -scaler["cap"], scaler["cap"]).astype(np.float32)
        result = np.concatenate((scaled, observed.astype(np.float32), clock[..., None].astype(np.float32)), axis=-1)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite lazy batch after normalization")
        return result
