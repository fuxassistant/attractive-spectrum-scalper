"""Reusable formulas extracted from spectrum/lab.py; see docs/MIGRATION.json."""
from dataclasses import dataclass
import numpy as np
from spectrum.leakage import check_model_input


def reject_future(names, columns=None):
    """Model input never carries future information: name, type and values are checked (spectrum.leakage)."""
    check_model_input(names, columns)


@dataclass(frozen=True)
class FeatureBatch:
    """Named frame observations with separate endpoint and publication clocks.

    ``known=False`` means unknown; its numerical storage is zero. Consumers
    must join on available_ns, never backdate observations to t_ns.
    """
    t_ns: np.ndarray
    available_ns: np.ndarray
    values: np.ndarray
    known: np.ndarray
    names: tuple[str, ...]
    units: tuple[str, ...]

    def __post_init__(self):
        reject_future(self.names, self.values)
        reject_future(self.names, self.known)
        times = np.asarray(self.t_ns)
        available = np.asarray(self.available_ns)
        values = np.asarray(self.values)
        known = np.asarray(self.known)
        if times.dtype != np.int64 or available.dtype != np.int64:
            raise TypeError("Feature clocks must be integer UTC nanoseconds")
        if times.ndim != 1 or available.shape != times.shape:
            raise ValueError("Feature clocks must have matching one-dimensional shapes")
        if values.shape != (len(times), len(self.names)) or known.shape != values.shape:
            raise ValueError("Feature matrix, masks and names must align")
        if known.dtype != bool or not np.isfinite(values).all():
            raise ValueError("Finite storage and explicit boolean masks required")
        reject_future(self.names, values)
        if len(set(self.names)) != len(self.names) or len(self.units) != len(self.names):
            raise ValueError("Unique feature names and one unit per feature required")
        if np.any(np.diff(times) <= 0) or np.any(available < times):
            raise ValueError("Ordered endpoints and publication no earlier than endpoint required")
        if np.any(values[~known] != 0):
            raise ValueError("Unknown feature storage must be zero")

    @classmethod
    def from_columns(cls, t_ns, available_ns, values, known, units):
        names = tuple(values)
        reject_future(names, values)
        reject_future(names, known)
        if set(known) != set(names) or set(units) != set(names):
            raise ValueError("Every feature requires a mask and unit")
        return cls(np.asarray(t_ns, dtype=np.int64).copy(),
                   np.asarray(available_ns, dtype=np.int64).copy(),
                   np.column_stack([values[n] for n in names]),
                   np.column_stack([known[n] for n in names]),
                   names, tuple(units[n] for n in names))

def rolling_valid(valid,n):
    prefix=np.r_[0,np.cumsum(np.asarray(valid,dtype=np.int64))]
    out=np.zeros(len(valid),bool)
    if len(out)>=n:out[n-1:]=prefix[n:]-prefix[:-n]==n
    return out


class Context:
    def __init__(self,frame,calendar):
        reject_future(frame.columns, frame)
        self.df=frame;self.t=frame.t_ns.to_numpy()
        self.closed=calendar.closed(self.t)
        if 'scheduled_closed' in frame and not np.array_equal(frame.scheduled_closed.to_numpy(bool), self.closed):
            raise ValueError('scheduled_closed disagrees with the selected calendar; regenerate stale data')
        self.supported=frame.valid.to_numpy()&~self.closed
        if 'reference_observed_open' in frame:
            self.supported &= frame.reference_observed_open.to_numpy(bool)
        self.admitted=(self.supported&
                       ((frame.available_ns-frame.t_ns).to_numpy()<=10**9))
        continuity=frame.segment_id.to_numpy() if 'segment_id' in frame else frame.epoch.to_numpy()
        self.same=np.r_[False,(np.diff(self.t)==10**9)&(np.diff(continuity)==0)]
        # A late publication can prove supported PAST without being a fresh
        # decision origin. Once known, that past remains usable by later windows.
        self.edge=self.supported&np.r_[False,self.supported[:-1]]&self.same
        self.values={};self.masks={};self.units={}

    def add(self,name,values,known,unit):
        reject_future([name], {name: values})
        reject_future([name], {name: known})
        if name in self.values:raise ValueError('Duplicate feature '+name)
        v=np.asarray(values,float);mask=np.asarray(known,bool)&np.isfinite(v)&self.admitted
        if v.shape!=self.t.shape or mask.shape!=self.t.shape:raise ValueError('Feature shape')
        self.values[name]=np.where(mask,v,0.);self.masks[name]=mask;self.units[name]=unit

    def delta(self,column):
        v=self.df[column].to_numpy(float)
        return np.r_[0.,np.diff(v)]


@dataclass(frozen=True)
class FeaturePlugin:
    name: str
    version: str
    compute: object
    lookback_seconds: int
