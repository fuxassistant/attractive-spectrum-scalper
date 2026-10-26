"""Bounded live/replay adapter over the SAME feature formulas as dataset jobs."""
from pathlib import Path
import pandas as pd
from spectrum.features.pipeline import compute
from spectrum.features.pipeline import REGISTRY
from spectrum.features.pipeline import load_plugins
from spectrum.market.calendar import MarketCalendar
from spectrum.core.storage import sha
from spectrum.core.state import seal
from spectrum.core.state import unseal


class FeatureStream:
    def __init__(self,families=('motion','flow','spectral'),*,plugins=(),calendar=None,source_id=''):
        self.plugins=tuple(plugins);self.plugin_sources=load_plugins(plugins)
        self.families=tuple(families);self.calendar=calendar or MarketCalendar();self.source_id=source_id
        if not self.families or len(set(self.families)) != len(self.families) or any(n not in REGISTRY for n in self.families):
            raise ValueError('Unknown, empty or duplicate feature family selection')
        self.halo=max(REGISTRY[n].lookback_seconds for n in self.families)+1
        self.tail=None;self.last_ns=None

    def feed(self,frame):
        if not len(frame):raise ValueError('Empty feature batch')
        if not frame.t_ns.is_monotonic_increasing or frame.t_ns.duplicated().any():raise ValueError('Feature clock order')
        if self.last_ns is not None and int(frame.t_ns.iloc[0])<=self.last_ns:raise ValueError('Feature source regressed')
        joined=frame if self.tail is None else pd.concat([self.tail,frame],ignore_index=True)
        result=compute(joined,self.families,self.calendar);n=len(frame)
        self.tail=joined.tail(self.halo).copy();self.last_ns=int(frame.t_ns.iloc[-1])
        # Detached output; no caller can mutate retained history through a view.
        return dict(t_ns=frame.t_ns.to_numpy(copy=True),available_ns=frame.available_ns.to_numpy(copy=True),
                    values={k:v[-n:].copy() for k,v in result.values.items()},
                    known={k:v[-n:].copy() for k,v in result.masks.items()},units=dict(result.units))

    def checkpoint(self):
        return seal(dict(contract='FEATURE_STREAM_V3',source_id=self.source_id,families=self.families,plugins=self.plugins,
                         plugin_sources=self.plugin_sources,calendar=self.calendar.contract(),
                         formula_sha256=implementation_identity(),last_ns=self.last_ns,
                         adapter_sha256=sha(__file__),calendar_sha256=sha(Path(inspect.getfile(MarketCalendar))),
                         tail=None if self.tail is None else self.tail.to_dict(orient='list')))

    def feed_batch(self, frame):
        """Typed result using exactly the same computation as feed()."""
        from .contract import FeatureBatch
        output = self.feed(frame)
        return FeatureBatch.from_columns(output['t_ns'], output['available_ns'],
                                         output['values'], output['known'], output['units'])

    @classmethod
    def restore(cls,state,*,expected_source_id):
        s=unseal(state)
        if s['contract']!='FEATURE_STREAM_V3' or s['source_id']!=expected_source_id:raise ValueError('Feature source/contract drift')
        if s['formula_sha256']!=implementation_identity():raise ValueError('Feature formula drift')
        if s['adapter_sha256']!=sha(__file__) or s['calendar_sha256']!=sha(Path(inspect.getfile(MarketCalendar))):raise ValueError('Feature adapter/calendar drift')
        obj=cls(s['families'],plugins=s['plugins'],calendar=MarketCalendar.restore(s['calendar']),source_id=expected_source_id)
        if s['plugin_sources']!=obj.plugin_sources:raise ValueError('Plugin source drift')
        obj.tail=None if s['tail'] is None else pd.DataFrame(s['tail']);obj.last_ns=s['last_ns'];return obj

from .pipeline import implementation_identity
import inspect
