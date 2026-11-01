"""Causal, falsifiable local hypotheses and delayed online recalibration.

This experiment does not change Spectrum runtime or historical receipts.
The five templates can overlap; none is a true regime or an order instruction.
"""
from __future__ import annotations

import heapq
import numpy as np
import pandas as pd

NS = 1_000_000_000
CLASSES = ('RETURN_FIRST', 'ADVERSE_FIRST', 'TIMEOUT')
FAMILIES = {
    'memory_return': ('Память удержания', 0),
    'permeable_departure': ('Проходимость и уход', 1),
    'failed_escape': ('Отмена ухода', 0),
    'place_migration': ('Старое место теряет актуальность', 1),
    'tight_range': ('Тесный диапазон сохраняется', 2),
}
PATH_COLUMNS = ['past_net30_r', 'past_range120_r', 'past_net120_r',
                'past_migration_r', 'past_zone_share_old', 'past_zone_share_new',
                'past_coverage', 'past_path30_r', 'past_efficiency30']
MEMORY_COLUMNS = ['x_mem_recent_pass_fraction', 'x_mem_log_median_dwell_seconds',
                  'x_mem_log_recent_gate_aborts', 'x_mem_log_completed_visits']


def decision_cohort(frame, seconds=300):
    """Past-only, one zone per clock and non-overlapping forward windows."""
    f = frame.sort_values(['t_ns', 'distance_r', 'zone_low_nano', 'candidate_id'], kind='stable')
    f = f.drop_duplicates('t_ns').copy()
    keep = []; last = None
    for i, t in zip(f.index, f.t_ns):
        if last is None or int(t)-last >= seconds*NS:
            keep.append(i); last = int(t)
    return f.loc[keep].reset_index(drop=True)


def causal_paths(quotes, frame, history_seconds=600, gap_seconds=30):
    """One-second held-quote samples, gated by EVERY native publication.

    Sampling is a documented occupancy proxy, not exact time at price. The
    native validity/epoch/gap audit prevents a one-second grid hiding outages.
    All samples lie in [t-600s,t); the endpoint t is used only for net changes.
    """
    qt = quotes.available_ns.to_numpy(np.int64)
    if len(qt) == 0 or np.any(np.diff(qt)<0):
        raise ValueError('Nonempty ordered publication tape required')
    mid2 = quotes.bid_nano.to_numpy(np.int64)+quotes.ask_nano.to_numpy(np.int64)
    epoch = quotes.epoch.to_numpy(np.int64)
    good = (quotes.valid & quotes.epoch.ge(0) & quotes.bid_nano.gt(0) &
            quotes.ask_nano.gt(quotes.bid_nano)).to_numpy(bool)
    bad = np.r_[0, np.cumsum(~good)]
    breaks = np.r_[False, (np.diff(qt)>gap_seconds*NS) | (np.diff(epoch)!=0)]
    breaks = np.r_[0, np.cumsum(breaks)]
    rows = []
    offsets = np.arange(-history_seconds,0,dtype=np.int64)*NS
    for r in frame.itertuples():
        t = int(r.t_ns); grid = t+offsets
        ids = np.searchsorted(qt,grid,'right')-1
        end = int(np.searchsorted(qt,t,'right'))-1
        known = bool(ids[0]>=0 and end>=0)
        if known:
            start = int(ids[0])
            known = bool(bad[end+1]==bad[start] and breaks[end+1]==breaks[start+1]
                         and grid[0]-qt[start]<=gap_seconds*NS and t-qt[end]<=gap_seconds*NS)
        row = dict(candidate_id=int(r.candidate_id),past_known=known,
                   past_max_available_ns=int(qt[end]) if end>=0 else -1,
                   past_coverage=1. if known else 0.)
        for c in PATH_COLUMNS:
            if c!='past_coverage': row[c]=np.nan
        if known:
            p = mid2[ids]/(2*NS); current = mid2[end]/(2*NS)
            risk = float(r.risk_per_oz); side = int(r.side)
            if not np.isfinite(risk) or risk<=0: raise ValueError('Positive known risk required')
            z = np.r_[p[-30:],current]
            path = np.abs(np.diff(z)).sum()/risk
            inzone = (2*r.zone_low_nano<=mid2[ids]) & (mid2[ids]<2*r.zone_high_nano)
            half=history_seconds//2
            row.update(past_net30_r=side*(current-p[-30])/risk,
                       past_range120_r=(max(p[-120:].max(),current)-min(p[-120:].min(),current))/risk,
                       past_net120_r=side*(current-p[-120])/risk,
                       past_migration_r=side*(p[half:].mean()-p[:half].mean())/risk,
                       past_zone_share_old=float(inzone[:half].mean()),
                       past_zone_share_new=float(inzone[half:].mean()),
                       past_path30_r=path,
                       past_efficiency30=abs(current-p[-30])/risk/path if path>0 else 0.)
        rows.append(row)
    return pd.DataFrame(rows)


def define_hypotheses(frame, config, thresholds=None):
    f=frame.copy(); spec=config['thresholds']
    abort_rate=np.expm1(f.x_mem_log_recent_gate_aborts)/np.minimum(f.completed_visits,8)
    if thresholds is None:
        train=f.block.eq('train')
        if not train.any(): raise ValueError('Training prefix required for X thresholds')
        thresholds={'median_dwell_seconds':float(f.loc[train,'median_dwell_seconds'].quantile(spec['memory_dwell_quantile'])),
                    'aborts_per_recent_visit':float(abort_rate[train].median())}
    f['past_aborts_per_visit']=abort_rate
    f['h_memory_return']=(f.median_dwell_seconds>=thresholds['median_dwell_seconds']) & (f.x_mem_recent_pass_fraction<=spec['recent_pass_low'])
    f['h_permeable_departure']=(f.x_mem_recent_pass_fraction>=spec['recent_pass_high']) & f.x_mem_last_pass.eq(1) & f.past_known & f.past_net30_r.le(0)
    f['h_failed_escape']=f.past_known & f.past_net30_r.ge(spec['return_progress_r']) & abort_rate.ge(thresholds['aborts_per_recent_visit'])
    f['h_place_migration']=f.past_known & (f.median_dwell_seconds>=thresholds['median_dwell_seconds']) & f.past_migration_r.le(-spec['migration_away_r']) & f.past_zone_share_old.gt(0) & (f.past_zone_share_new<.5*f.past_zone_share_old)
    f['h_tight_range']=f.past_known & f.past_range120_r.le(spec['range120_r']) & f.past_net120_r.abs().le(spec['net120_abs_r'])
    f['active_hypotheses']=f[['h_'+k for k in FAMILIES]].sum(axis=1)
    return f,thresholds


class DelayedCalibrator:
    """Observed/expected frequency correction with common delayed maturity.

    Does not count correlated active hypotheses as independent evidence.
    Counts are fractional exponentially discounted support, not sample sizes.
    """
    def __init__(self,half_life_seconds=7200,prior_count=20):
        self.half_life_ns=half_life_seconds*NS; self.prior=float(prior_count)
        self.observed={};self.expected={};self.last={};self.pending=[];self.seq=0
        self.max_used_origin=-1; self.max_used_release=-1; self.used=0

    def _decay(self,key,t):
        if key not in self.last:
            self.observed[key]=np.zeros(3);self.expected[key]=np.zeros(3);self.last[key]=t
        factor=2.**(-(t-self.last[key])/self.half_life_ns)
        self.observed[key]*=factor;self.expected[key]*=factor;self.last[key]=t

    def advance(self,t):
        while self.pending and self.pending[0][0]<t:
            release,_,origin,keys,p,label=self.pending[0]
            heapq.heappop(self.pending)
            if label<0:continue
            for key in ('global',)+tuple(keys):
                self._decay(key,release)
                self.observed[key][label]+=1;self.expected[key]+=p
            self.used+=1;self.max_used_origin=max(self.max_used_origin,origin)
            self.max_used_release=max(self.max_used_release,release)

    def predict(self,t,p,keys,local=False):
        self.advance(t); factors=[]; support=[]
        for key in ('global',)+(tuple(keys) if local else ()):
            self._decay(key,t)
            # Symmetric pseudocounts shrink each correction to one.
            factors.append((self.observed[key]+self.prior/3)/(self.expected[key]+self.prior/3))
            support.append(float(self.observed[key].sum()))
        q=p*np.mean(factors,axis=0); q=q/q.sum()
        return q,{'feedback_count':self.used,'max_feedback_origin_ns':self.max_used_origin,
                  'max_feedback_release_ns':self.max_used_release,'global_effective_support':support[0],
                  'local_effective_support':float(np.mean(support[1:])) if len(support)>1 else 0.}

    def enqueue(self,origin,release,p,label,keys):
        if release<origin:raise ValueError('Feedback precedes prediction')
        self.seq+=1
        heapq.heappush(self.pending,(int(release),self.seq,int(origin),tuple(keys),np.asarray(p).copy(),int(label)))
