"""Interpretable amplitude/time descriptors and explicitly masked book evidence."""
import numpy as np
import pandas as pd
from numba import njit

from .tokens import NS, TICK, quote_arrays

PATH_NAMES = ['net_ticks', 'range_ticks', 'path_ticks', 'efficiency', 'up_ticks',
              'down_ticks', 'direction_changes', 'largest_step_ticks',
              'above_start_fraction', 'below_start_fraction', 'zone_dwell_seconds',
              'coverage', 'shape_q1', 'shape_q2', 'shape_q3', 'shape_q4',
              'rise_from_start_ticks', 'fall_from_start_ticks', 'retreat_from_high_ticks',
              'rebound_from_low_ticks', 'mean_spread_ticks']
FLOW_FIELDS = ['trade_buy', 'trade_sell', 'trade_unknown', 'bid_near_fill',
               'ask_near_fill', 'bid_near_withdrawal', 'ask_near_withdrawal',
               'bid_near_refill', 'ask_near_refill', 'bid_near_add', 'ask_near_add']
FLOW_NAMES = FLOW_FIELDS + ['flow_imbalance_low', 'flow_imbalance_high',
                           'unknown_volume_fraction', 'flow_age_seconds',
                           'flow_coverage', 'book_coverage']
MODEL_NAMES = PATH_NAMES + ['elapsed_seconds', 'net_per_amplitude', 'range_per_amplitude',
    'path_per_amplitude', 'elapsed_per_limit', 'net_ticks_per_second',
    'edge_returns', 'gate_attempts', 'gate_aborts', 'distance_to_zone_ticks',
    'pending_seconds', 'is_running', 'is_friction', 'is_retracing', 'is_quiet'] + FLOW_NAMES + [
    'age_seconds', 'offset_current_ticks', 'is_previous_session', 'is_retrieved']


@njit(cache=True)
def _measure_many(qt, price, spread, good, epoch, starts, ends, si, ei, lows, widths, amplitudes, gap):
    out = np.zeros((len(starts), 21))
    known = np.ones(len(starts), np.bool_)
    for r in range(len(starts)):
        a = starts[r]; b = ends[r]; first = si[r]; last = ei[r]
        anchor = price[first]; p = price[last]; lo = anchor; hi = anchor
        path = up = down = turns = largest = support = above = below = dwell = 0
        weighted_spread = 0.0
        last_sign = 0
        complete = good[first] and good[last]
        for j in range(first, last+1):
            if not good[j] or epoch[j] != epoch[first]:
                complete = False
            if good[j]:
                lo = min(lo, price[j]); hi = max(hi, price[j])
                dt = max(0, min(b, qt[j]+gap, qt[j+1] if j < last else b)-max(a, qt[j]))
                support += dt
                weighted_spread += spread[j] / TICK * (dt/NS)
                if price[j] > anchor: above += dt
                if price[j] < anchor: below += dt
                if widths[r] > 0 and lows[r] <= price[j] < lows[r]+widths[r]: dwell += dt
            if j > first:
                if good[j] and good[j-1] and epoch[j] == epoch[j-1] and qt[j]-qt[j-1] <= gap:
                    change = price[j]-price[j-1]
                    path += abs(change); up += max(0, change); down += max(0, -change)
                    largest = max(largest, abs(change))
                    sign = 1 if change > 0 else -1 if change < 0 else 0
                    if sign:
                        if last_sign and last_sign != sign: turns += 1
                        last_sign = sign
                else:
                    complete = False
        duration = b-a
        known[r] = complete and support == duration
        out[r, 0] = (p-anchor)/TICK
        out[r, 1] = (hi-lo)/TICK
        out[r, 2] = path/TICK
        out[r, 3] = abs(p-anchor)/path if path else 0.
        out[r, 4] = up/TICK; out[r, 5] = down/TICK
        out[r, 6] = turns; out[r, 7] = largest/TICK
        out[r, 8] = above/duration if duration else 0.
        out[r, 9] = below/duration if duration else 0.
        out[r, 10] = dwell/NS
        out[r, 11] = support/duration if duration else (1. if good[last] else 0.)
        for k in range(1, 5):
            clock = a + duration*k//4
            j = min(last, max(first, np.searchsorted(qt, clock, side='right')-1))
            out[r, 11+k] = (price[j]-anchor)/(TICK*amplitudes[r])
        out[r, 16] = (hi-anchor)/TICK; out[r, 17] = (anchor-lo)/TICK
        out[r, 18] = (hi-p)/TICK; out[r, 19] = (p-lo)/TICK
        out[r, 20] = weighted_spread/(support/NS) if support else spread[last]/TICK
    return out, known


def path_measures(tokens, quotes, gap_seconds=30):
    if tokens.empty:
        return pd.DataFrame(columns=PATH_NAMES+['path_known'])
    qt, price, good, epoch = quote_arrays(quotes)
    values, known = _measure_many(qt, price, (quotes.ask_nano-quotes.bid_nano).to_numpy(np.int64),
        good, epoch, tokens.start_ns.to_numpy(np.int64), tokens.available_ns.to_numpy(np.int64),
        tokens.start_index.to_numpy(np.int64), tokens.source_index.to_numpy(np.int64),
        tokens.zone_low_nano.to_numpy(np.int64), (tokens.zone_high_nano-tokens.zone_low_nano).to_numpy(np.int64),
        tokens.amplitude_ticks.to_numpy(float), int(gap_seconds*NS))
    f = pd.DataFrame(values, columns=PATH_NAMES)
    f['path_known'] = known
    return f


def annotate_zone_history(tokens):
    """Only already published updates contribute; final outcomes are not backfilled."""
    f = tokens.copy()
    for kind, name in [('EDGE_RETURN','edge_returns'), ('GATE_START','gate_attempts'), ('GATE_ABORT','gate_aborts')]:
        f[name] = f.kind.eq(kind).groupby(f.story_id, sort=False).cumsum().astype('int64')
    gate = f.available_ns.where(f.kind.eq('GATE_START'), -1)
    reset = f.kind.isin(['GATE_ABORT','DEPARTURE_CONFIRMED','CENSORED','ARRIVAL','PRESENT_ON_RESTART'])
    # Carry a gate only inside the same visit and until its abort/confirmation.
    marker = gate.astype('Int64').where(f.kind.eq('GATE_START') | reset)
    f['pending_start_ns'] = marker.groupby(f.story_id, sort=False).ffill().fillna(-1).astype('Int64')
    f['pending_seconds'] = np.where(f.pending_start_ns >= 0, (f.available_ns-f.pending_start_ns)/NS, 0.)
    return f


class FlowView:
    """Fixed physical windows of *published complete buckets*, not future interpolation.

    These contextual windows are deliberately named separately from the native
    story interval. An incomplete field is masked even when some quantities are
    observed. Unknown trade side still contributes to the admissible imbalance bounds.
    """
    def __init__(self, fast):
        self.fast = fast
        self.av = fast.available_ns.to_numpy(np.int64)
        self.t = fast.t_ns.to_numpy(np.int64)
        if not fast.available_ns.is_monotonic_increasing or not fast.t_ns.is_monotonic_increasing:
            raise ValueError('Ordered flow publication and bucket clocks required')

    def measure(self, token):
        result = {name: np.nan for name in FLOW_NAMES}
        j = int(np.searchsorted(self.av, int(token.available_ns), side='right'))-1
        seconds = int(token.flow_seconds)
        if j < 0:
            return result
        end = self.t[j]; start = end-seconds*NS
        i = int(np.searchsorted(self.t, start, side='right'))
        f = self.fast.iloc[i:j+1]
        coverage = len(f) == seconds and (len(f) == 1 or np.all(np.diff(f.t_ns) == NS))
        result['flow_age_seconds'] = (int(token.available_ns)-int(end))/NS
        fresh = 0 <= result['flow_age_seconds'] <= seconds
        trade_ok = coverage and fresh and bool(f.trade_known.all()) and f.epoch.nunique() == 1
        book_ok = coverage and fresh and bool(f.ledger_known.all()) and f.epoch.nunique() == 1
        result['flow_coverage'] = float(f.trade_known.sum())/seconds
        result['book_coverage'] = float(f.ledger_known.sum())/seconds
        for name in FLOW_FIELDS:
            if (trade_ok if name.startswith('trade_') else book_ok):
                result[name] = float(f[name].sum())
        if trade_ok:
            buy, sell, unknown = (result[n] for n in FLOW_FIELDS[:3])
            total = buy+sell+unknown
            if total > 0:
                result['flow_imbalance_low'] = (buy-sell-unknown)/total
                result['flow_imbalance_high'] = (buy-sell+unknown)/total
                result['unknown_volume_fraction'] = unknown/total
        return result


def enrich(tokens, quotes, fast=None):
    f = tokens.reset_index(drop=True).copy()
    metrics = path_measures(f, quotes)
    for n in metrics:
        f[n] = metrics[n].to_numpy()
    for n in ('edge_returns','gate_attempts','gate_aborts','pending_seconds'):
        if n not in f: f[n] = 0.
        f[n] = f[n].fillna(0.)
    f['elapsed_seconds'] = (f.available_ns-f.start_ns)/NS
    f['net_per_amplitude'] = f.net_ticks/f.amplitude_ticks
    f['range_per_amplitude'] = f.range_ticks/f.amplitude_ticks
    f['path_per_amplitude'] = f.path_ticks/f.amplitude_ticks
    f['elapsed_per_limit'] = f.elapsed_seconds/f.time_limit_seconds
    f['net_ticks_per_second'] = f.net_ticks/f.elapsed_seconds.replace(0, np.nan)
    f['distance_to_zone_ticks'] = np.where(f.family.eq('zone'),
        np.maximum(0, np.maximum(f.zone_low_nano-f.mid_nano, f.mid_nano-f.zone_high_nano))/TICK, np.nan)
    f['is_running'] = (f.net_ticks.abs().ge(f.amplitude_ticks) & f.efficiency.ge(.65)).astype(float)
    f['is_friction'] = (f.path_ticks.ge(2*f.amplitude_ticks) & f.efficiency.le(.25)).astype(float)
    retrace = np.maximum(f.retreat_from_high_ticks.where(f.rise_from_start_ticks >= f.amplitude_ticks, 0),
                        f.rebound_from_low_ticks.where(f.fall_from_start_ticks >= f.amplitude_ticks, 0))
    f['is_retracing'] = retrace.ge(.5*f.amplitude_ticks).astype(float)
    f['is_quiet'] = (f.range_ticks.lt(.5*f.amplitude_ticks) & f.elapsed_seconds.ge(.5*f.time_limit_seconds)).astype(float)
    if fast is not None:
        flow = FlowView(fast)
        evidence = pd.DataFrame([flow.measure(r) for r in f.itertuples()])
        for name in FLOW_NAMES: f[name] = evidence[name].to_numpy()
    else:
        for name in FLOW_NAMES: f[name] = np.nan
    return f


def encode(tokens):
    """Numeric input and independent missingness; metadata carries causal graph addresses."""
    values = tokens.reindex(columns=MODEL_NAMES).to_numpy(float)
    known = np.isfinite(values)
    physical = [MODEL_NAMES.index(n) for n in PATH_NAMES if n != 'coverage']
    physical += [MODEL_NAMES.index(n) for n in ('net_per_amplitude','range_per_amplitude',
        'path_per_amplitude','net_ticks_per_second','is_running','is_friction','is_retracing','is_quiet')]
    known[:, physical] &= tokens.path_known.to_numpy(bool)[:, None]
    vocab = {'family':['motion','zone'], 'scale':['fast','middle','slow'],
        'phase':['OBSERVED_START','FORMING','ABORTED','CONFIRMED','TIME_LIMIT','CENSORED','OBSERVED_JUMP'],
        'kind':['START','DIRECTION_SET','PROGRESS','TURN_CONFIRMED','TIME_LIMIT','CENSORED',
                'ARRIVAL','PRESENT_ON_RESTART','EDGE_EXIT','EDGE_RETURN','GATE_START','GATE_ABORT',
                'DEPARTURE_CONFIRMED','JUMP_OVER','LIVE_STATE']}
    categories = np.column_stack([tokens[n].map({v:i+1 for i,v in enumerate(vocab[n])}).fillna(0).to_numpy(np.int16)
                                  for n in vocab])
    return dict(names=MODEL_NAMES, values=np.where(known, values, 0.).astype(np.float32), known=known,
                categorical_names=list(vocab), categorical_values=categories,
                categorical_known=categories>0, categorical_vocab=vocab)
