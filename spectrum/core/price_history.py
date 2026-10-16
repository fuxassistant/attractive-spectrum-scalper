"""Bounded causal OHLC memory: rolling price, native bands and calendar ranges."""
from collections import deque
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import math
import numpy as np
from .schema import NS, MINUTE

DAY = 1440 * MINUTE
SESSIONS = (('sydney', 'Australia/Sydney', 7, 16), ('tokyo', 'Asia/Tokyo', 9, 18),
            ('london', 'Europe/London', 8, 17), ('newyork', 'America/New_York', 8, 17))
OVERLAPS = (('sydney', 'tokyo'), ('tokyo', 'london'), ('london', 'newyork'), ('newyork', 'sydney'))
NAN = float('nan')


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else NAN


class NativeBand:
    def __init__(self, minutes):
        self.minutes = minutes
        self.end = self.epoch = None
        self.count = 0
        self.good = True
        self.closes = deque(maxlen=60)
        self.middles = deque(maxlen=4)
        self.published = None

    def update(self, bar, continuous):
        period, t = self.minutes * MINUTE, bar['t_ns']
        end = ((t - 1) // period + 1) * period
        if self.end != end:
            if self.end is not None and end - self.end != period:
                self.closes.clear()
                self.middles.clear()
            self.end, self.epoch = end, bar['epoch']
            self.count, self.good = 0, True
        self.count += 1
        self.good &= bar['valid'] and bar['epoch'] == self.epoch
        if not continuous:
            self.closes.clear()
            self.middles.clear()
        if t != end:
            return
        if self.good and self.count == self.minutes:
            self.closes.append(bar['c'])
        else:
            self.closes.clear()
            self.middles.clear()
        middle = float(np.mean(list(self.closes)[-20:])) if len(self.closes) >= 20 else NAN
        half = float(2 * np.std(list(self.closes)[-20:], ddof=1)) if len(self.closes) >= 20 else NAN
        sma60 = float(np.mean(self.closes)) if len(self.closes) == 60 else NAN
        self.middles.append(middle)
        slope = (middle - self.middles[0]) / (3 * self.minutes) if len(self.middles) == 4 else NAN
        self.published = (t, self.epoch, middle, half, sma60, slope)

    def values(self, t, epoch, price, scale):
        if self.published is None or self.published[1] != epoch or t - self.published[0] >= self.minutes * MINUTE:
            return [NAN] * 5
        _, _, middle, half, sma60, slope = self.published
        return [ratio(price - middle, half), half / scale, (price - middle) / scale,
                (price - sma60) / scale, slope / scale]

    def state_dict(self):
        return dict(minutes=self.minutes, end=self.end, epoch=self.epoch, count=self.count,
                    good=bool(self.good), closes=list(self.closes), middles=list(self.middles), published=self.published)

    def load_state_dict(self, state):
        for name in ('end', 'epoch', 'count', 'good', 'published'):
            setattr(self, name, state[name])
        self.closes.extend(state['closes'])
        self.middles.extend(state['middles'])


def empty_range(key):
    return dict(key=key, high=NAN, low=NAN, open=NAN, count=0, coverage=0.)


def update_range(state, bar):
    state['coverage'] += bar['coverage']
    if bar['valid']:
        if not state['count']:
            state['high'], state['low'], state['open'] = bar['h'], bar['l'], bar['o']
        else:
            state['high'] = max(state['high'], bar['h'])
            state['low'] = min(state['low'], bar['l'])
        state['count'] += 1


class PriceHistory:
    def __init__(self, tick_size):
        self.tick_size = tick_size
        self.last = None
        self.closes = deque(maxlen=61)
        self.ranges = deque(maxlen=60)
        self.native = {n: NativeBand(n) for n in (1, 3, 6, 9)}
        self.day = self.previous_day = self.week = self.previous_week = None
        self.sessions = {}
        self.zones = {name: ZoneInfo(zone) for name, zone, _, _ in SESSIONS}

    def update(self, bar):
        t, valid, epoch = bar['t_ns'], bar['valid'], bar['epoch']
        continuous = (self.last is not None and valid and self.last['valid'] and
                      t - self.last['t_ns'] == MINUTE and epoch == self.last['epoch'])
        if not continuous:
            self.ranges.clear()
            self.closes.clear()
        p, o, h, l = (bar[k] if valid else NAN for k in ('c', 'o', 'h', 'l'))
        if valid:
            tr = h - l
            if continuous:
                tr = max(tr, abs(h - self.last['c']), abs(l - self.last['c']))
            self.ranges.append(tr)
            self.closes.append(p)
        ready = len(self.ranges) >= 20
        scale = max(float(np.median(self.ranges)) if ready else self.tick_size, self.tick_size)
        result = dict(bar_valid=float(valid), bar_coverage=bar['coverage'],
                      scale_ticks_log=math.log1p(scale / self.tick_size) if ready else NAN,
                      body_scale=(p-o)/scale, range_scale=(h-l)/scale,
                      upper_wick_scale=(h-max(o,p))/scale, lower_wick_scale=(min(o,p)-l)/scale)
        for lag in (1, 3, 6, 9, 30, 60):
            result[f'return_{lag}m_scale'] = (p-self.closes[-lag-1])/scale if len(self.closes) > lag else NAN
        for minutes, native in self.native.items():
            native.update(bar, continuous)
            fields = ('position', 'width_scale', 'middle_distance_scale', 'sma60_distance_scale', 'middle_slope_scale_per_min')
            for name, value in zip(fields, native.values(t, epoch, p, scale)):
                result[f'bb_m{minutes}_{name}'] = value
        self._calendar(bar, p, scale, result)
        self.last = dict(bar)
        return result, scale, ready and valid

    def _calendar(self, bar, price, scale, out):
        opened = datetime.fromtimestamp((bar['t_ns']-MINUTE)/NS, timezone.utc)
        day_key = (bar['t_ns']-MINUTE) // DAY
        week_key = day_key - opened.weekday()
        if self.day is None or self.day['key'] != day_key:
            self.previous_day, self.day = self.day, empty_range(day_key)
        if self.week is None or self.week['key'] != week_key:
            self.previous_week = self.week if self.week is not None and self.week['key'] == week_key-7 else None
            self.week = empty_range(week_key)
        update_range(self.day, bar)
        update_range(self.week, bar)
        previous_day = self.previous_day or empty_range(day_key)
        previous_week = self.previous_week or empty_range(week_key)
        for label, level in (
            ('day_high', self.day['high']), ('day_low', self.day['low']), ('day_open', self.day['open']),
            ('previous_day_high', previous_day['high']), ('previous_day_low', previous_day['low']),
            ('week_high', self.week['high']), ('week_low', self.week['low']),
            ('previous_week_high', previous_week['high']), ('previous_week_low', previous_week['low']),
        ):
            out[label+'_distance_scale'] = (price-level)/scale
        out.update(day_range_scale=(self.day['high']-self.day['low'])/scale if bar['valid'] else NAN,
                   day_position=ratio(price-self.day['low'], self.day['high']-self.day['low']),
                   day_observed_minutes_log=math.log1p(self.day['count']),
                   day_coverage_minutes=self.day['coverage'],
                   previous_day_age_days=day_key-previous_day['key'] if self.previous_day else NAN,
                   previous_day_coverage_minutes=previous_day['coverage'] if self.previous_day else NAN,
                   week_position=ratio(price-self.week['low'], self.week['high']-self.week['low']),
                   week_coverage_minutes=self.week['coverage'],
                   previous_week_coverage_minutes=previous_week['coverage'] if self.previous_week else NAN,
                   utc_time_sin=math.sin((opened.hour*60+opened.minute)*2*math.pi/1440),
                   utc_time_cos=math.cos((opened.hour*60+opened.minute)*2*math.pi/1440),
                   utc_weekday=opened.weekday())
        decision = datetime.fromtimestamp(bar['t_ns']/NS, timezone.utc)
        active = {}
        for name, _, start, end in SESSIONS:
            local = decision.astimezone(self.zones[name])
            minute = local.hour*60 + local.minute
            on = local.weekday() < 5 and start*60 <= minute < end*60
            active[name] = on
            key = local.date().isoformat()
            if name not in self.sessions or self.sessions[name]['key'] != key:
                self.sessions[name] = empty_range(key)
            member = on and minute > start*60
            state = self.sessions[name]
            if member:
                update_range(state, bar)
            out[name+'_active'] = float(on)
            out[name+'_progress'] = (minute-start*60)/((end-start)*60) if on else NAN
            out[name+'_range_scale'] = (state['high']-state['low'])/scale if member and bar['valid'] else NAN
            out[name+'_position'] = ratio(price-state['low'], state['high']-state['low']) if member else NAN
        for first, second in OVERLAPS:
            out[first+'_'+second+'_overlap'] = float(active[first] and active[second])

    def state_dict(self):
        return dict(last=self.last, closes=list(self.closes), ranges=list(self.ranges),
                    native={str(n): band.state_dict() for n, band in self.native.items()},
                    day=self.day, previous_day=self.previous_day, week=self.week,
                    previous_week=self.previous_week, sessions=self.sessions)

    def load_state_dict(self, state):
        for name in ('last', 'day', 'previous_day', 'week', 'previous_week', 'sessions'):
            setattr(self, name, state[name])
        self.closes.extend(state['closes'])
        self.ranges.extend(state['ranges'])
        for n, band in self.native.items():
            band.load_state_dict(state['native'][str(n)])
