"""Canonical inputs: units, clocks, venue hours, trading regimes and cost defaults.

Every timestamp is int64 UTC nanoseconds. Schedules are declared as local
wall-clock times in IANA zones and converted per timestamp, so daylight-saving
shifts live in the timezone database rather than in constants. Each declared
value carries its unit, epistemic status and source (resources/canon_v1.json).
"""
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from importlib.resources import files
from zoneinfo import ZoneInfo
import hashlib
import json
import numpy as np
import pandas as pd
from spectrum.resources import contract

STATUSES = ('EVIDENCED', 'CONVENTION', 'ASSUMED', 'DEFAULT')
WEEKDAYS = ('MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN')
DAY_MINUTES = 24 * 60


@dataclass(frozen=True)
class Constant:
    name: str
    value: object
    unit: str
    status: str
    source: str


def _minute(text):
    hour, minute = map(int, text.split(':'))
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError('Invalid wall-clock time ' + text)
    return 60 * hour + minute


def _weekdays(names):
    if not names or len(set(names)) != len(names) or not set(names) <= set(WEEKDAYS):
        raise ValueError('Unique weekday names MON..SUN required')
    return tuple(WEEKDAYS.index(name) for name in names)


def _status(status):
    if status not in STATUSES:
        raise ValueError('Unknown canon status ' + str(status))
    return status


def _local(t_ns, zone):
    """Local weekday (MON=0), minute of day and local date for UTC ns."""
    stamps = pd.DatetimeIndex(pd.to_datetime(np.asarray(t_ns, np.int64), utc=True)).tz_convert(_zone(zone))
    return (stamps.dayofweek.to_numpy(), (stamps.hour * 60 + stamps.minute).to_numpy(),
            stamps.tz_localize(None).normalize())


def _at(date, minute, zone):
    """UTC ns of a local wall-clock minute; schedules never sit in DST-ambiguous hours."""
    local = pd.Timestamp(date) + pd.Timedelta(minutes=minute)
    return int(local.tz_localize(_zone(zone), ambiguous='raise', nonexistent='raise').value)


@lru_cache(maxsize=32)
def _zone(name):
    """Use the packaged, versioned tzdata on every platform, not host-specific rules."""
    with files('tzdata.zoneinfo').joinpath(*name.split('/')).open('rb') as stream:
        return ZoneInfo.from_file(stream, key=name)


def _local_dates(start_ns, end_ns, zone, pad_days):
    _, _, dates = _local([start_ns, end_ns], zone)
    return pd.date_range(dates[0] - pd.Timedelta(days=pad_days), dates[1] + pd.Timedelta(days=pad_days), freq='D')


def _clip(rows, start_ns, end_ns):
    return [(a, b) for a, b in rows if b > start_ns and a < end_ns]


@dataclass(frozen=True)
class LocalWindow:
    """Same-day local window [open, close) on listed local weekdays."""
    name: str
    zone: str
    open_minute: int
    close_minute: int
    weekdays: tuple[int, ...]
    status: str
    source: str

    def __post_init__(self):
        if not 0 <= self.open_minute < self.close_minute <= DAY_MINUTES:
            raise ValueError('Session must open and close on the same local day: ' + self.name)

    def contains(self, t_ns):
        day, minute, _ = _local(t_ns, self.zone)
        return np.isin(day, self.weekdays) & (minute >= self.open_minute) & (minute < self.close_minute)

    def intervals(self, start_ns, end_ns):
        rows = [(_at(d, self.open_minute, self.zone), _at(d, self.close_minute, self.zone))
                for d in _local_dates(start_ns, end_ns, self.zone, 1) if d.dayofweek in self.weekdays]
        return _clip(rows, start_ns, end_ns)


@dataclass(frozen=True)
class TradingDayHours:
    """Trading day D runs from D-1 at open to D at close, local time.

    The gap [close, open) is the daily break. A trading day that is not listed,
    or is a declared holiday, is closed in full. Holiday coverage is reported
    separately: an empty list is not a claim that no holiday exists.
    """
    name: str
    zone: str
    open_minute: int
    close_minute: int
    trading_days: tuple[int, ...]
    holidays: tuple[str, ...]
    holiday_coverage: str
    status: str
    source: str

    def __post_init__(self):
        if not 0 <= self.close_minute < self.open_minute < DAY_MINUTES:
            raise ValueError('Venue hours must span local midnight: ' + self.name)

    def trading_date(self, t_ns):
        _, minute, dates = _local(t_ns, self.zone)
        return dates + pd.to_timedelta((minute >= self.open_minute).astype(int), unit='D')

    def contains(self, t_ns):
        _, minute, _ = _local(t_ns, self.zone)
        day = self.trading_date(t_ns)
        in_hours = (minute >= self.open_minute) | (minute < self.close_minute)
        holiday = np.isin(np.asarray(day.strftime('%Y-%m-%d')), np.asarray(self.holidays, dtype=str))
        return in_hours & np.isin(day.dayofweek, self.trading_days) & ~holiday

    def intervals(self, start_ns, end_ns):
        rows = [(_at(d - pd.Timedelta(days=1), self.open_minute, self.zone), _at(d, self.close_minute, self.zone))
                for d in _local_dates(start_ns, end_ns, self.zone, 2)
                if d.dayofweek in self.trading_days and d.strftime('%Y-%m-%d') not in self.holidays]
        return _clip(rows, start_ns, end_ns)


@dataclass(frozen=True)
class Anchor:
    """A recurring local wall-clock moment, e.g. an auction or a data release."""
    name: str
    zone: str
    minute: int
    weekdays: tuple[int, ...]
    status: str
    source: str

    def times(self, start_ns, end_ns):
        values = [_at(d, self.minute, self.zone)
                  for d in _local_dates(start_ns, end_ns, self.zone, 1) if d.dayofweek in self.weekdays]
        return np.asarray([t for t in values if start_ns <= t < end_ns], np.int64)


@dataclass(frozen=True)
class CrossZoneWindow:
    """From a local time in one zone to a local time in another, on weekdays of the first."""
    name: str
    from_zone: str
    from_minute: int
    to_zone: str
    to_minute: int
    status: str
    source: str

    def contains(self, t_ns):
        day, start, _ = _local(t_ns, self.from_zone)
        _, end, _ = _local(t_ns, self.to_zone)
        return (day < 5) & (start >= self.from_minute) & (end < self.to_minute)


class RegimeClock:
    """Which sessions, overlaps and venues are active at each UTC timestamp."""

    def __init__(self, canon):
        self.canon = canon

    def flags(self, t_ns):
        t = np.asarray(t_ns, np.int64)
        table = {name: window.contains(t) for name, window in self.canon.sessions.items()}
        for name, members in self.canon.overlaps.items():
            table[name] = np.logical_and.reduce([table[member] for member in members])
        for name, venue in self.canon.venues.items():
            table[name + '_open'] = venue.contains(t)
        for name, window in self.canon.research_windows.items():
            table[name + '_window'] = window.contains(t)
        return pd.DataFrame(table, index=pd.Index(t, name='t_ns'))

    def label(self, t_ns):
        """Active sessions joined by '+', in canon order; 'NONE' between sessions."""
        flags = self.flags(t_ns)
        names = list(self.canon.sessions)
        return np.asarray(['+'.join(n.upper() for n in names if row[n]) or 'NONE'
                           for row in flags[names].to_dict('records')])

    def intervals(self, start_ns, end_ns):
        """UTC [open_ns, close_ns) rows for sessions, overlaps and venues inside a range."""
        rows = []
        spans = {name: window.intervals(start_ns, end_ns) for name, window in self.canon.sessions.items()}
        for name, span in spans.items():
            rows += [('session', name, a, b) for a, b in span]
        for name, members in self.canon.overlaps.items():
            common = spans[members[0]]
            for member in members[1:]:
                common = [(max(a, c), min(b, d)) for a, b in common for c, d in spans[member]
                          if max(a, c) < min(b, d)]
            rows += [('overlap', name, a, b) for a, b in common]
        for name, venue in self.canon.venues.items():
            rows += [('venue', name, a, b) for a, b in venue.intervals(start_ns, end_ns)]
        frame = pd.DataFrame(rows, columns=['kind', 'name', 'open_ns', 'close_ns'])
        return frame.sort_values(['open_ns', 'kind', 'name'], ignore_index=True)


@dataclass(frozen=True)
class Canon:
    version: str
    fingerprint: str
    constants: dict
    sessions: dict
    overlaps: dict
    venues: dict
    anchors: dict
    research_windows: dict
    mbo: dict
    document: dict

    def value(self, name):
        return self.constants[name].value

    def notes(self, topic):
        """How existing code measures a topic (closed_hours, quote_age, unknown, units, costs)."""
        return self.document['legacy_conventions'][topic]

    def regimes(self):
        return RegimeClock(self)

    def spread_or_default(self, measured_usd, known):
        """Measured spread where known, else the declared default, plus the defaulted mask."""
        values = np.asarray(measured_usd, float)
        defaulted = ~np.asarray(known, bool)
        return np.where(defaulted, self.value('spread_default_usd_per_oz'), values), defaulted


def _fingerprint(document):
    text = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=4)
def load_canon(name='canon_v1'):
    document = contract(name)
    constants = {key: Constant(key, item['value'], item['unit'], _status(item['status']), item['source'])
                 for key, item in document['constants'].items()}
    sessions = {key: LocalWindow(key, item['zone'], _minute(item['open']), _minute(item['close']),
                                 _weekdays(item['weekdays']), _status(item['status']), item['source'])
                for key, item in document['sessions'].items()}
    overlaps = {key: tuple(members) for key, members in document['overlaps'].items()}
    for key, members in overlaps.items():
        if len(members) < 2 or not set(members) <= set(sessions):
            raise ValueError('Overlap must name at least two declared sessions: ' + key)
    venues = {key: TradingDayHours(key, item['zone'], _minute(item['open']), _minute(item['close']),
                                   _weekdays(item['trading_days']), tuple(item['holidays']),
                                   item['holiday_coverage'], _status(item['status']), item['source'])
              for key, item in document['venues'].items()}
    anchors = {key: Anchor(key, item['zone'], _minute(item['at']), _weekdays(item['weekdays']),
                           _status(item['status']), item['source'])
               for key, item in document['anchors'].items()}
    windows = {key: CrossZoneWindow(key, item['from']['zone'], _minute(item['from']['at']),
                                    item['to']['zone'], _minute(item['to']['at']),
                                    _status(item['status']), item['source'])
               for key, item in document['research_windows'].items()}
    for rule in document['mbo']['rules']:
        _status(rule['status'])
    for notes in document['legacy_conventions'].values():
        for note in notes:
            _status(note['status'])
    frozen = map(MappingProxyType, (constants, sessions, overlaps, venues, anchors, windows))
    return Canon(document['version'], _fingerprint(document), *frozen, document['mbo'], document)
