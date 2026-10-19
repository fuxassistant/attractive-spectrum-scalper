"""Schema library with assimilation, accommodation, quarantine and a router.

A schema is (context, program, plan) plus expectation statistics and lineage:
  context  - conjunction of discrete causal coordinates (empty for a root)
  program  - side rule over coordinates (with/against trend or flow, range
             revert/break, long, short); the plan is a fixed exit protocol
  stats    - decayed mean/variance of matured net outcomes, prequential errors,
             a Drescher-style marginal attribution table of residuals per
             coordinate value outside the context
  lineage  - parent, trigger statistics, creation time, status history

Learning is prequential and off-policy: every matured sibling outcome of every
origin updates every applicable schema (no-impact replay counterfactuals), but
only after its exit time. A child learns only from origins after its creation,
so its confirmation never reuses the evidence that created it.

The router acts as one 1 oz account: at an idle origin it takes the applicable
ACTIVE schema with the best lower confidence bound if that bound is positive;
an active child supersedes its ancestors. WAIT is the default.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

COORDINATES = ('trend', 'flow', 'range', 'vol', 'event', 'phase', 'book')
PROGRAMS = ('with_trend', 'against_trend', 'with_flow', 'against_flow', 'range_revert', 'range_break', 'long', 'short')
PLANS = (2, 3)
UNKNOWN = ''


@dataclass(frozen=True)
class Params:
    """Explicit, logged balance between stability and plasticity."""
    half_life_days: float = 40.        # forgetting of decayed stats; history is kept
    min_n: float = 40.                 # effective outcomes before a schema may act
    min_days: int = 8
    kappa: float = 2.5                 # LCB = mean - kappa * day-clustered SE; calibrated on synthetic null/planted worlds
    surprise_z: float = 2.             # reporting only; surprises never hide losses
    spawn_n: int = 80                  # marginal attribution cell support
    spawn_days: int = 10
    spawn_t: float = 4.0               # vigilance: day-clustered |t|; Bonferroni ~300 cells x weekly looks
    checkpoint_days: int = 5           # cells are tested only at checkpoints: testing every origin is optional stopping
    max_children: int = 4
    max_depth: int = 2
    max_schemas: int = 96
    quarantine_days: int = 15          # ONE preregistered test after this many fresh days (no peeking)
    confirm_t: float = 2.              # one-sided day-clustered t of residual vs parent, trigger sign
    recent_days: int = 5               # fast refutation: suspend acting while the recent day mean <= 0
    accommodation: bool = True
    learning: bool = True
    freeze_at_block: str = ''          # stop all updates from this block on


def coordinates(frame):
    """Discrete causal coordinates; thresholds declared before any readout."""
    f = frame
    def pick(conds, values, known):
        out = np.select(conds, values, default=values[-1]).astype(object)
        out[~known] = UNKNOWN
        return out
    r = f.price_ret900_sigma.to_numpy()
    trend = pick([r < -.5, r > .5, True], ['down', 'up', 'flat'], f.price_ret900_known.to_numpy(bool))
    lo, hi = f.flow_pressure900_low.to_numpy(), f.flow_pressure900_high.to_numpy()
    flow = pick([lo > .05, hi < -.05, True], ['buy', 'sell', 'neutral'], f.flow_pressure900_known.to_numpy(bool))
    p = f.price_range1800_position.to_numpy()
    rng = pick([p < .25, p > .75, True], ['low', 'high', 'mid'], f.price_range1800_known.to_numpy(bool))
    ratio = f.sigma600_ticks_per_sqrt_s.to_numpy() / np.maximum(f.sigma3600_ticks_per_sqrt_s.to_numpy(), 1e-9)
    vol = pick([ratio > 1.25, ratio < .8, True], ['expanding', 'contracting', 'normal'],
               (f.sigma600_known & f.sigma3600_known).to_numpy(bool))
    active = (f.kind == 'event').to_numpy() | (f.prior_events_300s.to_numpy() > 0)
    event = pick([active, True], ['active', 'quiet'], f.event_bank_known.to_numpy(bool))
    m = f.minutes_since_london.to_numpy()
    phase = pick([m < 300, m < 450, True], ['london', 'overlap', 'late'], np.ones(len(f), bool))
    imb = f.flow_depth12_imbalance.to_numpy()
    book = pick([imb > .1, imb < -.1, True], ['bid_heavy', 'ask_heavy', 'balanced'], f.flow_depth12_known.to_numpy(bool))
    return dict(trend=trend, flow=flow, range=rng, vol=vol, event=event, phase=phase, book=book)


def program_side(program, c):
    """c: dict coordinate -> value for one origin. None when undefined."""
    t, f, r = c['trend'], c['flow'], c['range']
    return {'with_trend': {'up': 1, 'down': -1}.get(t), 'against_trend': {'up': -1, 'down': 1}.get(t),
            'with_flow': {'buy': 1, 'sell': -1}.get(f), 'against_flow': {'buy': -1, 'sell': 1}.get(f),
            'range_revert': {'low': 1, 'high': -1}.get(r), 'range_break': {'high': 1, 'low': -1}.get(r),
            'long': 1, 'short': -1}[program]


class Stat:
    """Decayed weighted moments plus undecayed totals and per-day sums.

    Outcomes inside one day overlap in time (5-minute origins, 10-30 minute
    plans) and mirror programs share paths, so inference uses the day as the
    unit: t_day and se_day come from day means, never from raw outcome counts.
    """
    __slots__ = ('w', 's', 'ss', 'day', 'n', 'total', 'days', 'by_day', 'se_cache', 'recent_cache')

    def __init__(self):
        self.w = self.s = self.ss = 0.; self.day = 0; self.n = 0; self.total = 0.; self.days = set()
        self.by_day = {}; self.se_cache = float('inf'); self.recent_cache = 0.

    def decay(self, day, half_life):
        if self.w and day > self.day and np.isfinite(half_life):
            k = .5 ** ((day - self.day) / half_life); self.w *= k; self.s *= k; self.ss *= k
        self.day = max(self.day, day)

    def add(self, y, day, half_life):
        self.decay(day, half_life); self.w += 1; self.s += y; self.ss += y * y
        self.n += 1; self.total += y; self.days.add(day)
        v = self.by_day.setdefault(day, [0., 0]); v[0] += y; v[1] += 1

    def day_means(self, since=None):
        return np.array([a / n for d, (a, n) in self.by_day.items() if since is None or d >= since])

    def t_day(self):
        m = self.day_means()
        if len(m) < 3 or m.std(ddof=1) <= 0:
            return 0.
        return float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m))))

    def refresh_se(self, day, window, recent):
        """Day-clustered standard error over the window and the mean of the last
        `recent` observed days, cached for the router."""
        m = self.day_means(day - window if np.isfinite(window) else None)
        self.se_cache = float(m.std(ddof=1) / np.sqrt(len(m))) if len(m) >= 3 else float('inf')
        last = sorted(self.by_day)[-recent:]
        self.recent_cache = float(np.mean([self.by_day[d][0] / self.by_day[d][1] for d in last])) if last else 0.

    @property
    def mean(self): return self.s / self.w if self.w else 0.

    @property
    def sd(self):
        if self.w < 2: return float('inf')
        return float(np.sqrt(max(self.ss / self.w - self.mean ** 2, 1e-12) * self.w / (self.w - 1)))


@dataclass
class Schema:
    id: int
    program: str
    plan: int
    context: tuple = ()
    parent: int = -1
    depth: int = 0
    status: str = 'ACTIVE'
    created_ns: int = 0
    trigger: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    stat: Stat = field(default_factory=Stat)
    confirm: Stat = field(default_factory=Stat)   # residual vs parent on fresh data
    cells: dict = field(default_factory=dict)     # (coord, value) -> Stat of residuals
    remainder: Stat = field(default_factory=Stat) # outcomes outside active children's contexts
    active_children: set = field(default_factory=set)
    children: int = 0
    acted: int = 0
    acted_net: float = 0.
    surprises: int = 0
    err: dict = field(default_factory=dict)       # month -> [sum sq err, n, sum sq err of global mean]

    def matches(self, c):
        return all(c[k] == v for k, v in self.context)

    def describe(self):
        ctx = ' & '.join(f'{k}={v}' for k, v in self.context) or '*'
        return f'#{self.id} [{ctx}] -> {self.program}/P{self.plan}'


class Library:
    def __init__(self, params=Params()):
        self.p = params; self.schemas = []; self.events = []
        self.global_stat = {plan: Stat() for plan in PLANS}
        for plan in PLANS:
            for program in PROGRAMS:
                self.schemas.append(Schema(len(self.schemas), program, plan))

    # ---- assimilation -------------------------------------------------
    def learn(self, c, intent_ns, day, month, plan, side, y):
        p = self.p; g = self.global_stat[plan]
        g_mean = g.mean if g.w else 0.
        before = [k.stat.mean for k in self.schemas]  # parents judged before this outcome
        for k in self.schemas:
            if k.status == 'RETIRED' or k.plan != plan or not k.matches(c):
                continue
            if k.parent >= 0 and intent_ns <= k.created_ns:
                continue  # fresh evidence only for children
            if program_side(k.program, c) != side:
                continue
            k.stat.decay(day, p.half_life_days)
            ready = k.stat.w >= p.min_n
            mu, sd = k.stat.mean, k.stat.sd
            if ready:
                e = k.err.setdefault(month, [0., 0, 0.])
                e[0] += (y - mu) ** 2; e[1] += 1; e[2] += (y - g_mean) ** 2
                if abs(y - mu) > p.surprise_z * sd:
                    k.surprises += 1
            if k.status == 'QUARANTINE':
                k.confirm.add(y - before[k.parent], day, float('inf'))
            k.stat.add(y, day, p.half_life_days)
            if k.active_children and not any(self.schemas[j].matches(c) for j in k.active_children):
                k.remainder.add(y, day, p.half_life_days)
            if ready and p.accommodation:
                r = y - mu
                for coord in COORDINATES:
                    if c[coord] == UNKNOWN or any(coord == ck for ck, _ in k.context):
                        continue
                    k.cells.setdefault((coord, c[coord]), Stat()).add(r, day, float('inf'))
        g.add(y, day, float('inf'))

    # ---- accommodation and quarantine ----------------------------------
    def review(self, now_ns, day, checkpoint=False):
        p = self.p
        for k in list(self.schemas):
            if k.status == 'ACTIVE':
                k.stat.refresh_se(day, 2 * p.half_life_days, p.recent_days)
                if k.active_children:
                    k.remainder.refresh_se(day, 2 * p.half_life_days, p.recent_days)
            if k.status == 'QUARANTINE':
                s = k.confirm; d = len(s.days)
                if d >= p.quarantine_days:
                    t = s.t_day(); ok = np.sign(k.trigger['mean_residual']) * t >= p.confirm_t
                    self._status(k, 'ACTIVE' if ok else 'RETIRED', now_ns,
                                 dict(confirm_t=float(t), n=s.n, days=d, reason='confirmed' if ok else 'refuted'))
            if (not checkpoint or not p.accommodation or k.status != 'ACTIVE' or k.depth >= p.max_depth
                    or k.children >= p.max_children):
                continue
            for cell, s in list(k.cells.items()):
                if len(self.schemas) >= p.max_schemas or k.children >= p.max_children:
                    break
                if s.n < p.spawn_n or len(s.days) < p.spawn_days:
                    continue
                t = s.t_day()
                if abs(t) < p.spawn_t:
                    continue
                context = tuple(sorted(k.context + (cell,)))
                if any(o.context == context and o.program == k.program and o.plan == k.plan for o in self.schemas):
                    del k.cells[cell]; continue
                child = Schema(len(self.schemas), k.program, k.plan, context, parent=k.id, depth=k.depth + 1,
                               status='QUARANTINE', created_ns=now_ns,
                               trigger=dict(cell=list(cell), mean_residual=float(s.mean), t=float(t), n=s.n,
                                            days=len(s.days), parent_mean=float(k.stat.mean)))
                child.history.append(dict(ns=str(now_ns), status='QUARANTINE'))
                self.schemas.append(child); k.children += 1; del k.cells[cell]
                self.events.append(dict(kind='spawn', ns=str(now_ns), day=day, schema=child.id, parent=k.id,
                                        describe=child.describe(), **child.trigger))

    def _status(self, k, status, now_ns, info):
        k.status = status; k.history.append(dict(ns=str(now_ns), status=status, **info))
        if status == 'ACTIVE' and k.parent >= 0:
            parent = self.schemas[k.parent]
            parent.active_children.add(k.id)
            parent.remainder = Stat()  # credit re-attributed: the parent restarts on what children do not cover
        self.events.append(dict(kind=status.lower(), ns=str(now_ns), schema=k.id, describe=k.describe(), **info))

    # ---- router ---------------------------------------------------------
    def choose(self, c, legal):
        """legal: plans whose branch exists at this origin (fits the session)."""
        p = self.p; candidates = []
        for k in self.schemas:
            if k.status != 'ACTIVE' or k.plan not in legal or not k.matches(c):
                continue
            side = program_side(k.program, c)
            st = k.remainder if k.active_children else k.stat
            if side is None or st.w < p.min_n or len(st.days) < p.min_days:
                continue
            candidates.append((k, side))
        if not candidates:
            return None
        ids = {k.id for k, _ in candidates}
        superseded = {k.parent for k, _ in candidates if k.parent in ids}
        best = None
        for k, side in candidates:
            if k.id in superseded:
                continue
            st = k.remainder if k.active_children else k.stat
            if st.recent_cache <= 0:
                continue  # suspended by recent evidence; it keeps learning
            lcb = st.mean - p.kappa * st.se_cache
            if lcb > 0 and (best is None or lcb > best[2]):
                best = (k, side, lcb)
        return best


def run(frame, outcomes, params=Params(), shuffle_seed=None):
    """frame: origins (index origin_id) sorted by intent.
    outcomes: (origin_id, plan, side) -> (status, net or None, known_ns).
    Only COMPLETE outcomes are learned, after known_ns. NOT_FILLED acts as flat
    cash; CENSORED blocks the account without an invented fill."""
    lib = Library(params)
    coords = coordinates(frame)
    if shuffle_seed is not None:  # destroy the context-outcome link within each day
        rng = np.random.default_rng(shuffle_seed)
        for name in COORDINATES:
            values = coords[name].copy()
            for idx in frame.groupby('date').indices.values():
                values[idx] = values[rng.permutation(idx)]
            coords[name] = values
    ids = frame.index.to_numpy(); intents = frame.intent_ns.astype('int64').tolist()  # Python ints: JSON-safe lineage
    days = frame.day_index.astype(int).tolist(); months = frame.date.str[:7].to_numpy(); blocks = frame.block.to_numpy()
    queue = []
    for i, oid in enumerate(ids):
        for plan in PLANS:
            for side in (-1, 1):
                status, net, known = outcomes.get((oid, plan, side), (None, None, None))
                if status == 'COMPLETE':
                    queue.append((known, i, plan, side, net))
    queue.sort(key=lambda q: q[0]); qi = 0
    busy_until = -1; trades = []; frozen = False
    for i in range(len(ids)):
        now = intents[i]
        frozen = bool(params.freeze_at_block) and blocks[i] >= params.freeze_at_block
        c = {name: coords[name][i] for name in COORDINATES}
        while qi < len(queue) and queue[qi][0] < now:
            known, j, plan, side, net = queue[qi]; qi += 1
            if params.learning and not frozen:
                cj = {name: coords[name][j] for name in COORDINATES}
                lib.learn(cj, intents[j], days[j], months[j], plan, side, net)
        if params.learning and not frozen and (i == 0 or days[i] != days[i - 1] or i % 25 == 0):
            lib.review(now, days[i], checkpoint=i > 0 and days[i] != days[i - 1] and days[i] % params.checkpoint_days == 0)
        if now < busy_until:
            continue
        legal = {plan for plan in PLANS if outcomes.get((ids[i], plan, 1), ('',))[0] in ('COMPLETE', 'CENSORED', 'NOT_FILLED')}
        choice = lib.choose(c, legal)
        if choice is None:
            continue
        k, side, lcb = choice
        status, net, known = outcomes[(ids[i], k.plan, side)]
        if status == 'NOT_FILLED':
            net, known = 0., now
        k.acted += 1
        trade = dict(origin_id=ids[i], date=frame.date.iat[i], block=blocks[i], intent_ns=int(now), schema=k.id,
                     describe=k.describe(), depth=k.depth, program=k.program, plan=k.plan, side=side, lcb=float(lcb),
                     expected=float(k.stat.mean), status=status, net=net if status != 'CENSORED' else None)
        if status != 'CENSORED':
            k.acted_net += net; busy_until = known
        else:
            busy_until = known  # unknown path: no invented fill, account blocked to the branch horizon
        trades.append(trade)
    return lib, pd.DataFrame(trades)
