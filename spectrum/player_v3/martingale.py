"""Bounded, non-overlapping martingale population and chronological selection.

This is a transparent sizing baseline. It uses only completed M1 closes;
there is no hidden future-best direction or exchange-margin assumption.
"""
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from spectrum.player.contract import Contract, NS
from spectrum.player_v3.gym import evaluate_policy, risk_contract


@dataclass(frozen=True)
class MartingaleConfig:
    mode: str = 'trend'
    lookback_minutes: int = 5
    threshold_ticks: float = 2.
    plan: int = 0
    base_oz: float = 1.
    multiplier: float = 1.5
    cap_oz: float = 4.

    def __post_init__(self):
        if self.mode not in ('trend', 'fade') or self.plan not in (0, 1, 2):
            raise ValueError('Unknown direction mode or exit plan')
        if (self.lookback_minutes < 1 or self.threshold_ticks < 0 or self.base_oz <= 0
                or self.multiplier < 1 or self.cap_oz < self.base_oz):
            raise ValueError('Invalid martingale configuration')

    def dict(self):
        return asdict(self)


class MartingalePolicy:
    def __init__(self, config=None):
        self.config = config or MartingaleConfig()
        self._signals = {}
        self.reset()

    def reset(self):
        self.next_size = self.config.base_oz
        self.loss_streak = 0
        self.series_id = None

    def signal(self, day):
        key = id(day)
        if key not in self._signals:
            x = np.asarray(day.x)
            t = np.asarray(day.bar_t, dtype=np.int64)
            lookback = self.config.lookback_minutes
            signals = np.zeros(len(x), dtype=np.float64)
            for i in range(lookback, len(x)):
                # The first change needs the preceding candle to be observed,
                # hence lookback + 1 known candles and lookback contiguous gaps.
                if (np.all(x[i - lookback:i + 1, 8] > .5)
                        and np.all(np.diff(t[i - lookback:i + 1]) == 60 * NS)):
                    signals[i] = float(x[i - lookback + 1:i + 1, 3].sum())
            self._signals[key] = signals
        return self._signals[key]

    def __call__(self, env):
        mask = env.mask()
        if not mask[1:].any():
            return 0, self.next_size
        change = self.signal(env.day)[env.i]
        if abs(change) <= self.config.threshold_ticks:
            return 0, self.next_size
        side = 1 if change > 0 else -1
        if self.config.mode == 'fade':
            side *= -1
        action = (1 if side > 0 else 4) + self.config.plan
        return (action, self.next_size) if mask[action] else (0, self.next_size)

    def on_trade(self, trade):
        # A profitable leg can leave the cumulative series below its starting
        # equity. Only the replay engine can establish recovery after costs.
        # Require that explicit state instead of silently using the old
        # "every positive trade resets" interpretation.
        status = trade['series_status']
        if status not in ('active', 'recovered', 'ruined', 'censored'):
            raise ValueError(f'Unknown martingale series status: {status}')
        self.series_id = trade.get('series_id')
        if status == 'recovered':
            self.loss_streak = 0
            self.next_size = self.config.base_oz
        elif trade['net'] < 0:
            self.loss_streak += 1
            self.next_size = min(self.config.cap_oz, self.next_size * self.config.multiplier)
        # An unrecovered positive/zero leg preserves the current size. Session
        # boundaries never invoke reset(); only a separately funded population
        # does so in evaluate_policy().


def parameter_grid():
    """180 distinct policies: 18 entry/exit rules x 10 size schedules.

    multiplier=1 and cap=1 duplicates are canonicalized to a single 1-oz
    control for each entry/exit rule. The grid is fixed before evaluation.
    """
    schedules = [(1., 1.)] + list(product((1.25, 1.5, 2.), (2., 4., 8.)))
    return [MartingaleConfig(mode=mode, threshold_ticks=threshold, plan=plan,
                             multiplier=multiplier, cap_oz=cap)
            for mode, plan, threshold in product(('trend', 'fade'), range(3), (0., 2., 5.))
            for multiplier, cap in schedules]


def _summary(result):
    return {k: v for k, v in result.items() if k not in ('environments', 'trade_rows', 'decisions')}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _identity(days):
    answer = []
    for day in days:
        manifest = Path(day.path) / 'manifest.json' if hasattr(day, 'path') else None
        answer.append(dict(date=day.meta.get('date'), manifest_sha256=(
            hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest and manifest.exists() else None)))
    return answer


CYCLE_COLUMNS = ('series_id', 'start_ns', 'last_exit_ns', 'start_equity',
                 'floor', 'last_flat_equity', 'net', 'trades', 'wins',
                 'min_equity', 'max_loss_fraction', 'status', 'first_date',
                 'last_date', 'holding_span_seconds', 'marked_equity', 'end_ns')


def cycle_rows(trades, series_history=()):
    """Collapse completed legs, retaining unrecovered/censored series as such.

    The last flat equity is an observed value, not a fabricated terminal exit.
    An active status means that the series has not recovered at its last
    recorded close. It must not be counted as a successfully completed cycle.
    """
    grouped = {}
    for trade in trades:
        key = int(trade['series_id'])
        if key not in grouped:
            grouped[key] = dict(
                series_id=key, start_ns=str(trade['series_start_ns']),
                start_equity=float(trade['series_start_equity']),
                floor=float(trade['series_floor']), trades=0, wins=0,
                min_equity=float(trade['series_min_equity']),
                max_loss_fraction=max(0., (trade['series_start_equity'] - trade['series_min_equity']) / trade['series_start_equity']),
                first_date=trade.get('date', ''))
        row = grouped[key]
        if not np.isclose(row['start_equity'], trade['series_start_equity'], rtol=0, atol=1e-8):
            raise ValueError('Series base changed before recovery')
        row.update(last_exit_ns=str(trade['exit_ns']),
                   last_flat_equity=float(trade['series_start_equity'] + trade['series_cumulative_net']),
                   net=float(trade['series_cumulative_net']),
                   status=trade['series_status'], last_date=trade.get('date', ''))
        row['trades'] += 1
        row['wins'] += int(trade['net'] > 0)
        row['min_equity'] = min(row['min_equity'], trade['series_min_equity'])
        row['max_loss_fraction'] = max(row['max_loss_fraction'], max(0.,
            (trade['series_start_equity'] - trade['series_min_equity']) / trade['series_start_equity']))
        row['holding_span_seconds'] = (int(row['last_exit_ns']) - int(row['start_ns'])) / NS
    # Include a series with no closed leg (e.g. missing terminal support), and
    # preserve an open-terminal censor instead of calling its last close active.
    for snapshot in series_history:
        key = int(snapshot['series_id'])
        if key not in grouped:
            grouped[key] = dict(series_id=key, start_ns=str(snapshot['series_start_ns']),
                                start_equity=float(snapshot['series_start_equity']),
                                floor=float(snapshot['series_start_equity']) * .5,
                                last_flat_equity=None, net=None, trades=0, wins=0,
                                first_date='', last_date='', last_exit_ns=None,
                                holding_span_seconds=None)
        row = grouped[key]
        row.update(status=snapshot['series_status'], min_equity=float(snapshot['series_min_equity']),
                   max_loss_fraction=max(0., (snapshot['series_start_equity'] - snapshot['series_min_equity']) / snapshot['series_start_equity']),
                   marked_equity=float(snapshot['liquidation_equity']), end_ns=snapshot.get('end_ns'))
    return list(grouped.values())


def search_population(train_days, validation_days, output, contract=None, configs=None, training_groups=None):
    """Select with validation only; no evaluation-day argument is accepted.

    Highest uncensored validation score; then training score; then earliest
    grid index. An economic deployment choice is WAIT unless the winner beats
    WAIT's score of zero. The experimental winner is retained for diagnosis.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    c = contract or Contract(initial_capital=200.)
    configs = list(configs if configs is not None else parameter_grid())
    training_groups = list(training_groups) if training_groups is not None else [train_days]
    def training_result(policy):
        groups = [evaluate_policy(group, policy, c) for group in training_groups]
        if len(groups) == 1:
            return groups[0]
        complete = all(g['complete'] for g in groups)
        answer = dict(complete=complete, censored=not complete,
                      score=float(np.mean([g['score'] for g in groups])) if complete else None,
                      ruined=any(g['ruined'] for g in groups),
                      trades=sum(g['trades'] for g in groups), wins=sum(g['wins'] for g in groups),
                      cohorts=[_summary(g) for g in groups], cohort_count=len(groups),
                      aggregation='Mean score, equity, net, lifetime drawdown, maximum series loss fraction, and lifespan across separately funded weekly populations.',
                      total_net=sum(g['net'] for g in groups),
                      worst_drawdown=max(g['max_drawdown'] for g in groups),
                      worst_series_loss_fraction=max(g['max_series_loss_fraction'] for g in groups),
                      ruined_populations=sum(g['ruined'] for g in groups))
        for key in ('net', 'final_equity', 'max_drawdown', 'max_series_loss_fraction',
                    'survival_fraction', 'lifetime_eligible_seconds'):
            answer[key] = float(np.mean([g[key] for g in groups]))
        return answer
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), initial_capital=c.initial_capital,
        risk_version=risk_contract()['version'], risk_contract=risk_contract(),
        user_correction='The 50% loss limit applies to a whole martingale series, measured against capital immediately before its first actual fill. It is not half the original deposit and not an individual-trade limit.',
        series_floor_rule='0.5 * series_start_equity; equity strictly below this floor ends the population. Exactly 50% loss remains alive (cash arithmetic tolerance 1e-8).',
        series_start='First actual fill while no series is active; base is cash immediately before the full round-trip fee is reserved.',
        series_end='Only flat equity at or above the fixed series starting equity recovers the series. Partial profitable legs and session/day boundaries do not reset it.',
        fee_roundtrip_per_oz=c.fee_roundtrip,
        latency_ns=str(c.latency_ns), position_rule='One position, immutable exit plan; no averaging into losses.',
        size_rule='First actual entry 1 oz. After each net loss multiply next size up to cap; partial wins or zero-net legs hold size while unrecovered. Reset to base_oz only on complete series recovery. Size and series carry across days.',
        score='log(final_equity/initial_capital) + .25*(survival_fraction-1) - .25*max_series_loss_fraction',
        score_weights={'log_growth': 1., 'survival_shortfall': .25, 'maximum_series_loss_fraction': .25},
        score_risk='Maximum marked loss relative to the starting equity of its own series, including costs. Lifetime peak drawdown is reported separately and does not determine the series risk penalty.',
        survival='Observed legal-origin seconds until the first strictly-more-than-50% series loss; surviving endpoint is right-censored lifetime.',
        utility_floor='Cash is never floored; log utility alone uses max(equity/initial, 1e-8).',
        support_censor='Halt population at first support-loss diagnostic close; no score, no claimed ruin from that ambiguous close.',
        selection='Highest validation score among complete train and validation runs, then training score, then earliest grid index.',
        deployment_rule='WAIT if selected validation score is not strictly greater than zero.',
        known_limits='Fixed 1-oz GC quote proxy, not actual exchange contract/margin. Historical development-exposed periods.',
        train=_identity(train_days), validation=_identity(validation_days),
        training_population_resets=[_identity(group) for group in training_groups],
        grid=[config.dict() for config in configs])
    _write_json(output / 'protocol.json', protocol)
    rows = []
    results = []
    started = time.perf_counter()
    for index, config in enumerate(configs):
        stamp = time.perf_counter()
        policy = MartingalePolicy(config)
        train = training_result(policy)
        validation = evaluate_policy(validation_days, policy, c)
        result = dict(index=index, config=config.dict(), train=_summary(train), validation=_summary(validation))
        results.append(result)
        row = dict(index=index, **config.dict(), elapsed_seconds=time.perf_counter() - stamp)
        for split, value in (('train', train), ('validation', validation)):
            for key in ('score', 'net', 'final_equity', 'max_drawdown', 'max_series_loss_fraction',
                        'trades', 'wins', 'ruined', 'complete',
                        'survival_fraction', 'lifetime_eligible_seconds'):
                row[f'{split}_{key}'] = value[key]
        rows.append(row)
        # Incremental receipt survives interruption without pretending COMPLETE.
        pd.DataFrame(rows).to_csv(output / 'population.csv', index=False)
        print(json.dumps(dict(martingale_index=index, policies=len(configs),
                              validation_score=validation['score'], validation_net=validation['net'],
                              train_ruined=train['ruined'], seconds=row['elapsed_seconds'])), flush=True)
    eligible = [r for r in results if r['train']['complete'] and r['validation']['complete']]
    _write_json(output / 'population.json', results)
    if not eligible:
        _write_json(output / 'blocked.json', dict(reason='No complete train and validation population.'))
        raise RuntimeError('No uncensored martingale candidate')
    winner = max(eligible, key=lambda r: (r['validation']['score'], r['train']['score'], -r['index']))
    selected = dict(index=winner['index'], config=winner['config'], train=winner['train'], validation=winner['validation'],
                    deployment_choice='martingale' if winner['validation']['score'] > 0 else 'WAIT',
                    status='FROZEN_BEFORE_EVALUATION', elapsed_seconds=time.perf_counter() - started,
                    frozen_utc=datetime.now(timezone.utc).isoformat(), grid_size=len(configs),
                    protocol_sha256=hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest(),
                    source_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in ('gym.py', 'martingale.py')})
    _write_json(output / 'frozen_before_evaluation.json', selected)
    return selected


def evaluate_frozen(days, frozen, output, contract=None):
    """Separate post-freeze evaluation, including a matched fixed-size control."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if frozen.get('status') != 'FROZEN_BEFORE_EVALUATION':
        raise ValueError('A train/validation freeze is required')
    for name, expected in frozen.get('source_sha256', {}).items():
        actual = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Frozen replay source changed: {name}')
    c = contract or Contract(initial_capital=200.)
    config = MartingaleConfig(**frozen['config'])
    alternatives = (
        ('selected_martingale', MartingalePolicy(config)),
        ('matched_1oz', MartingalePolicy(replace(config, multiplier=1., cap_oz=config.base_oz))),
        ('wait', lambda env: 0),
    )
    summary = dict(frozen=frozen, evaluation=_identity(days), results={})
    for label, policy in alternatives:
        result = evaluate_policy(days, policy, c, record=True)
        dest = output / label
        dest.mkdir()
        _write_json(dest / 'summary.json', _summary(result))
        pd.DataFrame(result['trade_rows']).to_csv(dest / 'trades.csv', index=False)
        cycles = cycle_rows(result['trade_rows'], result.get('series_rows', ()))
        pd.DataFrame(cycles, columns=CYCLE_COLUMNS).to_csv(dest / 'cycles.csv', index=False)
        pd.DataFrame(result['decisions']).to_parquet(dest / 'decisions.parquet', index=False)
        for env in result['environments']:
            np.save(dest / f"quote_equity_{env.day.meta['date']}.npy", env.trace)
            np.save(dest / f"quote_series_floor_{env.day.meta['date']}.npy", env.trace_series_floor)
            np.save(dest / f"quote_series_id_{env.day.meta['date']}.npy", env.trace_series_id)
        _write_json(dest / 'cycle_summary.json', dict(
            series_with_closed_legs=len(cycles), recovered=sum(x['status'] == 'recovered' for x in cycles),
            ruined=sum(x['status'] == 'ruined' for x in cycles),
            censored=sum(x['status'] == 'censored' for x in cycles),
            active_at_last_close=sum(x['status'] == 'active' for x in cycles),
            observed_bases=sorted(set(x['start_equity'] for x in cycles)),
            note='Rows aggregate completed legs. Active series are not counted as recovered; open terminal positions remain censored in summary.json.'))
        summary['results'][label] = _summary(result)
    _write_json(output / 'summary.json', summary)
    return summary
