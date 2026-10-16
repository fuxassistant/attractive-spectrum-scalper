"""Default feature definitions. New families need no changes in the stream loop."""
import math
from .feature_registry import Field, Family
from .price_history import SESSIONS, OVERLAPS, ratio, NAN
from .schema import NS

WINDOWS = (6, 30, 180)
OHLC_NAMES = (
    'bar_valid', 'bar_coverage', 'scale_ticks_log', 'body_scale', 'range_scale',
    'upper_wick_scale', 'lower_wick_scale',
    *(f'return_{n}m_scale' for n in (1, 3, 6, 9, 30, 60)),
    *(f'bb_m{n}_{field}' for n in (1, 3, 6, 9) for field in (
        'position', 'width_scale', 'middle_distance_scale', 'sma60_distance_scale', 'middle_slope_scale_per_min')),
    *(name+'_distance_scale' for name in (
        'day_high', 'day_low', 'day_open', 'previous_day_high', 'previous_day_low',
        'week_high', 'week_low', 'previous_week_high', 'previous_week_low')),
    'day_range_scale', 'day_position', 'day_observed_minutes_log', 'day_coverage_minutes',
    'previous_day_age_days', 'previous_day_coverage_minutes', 'week_position',
    'week_coverage_minutes', 'previous_week_coverage_minutes', 'utc_time_sin', 'utc_time_cos', 'utc_weekday',
    *(name+'_'+field for name, _, _, _ in SESSIONS for field in ('active', 'progress', 'range_scale', 'position')),
    *(a+'_'+b+'_overlap' for a, b in OVERLAPS),
)


def log(value):
    return math.log1p(value) if value >= 0 else NAN


def quotes(ctx):
    get, ready = ctx.current, ctx.snapshot_valid
    bid, ask, mid = get('bid'), get('ask'), get('mid')
    bs, ass = get('bid_size'), get('ask_size')
    result = dict(snapshot_valid=float(ready), quote_age_seconds=ctx.quote_age_seconds,
                  spread_ticks=(ask-bid)/ctx.tick_size if ready and ask > bid else NAN,
                  l1_imbalance=ratio(bs-ass, bs+ass) if ready else NAN,
                  microprice_minus_mid_ticks=ratio((ask-mid)*bs+(bid-mid)*ass, bs+ass)/ctx.tick_size if ready else NAN)
    for seconds in ctx.windows:
        w = ctx.window(seconds)
        tag = f'w{seconds}s_'
        result.update({tag+'supported': float(w.supported), tag+'progress_ticks': w.progress,
                       tag+'path_ticks_log': log(w.delta('path_ticks')),
                       tag+'path_efficiency': ratio(abs(w.progress), w.delta('path_ticks')),
                       tag+'up_fraction': ratio(w.delta('up_ticks'), w.delta('path_ticks')),
                       tag+'event_count_log': log(w.delta('event_count'))})
    return result


def trade(ctx):
    result = {}
    for seconds in ctx.windows:
        w = ctx.window(seconds)
        buy, sell, unknown = (w.delta('trade_'+side) for side in ('buy', 'sell', 'unknown'))
        volume, path = buy+sell+unknown, w.delta('path_ticks')
        tag = f'w{seconds}s_'
        result.update({tag+'volume_log': log(volume), tag+'trades_per_second': w.delta('trade_count')/seconds,
                       tag+'classified_fraction': ratio(buy+sell, volume), tag+'trade_imbalance': ratio(buy-sell, buy+sell),
                       tag+'ticks_per_100_contracts': ratio(100*w.progress, volume),
                       tag+'path_per_100_contracts': ratio(100*path, volume),
                       tag+'contracts_per_traversed_tick': ratio(volume, path)})
    return result


def depth(ctx):
    result = {}
    for seconds in ctx.windows:
        w = ctx.window(seconds)
        for side in ('bid', 'ask'):
            added, removed = w.delta('add_'+side), w.delta('remove_'+side)
            near = ctx.current('depth_'+side+'_12')
            tag = f'w{seconds}s_{side}_'
            result.update({tag+'add_log': log(added), tag+'remove_log': log(removed),
                           tag+'net_add_over_depth': ratio(added-removed, near),
                           tag+'churn_over_depth': ratio(added+removed, near),
                           tag+'fill_matched_remove_fraction': ratio(w.delta('matched_fill_remove_'+side), removed),
                           tag+'post_trade_add_log': log(w.delta('post_trade_add_'+side))})
    for ticks in (1, 4, 12):
        bid, ask = ctx.current(f'depth_bid_{ticks}'), ctx.current(f'depth_ask_{ticks}')
        result[f'depth_{ticks}_total_log'] = log(bid+ask) if ctx.snapshot_valid else NAN
        result[f'depth_{ticks}_imbalance'] = ratio(bid-ask, bid+ask) if ctx.snapshot_valid else NAN
    for quantity in (1, 5, 10):
        for side in ('buy', 'sell'):
            result[f'sweep_{side}_{quantity}_excess_ticks'] = ctx.current(f'sweep_{side}_{quantity}')/ctx.tick_size if ctx.snapshot_valid else NAN
    return result


def default_families(windows=WINDOWS):
    quote_names = ('snapshot_valid', 'quote_age_seconds', 'spread_ticks', 'l1_imbalance', 'microprice_minus_mid_ticks') + tuple(
        f'w{n}s_{field}' for n in windows for field in (
            'supported', 'progress_ticks', 'path_ticks_log', 'path_efficiency', 'up_fraction', 'event_count_log'))
    trade_names = tuple(f'w{n}s_{field}' for n in windows for field in (
        'volume_log', 'trades_per_second', 'classified_fraction', 'trade_imbalance',
        'ticks_per_100_contracts', 'path_per_100_contracts', 'contracts_per_traversed_tick'))
    depth_names = tuple(f'w{n}s_{side}_{field}' for n in windows for side in ('bid', 'ask') for field in (
        'add_log', 'remove_log', 'net_add_over_depth', 'churn_over_depth', 'fill_matched_remove_fraction', 'post_trade_add_log')) + tuple(
        f'depth_{n}_{field}' for n in (1, 4, 12) for field in ('total_log', 'imbalance')) + tuple(
        f'sweep_{side}_{quantity}_excess_ticks' for quantity in (1, 5, 10) for side in ('buy', 'sell'))
    def fields(names):
        return tuple(Field(name, _unit(name)) for name in names)
    return (Family('ohlc', '2', fields(OHLC_NAMES), lambda context: context.price),
            Family('quotes', '2', fields(quote_names), quotes),
            Family('trade', '2', fields(trade_names), trade),
            Family('depth', '2', fields(depth_names), depth))


def _unit(name):
    if name.endswith('seconds'):
        return 'seconds'
    if name.endswith('minutes'):
        return 'minutes'
    if name.endswith('days'):
        return 'days'
    if name.endswith('ticks'):
        return 'ticks'
    if name.endswith('per_100_contracts'):
        return 'ticks/100 contracts'
    if name.endswith('per_traversed_tick'):
        return 'contracts/tick'
    if name.endswith('trades_per_second'):
        return 'trades/second'
    return 'dimensionless'
