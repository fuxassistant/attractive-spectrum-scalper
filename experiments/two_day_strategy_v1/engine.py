"""Frozen entry experts, two-day selection and one-position accounting.

No future outcome is used to construct a mask. Retrospective scores and forward
deployment are separate objects. The existing execution engine is authoritative.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from experiments.mechanical_lab.portfolio import AccountConfig, account

NS = 1_000_000_000
CATALOG = {
    'toward_all': dict(title='Все входы к зоне', direction=1, meaning='Возвраты к прежним областям оплачивались при общей геометрии.', rule='Каждая допустимая возможность.'),
    'away_all': dict(title='Все входы от зоны', direction=-1, meaning='Продолжение удаления оплачивалось при общей геометрии.', rule='Каждая допустимая возможность, противоположная сторона.'),
    'memory_return': dict(title='Возврат к удерживавшей области', direction=1, meaning='Возвраты после истории удержания были вознаграждены в выбранной популяции.', rule='Флаг памяти удержания из атласа.'),
    'permeable_departure': dict(title='Уход от проходимой области', direction=-1, meaning='Продолжение ухода от ранее проходимых областей было вознаграждено.', rule='Флаг проходимости и ухода из атласа.'),
    'failed_escape': dict(title='Возврат после отмены ухода', direction=1, meaning='Уже начавшаяся обратная реакция и история отмен давали оплачиваемые возвраты.', rule='Флаг отмены ухода из атласа.'),
    'place_migration': dict(title='Продолжение миграции', direction=-1, meaning='Удаление от теряющей посещаемость старой области было вознаграждено.', rule='Флаг миграции из атласа.'),
    'flow_away': dict(title='Поток и продвижение от зоны', direction=-1, meaning='Совпавшие направление сделок и продвижение цены оплачивали дальнейший уход.', rule='Известный поток: imbalance ≤ −0.25; движение за 30с ≤ −0.25R; дисбаланс лучшей глубины ≤ 0.'),
    'flow_return': dict(title='Поток и продвижение к зоне', direction=1, meaning='Совпавшие направление сделок и продвижение цены оплачивали возврат.', rule='Известный поток: imbalance ≥ 0.25; движение за 30с ≥ 0.25R; дисбаланс лучшей глубины ≥ 0.'),
    'stalled_flow_return': dict(title='Поток от зоны без продвижения', direction=1, meaning='Возврат после направленного потока без заметного продвижения был вознаграждён; это ещё не доказательство поглощения.', rule='Известный imbalance ≤ −0.25; |движение 30с| ≤ 0.1R; активность ≥ медианы префикса; лучшая глубина ≥ 0.'),
    'aligned_return': dict(title='Возврат при согласии масштабов', direction=1, meaning='Обратное движение за 30с и смещение средних за 10мин совпадали с оплачиваемым возвратом.', rule='Движение 30с ≥ 0.25R и смещение средних половин 10мин ≥ 0.25R.'),
    'WAIT': dict(title='WAIT — не открываться', direction=0, meaning='Ни одна стратегия не получила достаточной опоры на обоих днях при заданной стоимости.', rule='Нулевая позиция и нулевой денежный результат.'),
}

PROFILE_COLUMNS = {
    'distance_r': 'Расстояние до цели / R', 'risk_per_oz': 'R, USD/oz',
    'spread_per_oz': 'Спред, USD/oz', 'past_net30_r': 'Продвижение к зоне за 30с / R',
    'past_migration_r': 'Смещение к зоне за 10мин / R',
    'past_efficiency30': 'Эффективность пути за 30с',
    'x_mbo_mbo_now_trade_imbalance': 'Дисбаланс сделок к зоне',
    'x_mbo_mbo_now_trade_unknown_fraction': 'Доля объёма с неизвестной стороной',
    'x_mbo_mbo_now_depth_1_toward_imbalance': 'Дисбаланс лучшей глубины к зоне',
    'x_base_mbo_now_trade_total_log': 'log(1 + объём сделок 30с)',
}


def build_masks(f, config, activity_threshold):
    """X-only rules. Threshold is fixed on the earlier initialization prefix."""
    r = config['rules']; known = f.past_known.astype(bool)
    flow = known & f.x_base_mbo_now_known.eq(1)
    flow &= f.x_mbo_mbo_now_trade_unknown_fraction.le(r['max_unknown_trade_fraction'])
    imbalance = f.x_mbo_mbo_now_trade_imbalance
    depth = f.x_mbo_mbo_now_depth_1_toward_imbalance
    net = f.past_net30_r
    masks = {'toward_all': pd.Series(True,index=f.index), 'away_all': pd.Series(True,index=f.index)}
    for key in ('memory_return','permeable_departure','failed_escape','place_migration'):
        masks[key] = f['h_'+key].astype(bool)
    masks['flow_away'] = flow & imbalance.le(-r['flow_imbalance']) & net.le(-r['progress_r']) & depth.le(0)
    masks['flow_return'] = flow & imbalance.ge(r['flow_imbalance']) & net.ge(r['progress_r']) & depth.ge(0)
    masks['stalled_flow_return'] = flow & imbalance.le(-r['flow_imbalance']) & net.abs().le(r['stalled_abs_progress_r']) & depth.ge(0) & f.x_base_mbo_now_trade_total_log.ge(activity_threshold)
    masks['aligned_return'] = known & net.ge(r['progress_r']) & f.past_migration_r.ge(r['aligned_migration_r'])
    masks['WAIT'] = pd.Series(False,index=f.index)
    return pd.DataFrame({k:v.fillna(False).astype(bool) for k,v in masks.items()},index=f.index)


def offered(f, masks, strategy):
    """Preserve source geometry; only the requested execution side changes."""
    x = f.loc[masks[strategy]].drop(columns=['deadline_ns'],errors='ignore').copy()
    x['original_signal_side'] = x.side
    x['side'] = x.side * CATALOG[strategy]['direction']
    x['selected_strategy'] = strategy
    return x


def replay(candidates, outcomes, config, *, direction='signal', seed=0):
    stats, log = account(candidates, outcomes, sizing='fixed', direction=direction,
                         seed=seed, config=AccountConfig(**config['account']))
    if len(log):
        if np.any(log.signal_ns.to_numpy()[1:] <= log.exit_ns.to_numpy()[:-1]):
            raise AssertionError('Overlapping executed positions')
        meta = candidates[['candidate_id','selected_strategy','original_signal_side']]
        log = log.merge(meta,on='candidate_id',validate='one_to_one')
    return stats, log


def pair_record(strategy, days, candidates, outcomes, config):
    record = dict(pair_start=days[0],pair_end=days[1],strategy=strategy,
                  title=CATALOG[strategy]['title'],offered=len(candidates))
    for condition in ('base','stress'):
        s, log = replay(candidates,outcomes[condition],config)
        record[condition+'_net_usd'] = s['net_usd']
        record[condition+'_complete'] = s['capital_curve_complete']
        record[condition+'_drawdown_usd'] = s['max_drawdown_usd']
        record[condition+'_unresolved'] = s['unresolved']
        record[condition+'_trades'] = s['trades']
        record[condition+'_largest_trade_usd'] = max(0.,float(log.pnl_usd.max())) if len(log) else 0.
        record[condition+'_without_best_trade_usd'] = s['net_usd']-record[condition+'_largest_trade_usd']
        record[condition+'_max_feedback_ns'] = int(log.exit_ns.max()) if len(log) else -1
        for i,date in enumerate(days):
            z = log[log.date.eq(date)] if len(log) else log
            record[f'{condition}_day{i+1}_net_usd'] = float(z.pnl_usd.sum()) if len(z) else 0.
            record[f'{condition}_day{i+1}_trades'] = len(z)
    return record


def select_pair(records, config):
    """Pure two-day fit; no following-day columns are accepted or consulted."""
    z = pd.DataFrame(records).copy(); n = config['selection']['min_trades_each_day']
    support = z.base_day1_trades.ge(n) & z.base_day2_trades.ge(n)
    raw = z.loc[z.base_complete & support & z.strategy.ne('WAIT')]
    raw = raw.sort_values(['base_net_usd','strategy'],ascending=[False,True])
    z['eligible'] = z.base_complete & z.stress_complete & support & z.stress_day1_trades.ge(n) & z.stress_day2_trades.ge(n)
    for condition in ('base','stress'):
        for d in (1,2): z['eligible'] &= z[f'{condition}_day{d}_net_usd'].gt(0)
    z['score_usd'] = np.minimum(z.stress_day1_net_usd,z.stress_day2_net_usd)
    rank = z[z.eligible & z.strategy.ne('WAIT')].sort_values(['score_usd','strategy'],ascending=[False,True])
    selected = str(rank.iloc[0].strategy) if len(rank) else 'WAIT'
    raw_leader = str(raw.iloc[0].strategy) if len(raw) else 'WAIT'
    runner = str(rank.iloc[1].strategy) if len(rank)>1 else 'WAIT'
    gap = float(rank.iloc[0].score_usd-(rank.iloc[1].score_usd if len(rank)>1 else 0)) if len(rank) else 0.
    return dict(selected_strategy=selected,raw_leader=raw_leader,runner_up=runner,
                score_gap_usd=gap,eligible_count=len(rank)),z


def profiles(f, selected, *, pair_start, pair_end, period):
    rows=[]
    for field,title in PROFILE_COLUMNS.items():
        allx=f[field].dropna();sx=selected[field].dropna()
        rows.append(dict(pair_start=pair_start,pair_end=pair_end,period=period,field=field,title=title,
                         population_n=len(f),selected_n=len(selected),nonmissing_n=len(sx),
                         all_median=float(allx.median()) if len(allx) else np.nan,
                         selected_median=float(sx.median()) if len(sx) else np.nan))
    return rows


def acceptance_rates(history, mask, strata, prior):
    """Past pair only, preserving WAIT as exactly zero activity."""
    rate=float(mask.mean()) if len(mask) else 0.
    tmp=history[strata].copy();tmp['accepted']=np.asarray(mask,dtype=float)
    groups=tmp.groupby(strata,dropna=False).accepted.agg(['sum','count'])
    mapping={tuple(k):(float(v['sum'])+prior*rate)/(int(v['count'])+prior) for k,v in groups.iterrows()}
    return rate,mapping


def stable_uniform(ids,seed):
    with np.errstate(over='ignore'):
        x=np.asarray(ids,dtype=np.uint64)+np.uint64(seed)*np.uint64(0x9e3779b97f4a7c15)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        x=x^(x>>np.uint64(31))
    return (x>>np.uint64(11)).astype(float)/2**53
