"""Sequential one-position account; outcome information is consumed only on exit."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

SIZING_CATALOG = {
    'fixed': 'One GC contract.',
    'martingale_capped': 'After losses 1->2->4; after a loss at4 WAIT until next UTC day; profit resets1.',
    'anti_martingale': 'After wins 1->2->4, stay4; loss resets1. Reset at next UTC day.',
}

class OutcomeLookup(dict):
    """Validated reusable lookup; avoids rebuilding it for every coin seed."""

def prepare_outcomes(paired):
    if paired.duplicated(['origin_index','side']).any():raise ValueError('Duplicate origin/side outcomes')
    return OutcomeLookup({(int(r.origin_index),int(r.side)):r for r in paired.itertuples(index=False)})

@dataclass(frozen=True)
class AccountConfig:
    initial_capital_usd: float = 100000.0
    contract_oz: float = 100.0
    max_planned_risk_fraction: float = .02
    daily_loss_fraction: float = .02
    max_drawdown_fraction: float = .10
    max_contracts: int = 4

    def __post_init__(self):
        if not np.isfinite(list(self.__dict__.values())).all():raise ValueError('Finite account parameters required')
        if self.initial_capital_usd<=0 or self.contract_oz<=0:raise ValueError('Positive capital and contract size required')
        for name in ('max_planned_risk_fraction','daily_loss_fraction','max_drawdown_fraction'):
            if not 0<getattr(self,name)<=1:raise ValueError('Risk fractions must be in (0,1]')
        if isinstance(self.max_contracts,bool) or int(self.max_contracts)!=self.max_contracts or self.max_contracts<1:
            raise ValueError('Positive integer max_contracts required')

def account(candidates, paired, *, sizing='fixed', direction='signal', seed=0,
            config=None):
    """No future filtering. A censored filled trade terminates the entire account.

    paired is indexed by origin_index/side, independent of signal or veto. Each
    split starts fresh; this is explicit and is not a continuous monthly curve.
    """
    cfg = config or AccountConfig()
    if sizing not in SIZING_CATALOG:
        raise ValueError('Unknown sizing')
    if direction not in ('signal', 'reverse', 'coin'):
        raise ValueError('Unknown direction')
    if candidates.empty:
        return summarize([], cfg, {}), pd.DataFrame()
    lookup=paired if isinstance(paired,OutcomeLookup) else prepare_outcomes(paired)
    cash = peak = cfg.initial_capital_usd
    maxdd = 0.0
    busy_until = -1
    qty = 1
    day = None
    day_start = cash
    day_wait = False
    counts = dict(offered=len(candidates), overlap=0, risk_rejected=0, wait_rejected=0,
                  empty=0, ineligible=0, censored=0, unresolved=0, halted=False)
    records = []
    ordered=candidates[['t_ns','candidate_id','origin_index','side','date','regime','time_regime']].sort_values(['t_ns', 'candidate_id'])
    # A stable random bit per (seed,event), independent of veto/overlap and RNG
    # consumption. SplitMix64 mixing; this is a control, not cryptography.
    with np.errstate(over='ignore'):
        bits=ordered.candidate_id.to_numpy(np.uint64)+np.uint64(seed)*np.uint64(0x9e3779b97f4a7c15)
        bits=(bits^(bits>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        bits=(bits^(bits>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        bits=((bits^(bits>>np.uint64(31)))&np.uint64(1)).astype(np.int8)*2-1
    for i, c in enumerate(ordered.itertuples(index=False)):
        if c.date != day:
            day, day_start, qty, day_wait = c.date, cash, 1, False
        if c.t_ns <= busy_until:
            counts['overlap'] += 1
            continue
        if day_wait or day_start-cash >= cfg.daily_loss_fraction*day_start or maxdd >= cfg.max_drawdown_fraction*cfg.initial_capital_usd:
            counts['wait_rejected'] += 1
            continue
        side = int(c.side) if direction == 'signal' else -int(c.side) if direction == 'reverse' else int(bits[i])
        r = lookup[(int(c.origin_index), side)]
        # Risk is frozen at signal and known before execution; no status/outcome
        # affects admission. The same caps apply to all position sizing rules.
        planned = qty*cfg.contract_oz*r.risk_per_oz
        if qty > cfg.max_contracts or planned > max(0,cash)*cfg.max_planned_risk_fraction:
            counts['risk_rejected'] += 1
            continue
        if r.status == 'INELIGIBLE':
            counts['ineligible'] += 1
            continue
        if r.status == 'EMPTY':
            counts['empty'] += 1
            busy_until = c.t_ns+2_000_000_000
            continue
        if r.status != 'COMPLETE':
            counts['censored'] += 1
            if r.entry_ns > 0:
                counts['unresolved'] += 1
                counts['halted'] = True
                break
            day_wait = True
            continue
        scale = qty*cfg.contract_oz
        pnl = r.net_per_oz*scale
        # Exact within-trade drawdown plus drawdown from preceding account peak.
        maxdd = max(maxdd, peak-(cash+r.mae_per_oz*scale), r.max_drawdown_per_oz*scale)
        peak = max(peak, cash+max(0,r.mfe_per_oz)*scale)
        cash += pnl
        peak = max(peak,cash)
        busy_until = int(r.exit_ns)
        records.append(dict(candidate_id=int(c.candidate_id), origin_index=int(c.origin_index),
            date=c.date, regime=c.regime, time_regime=c.time_regime,
            signal_ns=int(c.t_ns), side=side, qty=qty, planned_risk_usd=planned,
            entry_ns=int(r.entry_ns), exit_ns=int(r.exit_ns), reason=r.reason,
            pnl_usd=pnl, net_per_oz=r.net_per_oz, r_multiple=r.net_per_oz/r.risk_per_oz,
            equity_usd=cash, max_drawdown_usd=maxdd,
            holding_seconds=(r.exit_ns-r.entry_ns)/1e9))
        if sizing == 'martingale_capped':
            if pnl < 0 and qty == 4:
                day_wait = True
            qty = min(4,qty*2) if pnl < 0 else 1
        elif sizing == 'anti_martingale':
            qty = min(4,qty*2) if pnl > 0 else 1
    counts.update(max_drawdown_usd=maxdd, terminal_cash_usd=cash,
                  capital_curve_complete=not counts['unresolved'])
    return summarize(records,cfg,counts), pd.DataFrame(records)

def summarize(records,cfg,counts):
    f = pd.DataFrame(records)
    out = dict(trades=len(f), active_days=0, net_usd=0., mean_usd=0., win_rate=0.,
               mean_r=0., profit_factor=0., exposure_seconds=0., contracts=0,
               max_drawdown_usd=0., terminal_cash_usd=cfg.initial_capital_usd,
               capital_curve_complete=True, unresolved=0)
    if len(f):
        gain=f.loc[f.pnl_usd>0,'pnl_usd'].sum(); loss=-f.loc[f.pnl_usd<0,'pnl_usd'].sum()
        daily=f.groupby('date').pnl_usd.sum()
        out.update(active_days=int(f.date.nunique()), net_usd=float(f.pnl_usd.sum()),
            mean_usd=float(f.pnl_usd.mean()), win_rate=float((f.pnl_usd>0).mean()),
            mean_r=float(f.r_multiple.mean()), profit_factor=float(gain/loss) if loss else None,
            exposure_seconds=float(f.holding_seconds.sum()), contracts=int(f.qty.sum()),
            largest_day_share_of_abs_net=float(daily.abs().max()/daily.abs().sum()) if daily.abs().sum() else 0.)
    out.update(counts)
    return out
