"""Reusable formulas extracted from spectrum/cognition/signatures.py; see docs/MIGRATION.json."""
from spectrum.core.schema import NS
TICK_NANO = 100_000_000
from .config import PLANS
MENU_PLANS = tuple(PLANS) + ((80, 40, 1800),)
SESSION_SLACK_SECONDS = 5

def branch(base, tape, side, plan_id, intent, mid_ticks):
    plan = MENU_PLANS[plan_id]
    row = dict(side=side, plan_id=plan_id, take_ticks=plan[0], stop_ticks=plan[1], max_hold_s=plan[2])
    horizon = intent + (plan[2] + SESSION_SLACK_SECONDS) * NS
    if horizon >= tape.deadline_ns:
        return dict(row, status='EXCLUDED_SESSION_TIME')
    r = base.clone()
    if not r.open_at(intent, side, plan_id):
        return dict(row, status='NOT_ELIGIBLE')
    s = r.reference_exit(horizon)
    if s.entry_ns <= 0:
        return dict(row, status='NOT_FILLED')
    out = dict(row, entry_ns=int(s.entry_ns), entry_latency_ms=(s.entry_ns - intent) / 1e6,
               entry_cost_ticks=side * (s.entry_price_nano / TICK_NANO - mid_ticks),
               mfe_usd=float(s.position_mfe), mae_usd=float(s.position_mae), fees_usd=float(s.total_fees))
    if s.censored:
        return dict(out, status='CENSORED', exit_reason=s.exit_reason)
    return dict(out, status='COMPLETE', exit_reason=s.exit_reason, net_usd=float(s.cash - s.initial_capital),
                gross_usd=float(s.cash - s.initial_capital + s.total_fees), hold_s=(s.exit_ns - s.entry_ns) / NS,
                exit_ns=int(s.exit_ns))
