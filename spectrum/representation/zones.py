"""Addressable past landmarks and prefix-only interaction tokens.

V1 uses previous London-cycle H/L/VWAP/POC and declared offset controls.
It does not claim participant identity or full persistent-depth reconstruction.
"""
import numpy as np
import pandas as pd
from spectrum.representation.representation import NS
from spectrum.representation.representation import MINUTE
from spectrum.representation.representation import TICK
from spectrum.representation.representation import anchor

MECHANISMS = ('EXECUTED_DEPLETION', 'WITHDRAWAL', 'REFILL_UNDER_PRESSURE', 'AGGRESSION_DECAY')


def zone_and_event_tables(fast, context, levels, date, symbol, *, start_ns=None, end_ns=None):
    current = anchor(date) if start_ns is None else int(start_ns)
    end = current + 17 * 3600 * NS if end_ns is None else int(end_ns)
    mature_rows = context[context.previous_recognized_ns > 0]
    if len(mature_rows) == 0: raise ValueError('No matured previous-cycle levels')
    init = mature_rows.iloc[0]
    recognized = int(init.previous_recognized_ns)
    data = fast[(fast.available_ns >= recognized) & (fast.t_ns <= end)].copy()
    events, states = [], []
    for mech in MECHANISMS:
        active = data[mech].to_numpy(bool)
        starts = active & ~np.r_[False, active[:-1]]
        for r in data[starts].itertuples():
            events.append(dict(event_id=f'{symbol}:{date}:{r.t_ns}:{mech}', zone_id='', kind=mech,
                               t_ns=int(r.t_ns), available_ns=int(r.available_ns), direction=int(np.sign(r.ret30_ticks)),
                               price=float(r.mid), evidence_quantity=float(r.trade_quantity30),
                               source='prefix_30s_descriptor', recognized_ns=int(r.available_ns)))
    levels = levels[(levels.available_ns >= recognized) & (levels.t_ns <= end)]
    for number, kind in enumerate(('high', 'low', 'vwap', 'poc')):
        price = float(init['previous_' + kind])
        for control in (False, True):
            # Chosen before outcomes. Controls are distance-shifted, NOT matched
            # experiments; their exposure/distance difference is reported.
            center = price + ((43 if number % 2 == 0 else -43) * TICK if control else 0)
            zone_id = f'{symbol}:{date}:previous_{kind}' + (':offset_control' if control else '')
            l = levels[(levels.price_nano / 1e9 - center).abs() <= 2 * TICK + 1e-8]
            quantity_cols = ['linked_fill', 'unlinked_remove', 'ambiguous_remove', 'refill', 'add']
            ledger = l.groupby('available_ns')[quantity_cols].sum().sort_index().cumsum()
            if len(ledger):
                aligned = pd.merge_asof(data[['available_ns']], ledger.reset_index(), on='available_ns').fillna(0)
            else:
                aligned = pd.DataFrame(0., index=data.index, columns=quantity_cols)
            dist = (data.mid.to_numpy() - center) / TICK
            valid = data.known.to_numpy()
            near = valid & (np.abs(dist) <= 2)
            contact = near & ~np.r_[False, near[:-1]]
            contacts = contact.cumsum()
            last_side = 0; last_contact = None
            outside_start = None; outside_side = 0; persistence_emitted = False
            for k, r in enumerate(data.itertuples()):
                tags = []
                side = int(np.sign(dist[k])) if valid[k] and abs(dist[k]) > 2 else 0
                if not valid[k]:
                    last_side = 0; last_contact = None
                    outside_start = None; outside_side = 0; persistence_emitted = False
                if contact[k]:
                    tags.append('CONTACT' if contacts[k] == 1 else 'RETURN')
                    last_contact = int(r.t_ns)
                if side and last_side and side != last_side:
                    tags.append('CROSS')
                if side: last_side = side
                if k > 0 and valid[k] and valid[k - 1] and abs(dist[k]) <= 10 < abs(dist[k - 1]): tags.append('APPROACH')
                if not side:
                    outside_start = None; outside_side = 0; persistence_emitted = False
                elif outside_start is None or side != outside_side:
                    outside_start = int(r.t_ns); outside_side = side; persistence_emitted = False
                if last_contact is not None and outside_start is not None and int(r.t_ns) - outside_start >= 30 * NS and not persistence_emitted:
                    tags.append('PERSISTENCE_CONFIRMED'); persistence_emitted = True
                for tag in tags:
                    events.append(dict(event_id=f'{zone_id}:{r.t_ns}:{tag}', zone_id=zone_id, kind=tag,
                                       t_ns=int(r.t_ns), available_ns=int(r.available_ns), direction=side,
                                       price=float(r.mid), evidence_quantity=np.nan, source='past_landmark_contact',
                                       recognized_ns=int(r.available_ns)))
                if k == 0 or tags or r.t_ns % (30 * NS) == 0:
                    totals = aligned.iloc[k]
                    states.append(dict(zone_id=zone_id, kind='previous_' + kind, is_control=control,
                        t_ns=int(r.t_ns), available_ns=int(r.available_ns), recognized_ns=recognized,
                        price=center, lower=center - .2, upper=center + .2, coordinate_version=1,
                        origin_ns=int(init.context_start_ns), age_seconds=(int(r.t_ns) - recognized) / NS,
                        contacts=int(contacts[k]), distance_ticks=float(dist[k]), known=bool(valid[k]),
                        linked_fill=float(totals.linked_fill), withdrawal_proxy=float(totals.unlinked_remove),
                        unresolved_remove=float(totals.ambiguous_remove), refill=float(totals.refill),
                        add=float(totals['add']), remaining_visible_qty=np.nan,
                        last_contact_ns=last_contact or 0, status='CONTACT' if near[k] else 'OUTSIDE' if valid[k] else 'UNKNOWN'))
    z = pd.DataFrame(states).sort_values(['available_ns', 'zone_id']).reset_index(drop=True)
    e = pd.DataFrame(events).sort_values(['available_ns', 'event_id']).reset_index(drop=True)
    return z, e
