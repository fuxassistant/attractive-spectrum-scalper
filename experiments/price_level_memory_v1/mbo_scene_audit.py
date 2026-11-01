"""Localize MBO evidence to the previous best bid that actually disappeared."""
import numpy as np
import pandas as pd

TICK_NANO = 100_000_000


def audit_scene(events, mutations, trades, quotes):
    """Return (bid_stepdowns, summary) from complete visible scene publications.

    quotes must contain the immediately preceding full-book publication and all
    scene publications in original order. Only canonical BBO is absolute stock;
    no unchanging deep orders are inferred from selected-order reconstruction.
    Gross downward microsteps are explicitly separated from net displacement.
    """
    if events.empty:
        raise ValueError("Nonempty complete scene required")
    for col in ("source_ordinal", "available_ns", "bid_nano", "ask_nano", "bid_size", "ask_size"):
        if quotes[col].dtype.kind not in "iu":
            raise ValueError("Canonical quotes require integer prices, sizes, ordinals, and clocks")
    if np.any(np.diff(quotes.source_ordinal.to_numpy(np.int64)) <= 0):
        raise ValueError("Canonical publication ordinals must be strictly increasing")
    e = events.sort_values("last_source_ordinal").reset_index(drop=True)
    ordinals = quotes.source_ordinal.to_numpy(np.int64)
    pos = np.searchsorted(ordinals, e.last_source_ordinal.to_numpy(np.int64))
    if (pos >= len(quotes)).any() or not np.array_equal(ordinals[pos], e.last_source_ordinal.to_numpy(np.int64)):
        raise ValueError("Scene publications absent from canonical quote tape")
    if pos[0] == 0 or np.any(np.diff(pos) != 1):
        raise ValueError("Need preceding BBO and consecutive complete scene publications")
    after = quotes.iloc[pos].reset_index(drop=True)
    before = quotes.iloc[pos - 1].reset_index(drop=True)
    if not np.array_equal(after.available_ns.to_numpy(), e.available_ns.to_numpy()):
        raise ValueError("Scene availability differs from canonical quote tape")
    for name, source in (("before_bid_nano", before.bid_nano), ("before_ask_nano", before.ask_nano),
                         ("after_bid_nano", after.bid_nano), ("after_ask_nano", after.ask_nano)):
        if not np.array_equal(e[name].to_numpy(), source.to_numpy()):
            raise ValueError("Scene BBO differs from canonical quote tape")
    if not (e.stock_supported & e.price_supported).all():
        raise ValueError("Stock/price support required for disappearance attribution")
    m_groups = dict(tuple(mutations.groupby("event_uid")))
    t_groups = dict(tuple(trades.groupby("event_uid")))
    rows = []
    for i, ev in enumerate(e.itertuples()):
        if ev.after_bid_nano >= ev.before_bid_nano:
            continue
        local = m_groups.get(ev.event_uid, mutations.iloc[:0])
        local = local.loc[(local.side == "B") & (local.price_nano == ev.before_bid_nano)]
        if not local.attribution_known.all():
            raise ValueError("Unsupported mutation cannot explain best-bid disappearance")
        removed = local.loc[local.quantity < 0]
        t = t_groups.get(ev.event_uid, trades.iloc[:0])
        t = t.loc[t.price_nano == ev.before_bid_nano]
        row = dict(window_id=ev.window_id, event_uid=ev.event_uid,
            available_ns=int(ev.available_ns), before_bid_nano=int(ev.before_bid_nano),
            after_bid_nano=int(ev.after_bid_nano),
            step_ticks=(int(ev.before_bid_nano)-int(ev.after_bid_nano))/TICK_NANO,
            previous_bid_qty=int(before.iloc[i].bid_size),
            add_qty=int(local.added_qty.sum()), remove_qty=int(removed.removed_qty.sum()),
            passive_fill_qty=int(removed.passive_fill.sum()),
            aggressor_fill_qty=int(removed.aggressor_fill.sum()),
            unknown_fill_qty=int(removed.linked_unknown_role_fill.sum()),
            ambiguous_qty=int(removed.ambiguous_remove.sum()),
            unlinked_cancel_qty=int(removed.unlinked_cancel_qty.sum()),
            unlinked_modify_qty=int(removed.unlinked_modify_qty.sum()),
            unlinked_reprice_qty=int(removed.unlinked_reprice_out_qty.sum()),
            sell_trade_qty_at_old_bid=int(t.loc[t.side == "A", "quantity"].sum()),
            buy_trade_qty_at_old_bid=int(t.loc[t.side == "B", "quantity"].sum()),
            unknown_trade_qty_at_old_bid=int(t.loc[~t.side.isin(["A", "B"]), "quantity"].sum()))
        row["stock_residual"] = row["previous_bid_qty"] + row["add_qty"] - row["remove_qty"]
        linked = row["passive_fill_qty"] + row["aggressor_fill_qty"] + row["unknown_fill_qty"]
        unlinked = row["unlinked_cancel_qty"] + row["unlinked_modify_qty"] + row["unlinked_reprice_qty"]
        if row["stock_residual"] or linked + unlinked + row["ambiguous_qty"] != row["remove_qty"]:
            raise ValueError("Disappearing best bid does not reconcile against full BBO stock")
        row["evidence_class"] = ("ambiguous" if row["ambiguous_qty"] else "mixed" if linked and unlinked
                                  else "linked_fill_only" if linked else "unlinked_only" if unlinked else "unknown")
        rows.append(row)
    stepdowns = pd.DataFrame(rows)
    initial, final = before.iloc[0], after.iloc[-1]
    path = quotes.iloc[pos[0]-1:pos[-1]+1]
    mids = (path.bid_nano.to_numpy(np.int64) + path.ask_nano.to_numpy(np.int64)) // 2
    summary = dict(publications=len(e), preceding_available_ns=int(initial.available_ns),
        first_available_ns=int(e.available_ns.iloc[0]), last_available_ns=int(e.available_ns.iloc[-1]),
        first_available_utc=pd.Timestamp(int(e.available_ns.iloc[0]),tz="UTC").isoformat(),
        last_available_utc=pd.Timestamp(int(e.available_ns.iloc[-1]),tz="UTC").isoformat(),
        before_bid_per_oz=int(initial.bid_nano)/1e9, before_ask_per_oz=int(initial.ask_nano)/1e9,
        after_bid_per_oz=int(final.bid_nano)/1e9, after_ask_per_oz=int(final.ask_nano)/1e9,
        before_mid_per_oz=int(mids[0])/1e9, after_mid_per_oz=int(mids[-1])/1e9,
        net_bid_ticks=(int(final.bid_nano)-int(initial.bid_nano))/TICK_NANO,
        net_ask_ticks=(int(final.ask_nano)-int(initial.ask_nano))/TICK_NANO,
        net_mid_ticks=(int(mids[-1])-int(mids[0]))/TICK_NANO,
        net_mid_per_oz=(int(mids[-1])-int(mids[0]))/1e9,
        path_min_mid_per_oz=int(mids.min())/1e9, path_max_mid_per_oz=int(mids.max())/1e9,
        before_spread_ticks=(int(initial.ask_nano)-int(initial.bid_nano))/TICK_NANO,
        after_spread_ticks=(int(final.ask_nano)-int(final.bid_nano))/TICK_NANO,
        bid_stepdown_publications=len(rows), sum_downward_bid_step_ticks=float(stepdowns.step_ticks.sum()) if rows else 0.,
        stock_reconciliation_passed=True, evidence_classes=stepdowns.evidence_class.value_counts().to_dict() if rows else {},
        trade_qty=int(trades.quantity.sum()), sell_trade_qty=int(trades.loc[trades.side == "A","quantity"].sum()),
        buy_trade_qty=int(trades.loc[trades.side == "B","quantity"].sum()),
        unknown_trade_qty=int(trades.loc[~trades.side.isin(["A","B"]),"quantity"].sum()),
        absolute_depth_scope="Canonical BBO only; no absolute deep-level reconstruction")
    for col in ("previous_bid_qty", "add_qty", "remove_qty", "passive_fill_qty", "aggressor_fill_qty", "unknown_fill_qty",
                "ambiguous_qty", "unlinked_cancel_qty", "unlinked_modify_qty", "unlinked_reprice_qty"):
        summary[col] = int(stepdowns[col].sum()) if rows else 0
    # Compact chronological evidence anchors, all already known at the scene's
    # final availability. A within-scene extreme is descriptive, not a forecast.
    indexes = {0, len(e)-1, int(np.argmin(mids[1:]))}
    if rows:
        first_step = str(stepdowns.event_uid.iloc[0])
        indexes.add(int(np.flatnonzero(e.event_uid.to_numpy() == first_step)[0]))
        passive = stepdowns.loc[stepdowns.passive_fill_qty > 0]
        if len(passive):
            indexes.add(int(np.flatnonzero(e.event_uid.to_numpy() == passive.event_uid.iloc[0])[0]))
    step_by_uid = {r["event_uid"]: r for r in rows}
    sequence = []
    for i in sorted(indexes):
        ev = e.iloc[i]
        t = t_groups.get(ev.event_uid, trades.iloc[:0])
        item = dict(available_ns=int(ev.available_ns), utc=pd.Timestamp(int(ev.available_ns),tz="UTC").isoformat(),
            event_uid=ev.event_uid, before_bid_per_oz=int(ev.before_bid_nano)/1e9,
            after_bid_per_oz=int(ev.after_bid_nano)/1e9,
            after_mid_per_oz=(int(ev.after_bid_nano)+int(ev.after_ask_nano))/2e9,
            sell_trade_qty=int(t.loc[t.side=="A","quantity"].sum()),
            buy_trade_qty=int(t.loc[t.side=="B","quantity"].sum()),
            unknown_trade_qty=int(t.loc[~t.side.isin(["A","B"]),"quantity"].sum()))
        if ev.event_uid in step_by_uid:
            item["old_best_bid_evidence"] = step_by_uid[ev.event_uid]
        sequence.append(item)
    summary["visible_sequence_anchors"] = sequence
    unlinked_events = summary["evidence_classes"].get("unlinked_only",0)
    summary["interpretation_ru"] = [
        f"За наблюдаемый интервал mid изменился на {summary['net_mid_per_oz']:+.2f} USD/oz ({summary['net_mid_ticks']:+g} тиков).",
        f"{len(rows)} понижений лучшего bid — микроизменения внутри траектории; их сумма не равна чистому движению рынка.",
        f"В {unlinked_events} из {len(rows)} исчезновений прежнего лучшего bid уменьшения не имеют связанной F. Это описание источника, а не доказательство намерений или отсутствия исполнения.",
        "Объём T учитывается один раз. F связывает исполнения с уменьшениями заявок и не добавляется к объёму торгов.",
    ]
    return stepdowns, summary
