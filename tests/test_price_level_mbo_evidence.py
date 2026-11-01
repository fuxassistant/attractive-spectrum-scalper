from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments.price_level_memory_v1.mbo_evidence import (
    NS, event_evidence, load_evidence, summarize_price_evidence,
    _apply_subset_record, _complete_subset_event,
)
from spectrum.core.vendor.l3 import MboRecord, F_LAST, F_SNAPSHOT, F_MAYBE_BAD_BOOK
from spectrum.core.vendor.l3_v2 import L3EventEngineV2
from spectrum.representation.raw import FIELDS


class EventBuilder:
    def __init__(self):
        self.engine = L3EventEngineV2(publisher_id=1, instrument_id=1, channel_id=1)
        self.ordinal = 0
        self.records = []
        self.events = []
        self.feed([("R", "N", 0, 0, 0, F_SNAPSHOT),
                   ("A", "B", 1, 100., 10, F_SNAPSHOT),
                   ("A", "B", 3, 99.9, 5, F_SNAPSHOT),
                   ("A", "A", 2, 100.2, 10, F_SNAPSHOT | F_LAST)])

    def feed(self, rows):
        event = None
        for action, side, oid, price, size, flags in rows:
            self.ordinal += 1
            record = MboRecord(self.ordinal, 1, 1, 1,
                self.ordinal * 100, self.ordinal * 100 + 1, self.ordinal,
                oid, round(price * 1e9), size, flags, action, side)
            self.records.append(record)
            event = self.engine.append(record)
            if event is not None:
                self.events.append(event)
        return event


def test_sell_trade_and_passive_bid_fill_count_t_once_not_f_again():
    b = EventBuilder()
    event = b.feed([("T", "A", 99, 100., 3, 0),
                    ("F", "B", 1, 100., 3, 0),
                    ("C", "B", 1, 100., 3, F_LAST)])
    out = event_evidence(event)
    assert out["event"]["stock_supported"] and out["event"]["price_supported"]
    assert sum(t["quantity"] for t in out["trades"]) == 3
    assert out["trades"][0]["side"] == "A"
    row = out["mutations"][0]
    assert row["action"] == "C" and row["removed_qty"] == 3
    assert row["passive_fill"] == 3 and row["unlinked_remove"] == 0
    assert row["unlinked_cancel_qty"] == 0  # Raw C is not cancellation intent.
    assert out["links"][0]["fill_ordinal"] == event.deltas[1].source_ordinal
    assert out["links"][0]["mutation_ordinal"] == event.deltas[2].source_ordinal
    assert out["links"][0]["quantity"] == 3


def test_bid_reprice_is_separate_from_cancel_and_execution():
    b = EventBuilder()
    out = event_evidence(b.feed([("M", "B", 1, 99.8, 7, F_LAST)]))
    remove, add = out["mutations"]
    assert remove["kind"] == "REPRICE" and add["kind"] == "REPRICE"
    assert remove["reprice_out_qty"] == 10 and add["reprice_in_qty"] == 7
    assert remove["unlinked_reprice_out_qty"] == 10
    assert remove["unlinked_cancel_qty"] == remove["unlinked_modify_qty"] == 0
    assert not out["trades"]
    assert remove["prior_price_nano"] == 100_000_000_000
    assert remove["resulting_price_nano"] == 99_800_000_000


def test_existing_bid_aggressor_is_not_sell_consuming_resting_bid():
    b = EventBuilder()
    event = b.feed([("M", "B", 1, 100.2, 3, 0),
                    ("T", "B", 1, 100.2, 3, 0),
                    ("F", "B", 1, 100.2, 3, 0),
                    ("F", "A", 2, 100.2, 3, 0),
                    ("C", "B", 1, 100.2, 3, 0),
                    ("C", "A", 2, 100.2, 3, F_LAST)])
    out = event_evidence(event)
    assert sum(t["quantity"] for t in out["trades"]) == 3
    bid_old = next(m for m in out["mutations"] if m["side"] == "B" and m["quantity"] < 0 and m["price_nano"] == 100_000_000_000)
    bid_new = next(m for m in out["mutations"] if m["side"] == "B" and m["quantity"] < 0 and m["price_nano"] == 100_200_000_000)
    ask = next(m for m in out["mutations"] if m["side"] == "A")
    assert bid_old["linked_fill"] == 0 and bid_old["unlinked_reprice_out_qty"] == 10
    assert bid_new["aggressor_fill"] == 3 and bid_new["passive_fill"] == 0
    assert ask["passive_fill"] == 3
    assert sum(m["linked_fill"] for m in out["mutations"]) == 6  # Both resting sides, one3-contract trade.


def test_cancel_without_fill_is_explicit_unlinked_proxy():
    b = EventBuilder()
    out = event_evidence(b.feed([("C", "B", 1, 100., 4, F_LAST)]))
    row = out["mutations"][0]
    assert row["unlinked_cancel_qty"] == row["unlinked_remove"] == 4
    assert row["linked_fill"] == 0 and not out["trades"]


def test_unresolved_f_is_retained_without_inventing_book_decrement():
    b = EventBuilder()
    out = event_evidence(b.feed([("T", "A", 99, 100., 3, 0), ("F", "B", 1, 100., 3, F_LAST)]))
    assert out["event"]["unresolved_fill_qty"] == 3
    assert not out["mutations"] and out["fills"][0]["linked"] == 0


def test_unsupported_lifecycle_attribution_is_unknown_not_zero():
    b = EventBuilder()
    out = event_evidence(b.feed([("C", "B", 1, 100., 4, F_LAST | F_MAYBE_BAD_BOOK)]))
    row = out["mutations"][0]
    assert not row["attribution_known"] and row["linked_fill"] is None
    assert row["unlinked_remove"] is None


def test_complete_group_and_delivery_clock_are_required():
    b = EventBuilder()
    event = b.feed([("T", "A", 99, 100., 3, 0), ("F", "B", 1, 100., 3, F_LAST)])
    with pytest.raises(ValueError, match="F_LAST"):
        event_evidence(replace(event, deltas=event.deltas[:-1]))
    out = event_evidence(event, received_at_ns=10000)
    assert out["trades"][0]["available_ns"] == 10000
    with pytest.raises(ValueError, match="Delivery"):
        event_evidence(event, received_at_ns=0)


def sidecars(tmp_path):
    rows = []
    for end, available, side, price in [(NS, NS + 20, 1, 100_100_000_000),
                                         (2 * NS, 2 * NS + 20, -1, 100_300_000_000),
                                         (3 * NS, 3 * NS + 20, 1, 100_600_000_000)]:
        row = dict(t_ns=end, available_ns=available, side=side, price_nano=price,
                   **{name: 0. for name in FIELDS})
        row.update(add=4., remove=6., linked_fill=3., passive_fill=2., aggressor_fill=.5,
                   ambiguous_remove=1., unlinked_remove=2., refill=1.)
        rows.append(row)
    levels = pd.DataFrame(rows)
    trades = pd.DataFrame(dict(t_ns=[NS // 2, NS + 100, 2 * NS + 100],
        available_ns=[NS // 2, NS + 100, 4 * NS], ordinal=[1, 2, 3],
        price_nano=[100_100_000_000, 100_300_000_000, 100_600_000_000], quantity=[3, 4, 5],
        side=[-1, 1, 0], supported=[True, True, False], epoch=[1, 1, 1]))
    levels.to_parquet(tmp_path / "levels_1s.parquet", index=False)
    trades.to_parquet(tmp_path / "trades.parquet", index=False)
    manifest = dict(status="COMPLETE", stats=dict(max_stock_residual=0),
        outputs={n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
                 for n in ("levels_1s.parquet", "trades.parquet")})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_sidecar_loader_respects_physical_and_availability_clocks(tmp_path):
    root = sidecars(tmp_path)
    levels, trades, receipt = load_evidence(root, 0, 3 * NS, asof_ns=2 * NS + 30)
    assert len(levels) == len(trades) == 2
    assert receipt["max_available_ns"] == 2 * NS + 20
    np.testing.assert_allclose(levels.linked_unknown_role_fill, .5)
    assert trades.buy_trade_qty.sum() == 4 and trades.sell_trade_qty.sum() == 3
    # A fractional left boundary cannot include a whole bucket's hidden prefix.
    levels2, _, _ = load_evidence(root, NS // 2, 3 * NS, asof_ns=2 * NS + 30)
    assert list(levels2.t_ns) == [2 * NS]


def test_absolute_price_bins_preserve_distinct_trade_and_fill_channels(tmp_path):
    levels, trades, _ = load_evidence(sidecars(tmp_path), 0, 3 * NS, asof_ns=4 * NS)
    summary = summarize_price_evidence(levels, trades)
    assert list(summary.price_bin_nano) == [100_000_000_000, 100_500_000_000]
    first = summary.iloc[0]
    assert first.bid_passive_fill == 2 and first.ask_passive_fill == 2
    assert first.trade_buy_qty == 4 and first.trade_sell_qty == 3
    assert summary.trade_unsupported_qty.sum() == 5
    assert summary.trade_unknown_qty.sum() == 5
    assert first.source_max_available_ns == 2 * NS + 20
    assert first.price_low == 100. and first.price_high == 100.5


def test_sidecar_hash_failure_and_incomplete_manifest_rejected(tmp_path):
    root = sidecars(tmp_path)
    with (root / "trades.parquet").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_evidence(root, 0, 3 * NS)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["status"] = "RUNNING"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="COMPLETE"):
        load_evidence(root, 0, 3 * NS)


def test_invalid_removal_partition_and_float_time_rejected(tmp_path):
    root = sidecars(tmp_path)
    levels, trades, _ = load_evidence(root, 0, 3 * NS)
    levels.loc[0, "linked_fill"] = 8.
    with pytest.raises(ValueError, match="partition"):
        summarize_price_evidence(levels, trades)
    with pytest.raises(TypeError, match="integer"):
        load_evidence(root, 0., 3 * NS)


def test_empty_window_has_stable_schema(tmp_path):
    levels, trades, _ = load_evidence(sidecars(tmp_path), 5 * NS, 6 * NS)
    summary = summarize_price_evidence(levels, trades)
    assert summary.empty and "bid_passive_fill" in summary and "trade_sell_qty" in summary


def test_selected_order_prefix_matches_full_engine_deltas_and_uses_full_bbo():
    b = EventBuilder()
    # This unchanged order is absent from the selected-order prefix but present
    # in independently reconstructed canonical BBO. It must not disappear there.
    previous = b.feed([("A", "B", 999, 100., 50, F_LAST)])
    before_count = len(b.records)
    expected = b.feed([("T", "A", 99, 100., 3, 0),
                       ("F", "B", 1, 100., 3, 0),
                       ("C", "B", 1, 100., 3, F_LAST)])
    target_ids = {r.order_id for r in b.records[before_count:]}
    subset = L3EventEngineV2(publisher_id=1, instrument_id=1, channel_id=1)
    for r in b.records[:before_count]:
        if r.action == "R" or r.order_id in target_ids and r.action in ("A", "C", "M"):
            _apply_subset_record(subset, r)
    assert subset.bids[100_000_000_000].size == 10
    deltas = [_apply_subset_record(subset, r) for r in b.records[before_count:]]
    assert tuple(deltas) == expected.deltas

    def quote(event):
        return pd.Series(dict(available_ns=event.available_at_ns, source_ordinal=event.last_source_ordinal,
            epoch=event.book.continuity_epoch, bid_nano=event.book.best_bid_nano,
            ask_nano=event.book.best_ask_nano, bid_size=event.book.bid_size_l1,
            ask_size=event.book.ask_size_l1, valid=event.book.valid))

    reconstructed = _complete_subset_event(deltas, quote(previous), quote(expected), "source-hash")
    assert reconstructed.book_before.bid_size_l1 == 60 and reconstructed.book.bid_size_l1 == 57
    exact, selected = event_evidence(expected), event_evidence(reconstructed)
    for kind in ("mutations", "trades", "fills", "links"):
        assert exact[kind] == selected[kind]


def test_selected_order_scene_refuses_clock_or_group_mismatch():
    b = EventBuilder()
    previous = b.events[-1]
    event = b.feed([("C", "B", 1, 100., 3, F_LAST)])
    row = pd.Series(dict(available_ns=event.available_at_ns + 1, source_ordinal=event.last_source_ordinal,
        epoch=1, bid_nano=100_000_000_000, ask_nano=100_200_000_000, bid_size=7, ask_size=10, valid=True))
    with pytest.raises(ValueError, match="clock"):
        _complete_subset_event(event.deltas, row, row, "source-hash")


def stepdown_fixture():
    base = 1_787_558_880_000_000_000
    quotes = pd.DataFrame(dict(source_ordinal=[10,20,30,40],
        available_ns=[base,base+1001,base+2003,base+3007],
        bid_nano=[100_000_000_000,99_900_000_000,100_100_000_000,100_000_000_000],
        ask_nano=[100_200_000_000,100_100_000_000,100_300_000_000,100_200_000_000],
        bid_size=[5,4,6,7],ask_size=[10,10,10,10]))
    events = pd.DataFrame([dict(event_uid=f"e{i}", window_id="test", last_source_ordinal=int(quotes.iloc[i].source_ordinal),
        available_ns=int(quotes.iloc[i].available_ns), stock_supported=True, price_supported=True,
        before_bid_nano=int(quotes.iloc[i-1].bid_nano),before_ask_nano=int(quotes.iloc[i-1].ask_nano),
        after_bid_nano=int(quotes.iloc[i].bid_nano),after_ask_nano=int(quotes.iloc[i].ask_nano)) for i in range(1,4)])
    rows=[]
    for uid,px,qty,passive,unlinked in [("e1",100_000_000_000,2,0,0),("e1",100_000_000_000,-7,2,5),
                                      ("e2",100_100_000_000,6,0,0),("e3",100_100_000_000,-6,0,6)]:
        rows.append(dict(event_uid=uid,side="B",price_nano=px,quantity=qty,added_qty=max(qty,0),removed_qty=max(-qty,0),
            attribution_known=True,passive_fill=passive,aggressor_fill=0,linked_unknown_role_fill=0,ambiguous_remove=0,
            unlinked_cancel_qty=unlinked,unlinked_modify_qty=0,unlinked_reprice_out_qty=0))
    trades=pd.DataFrame([dict(event_uid="e1",side="A",price_nano=100_000_000_000,quantity=2)])
    return events,pd.DataFrame(rows),trades,quotes


def test_stepdown_stock_accounts_for_same_publication_additions_and_net_path():
    from experiments.price_level_memory_v1.mbo_scene_audit import audit_scene
    events,mutations,trades,quotes=stepdown_fixture()
    steps,summary=audit_scene(events,mutations,trades,quotes)
    assert list(steps.stock_residual)==[0,0]
    assert steps.iloc[0].previous_bid_qty==5 and steps.iloc[0].remove_qty==7 and steps.iloc[0].add_qty==2
    assert summary["net_mid_ticks"]==0 and summary["sum_downward_bid_step_ticks"]==2
    assert summary["evidence_classes"]=={"mixed":1,"unlinked_only":1}
    assert summary["first_available_ns"]==1_787_558_880_000_001_001
    assert summary["last_available_ns"]==1_787_558_880_000_003_007
    assert summary["trade_qty"]==2 and summary["passive_fill_qty"]==2


def test_stepdown_audit_fails_missing_stock_and_future_clock_substitution():
    from experiments.price_level_memory_v1.mbo_scene_audit import audit_scene
    events,mutations,trades,quotes=stepdown_fixture()
    wrong=mutations.copy();wrong.loc[1,"removed_qty"]-=1
    with pytest.raises(ValueError,match="reconcile"):
        audit_scene(events,wrong,trades,quotes)
    future=events.copy();future.loc[0,"available_ns"]+=1
    with pytest.raises(ValueError,match="availability"):
        audit_scene(future,mutations,trades,quotes)
