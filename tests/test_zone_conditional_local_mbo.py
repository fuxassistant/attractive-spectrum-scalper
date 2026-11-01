"""Sparse local evidence needs coverage, publication and conservation proofs."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments.zone_conditional_v1.local_mbo import local_mbo_diagnostics
from experiments.zone_conditional_v1.mbo_features import GlobalMBOContext
from spectrum.representation.raw import FIELDS
from test_zone_conditional_mbo import BASE, NS, fixture


def write_source(path, levels, trades):
    path.mkdir(exist_ok=True)
    levels.to_parquet(path/"levels_1s.parquet", index=False)
    trades.to_parquet(path/"trades.parquet", index=False)
    manifest = dict(status="COMPLETE", route=[1, 2, 3], source=dict(partition_sha256="source"),
                    stats=dict(max_stock_residual=0), outputs={})
    for name in ("levels_1s.parquet", "trades.parquet"):
        manifest["outputs"][name] = hashlib.sha256((path/name).read_bytes()).hexdigest()
    (path/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def setup_local(tmp_path):
    s, q, c = fixture(); s["c_trade_count"] = np.arange(len(s))*3
    q["available_ns"] += NS//20
    c["zone_low_nano"] = 100_000_000_000; c["zone_high_nano"] = 101_000_000_000
    levels = []; trades = []
    for second in range(1, 101):
        for side, price in ((1, 100_100_000_000), (-1, 100_200_000_000)):
            amounts = dict.fromkeys(FIELDS, 0.)
            if side == 1:
                amounts.update(add=10., remove=4., linked_fill=3., passive_fill=2., unlinked_remove=1., refill=2.)
            else:
                amounts.update(add=5., remove=2., linked_fill=1., passive_fill=1., unlinked_remove=1., refill=1.)
            levels.append(dict(t_ns=BASE+second*NS, available_ns=BASE+second*NS+NS//20,
                               side=side, price_nano=price, **amounts))
        for k, (side, qty) in enumerate(((1, 3), (-1, 1), (0, 2))):
            t = BASE+second*NS-NS//2+k
            trades.append(dict(t_ns=t, available_ns=t, ordinal=second*3+k,
                               price_nano=100_100_000_000, quantity=qty, side=side,
                               supported=True, epoch=0))
    levels = pd.DataFrame(levels); trades = pd.DataFrame(trades)
    raw = tmp_path/"raw"; write_source(raw, levels, trades)
    ctx = GlobalMBOContext(s, q, selection_available_ns=BASE+50*NS, coverage_end_ns=BASE+101*NS)
    return raw, c, ctx, levels, trades


def call(c, ctx, raw):
    return local_mbo_diagnostics(c, ctx, raw, expected_route=[1, 2, 3],
                                 expected_source_partition_sha256="source")


def test_local_flow_reconciles_and_t_is_not_double_counted(tmp_path):
    raw, c, ctx, _, _ = setup_local(tmp_path)
    f, receipt = call(c, ctx, raw); r = f.loc[f.window == "now"].iloc[0]
    assert receipt["known_rows"] == 2 and r.known
    assert r.observed_trade_total_qty == 180
    assert r.observed_trade_toward_qty == 90
    assert r.observed_support_passive_fill_qty == 60
    assert r.observed_support_linked_unknown_role_fill_qty == 30
    assert r.observed_support_unlinked_remove_qty == 30
    assert (f.max_source_available_ns <= f.query_ns).all()


def test_verified_empty_zone_is_zero_activity_not_missing(tmp_path):
    raw, c, ctx, _, _ = setup_local(tmp_path)
    c["zone_low_nano"] = 900_000_000_000; c["zone_high_nano"] = 901_000_000_000
    f, _ = call(c, ctx, raw)
    assert f.known.all() and (f.observed_level_rows == 0).all()
    assert (f.observed_trade_total_qty == 0).all()
    assert (f.observed_support_passive_fill_qty == 0).all()


def test_missing_source_never_becomes_zero(tmp_path):
    _, c, ctx, _, _ = setup_local(tmp_path)
    f, receipt = call(c, ctx, None)
    assert not f.known.any() and not f.source_present.any()
    assert f.observed_trade_total_qty.isna().all()
    assert f.observed_support_passive_fill_qty.isna().all()
    assert receipt["status"] == "MISSING_LOCAL_SIDECARS"


def test_missing_price_rows_break_conservation_and_empty_zone_remains_unknown(tmp_path):
    raw, c, ctx, levels, trades = setup_local(tmp_path)
    levels = levels.loc[levels.t_ns != BASE+45*NS]
    write_source(raw, levels, trades)
    c["zone_low_nano"] = 900_000_000_000; c["zone_high_nano"] = 901_000_000_000
    f, _ = call(c, ctx, raw); r = f.loc[f.window == "now"].iloc[0]
    assert not r.level_window_known and r.trade_window_known
    assert np.isnan(r.observed_support_add_qty)
    assert r.observed_trade_total_qty == 0


def test_delayed_local_publication_cannot_backfill_historical_window(tmp_path):
    raw, c, ctx, levels, trades = setup_local(tmp_path)
    levels.loc[levels.t_ns == BASE+40*NS, "available_ns"] = BASE+42*NS
    write_source(raw, levels, trades)
    f, _ = call(c, ctx, raw)
    assert not f.loc[f.window == "dep", "level_window_known"].iloc[0]
    assert f.loc[f.window == "now", "level_window_known"].iloc[0]


def test_exact_native_boundary_prevents_different_interval_conventions_being_equated(tmp_path):
    raw, c, ctx, _, _ = setup_local(tmp_path)
    # A publication at the inclusive canonical endpoint is in the next local
    # ledger bucket. Even matching totals cannot prove that local bucket exact.
    ctx.qt[60] = BASE+60*NS
    f, _ = call(c, ctx, raw); r = f.loc[f.window == "now"].iloc[0]
    assert r.trade_window_known and not r.native_boundaries_clear
    assert not r.level_window_known


def test_unsupported_t_cannot_be_a_complete_trade_window(tmp_path):
    raw, c, ctx, levels, trades = setup_local(tmp_path)
    trades.loc[trades.ordinal == 45*3, "supported"] = False
    write_source(raw, levels, trades)
    f, _ = call(c, ctx, raw)
    assert not f.loc[f.window == "now", "trade_window_known"].iloc[0]


def test_sidecar_source_identity_and_hash_are_required(tmp_path):
    raw, c, ctx, _, _ = setup_local(tmp_path)
    with pytest.raises(ValueError, match="route"):
        local_mbo_diagnostics(c, ctx, raw, expected_route=[1, 4, 3], expected_source_partition_sha256="source")
    with (raw/"trades.parquet").open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="hash"):
        call(c, ctx, raw)


def test_optional_departure_corridor_uses_recorded_departure_not_decision_price(tmp_path):
    raw, c, ctx, _, _ = setup_local(tmp_path)
    c["departure_mid_nano"] = 102_000_000_000
    c["decision_mid_per_oz"] = 120.
    f, _ = call(c, ctx, raw)
    assert set(f.region) == {"zone", "departure_corridor"}
    corridor = f.loc[f.region == "departure_corridor"]
    assert (corridor.region_high_nano == 102_100_000_000).all()
