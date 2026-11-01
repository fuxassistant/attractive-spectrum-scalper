"""Causal, completed-M1 mechanical proposals, independent of future outcomes.

All prices and risk hints are USD per ounce. ``t_ns`` is the information
availability clock, while ``bar_end_ns`` is the canonical minute boundary.
Indicators restart after an invalid minute, missing minute, epoch or UTC day.
The UTC research window is a deliberately fixed experiment restriction; its
labels do not claim to encode the exchange holiday calendar or local sessions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NS = 1_000_000_000
MINUTE_NS = 60 * NS
START_MINUTE = 7 * 60
END_MINUTE = 20 * 60 + 30


def _spec(description: str, style: str, **parameters: float) -> dict:
    return dict(description_ru=description, style=style, parameters=parameters)


SIGNAL_CATALOG = {
    "bb_breakout": _spec("Закрытие пересекает внешнюю полосу BB20", "continuation"),
    "bb_reentry": _spec("Закрытие возвращается внутрь BB20 после выхода", "reversion"),
    "bb_squeeze_breakout": _spec("Выход за предыдущую BB после сжатия ширины", "continuation", squeeze_ratio=0.80),
    "range_breakout": _spec("Закрытие пробивает диапазон предыдущих 20 минут", "continuation"),
    "range_failure": _spec("Прокол диапазона предыдущих 20 минут с возвратом", "reversion"),
    "momentum_second": _spec("Вторая свеча продолжает импульс первой", "continuation", first_body_atr=0.8, second_body_atr=0.15),
    "large_candle_fade": _spec("Контрдвижение после большой направленной свечи", "reversion", body_atr=1.4),
    "wick_rejection": _spec("Длинная тень и закрытие у противоположного края", "reversion", wick_fraction=0.60, range_atr=0.8),
    "ema_pullback": _spec("Касание EMA20 и возврат по тренду EMA20/60", "continuation", trend_atr=0.25),
    "opening_range_breakout": _spec("Пробой завершённого диапазона 07:00–07:30 UTC", "continuation"),
    "opening_range_failure": _spec("Прокол завершённого утреннего диапазона и возврат", "reversion"),
    "inside_bar_breakout": _spec("Выход закрытия из материнской свечи после внутренней", "continuation"),
    "three_bar_reversal": _spec("Разворот после двух направленных свечей с поглощением тела", "reversion", prior_body_atr=0.15),
    "trend_pullback_resume": _spec("После встречной свечи цена возобновляет тренд", "continuation", trend_atr=0.4),
    "stretch_reversal": _spec("Разворотная свеча далеко от EMA60", "reversion", stretch_atr=2.5),
    "compression_release": _spec("Выход из тесного диапазона трёх предыдущих свечей", "continuation", compression_atr=1.2),
    "control_clock": _spec("Контрольные моменты каждые пять минут; сторона выбирается контролем", "control"),
}

VETO_CATALOG = {
    "all": "Без дополнительного вето после общей проверки готовности",
    "trend_aligned": "Направление совпадает с EMA20/60; эффективность движения >= 0.20",
    "range_only": "Причинный режим диапазона: эффективность < 0.20",
    "not_extended": "Не покупать выше +2.5 BB-sigma и не продавать ниже -2.5",
    "normal_volatility": "ATR14 / предыдущий средний ATR100 находится в [0.70, 1.50]",
    "candle_confirmed": "Тело свечи подтверждает сторону, закрытие в соответствующей половине",
}


def _segment_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute only within one uninterrupted segment; never backfill a value."""
    g = g.copy()
    o, h, l, c = (g[k] for k in ("o", "h", "l", "c"))
    pc = c.shift(1)
    g["previous_close"] = pc
    true_range = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    # The first segment minute has no known previous close: use its observed
    # high-low range rather than importing a close across a discontinuity.
    g["atr14"] = true_range.rolling(14, min_periods=14).mean()
    atr = g["atr14"].where(g["atr14"] > 0)
    g["atr_ratio"] = atr / atr.shift(1).rolling(100, min_periods=100).mean()
    g["bb_mid"] = c.rolling(20, min_periods=20).mean()
    sigma = c.rolling(20, min_periods=20).std(ddof=1)
    g["bb_upper"] = g["bb_mid"] + 2 * sigma
    g["bb_lower"] = g["bb_mid"] - 2 * sigma
    g["bb_z"] = (c - g["bb_mid"]) / sigma.where(sigma > 0)
    g["bb_width"] = 4 * sigma
    g["bb_width_ratio"] = g["bb_width"] / g["bb_width"].shift(1).rolling(100, min_periods=100).mean()
    g["ema20"] = c.ewm(span=20, adjust=False, min_periods=20).mean()
    g["ema60"] = c.ewm(span=60, adjust=False, min_periods=60).mean()
    g["trend_atr"] = (g["ema20"] - g["ema60"]) / atr
    g["stretch_atr"] = (c - g["ema60"]) / atr
    path = c.diff().abs().rolling(20, min_periods=20).sum()
    g["efficiency20"] = (c - c.shift(20)).abs() / path.where(path > 0)
    # A truly constant path is range, with efficiency zero rather than unknown.
    g.loc[path.eq(0), "efficiency20"] = 0.0
    for width in (3, 20):
        # Excluding the trigger candle is essential: a close cannot exceed its
        # own high, and including it would silently suppress breakouts.
        g[f"prior_high{width}"] = h.shift(1).rolling(width, min_periods=width).max()
        g[f"prior_low{width}"] = l.shift(1).rolling(width, min_periods=width).min()
    span = (h - l).where(h > l)
    g["body_atr"] = (c - o) / atr
    g["body_abs_atr"] = g["body_atr"].abs()
    g["range_atr"] = (h - l) / atr
    g["close_location"] = ((c - l) / span).fillna(0.5)
    g["upper_wick_fraction"] = ((h - pd.concat([o, c], axis=1).max(axis=1)) / span).fillna(0.0)
    g["lower_wick_fraction"] = ((pd.concat([o, c], axis=1).min(axis=1) - l) / span).fillna(0.0)
    for col in ("o", "h", "l", "c", "body_atr", "bb_upper", "bb_lower", "bb_width_ratio", "prior_high20", "prior_low20"):
        g[f"prev_{col}"] = g[col].shift(1)
    for col in ("o", "h", "l", "c", "body_atr"):
        g[f"prev2_{col}"] = g[col].shift(2)
    g["opening_high"] = np.nan
    g["opening_low"] = np.nan
    bar_minute = pd.to_datetime(g["bar_end_ns"], utc=True).dt.hour * 60 + pd.to_datetime(g["bar_end_ns"], utc=True).dt.minute
    opening = (bar_minute > START_MINUTE) & (bar_minute <= START_MINUTE + 30)
    # OR is available only when all 30 uninterrupted completed minutes exist.
    # Assignment to later rows is safe: those rows occur after the entire OR.
    if opening.sum() == 30:
        known = bar_minute >= START_MINUTE + 30
        g.loc[known, "opening_high"] = h[opening].max()
        g.loc[known, "opening_low"] = l[opening].min()
    g["warmup_bars"] = np.arange(1, len(g) + 1)
    return g


def build_features(bars: pd.DataFrame, min_warmup: int = 120) -> pd.DataFrame:
    """Return causal features from canonical M1 bars, including unusable rows.

    Invalid OHLC, partial coverage and epoch discontinuities invalidate the
    current minute and restart warmup. Availability must be monotone and cannot
    precede the completed bar. Input may cover multiple UTC dates.
    """
    required = {"t_ns", "available_ns", "o", "h", "l", "c", "epoch", "valid", "coverage"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"Missing canonical bar columns: {sorted(missing)}")
    if min_warmup < 120:
        raise ValueError("min_warmup must be at least 120 consecutive minutes")
    if any(bars[col].dtype.kind not in "iu" for col in ("t_ns", "available_ns")):
        raise ValueError("Bar clocks must be integer nanoseconds")
    if any(c.startswith(("future_", "outcome_", "target_")) for c in bars.columns):
        raise ValueError("Future/outcome/target columns are forbidden in feature input")
    g = bars.loc[:, sorted(required | ({"volume"} & set(bars.columns)))].copy()
    g = g.sort_values("t_ns", kind="stable").reset_index(drop=True)
    g["bar_end_ns"] = g["t_ns"].astype("int64")
    g["available_ns"] = g["available_ns"].astype("int64")
    if g["bar_end_ns"].duplicated().any():
        raise ValueError("Duplicate M1 bar boundaries")
    if (g["available_ns"] < g["bar_end_ns"]).any() or (g["available_ns"].diff().dropna() < 0).any():
        raise ValueError("Bar availability must be monotone and at or after bar end")
    g["t_ns"] = g["available_ns"]
    g["bar_publication_delay_seconds"] = (g["available_ns"] - g["bar_end_ns"]) / NS
    clock = pd.to_datetime(g["t_ns"], utc=True)
    g["date"] = pd.to_datetime(g["bar_end_ns"], utc=True).dt.strftime("%Y-%m-%d")
    g["minute_utc"] = clock.dt.hour * 60 + clock.dt.minute
    prices = g[["o", "h", "l", "c"]]
    g["bar_usable"] = (
        g["valid"].fillna(False).astype(bool) & g["epoch"].ge(0)
        & np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        & g["h"].ge(prices[["o", "c"]].max(axis=1))
        & g["l"].le(prices[["o", "c"]].min(axis=1))
        & g["h"].ge(g["l"]) & g["coverage"].ge(1 - 1e-9)
    )
    reset = (g["date"].ne(g["date"].shift()) | g["epoch"].ne(g["epoch"].shift())
             | g["bar_end_ns"].diff().ne(MINUTE_NS) | ~g["bar_usable"]
             | ~g["bar_usable"].shift(1, fill_value=False))
    g["segment_id"] = reset.cumsum().astype("int64")
    if len(g):
        g = pd.concat([_segment_features(part) for _, part in g.groupby("segment_id", sort=False)]).sort_index()
    else:
        g = _segment_features(g)
    for native, pretty in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close")):
        g[pretty] = g[native]
    g["trend_regime"] = np.select([g["efficiency20"] < 0.20, g["efficiency20"] >= 0.35], ["range", "trend"], default="transition")
    g["vol_regime"] = np.select([g["atr_ratio"] < 0.80, g["atr_ratio"] > 1.25], ["low", "high"], default="normal")
    g["time_regime"] = np.select([g["minute_utc"] < 12 * 60, g["minute_utc"] < 17 * 60], ["07_12_utc", "12_17_utc"], default="17_2030_utc")
    g["regime"] = g["trend_regime"] + "_" + g["vol_regime"]
    g["ready"] = (g["bar_usable"] & g["warmup_bars"].ge(min_warmup)
                  & np.isfinite(g[["atr14", "atr_ratio", "bb_z", "trend_atr", "efficiency20"]]).all(axis=1)
                  & g["atr14"].gt(0) & g["minute_utc"].ge(START_MINUTE) & g["minute_utc"].lt(END_MINUTE))
    return g


def _conditions(f: pd.DataFrame, signal_id: str, p: dict) -> tuple[pd.Series, pd.Series]:
    c, o, h, l = (f[k] for k in ("c", "o", "h", "l"))
    pc = f["prev_c"]
    if signal_id == "bb_breakout":
        return (c > f.bb_upper) & (pc <= f.prev_bb_upper), (c < f.bb_lower) & (pc >= f.prev_bb_lower)
    if signal_id == "bb_reentry":
        inside = (c >= f.bb_lower) & (c <= f.bb_upper)
        return (pc < f.prev_bb_lower) & inside, (pc > f.prev_bb_upper) & inside
    if signal_id == "bb_squeeze_breakout":
        squeeze = f.prev_bb_width_ratio < p["squeeze_ratio"]
        return squeeze & (c > f.prev_bb_upper) & (pc <= f.prev_bb_upper), squeeze & (c < f.prev_bb_lower) & (pc >= f.prev_bb_lower)
    if signal_id == "range_breakout":
        return (c > f.prior_high20) & (pc <= f.prev_prior_high20), (c < f.prior_low20) & (pc >= f.prev_prior_low20)
    if signal_id == "range_failure":
        inside = (c > f.prior_low20) & (c < f.prior_high20)
        return (l < f.prior_low20) & inside & (c > o), (h > f.prior_high20) & inside & (c < o)
    if signal_id == "momentum_second":
        return ((f.prev_body_atr > p["first_body_atr"]) & (f.body_atr > p["second_body_atr"]) & (c > f.prev_h),
                (f.prev_body_atr < -p["first_body_atr"]) & (f.body_atr < -p["second_body_atr"]) & (c < f.prev_l))
    if signal_id == "large_candle_fade":
        return f.body_atr < -p["body_atr"], f.body_atr > p["body_atr"]
    if signal_id == "wick_rejection":
        active = f.range_atr >= p["range_atr"]
        return active & (f.lower_wick_fraction >= p["wick_fraction"]), active & (f.upper_wick_fraction >= p["wick_fraction"])
    if signal_id == "ema_pullback":
        return ((f.trend_atr > p["trend_atr"]) & (l <= f.ema20) & (c > f.ema20) & (c > o),
                (f.trend_atr < -p["trend_atr"]) & (h >= f.ema20) & (c < f.ema20) & (c < o))
    if signal_id == "opening_range_breakout":
        return (c > f.opening_high) & (pc <= f.opening_high), (c < f.opening_low) & (pc >= f.opening_low)
    if signal_id == "opening_range_failure":
        inside = (c > f.opening_low) & (c < f.opening_high)
        return (l < f.opening_low) & inside & (c > o), (h > f.opening_high) & inside & (c < o)
    if signal_id == "inside_bar_breakout":
        inside = (f.prev_h <= f.prev2_h) & (f.prev_l >= f.prev2_l)
        return inside & (c > f.prev2_h), inside & (c < f.prev2_l)
    if signal_id == "three_bar_reversal":
        threshold = p["prior_body_atr"]
        return ((f.prev2_body_atr < -threshold) & (f.prev_body_atr < -threshold) & (c > f.prev_o) & (c > o),
                (f.prev2_body_atr > threshold) & (f.prev_body_atr > threshold) & (c < f.prev_o) & (c < o))
    if signal_id == "trend_pullback_resume":
        return ((f.trend_atr > p["trend_atr"]) & (f.prev_body_atr < 0) & (c > f.prev_h) & (c > o),
                (f.trend_atr < -p["trend_atr"]) & (f.prev_body_atr > 0) & (c < f.prev_l) & (c < o))
    if signal_id == "stretch_reversal":
        return (f.stretch_atr < -p["stretch_atr"]) & (c > o), (f.stretch_atr > p["stretch_atr"]) & (c < o)
    if signal_id == "compression_release":
        compressed = (f.prior_high3 - f.prior_low3) < p["compression_atr"] * f.atr14
        return compressed & (c > f.prior_high3), compressed & (c < f.prior_low3)
    if signal_id == "control_clock":
        return f.minute_utc.mod(5).eq(0), pd.Series(False, index=f.index)
    raise ValueError(f"Unknown signal: {signal_id}")


def generate_candidates(features: pd.DataFrame, signal_ids=None, parameters=None) -> pd.DataFrame:
    """Enumerate alternative proposals, not a jointly tradable portfolio.

    ``parameters`` maps signal IDs to overrides for that signal's documented
    numeric thresholds. Candidate IDs are assigned by the experiment runner.
    Both directions remain separate records when a formula permits both.
    """
    signal_ids = list(SIGNAL_CATALOG) if signal_ids is None else list(signal_ids)
    if len(signal_ids) != len(set(signal_ids)):
        raise ValueError("Signal IDs must be unique")
    parameters = {} if parameters is None else parameters
    unknown = set(signal_ids) - set(SIGNAL_CATALOG)
    if unknown or set(parameters) - set(SIGNAL_CATALOG):
        raise ValueError(f"Unknown signal IDs: {sorted(unknown | (set(parameters) - set(SIGNAL_CATALOG)))}")
    rows = []
    for signal_id in signal_ids:
        defaults = SIGNAL_CATALOG[signal_id]["parameters"]
        overrides = parameters.get(signal_id, {})
        if set(overrides) - set(defaults):
            raise ValueError(f"Unknown parameters for {signal_id}: {sorted(set(overrides) - set(defaults))}")
        p = {**defaults, **overrides}
        if any(not np.isfinite(v) or v <= 0 for v in p.values()):
            raise ValueError("Signal parameters must be finite positive numbers")
        for side, mask in zip((1, -1), _conditions(features, signal_id, p)):
            selected = features.loc[features.ready & mask.fillna(False)].copy()
            selected["signal_id"] = signal_id
            selected["side"] = side
            selected["risk_hint"] = selected["atr14"]
            selected["candle_risk"] = np.where(side > 0, selected.c - selected.l, selected.h - selected.c)
            rows.append(selected)
    if not rows:
        return features.iloc[:0].assign(signal_id=pd.Series(dtype=str), side=pd.Series(dtype="int8"), risk_hint=pd.Series(dtype=float), candle_risk=pd.Series(dtype=float))
    return pd.concat(rows, ignore_index=True).sort_values(["t_ns", "signal_id", "side"], kind="stable").reset_index(drop=True)


def accept_mask(candidates: pd.DataFrame, veto_id: str) -> pd.Series:
    """Causal side-aware gates. Liquidity/cost gates belong to the quote runner."""
    if veto_id not in VETO_CATALOG:
        raise ValueError(f"Unknown veto: {veto_id}")
    f = candidates
    if veto_id == "all":
        result = pd.Series(True, index=f.index)
    elif veto_id == "trend_aligned":
        result = (f.side * f.trend_atr > 0.20) & (f.efficiency20 >= 0.20)
    elif veto_id == "range_only":
        result = f.efficiency20 < 0.20
    elif veto_id == "not_extended":
        result = f.side * f.bb_z <= 2.5
    elif veto_id == "normal_volatility":
        result = f.atr_ratio.between(0.70, 1.50, inclusive="both")
    else:
        result = (f.side * f.body_atr > 0) & (f.side * (f.close_location - 0.5) >= 0)
    return (f.ready & result).fillna(False).astype(bool)
