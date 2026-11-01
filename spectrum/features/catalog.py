"""Discover reusable families without importing models or historical experiments.

Families keep their native clocks and input contracts. Sharing a catalog does
not make trade-exact VWAP equivalent to a midpoint proxy, or M1 equal to 1s.
"""
from dataclasses import asdict, dataclass
from importlib import import_module


@dataclass(frozen=True)
class FamilySpec:
    name: str
    group: str
    entrypoint: str
    input_contract: str
    clock: str
    description: str
    availability: str = "Published prefix only; retain support masks and continuity."

    def resolve(self):
        module, attribute = self.entrypoint.split(":", 1)
        return getattr(import_module(module), attribute)

    def describe(self) -> dict:
        return asdict(self)


def _family(name, group, entrypoint, input_contract, clock, description, **kwargs):
    return FamilySpec(name, group, entrypoint, input_contract, clock, description, **kwargs)


FAMILIES = (
    _family("core.ohlc", "price", "spectrum.core.feature_families:default_families", "core context / Family('ohlc')", "completed M1", "OHLC, BB, day/week levels, session geometry"),
    _family("core.quotes", "price", "spectrum.core.feature_families:quotes", "core context", "publication + completed M1", "BBO, microprice, path, efficiency and event intensity"),
    _family("core.trade", "flow", "spectrum.core.feature_families:trade", "core context", "publication + completed M1", "Classified and unknown trade volume; effort per displacement"),
    _family("core.depth", "liquidity", "spectrum.core.feature_families:depth", "core context", "publication + completed M1", "Depth, churn, fills, replenishment and sweep cost"),
    _family("motion", "price", "spectrum.features.families.motion:motion", "Context / seconds", "1s", "Spread, displacement, path, efficiency and depth"),
    _family("flow", "flow", "spectrum.features.families.flow:flow", "Context / seconds", "1s", "Rolling published MBO counters"),
    _family("spectral", "spectrum", "spectrum.features.families.spectral:spectral", "Context / seconds", "1s", "Multi-window return and flow spectra"),
    _family("spectral_activity", "spectrum", "spectrum.features.families.spectral_activity:activity", "Context / all T counters plus unknown-side volume", "1s", "Total/observed-signed/unknown activity spectra, exact white reference and side uncertainty bounds"),
    _family("depth_balance", "liquidity", "spectrum.features.families.depth_balance:calculate", "Context / seconds", "1s", "Visible depth balance at twelve ticks"),
    _family("volatility_vwap_proxy", "volatility_geometry", "spectrum.features.families.volatility_vwap:proxy", "Context / midpoint seconds", "1s", "Volume-weighted midpoint proxy, scale and dispersion"),
    _family("volatility_vwap_exact", "volatility_geometry", "spectrum.features.families.volatility_vwap:exact", "Context / exact trade moments", "1s", "Trade-exact weighted geometry; requires trade first/second moments"),
    _family("opportunity_baseline", "volatility_geometry", "spectrum.features.families.opportunity:causal", "supported second grid + origin indices", "as-of decision", "Causal sigma, spread and baseline support"),
    _family("signature.price", "price", "spectrum.features.families.signature:price_family", "supported grid + origins + known sigma", "as-of decision", "Normalized returns, ranges and running session anchors"),
    _family("signature.flow", "flow", "spectrum.features.families.signature:flow_family", "published MBO seconds + intents", "as-of decision", "Long-window flow, rate and book pressure"),
    _family("signature.micro", "liquidity", "spectrum.features.families.signature:micro_family", "prepared minute arrays + intents", "completed M1", "Explicit projection of thirty book, flow and spectral fields"),
    _family("episodes", "events", "spectrum.features.families.episodes:features", "seconds + segment tokens", "1s", "Recognized flow-response episodes and their causal age"),
    _family("flow_context.local", "flow", "spectrum.features.families.flow_context:local_features", "seconds + segment tokens", "1s", "Local multi-scale flow and price response"),
    _family("flow_context.day", "context", "spectrum.features.families.flow_context:context_features", "current/previous minute tapes + origins", "completed M1 / as-of decision", "Compressed previous and current day geometry"),
    _family("flow_context.spectrum", "spectrum", "spectrum.features.families.flow_context:trailing_spectra", "minute tape + supported windows", "completed M1", "Trailing price/volume STFT"),
    _family("volatility", "volatility_geometry", "spectrum.features.families.volatility:features", "supported second arrays + session metadata", "1s", "Sigma, ranges, activity spectra, efficiency, flow and session clocks"),
    _family("anchored_profile", "context", "spectrum.features.families.anchored_profile:features_day", "completed previous session + current arrays", "sampled 1s prefix", "Anchored VWAP, volume profile, POC, value area and path"),
    _family("response", "spectrum", "spectrum.features.families.response:window_features", "published seconds + window endpoint", "512s trailing window", "Auto/cross spectra, coherence and measured flow-price response"),
    _family("landmarks", "events", "spectrum.features.families.landmarks:detect_bursts", "published window descriptors", "recognition time", "Causal bursts and price landmarks"),
    _family("event_context", "events", "spectrum.features.families.event_context:event_context", "intent times + recognized event times", "strictly before intent", "Event count, age and explicit event-bank coverage"),
    _family("market_history", "context", "spectrum.features.families.market_history:market_features", "minute arrays + exact quote arrays", "as-of decision", "Completed M1 history and latest quote contrast"),
    _family("scalper_context", "context", "spectrum.features.families.scalper_context:features", "supported grid + origins + published seconds", "as-of decision", "Short-horizon returns, range position, volatility ratios and flow"),
    _family("side_orientation", "context", "spectrum.features.families.scalper_context:side_features", "causal scalper context + chosen side", "as-of decision", "Side-oriented returns, pressure bounds and range location"),
    _family("observation_summary", "representation", "spectrum.features.families.observation_summary:summary_at", "fast/context tables + explicit origins", "publication watermark", "Scalar price/book/context projections with causal eligibility"),
    _family("queue.flow", "liquidity", "spectrum.features.families.queue_state:window_flows", "MBO tape + decision BBO addresses", "strictly before intent", "Adds, cancels, fills and aggressor volume at decision prices"),
    _family("queue.depth", "liquidity", "spectrum.features.families.queue_state:level_sizes", "MBO tape + decision BBO addresses", "strictly before intent", "Resting size using the shared FIFO queue kernel"),
    _family("crossing_geometry", "volatility_geometry", "spectrum.features.alignment:geometry_features", "origin scene with causal BB geometry", "as-of decision", "BB position, width and risk scale; future map columns ignored"),
    _family("memory.minutes", "context", "spectrum.representation.memory:minute_values", "completed minute tape", "publication time", "Mixed-resolution minute observations"),
    _family("memory.anchors", "context", "spectrum.representation.memory:anchor_values", "published seconds/minutes + previous session", "publication time", "Session/opening anchors and their known masks"),
    _family("memory.events", "events", "spectrum.representation.memory:encode_events", "recognized event table", "event availability", "Explicit event allowlist without future labels"),
    _family("representation.sequences", "representation", "spectrum.representation.windows:SequenceStore", "origin frame + float32 matrix + family column groups", "physical continuity + cadence", "Lazy windows, explicit missing-X positions and common training-row scaler"),
    _family("representation.fast", "representation", "spectrum.representation.representation:make_fast", "canonical seconds + interaction ledger + exact trades", "1s publication watermark", "BBO, path, aggression, liquidity mutation and masks"),
    _family("representation.minutes", "representation", "spectrum.representation.representation:make_minutes", "fast frame + exact trades + canonical bars", "completed M1 watermark", "Minute price, trade VWAP, bands, volume and STFT"),
    _family("representation.context", "representation", "spectrum.representation.representation:anchored_context", "minutes + trades + explicit current/previous date", "two-session prefix", "Anchored VWAP, profile and preceding session context"),
    _family("representation.zones", "representation", "spectrum.representation.zones:zone_and_event_tables", "fast/context frames + per-price ledger", "version availability", "Versioned zones, interactions and event tokens"),
)


def families(group: str | None = None) -> tuple[FamilySpec, ...]:
    return tuple(f for f in FAMILIES if group is None or f.group == group)


def get_family(name: str) -> FamilySpec:
    for family in FAMILIES:
        if family.name == name:
            return family
    raise KeyError(f"Unknown feature family: {name}")
