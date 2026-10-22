"""V2-identical observations with separate quote continuity and missingness.

The current-origin admission policy and feature values remain frozen. A late or
inadmissible one-second feature row is not itself a quote-support break. Missing
five-second rows remain positions on the clock; only their observation masks
are zero. The lazy reader owns padding and must reset on date/cadence changes
and the full-publication physical support token, including hidden transitions.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from spectrum.research_v2 import features as v2

NS = 1_000_000_000


def _metadata_proof(day, origins_ns, config, base_receipt):
    """Re-read only source metadata; verify it is the payload already used by V2.

    Hashes bind the complete immutable feature files. No quote file is loaded or
    hashed here; the caller supplies and certifies the same full quote tape.
    """
    source_root = Path(config['data']['source_root'])
    hashes = base_receipt['source_sha256']
    catalog_path = source_root/'manifest.json'
    catalog_key = str(catalog_path.resolve())
    if catalog_key not in hashes or v2._sha(catalog_path) != hashes[catalog_key]:
        raise ValueError('V3 metadata catalog differs from the consumed V2 source')
    catalog = v2._json(catalog_path)
    captures = {r['key']: r for r in catalog['captures'] if r.get('status') == 'COMPLETE'}
    parts = []
    for number, part in enumerate(day.meta['quote_parts']):
        capture = captures.get(part['key'])
        if capture is None:
            raise ValueError('Missing bound V2 feature capture')
        path = Path(capture['features'])/'features.parquet'
        key = str(path.resolve())
        if key not in hashes or v2._sha(path) != hashes[key]:
            raise ValueError('V3 metadata feature payload differs from consumed V2 values')
        before = path.stat()
        frame = pq.read_table(path, columns=list(v2.SOURCE_METADATA),
            filters=[('t_ns', '>=', int(part['start_ns'])), ('t_ns', '<', int(part['end_ns']))]).to_pandas()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Feature payload changed while metadata was read')
        frame['source_part'] = number
        parts.append(frame)
    if not parts:
        raise ValueError('At least one consumed feature part required')
    source = pd.concat(parts, ignore_index=True).sort_values('t_ns').reset_index(drop=True)
    _, proof = v2.asof_source(source, origins_ns, [],
        float(config['data'].get('max_feature_age_seconds', 2.)))
    return proof


def extend_history_metadata(base, q, origins_ns, source_proof, config):
    """Add two continuity views without changing a single V2 X value or gate.

    `segment_token` deliberately remains the legacy alias. A restored-history
    reader selects `physical_segment_token` explicitly, ignores feature holes as
    reset events, and uses `feature_known_mask` for observation rather than row
    presence. No forward-filled values are manufactured by this adapter.
    """
    t = np.asarray(origins_ns, np.int64)
    if t.ndim != 1 or not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError('Nonempty strictly chronological origins required')
    raw = np.asarray(base['x_raw'])
    available = np.asarray(base['feature_available'])
    legacy = np.asarray(base['segment_token'])
    if raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[0] != len(t):
        raise ValueError('Frozen V2 float32[N,F] observations required')
    if available.shape != t.shape or available.dtype != bool:
        raise ValueError('Aligned V2 boolean admission required')
    if legacy.shape != t.shape or not np.issubdtype(legacy.dtype, np.integer):
        raise ValueError('Aligned V2 integer continuity tokens required')
    required = ('ready', 'row_index', 'age_seconds', 'endpoint_ns', 'publication_ns', 'source_segment')
    if not all(k in source_proof and np.asarray(source_proof[k]).shape == t.shape for k in required):
        raise ValueError('Complete aligned source-publication proof required')
    selected = np.asarray(source_proof['row_index']) >= 0
    endpoint = np.asarray(source_proof['endpoint_ns'], np.int64)
    published = np.asarray(source_proof['publication_ns'], np.int64)
    age = np.asarray(source_proof['age_seconds'], float)
    if np.any(selected & ((endpoint > t) | (published > t))):
        raise ValueError('Selected source endpoint/publication is in the future')
    if not np.allclose(age[selected], (t[selected]-endpoint[selected])/NS, atol=1e-12, rtol=0):
        raise ValueError('Source endpoint age metadata disagrees with its clock')
    age_limit = float(config['data'].get('max_feature_age_seconds', 2.))
    if not np.isfinite(age_limit) or age_limit <= 0:
        raise ValueError('Positive feature age limit required')
    gap = float(config['execution'].get('max_quote_gap_seconds', 60.))
    # One full-publication scan retains invalid/epoch/gap events even when both
    # observed five-second origins are individually valid.
    states = v2.quote_state(q, np.r_[t, published], gap)
    n = len(t)
    physical_ready = states['ready'][:n]
    physical_token = states['segment'][:n].astype(np.int64, copy=True)
    source_same_support = (physical_ready & states['ready'][n:]
                           & (physical_token == states['segment'][n:]))
    rebuilt_available = np.asarray(source_proof['ready'], bool) & source_same_support
    if not np.array_equal(available, rebuilt_available):
        raise ValueError('V3 metadata would change the frozen V2 admission cohort')
    if np.any(available & (~selected | ~np.isfinite(age) | (age < 0) | (age > age_limit))):
        raise ValueError('Admitted source violates endpoint/publication age bounds')
    legacy_from_proof = v2.common_segments(t, available, np.asarray(source_proof['source_segment']),
        physical_token, config['data']['cadence_seconds'])
    if not np.array_equal(legacy, legacy_from_proof):
        raise ValueError('Reconstructed metadata disagrees with frozen V2 segment tokens')
    quote_ordinal = states['ordinal'][:n]
    quote_age = np.where(quote_ordinal >= 0, (t-q.quote_t[np.maximum(quote_ordinal, 0)])/NS, np.nan)
    observed = np.isfinite(raw) & available[:, None]
    if np.any(observed & ~physical_ready[:, None]):
        raise ValueError('Observed X outside physical quote support')
    receipts = copy.deepcopy(base['receipts'])
    receipts.setdefault('implementation_sha256', {})[str(Path(__file__).resolve())] = v2._sha(__file__)
    same_physical = (physical_ready[1:] & physical_ready[:-1]
                     & (physical_token[1:] == physical_token[:-1]))
    reset_removed = same_physical & (legacy[1:] != legacy[:-1])
    receipts['v3_history'] = dict(
        version='QUOTE_CONTINUITY_SEPARATE_FROM_FEATURE_MISSINGNESS_V3',
        v2_values_unchanged=True, v2_current_origin_admission_unchanged=True,
        v2_legacy_tokens_reproduced=True, extra_feature_channels=0,
        origins=len(t), physical_support_origins=int(physical_ready.sum()),
        current_feature_available=int(available.sum()),
        missing_feature_rows_inside_physical_support=int((physical_ready & ~available).sum()),
        adjacent_legacy_changes_with_same_supported_quote_token=int(reset_removed.sum()),
        physical_token='Full quote publication invalid/epoch/gap boundary; -1 when current quote support is unavailable. Day/cadence boundaries remain the reader responsibility.',
        observation='feature_known_mask = finite frozen x_raw AND frozen feature_available. False cells must not be used even when x_raw has a finite geometry/context value.',
        missing_row='Keep the real five-second position with zero values/masks and clock_row_present=1. Only actual left padding has row-present=0. A missing feature row does not reset supported history.',
        legacy_alias='segment_token remains the V2 token. Restored mode must explicitly select physical_segment_token; strict mode uses legacy_segment_token.',
        source_clock='Exact asof endpoint/publication/age metadata, publication prefix and unchanged <=2s endpoint-age policy; no additional freshness channel is appended to X.',
        no_forward_fill=True, no_quote_payload_read=True,
    )
    result = dict(base)
    result.update(legacy_segment_token=legacy.astype(np.int64, copy=True),
        physical_segment_token=physical_token, physical_support_available=physical_ready.copy(),
        feature_known_mask=observed, clock_row_present=np.ones(n, bool),
        source_row_index=np.asarray(source_proof['row_index'], np.int64).copy(),
        source_endpoint_ns=endpoint.copy(), source_available_ns=published.copy(),
        source_age_seconds=age.copy(), source_row_available=np.asarray(source_proof['ready'], bool).copy(),
        source_segment_token=np.asarray(source_proof['source_segment'], np.int64).copy(),
        source_same_quote_support=source_same_support, quote_age_seconds=quote_age,
        receipts=receipts)
    return result


def build_features(day, q, origins_ns, scene, config):
    """Build exactly V2 values/admission, then add causal continuity metadata."""
    base = v2.build_features(day, q, origins_ns, scene, config)
    proof = _metadata_proof(day, origins_ns, config, base['receipts'])
    return extend_history_metadata(base, q, origins_ns, proof, config)
