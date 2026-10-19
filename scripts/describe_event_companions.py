"""Descriptive companions; retrospective groups must never become causal X."""
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
from spectrum.ingest import save_json
from spectrum.core.storage import sha

out=Path(sys.argv[1] if len(sys.argv)>1 else 'reports/event_landmarks_v1/run_003')
d=pq.read_table(out/'events.parquet').to_pandas();d=d[d.row_kind=='event'].copy()
d['response_per_100']=np.divide(abs(d.move_ticks)*100,d.volume,out=np.zeros(len(d)),where=d.volume>0)
d['depth_down']=d.opposite_depth_change_fraction<0
d['fill_volume']=d.matched_fill_remove_bid+d.matched_fill_remove_ask
d['recovery_volume']=d.post_trade_add_bid+d.post_trade_add_ask
d['future_group']=np.where(d.event_class=='fresh_impulse',np.where(d.retrospectively_no_repeat==True,'isolated','series_start'),np.where(d.retrospectively_no_repeat==True,'series_end','series_middle'))
rows=[]
for split,g in [('all',d),('evaluation',d[d.date>='2026-05-01'])]:
    for family in ('event_class','activity','future_group'):
        z=g if family!='future_group' else g[g.recurrence_mature==True]
        for name,a in z.groupby(family):
            row=dict(split=split,family=family,group=name,n=len(a),days=int(a.date.nunique()),
                median_abs_move=float(a.move_ticks.abs().median()),median_volume=float(a.volume.median()),
                median_response_per100=float(a.response_per_100.median()),same_way_robust_flow_share=float(a.same_way_flow_robust.mean()),
                opposing_robust_flow_share=float(a.opposing_flow_robust.mean()),depth_down_share=float(a.depth_down.mean()),
                median_opposite_depth_change=float(a.opposite_depth_change_fraction.median()),
                spread_widen_share=float((a.spread_change>1e-7).mean()),
                fill_present_share=float((a.fill_volume>0).mean()),recovery_present_share=float((a.recovery_volume>0).mean()),
                median_fill_volume=float(a.fill_volume.median()),median_recovery_volume=float(a.recovery_volume.median()),
                median_coherence=float(a.classified_flow_to_return_original.median()),
                median_peak_energy_share=float(a.return_peak5_share.median()))
            rows.append(row)
save_json(out/'companions.json',dict(events_sha256=sha(out/'events.parquet'),groups=rows,
    limitations=['future_group is retrospective only; not available at entry.',
    'Depth is relative top12, not a fixed price-level queue: movement/reindexing also changes it.',
    'Matched fill and post-trade add are reconstructed evidence/proxies, not intent or hidden liquidity.',
    'Per100 response is an observed ratio with varying volatility, not estimated price impact.',
    'No hypothesis tests or fitted selection rule from these tables.']))
for r in rows:
    if r['split']=='evaluation':print(r)
