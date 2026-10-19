"""Explicit allowlist of event X; controls are diagnostic matching rows only."""
from pathlib import Path
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from spectrum.ingest import save_json
from spectrum.core.storage import sha

out=Path(sys.argv[1] if len(sys.argv)>1 else 'reports/event_landmarks_v1/run_003')
d=pq.read_table(out/'events.parquet').to_pandas();d=d[d.row_kind=='event'].copy()
identity=['key','date','symbol','t_ns','decision_ns','match_id']
x=['event_class','activity','prior_events_180','interval_cv','interval_cv_known','last_gap_s',
   'move_threshold','volume_threshold','event_start_price','event_end_price','direction',
   'move_ticks','volume','classified_flow','unknown_volume','unknown_fraction','pressure_lower','pressure_upper',
   'flow_direction_robust','baseline_volatility','baseline_unknown','spread_before','spread_after',
   'depth_bid_change_fraction','depth_ask_change_fraction','add_bid','add_ask','remove_bid','remove_ask',
   'matched_fill_remove_bid','matched_fill_remove_ask','post_trade_add_bid','post_trade_add_ask',
   'opposite_depth_change_fraction','same_way_flow_robust','opposing_flow_robust','spread_change',
   'return_peak5_share','flow_peak5_share']
for pair in ('classified_flow_to_return','volume_to_abs_return'):
    for variant in ('original','remove5','interpolate15'):
        x.extend([f'{pair}_{variant}',f'{pair}_{variant}_known'])
y=['recurrence_mature','future_events_180','retrospectively_no_repeat','retrospectively_isolated','recurrence_label_available_ns']
for h in (30,120):
    y.extend(f'{s}_{h}' for s in ('mature','label_available_ns','y','continue_net','fade_net','abs_response','origin_retest','extends3ticks','both_landmarks'))
assert not set(x)&set(y)
assert set(d.columns)-set(identity+x+y)=={'row_kind','bin','end_index'}
for name,cols in [('causal_events',x),('event_outcomes',y)]:
    pq.write_table(pa.Table.from_pandas(d[identity+cols],preserve_index=False),out/(name+'.parquet'),compression='zstd')
save_json(out/'feature_contract.json',dict(version='event_landmarks_v1',rows=len(d),identity=identity,x_allowlist=x,y_only=y,
    source_events_sha256=sha(out/'events.parquet'),x_sha256=sha(out/'causal_events.parquet'),y_sha256=sha(out/'event_outcomes.parquet'),
    admission='representation research only; no fitted policy or economic admission',
    controls='Excluded from X/Y exports. In diagnostic events.parquet, control event_class/activity refer to the MATCHED event, not the control own causal state.',
    maturity='Only consume Y after its label_available_ns and when mature; direction0 disallows directional economic proxies.',
    causal_time='X available at decision_ns. Current start/end landmarks are observed geometry, not inferred support/resistance.'))
print({'rows':len(d),'causal_features':len(x),'outcome_fields':len(y)})
