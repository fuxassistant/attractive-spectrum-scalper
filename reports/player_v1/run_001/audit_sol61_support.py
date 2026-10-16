import json
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd
from spectrum.player.data import Day
from spectrum.player.contract import NS
r=Path('data/research/player_v1/run_001');days={};records=[]
for label in ('selected_train','selected_test','untrained_test'):
 for p in sorted((r/label).iterdir()):
  try:tr=pd.read_csv(p/'trades.csv')
  except pd.errors.EmptyDataError:continue
  if p.name not in days:days[p.name]=Day(Path('data/derived/player_v1')/('week1' if p.name<'2026-05-10' else 'week2')/p.name)
  d=days[p.name]
  for trade in tr.to_dict('records'):
   if not trade['censored']:continue
   a=int(trade['entry_index']);z=int(trade['exit_index']);sl=slice(a,z+1);t=d.quote_t[sl];b=d.bid[sl];ask=d.ask[sl]
   masks=dict(invalid_flag=~d.valid[sl],crossed=ask<b,nonpositive_bid=b<=0,source_age=t-d.source_t[sl]>5*NS,epoch=d.epoch[sl]!=d.epoch[a],gap=np.r_[False,np.diff(t)>5*NS])
   hit=np.flatnonzero(np.logical_or.reduce(list(masks.values())));first=int(hit[0]);causes=[k for k,v in masks.items() if v[first]]
   records.append(dict(label=label,date=p.name,entry_index=a,first_loss_index=a+first,exit_index=z,first_causes=causes,any_causes=[k for k,v in masks.items() if v.any()],gap_seconds=float((t[first]-t[first-1])/NS) if first else None,bid_nano=int(b[first]),ask_nano=int(ask[first]),entry_hour_utc=pd.Timestamp(int(t[0]),tz='UTC').isoformat()))
summary={label:dict(censored=sum(x['label']==label for x in records),first_causes=dict(Counter(c for x in records if x['label']==label for c in x['first_causes'])),any_causes=dict(Counter(c for x in records if x['label']==label for c in x['any_causes']))) for label in ('selected_train','selected_test','untrained_test')}
receipt=Path('reports/player_v1/run_001/independent_sol61_receipt.json');j=json.loads(receipt.read_text());j['support_loss_diagnosis']=dict(summary=summary,trades=records);j['visual_inspection']=dict(files=['epochs.png','equity.png','policy.png']+[f'scene_{i:02d}.png' for i in range(8)],verdict='Inspected all eleven PNGs; agree with numeric negative/censored result. Scenes expose correct direction/barriers and feature availability gaps, cannot establish predictive edge.')
receipt.write_text(json.dumps(j,indent=2));print(json.dumps(summary,indent=2));print(json.dumps([x for x in records if x['label']=='selected_test'][:3],indent=2))
