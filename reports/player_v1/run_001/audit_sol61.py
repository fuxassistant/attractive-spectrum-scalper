import json,time
from pathlib import Path
import numpy as np,pandas as pd
from spectrum.player.data import Day
from spectrum.player.contract import Contract,PLANS,NS
root=Path('data/research/player_v1/run_001'); days={}; errors=[]; rows=[]; count=0
for folder in sorted(root.iterdir()):
 if not folder.is_dir():continue
 for p in sorted(folder.iterdir()):
  if not p.is_dir():continue
  date=p.name; week=1 if date<'2026-05-10' else 2
  if date not in days:days[date]=Day(Path('data/derived/player_v1')/f'week{week}'/date)
  d=days[date];m=json.loads((p/'metrics.json').read_text()); dec=pd.read_parquet(p/'decisions.parquet')
  try:tr=pd.read_csv(p/'trades.csv')
  except pd.errors.EmptyDataError:tr=pd.DataFrame()
  cash=0.;censor=0
  for r in tr.to_dict('records'):
   count+=1; a=int(r['entry_index']);z=int(r['exit_index']);side=int(r['side']);plan=PLANS[int(r['plan'])]
   ep=int(d.ask[a] if side>0 else d.bid[a]);xp=int(d.bid[z] if side>0 else d.ask[z]);net=side*(xp-ep)/1e9-.3;cash+=net;censor+=int(r['censored'])
   if abs(net-r['net'])>1e-8 or ep!=r['entry_nano'] or xp!=r['exit_nano']:errors.append([str(p),'cash/price',r])
   now=d.quote_t[a:z+1];src=d.source_t[a:z+1];bid=d.bid[a:z+1];ask=d.ask[a:z+1];epc=d.epoch[a:z+1]
   good=d.valid[a:z+1]&(bid>0)&(ask>=bid)&(now-src<=5*NS)
   loss=(~good)|(epc!=epc[0]);loss[1:]|=np.diff(now)>5*NS
   cens=np.maximum.accumulate(loss)
   delta=side*((bid if side>0 else ask)-ep)
   trigger=good&(cens|(now>=d.deadline)|(delta<=-plan[1]*100_000_000)|(delta>=plan[0]*100_000_000)|(now-now[0]>=plan[2]*NS))
   ids=np.flatnonzero(trigger)
   # risk can preempt these conditions, never follow them
   if len(ids) and ids[0]!=z-a:errors.append([str(p),'missed_first_hit',a+int(ids[0]),z,r['reason']])
   if not len(ids) and r['reason']!='risk_stop':errors.append([str(p),'no_trigger',r])
   dt=int(d.decision_t[int(r['decision_index'])]);
   if not(dt+100_000_000<int(now[0])<=dt+2_100_000_000):errors.append([str(p),'entry_latency',r])
   if not d.can_enter[int(r['decision_index'])]:errors.append([str(p),'entry_gate',r])
  if abs(cash-m['realized'])>1e-7 or (not m['open_terminal'] and abs(cash-m['net'])>1e-7):errors.append([str(p),'metric_cash',cash,m])
  if abs(dec.reward.sum()-m['net'])>1e-7:errors.append([str(p),'rewards'])
  if (dec.action.to_numpy()[~d.can_enter]!=0).any():errors.append([str(p),'gate_actions'])
  if (p/'quote_equity.npy').exists():
   eq=np.load(p/'quote_equity.npy',mmap_mode='r');dd=float(np.max(np.maximum.accumulate(eq)-eq))
   if abs(dd-m['max_drawdown'])>1e-7:errors.append([str(p),'drawdown',dd,m['max_drawdown']])
  rows.append(dict(label=folder.name,date=date,net=m['net'],trades=m['trades'],censored=m['censored'],risk_halted=m['risk_halted']))
out=Path('reports/player_v1/run_001');pd.DataFrame(rows).to_csv(out/'independent_sol61_recomputed.csv',index=False)
receipt=dict(replays=len(rows),trades=count,errors=errors,summary=pd.DataFrame(rows).groupby('label')[['net','trades','censored']].sum().to_dict('index'))
(out/'independent_sol61_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(replays=len(rows),trades=count,errors=errors[:12],error_count=len(errors)),indent=2))
