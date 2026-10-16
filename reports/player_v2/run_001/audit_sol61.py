"""Independent read-only numerical audit; writes receipts beside this script."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from spectrum.player.data import Day
from spectrum.player.model import Scaler
from spectrum.player_v2.policy import distribution,distribution_numpy,greedy,HierarchicalPlayer
OUT=Path(__file__).parent;SRC=ROOT/'data/research/player_v2/run_001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
day_cache={};receipt={};freezes=json.loads((SRC/'all_frozen_before_evaluation.json').read_text())
protocol=json.loads((SRC/'protocol.json').read_text())
receipt['hashes_match']=all(sha(ROOT/p)==h and sha(SRC/'source_snapshot'/p)==h for p,h in protocol['code_sha256'].items())
receipt['selected_hash_match']=all(sha(SRC/f"seed_{f['seed']}"/'selected.pt')==f['selected_sha256'] for f in freezes)
# All 64 downstream masks and broad logits, including forced WAIT.
rng=np.random.default_rng(4721);maxdiff=0.;maxnorm=0.;illegal=0
for bits in range(64):
 mask=np.r_[True,[bool(bits&(1<<j)) for j in range(6)]]
 for k in range(12):
  logits=rng.normal(0,8,10);n=distribution_numpy(logits,mask)
  t=distribution(torch.tensor(logits)[None],torch.tensor(mask)[None])
  maxdiff=max(maxdiff,max(np.max(np.abs(np.asarray(a)-b.detach().numpy()[0])) for a,b in zip(n,t)))
  maxnorm=max(maxnorm,abs(n[0].sum()-1));illegal+=int(n[0][~mask].sum()!=0)
  assert mask[greedy(n[2],n[3],n[4])]
receipt['distribution']={'cases':768,'numpy_torch_max_abs':maxdiff,'sum_error':maxnorm,'illegal_nonzero_cases':illegal}
ledger=pd.read_csv(OUT/'all_trades.csv');errors=[];decision_checks=0;max_net_error=0.;max_metric_error=0.;gradient_rows=[]
for seed in [61015,61016,61017]:
 sd=SRC/f'seed_{seed}';hist=pd.read_csv(sd/'epochs.csv');f=next(f for f in freezes if f['seed']==seed)
 valid=hist[hist.validation_censored==0];expected=int(valid.loc[valid.validation_net.idxmax(),'epoch'])
 assert expected==f['selected_epoch']
 for ep in hist.epoch:
  ms=[json.loads(p.read_text()) for p in (sd/f'train_epoch_{ep:03d}').glob('*/metrics.json')]
  row=hist[hist.epoch==ep].iloc[0]
  assert sum(m['complete'] for m in ms)==row.clean_days
  gradient_rows.append({'seed':seed,'epoch':int(ep),'clean':int(row.clean_days),'updated':bool(row.updated)})
 for p in sd.glob('*/*/metrics.json'):
  dp=p.parent;date=dp.name;week='week1' if date<'2026-05-11' else 'week2'
  if date not in day_cache:day_cache[date]=Day(ROOT/'data/derived/player_v1'/week/date,verify=False)
  day=day_cache[date]
  m=json.loads(p.read_text());tr=ledger[(ledger.seed==seed)&(ledger.simulation==dp.parent.name)&(ledger.date==date)]
  trace=np.load(dp/'policy_trace.npz');h=np.load(dp/'hierarchy.npz');dec=pd.read_parquet(dp/'decisions.parquet')
  assert np.all(trace['mask'][np.arange(len(dec)),dec.action.to_numpy(int)])
  assert np.allclose(trace['probability'].sum(-1),1,atol=1e-6)
  assert np.all(trace['probability'][~trace['mask']]==0)
  assert np.all(day.quote_t[day.cuts[day.cuts>0]-1]<=day.decision_t[day.cuts>0])
  assert abs(m['net']-m['reward_sum'])<1e-4
  if not m['open_terminal']:max_metric_error=max(max_metric_error,abs(m['net']-tr.net.sum()))
  for r in tr.itertuples():
   i=int(r.entry_index);j=int(r.exit_index);di=int(r.decision_index)
   assert int(day.quote_t[i])==r.entry_ns and int(day.quote_t[j])==r.exit_ns
   assert r.entry_ns>int(day.decision_t[di])+100_000_000
   assert r.entry_ns<=int(day.decision_t[di])+2_100_000_000
   assert r.entry_nano==(day.ask[i] if r.side>0 else day.bid[i])
   assert r.exit_nano==(day.bid[j] if r.side>0 else day.ask[j])
   err=abs(r.net-(r.side*(int(day.bid[j] if r.side>0 else day.ask[j])-int(day.ask[i] if r.side>0 else day.bid[i]))/1e9-.3))
   max_net_error=max(max_net_error,err)
   assert int(dec.action.iloc[di])==1+(0 if r.side>0 else 3)+r.plan
  decision_checks+=len(dec)
receipt['ledger']={'trades':len(ledger),'decision_checks':decision_checks,'max_trade_net_error':max_net_error,'max_closed_episode_metric_error':max_metric_error}
receipt['gradient']={'episodes':len(gradient_rows)*4,'admitted_complete':sum(r['clean'] for r in gradient_rows),'zero_update_epochs':sum(not r['updated'] for r in gradient_rows)}
receipt['policy_results']=pd.DataFrame(json.loads((SRC/'summary.json').read_text())).drop(columns='days').to_dict('records')
receipt['epochs']=[]
for seed in [61015,61016,61017]:
 h=pd.read_csv(SRC/f'seed_{seed}'/'epochs.csv')
 f=next(f for f in freezes if f['seed']==seed)
 receipt['epochs'].append({'seed':seed,'selected_epoch':f['selected_epoch'],'epoch1_intent':float(h.iloc[0].validation_intent),'epoch60_intent':float(h.iloc[-1].validation_intent),'first10_train_net_mean':float(h.iloc[:10].train_net.mean()),'last10_train_net_mean':float(h.iloc[-10:].train_net.mean()),'validation_positive_epochs':int((h.validation_net>0).sum())})
sample=ledger[ledger.simulation.str.startswith('sampled_test')];clean=sample[sample.censored==0]
receipt['sampled_intent']=[]
for seed,g in clean.groupby('seed'):
 allg=sample[sample.seed==seed]
 receipt['sampled_intent'].append({'seed':int(seed),'total_trades':len(allg),'censored':int(allg.censored.sum()),'clean_trades':len(g),'mean_net':float(g.net.mean()),'win_rate':float((g.net>0).mean()),'intent_min':float(allg.participation_intent.min()),'intent_max':float(allg.participation_intent.max()),'pearson_intent_net':float(g.participation_intent.corr(g.net))})
attrs=json.loads((OUT/'attributions.json').read_text());features=[]
for a in attrs:
 for p in a['points']:
  for f in p['features']:features.append({'seed':a['seed'],'name':f['name'],'abs':abs(f['signed_sensitivity']),'signed':f['signed_sensitivity']})
f=pd.DataFrame(features);receipt['sensitivities_top']=f.groupby('name').agg(appearances=('abs','size'),mean_abs=('abs','mean'),mean_signed=('signed','mean')).sort_values('mean_abs',ascending=False).head(12).reset_index().to_dict('records')
torch.set_num_threads(4)
scaler=Scaler.load(SRC/'scaler.npz');refit=Scaler().fit([day_cache[f'2026-05-{i:02d}'] for i in range(4,8)])
model=HierarchicalPlayer();model.load_state_dict(torch.load(SRC/'seed_61017/selected.pt',weights_only=True));model.eval()
day=day_cache['2026-05-14'];x=torch.tensor(scaler.transform(day.x))[None];account=torch.zeros(1,len(day.x),7);i=700
with torch.no_grad():
 y=model(x,account)[0][0,i];prefix=model(x[:,:i+1],account[:,:i+1])[0][0,i]
 changed=x.clone();changed[:,i+1:]=7;future=model(changed,account)[0][0,i]
receipt['causality_probe']={'index':i,'prefix_full_max':float((prefix-y).abs().max()),'future_perturbation_max':float((future-y).abs().max()),'scaler_train_refit_max':float(max(np.max(np.abs(scaler.center-refit.center)),np.max(np.abs(scaler.scale-refit.scale))))}
(OUT/'audit_sol61_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
