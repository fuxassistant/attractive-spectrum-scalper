"""Export synchronized local audit UI, empirical intent diagnostics and sensitivities.

No fitting or checkpoint selection. Browser data are split into loadable JS
chunks so the inspector works directly from file:// without a server/CDN.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectrum.player.data import Day
from spectrum.player.model import Scaler
from spectrum.player_v2.policy import HierarchicalPlayer
from spectrum.core.storage import sha
from spectrum.ingest import save_json


def rows(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def sensitivity(model,scaler,day,accounts,indices):
    """Gradient x input wrt gate log odds, prefix-only; account reported separately."""
    model.eval();result=[];names=day.meta['feature_names'];x=scaler.transform(day.x)
    account_names=['position','realized/30','liquidation/30','drawdown/30','last_pnl/5','risk_halt','trade_count/100']
    for i in indices:
        xt=torch.tensor(x[:i+1][None],requires_grad=True);at=torch.tensor(accounts[:i+1][None],requires_grad=True)
        logits,_=model(xt,at);score=logits[0,-1,1]-logits[0,-1,0]
        gx,ga=torch.autograd.grad(score,(xt,at))
        contributions=(gx[0]*xt[0]).detach().numpy();by_feature=contributions.sum(axis=0);by_time=contributions.sum(axis=1)
        top=np.argsort(np.abs(by_feature))[-10:][::-1]
        result.append(dict(index=int(i),log_odds=float(score.detach()),features=[dict(name=names[j],signed_sensitivity=float(by_feature[j]),current_scaled_value=float(x[i,j])) for j in top],
                           recent_time_contributions=[dict(index=int(j),value=float(by_time[j])) for j in np.argsort(np.abs(by_time))[-12:][::-1]],
                           account=[dict(name=n,value=float(v)) for n,v in zip(account_names,(ga[0,-1]*at[0,-1]).detach().numpy())],
                           caveat='Local gradient x standardized input around actual prefix. Not SHAP, not causal attribution, not a profit forecast. Flags may be sensitive but are not physically continuous interventions.'))
    return result


def build(root='.',run='run_001'):
    root=Path(root).resolve();src=root/'data/research/player_v2'/run;out=root/'reports/player_v2'/run;chunks=out/'data';chunks.mkdir(exist_ok=True)
    protocol=json.loads((src/'protocol.json').read_text());freezes=json.loads((src/'all_frozen_before_evaluation.json').read_text());summary=json.loads((src/'summary.json').read_text())
    days={p.name:Day(p,verify=False) for w in ('week1','week2') for p in sorted((root/'data/derived/player_v1'/w).glob('2026*'))}
    bars=pd.concat([pq.read_table(p).to_pandas() for p in sorted((root/'data/derived/market_v1').glob('capture_*/bars.parquet'))]).set_index('t_ns')
    if bars.index.duplicated().any():raise ValueError('Ambiguous bar lineage')
    scaler=Scaler.load(src/'scaler.npz');torch.set_num_threads(4)
    manifest=[];alltr=[];outcome=[];allhist=[];attributions=[]
    feature_names=['spread_ticks','imbalance_l1','displacement_30s','efficiency_30s','trade_buy_30s','trade_sell_30s','fft_mid_change_128s_power','fft_mid_change_128s_entropy']
    for freeze in freezes:
        seed=freeze['seed'];sd=src/f'seed_{seed}';model=HierarchicalPlayer();model.load_state_dict(torch.load(sd/'selected.pt',weights_only=True,map_location='cpu'))
        hist=pd.read_csv(sd/'epochs.csv');allhist.append(hist)
        for sim in sorted(sd.iterdir()):
            if not sim.is_dir():continue
            for dp in sorted(sim.glob('2026*')):
                t=rows(dp/'trades.csv')
                if len(t):t['seed']=seed;t['simulation']=sim.name;t['date']=dp.name;alltr.append(t)
            # Include all validation epochs for actual policy evolution plus selected daily replays.
            if sim.name not in ('selected_train','selected_test','untrained_test','sampled_test_101') and not sim.name.startswith('validation_epoch_'):continue
            checkpoint=('untrained.pt' if sim.name=='untrained_test' or sim.name=='validation_epoch_000' else
                        sim.name.replace('validation_','')+'.pt' if sim.name.startswith('validation_epoch_') else 'selected.pt')
            model.load_state_dict(torch.load(sd/checkpoint,weights_only=True,map_location='cpu'));model.eval()
            for dp in sorted(sim.glob('2026*')):
                day=days[dp.name];dec=pd.read_parquet(dp/'decisions.parquet');h=np.load(dp/'hierarchy.npz');p=np.load(dp/'policy_trace.npz')
                tr=rows(dp/'trades.csv');m=json.loads((dp/'metrics.json').read_text());bar=bars.reindex(day.bar_t)
                device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
                with torch.no_grad():
                    logits,_=model(torch.from_numpy(scaler.transform(day.x))[None].to(device),torch.from_numpy(p['account'])[None].to(device))
                    raw_direction=logits[0,:,2:4].softmax(-1).cpu().numpy();raw_plans=logits[0,:,4:].reshape(-1,2,3).softmax(-1).cpu().numpy()
                    reproduced_intent=logits[0,:,:2].softmax(-1)[:,1].cpu().numpy()
                if not np.allclose(reproduced_intent,h['intent'],atol=1e-6,rtol=1e-5):raise ValueError(f'UI checkpoint mismatch {seed}/{sim.name}/{dp.name}: max {np.max(np.abs(reproduced_intent-h["intent"]))}')
                model.cpu()
                selected=sim.name in ('selected_train','selected_test')
                attr=[]
                if selected:
                    eligible=p['mask'][:,1:].any(axis=1);candidates=np.flatnonzero(eligible)
                    ids=[]
                    if len(candidates):
                        ids=[int(candidates[np.argmax(h['intent'][candidates])]),int(candidates[np.argmax(np.diff(h['intent'],prepend=h['intent'][0])[candidates])])]
                        ids+=np.random.default_rng(61015+seed+int(dp.name[-2:])).choice(candidates,min(2,len(candidates)),replace=False).tolist()
                        if len(tr):ids+=[int(tr.iloc[0].decision_index)]
                    attr=sensitivity(model,scaler,day,p['account'],sorted(set(ids)))
                    attributions.append(dict(seed=seed,simulation=sim.name,date=dp.name,points=attr))
                ix=[day.meta['feature_names'].index(n) for n in feature_names]
                trades=[]
                for row in tr.to_dict('records'):
                    row['entry_ns']=str(row['entry_ns']);row['exit_ns']=str(row['exit_ns'])
                    row['entry_ms']=int(row['entry_ns'])//1_000_000;row['exit_ms']=int(row['exit_ns'])//1_000_000
                    trades.append(row)
                payload=dict(seed=seed,simulation=sim.name,date=dp.name,selected_epoch=freeze['selected_epoch'],metrics=m,
                             t=(day.decision_t//1_000_000).tolist(),proof_ns=[str(t) for t in day.decision_t],
                             candles=bar[['o','h','l','c']].where(bar.valid,axis=0).to_numpy().round(4).tolist(),
                             intent=h['intent'].round(6).tolist(),gate=h['gate'].round(6).tolist(),direction=h['direction'].round(6).tolist(),plans=h['plans'].round(6).tolist(),
                             raw_direction=raw_direction.round(6).tolist(),raw_plans=raw_plans.round(6).tolist(),plan_legal=p['mask'][:,1:].reshape(-1,2,3).tolist(),
                             blocked=h['blocked'].tolist(),action=dec.action.tolist(),equity=dec.equity_before.round(5).tolist(),
                             value=p['value'].round(5).tolist(),feature_names=feature_names,features=scaler.transform(day.x)[:,ix].round(4).tolist(),
                             trades=trades,attribution=attr,london_ms=int(day.meta['london_ns'])//1_000_000,ny_ms=int(day.meta['ny_close_ns'])//1_000_000)
                # JSON null for unsupported candles, never NaN JavaScript arithmetic.
                encoded=json.dumps(payload,ensure_ascii=False,allow_nan=True,separators=(',',':')).replace('NaN','null')
                key=f'{seed}__{sim.name}__{dp.name}';file=f'data/{key}.js';(out/file).write_text('window.loadDataset('+encoded+');',encoding='utf-8')
                manifest.append(dict(key=key,seed=seed,simulation=sim.name,date=dp.name,file=file))
        print(f'Inspector exported seed {seed}',flush=True)
    ledger=pd.concat(alltr,ignore_index=True);ledger.to_csv(out/'all_trades.csv',index=False)
    # Fixed bins, include censored count explicitly; no exclusion-based policy claim.
    for (seed,sim),group in ledger.groupby(['seed','simulation']):
        if sim not in ('selected_test','sampled_test_101','sampled_test_202','sampled_test_303'):continue
        for lo,hi in zip([0,.2,.4,.6,.8],[.2,.4,.6,.8,1.000001]):
            g=group[(group.participation_intent>=lo)&(group.participation_intent<hi)];clean=g[g.censored==0]
            outcome.append(dict(seed=int(seed),simulation=sim,low=lo,high=min(hi,1),trades=len(g),censored=int(g.censored.sum()),days=int(g.date.nunique()),
                                observed_net=float(g.net.sum()),clean_count=len(clean),clean_mean=float(clean.net.mean()) if len(clean) else None,
                                clean_win_rate=float((clean.net>0).mean()) if len(clean) else None,
                                note='Descriptive selected/explored actions; not calibrated opportunity probability or causal benefit of thresholding.'))
    save_json(out/'intent_outcomes.json',outcome);save_json(out/'attributions.json',attributions)
    history=pd.concat(allhist,ignore_index=True);history.to_csv(out/'epochs.csv',index=False)
    chart(out,history,summary,outcome)
    meta=dict(datasets=manifest,freezes=freezes,summary=summary,outcomes=outcome,history=json.loads(history.to_json(orient='records')),
              protocol=protocol,feature_names=feature_names)
    (out/'manifest.js').write_text('window.AUDIT='+json.dumps(meta,ensure_ascii=False,allow_nan=False,separators=(',',':'))+';',encoding='utf-8')
    template=(root/'scripts/player_v2_ui.html').read_text(encoding='utf-8');(out/'index.html').write_text(template,encoding='utf-8')
    save_json(out/'export_receipt.json',dict(datasets=len(manifest),trades=len(ledger),attribution_points=sum(len(x['points']) for x in attributions),files={p.name:sha(p) for p in out.glob('*.png')},
                                           method='Source candles aligned by bar_t; price unknowns null; model series use decision proof clocks; actual trace of selected epoch, no synthetic signal.'))


def chart(out,h,summary,bins):
    plt.rcParams.update({'figure.dpi':130,'text.parse_math':False,'axes.grid':True,'grid.alpha':.2})
    fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for seed,g in h.groupby('seed'):
        ax[0,0].plot(g.epoch,g.validation_intent,label=str(seed));ax[0,1].plot(g.epoch,g.validation_net,label=str(seed))
        ax[1,0].plot(g.epoch,g.train_net,label=str(seed));ax[1,1].plot(g.epoch,g.clean_days,label=str(seed))
    for a,title in zip(ax.flat,['Validation mean participation intent','Validation greedy net USD','Sampled training observed net USD (all 4 days)','Complete training days admitted to gradient']):a.set_title(title);a.set_xlabel('Epoch');a.legend()
    ax[0,0].axhline(.5,color='black',ls=':');fig.savefig(out/'epochs.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for r in summary:
        if r['policy'] not in ('selected_test','untrained_test'):continue
        ax.plot(range(6),np.r_[1000,1000+np.cumsum([d['net'] for d in r['days']])],label=f'{r["seed"]} {r["policy"]} ({r["censored"]} censored)',ls='-' if r['policy']=='selected_test' else '--')
    ax.axhline(1000,color='grey',label='WAIT');ax.set_xticks(range(6),['Start','May11','May12','May13','May14','May15']);ax.set_ylabel('Day-end observed equity USD');ax.legend(fontsize=8)
    ax.set_title('Development week comparison; censored paths retained; no fresh OOS claim');fig.savefig(out/'equity.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    b=pd.DataFrame(bins);b=b[b.simulation.str.startswith('sampled_test')]
    for seed,g in b.groupby('seed'):
        means=[];xs=[]
        for lo,gg in g.groupby('low'):
            clean=gg[gg.clean_count>0]
            if len(clean):xs.append(lo+.1);means.append(np.average(clean.clean_mean,weights=clean.clean_count))
        ax.plot(xs,means,'o-',label=f'{seed}: three fixed stochastic replays')
    ax.axhline(0,color='grey');ax.set_xlim(0,1);ax.set_xlabel('Participation intention bin');ax.set_ylabel('Mean net per uncensored observed trade USD')
    ax.set_title('Descriptive action outcomes; sparse bins / repeated days are not independent proof');ax.legend(fontsize=8);fig.savefig(out/'intent_outcomes.png');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--run',default='run_001');a=p.parse_args();build(a.root,a.run)
