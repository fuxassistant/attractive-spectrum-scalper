"""Audit-only replays after freeze; never selects or updates a policy."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from spectrum.player.contract import Contract
from spectrum.player_v3.data import Day
from spectrum.player_v3.policy import Standardizer,PreparedDay,make_model
from spectrum.player_v3.experiment import rollout
from spectrum.player_v3.gym import risk_contract
from spectrum.core.storage import sha
from spectrum.ingest import save_json


def export(data_root,report_root,martingale_root):
    root=Path(report_root);summary=json.loads((root/'summary.json').read_text())
    if summary['status']!='COMPLETE':raise ValueError('Frozen completed comparison required')
    saved_contract=json.loads((root/'contract.json').read_text())
    if saved_contract.get('risk_contract')!=risk_contract():raise ValueError('Risk contract mismatch; historical fixed-floor runs cannot be replayed by series-risk code')
    for name,digest in saved_contract['source_hashes'].items():
        if sha(name)!=digest:raise ValueError('Frozen implementation changed before replay: '+name)
    freeze_sha=sha(root/'freeze.json');torch.set_num_threads(4)
    if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.76)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scale=Standardizer.load(root/'normalization.json');c=Contract(**saved_contract['contract'])
    val=[PreparedDay(Day(p),scale) for p in sorted(Path(data_root).glob('2026-05-*')) if '2026-05-11'<=p.name<='2026-05-15']
    test=[PreparedDay(Day(p),scale) for p in sorted(Path(data_root).glob('2026-05-*')) if p.name>='2026-05-18']
    rows=[]
    for item in summary['models']:
        model=make_model(item['variant']).to(device);model.load_state_dict(torch.load(item['checkpoint'],weights_only=True,map_location=device))
        label=f"{item['variant']}_s{item['seed']}_sample101"
        _,metric=rollout(model,test,c,rng=np.random.default_rng(101),record_dir=root/'replays'/label)
        reference=item['stochastic_evaluation'][0]
        for key in ('final_equity','trades','ruined','complete','censored'):
            if metric[key]!=reference[key]:raise ValueError('Posthoc replay differs: '+label+' '+key)
        rows.append(dict(label=label,split='evaluation',role='same frozen stochastic101 audited replay',metrics=metric))
        if item['variant']=='mixed_full' and item['seed']==61018:
            for epoch in (4,8):
                checkpoint=Path(item['checkpoint']).parent/f'epoch_{epoch:02d}.pt'
                model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=device))
                for simulation in ('greedy','sample101'):
                    label=f'mixed_full_s61018_epoch{epoch}_validation_{simulation}'
                    rng=None if simulation=='greedy' else np.random.default_rng(101)
                    _,metric=rollout(model,val,c,rng=rng,record_dir=root/'replays'/label)
                    rows.append(dict(label=label,split='validation',role='epoch evolution audit, no selection',checkpoint_sha256=sha(checkpoint),metrics=metric))
        del model;torch.cuda.empty_cache()
    if sha(root/'freeze.json')!=freeze_sha:raise ValueError('Freeze changed during posthoc audit')
    summary['martingale']=json.loads((Path(martingale_root)/'audit_summary.json').read_text())
    summary['ablations']=[dict(variant=m['variant'],seed=m['seed'],removed=key,**{k:v for k,v in value.items() if k!='days'})
                          for m in summary['models'] for key,value in m.get('ablations',{}).items()]
    summary['audit_replays']=rows
    save_json(root/'summary.json',summary)
    receipt=dict(status='COMPLETE',freeze_sha256=freeze_sha,implementation_sha256=sha(__file__),replays=rows,
        martingale_source=str(Path(martingale_root).resolve()),martingale_summary_sha256=sha(Path(martingale_root)/'audit_summary.json'))
    save_json(root/'posthoc_receipt.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/research/player_v3/memory_001')
    p.add_argument('--report',default='reports/player_v3/series_001');p.add_argument('--martingale',default='reports/player_v3/martingale_series_001')
    a=p.parse_args();print(json.dumps(export(a.data,a.report,a.martingale)['status']))
