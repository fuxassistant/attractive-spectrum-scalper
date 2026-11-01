"""CLI: catalog | run NAME [--workers 2] [--recipe FILE] | evaluate DIRECTORY."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import json
import os
import shutil
from pathlib import Path
import sys
import time
from .data import ROOT,dates_for,dump,digest,prepare_day
from .signals import SIGNAL_CATALOG,VETO_CATALOG
from .execution import EXIT_PLANS,ExecutionConfig,resolve_plan
from .portfolio import SIZING_CATALOG,AccountConfig

HERE=Path(__file__).resolve().parent

def catalog():
    return dict(entries=SIGNAL_CATALOG,vetos=VETO_CATALOG,
        exits={k:asdict(v) for k,v in EXIT_PLANS.items()},sizing=SIZING_CATALOG,
        recipe_example={'entry':{'id':'bb_reentry','parameters':{}},'veto':{'id':'range_only'},
                        'exit':{'id':'breakeven','parameters':{'take_r':2.0,'be_trigger_r':1.0}},'sizing':'fixed'})

def validate(config):
    AccountConfig(**config['account']);ExecutionConfig(**config['execution'])
    ExecutionConfig(**dict(config['execution'],**config['stress']))
    if config['execution']['pending_timeout_seconds']!=2. or config['stress'].get('pending_timeout_seconds',2.)!=2.:
        raise ValueError('v1 account reserves2s per pending order; timeout is frozen')
    if config['research_hours_utc']!=['07:00','20:30']:raise ValueError('v1 feature research hours are frozen')
    fixed_gates=json.loads((HERE/'config.json').read_text(encoding='utf-8'))['candidate_gates']
    if config['candidate_gates']!=fixed_gates:raise ValueError('v1 acceptance contract is frozen; do not tune criteria with recipes')
    if config['coin_seeds']<199:raise ValueError('At least199 coin seeds required')
    recipe=config.get('recipe')
    if recipe:
        if set(recipe)!={'entry','veto','exit','sizing'}:raise ValueError('Recipe fields: entry,veto,exit,sizing')
        for module in ('entry','exit'):
            if set(recipe[module])-{'id','parameters'}:raise ValueError('Only id and parameters accepted')
        if set(recipe['veto'])!={'id'}:raise ValueError('Veto takes id only')
        if recipe['entry']['id'] not in SIGNAL_CATALOG or recipe['veto']['id'] not in VETO_CATALOG or recipe['exit']['id'] not in EXIT_PLANS or recipe['sizing'] not in SIZING_CATALOG:
            raise ValueError('Unknown module')
        params=recipe['entry'].get('parameters',{})
        allowed=SIGNAL_CATALOG[recipe['entry']['id']]['parameters']
        if set(params)-set(allowed):raise ValueError('Unknown entry knobs')
        resolve_plan(dict(asdict(EXIT_PLANS[recipe['exit']['id']]),**recipe['exit'].get('parameters',{})))

def build(root,config,workers,stress=False,only=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    dump(root/'frozen_config.json',config);dump(root/'catalog.json',catalog())
    started=time.perf_counter()
    identity={str(p.relative_to(ROOT)):digest(p) for p in sorted(HERE.glob('*')) if p.suffix in ('.py','.json') and p.name!='report.py'}
    (root/'source_snapshot').mkdir()
    for path in identity:shutil.copyfile(ROOT/path,root/'source_snapshot'/Path(path).name)
    dump(root/'manifest.json',dict(status='RUNNING',pid=os.getpid(),code_identity=identity,config_sha256=digest(root/'frozen_config.json')))
    jobs=[(date,split,index) for index,(date,split) in enumerate(dates_for(config)) if only is None or split in only]
    receipts=[]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(prepare_day,d,s,i,config,str(root),stress):d for d,s,i in jobs}
            for f in as_completed(pending):
                receipt=f.result();receipts.append(receipt)
                print(json.dumps(receipt,ensure_ascii=False),flush=True)
                dump(root/'progress.json',dict(done=len(receipts),total=len(jobs),elapsed_seconds=time.perf_counter()-started))
        if any(digest(ROOT/path)!=sha for path,sha in identity.items()):raise RuntimeError('Implementation changed during build; use a new immutable run')
        dump(root/'manifest.json',dict(status='COMPLETE',code_identity=identity,config_sha256=digest(root/'frozen_config.json'),
             elapsed_seconds=time.perf_counter()-started,days=sorted(receipts,key=lambda x:x['date'])))
    except Exception as e:
        dump(root/'manifest.json',dict(status='FAILED',error=repr(e),code_identity=identity,completed_days=len(receipts)))
        raise

def evaluation_receipt(root,summary,started):
    outputs=[p for p in root.iterdir() if p.suffix=='.parquet' or p.name in ('selected.json','admission.json')]
    dump(root/'evaluation.json',dict(status='COMPLETE',**summary,elapsed_seconds=time.perf_counter()-started,
        evaluator_sha256=digest(HERE/'evaluate.py'),artifact_sha256={p.name:digest(p) for p in sorted(outputs)}))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('catalog')
    r=sub.add_parser('run');r.add_argument('name');r.add_argument('--workers',type=int,default=2);r.add_argument('--recipe');r.add_argument('--config',default=str(HERE/'config.json'));r.add_argument('--skip-stress',action='store_true')
    e=sub.add_parser('evaluate');e.add_argument('directory');e.add_argument('--stress-directory')
    a=p.parse_args()
    if a.command=='catalog':print(json.dumps(catalog(),ensure_ascii=False,indent=2));return
    if a.command=='evaluate':
        from .evaluate import evaluate
        root=Path(a.directory);config=json.loads((root/'frozen_config.json').read_text(encoding='utf-8'))
        validate(config)
        if (root/'evaluation.json').exists():raise ValueError('Evaluation already completed; preserve immutable results')
        started=time.perf_counter();summary=evaluate(root,config,a.stress_directory)
        evaluation_receipt(root,summary,started)
        print(summary);return
    if not a.name or Path(a.name).name!=a.name or a.name in ('.','..'):raise ValueError('Use a simple new run name')
    config=json.loads(Path(a.config).read_text(encoding='utf-8'))
    if a.recipe:config['recipe']=json.loads(Path(a.recipe).read_text(encoding='utf-8'))
    validate(config)
    root=ROOT/'research_results/mechanical_lab'/a.name
    build(root,config,max(1,min(a.workers,4)))
    stress_root=None
    if not a.skip_stress:
        stress_root=root/'stress'
        build(stress_root,config,max(1,min(a.workers,4)),True,('diagnostic_test','extension'))
    from .evaluate import evaluate
    evaluation_started=time.perf_counter()
    summary=evaluate(root,config,stress_root)
    evaluation_receipt(root,summary,evaluation_started)
    print(summary,flush=True)

if __name__=='__main__':main()
