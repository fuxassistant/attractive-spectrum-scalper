"""Evaluate the August-frozen basket unchanged on each available month."""
from pathlib import Path
from dataclasses import asdict
import argparse,time
from spectrum.finalized import read_json
from spectrum.ingest import save_json
from spectrum.core.storage import sha
from .basket import BasketConfig
from .basket_search import evaluate_record
from .equity_envelope import attach
from .multimonth import DEFAULT
from .data import MonthDay

OUTPUT='reports/basket_multimonth_v1'


def evaluate_period(path,dates,dest,recipe_path,config):
    path=Path(path);dest=Path(dest)
    recipe=read_json(recipe_path)
    if asdict(config)!=recipe['config']:raise ValueError('Configuration differs from frozen recipe')
    if sha(Path(__file__).with_name('basket.py'))!=recipe['source']['basket.py']:raise ValueError('Numerical kernel drift')
    identity=dict(day_manifests={date:sha(path/date/'manifest.json') for date in dates},recipe_sha256=sha(recipe_path),
        evaluator_sha256=sha(Path(__file__).with_name('basket_search.py')),
        envelope_sha256=sha(Path(__file__).with_name('equity_envelope.py')))
    if (dest/'complete.json').exists():
        seal=read_json(dest/'complete.json')
        if seal['identity']!=identity:raise ValueError('Completed evaluation inputs/code changed')
        for name,digest in seal['files'].items():
            if sha(dest/name)!=digest:raise ValueError('Completed evaluation file changed: '+name)
        record=read_json(dest/'result.json');env=read_json(dest/'equity_envelope.json')
    else:
        record=evaluate_record(path,dates,config,dest)
        env=attach(path,dest/'result.json')
        save_json(dest/'complete.json',dict(identity=identity,files={name:sha(dest/name) for name in ('result.json','equity_envelope.json')}))
    return record,env


def evaluate(data=DEFAULT,output=OUTPUT):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    recipe_path=Path('reports/basket_search_august_1500/frozen.json');recipe=read_json(recipe_path)
    if sha(Path(__file__).with_name('basket.py'))!=recipe['source']['basket.py']:raise ValueError('Numerical kernel changed after August freeze')
    catalog=read_json(Path(data)/'catalog.json');config=BasketConfig(**recipe['config'])
    protocol=dict(recipe_sha256=sha(recipe_path),config=recipe['config'],data_catalog_sha256=sha(Path(data)/'catalog.json'),
        implementation={name:sha(Path(__file__).with_name(name)) for name in
            ('multimonth_evaluate.py','basket_search.py','basket.py','data.py','equity_envelope.py')},
        initial_capital=1500,kernel_sha256=recipe['source']['basket.py'],selection='No new fitting or selection. Each month independently funded; no joining over missing calendar days.',
        exposure='August is selection data. Other months are historical transfer checks, not guaranteed pristine unseen data.',
        metric='Minute and daily min/max balance and liquidation equity using all exposed quotes; incomplete/censored periods remain explicit')
    if (root/'protocol.json').exists() and read_json(root/'protocol.json')!=protocol:
        raise ValueError('Evaluation protocol drift; use a new output directory')
    save_json(root/'protocol.json',protocol)
    rows=[];failures=[]
    for month in catalog['months']:
        if month['status']!='COMPLETE':failures.append(dict(month=month['month'],reason='Input preparation not complete'));continue
        path=Path(month['path']);m=read_json(path/'manifest.json');dest=root/month['month']
        try:
            record,env=evaluate_period(path,[d['date'] for d in m['days']],dest,recipe_path,config)
            state=record['result']
            censor=None
            if state['censored']:
                date=record['days'][-1]['date'];last=record['events'][-1];daily=env['days'][-1]
                reason='OPEN_POSITION_AT_END_OF_TAPE'
                if int(last['kind'])==7:
                    d=MonthDay(path/date)
                    reason='INVALID_BOOK_PUBLICATION' if not d.valid[int(last['quote_index'])] else 'SUPPORT_SEGMENT_TRANSITION'
                censor=dict(date=date,reason=reason,observed_until_ns=daily['summary']['observed_until_ns'],
                    last_known_equity=state['equity'],closed_baskets=state['closed_baskets'])
            row=dict(month=month['month'],input_days=len(m['days']),replayed_days=len(record['days']),coverage=m['coverage'],
                missing_weekdays=m['missing_weekdays'],result=state,censor_detail=censor,
                equity_min=min(d['summary']['equity_min'] for d in env['days']),equity_max=max(d['summary']['equity_max'] for d in env['days']),
                balance_min=min(d['summary']['balance_min'] for d in env['days']),balance_max=max(d['summary']['balance_max'] for d in env['days']),
                daily=[dict(date=d['date'],**d['summary']) for d in env['days']],
                invalid_activity_publications=sum(d['diagnostics']['invalid_publications'] for d in m['days']),
                quality_days=[dict(date=d['date'],**{key:d['diagnostics'][key] for key in
                    ('symbol','observed_activity_minutes','expected_activity_minutes','legal_minute_origins','invalid_publications','max_pause_seconds')})
                    for d in m['days'] if d['diagnostics']['observed_activity_minutes']<d['diagnostics']['expected_activity_minutes']
                    or d['diagnostics']['invalid_publications'] or (d['diagnostics']['max_pause_seconds'] or 0)>60],
                zero_entry_days=[d['date'] for d in m['days'] if not d['diagnostics']['legal_minute_origins']])
            rows.append(row);print(dict(month=month['month'],net=state['net'],alive=state['alive'],censored=state['censored'],coverage=m['coverage']),flush=True)
        except Exception as exc:
            failures.append(dict(month=month['month'],reason=str(exc)));print(f"ERROR {month['month']}: {exc}",flush=True)
        save_json(root/'progress.json',dict(status='RUNNING',months=rows,failures=failures))
    result=dict(status='COMPLETE' if not failures else 'PARTIAL',months=rows,failures=failures,seconds=time.perf_counter()-started)
    save_json(root/'summary.json',result);save_json(root/'progress.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=DEFAULT);p.add_argument('--output',default=OUTPUT)
    a=p.parse_args();evaluate(a.data,a.output)
