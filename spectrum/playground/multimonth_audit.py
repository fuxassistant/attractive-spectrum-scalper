"""Independent accounting and old/new August input parity for month transfer."""
from pathlib import Path
import argparse,time
import numpy as np
from spectrum.finalized import read_json
from spectrum.ingest import save_json
from spectrum.core.storage import sha
from .data import MonthDay
from .basket_audit import audit_record
from .multimonth import DEFAULT
from .multimonth_evaluate import OUTPUT


def audit_period(data,destination):
    dest=Path(destination);path=dest/'accounting_audit.json'
    identity=dict(result_sha256=sha(dest/'result.json'),evaluation_seal_sha256=sha(dest/'complete.json'),
        audit_code_sha256=sha(Path(__file__).with_name('basket_audit.py')))
    if path.exists():
        receipt=read_json(path)
        if receipt['identity']!=identity:raise ValueError('Accounting audit identity changed')
        return receipt['audit']
    result=audit_record(data,dest/'result.json')
    save_json(path,dict(identity=identity,audit=result));return result


def run(data=DEFAULT,output=OUTPUT):
    started=time.perf_counter();root=Path(output);summary=read_json(root/'summary.json')
    old=Path('data/derived/playground_august_1500');august=Path(data)/'2026-08'
    parity=[]
    for row in read_json(old/'manifest.json')['days']:
        date=row['date'];a=MonthDay(old/date);b=MonthDay(august/date,verify=True)
        for key in ('x','decision_t','bar_t','can_enter','execution_valid','execution_segment'):
            if not np.array_equal(getattr(a,key),getattr(b,key)):raise AssertionError(f'August input drift {date}: {key}')
        parity.append(date)
    original=read_json('reports/basket_search_august_1500/month/result.json')
    current=read_json(root/'2026-08/result.json')
    for key in ('result','events','days','config'):
        if original[key]!=current[key]:raise AssertionError('August replay drift: '+key)
    audits=[]
    for row in summary['months']:
        month=row['month']
        if row['result']['censored']:
            audits.append(dict(month=month,status='CENSORED_NOT_ECONOMIC_RESULT'))
            continue
        audit=audit_period(Path(data)/month,root/month)
        if abs(row['equity_max']-row['result']['peak'])>1e-7:raise AssertionError('Envelope peak mismatch: '+month)
        env=read_json(root/month/'equity_envelope.json')
        if abs(env['days'][-1]['summary']['balance_close']-row['result']['equity'])>1e-7:raise AssertionError('Envelope terminal mismatch: '+month)
        audits.append(dict(month=month,**audit));print(month,audit['status'],flush=True)
    result=dict(status='PASS',august_parity_days=parity,august_events_identical=True,accounting=audits,
        seconds=time.perf_counter()-started,
        scope='Independent per-exposed-quote accounting on uncensored runs; censored results remain unknown, not passed economic outcomes. Calendar coverage remains separate.')
    save_json(root/'audit.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=DEFAULT);p.add_argument('--output',default=OUTPUT)
    a=p.parse_args();print(run(a.data,a.output))
