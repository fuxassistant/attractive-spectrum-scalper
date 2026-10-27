"""Run the fixed-week, full-quote first-crossing oracle experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import math
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from spectrum.crossings import open_bank
from spectrum.canon import load_canon
from spectrum.future.crossings import future_crossings
from spectrum.economics import future_crossing_marks
from spectrum.leakage import export_future_values
from experiments.topn_crossings.oracle import make_windows, prepare, outcome_representatives, summarize_window, write_labels


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def verify_samples(rows, bank, tape, coverage_end, seed=23):
    """Independent public API checks of random representative candidates."""
    if not len(rows):
        return 0
    picks = rows.sample(min(24, len(rows)), random_state=seed)
    for row in picks.itertuples():
        # One ns before the half-open end; no first crossing may be skipped.
        y = future_crossings(bank, tape, [row.entry_ns], row.exit_limit_ns-row.entry_ns-1, coverage_end)
        marks = future_crossing_marks(y)
        raw = pd.DataFrame(export_future_values(marks), columns=marks.columns)
        value = raw[(raw.scale == row.scale) & (raw.role == row.role)].iloc[0]
        if value.future_event_id != row.event_id or value.future_status != 'CROSSED':
            raise ValueError('First-crossing parity failure')
        actual = value.future_long_usd if row.side == 'LONG' else value.future_short_usd
        if not np.isclose(actual, row.net, atol=1e-9, rtol=0):
            raise ValueError('Economics parity failure')
    return len(picks)


def run(config_path, output, modes=None, days=None):
    config = json.loads(Path(config_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if modes:
        config['window_modes'] = modes
    if days:
        config['days'] = days
    receipt = dict(status='RUNNING', created_utc=datetime.now(timezone.utc).isoformat(),
                   config=config, config_sha256=sha(config_path), python=sys.version,
                   platform=platform.platform(), code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   source_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                   fee_usd_per_oz=load_canon().value('fee_roundtrip_usd_per_oz'),
                   source_manifests=[], outputs={}, public_api_parity_samples=0,
                   training_performed=False, trading_admission=False)
    for name in receipt['source_code_sha256']:
        snapshot=output/'source_snapshot'/name
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,snapshot)
    shutil.copy2(config_path,output/'source_snapshot/config.json')
    def save():
        (output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
    save()
    baseline = {r['path']:r['sha256'] for r in map(json.loads,(ROOT/'baselines/0.3.0/data-files.jsonl').read_text().splitlines())}
    summaries=[]; curves=[]; tops=[]
    begin=time.perf_counter()
    try:
        for day in config['days']:
            source=ROOT/config['bank_root']/day
            for name in ('manifest.json','events.parquet','tape.parquet'):
                path=source/name
                if sha(path)!=baseline[path.relative_to(ROOT/'data').as_posix()]:
                    raise ValueError('Source drift from baseline: '+str(path))
            bank,tape,manifest=open_bank(source)
            receipt['source_manifests'].append(dict(day=day,sha256=sha(source/'manifest.json'),source_id=manifest['source_id'],
                events=len(bank),origins=len(tape['t']),files=manifest['files']))
            valid,ends=prepare(tape,manifest['coverage_end_ns'])
            print(json.dumps(dict(day=day,origins=len(tape['t']),events=len(bank))),flush=True)
            for mode in config['window_modes']:
                windows=make_windows(tape['t'],bank,manifest['coverage_end_ns'],mode)
                rows,stats=outcome_representatives(bank,tape,windows,valid,ends)
                if len(rows):
                    receipt['public_api_parity_samples']+=verify_samples(rows,bank,tape,manifest['coverage_end_ns'])
                    if rows.duplicated(['window','event_id']).any():
                        raise ValueError('Duplicate outcome')
                    if not ((rows.entry_ns<rows.exit_ns)&(rows.exit_ns<rows.exit_limit_ns)&(rows.entry_ns>=rows.window_start)&(rows.entry_ns<rows.window_end)).all():
                        raise ValueError('Window/time violation')
                partition=output/'labels'/day
                partition.mkdir(parents=True,exist_ok=True)
                path=partition/(mode+'.parquet')
                write_labels(rows,path)
                receipt['outputs'][path.relative_to(output).as_posix()]=sha(path)
                for window in stats.itertuples(index=False):
                    subset=rows[rows.window==window.window] if len(rows) else rows
                    result,curve=summarize_window(subset,config['max_rank'])
                    result.update(window._asdict());result.update(day=day,mode=mode)
                    result['duration_s']=(window.end-window.start)/1e9
                    summaries.append(result)
                    curves.extend(dict(day=day,mode=mode,window=window.window,**v) for v in curve)
                if len(rows):
                    top=rows[(rows['rank']>0)&(rows['rank']<=config['max_rank'])].copy()
                    top['day']=day;top['mode']=mode;tops.append(top)
                print(json.dumps(dict(day=day,mode=mode,windows=len(stats),outcomes=len(rows),seconds=round(time.perf_counter()-begin,2))),flush=True)
        summary=pd.DataFrame(summaries)
        for name,frame in [('windows',summary),('curves',pd.DataFrame(curves)),('top',pd.concat(tops,ignore_index=True) if tops else pd.DataFrame())]:
            path=output/(name+'.parquet');write_labels(frame,path)
            receipt['outputs'][path.name]=sha(path)
        aggregate=[]
        for mode,g in summary.groupby('mode',sort=False):
            active=g[(g.origins>0)&~g.partial]
            aggregate.append(dict(mode=mode,windows=len(g),full_active_windows=len(active),
                zero_positive_windows=int((active.positive_outcomes==0).sum()),
                duration_median_s=float(active.duration_s.median()),duration_p10_s=float(active.duration_s.quantile(.1)),duration_p90_s=float(active.duration_s.quantile(.9)),
                positive_outcomes_median=float(active.positive_outcomes.median()),n80_median=float(active.n80.median()),
                top1_share_median=float(active.top1_share.median()),top10_share_median=float(active.top10_share.median()),
                top10_flicker_mean=float(active.top10_flicker_fraction.mean()),
                top10_same_entry_mean=float(active.top10_same_entry_fraction.mean()),
                top10_same_exit_mean=float(active.top10_same_exit_fraction.mean()),
                ready_query_fraction=float(1-active.not_ready.sum()/max(1,active.queries.sum())),
                window_censored_fraction=float(active.window_censored.sum()/max(1,(active.queries-active.not_ready).sum())),
                support_end_fraction=float(active.support_end.sum()/max(1,(active.queries-active.not_ready).sum()))))
        path=output/'summary.json';path.write_text(json.dumps(finite_json(aggregate),indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
        receipt['outputs'][path.name]=sha(path)
        receipt.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-begin)
        save()
        print(json.dumps(dict(status=receipt['status'],parity_samples=receipt['public_api_parity_samples'])),flush=True)
    except Exception as exc:
        receipt.update(status='FAILED',error=repr(exc));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--output',required=True)
    parser.add_argument('--modes',nargs='+')
    parser.add_argument('--days',nargs='+')
    args=parser.parse_args()
    run(args.config,args.output,args.modes,args.days)
