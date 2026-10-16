"""Pipeline already completed captures while the remaining books reconstruct."""
import sys,time,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectrum.lab import generate_capture
from spectrum.core.storage import sha
from spectrum.ingest import save_json
import psutil

if __name__=='__main__':
    market=Path('data/derived/market_v1');output=Path('data/derived/features_v1');output.mkdir(parents=True,exist_ok=False)
    raw=json.loads(Path('data/derived/raw_v1/manifest.json').read_text());names=[Path(e['name']).stem for e in raw['partitions']]
    submitted=set();jobs={};results=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        while len(results)<len(names):
            for name in names:
                if name not in submitted and (market/name/'manifest.json').exists():
                    jobs[pool.submit(generate_capture,market/name,output/name,['motion','flow','spectral'])]=name;submitted.add(name)
            done,_=wait(jobs,timeout=1,return_when=FIRST_COMPLETED) if jobs else (set(),None)
            for f in done:
                name=jobs.pop(f);r=f.result();r['capture']=name;results.append(r)
                print(json.dumps(dict(capture=name,rows=r['rows'],seconds=r['elapsed_seconds'])),flush=True)
            if not jobs:
                if len(submitted)<len(names):
                    job=json.loads((market/'job.json').read_text())
                    if not psutil.pid_exists(job['pid']):raise RuntimeError('Raw generation stopped before remaining captures completed')
                time.sleep(1)
    deadline=time.monotonic()+30
    while not (market/'manifest.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('Missing final market manifest')
        time.sleep(1)
    results.sort(key=lambda r:r['capture'])
    result=dict(status='COMPLETE',captures=results,elapsed_seconds=time.perf_counter()-started,workers=3,families=['motion','flow','spectral'],
                market_manifest_sha256=sha(market/'manifest.json'),
                timing_scope='pipelined elapsed includes waiting for unfinished raw captures; see per-capture times and matched benchmark for compute performance')
    save_json(output/'manifest.json',result);print('FEATURE_DATASET_COMPLETE',flush=True)
