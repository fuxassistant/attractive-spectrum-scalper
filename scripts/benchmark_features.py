"""Matched real capture workloads: serial vs two processes, exact saved parity."""
import sys,json,time,shutil,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spectrum.lab import generate
from spectrum.ingest import save_json
from spectrum.core.storage import sha
import psutil

if __name__=='__main__':
    root=Path('.tmp/feature_benchmark');root.mkdir(parents=True,exist_ok=False)
    market=root/'market';market.mkdir();captures=[]
    for name in ('capture_001','capture_005'):
        dest=market/name;dest.mkdir()
        for file in ('manifest.json','seconds.parquet'):shutil.copy2(Path('data/derived/market_v1')/name/file,dest/file)
        captures.append(dict(input={'name':name+'.bin'}))
    save_json(market/'manifest.json',dict(captures=captures))
    results=[]
    for workers in (1,2):
        out=root/f'workers_{workers}'
        result=generate(market,out,['motion','flow','spectral'],workers)
        results.append(dict(workers=workers,seconds=result['elapsed_seconds'],rows=sum(x['rows'] for x in result['captures']),
                            pids=[x['pid'] for x in result['captures']],rss_samples=[x['rss_sample'] for x in result['captures']]))
    hashes=[]
    for name in ('capture_001','capture_005'):
        a=sha(root/'workers_1'/name/'features.parquet');b=sha(root/'workers_2'/name/'features.parquet')
        assert a==b;hashes.append(dict(capture=name,sha256=a))
    result=dict(status='PASS',workloads=results,speedup=results[0]['seconds']/results[1]['seconds'],parity=hashes,
                platform=platform.platform(),python=sys.version,cpu_logical=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total,
                caveat='single matched pair while raw generation is running; includes process startup, read, FFT, Parquet write and readback; not isolated scaling benchmark')
    save_json('reports/feature_performance.json',result);print(json.dumps(result,indent=2))
