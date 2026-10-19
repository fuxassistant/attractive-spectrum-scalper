"""Loopback-only interactive playground; one reproducible experiment at a time."""
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import re
import threading
import numpy as np
from spectrum.finalized import read_json
from spectrum.player_v3.martingale import MartingaleConfig
from .runner import run,freeze,DEFAULT_DATA,DEFAULT_OUTPUT
from .data import MonthDay


def wire(x):
    if isinstance(x,dict):return {k:wire(v) for k,v in x.items()}
    if isinstance(x,list):return [wire(v) for v in x]
    if isinstance(x,int) and abs(x)>2**53-1:return str(x)
    return x


def serve(data=DEFAULT_DATA,output=DEFAULT_OUTPUT,port=8794):
    data=Path(data).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=read_json(data/'manifest.json')
    if manifest['status']!='COMPLETE':raise ValueError('Month not prepared')
    pool=ThreadPoolExecutor(max_workers=1);lock=threading.Lock();state=dict(status='idle',id=None,error=None)
    def config(payload):
        allowed={'mode','lookback_minutes','threshold_ticks','plan','multiplier','cap_oz'}
        if set(payload)-allowed:raise ValueError('Unknown configuration field')
        c=MartingaleConfig(**payload)
        if not 1<=c.lookback_minutes<=120 or not 1<=c.multiplier<=4 or not 1<=c.cap_oz<=64 or not 0<=c.threshold_ticks<=100:raise ValueError('Parameter outside playground bounds')
        if not all(np.isfinite(v) for v in (c.multiplier,c.cap_oz,c.threshold_ticks)):raise ValueError('Nonfinite parameter')
        return c
    def task(cfg,split,control):
        try:
            frozen=freeze(data,output,cfg) if split=='holdout' else None
            result=run(data,output,cfg,split,control,frozen)
            with lock:state.update(status='complete',id=result['id'])
        except Exception as exc:
            with lock:state.update(status='failed',error=str(exc))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_json(self,obj,status=200):
            payload=json.dumps(wire(obj),allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            try:
                u=urlparse(self.path);params=parse_qs(u.query)
                if u.path in ('/','/index.html'):
                    payload=Path(__file__).with_name('index.html').read_bytes()
                    self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload);return
                if u.path=='/api/status':
                    with lock:s=dict(state)
                    s.update(month=manifest['month'],days=[r['date'] for r in manifest['days']],split=manifest['split'],initial_capital=1500.,
                        runs=[dict(id=p.parent.name) for p in sorted((output/'runs').glob('*/result.json'))])
                    return self.send_json(s)
                if u.path=='/api/result':
                    ident=params.get('id',[''])[0]
                    if not re.fullmatch(r'[A-Za-z0-9_-]+',ident):raise ValueError('Invalid run id')
                    return self.send_json(read_json(output/'runs'/ident/'result.json'))
                if u.path=='/api/market':
                    date=params.get('date',[''])[0]
                    if date not in [r['date'] for r in manifest['days']]:raise ValueError('Unknown date')
                    day=MonthDay(data/date)
                    price=[float(v) if valid and v else None for v,valid in zip(day.decision_mid,day.x[:,8]>.5)]
                    return self.send_json(dict(date=date,t=[str(t) for t in day.decision_t],price=price,can_enter=day.can_enter.tolist(),
                        london_ns=day.meta['london_ns'],ny_close_ns=day.meta['ny_close_ns'],diagnostics=read_json(data/date/'diagnostics.json')))
                self.send_json(dict(error='Not found'),404)
            except Exception as exc:self.send_json(dict(error=str(exc)),400)
        def do_POST(self):
            try:
                if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{port}',f'http://localhost:{port}'):raise ValueError('Foreign origin refused')
                if self.path!='/api/run':return self.send_json(dict(error='Not found'),404)
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<8192 or self.headers.get_content_type()!='application/json':raise ValueError('Expected bounded JSON')
                payload=json.loads(self.rfile.read(length));cfg=config(payload.get('config',{}));split=payload.get('split','train');control=payload.get('control','martingale')
                if split not in ('train','validation','holdout'):raise ValueError('Invalid split')
                if split=='holdout' and payload.get('freeze') is not True:raise ValueError('Freeze recipe before holdout')
                with lock:
                    if state['status']=='running':return self.send_json(dict(error='An experiment is already running'),409)
                    state.update(status='running',error=None,id=None)
                pool.submit(task,cfg,split,control);self.send_json(dict(status='running'),202)
            except Exception as exc:self.send_json(dict(error=str(exc)),400)
    print(f'http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=DEFAULT_DATA);p.add_argument('--output',default=DEFAULT_OUTPUT);p.add_argument('--port',type=int,default=8794)
    a=p.parse_args();serve(a.data,a.output,a.port)
