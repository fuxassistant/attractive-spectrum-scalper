"""Build a portable interactive audit of the frozen basket policy."""
from pathlib import Path
import argparse,json
import numpy as np
from spectrum.finalized import read_json
from .data import MonthDay
from .runner import DEFAULT_DATA
from .basket_search import DEFAULT_OUTPUT


def build(data=DEFAULT_DATA,output=DEFAULT_OUTPUT):
    root=Path(output);manifest=read_json(Path(data)/'manifest.json');markets={}
    for item in manifest['days']:
        d=MonthDay(Path(data)/item['date'])
        markets[item['date']]=dict(t=[str(int(t)) for t in d.decision_t],
            price=[float(p) if k else None for p,k in zip(d.decision_mid,d.x[:,8]>.5)],
            london=d.meta['london_ns'],close=d.meta['ny_close_ns'])
    payload=dict(summary=read_json(root/'summary.json'),audit=read_json(root/'audit.json'),markets=markets,
        records={n:read_json(root/n/'result.json') for n in ('month','same_grid_fixed_adds','same_signal_no_adds')})
    html=Path(__file__).with_name('basket_report.html').read_text(encoding='utf8')
    html=html.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    (root/'index.html').write_text(html,encoding='utf8')
    return root/'index.html'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=DEFAULT_DATA);p.add_argument('--output',default=DEFAULT_OUTPUT)
    a=p.parse_args();print(build(a.data,a.output))
