"""Small CLI: inspect, ingest, build, features, replay, audit."""
import argparse, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('ingest');x.add_argument('--source',required=True);x.add_argument('--output',required=True)
    x.add_argument('--route',nargs=3,type=int,default=[1,19181,22]);x.add_argument('--member')
    x=sub.add_parser('build');x.add_argument('--raw',required=True);x.add_argument('--output',required=True);x.add_argument('--workers',type=int,default=4)
    x=sub.add_parser('features');x.add_argument('--market',required=True);x.add_argument('--output',required=True);x.add_argument('--workers',type=int,default=4)
    x.add_argument('--families',nargs='+',default=['motion','flow','spectral'])
    x.add_argument('--plugin',action='append',default=[],help='Importable module:FeaturePlugin attribute')
    x=sub.add_parser('status');x.add_argument('root')
    x=sub.add_parser('replay');x.add_argument('capture');x.add_argument('--limit',type=int,default=5)
    x=sub.add_parser('audit');x.add_argument('--market',required=True);x.add_argument('--features',required=True);x.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='ingest':
        from .ingest import partition
        r=partition(a.source,a.output,a.route,a.member)
    elif a.command=='build':
        from .build import build
        if not 1<=a.workers<=12:p.error('workers must be 1..12')
        r=build(a.raw,a.output,a.workers)
    elif a.command=='features':
        from .lab import generate
        if not 1<=a.workers<=12:p.error('workers must be 1..12')
        r=generate(a.market,a.output,a.families,a.workers,a.plugin)
    elif a.command=='status':
        root=Path(a.root)
        r=[]
        for capture in sorted(root.glob('capture_*')):
            mf=capture/'manifest.json';pf=capture/'progress.json'
            if mf.exists():
                m=json.loads(mf.read_text());r.append(dict(capture=capture.name,status=m['status'],records=m['records'],seconds=m['elapsed_seconds']))
            elif pf.exists():r.append(dict(capture=capture.name,status='RUNNING_OR_INTERRUPTED',**json.loads(pf.read_text())))
    elif a.command=='replay':
        from .tape import TapeReplay
        stream=TapeReplay(a.capture)
        r=[]
        for row in stream:
            r.append(row)
            if len(r)>=a.limit:break
    else:
        from .audit import audit
        r=audit(a.market,a.features,a.output)
    print(json.dumps(r,indent=2,default=str))

if __name__=='__main__':main()
