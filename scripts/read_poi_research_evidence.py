"""List/read a preserved evidence item without extracting executable archives."""
import argparse,json,zipfile
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('pattern');p.add_argument('--read',action='store_true')
p.add_argument('--capsule',default='reports/poi_research_review_20261018/research_evidence.zip')
a=p.parse_args()
with zipfile.ZipFile(a.capsule) as z:
    index=json.loads(z.read('index.json'))
    matches=[e for e in index['entries'] if a.pattern.lower() in (str(e['archive'])+' '+e['origin']).lower()]
    if a.read:
        if len(matches)!=1:raise SystemExit(f'Expected one match, found {len(matches)}; list and use an exact path.')
        print(z.read(matches[0]['member']).decode('utf8'))
    else:
        for e in matches:print(json.dumps({k:e[k] for k in ('archive','origin','bytes','sha256')},ensure_ascii=False))
