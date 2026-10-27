"""Verify saved evidence and summarize ambiguity/overlap, not trading returns."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    f=pd.read_parquet(path)
    return f.rename(columns={c:c.removeprefix('future_') for c in f})


def analyze(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='COMPLETE':raise ValueError('Incomplete labels')
    for path,digest in manifest['outputs'].items():
        if sha(root/path)!=digest:raise ValueError('Artifact drift: '+path)
    for path,digest in manifest['source_code_sha256'].items():
        snapshot=root/'source_snapshot'/path
        if sha(snapshot)!=digest:raise ValueError('Generation code snapshot drift')
    mm=json.loads((root/'matching/manifest.json').read_text())
    if sha(root/'matching/matching-source.py')!=mm['code_sha256']:raise ValueError('Matching code snapshot drift')
    if mm['parent_manifest_sha256']!=sha(root/'manifest.json'):raise ValueError('Matching lineage drift')
    for path,digest in mm['files'].items():
        if sha(root/'matching'/path)!=digest:raise ValueError('Matching artifact drift')
    top=read(root/'top.parquet');windows=read(root/'windows.parquet')
    curves=read(root/'curves.parquet');mc=read(root/'matching/curves.parquet');assign=read(root/'matching/assignments.parquet')
    if (mc.mass>mc.independent_mass+1e-7).any():raise ValueError('Matching exceeds independent upper bound')
    matching=[]
    for mode,g in mc[~mc.partial].groupby('mode'):
        a=assign[(assign['mode']==mode)&(assign.k==10)]
        overlap=[];near=[];both=[]
        for _,part in a.groupby(['day','window']):
            if part.entry_ns.duplicated().any() or part.event_id.duplicated().any():raise ValueError('Nonunique assignment')
            entry=part.entry_ns.to_numpy();exit=part.exit_ns.to_numpy();n=len(part)
            both.append(part.side.nunique()==2)
            if n>1:
                matrix=(np.maximum(entry[:,None],entry[None,:])<np.minimum(exit[:,None],exit[None,:]))
                overlap.append(float(matrix[np.triu_indices(n,1)].mean()))
                distances=np.abs(entry[:,None]-entry[None,:]).astype(float);np.fill_diagonal(distances,np.inf)
                near.append(float((distances.min(axis=1)<1e9).mean()))
        matching.append(dict(mode=mode,top10_retention_mean=float(g[g.k==10].retention.mean()),
            top32_retention_mean=float(g[g.k==32].retention.mean()),
            top10_pairwise_overlap_mean=float(np.mean(overlap)),
            top10_entries_with_neighbor_within_1s_mean=float(np.mean(near)),
            top10_both_sides_window_fraction=float(np.mean(both))))
    repeated=[]
    for mode,g in top.groupby('mode'):
        counts=g.groupby('event_id').window.count()
        repeated.append(dict(mode=mode,reused_event_ids=int((counts>1).sum()),
            top_rows=len(g),unique_events=g.event_id.nunique()))
    curve_summary=[]
    for (mode,control),g in curves[curves.k==10].groupby(['mode','control']):
        full=windows[(windows['mode']==mode)&~windows.partial&(windows.origins>0)][['day','window']]
        g=g.merge(full,on=['day','window'])
        curve_summary.append(dict(mode=mode,control=control,mean_coverage_event_mass=float(g.coverage_of_event_mass.mean())))
    sets={m:set(g[g['rank']<=10].event_id) for m,g in top.groupby('mode')}
    compare=[]
    for a,b in [('time_30m','time_30m_shift15'),('time_30m','time_30m_h60'),('time_60m','time_60m_h60')]:
        compare.append(dict(a=a,b=b,weekly_top10_event_set_jaccard=len(sets[a]&sets[b])/len(sets[a]|sets[b])))
    raw=[];verified=0
    for source in manifest['source_manifests']:
        day=source['day'];path=ROOT/manifest['config']['bank_root']/day
        if sha(path/'manifest.json')!=source['sha256']:raise ValueError('Bank manifest drift')
        bank=pd.read_parquet(path/'events.parquet')
        clock=pd.read_parquet(path/'tape.parquet',columns=['t_ns']).t_ns.to_numpy()
        if not (np.diff(clock)>0).all():raise ValueError('Matching requires unique ordered origin timestamps')
        raw.append(dict(day=day,events=len(bank),flicker_fraction=float((~bank.opportunity).mean())))
        # Independent vector audit of EVERY saved winner against the full raw bank.
        for mode in manifest['config']['window_modes']:
            rows=read(root/'labels'/day/(mode+'.parquet'))
            if rows.empty:continue
            if rows.duplicated(['window','event_id']).any():raise ValueError('Duplicate outcome representative')
            for (scale,role),part in rows.groupby(['scale','role']):
                ev=bank[(bank.scale==scale)&(bank.role==('lower','middle','upper').index(role))]
                times=ev.t_ns.to_numpy()
                for entry_column in ('entry_ns','best_long_entry_ns','best_short_entry_ns'):
                    ix=np.searchsorted(times,part[entry_column].to_numpy(),side='right')
                    if (ix>=len(ev)).any() or not np.array_equal(ev.event_id.to_numpy()[ix],part.event_id.to_numpy()):
                        raise ValueError('Skipped first raw crossing')
                    verified+=len(part)
    result=dict(status='PASS',scope='artifact hashes + all saved first-crossing winner links + matching uniqueness/upper bound',
        parent_manifest_sha256=sha(root/'manifest.json'),matching_manifest_sha256=sha(root/'matching/manifest.json'),
        first_crossing_links_verified=verified,public_api_parity_samples=manifest['public_api_parity_samples'],
        total_quote_origins=sum(s['origins'] for s in manifest['source_manifests']),
        total_raw_events=sum(s['events'] for s in manifest['source_manifests']),
        matching=matching,reused_outcomes=repeated,controls_top10=curve_summary,
        window_sensitivity=compare,raw_bank=raw,capital_profit=False)
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');a=p.parse_args();analyze(a.output)
