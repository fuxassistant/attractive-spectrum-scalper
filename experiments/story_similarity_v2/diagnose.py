"""Explain the exposed visual counterexample without changing the frozen metric."""
import argparse
import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.scalper_adapter_v1.adapter import live_states
from experiments.scalper_view_audit_v1.audit import dump
from .run import ROOT,HERE,read,load_corpus
from .search import SimilaritySearch
from .metric import score_components


def main(run):
    folder=ROOT/run
    if (folder/'diagnostics.json').exists():raise FileExistsError('Diagnostic already exists')
    cfg=read(folder/'config.json');manifest=read(folder/'manifest.json')
    for name in ('matches.json','atlas_index.json','sweep_matches.parquet','statuses.parquet'):
        assert sha(folder/name)==manifest['artifacts'][name]
    sessions,_=load_corpus(cfg);search=SimilaritySearch(sessions,cfg)
    clock=int(pd.Timestamp(cfg['query_session']+'T12:00:00Z').value)
    token=live_states(search.sessions[cfg['query_session']],clock).query("family == 'motion' and scale == 'slow'").iloc[0]
    prepared=search.prepare(token,'previous')
    case=next(c for c in read(folder/'atlas_index.json') if c['case_id']=='120000_slow')
    prior_folder=ROOT/'research_results/story_similarity_v2/verified_v2'
    prior=next(c for c in read(prior_folder/'atlas_index.json') if c['case_id']=='120000_slow')
    candidates=[('legacy',case['methods']['legacy']['token_id']),
                ('before_whole_trajectory',prior['methods']['rhythm_gate']['token_id']),
                ('after_whole_trajectory',case['methods']['rhythm_gate']['token_id'])]
    rows=[]
    for name,token_id in candidates:
        ix=int(np.flatnonzero(prepared['ids']==search.base.lookup[token_id])[0])
        parts={k:float(v[ix]) if np.isfinite(v[ix]) else None for k,v in prepared['parts'].items()}
        checks={}
        for method in cfg['methods']:
            score,eligible,accepted=score_components(prepared['parts'],method,cfg)
            checks[method['name']]=dict(score=float(score[ix]),eligible=bool(eligible[ix]),accepted=bool(accepted[ix]),
                weighted_components={k:float(w*parts[k]) for k,w in method['weights'].items() if w and parts.get(k) is not None})
        index=search.parent_indices[prepared['ids'][ix]]
        amplitude=search.features['amplitude'][index]
        rows.append(dict(name=name,token_id=token_id,parts=parts,methods=checks,
                         slot_net_amplitudes=amplitude[:,0].tolist(),slot_range_amplitudes=amplitude[:,1].tolist(),
                         slot_seconds=search.features['duration'][index].tolist()))
    sweep=pd.read_parquet(folder/'sweep_matches.parquet')
    sweep['story_id']=[search.base.frame.iloc[search.base.lookup[x]].story_id for x in sweep.token_id]
    stability=[]
    for query_id,g in sweep.groupby('query_token_id',sort=False):
        counts=g.story_id.value_counts()
        stability.append(dict(query_token_id=query_id,scale=g.scale.iloc[0],query_ns=int(g.query_ns.iloc[0]),
            settings_with_neighbor=len(g),distinct_first_token=int(g.token_id.nunique()),
            distinct_first_story=int(g.story_id.nunique()),most_common_story_count=int(counts.iloc[0])))
    statuses=pd.read_parquet(folder/'statuses.parquet')
    result=dict(status='EXPOSED_VISUAL_COUNTEREXAMPLE_DIAGNOSED',manifest_sha256=sha(folder/'manifest.json'),
        source_sha256=sha(__file__),previous_atlas_sha256=sha(prior_folder/'atlas_index.json'),
        query_slot_net_amplitudes=prepared['query_pack']['amplitude'][:,0].tolist(),
        query_slot_range_amplitudes=prepared['query_pack']['amplitude'][:,1].tolist(),
        query_slot_seconds=prepared['query_pack']['duration'].tolist(),noon_candidates=rows,
        first_neighbor_parameter_stability=stability,
        status_counts=statuses.groupby(['scope','method','status']).size().rename('count').reset_index().to_dict('records'),
        limitation='Post-hoc explanation on exposed examples; no parameter selection or new outcome evidence')
    dump(folder/'diagnostics.json',result)
    print(dict(noon_candidates=rows,query_nets=result['query_slot_net_amplitudes'],
               median_distinct_first_stories=float(np.median([r['distinct_first_story'] for r in stability]))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=read(HERE/'config.json')['output']);main(p.parse_args().run)
