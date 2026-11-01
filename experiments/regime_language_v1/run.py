"""Bounded probe of existing sentences through price-time and macro VWAP context."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.scalper_view_audit_v1.audit import dump
from experiments.story_similarity_v2.run import load_corpus,read
from .measures import TimeAtPrice,VwapContext,anchor_relations,terms,scalar_time_mean,NS

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEFAULT='research_results/regime_language_v1/verified_v1'


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,list):return [serial(v) for v in value]
    return value


def describe(frame,session,map_,vwap,previous_profile):
    frame=frame.sort_values('available_ns',kind='stable')
    left=int(frame.start_ns.iloc[0]);right=int(frame.available_ns.iloc[-1]);A=int(frame.amplitude_ticks.iloc[0])
    p0=int(frame.start_mid_nano.iloc[0])/NS;p1=int(frame.mid_nano.iloc[-1])/NS
    profile=map_.facets(left,right,A)
    start=vwap.at(left);end=vwap.at(right)
    relations=anchor_relations(start,end,p0,p1,profile['time_mean'],A)
    net=(p1-p0)/(A*.1);path=float(frame.path_nano.sum())/(A*100_000_000)
    variants=[]
    for width,shift in [(5,2.5),(10,0)]:
        other=map_.facets(left,right,A,width,shift)
        assert other['time_mean']==profile['time_mean'] and other['time_sigma']==profile['time_sigma']
        assert other['center_migration_A']==profile['center_migration_A']
        variants.append(dict(width_ticks=width,shift_ticks=shift,mode_price=other['mode_price'],
            largest_bin_fraction=other['largest_bin_fraction'],half_profile_overlap=other['half_profile_overlap']))
    occupied=profile['bin_low_nano'];prior=previous_profile
    use=np.isin(prior['bin_low_nano'],occupied)
    share=float(prior['bin_seconds'][use].sum()/prior['supported_seconds']) if prior['supported_seconds'] else None
    result=dict(session_id=session.session_id,scale=frame.scale.iloc[0],start_ns=left,available_ns=right,
        source_index=int(frame.source_index.iloc[-1]),start_index=int(frame.start_index.iloc[0]),
        amplitude_ticks=A,start_price=p0,end_price=p1,net_A=net,path_A=path,
        efficiency=abs(net)/path if path else 0.,profile=profile,
        anchor_start=start,anchor_end=end,anchor_relations=relations,
        prior_observed_time_share_at_occupied_bins=share,prior_session_time_coverage=prior['coverage'],
        grid_variants=variants,terms=terms(profile,net,path,(right-left)/NS/float(frame.time_limit_seconds.sum()),relations))
    if profile['known']:
        assert abs(profile['bin_seconds'].sum()-profile['duration_seconds'])<1e-8
    for row in relations.values():
        if row['relative_identity_residual'] is not None:assert abs(row['relative_identity_residual'])<1e-10
    return result


def main(out):
    folder=ROOT/out;folder.mkdir(parents=True,exist_ok=False);begun=time.perf_counter()
    parent=ROOT/'research_results/story_similarity_v2/verified_v3'
    config=read(parent/'config.json');pm=read(parent/'manifest.json')
    assert sha(parent/'query_contexts.parquet')==pm['artifacts']['query_contexts.parquet']
    files=list(HERE.glob('*.py'))+[ROOT/'tests/test_regime_language.py']
    source={str(p.relative_to(ROOT)):sha(p) for p in files}
    for p in files:
        to=folder/'source_snapshot'/p.relative_to(ROOT);to.parent.mkdir(parents=True,exist_ok=True);to.write_bytes(p.read_bytes())
    contract=dict(version='REGIME_LANGUAGE_PROBE_V1',parent=str(parent.relative_to(ROOT)),
        segmentation='Reuse existing causal motion intervals; no learned regime boundary or retokenization claimed',
        price_time='Exact held native midpoint time integral on [start,available); absolute 5-tick bins',
        profile_controls=[dict(width_ticks=5,shift_ticks=2.5),dict(width_ticks=10,shift_ticks=0)],
        vwaps=['current venue session','frozen previous venue session','previous plus current','rolling 60 completed minutes','rolling 240 completed minutes'],
        clock='Context publication strictly before token clock; endpoint age and source coverage retained',
        vocabulary='Overlapping evidence facets with numbers and masks, not exhaustive market regimes',
        future_outcomes=False,policy=False)
    dump(folder/'config.json',contract);dump(folder/'manifest.json',dict(status='RUNNING',source_sha256=source))
    sessions,inputs=load_corpus(config);by_date={s.session_id:s for s in sessions}
    current=by_date[config['query_session']];previous=by_date[current.previous_session_id]
    map_=TimeAtPrice(current.quotes);vwap=VwapContext(current.context)
    previous_profile=TimeAtPrice(previous.quotes).profile(previous.start_ns,previous.end_ns)
    inputs[str(parent/'manifest.json')]=sha(parent/'manifest.json')
    inputs[str(parent/'query_contexts.parquet')]=sha(parent/'query_contexts.parquet')
    raw=pd.read_parquet(parent/'query_contexts.parquet');tokens=[];sentences=[]
    for token_id,frame in raw.drop_duplicates('token_id').groupby('token_id',sort=False):
        r=describe(frame,current,map_,vwap,previous_profile);r.update(token_id=token_id,kind=frame.kind.iloc[0],phase=frame.phase.iloc[0])
        tokens.append(r)
    scalar_checks=0;prefix_checks=0
    for query_id,frame in raw.groupby('query_id',sort=False):
        r=describe(frame,current,map_,vwap,previous_profile)
        r.update(sentence_id=query_id,token_ids=frame.sort_values('available_ns',kind='stable').token_id.tolist())
        mean,support=scalar_time_mean(current.quotes,r['start_ns'],r['available_ns'])
        assert support==round(r['profile']['supported_seconds']*NS)
        assert np.isclose(mean,r['profile']['time_mean'],rtol=0,atol=1e-8);scalar_checks+=1
        # A context update after the query cannot change any anchor. Profile
        # endpoints use no quotes at or after their right-open boundary.
        clipped=VwapContext(current.context[current.context.available_ns<r['available_ns']])
        assert clipped.at(r['available_ns'])==r['anchor_end'];prefix_checks+=1
        sentences.append(r)
    selected=[]
    for stamp in ('07:30:00','12:00:00','18:30:00'):
        clock=int(pd.Timestamp(config['query_session']+'T'+stamp+'Z').value)
        r=next(x for x in sentences if x['scale']=='slow' and x['available_ns']==clock)
        case=stamp.replace(':','')+'_slow';r['case_id']=case;selected.append(r['sentence_id'])
        q=current.quotes.iloc[r['start_index']:r['source_index']+1]
        q[['available_ns','bid_nano','ask_nano','valid','epoch']].to_parquet(folder/(case+'_quotes.parquet'),index=False)
        current.context[(current.context.t_ns>clock-4*3600*NS)&(current.context.available_ns<clock)].to_parquet(folder/(case+'_context.parquet'),index=False)
    # An actual noon source truncation, independent of endpoint selection.
    noon=int(pd.Timestamp(config['query_session']+'T12:00:00Z').value)
    short=TimeAtPrice(current.quotes[current.quotes.available_ns<noon])
    r=next(x for x in sentences if x['scale']=='slow' and x['available_ns']==noon)
    full=r['profile'];cut=short.facets(r['start_ns'],noon,r['amplitude_ticks'])
    assert json.dumps(serial(full),sort_keys=True)==json.dumps(serial(cut),sort_keys=True)
    dump(folder/'tokens.json',serial(tokens));dump(folder/'sentences.json',serial(sentences))
    dump(folder/'previous_profile.json',serial(previous_profile));dump(folder/'atlas_index.json',selected)
    counts={}
    for r in sentences:
        for word in r['terms']:counts[word]=counts.get(word,0)+1
    anchor_known={name:sum(r['anchor_relations'][name]['known'] for r in sentences) for name in sentences[0]['anchor_relations']}
    summary=dict(status='COMPLETE_DESCRIPTIVE_REGIME_PROBE',sentences=len(sentences),unique_motion_tokens=len(tokens),
        term_counts=counts,anchor_known_queries=anchor_known,scalar_time_integrals=scalar_checks,
        context_prefix_checks=prefix_checks,native_noon_prefix_equal=True,
        previous_observed_seconds=previous_profile['supported_seconds'],previous_time_coverage=previous_profile['coverage'],
        elapsed_seconds=time.perf_counter()-begun,
        limits=['Same exposed dates; no unseen semantic or economic confirmation',
                'No learned primary segmentation or encoder; old motion boundaries retained',
                'Time occupancy describes midpoint persistence, not transaction volume or participant intent',
                'VWAP means use complete published minutes and carry their publication age; they do not update on every native quote',
                'Previous-session time distribution covers observed supported intervals only'])
    dump(folder/'summary.json',summary)
    artifacts={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'}
    dump(folder/'manifest.json',dict(status=summary['status'],source_sha256=source,input_sha256=inputs,artifacts=artifacts))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=DEFAULT);main(p.parse_args().out)
