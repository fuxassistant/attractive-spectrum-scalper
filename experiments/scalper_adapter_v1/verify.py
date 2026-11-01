"""Post-generation checks formulated against saved records and native sources."""
import argparse
from html.parser import HTMLParser
import json

import numpy as np
import pandas as pd

from spectrum.io import sha
from spectrum.representation.session_store import open_session_store
from experiments.scalper_view_audit_v1.audit import dump
from .adapter import ROOT, HERE
from .tokens import NS, TICK


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.urls=[]
    def handle_starttag(self,tag,attrs):
        self.urls.extend(v for k,v in attrs if k in ('src','href'))


def main(run):
    out = ROOT/run
    m = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    r = json.loads((out/'report_receipt.json').read_text(encoding='utf-8'))
    cfg = json.loads((out/'config.json').read_text(encoding='utf-8'))
    assert m['status']=='COMPLETE_DESCRIPTIVE_ADAPTER'
    checks = 0
    for bank,base in ((m['source_sha256'],ROOT),(m['input_sha256'],ROOT),
                      (m['artifacts'],out),(r['artifacts'],out),(r['source_sha256'],ROOT)):
        for path,digest in bank.items():
            assert sha(base/path)==digest, path
            checks += 1
    assert r['input_manifest_sha256']==sha(out/'manifest.json')
    path_checks = token_checks = 0
    for date in cfg['sessions']:
        store = open_session_store(ROOT/cfg['source_root']/date)
        q = store.quotes[store.quotes.available_ns.between(store.start,store.end)].reset_index(drop=True)
        price = (q.bid_nano.to_numpy(np.int64)+q.ask_nano.to_numpy(np.int64))//2
        prefix = np.r_[0,np.cumsum(np.abs(np.diff(price)),dtype=np.int64)]
        qt = q.available_ns.to_numpy(np.int64)
        for name in ('motion','zone_updates'):
            f = pd.read_parquet(out/date/(name+'.parquet'))
            assert f.token_id.is_unique
            assert f.available_ns.dtype.kind in 'iu' and f.start_ns.dtype.kind in 'iu'
            si=f.start_index.to_numpy(np.int64); ei=f.source_index.to_numpy(np.int64)
            assert (si<=ei).all() and (qt[ei]<=f.available_ns.to_numpy()).all()
            assert (qt[si]<=f.start_ns.to_numpy()).all()
            assert np.array_equal(f.mid_nano.to_numpy(),price[ei])
            token_checks += len(f)
            if name=='motion':
                assert np.array_equal(f.path_nano.to_numpy(),prefix[ei]-prefix[si])
                assert np.array_equal(f.up_nano.to_numpy()+f.down_nano.to_numpy(),f.path_nano.to_numpy())
                assert (f.extreme_ns<=f.available_ns).all()
                turns=f[f.kind.eq('TURN_CONFIRMED')]
                assert (turns.direction*(turns.extreme_nano-turns.mid_nano)>=turns.amplitude_ticks*TICK).all()
                path_checks += len(f)
    observation_checks = neighbor_checks = 0
    for n in (1,2,3):
        name='query_'+str(n)
        table=pd.read_parquet(out/(name+'_tokens.parquet'))
        receipt=json.loads((out/(name+'_receipt.json')).read_text())
        model=np.load(out/(name+'_model.npz'),allow_pickle=False)
        assert table.available_ns.le(receipt['clock_ns']).all()
        assert table.available_ns.is_monotonic_increasing and table.token_id.is_unique
        for _, group in table.groupby(['session_id','available_ns']):
            assert group.source_index.is_monotonic_increasing
        assert table.session_id.ne(receipt['session_id']).sum()==receipt['previous_retained']>0
        assert model['values'].shape==(len(table),len(receipt['feature_names']))
        assert np.isfinite(model['values']).all()
        assert (model['values'][~model['known']]==0).all()
        assert model['categorical_values'].shape==(len(table),4)
        lookup=table.set_index('token_id')
        for row in table.itertuples():
            if row.next_selected_observed_update_id:
                nxt=lookup.loc[row.next_selected_observed_update_id]
                assert row.story_id==nxt.story_id and row.available_ns<=nxt.available_ns<=receipt['clock_ns']
            if row.kind=='LIVE_STATE' and row.family=='motion':
                assert row.up_nano/TICK==row.up_ticks and row.down_nano/TICK==row.down_ticks
                assert row.turns==row.direction_changes
            observation_checks += 1
        for match in json.loads((out/(name+'_similarity.json')).read_text()):
            assert match['available_ns']<match['query_context_start_ns']<=match['query_available_ns']
            assert set(match['context_token_ids']).issubset(lookup.index)
            assert match['distance']>=0
            neighbor_checks += 1
        memory=pd.read_parquet(out/(name+'_price_memory.parquet'))
        assert (memory.available_ns<=receipt['clock_ns']).all()
        assert (memory.dwell_seconds>=0).all()
        assert (memory.pass_count+memory.reject_count<=memory.confirmed_visits).all()
    parser=Links();parser.feed((out/'report.html').read_text(encoding='utf-8'))
    for target in parser.urls:
        assert (out/target).is_file(), target
    result=dict(status='PASS_BOUNDED_ADAPTER_CHECKS',manifest_sha256=sha(out/'manifest.json'),
        source_sha256=sha(__file__),hash_checks=checks,native_token_endpoint_checks=token_checks,
        independent_cumulative_native_path_checks=path_checks,observation_row_checks=observation_checks,
        past_only_neighbor_checks=neighbor_checks,local_links_checked=len(parser.urls),
        review_scope='Separate arithmetic/check program in the same task, not an independent agent or human review',
        economic_or_live_admission=False)
    dump(out/'validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=json.loads((HERE/'config.json').read_text())['output'])
    main(p.parse_args().run)
