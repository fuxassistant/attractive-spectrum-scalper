"""Independent checks of native timer support and generated audit receipts."""
from pathlib import Path
import json
from html.parser import HTMLParser

import numpy as np
import pandas as pd

from .audit import ROOT,HERE,digest,dump,NS


def main():
    cfg=json.loads((HERE/'config.json').read_text());out=ROOT/cfg['output']
    audit=json.loads((out/'manifest.json').read_text());report=json.loads((out/'report_receipt.json').read_text())
    assert audit['status']=='COMPLETE_DESCRIPTIVE_ONLY'
    count=0
    for m in (audit,report):
        for name,value in m.get('input_sha256',{}).items():assert digest(ROOT/name)==value;count+=1
        for name,value in m.get('source_sha256',{}).items():assert digest(ROOT/name)==value;count+=1
        for name,value in m['artifacts'].items():assert digest(out/name)==value;count+=1
    sm=json.loads((ROOT/cfg['source_run']/'manifest.json').read_text())
    latency=pd.read_parquet(out/'recognition_latency.parquet');timer_checks=0;prefix_scene_checks=0
    for date in cfg['dates']:
        r=next(r for r in sm['receipts'] if r['date']==date)
        q=pd.read_parquet(Path(r['source'])/'quotes.parquet',columns=['available_ns','bid_nano','ask_nano','valid','epoch','source_ordinal'])
        ev=pd.read_parquet(ROOT/cfg['source_run']/'days'/date/'events.parquet')
        gate=ev[ev.event_type.eq('GATE_START')].drop_duplicates('episode_id',keep='last').set_index('episode_id')
        qt=q.available_ns.to_numpy(np.int64);ordinals=q.source_ordinal.to_numpy(np.int64)
        assert np.all(ordinals[1:]>ordinals[:-1])
        mid=(q.bid_nano.to_numpy(np.int64)+q.ask_nano.to_numpy(np.int64))//2
        good=q.valid.to_numpy(bool)&(q.ask_nano>q.bid_nano).to_numpy()&q.epoch.ge(0).to_numpy();epoch=q.epoch.to_numpy(np.int64)
        for row in latency[(latency.date==date)&(latency.variant=='base')].itertuples():
            g=gate.loc[row.episode_id]
            lo=int(np.searchsorted(ordinals,int(g.source_ordinal),'left'))
            hi=int(np.searchsorted(qt,int(row.confirmation_ns),'left'))
            assert qt[lo]==g.available_ns==row.last_gate_ns
            assert row.confirmation_ns-int(g.available_ns)==3*NS and hi>lo
            assert good[lo:hi].all() and (epoch[lo:hi]==epoch[lo]).all()
            assert np.diff(qt[lo:hi]).max(initial=0)<=30*NS and row.confirmation_ns-qt[hi-1]<=30*NS
            far=(mid[lo:hi]>=int(row.zone_low_nano)+2*NS) if row.side==1 else (mid[lo:hi]<int(row.zone_low_nano)-NS)
            assert far.all() and row.confirmation_mid_nano==mid[hi-1]
            timer_checks+=1
    cases=json.loads((out/'atlas_index.json').read_text())
    for case in cases:
        e=pd.read_parquet(out/(case['case_id']+'_events.parquet'))
        assert (e.known_at_decision==(e.available_ns<=case['decision_ns'])).all()
        h=e[(e.episode_id==case['episode_id'])&e.known_at_decision]
        assert not h.event_type.eq('DEPARTURE_CONFIRMED').any()
        assert ((h.available_ns==case['decision_ns'])&h.event_type.eq('GATE_START')).any()
        prefix_scene_checks+=1
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.local=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            for name in ('href','src'):
                if name in a and not a[name].startswith('https:'):self.local.append(a[name])
    parser=Links();parser.feed((out/'report.html').read_text(encoding='utf-8'))
    for p in parser.local:assert (out/p).resolve().is_file(),p
    result=dict(status='PASS_BOUNDED_DESCRIPTIVE_CHECKS',hash_checks=count,
        native_continuous_gate_timer_checks=timer_checks,atlas_asof_boundary_checks=prefix_scene_checks,
        local_links=len(parser.local),verifier_sha256=digest(__file__),
        audit_manifest_sha256=digest(out/'manifest.json'),report_receipt_sha256=digest(out/'report_receipt.json'),
        limits=['Verifier written/run in same task; no independent agent review.',
                'Two exposed days; changed parameter definitions are not truth/error labels.',
                'No economic outcome, predictive test or broad model capability inferred.'])
    dump(out/'validation.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
