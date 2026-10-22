"""Feature usage, geometry-only comparisons and explicit context limits."""
import json
import pickle
import numpy as np
from spectrum.cognition import bb_crossing as bb,flow_context as fc
from scripts.bb_crossing_build import DATA,REPORT
from scripts.flow_context_coverage import save,sha


def main():
    families={
        'base':set(fc.FAMILIES['baseline'])-{'side'},
        'short_flow_and_returns':set(fc.FAMILIES['rich_flow'])-set(fc.FAMILIES['baseline']),
        'day_summaries':set(fc.DAY),'spectrum_summaries':set(fc.SPECTRUM),'bb_geometry':set(bb.GEOMETRY)}
    records={}
    for path in sorted(DATA.glob('model_B_*_rich*.pkl')):
        with path.open('rb') as src:saved=pickle.load(src)
        model=saved['model'];names=saved['columns'];used=set();splits={}
        for iteration in model._predictors:
            for tree in iteration:
                nodes=tree.nodes;idx=nodes['feature_idx'][nodes['is_leaf']==0]
                for i in idx:
                    name=names[int(i)];used.add(name);splits[name]=splits.get(name,0)+1
        records[path.name]=dict(input_columns=len(names),used_columns=len(used),
            groups={group:dict(available=len(values),used=len(values&used),split_nodes=sum(splits.get(v,0) for v in values)) for group,values in families.items()},
            unused=sorted(set(names)-used))
    save(REPORT/'feature_context_audit.json',dict(status='COMPLETE',models=records,script_sha256=sha(__file__),
        supplied='81 scalars: 5 baseline +31 local flow/returns/volume +21 day/previous-day +8 STFT +16 live BB geometry',
        day_scope='Today London prefix and previous same-contract weekday London-to-close summaries, with known/coverage masks; combined VWAP proxy includes current overnight. No full raw two-day sequence or recurrent state.',
        spectra='128 M1 changes/log-volume windows, three power fractions and entropy. No phase/coherence/absolute power; no full-scale bank of spectra.',
        vwap='Volume-weighted available M1 midpoint proxy, not exact trade-price VWAP',
        warning='Feature split use establishes access/use, not incremental predictive or economic value. Compare rich vs geometry on paired chronological days.'))


if __name__=='__main__':main()
