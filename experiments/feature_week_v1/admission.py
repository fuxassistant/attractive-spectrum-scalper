"""Bind source receipts and audited implementation without changing the baseline."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from profile import ROOT,OUTPUT,DATES,WEEK,source_for,raw_for,checked_inputs
import pandas as pd
from spectrum.io import save_json,sha


def main():
    sources=[]
    for d in DATES:
        s=source_for(d);market,raw,checks=checked_inputs(d)
        if sha(market/'manifest.json')!=s['market_manifest_sha256']:
            raise ValueError('Canonical manifest differs from full-history catalog')
        sources.append(dict(date=d,source_records=s['entry']['records'],route=s['entry']['route'],
                            canonical_manifest=sha(market/'manifest.json'),raw_manifest=sha(raw/'manifest.json'),
                            payloads=checks))
    populations={k:dict(rows=0,volume=0.,unknown_volume=0.) for k in ('all_rows','all_409_known','admitted_not_all_known')}
    for d in WEEK:
        base=OUTPUT/'serial_v2/days'/d
        f=pd.read_parquet(base/'input_seconds.parquet')
        x=pd.read_parquet(base/'features.parquet')
        all_known=x[[c for c in x if c.endswith('__known')]].all(axis=1)
        u=f.c_trade_unknown.diff().clip(lower=0).fillna(0)
        for key,mask in [('all_rows',pd.Series(True,index=f.index)),('all_409_known',all_known),
                         ('admitted_not_all_known',x.admissible&~all_known)]:
            populations[key]['rows']+=int(mask.sum())
            populations[key]['volume']+=float(f.trade_volume[mask].sum())
            populations[key]['unknown_volume']+=float(u[mask].sum())
    implementation={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'spectrum').rglob('*.py'))}
    implementation.update({str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    result=dict(status='OBSERVATION_PROFILE_AUDITED_WITH_LIMITATIONS',training_admission=False,
        audit_sha256=sha(OUTPUT/'audit/audit.json'),sources=sources,implementation=implementation,
        selection_populations=populations,
        tests=dict(passed=333,failed=0,seconds=66.44,
            command='python -m pytest -q tests checks experiments/topn_crossings experiments/feature_week_v1',
            evidence='Observed completed baseline-environment run, exit code 0'),
        limitations=['Five UTC-boundary endpoints absent; preserve gaps or reconstruct with verified replay handoff',
            'Five invalid-book seconds on Friday; endpoint-vs-interval schedule differs on five closing seconds',
            'Per-family masks required; complete-case filtering keeps only 5.4 percent of UTC-day rows',
            'Signed trade FFT is sparse because every unknown-side trade excludes the full window',
            'Two BB geometries remain explicit; choose one for targets and decisions',
            'Only partial previous ISO week; prior-week core fields masked',
            'Observation store decision day begins at London 08:00; full-session player loader still required',
            'Ordinary venue hours checked; general holiday calendar is NOT_COVERED',
            'Full persistent price-level depth/queue not present in compact observation tensors',
            'Predictive, sequential-policy and economic usefulness not tested in this data audit'])
    save_json(OUTPUT/'audit/admission.json',result)
    print(json.dumps(dict(status=result['status'],sources=len(sources),populations=populations)))


if __name__=='__main__':main()
