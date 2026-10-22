"""Independent full-tape first-passage checks and loss contract tests."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from scripts.crossing_map_build import OUT,REPORT,DATES
from spectrum.playground.data import MonthDay,QUOTE_CACHE
from spectrum.cognition.execution import QuoteTape
from scripts.flow_context_coverage import save,sha


def main():
    checks=[]
    for date in DATES:
        roots=pd.read_parquet(OUT/(date+'_roots.parquet'))
        day=MonthDay(Path('data/derived/basket_multimonth_v1/2026-02')/date);q=QuoteTape.from_day(day)
        candidates=roots.loc[(roots.entry_ordinal>=0)&np.isfinite(roots.risk_R)]
        for _,row in candidates.iloc[np.linspace(0,len(candidates)-1,20,dtype=int)].iterrows():
            e=int(row.entry_ordinal);end=min(int(row.quote_support_end_ns),int(row.entry_ns)+1800*10**9+1)
            stop=np.searchsorted(q.quote_t,end,side='left')
            nets=[(q.bid[e:stop]-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[e:stop])/1e9-.3]
            for side,net in zip(['L','S'],nets):
                loss=np.flatnonzero(net<=-row.risk_R);limit=int(loss[0]) if len(loss) else len(net)
                for g,multiple in enumerate([.5,1.,2.]):
                    hit=np.flatnonzero(net[:limit]>=multiple*row.risk_R)
                    expected=int(q.quote_t[e+hit[0]]) if len(hit) else 0
                    if expected!=int(row[f'gain_{side}_{g}']):raise ValueError('First-passage mismatch')
                    checks.append(dict(date=date,t_ns=str(int(row.t_ns)),side=side,gain_multiple=multiple,matched=True))
        QUOTE_CACHE.clear()
    test=subprocess.run([sys.executable,'-m','pytest','tests/test_crossing_map.py','tests/test_bb_crossing.py','tests/test_vwap_compare.py','-q',
        '--basetemp=.tmp/pytest_crossing_map_verified'],capture_output=True,text=True)
    if test.returncode:raise ValueError(test.stdout+test.stderr)
    save(REPORT/'validation.json',dict(status='PASS',raw_first_passage_checks=len(checks),checks=checks,
        tests=test.stdout.strip(),script_sha256=sha(__file__)))
    print(json.dumps(dict(status='PASS',first_passage_checks=len(checks),tests=test.stdout.strip())),flush=True)


if __name__=='__main__':main()
