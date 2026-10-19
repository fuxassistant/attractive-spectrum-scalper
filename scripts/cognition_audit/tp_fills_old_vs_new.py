"""Same take-profit probes through the old (v2, git) and current (v4) kernels."""
import importlib.util
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import os; os.chdir(repo)
from spectrum.cognition.passive import day_tapes, captures_by_date, LATENCY
from spectrum.cognition.signatures import MENU_PLANS
from spectrum.cognition import queue as v4

scratch = Path(sys.argv[2])
old_src = subprocess.run(['git', 'show', 'cda127a:spectrum/cognition/queue.py'], capture_output=True, text=True, check=True).stdout
(scratch / 'queue_v2_old.py').write_text(old_src.replace('@njit(cache=True)', '@njit(cache=False)'), encoding='utf8')
spec = importlib.util.spec_from_file_location('queue_v2_old', scratch / 'queue_v2_old.py'); v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)

date = sys.argv[3]
f = pd.read_parquet(f'data/derived/passive_execution_v4/{date}.parquet', filters=[('variant', '==', 'TQ'), ('status', '==', 'COMPLETE')])
member = captures_by_date(months=(date[:7],))[date][0]['member_key']
for _, tape, _ in day_tapes(member, [date]):
    pass
take = f.plan_id.map(lambda p: MENU_PLANS[p][0]).to_numpy(); hold = f.plan_id.map(lambda p: MENU_PLANS[p][2]).to_numpy()
tp = f.entry_px.astype('int64').to_numpy() + f.side.to_numpy() * take * 100_000_000
place = f.entry_ns.astype('int64').to_numpy() + LATENCY; end = f.entry_ns.astype('int64').to_numpy() + hold * 10**9
args = (place, -f.side.to_numpy(), tp, end)
s2, f2, k2, a2 = v2.probe(tape, *args); s4, f4, k4, a4 = v4.probe(tape, *args)
d = pd.DataFrame(dict(s2=s2, f2=f2, k2=k2, s4=s4, f4=f4, k4=k4, ahead=a4, side=-f.side.to_numpy(), tp=tp, place=place))
print('TP probes', len(d), '| v2 filled', int((d.s2 == 1).sum()), '| v4 filled', int((d.s4 == 1).sum()))
print('v4 fill kinds:', d[d.s4 == 1].k4.map(v4.KIND).value_counts().to_dict())
new = d[(d.s4 == 1) & ((d.s2 != 1) | (d.f4 < d.f2))]
print('filled only/earlier in v4:', len(new), 'by kind', new.k4.map(v4.KIND).value_counts().to_dict())
new.to_parquet(scratch / f'tp_new_{date}.parquet')

# necessary condition: a resting limit cannot fill before a trade prints at or through its price
T = ord('T'); tr = np.flatnonzero(tape['action'] == T); tts = tape['ts'][tr]; tpx = tape['price'][tr]
def first_trade(p, side, price):
    lo = np.searchsorted(tts, p, side='left')
    hit = np.flatnonzero((tpx[lo:] >= price) if side < 0 else (tpx[lo:] <= price))
    return int(tts[lo + hit[0]]) if len(hit) else None
for name, s_, f_ in (('v2', d.s2, d.f2), ('v4', d.s4, d.f4)):
    bad = 0; n = 0
    for i in np.flatnonzero(s_.to_numpy() == 1):
        n += 1; ft = first_trade(int(d.place.iat[i]), int(d.side.iat[i]), int(d.tp.iat[i]))
        bad += ft is None or int(f_.iat[i]) < ft
    print(name, 'filled', n, 'fills before any trade at/through the limit (impossible):', bad)
