import itertools
import numpy as np
import pandas as pd
import pytest

from experiments.topn_crossings.matching import exact_matching


def test_matching_reassigns_old_best_instead_of_greedy():
    edges=pd.DataFrame(dict(event_id=['a','a','b'],entry_ns=[1,2,1],net=[10.,8.,9.]))
    solutions=exact_matching(edges,3)
    assert [s.net.sum() for s in solutions]==[10.,17.]
    assert set(zip(solutions[1].event_id,solutions[1].entry_ns))=={('a',2),('b',1)}


def test_exact_matching_against_exhaustive_random_graphs():
    rng=np.random.default_rng(113)
    for _ in range(50):
        rows=[dict(event_id=str(a),entry_ns=b,net=float(rng.integers(1,30))/10)
              for a in range(3) for b in range(4) if rng.random()<.75]
        edges=pd.DataFrame(rows)
        solutions=exact_matching(edges,3)
        for k in range(1,4):
            expected=0.
            for size in range(1,k+1):
                for combo in itertools.combinations(rows,size):
                    if len({r['event_id'] for r in combo})==size and len({r['entry_ns'] for r in combo})==size:
                        expected=max(expected,sum(r['net'] for r in combo))
            actual=solutions[min(k,len(solutions))-1].net.sum() if solutions else 0.
            assert actual==pytest.approx(expected)


def test_top_k_origin_pruning_matches_unpruned_market_graph():
    from experiments.topn_crossings.test_oracle import fixture
    from experiments.topn_crossings.oracle import prepare,outcome_representatives
    from experiments.topn_crossings.matching import candidate_edges
    tape,bank=fixture();valid,ends=prepare(tape,9_000_000_000,calendar=False)
    windows=pd.DataFrame(dict(window=[0],start=[0],end=[9_000_000_000],partial=False,horizon_ns=0,maturity_ns=9_000_000_000))
    reps,_=outcome_representatives(bank,tape,windows,valid,ends)
    edges=candidate_edges(bank,tape,windows,valid,ends,reps,2)
    full=[]
    for i,t in enumerate(tape['t']):
        future=bank[bank.t_ns>t]
        if not len(future):continue
        e=future.iloc[0]
        net=max(e.bid-tape['ask'][i]-.3,tape['bid'][i]-e.ask-.3)
        if net>0:full.append(dict(event_id=e.event_id,entry_ns=t,net=net))
    pruned=exact_matching(edges,2);unpruned=exact_matching(pd.DataFrame(full),2)
    assert [s.net.sum() for s in pruned]==pytest.approx([s.net.sum() for s in unpruned])
