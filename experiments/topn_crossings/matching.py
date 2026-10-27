"""Exact cardinality matching: one origin and one raw outcome per assignment.

This is still a set of potentially overlapping alternatives, not one-position
trading. The best assignment at k+1 may replace the assignment at k.
"""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from spectrum.crossings import open_bank
from spectrum.crossings.bank import SCALES, ROLES
from spectrum.future.crossings import ORIGIN_KEYS
from spectrum.economics import future_crossing_marks
from spectrum.leakage import mark_future, export_future_values
from experiments.topn_crossings.oracle import prepare, make_windows, write_labels


def candidate_edges(bank,tape,windows,valid,ends,reps,k=32):
    """Top K distinct origins per outcome are sufficient for any size<=K optimum.

    At most K-1 other origins can be occupied, hence a retained origin can
    always replace an omitted edge without decreasing the objective.
    """
    t=tape['t'];bid=tape['bid'];ask=tape['ask']
    wid=np.searchsorted(windows.end.to_numpy(),t,side='right')
    w=np.minimum(wid,len(windows)-1)
    eligible=valid&(wid<len(windows))&(t>=windows.start.to_numpy()[w])
    h=windows.horizon_ns.to_numpy()[w]
    limit=np.where(h>0,t+h+1,windows.end.to_numpy()[w])
    parts=[]
    for scale in SCALES:
        support=tape['support'][scale]
        for role in range(3):
            ev=bank[(bank.scale==scale)&(bank.role==role)].reset_index(drop=True)
            if not len(ev):continue
            ix=np.searchsorted(ev.t_ns.to_numpy(),t,side='right');safe=np.minimum(ix,len(ev)-1)
            found=eligible&(support>=0)&(ix<len(ev))&(ev.support.to_numpy()[safe]==support)&(ev.t_ns.to_numpy()[safe]<=ends[scale])&(ev.t_ns.to_numpy()[safe]<limit)
            positive=reps[(reps.scale==scale)&(reps.role==ROLES[role])&(reps.score>0)]
            event_lookup={name:i for i,name in enumerate(ev.event_id)}
            allowed=np.array([int(r.window)*(len(ev)+1)+event_lookup[r.event_id] for r in positive.itertuples()],np.int64)
            key=w*(len(ev)+1)+ix
            found &= np.isin(key,allowed)
            origins=np.flatnonzero(found)
            if not len(origins):continue
            f=pd.DataFrame(dict(group=key[origins],origin=origins,ask=ask[origins],bid=bid[origins]))
            li=f.sort_values(['group','ask','origin'],kind='stable').groupby('group',sort=False).head(k).origin.to_numpy()
            si=f.sort_values(['group','bid','origin'],ascending=[True,False,True],kind='stable').groupby('group',sort=False).head(k).origin.to_numpy()
            origin=np.unique(np.r_[li,si]);e=ev.iloc[ix[origin]]
            q=pd.DataFrame(dict(t_ns=t[origin],scale=scale,role=ROLES[role],origin_bid=bid[origin],origin_ask=ask[origin],
                                future_status='CROSSED',future_bid=e.bid.to_numpy(),future_ask=e.ask.to_numpy()))
            marks=future_crossing_marks(mark_future(q,__name__,ORIGIN_KEYS))
            long=export_future_values(marks.future_long_usd);short=export_future_values(marks.future_short_usd)
            choose_long=np.round(long,8)>=np.round(short,8)
            part=pd.DataFrame(dict(window=w[origin],event_id=e.event_id.to_numpy(),entry_ns=t[origin],exit_ns=e.t_ns.to_numpy(),
                side=np.where(choose_long,'LONG','SHORT'),net=np.where(choose_long,long,short),scale=scale,role=ROLES[role],
                entry_bid=bid[origin],entry_ask=ask[origin],exit_bid=e.bid.to_numpy(),exit_ask=e.ask.to_numpy(),
                ordinal=e.ordinal.to_numpy(),cluster_id=e.cluster_event_id.to_numpy(),
                maturity_ns=windows.maturity_ns.to_numpy()[w[origin]]))
            part['score']=part.net.round(8)
            part=part[part.score>0].sort_values(['window','event_id','score','entry_ns'],ascending=[True,True,False,True],kind='stable')
            parts.append(part.groupby(['window','event_id'],sort=False).head(k))
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def exact_matching(edges,kmax):
    """Successive shortest augmenting paths on a bipartite min-cost flow graph.

    Costs are integer 1e-8 USD/oz, matching the recorded tie precision.
    Yields the entire optimal assignment for each non-empty cardinality.
    """
    if edges.empty:return []
    edges=edges.reset_index(drop=True)
    events=sorted(edges.event_id.unique());origins=sorted(edges.entry_ns.unique())
    ne=len(events);no=len(origins);source=0;sink=1+ne+no;n=sink+1
    event_node={e:i+1 for i,e in enumerate(events)}
    origin_node={o:ne+1+i for i,o in enumerate(origins)}
    graph=[[] for _ in range(n)]
    def add(a,b,cost):
        edge=[b,len(graph[b]),1,cost];reverse=[a,len(graph[a]),0,-cost]
        graph[a].append(edge);graph[b].append(reverse)
        return edge
    for v in event_node.values():add(source,v,0)
    for v in origin_node.values():add(v,sink,0)
    references=[];pot=[0]*n
    for i,r in enumerate(edges.itertuples()):
        cost=-int(round(r.net*100_000_000));u=event_node[r.event_id];v=origin_node[r.entry_ns]
        references.append((i,add(u,v,cost)));pot[v]=min(pot[v],cost)
    pot[sink]=min(pot[v] for v in origin_node.values())
    solutions=[];infinity=10**40
    for k in range(1,kmax+1):
        dist=[infinity]*n;dist[source]=0;previous=[None]*n;queue=[(0,source)]
        while queue:
            d,u=heapq.heappop(queue)
            if d!=dist[u]:continue
            for ei,e in enumerate(graph[u]):
                v,_,capacity,cost=e
                if not capacity:continue
                nd=d+cost+pot[u]-pot[v]
                if nd<dist[v]:
                    dist[v]=nd;previous[v]=(u,ei);heapq.heappush(queue,(nd,v))
        if dist[sink]==infinity or dist[sink]+pot[sink]-pot[source]>=0:break
        for v in range(n):
            if dist[v]!=infinity:pot[v]+=dist[v]
        v=sink
        while v!=source:
            u,ei=previous[v];e=graph[u][ei];e[2]-=1;graph[v][e[1]][2]+=1;v=u
        selected=edges.iloc[[i for i,e in references if e[2]==0]].copy()
        if len(selected)!=k or selected.event_id.duplicated().any() or selected.entry_ns.duplicated().any():
            raise ValueError('Invalid matching')
        solutions.append(selected)
    return solutions


def read_y(path):
    f=pd.read_parquet(path)
    return f.rename(columns={c:c.removeprefix('future_') for c in f})


def run(root,modes,kmax):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());out=root/'matching';out.mkdir(exist_ok=False)
    source_bytes=Path(__file__).read_bytes()
    (out/'matching-source.py').write_bytes(source_bytes)
    curves=[];assignments=[];edge_count=0
    for day in m['config']['days']:
        bank,tape,meta=open_bank(ROOT/m['config']['bank_root']/day)
        valid,ends=prepare(tape,meta['coverage_end_ns'])
        for mode in modes:
            reps=read_y(root/'labels'/day/(mode+'.parquet'))
            windows=make_windows(tape['t'],bank,meta['coverage_end_ns'],mode)
            edges=candidate_edges(bank,tape,windows,valid,ends,reps,kmax)
            edge_count+=len(edges)
            for w in windows.itertuples():
                e=edges[edges.window==w.window] if len(edges) else edges
                independent=reps[(reps.window==w.window)&(reps.score>0)]
                solutions=exact_matching(e,kmax)
                previous_mass=0.
                for k in range(1,kmax+1):
                    selected=solutions[min(k,len(solutions))-1] if solutions else e
                    mass=float(selected.net.sum()) if len(selected) else 0.
                    independent_mass=float(independent.head(k).net.sum())
                    curves.append(dict(day=day,mode=mode,window=w.window,k=k,selected=len(selected),mass=mass,
                        independent_mass=independent_mass,retention=mass/independent_mass if independent_mass else None,
                        marginal=mass-previous_mass,partial=w.partial,window_start=w.start,window_end=w.end))
                    previous_mass=mass
                    if len(selected) and k in (1,3,10,32):
                        selected=selected.copy();selected['k']=k;selected['day']=day;selected['mode']=mode
                        assignments.append(selected)
            print(json.dumps(dict(day=day,mode=mode,edges=len(edges))),flush=True)
    write_labels(pd.DataFrame(curves),out/'curves.parquet')
    write_labels(pd.concat(assignments,ignore_index=True) if assignments else pd.DataFrame(),out/'assignments.parquet')
    receipt=dict(status='COMPLETE',exact_for='maximum net sum at up to K, unique origin timestamp AND event_id; overlapping intervals allowed',
                 kmax=kmax,pruned_edges=edge_count,modes=modes,
                 code_sha256=hashlib.sha256(source_bytes).hexdigest(),
                 parent_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
                 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.parquet')},
                 portfolio_profit=False)
    (out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');p.add_argument('--kmax',type=int,default=32)
    p.add_argument('--modes',nargs='+',default=['time_30m','time_60m','time_30m_h60','time_60m_h60'])
    a=p.parse_args();run(a.output,a.modes,a.kmax)
