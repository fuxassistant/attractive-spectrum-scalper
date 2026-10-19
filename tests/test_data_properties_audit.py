import numpy as np
from spectrum.data_properties_audit import Distribution, interarrival, publication_hash


def test_population_quantile_bounds_include_observed_rank_and_keep_tails():
    values=np.array([0,0,1,2,3,1000,10**6,5*10**9,5*10**9+1,60*10**9,-1],np.int64)
    d=Distribution();d.add(values[:4]);d.add(values[4:])
    r=d.result();ordered=np.sort(values[values>=0])
    assert r['negative']==1 and r['zeros']==2 and r['count']==10
    assert r['tail_counts_gt_ns'][str(5*10**9)]==2
    for q,bounds in r['quantile_ns_bounds'].items():
        value=ordered[max(1,int(np.ceil(float(q)*len(ordered))))-1]
        assert bounds[0]<=value<=bounds[1]


def test_chunk_boundary_intervals_are_not_lost_or_invented():
    d=Distribution();last=interarrival(d,[1,1,3]);last=interarrival(d,[],last)
    interarrival(d,[8,20],last)
    r=d.result();assert r['count']==4 and r['zeros']==1 and r['total_interval_seconds']==19/10**9


def test_publication_identity_preserves_exact_ns_and_order():
    times=np.array([1779097633117989959,1779097640652992020],np.int64)
    assert publication_hash([11,12],times)!=publication_hash([11,12],times+1)
    assert publication_hash([11,12],times)==publication_hash([11],times[:1])+publication_hash([12],times[1:])


def test_independent_book_does_not_double_remove_fill_then_cancel():
    from spectrum.data_semantics_audit import ReferenceBook
    from spectrum.core.vendor.glbx_dbn import MBO_DTYPE
    expected={5:dict(valid=True,bid_nano=100,ask_nano=110,bid_size=3,ask_size=2)}
    book=ReferenceBook(expected)
    actions=[(b'R',b'N',0,0,0),(b'A',b'B',1,100,5),(b'A',b'A',2,110,2),
             (b'F',b'B',1,100,2),(b'C',b'B',1,100,2)]
    for i,(action,side,oid,price,size) in enumerate(actions,1):
        row=np.zeros(1,MBO_DTYPE);row['action']=action;row['side']=side;row['order_id']=oid;row['price']=price;row['size']=size
        book.feed(row[0],i)
    assert book.checks==1 and not book.errors and book.orders[1][2]==3
