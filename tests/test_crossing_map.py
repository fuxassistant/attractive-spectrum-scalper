import numpy as np
from spectrum.crossings.map import vwap_geometry
from spectrum.crossings.map import NS


def test_geometry_identity_depends_on_scale_and_source_only():
    import spectrum.crossings.map as module
    parts=[dict(quote_file=dict(sha256='raw'),start_ns=1,end_ns=2)]
    before=module.geometry_id('VWAP',5,parts,'trades')
    assert module.geometry_id('VWAP',5,parts,'trades')==before
    assert module.geometry_id('VWAP',6,parts,'trades')!=before


def test_same_timestamp_future_publication_does_not_enter_vwap():
    t=np.array([60,60,60])*NS;k=np.arange(3);p=np.array([100.,110.,120.]);q=np.ones(3)
    m,h,s=vwap_geometry(t,k,k,t,p,q,1,np.zeros(3,dtype=int))
    assert m[1]==105 and h[1]==10
    mt,ht,_=vwap_geometry(t[:2],k[:2],k[:2],t[:2],p[:2],q[:2],1,np.zeros(2,dtype=int))
    np.testing.assert_allclose(m[:2],mt,equal_nan=True);np.testing.assert_allclose(h[:2],ht,equal_nan=True)
