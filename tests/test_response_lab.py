import numpy as np
import pandas as pd
from spectrum.response_lab import window_features,spectral_features,split_dates,WINDOW,COUNTERS


def frame(n=750):
    rng=np.random.default_rng(6)
    d=pd.DataFrame({'t_ns':pd.Timestamp('2026-02-12T10:00Z').value+np.arange(n)*10**9,
                    'mid':4400+np.cumsum(rng.normal(0,.1,n))})
    d['bid']=d.mid-.1;d['ask']=d.mid+.1
    for name in COUNTERS:d['c_'+name]=np.cumsum(rng.integers(0,5,n))
    for side in ('bid','ask'):
        for depth in (1,4,12):d[f'depth_{side}_{depth}']=100*depth
    return d


def test_unknown_volume_retained_and_prefix_invariant():
    d=frame();actual=window_features(d,600)
    assert actual['A__unknown_fraction_512']>0
    assert actual['A__pressure_lower_512']<=actual['A__pressure_upper_512']
    assert actual==window_features(d.iloc[:601],600)
    d.loc[601:,'mid']=1e8;d.loc[601:,'c_trade_buy']=1e12
    assert actual==window_features(d,600)
    assert all(np.isfinite(x) for x in actual.values())


def test_averaging_prevents_degenerate_coherence_and_detects_copy():
    rng=np.random.default_rng(123);x=rng.normal(size=WINDOW);y=rng.normal(size=WINDOW)
    def run(y):return spectral_features({'return':y,'classified_flow':x,'volume':abs(x),'book_flow':x,'abs_return':abs(y)})[1]
    independent=run(y);same=run(2*x)
    assert independent['classified_flow_to_return_8_16_coherence']<.6
    assert abs(same['classified_flow_to_return_8_16_coherence']-1)<1e-12
    assert abs(np.expm1(same['classified_flow_to_return_8_16_gain_log'])-2)<1e-12


def test_zero_spectra_are_masked_without_dropping_window():
    auto,cross=spectral_features({n:np.zeros(WINDOW) for n in ['return','classified_flow','volume','book_flow','abs_return']})
    assert not any(auto.values()) and not any(cross.values())


def test_chronological_boundaries():
    splits=split_dates(['2026-02-28','2026-03-01','2026-04-30','2026-05-01'])
    assert splits['train'].tolist()==[True,False,False,False]
    assert splits['validation'].tolist()==[False,True,True,False]
    assert splits['evaluation'].tolist()==[False,False,False,True]
