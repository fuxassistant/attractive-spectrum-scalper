"""Reusable formulas extracted from scripts/crossing_map_analyze.py; see docs/MIGRATION.json."""
import numpy as np
import pandas as pd
from spectrum.core.schema import NS

def target_frame(f,roots,scales,deadline,gain_multiple=2.):
    a=f.loc[f.scale.isin(scales)].sort_values(['t_ns','scale','role'])
    n=len(roots);width=len(scales)*3
    if len(a)!=n*width:raise ValueError('Incomplete rectangular map')
    def mat(name):return a[name].to_numpy().reshape(n,width)
    e=roots.entry_ns.to_numpy();risk=roots.risk_R.to_numpy();delay=mat('delay_s')
    observed=(mat('cross_ordinal')>=0)&(delay<=deadline)
    mature=mat('ready')&(observed|(mat('support_end_ns')>e[:,None]+deadline*NS))
    all_known=mature.all(axis=1)&np.isfinite(risk)&(e>0)
    out=roots[['date','t_ns','entry_ns','risk_R']].copy();out['all_known']=all_known
    for side in ['L','S']:
        mark=mat('mark_'+side);mae=mat('mae_'+side)
        out['raw_'+side]=np.any(observed&(mark>0),axis=1)
        out['amplitude_'+side]=np.any(observed&(mark>=gain_multiple*risk[:,None]),axis=1)
        out['safe_'+side]=np.any(observed&(mark>=gain_multiple*risk[:,None])&(mae>-risk[:,None]),axis=1)
        quality=np.tanh(np.maximum(mark,0)/risk[:,None])*np.exp(-np.maximum(-mae,0)/risk[:,None])*np.exp(-delay/deadline)
        out['quality_'+side]=np.max(np.where(observed&np.isfinite(quality),quality,0),axis=1)
    for name in ['raw','amplitude','safe']:
        event=out[name+'_L']|out[name+'_S'];out[name+'_event_known']=all_known|event
        out[name+'_state']=np.where(all_known,out[name+'_L'].astype(int)+2*out[name+'_S'].astype(int),-1)
    out['family']=a.family.iloc[0];out['deadline']=deadline
    return out
