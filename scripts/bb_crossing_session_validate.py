"""Validate full-session marks and explain selected unresolved execution."""
import gc
import json
import numpy as np
import pandas as pd
from spectrum.cognition import bb_crossing as bb
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay,QUOTE_CACHE
from scripts.bb_crossing_build import DATA,REPORT,MARKET
from scripts.flow_context_coverage import save,sha


def main():
    directory=DATA/'first_session';man=json.loads((directory/'manifest.json').read_text())
    parity=0;error=0.;late_checks=0;late_error=0.
    for day in man['days']:
        date=day['date']
        for suffix,expected in day['files'].items():
            if sha(directory/(date+'_'+suffix+'.parquet'))!=expected:raise ValueError('Full session hash drift')
        full=pd.read_parquet(directory/(date+'_profiles.parquet'))
        old=pd.read_parquet(DATA/(date+'_profiles.parquet'))
        f=old.merge(full,on=['date','t_ns','scale'],suffixes=('_old','_full'),validate='one_to_one')
        known=(f.execution_status==1)&(f.cross_ordinal_old>=0)
        if not f.loc[known,'cross_known'].all():raise ValueError('Lost short-horizon mature crossing')
        if not (f.loc[known,'cross_ordinal_old']==f.loc[known,'cross_ordinal_full']).all():raise ValueError('Changed first physical event')
        for a,b in [('mark_L_old','mark_L_full'),('mark_S_old','mark_S_full'),('cross_mae_L','mae_L'),('cross_mae_S','mae_S'),('cross_mfe_L','mfe_L'),('cross_mfe_S','mfe_S')]:
            if known.any():error=max(error,float(np.max(np.abs(f.loc[known,a]-f.loc[known,b]))))
        parity+=int(known.sum())
        if date in bb.contract()['sensitivity_dates']:
            source=MonthDay(MARKET/date[:7]/date);q=QuoteTape.from_day(source)
            for s in bb.SCALES:
                late=full.loc[(full.scale==s)&full.cross_known&full.later_than_30m]
                if not len(late):continue
                chosen=late.iloc[np.linspace(0,len(late)-1,min(8,len(late))).astype(int)]
                for r in chosen.itertuples():
                    e=r.entry_ordinal;x=r.cross_ordinal
                    if not q.valid[e:x+1].all() or (q.epoch[e:x+1]!=q.epoch[e]).any():raise ValueError('Late mark crosses unsupported quotes')
                    want=np.array([(q.bid[x]-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[x])/1e9-.3,
                        (q.bid[e:x+1].min()-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[e:x+1].max())/1e9-.3,
                        (q.bid[e:x+1].max()-q.ask[e])/1e9-.3,(q.bid[e]-q.ask[e:x+1].min())/1e9-.3])
                    got=np.array([r.mark_L,r.mark_S,r.mae_L,r.mae_S,r.mfe_L,r.mfe_S]);late_error=max(late_error,float(abs(want-got).max()));late_checks+=1
            QUOTE_CACHE.clear();del q,source;gc.collect()
    if error>1e-9 or late_error>1e-9:raise ValueError('Full-session money parity')
    summary=json.loads((REPORT/'summary.json').read_text());censors=[]
    for stage in ['A2','B']:
        for s in bb.SCALES:
            censor=summary['stages'][stage]['economics'][str(s)]['candidate_account']['censor']
            if not censor:continue
            date=censor['date'];t=int(censor['t_ns']);source=MonthDay(MARKET/date[:7]/date);q=QuoteTape.from_day(source)
            profile=pd.read_parquet(DATA/(date+'_profiles.parquet'));r=profile.loc[(profile.scale==s)&(profile.t_ns==t)].iloc[0]
            root=pd.read_parquet(DATA/(date+'_roots.parquet'));support=int(root.loc[root.t_ns==t,f'm{s}_support_until_ns'].iloc[0])
            e=int(r.entry_ordinal);x=int(r.exit_ordinal);req=int(r.request_ns)
            if req>=support:reason='GEOMETRY_SUPPORT_LOST'
            elif x<0:reason='NO_FUTURE_EXIT_PUBLICATION'
            elif not q.valid[e:x+1].all() or (q.epoch[e:x+1]!=q.epoch[e]).any() or (np.diff(q.quote_t[e:x+1])>bb.MINUTE).any():reason='QUOTE_SUPPORT_LOST'
            elif q.quote_t[x]>req+2*bb.NS:reason='EXIT_TIMEOUT_WITH_SUPPORTED_QUOTE_PATH'
            else:reason='UNCLASSIFIED_REQUIRES_REVIEW'
            censors.append(dict(stage=stage,scale=s,date=date,t_ns=str(t),entry_ns=str(int(r.entry_ns)),
                request_ns=str(req),first_future_quote_after_latency_ns=str(int(r.exit_ns)),
                first_exit_quote_delay_s=(int(r.exit_ns)-req)/bb.NS if x>=0 else None,
                reason=reason,mark_L=float(r.mark_L) if pd.notna(r.mark_L) else None,mark_S=float(r.mark_S) if pd.notna(r.mark_S) else None))
            QUOTE_CACHE.clear();del q,source;gc.collect()
    if any(r['reason']=='UNCLASSIFIED_REQUIRES_REVIEW' for r in censors):raise ValueError('Unexplained execution censor')
    save(REPORT/'full_session_validation.json',dict(status='PASS',script_sha256=sha(__file__),
        short_horizon_profile_parity=parity,money_max_error=error,independent_late_profile_checks=late_checks,late_money_max_error=late_error,
        manifest_sha256=sha(directory/'manifest.json')))
    save(REPORT/'execution_censor_diagnostics.json',dict(status='COMPLETE',censors=censors,
        meaning='A >2s absence of a new executable publication is a failure to resolve the chosen execution recipe, not automatically lost book support. Original accounts are not silently continued.'))
    print(json.dumps(dict(profile_parity=parity,late_checks=late_checks,max_error=max(error,late_error),censors=censors)),flush=True)


if __name__=='__main__':main()
