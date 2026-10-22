"""Label distributions, maturity, scale redundancy and target roughness."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.crossing_map_build import OUT,REPORT,DATES
from scripts.flow_context_coverage import save,sha
from spectrum.cognition.crossing_map import SCALES,NS,factorized_loss

SETS={'fast5':[1,3,5],'fast6':[1,3,6],'with9':[1,3,6,9],'all':[1,3,5,6,9]}


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


def summarize_targets(t,name):
    y=t[name+'_state'].to_numpy();known=y>=0;values=y[known]
    consecutive=known[1:]&known[:-1]&(np.diff(t.t_ns.to_numpy())==30*NS)&(t.date.to_numpy()[1:]==t.date.to_numpy()[:-1])
    out=dict(total=len(t),mature=int(known.sum()),unknown=int((~known).sum()),
        counts={k:int((values==v).sum()) for v,k in enumerate(['NONE','LONG_ONLY','SHORT_ONLY','BOTH'])},
        event_rate=float((values>0).mean()) if len(values) else None,
        both_rate=float((values==3).mean()) if len(values) else None,
        same_30s_pairs=int(consecutive.sum()),
        event_flip=float(np.mean(((y[1:]>0)!=(y[:-1]>0))[consecutive])) if consecutive.any() else None,
        state_flip=float(np.mean((y[1:]!=y[:-1])[consecutive])) if consecutive.any() else None)
    return out


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    if manifest['status']!='COMPLETE':raise ValueError('Incomplete map')
    for d in manifest['days']:
        for name,h in d['files'].items():
            if sha(OUT/name)!=h:raise ValueError('Changed map')
    f=pd.concat([pd.read_parquet(OUT/(date+'_map.parquet')) for date in DATES],ignore_index=True)
    roots=pd.concat([pd.read_parquet(OUT/(date+'_roots.parquet')) for date in DATES],ignore_index=True).sort_values('t_ns').reset_index(drop=True)
    bank=pd.concat([pd.read_parquet(OUT/(date+'_bank.parquet')) for date in DATES],ignore_index=True)
    labels={};target_frames=[];coverage={};redundancy={};interarrival={};extras={}
    for family in ['BB','VWAP']:
        a=f.loc[f.family==family];labels[family]={};coverage[family]={};redundancy[family]={};interarrival[family]={}
        for setname,scales in SETS.items():
            labels[family][setname]={}
            for d in [300,900,1800]:
                t=target_frame(a,roots,scales,d);t['scale_set']=setname;target_frames.append(t)
                labels[family][setname][str(d)]={name:summarize_targets(t,name) for name in ['raw','amplitude','safe']}
                known=t.all_known.to_numpy();pairs=known[1:]&known[:-1]&(np.diff(t.t_ns)==30*NS)
                u=np.maximum(t.quality_L,t.quality_S).to_numpy()
                labels[family][setname][str(d)]['soft_quality']=dict(mean=float(u[known].mean()),median=float(np.median(u[known])),
                    p90=float(np.quantile(u[known],.9)),mean_abs_change_30s=float(abs(np.diff(u))[pairs].mean()),
                    fraction_above_point5=float((u[known]>.5).mean()))
        extras[family]={}
        for mult in [.5,1.,2.]:
            t=target_frame(a,roots,SETS['fast6'],900,mult)
            extras[family][str(mult)]={name:summarize_targets(t,name) for name in ['amplitude','safe']}
        for s in SCALES:
            z=a.loc[a.scale==s];coverage[family][str(s)]={}
            for role in range(3):
                b=z.loc[z.role==role];ready=b.ready.to_numpy();known=b.cross_ordinal.to_numpy()>=0
                coverage[family][str(s)][str(role)]=dict(ready=int(ready.sum()),observed=int(known.sum()),
                    observed_by_seconds={str(d):float(((b.delay_s<=d)&ready).sum()/max(1,ready.sum())) for d in [300,900,1800,3600]},
                    delay_quantiles={str(p):float(b.loc[known,'delay_s'].quantile(p)) for p in [.1,.5,.9]},
                    median_abs_net=float(b.loc[known,'mark_L'].abs().median()),
                    long_positive_rate=float((b.loc[known,'mark_L']>0).mean()),short_positive_rate=float((b.loc[known,'mark_S']>0).mean()))
            rows=bank.loc[(bank.family==family)&(bank.scale==s)].copy();diffs=[]
            for _,part in rows.groupby('role'):
                delta=np.diff(part.t_ns.to_numpy())/NS
                diffs.extend(delta[delta>=0].tolist())
            interarrival[family][str(s)]=dict(intervals=len(diffs),under1_fraction=float((np.asarray(diffs)<1).mean()),median_s=float(np.median(diffs)))
        for s1,s2 in [(5,6),(6,9),(3,5)]:
            x=a.loc[a.scale==s1].set_index(['t_ns','role']);y=a.loc[a.scale==s2].set_index(['t_ns','role']).reindex(x.index)
            both=(x.cross_ordinal>=0)&(y.cross_ordinal>=0)
            dx=x.loc[both];dy=y.loc[both]
            redundancy[family][f'{s1}_{s2}']=dict(common_resolved_cells=int(both.sum()),
                exact_publication_fraction=float((dx.cross_ordinal==dy.cross_ordinal).mean()),
                delay_abs_median_s=float((dx.delay_s-dy.delay_s).abs().median()),
                delay_within60s_fraction=float(((dx.delay_s-dy.delay_s).abs()<=60).mean()),
                net_L_correlation=float(dx.mark_L.corr(dy.mark_L)),net_abs_median_delta=float((dx.mark_L-dy.mark_L).abs().median()))
    # Full-BBO first-passage benchmark is independent of which crossing family is used.
    first={}
    for d in [300,900,1800]:
        first[str(d)]={}
        for g,mult in enumerate([.5,1.,2.]):
            t=roots[['date','t_ns']].copy();states=[];knowns=[]
            for side in ['L','S']:
                hit=roots[f'gain_{side}_{g}'];stop=roots['stop_'+side];limit=roots.entry_ns+d*NS
                success=(hit>0)&(hit<=limit)
                known=(roots.entry_ns>0)&np.isfinite(roots.risk_R)&(success|((stop>0)&(stop<=limit))|(roots.quote_support_end_ns>limit))
                states.append(success.to_numpy());knowns.append(known.to_numpy())
            t['passage_state']=np.where(knowns[0]&knowns[1],states[0]+2*states[1],-1)
            first[str(d)][str(mult)]=summarize_targets(t,'passage')
    alltargets=pd.concat(target_frames,ignore_index=True);alltargets.to_parquet(OUT/'targets.parquet',index=False)
    incremental={};gradient_at_prior={};paired_sets={}
    for family in ['BB','VWAP']:
        incremental[family]={};gradient_at_prior[family]={};paired_sets[family]={}
        for d in [300,900,1800]:
            a=alltargets.loc[(alltargets.family==family)&(alltargets.scale_set=='fast6')&(alltargets.deadline==d)].set_index('t_ns')
            b=alltargets.loc[(alltargets.family==family)&(alltargets.scale_set=='with9')&(alltargets.deadline==d)].set_index('t_ns').reindex(a.index)
            common=a.all_known&b.all_known
            incremental[family][str(d)]=dict(common_mature=int(common.sum()),
                safe_event_added_by_m9=int(((a.safe_state==0)&(b.safe_state>0)&common).sum()),
                quality_median_gain=float((np.maximum(b.quality_L,b.quality_S)-np.maximum(a.quality_L,a.quality_S))[common].median()),
                quality_p90_gain=float((np.maximum(b.quality_L,b.quality_S)-np.maximum(a.quality_L,a.quality_S))[common].quantile(.9)))
            c=alltargets.loc[(alltargets.family==family)&(alltargets.scale_set=='fast5')&(alltargets.deadline==d)].set_index('t_ns').reindex(a.index)
            common=a.all_known&c.all_known
            paired_sets[family][str(d)]=dict(common_mature=int(common.sum()),
                amplitude_event_agreement=float(((a.amplitude_state>0)==(c.amplitude_state>0))[common].mean()),
                amplitude_direction_state_agreement=float((a.amplitude_state==c.amplitude_state)[common].mean()),
                safe_event_agreement=float(((a.safe_state>0)==(c.safe_state>0))[common].mean()),
                safe_only_fast5=int(((a.safe_state==0)&(c.safe_state>0)&common).sum()),
                safe_only_fast6=int(((a.safe_state>0)&(c.safe_state==0)&common).sum()))
        for name in ['raw','amplitude','safe']:
            a=alltargets.loc[(alltargets.family==family)&(alltargets.scale_set=='fast6')&(alltargets.deadline==900)&alltargets.all_known]
            train=a.date.isin(DATES[:2]);yy=(a[name+'_state']>0).to_numpy()
            prior=float(yy[train].mean());prior=float(np.clip(prior,1e-6,1-1e-6));eval_y=yy[~train]
            grad=prior-eval_y
            gradient_at_prior[family][name]=dict(fit_dates=DATES[:2],eval_dates=DATES[2:],fit_event_prior=prior,
                evaluation_rows=len(eval_y),evaluation_positive_rate=float(eval_y.mean()),
                mean_abs_logit_gradient=float(abs(grad).mean()),negative_example_abs_gradient=prior,positive_example_abs_gradient=1-prior,
                interpretation='Untrained constant fit-prior BCE diagnostic, not a fitted model')
    # Fixed mixed positive/negative/missing examples, finite differences for actual formula.
    z=np.array([-.7,.3,.8,.1]);dd=np.array([[.2,.1,-.2],[.3,.5,-.1],[.1,-.4,.9],[.3,.2,.1]])
    y=np.array([0,1,1,0]);direction=np.array([-1,0,2,-1]);known=np.array([1,1,1,0],bool);dk=np.array([0,1,1,0],bool)
    loss,ge,gd=factorized_loss(z,dd,y,direction,known,dk);errors=[]
    for array,gradient in [(z,ge),(dd,gd)]:
        for ix in np.ndindex(array.shape):
            old=array[ix];array[ix]=old+1e-6;p=factorized_loss(z,dd,y,direction,known,dk)[0]
            array[ix]=old-1e-6;m=factorized_loss(z,dd,y,direction,known,dk)[0];array[ix]=old
            errors.append(abs((p-m)/2e-6-gradient[ix]))
    save(REPORT/'summary.json',dict(status='COMPLETE',dates=DATES,origins=len(roots),cells=len(f),bank_events=len(bank),
        labels=labels,gain_sensitivity=extras,first_passage=first,coverage=coverage,redundancy=redundancy,interarrival=interarrival,
        interarrival_scope='Adjacent same-line timestamps across session banks, including gaps; no gap-based exclusion',
        incremental_m9=incremental,paired_fast5_fast6=paired_sets,event_gradient_at_fit_prior=gradient_at_prior,
        risk_query=dict(median_R=float(roots.risk_R.median()),p10_R=float(roots.risk_R.quantile(.1)),p90_R=float(roots.risk_R.quantile(.9)),
            cost_floor_fraction=float(np.isclose(roots.risk_R,2*(roots.decision_spread+.3))[roots.risk_R.notna()].mean())),
        factorized_gradient_max_error=max(errors),targets_sha256=sha(OUT/'targets.parquet'),
        code_sha256=sha(__file__),manifest_sha256=sha(OUT/'manifest.json'),
        interpretation='Class/quality stability and favorable frequency do not establish predictability or live profitability. All map checkpoints are alternative states of one fixed entry.'))
    print(json.dumps(dict(status='COMPLETE',origins=len(roots),cells=len(f),gradient_error=max(errors))),flush=True)


if __name__=='__main__':main()
