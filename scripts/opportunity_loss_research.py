"""Analytical loss probes and current crossing-target semantics; no learner fit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.flow_context_coverage import save, sha

OUT = Path('reports/opportunity_loss_research_v1')
DATA = Path('data/derived/crossing_map_target_v1')


def focal(p, y, gamma):
    pt = np.where(np.asarray(y) == 1, p, 1-p)
    return -(1-pt)**gamma*np.log(pt)


def focal_gradient(p, y, gamma):
    pt = np.where(np.asarray(y) == 1, p, 1-p)
    return (p-y)*(1-pt)**gamma*(1-gamma*pt*np.log(pt)/(1-pt))


def probability_loss(p, y, kind):
    ce = -y*np.log(p)-(1-y)*np.log1p(-p)
    if kind == 'BCE':
        return ce
    if kind == 'BCE+Brier':
        return .5*ce+.5*(p-y)**2
    if kind == 'proper_confidence':
        # epsilon=1/3: dL/dz=(p-y)*[epsilon+(1-epsilon)*(2p-1)^2].
        # Integrating that derivative gives an actual proper scoring rule.
        return ce-4/3*(p-y)**2
    return focal(p, y, float(kind.split('=')[1]))


def probability_gradient(p, y, kind):
    if kind == 'BCE':
        return p-y
    if kind == 'BCE+Brier':
        return (p-y)*(.5+p*(1-p))
    if kind == 'proper_confidence':
        return (p-y)*(1/3+2/3*(2*p-1)**2)
    return focal_gradient(p, y, float(kind.split('=')[1]))


def finish(fig, name):
    fig.savefig(OUT/name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text())
    old_summary=json.loads(Path('reports/crossing_map_target_v1/summary.json').read_text())
    assert manifest['status']=='COMPLETE'
    assert sha(DATA/'manifest.json')==old_summary['manifest_sha256']
    assert sha(DATA/'targets.parquet')==old_summary['targets_sha256']
    frames=[]
    for day in manifest['days']:
        for suffix in ['map','roots']:
            name=day['date']+'_'+suffix+'.parquet'
            assert sha(DATA/name)==day['files'][name]
        m=pd.read_parquet(DATA/(day['date']+'_map.parquet'))
        r=pd.read_parquet(DATA/(day['date']+'_roots.parquet'))
        frames.append(m.merge(r[['t_ns','entry_spread']],on='t_ns',validate='many_to_one'))
    mapping=pd.concat(frames,ignore_index=True)
    resolved=mapping.loc[mapping.cross_ordinal>=0]
    # For a COMMON checkpoint: NL-NS=2*delta(mid); NL+NS=-(spread_e+spread_u+2fee).
    err=float(np.abs(resolved.mark_L-resolved.mark_S-2*resolved.mid_delta).max())
    exit_spread=-resolved.mark_L-resolved.mark_S-resolved.entry_spread-.6
    assert err<1e-8 and exit_spread.min()>0
    same_checkpoint_both=int(((resolved.mark_L>0)&(resolved.mark_S>0)).sum())
    assert same_checkpoint_both==0
    t=pd.read_parquet(DATA/'targets.parquet')
    joint=[]
    for family in ['BB','VWAP']:
        for d in [300,900,1800]:
            a=t.loc[(t.family==family)&(t.scale_set=='fast6')&(t.deadline==d)&t.all_known]
            for name in ['raw','amplitude','safe']:
                L=a[name+'_L'].to_numpy(bool);S=a[name+'_S'].to_numpy(bool)
                joint.append(dict(family=family,deadline=d,target=name,n=len(a),
                    none=int((~L&~S).sum()),long_only=int((L&~S).sum()),short_only=int((~L&S).sum()),both=int((L&S).sum()),
                    positive_long=float(L.mean()),positive_short=float(S.mean()),
                    complement_failure_fraction=float((L==S).mean())))
    kinds=['BCE','focal=0.5','focal=1','focal=2','BCE+Brier','proper_confidence']
    eta=np.linspace(.01,.99,99);bayes={};errors={};examples=[]
    for kind in kinds:
        minimizers=[]
        for e in eta:
            f=lambda p:e*probability_loss(p,1,kind)+(1-e)*probability_loss(p,0,kind)
            opt=minimize_scalar(f,bounds=(1e-8,1-1e-8),method='bounded',options={'xatol':1e-12})
            assert opt.success
            minimizers.append(float(opt.x))
        bayes[kind]=minimizers
        checks=[]
        for y in [0,1]:
            for z in np.linspace(-7,7,61):
                dz=1e-5
                finite=(probability_loss(expit(z+dz),y,kind)-probability_loss(expit(z-dz),y,kind))/(2*dz)
                checks.append(abs(finite-probability_gradient(expit(z),y,kind)))
        errors[kind]=float(max(checks));assert errors[kind]<1e-7
        if kind in ['BCE','BCE+Brier','proper_confidence']:
            assert np.max(np.abs(np.asarray(minimizers)-eta))<2e-6
        for p in [.01,.1,.5,.9,.99]:
            examples.append(dict(loss=kind,p_true=p,abs_logit_gradient=float(abs(probability_gradient(p,1,kind))),
                                 multiplier_vs_BCE=float(probability_gradient(p,1,kind)/(p-1))))
    # A deliberately ideal opposite-label example, not gradients of a trained trading model.
    rng=np.random.default_rng(61022);X=rng.normal(size=(64,5));w=rng.normal(size=5)
    z=X@w;y=(rng.random(64)>.5).astype(float)
    gL=X.T@(expit(z)-y)/len(y)
    gS=(-X).T@(expit(-z)-(1-y))/len(y)
    cosine=float(np.dot(gL,gS)/(np.linalg.norm(gL)*np.linalg.norm(gS)))
    assert np.max(np.abs(gL-gS))<1e-12
    # Imbalance weights move the optimum even without focal modulation.
    weighted=[dict(eta=float(e),positive_weight_ratio=w,optimal_score=float(w*e/(w*e+1-e)))
              for e in [.1,.2,.5,.8,.9] for w in [.5,1.,2.]]
    fig,axes=plt.subplots(1,2,figsize=(14,5.4),layout='constrained')
    p=np.linspace(.001,.999,999)
    for kind in kinds:
        g=np.abs(probability_gradient(p,1,kind))
        axes[0].plot(p,g,label=kind);axes[1].plot(p,g/(1-p),label=kind)
    axes[0].set(xlabel='Probability of the observed true label',ylabel='Absolute d loss / d logit',title='One hard-label example; equal class weights')
    axes[1].set(xlabel='Probability of the observed true label',ylabel='Gradient multiplier relative to BCE',title='Focal suppresses easy cases; it does not identify noise')
    axes[0].legend(fontsize=9);axes[1].axhline(1,color='gray',ls=':',lw=1)
    fig.suptitle('Analytical derivatives checked by finite differences; no trained model',fontsize=14)
    finish(fig,'01_loss_gradients.png')
    fig,ax=plt.subplots(figsize=(8.5,7),layout='constrained')
    for kind in kinds:ax.plot(eta,bayes[kind],label=kind,lw=1.8)
    ax.set(xlabel='True conditional probability eta',ylabel='Score minimizing population expected loss',
           xlim=(0,1),ylim=(0,1),title='Focal scores need not equal probabilities\nUnrestricted constant-probability population calculation')
    ax.legend();ax.grid(alpha=.2)
    ax.text(.03,.61,'BCE, BCE+Brier and proper_confidence\ncoincide on the diagonal.',fontsize=10,
            bbox=dict(facecolor='white',edgecolor='#d0d0d0',alpha=.9))
    finish(fig,'02_population_calibration.png')
    fig,axes=plt.subplots(1,2,figsize=(14,6.3),layout='constrained',sharey=True)
    for ax,family in zip(axes,['BB','VWAP']):
        rows=[x for x in joint if x['family']==family];bottom=np.zeros(len(rows))
        for key,color in [('none','#9ca9b3'),('long_only','#2570aa'),('short_only','#cf733c'),('both','#479c7b')]:
            v=np.array([r[key]/r['n'] for r in rows])*100
            ax.bar(np.arange(len(rows)),v,bottom=bottom,label=key,color=color);bottom+=v
        ax.set(xticks=range(len(rows)),xticklabels=[str(r['deadline']//60)+'m\n'+r['target'] for r in rows],
               ylim=(0,112),ylabel='% of fully mature origins',title=family+' M1/M3/M6: side labels after searching the map')
        for i,r in enumerate(rows):ax.text(i,102,f"n={r['n']}",ha='center',fontsize=8)
        ax.legend(ncol=4,loc='upper center',fontsize=8)
    fig.suptitle('Same checkpoint: zero jointly positive LONG/SHORT cells. Different future checkpoints: BOTH is valid.\nUnknown excluded; across deadlines the admitted cohorts differ.',fontsize=13)
    finish(fig,'03_side_semantics.png')
    # Current cohort priors: gamma=2 does not magically add information to degenerate E.
    prior_probe=[]
    for name in ['raw','amplitude','safe']:
        row=next(r for r in joint if r['family']=='BB' and r['deadline']==900 and r['target']==name)
        prior=1-row['none']/row['n']
        for gamma in [0,.5,1,2]:
            gp=float(abs(focal_gradient(prior,1,gamma)));gn=float(abs(focal_gradient(prior,0,gamma)))
            mass_positive=prior*gp;mass_negative=(1-prior)*gn
            prior_probe.append(dict(target=name,gamma=gamma,constant_prior=prior,
                positive_gradient=gp,negative_gradient=gn,
                negative_share_abs_gradient=mass_negative/(mass_positive+mass_negative)))
    result=dict(status='COMPLETE_ANALYTICAL_RESEARCH_NOT_MODEL_BENCHMARK',
        source_manifest_sha256=sha(DATA/'manifest.json'),targets_sha256=sha(DATA/'targets.parquet'),
        script_sha256=sha(__file__),common_checkpoint=dict(cells=len(resolved),both_positive=same_checkpoint_both,
            max_delta_identity_error=err,minimum_inferred_exit_spread=float(exit_spread.min())),
        joint_side_labels=joint,gradient_checks=dict(points_per_loss=122,max_error_by_loss=errors),
        focal_bayes_minimizers={kind:[dict(eta=float(e),optimal_score=float(v)) for e,v in zip(eta,bayes[kind])] for kind in kinds},
        gradient_examples=examples,weighted_BCE_optima=weighted,constant_prior_gradient_mass=prior_probe,
        opposite_labels_toy=dict(shared_gradient_cosine=cosine,max_gradient_difference=float(np.max(abs(gL-gS))),
            meaning='Opposite labels plus opposite readouts give identical encoder gradients. Not a measurement on real backbone training.'),
        qualifications=['No predictive model was trained; no measured negative transfer or architecture ranking',
                        'The population minimizer calculation is not empirical model calibration',
                        'One development week with dependent origins, not out-of-sample profitability'],
        images={p.name:sha(p) for p in sorted(OUT.glob('*.png'))})
    save(OUT/'probes.json',result)
    print(json.dumps(dict(status=result['status'],checkpoint=result['common_checkpoint'],gradient_errors=errors,
                          toy_cosine=cosine,focal_eta_point1_gamma2=bayes['focal=2'][9])),flush=True)


if __name__=='__main__':main()
