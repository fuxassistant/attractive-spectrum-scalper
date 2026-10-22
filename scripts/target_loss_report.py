"""Visualize target, probabilities, loss and logit gradients together."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.special import expit
from scripts.flow_context_coverage import save, sha

DATA=Path('data/derived/target_loss_audit_v1');REPORT=Path('reports/target_loss_audit_v1')
C={'TAKE':'#16876b','STOP':'#c54958','TIME':'#c3982d','EMPTY':'#4284b9','UNKNOWN':'#b7b7b7'}
MODEL={'baseline':'#416aa8','rich_flow':'#6a48a8','rich_spectrum':'#bf437e','prior':'#777777','shuffled':'#aaaaaa'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})


def finish(fig,name):
    fig.savefig(REPORT/name,dpi=150,bbox_inches='tight',pad_inches=.2,facecolor='white');plt.close(fig)
    return name


def schema_plot(m):
    fig=plt.figure(figsize=(15,10),layout='constrained');gs=fig.add_gridspec(2,3,height_ratios=[1.1,1])
    ax=fig.add_subplot(gs[0,:]);ax.axis('off')
    ax.set_title('Actual learning contract: independent side-attempts, two proper classification losses',fontsize=14,pad=15)
    boxes=[(.02,.58,.24,.30,'Known decision-time X\n6 / 22 / 30 / 37 / 58 / 66 scalars\nNo row-order memory\nUniform weight per eligible row'),
           (.35,.62,.25,.27,'Fill head\ny = 1 for TAKE / STOP / TIME\ny = 0 for EMPTY\nUNKNOWN excluded'),
           (.35,.18,.25,.27,'Conditional path head\ny = one-hot TAKE / STOP / TIME\nOnly observed filled rows\nEMPTY / UNKNOWN excluded'),
           (.69,.39,.29,.32,'100 rounds of shallow trees\nmax_depth 3, min_leaf 500\nNewton leaf: -0.05 G / (H + 5)\nNo temporal smoothness penalty\nNo net/holding-time loss weight')]
    for x,y,w,h,text in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='#f1f5fa',edgecolor='#6d8599',transform=ax.transAxes))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',transform=ax.transAxes,fontsize=10)
    for start,end in [((.26,.73),(.35,.75)),((.26,.63),(.35,.32)),((.60,.75),(.69,.61)),((.60,.32),(.69,.48))]:
        ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=1.5))
    ax.text(.02,.02,'Future path chooses the hard class. Net magnitudes are used later for expected-net decisions, not for classifier gradients.',transform=ax.transAxes,fontsize=10)
    p=np.linspace(.005,.995,400)
    a=fig.add_subplot(gs[1,0]);a.plot(p,-np.log(p),color=C['TAKE'],label='y=1');a.plot(p,-np.log1p(-p),color=C['STOP'],label='y=0')
    a.set(title='Binary CE / target-class CE',xlabel='Predicted probability',ylabel='Per-row loss');a.legend()
    a=fig.add_subplot(gs[1,1]);a.plot(p,p-1,color=C['TAKE'],label='y=1: push probability up');a.plot(p,p,color=C['STOP'],label='y=0: push probability down');a.axhline(0,color='gray')
    a.set(title='Logit gradient g = p - y',xlabel='Predicted probability',ylabel='Gradient, not the tree update');a.legend(fontsize=8)
    a=fig.add_subplot(gs[1,2]);a.plot(p,p*(1-p),color='#416aa8')
    error=max(x['gradient_max_error'] for x in m['derivative_parity'])
    a.set(title='Hessian diagonal h = p(1-p)',xlabel='Predicted probability',ylabel='Curvature')
    a.text(.05,.07,f'Native sklearn parity error: {error:.2g}\nMulticlass uses one gradient per class;\nfull off-diagonal Hessian is not used.',transform=a.transAxes,fontsize=8)
    return finish(fig,'01_learning_scheme.png')


def distribution_plot(m):
    fig,ax=plt.subplots(2,2,figsize=(15,10),layout='constrained');d=pd.DataFrame(m['class_distribution'])
    fig.suptitle('Examples and gradient mass | all rows are not independent observations',fontsize=14)
    for a,mode in zip(ax[0],['P','T']):
        months=sorted(d.month.unique());bottom=np.zeros(len(months));totals=d.loc[d['mode']==mode].groupby('month').n.sum()
        for cat in ['TAKE','STOP','TIME','EMPTY','UNKNOWN']:
            v=d.loc[(d['mode']==mode)&(d.category==cat)].set_index('month').n.reindex(months).to_numpy()/totals.reindex(months).to_numpy()
            a.bar(months,v,bottom=bottom,color=C[cat],label=cat);bottom+=v
        a.axvline(2.5,color='black',ls='--');a.set(title=mode+' targets: Oct-Dec FIT | Jan-Feb evaluation',ylabel='Fraction of side-attempts');a.legend(ncol=3,fontsize=8);a.tick_params(axis='x',rotation=20)
        for i,month in enumerate(months):a.text(i,1.02,f'{totals[month]:,}',ha='center',fontsize=8)
    gg=pd.DataFrame(m['gradient_groups'])
    for a,head,cats in [(ax[1,0],'fill',['TAKE','STOP','TIME','EMPTY']),(ax[1,1],'path',['TAKE','STOP','TIME'])]:
        g=gg.loc[(gg['mode']=='P')&(gg['head']==head)&(gg.split=='FIT')&(gg.group=='class')]
        for offset,phase,field,label in [(-.25,'initial','n','Row share'),(0,'initial','grad_abs_sum','Initial |gradient| share'),(.25,'final','grad_abs_sum','Final |gradient| share')]:
            v=g.loc[g.phase==phase].set_index('value')[field].reindex(cats).to_numpy();a.bar(np.arange(len(cats))+offset,v/v.sum(),width=.24,label=label)
        a.set(xticks=range(len(cats)),xticklabels=cats,ylabel='Share within this head',title=f'P {head}, actual FIT: class frequency != gradient mass');a.legend(fontsize=8)
    return finish(fig,'02_examples_and_gradients.png')


def temporal_plot(m):
    fig,ax=plt.subplots(2,2,figsize=(15,10),layout='constrained');lags=[30,60,120,300,600,1800]
    fig.suptitle('Jagged outcomes and smooth forecasts are different objects | exact support/masks, no gap interpolation',fontsize=13)
    for key,label,color in [('take_conditional','Observed TAKE, filled run',C['TAKE']),('take_conditional_permuted','Within-day shuffled labels','#999999'),('p_take_conditional','Model P(TAKE | fill)','#bf437e'),('g_take','Final gradient residual','#416aa8')]:
        rows=[m['temporal'][f'P/B/{key}/{lag}'] for lag in lags]
        vals=[r.get('correlation',np.nan) if r['n']>=100 else np.nan for r in rows]
        ax[0,0].plot(lags,vals,'o-',label=label,color=color)
    ax[0,0].set(xscale='log',xlabel='Lag seconds',ylabel='Pearson correlation',ylim=(-.1,1.05),title='B/P conditional series (points need >=100 pairs)');ax[0,0].legend(fontsize=8)
    for key,label,color in [('take_conditional','Hard TAKE target',C['TAKE']),('p_take_conditional','Model P(TAKE | fill)','#bf437e'),('g_take','Logit-gradient residual','#416aa8')]:
        rows=[m['temporal'][f'P/B/{key}/{lag}'] for lag in lags]
        ax[0,1].plot(lags,[r.get('mean_abs_change',np.nan) if r['n']>=100 else np.nan for r in rows],'o-',label=label,color=color)
    ax[0,1].set(xscale='log',xlabel='Lag seconds',ylabel='Mean absolute change',title='Same filled-run pairs; low forecast variation is not skill');ax[0,1].legend(fontsize=8)
    for mode,style in [('P','-'),('T','--')]:
        for name,color in [('fill','#416aa8'),('class_conditional','#c54958')]:
            ll=[1,2,5,10,30,60];rows=[m['dense_temporal'][f'{mode}/{name}/{v}'] for v in ll]
            ax[1,0].plot(ll,[r.get('change_rate',np.nan) for r in rows],style+'o',color=color,label=f'{mode} {name}')
    ax[1,0].set(xscale='log',xlabel='Origin shift seconds',ylabel='Fraction of pairs changing class',title='Dense 1-second source: five fixed calendar days');ax[1,0].legend(fontsize=8)
    names=['fill','take_conditional','class_conditional'];x=np.arange(len(names))
    for offset,mode in [(-.18,'P'),(.18,'T')]:
        vals=[m['runs'][f'{mode}/B/{k}']['median_observed_span_seconds'] for k in names]
        bars=ax[1,1].bar(x+offset,vals,width=.36,label=mode)
        for bar,v in zip(bars,vals):ax[1,1].text(bar.get_x()+bar.get_width()/2,v+2,f'{v:.0f}s',ha='center',fontsize=8)
    ax[1,1].set(xticks=x,xticklabels=names,ylabel='Median observed grid span, seconds',title='Equal-label runs at 30s spacing; not trade duration');ax[1,1].legend()
    return finish(fig,'03_temporal_roughness.png')


def training_plot(m):
    fig,ax=plt.subplots(2,2,figsize=(15,10),layout='constrained');cur=pd.DataFrame(m['staged_curves'])
    fig.suptitle('Saved B-stage P rich_spectrum | actual FIT gradients vs evaluation residuals',fontsize=14)
    for head,a in [('fill',ax[0,0]),('path',ax[0,1])]:
        for split,color in [('FIT','#416aa8'),('B','#bf437e')]:
            d=cur.loc[(cur['head']==head)&(cur.split==split)]
            a.plot(d.iteration,d.loss,'o-',label=split,color=color)
        a.set(title=head+' native loss through saved boosting stages',xlabel='Completed iterations (0 = fitted class prior)',ylabel='Mean log-loss');a.legend()
    for head,color in [('fill','#416aa8'),('path','#bf437e')]:
        for split,style in [('FIT','-'),('B','--')]:
            d=cur.loc[(cur['head']==head)&(cur.split==split)]
            ax[1,0].plot(d.iteration,d.grad_sq_mean/d.grad_sq_mean.iloc[0],style+'o',label=f'{head} {split}',color=color)
    ax[1,0].set(title='Remaining squared logit-gradient magnitude',xlabel='Completed iterations',ylabel='Relative to initial prior');ax[1,0].legend(fontsize=8)
    leaves=m['first_tree_leaves']
    for head,color in [('fill','#416aa8'),('path','#bf437e')]:
        d=[r for r in leaves if r['head']==head]
        ax[1,1].scatter([r['cancellation_ratio'] for r in d],[abs(r['saved_update']) for r in d],s=[max(15,r['n']/300) for r in d],alpha=.7,color=color,label=head)
    ax[1,1].set(title='Actual first-tree leaves: cancellation and Newton update',xlabel='|sum(g)| / sum(|g|), 0 = full cancellation',ylabel='Absolute saved logit update (includes lr=.05)');ax[1,1].legend()
    error=max(r['error'] for r in leaves);ax[1,1].text(.03,.95,f'All {len(leaves)} leaves reconstructed\nmax error {error:.2g}; bubble size = row count',va='top',transform=ax[1,1].transAxes,fontsize=8)
    return finish(fig,'04_training_gradient_and_leaves.png')


def guess_plot(m,frame):
    fig,ax=plt.subplots(2,2,figsize=(15,10),layout='constrained');names=['prior','baseline','rich_flow','rich_spectrum','shuffled'];s=m['scores']
    fig.suptitle('What is the model guessing? | B/P, after-cost admission remains WAIT',fontsize=14)
    for i,key in enumerate(['fill_loss','path_loss']):
        vals=[s[f'P/{name}/B'][key] for name in names];ax[0,i].bar(names,vals,color=[MODEL[k] for k in names]);ax[0,i].set(title=key+' (train-frequency prior included)',ylabel='Log-loss, lower better');ax[0,i].tick_params(axis='x',rotation=20)
        ax[0,i].set_ylim(min(vals)*.98,max(vals)*1.02)
        for j,v in enumerate(vals):ax[0,i].text(j,v,f'{v:.5f}',ha='center',va='bottom',fontsize=8)
    r=s['P/rich_spectrum/B'];n=r['n_filled'];x=np.arange(3)
    ax[1,0].bar(x-.2,np.array(r['class_observed'])/n,.4,label='Observed class',color='#416aa8')
    ax[1,0].bar(x+.2,np.array(r['conditional_argmax_counts'])/n,.4,label='Argmax predicted class',color='#bf437e')
    ax[1,0].set(xticks=x,xticklabels=['TAKE','STOP','TIME'],title='Most likely class can be STOP while probabilities vary',ylabel='Fraction of known fills');ax[1,0].legend()
    b=frame.loc[(frame.split=='B')&(frame.category>=0)&(frame.category<3)].copy();b['take']=(b.category==0).astype(float)
    d=b.groupby('date')[['take','pc0_rich_spectrum']].mean();tt=pd.to_datetime(d.index)
    ax[1,1].plot(tt,d['take'],label='Observed daily TAKE rate',color=C['TAKE']);ax[1,1].plot(tt,d.pc0_rich_spectrum,label='Mean model P(TAKE | fill)',color='#bf437e')
    ax[1,1].set(title='Daily probability vs realized rate (not independent trials)',ylabel='Probability / frequency');ax[1,1].legend(fontsize=8);ax[1,1].tick_params(axis='x',rotation=20)
    return finish(fig,'05_prior_and_model_guesses.png')


def time_scene(record,frame,zoom=False):
    date=record['date'];dense=pd.read_parquet(DATA/(date+'_dense.parquet'));d=dense.loc[dense['mode']=='P'];london=int(record['london_ns'])
    root=frame.loc[(frame.date==date)&(frame.side==1)].sort_values('t_ns').copy();root.index=pd.to_datetime(root.t_ns,unit='ns',utc=True)
    low=london+60*60*10**9 if zoom else int(d.t_ns.min());high=london+120*60*10**9 if zoom else int(d.t_ns.max())
    fig,ax=plt.subplots(5,1,figsize=(16,12),sharex=True,gridspec_kw={'height_ratios':[1.6,.65,1,1,1.2]});fig.subplots_adjust(left=.08,right=.97,bottom=.08,top=.90,hspace=.28)
    split='FIT: final-model training residual' if date<'2026-01' else 'B: evaluation residual, not a training update'
    fig.suptitle(f'{date} | {split} | '+('fixed London +60 to +120 min' if zoom else 'full supported session')+'\nEach origin asks a separate 24/12-tick, 300-second question. Targets are future facts; forecasts use their own past.',fontsize=12)
    dl=d.loc[d.side==1].sort_values('t_ns');x=(dl.t_ns.to_numpy()-london)/3.6e12
    mid=dl.mid.to_numpy().copy();breaks=np.r_[False,(np.diff(dl.t_ns)!=10**9)|(np.diff(dl.epoch)!=0)];mid[breaks]=np.nan
    ax[0].plot(x,mid,color='#416aa8',lw=.7);ax[0].set_ylabel('GC mid $/oz')
    grid=np.arange(int(d.t_ns.min()),int(d.t_ns.max())+10**9,10**9,dtype=np.int64)
    matrix=np.stack([d.loc[d.side==side].set_index('t_ns').category.reindex(grid).fillna(-1).to_numpy()+1 for side in (1,-1)])
    ax[1].imshow(matrix,aspect='auto',interpolation='nearest',origin='upper',extent=[(grid[0]-london)/3.6e12,(grid[-1]-london)/3.6e12,1.5,-.5],vmin=0,vmax=4,cmap=ListedColormap([C[k] for k in ['UNKNOWN','TAKE','STOP','TIME','EMPTY']]))
    ax[1].set_yticks([0,1],['LONG','SHORT']);ax[1].set_title('Dense 1s hard labels: green TAKE | red STOP | gold TIME | blue EMPTY | grey unknown',fontsize=9)
    rx=(root.t_ns.to_numpy()-london)/3.6e12;known=root.category>=0;filled=known&(root.category<3)
    for a,field,title in [(ax[2],'pf_','P(fill)'),(ax[3],'pc0_','P(TAKE | fill)')]:
        for family in ['baseline','rich_flow','rich_spectrum']:
            values=root[field+family].to_numpy().copy();values[~root.audit_support.to_numpy()]=np.nan
            # Reindexing is deliberately avoided: draw only original forecasts.
            a.scatter(rx,values,s=4,color=MODEL[family],label=family,alpha=.85)
        a.set(ylim=(-.03,1.03),ylabel=title)
    fill_y=pd.Series(np.where(known,filled.astype(float),np.nan),index=root.index)
    take_y=pd.Series(np.where(filled,(root.category==0).astype(float),np.nan),index=root.index)
    for a,y in [(ax[2],fill_y),(ax[3],take_y)]:
        local=y.rolling('20min',center=True,min_periods=15).mean()
        a.plot(rx,local.to_numpy(),color='#d28b18',lw=1.3,label='20min centered realized rate: HINDSIGHT, not target / true probability')
        a.legend(fontsize=7,loc='upper right',ncol=2)
    g=root.g_path.to_numpy();show=np.isfinite(g)
    ax[4].vlines(rx[show],0,g[show],colors=np.where(g[show]<0,C['TAKE'],C['STOP']),lw=.5,alpha=.6)
    ax[4].scatter(rx[show],g[show],s=5,c=np.where(g[show]<0,C['TAKE'],C['STOP']))
    ax[4].axhline(0,color='gray',lw=.6);ax[4].set(ylabel='TAKE-logit gradient p - y',ylim=(-1.02,1.02),xlabel='Hours since London open (each probability point has its own causal past)')
    ax[4].set_title('Green negative gradient pushes TAKE up; red positive pushes it down. EMPTY/unknown have no path gradient.',fontsize=9)
    for a in ax:a.set_xlim((low-london)/3.6e12,(high-london)/3.6e12)
    return finish(fig,f'scene_{date}_'+('zoom' if zoom else 'session')+'.png')


def main():
    m=json.loads((REPORT/'measurements.json').read_text(encoding='utf-8'));frame=pd.read_parquet(DATA/'frame_P.parquet')
    files=[schema_plot(m),distribution_plot(m),temporal_plot(m),training_plot(m),guess_plot(m,frame)]
    for r in m['dense_dates']:
        files.append(time_scene(r,frame))
        if r['date'][:7] in ['2025-10','2026-01','2026-02']:files.append(time_scene(r,frame,True))
    body=''.join(f'<figure><img src="{html.escape(f)}" alt="{html.escape(f)}"><figcaption>{html.escape(f)}</figcaption></figure>' for f in files)
    page='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Target & loss audit</title><style>body{max-width:1600px;margin:30px auto;padding:0 22px;font:17px system-ui;background:#f3f5f7;color:#17232b}figure{margin:24px 0;background:white;padding:12px;border-radius:12px}img{width:100%;height:auto}figcaption{padding:8px;color:#506470}a{color:#146b86}</style><h1>Чему учится модель: target, loss и градиент</h1><p>Новый аудит сохранённого flow/context v2. Обучение не менялось. Жёсткие исходы, вероятности, локальная hindsight-частота и градиенты показаны раздельно.</p><p><a href="REPORT_RU.md">Результаты</a> · <a href="SOL61_TARGET_LOSS_AUDIT_RU.md">Независимый аудит Sol 6.1</a></p>'+body+'</html>'
    (REPORT/'index.html').write_text(page,encoding='utf-8',newline='\n')
    save(REPORT/'visual_receipt.json',dict(images={f:sha(REPORT/f) for f in files},measurements_sha256=sha(REPORT/'measurements.json'),script_sha256=sha(__file__)))
    print(json.dumps(dict(images=len(files),status='COMPLETE')))


if __name__=='__main__':main()
