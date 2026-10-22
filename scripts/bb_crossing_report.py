"""Scientific visual evidence for the crossing-bank experiment."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from spectrum.cognition import bb_crossing as bb
from scripts.bb_crossing_build import DATA,REPORT
from scripts.flow_context_coverage import save,sha

COLORS=['#176b9b','#dd9224','#a53745','#5a56a6']
LINE_COLORS=['#347fbd','#666666','#d76535','#8e57ac','#bbbbbb']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white','axes.titleweight':'bold','savefig.facecolor':'white'})


def read(p):return json.loads(Path(p).read_text())
def finish(fig,name):
    fig.savefig(REPORT/name,dpi=145,bbox_inches='tight');plt.close(fig)
def clock(t):return pd.to_datetime(np.asarray(t),utc=True)
def format_time(ax):
    ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M',tz='UTC'))
    ax.set_xlabel('UTC')


def load_all(suffix):
    man=read(DATA/'manifest.json')
    return pd.concat([pd.read_parquet(DATA/(r['date']+'_'+suffix+'.parquet')) for r in man['days']],ignore_index=True)


def bank_figures():
    roots=load_all('roots');profiles=load_all('profiles');receipt=read(REPORT/'data_receipt.json')
    fig,ax=plt.subplots(figsize=(15,6));ax.axis('off')
    boxes=[(.01,.57,.30,.31,'CAUSAL X AT DECISION','65 flow / day / spectrum scalars\n+16 current BB geometry scalars\nNo future crossing ID in X'),
        (.35,.57,.30,.31,'FUTURE TARGET BANK','4 native scales × lower / middle / upper\nStrict reversals; touches carry side\nStable physical IDs + ties + censor'),
        (.69,.57,.30,.31,'TWO DIFFERENT QUESTIONS','Classify next line / global scale+line\nPredict mean executed net separately\nChronological A2 and exposed B'),
        (.01,.10,.46,.31,'WHAT ONE TRADE PRODUCES','Same actual entry → 4 first post-entry crossing marks\nEach point: ID, Δt, net, MAE / MFE\nThese marks are not four accumulated profits'),
        (.53,.10,.46,.31,'WHAT EVENT WEIGHTING CHANGES','Uniform queries vs equal total FIT weight per event\nEvaluate row score and event-macro score\nIDs do not make neighboring events independent')]
    for x,y,w,h,title,body in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='#edf3f7',edgecolor='#93a6b8',transform=ax.transAxes))
        ax.text(x+.02,y+h-.06,title,transform=ax.transAxes,weight='bold',fontsize=12,va='top')
        ax.text(x+.02,y+h-.13,body,transform=ax.transAxes,fontsize=9.5,va='top',linespacing=1.4)
    fig.suptitle('Learning contract: first crossings, event IDs and a separate execution experiment',fontsize=16,weight='bold')
    finish(fig,'01_learning_scheme.png')

    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained');names=['LOWER','MIDDLE','UPPER','TIE','NONE ≤30m','CENSORED']
    classes=[];mult=[];unique=[];delays=[]
    for s in bb.SCALES:
        y=roots[f'm{s}_y'];classes.append([float((y==k).mean()) for k in [0,1,2,3,4,-1]])
        ids=roots.loc[roots[f'm{s}_event_ns']>0,f'm{s}_id'].value_counts();mult.append(ids);unique.append(len(ids))
        delays.append(roots[f'm{s}_delay_s'].dropna())
    bottom=np.zeros(4)
    for k,name in enumerate(names):
        vals=np.array(classes)[:,k];axes[0,0].bar([str(s) for s in bb.SCALES],vals,bottom=bottom,label=name,color=(LINE_COLORS+['#333333'])[k]);bottom+=vals
    axes[0,0].set(title='Target composition — all 173,724 origins',xlabel='Native minutes',ylabel='Fraction of origins',ylim=(0,1));axes[0,0].legend(fontsize=8,ncol=3)
    for s,c,counts in zip(bb.SCALES,COLORS,mult):
        axes[0,1].hist(counts,bins=np.arange(.5,62,2),histtype='step',density=True,label=f'M{s}: {len(counts):,} events',color=c)
    axes[0,1].set(title='Queries sharing one event (cap 1800s / 30s = 60)',xlabel='Queries per first-cross packet',ylabel='Event density',yscale='log');axes[0,1].legend()
    axes[1,0].boxplot(delays,tick_labels=[f'M{s}' for s in bb.SCALES],showfliers=False)
    axes[1,0].set(title='Time to first crossing — resolved crossings only',ylabel='Seconds',yscale='log')
    known=[];empty=[];censor=[]
    for s in bb.SCALES:
        f=profiles.loc[profiles.scale==s];known.append((f.execution_status==1).mean());empty.append((f.execution_status==0).mean());censor.append((f.execution_status<0).mean())
    x=np.arange(4);axes[1,1].bar(x,known,label='Resolved execution');axes[1,1].bar(x,empty,bottom=known,label='No entry')
    axes[1,1].bar(x,censor,bottom=np.array(known)+empty,label='Censored / no geometry',color='#a74d4d')
    axes[1,1].set(xticks=x,xticklabels=[f'M{s}' for s in bb.SCALES],ylim=(0,1),title='Post-entry profile support — 100ms taker',ylabel='Fraction of roots');axes[1,1].legend(fontsize=9)
    finish(fig,'02_bank_and_examples.png')

    profile_summary={};fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for s,c in zip(bb.SCALES,COLORS):
        p=profiles.loc[(profiles.scale==s)&(profiles.execution_status==1)&(profiles.cross_id!='')]
        sample=p.iloc[::max(1,len(p)//3000)]
        axes[0].scatter(sample.cross_after_entry_s,sample.mark_L,s=3,alpha=.12,color=c,label=f'M{s}')
        latency=p.net_L-p.mark_L;axes[1].hist(latency,bins=np.linspace(-2,2,121),histtype='step',density=True,label=f'M{s}',color=c)
        keys=[0,10,30,60,120,300,600,1800];med=[];xx=[];bins=[]
        for lo,hi in zip(keys[:-1],keys[1:]):
            v=p.loc[(p.cross_after_entry_s>lo)&(p.cross_after_entry_s<=hi),'mark_L']
            if len(v):xx.append((lo+hi)/2);med.append((v>0).mean())
            bins.append(dict(lo=lo,hi=hi,queries=len(v),positive=float((v>0).mean()) if len(v) else None))
        axes[2].plot(xx,med,'o-',color=c,label=f'M{s}')
        profile_summary[str(s)]=dict(resolved_cross_profiles=len(p),mark_L_positive=float((p.mark_L>0).mean()),net_L_positive=float((p.net_L>0).mean()),
            future_delay_bins=bins,latency_plot_tail_fraction=float((abs(latency)>2).mean()),
            latency_L_quantiles={str(k):float(latency.quantile(k)) for k in [.01,.1,.5,.9,.99]},
            cross_time_quantiles={str(k):float(p.cross_after_entry_s.quantile(k)) for k in [.1,.5,.9]},
            crossing_before_entry_queries=int(profiles.loc[profiles.scale==s,'decision_event_before_entry'].sum()),
            long_short_net_sum_max=float((p.net_L+p.net_S).max()))
    axes[0].set(xscale='log',xlabel='Seconds from actual entry',ylabel='LONG liquidation net at crossing ($ / 1 oz)',title='Overlapping query profiles; not account equity')
    axes[0].axhline(0,color='black',lw=.8);axes[0].set_ylim(np.quantile(profiles.mark_L.dropna(),[.01,.99]));axes[0].legend()
    axes[1].set(title='Exit latency effect: executed net − crossing mark',xlabel='$ / 1 oz (plot clipped to ±2; tails in JSON)',ylabel='Density');axes[1].legend()
    axes[2].set(title='Future-delay strata (bin counts in profile_summary.json)',xscale='log',xlabel='Realized delay-bin midpoint (seconds); not a live signal',ylabel='Fraction of LONG marks > 0',ylim=(0,1));axes[2].axhline(.5,color='black',ls='--',lw=.7);axes[2].legend()
    finish(fig,'03_time_and_net_profiles.png')
    save(REPORT/'profile_summary.json',profile_summary)

    scene_meta=[]
    for date in bb.contract()['visual_dates']:
        root=roots.loc[roots.date==date];profile=profiles.loc[profiles.date==date]
        good=profile.loc[(profile.execution_status==1)&(profile.cross_id!='')].groupby('t_ns').size()
        origin=int(good.index[good==4][0]);chosen=profile.loc[profile.t_ns==origin].sort_values('scale')
        scene=pd.read_parquet(DATA/(date+'_scene.parquet'));bank=pd.read_parquet(DATA/(date+'_bank.parquet'))
        start=origin-120*bb.NS;end=origin+max(900,int(chosen.cross_after_decision_s.max()+60))*bb.NS
        view=scene.loc[scene.t_ns.between(start,end)];events=bank.loc[bank.t_ns.between(view.t_ns.min(),view.t_ns.max())]
        fig,axes=plt.subplots(4,1,figsize=(15,12),sharex=True,layout='constrained')
        for ax,s in zip(axes,bb.SCALES):
            ax.plot(clock(view.t_ns),view.mid,color='#142c3d',lw=1.1,label='GC midpoint (5s display)')
            for r,offset in enumerate([-1,0,1]):
                ax.plot(clock(view.t_ns),view[f'm{s}']+offset*view[f'h{s}'],color=LINE_COLORS[r],lw=.8,alpha=.8,label=bb.ROLES[r])
            e=events.loc[events.scale==s];ax.scatter(clock(e.t_ns),e.mid,c=[LINE_COLORS[r] for r in e.role],s=7,alpha=.45)
            row=chosen.loc[chosen.scale==s].iloc[0];ev=bank.loc[(bank.scale==s)&(bank.ordinal==row.cross_ordinal)].iloc[0]
            ax.scatter(clock([ev.t_ns]),[ev.mid],s=110,marker='*',c='black',zorder=5)
            ax.axvline(clock([origin])[0],color='black',ls='--',lw=.9)
            ax.set_ylabel(f'M{s} price ($)');ax.set_title(f'M{s}: first post-entry {bb.ROLES[int(ev.role)]} after {row.cross_after_entry_s:.2f}s; ID suffix {str(row.cross_id).split("/")[-2:]}',loc='left',fontsize=10)
        axes[0].legend(ncol=4,fontsize=8,loc='upper left');format_time(axes[-1])
        fig.suptitle(f'{date} · developing native BB · first root with all 4 resolved crossing marks\nDots: actual quote-clock crossings; black stars: first post-entry events; dashed: decision',fontsize=13,weight='bold')
        finish(fig,date+'_crossings.png')
        fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
        for side,color in [('L','#176b9b'),('S','#ad4c51')]:
            order=chosen.sort_values('cross_after_entry_s');axes[0].scatter(order.cross_after_entry_s,order['mark_'+side],s=50,color=color,label=side+' mark at crossing')
            axes[0].scatter(order.cross_after_entry_s,order['net_'+side],marker='x',s=65,color=color,label=side+' executed +100ms')
            for r in order.itertuples():axes[0].annotate(f'M{r.scale}',(r.cross_after_entry_s,getattr(r,'mark_'+side)),xytext=(5,6 if side=='L' else -13),textcoords='offset points',fontsize=10)
        axes[0].axhline(0,color='black',lw=.7);axes[0].set(xlabel='Δt from the SAME actual entry (seconds)',ylabel='Net per alternative checkpoint ($ / 1 oz), fee included',title='Four alternative checkpoints — never summed');axes[0].legend(fontsize=8,ncol=2)
        x=np.arange(4);axes[1].bar(x-.18,chosen.cross_mae_L,.35,label='LONG prefix MAE',color=COLORS[0]);axes[1].bar(x+.18,chosen.cross_mae_S,.35,label='SHORT prefix MAE',color='#ad4c51')
        axes[1].set(xticks=x,xticklabels=[f'M{s}' for s in chosen.scale],ylabel='Worst liquidation net before crossing ($ / 1 oz)',title='The route to a crossing has its own risk');axes[1].legend(fontsize=9)
        fig.suptitle(f'{date} · first eligible shared entry · {clock([int(chosen.entry_ns.iloc[0])])[0]}',fontsize=13,weight='bold')
        finish(fig,date+'_trade_profile.png')
        scene_meta.append(dict(date=date,origin_ns=str(origin),entry_ns=str(int(chosen.entry_ns.iloc[0])),
            points=[dict(scale=int(r.scale),event_id=r.cross_id,event_ns=str(r.cross_ns),delay_s=r.cross_after_entry_s,mark_L=r.mark_L,net_L=r.net_L,mark_S=r.mark_S,net_S=r.net_S) for r in chosen.itertuples()]))
    save(REPORT/'scene_receipt.json',dict(selection='first root per fixed calendar day with all 4 resolved marks; uses future only for audit selection, never fitting or policy',scenes=scene_meta))


def model_figures():
    summary=read(REPORT/'summary.json');b=summary['stages']['B']['classification'];heads=['m1','m3','m6','m9','global'];x=np.arange(5)
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    for j,family in enumerate(['geometry','rich','rich_event']):
        for ax,metric,title in [(axes[0,0],'accuracy','Query accuracy'),(axes[0,1],'event_accuracy','Event-macro accuracy'),(axes[1,0],'log_loss','Query cross-entropy'),(axes[1,1],'event_log_loss','Event-macro cross-entropy')]:
            ax.bar(x+(j-1)*.25,[b[h][family]['scores'][metric] for h in heads],.24,label=family,color=COLORS[j])
            ax.set(xticks=x,xticklabels=[h.upper() for h in heads],title=title)
    axes[0,0].plot(x,[b[h]['geometry']['nearest_line_accuracy'] for h in heads],'ko--',label='Nearest line, no learner')
    for ax in axes[1]:ax.plot(x,[b[h]['geometry']['prior']['event_log_loss' if ax==axes[1,1] else 'log_loss'] for h in heads],'ko--',label='FIT prior')
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.suptitle('Jan–Feb B (already exposed): high next-line accuracy must beat geometry and event controls',fontsize=14,weight='bold')
    finish(fig,'04_model_comparison.png')

    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    axes[0].bar(x-.25,[b[h]['rich']['roughness']['target_flip'] for h in heads],.24,label='True class changes',color=COLORS[0])
    axes[0].bar(x,[b[h]['rich']['roughness']['argmax_flip'] for h in heads],.24,label='Predicted class changes',color=COLORS[1])
    axes[0].bar(x+.25,[b[h]['rich']['roughness']['event_id_flip'] for h in heads],.24,label='Physical event / NONE group ID changes',color=COLORS[2])
    axes[0].set(xticks=x,xticklabels=[h.upper() for h in heads],ylabel='Fraction of adjacent 30s pairs',title='Temporal change is discrete, not a smooth wave');axes[0].legend(fontsize=8)
    for family,c in zip(['rich','rich_event'],COLORS):axes[1].plot(x,[b[h][family]['roughness']['mean_prediction_total_variation'] for h in heads],'o-',color=c,label=family)
    axes[1].set(xticks=x,xticklabels=[h.upper() for h in heads],title='Probability movement between adjacent queries',ylabel='Mean total variation distance');axes[1].legend()
    gradients=b['m9'];classes=['LOWER','MIDDLE','UPPER','TIE','NONE']
    for j,fam in enumerate(['rich','rich_event']):
        values=np.zeros(5)
        for r in gradients[fam]['gradient']['by_true_class']:values[r['label']]=r['l1_gradient']
        values/=values.sum();axes[2].bar(np.arange(5)+(j-.5)*.36,values,.35,label=fam,color=COLORS[j])
    axes[2].set(xticks=np.arange(5),xticklabels=classes,title='M9 FIT gradient mass: g = w × (p − y)',ylabel='Share of final |gradient|');axes[2].tick_params(axis='x',rotation=20);axes[2].legend()
    finish(fig,'05_temporal_and_gradient.png')

    for date in ['2026-01-02','2026-02-02']:
        scene=pd.read_parquet(DATA/(date+'_scene.parquet'));start=scene.t_ns.min();end=start+3600*bb.NS
        root=pd.read_parquet(DATA/(date+'_roots.parquet')).set_index('t_ns')
        view=scene.loc[scene.t_ns.between(start,end)]
        fig,axes=plt.subplots(5,1,figsize=(15,12),sharex=True,layout='constrained')
        axes[0].plot(clock(view.t_ns),view.mid,color='#142c3d',lw=1.2);axes[0].set_ylabel('GC midpoint ($)')
        for i,s in enumerate([1,9]):
            f=pd.read_parquet(DATA/f'predictions_B_m{s}_rich.parquet');f=f.loc[(f.date==date)&f.t_ns.between(start,end)]
            ready=np.isfinite(root.reindex(f.t_ns)[f'bb{s}_position'].to_numpy())
            ax=axes[1+2*i];ay=axes[2+2*i]
            for r,c in enumerate(LINE_COLORS):
                ax.plot(clock(f.t_ns),f[f'p{r}'].where(ready),color=c,lw=1,label=['LOWER','MIDDLE','UPPER','TIE','NONE'][r])
                positive=f[f'm{s}_y']==r
                ax.scatter(clock(f.loc[positive,'t_ns']),np.ones(positive.sum())*(1.04+r*.025),color=c,s=12,marker='|')
            ax.set(ylim=(0,1.18),ylabel=f'M{s} probability');ax.set_title(f'M{s}: probabilities; target class marks above 1.0',loc='left');ax.legend(ncol=5,fontsize=8)
            ax.fill_between(clock(f.t_ns),0,1.18,where=~ready,color='#cccccc',alpha=.5,step='mid')
            if not ready.any():ax.text(.5,.35,'No supported origin geometry in this hour; unscored',transform=ax.transAxes,ha='center',weight='bold')
            ay.plot(clock(f.t_ns),f[f'm{s}_delay_s'],color=COLORS[i],lw=1,marker='.',markersize=3)
            ay.set(ylabel='Seconds to event');ay.set_title('Shared event gives a countdown; a new ID resets it',loc='left')
            if not f[f'm{s}_delay_s'].notna().any():
                ay.set_ylim(0,1800);ay.set_facecolor('#eeeeee')
                ay.text(.5,.5,'No mature first-crossing labels in this hour',transform=ay.transAxes,ha='center')
        format_time(axes[-1]);fig.suptitle(f'{date} · first London hour · rich model · grey = no causal origin geometry',fontsize=14,weight='bold')
        finish(fig,date+'_predictions.png')

    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    for ax,stage in zip(axes,['A2','B']):
        for s,c in zip(bb.SCALES,COLORS):
            stats=summary['stages'][stage]['economics'][str(s)]['candidate_account'];f=pd.read_parquet(DATA/f'equity_{stage}_m{s}.parquet')
            if len(f):ax.plot(clock(f.t_ns),f.equity,color=c,label=f'M{s}: {stats["fills"]} fills, '+('CENSORED prefix' if stats['censor'] else 'complete'))
            if len(f) and stats['censor']:ax.scatter(clock([f.t_ns.iloc[-1]]),[f.equity.iloc[-1]],marker='x',color=c,s=65)
        ax.axhline(1500,color='black',ls='--',lw=.8);ax.set(title=stage+' candidate accounts: no extra cooldown',ylabel='Equity ($), 1 reference oz, start $1500');ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(md.DateFormatter('%m-%d'));ax.tick_params(axis='x',rotation=25)
    fig.suptitle('Four separate preselected-scale accounts; unknown exposure stops a curve\nOperational B admission is decided on A2, not on the attractive part of a B curve',fontsize=13,weight='bold')
    finish(fig,'06_candidate_equity.png')


def page():
    summary=read(REPORT/'summary.json');d=read(REPORT/'data_receipt.json');p=read(REPORT/'profile_summary.json');diag=read(REPORT/'bank_diagnostics.json')
    b=summary['stages']['B']['classification'];rows=[]
    for h in ['m1','m3','m6','m9','global']:
        a=b[h]['geometry'];r=b[h]['rich'];w=b[h]['rich_event'];mult=r['multiplicity']
        rows.append(f'<tr><td>{h.upper()}</td><td>{a["nearest_line_accuracy"]:.1%}</td><td>{a["scores"]["accuracy"]:.1%}</td><td>{r["scores"]["accuracy"]:.1%}</td><td>{w["scores"]["accuracy"]:.1%}</td><td>{r["scores"]["event_accuracy"]:.1%} / {w["scores"]["event_accuracy"]:.1%}</td><td>{mult["distinct_events"]:,}</td><td>{r["roughness"]["target_flip"]:.1%}</td></tr>')
    figure_captions={
        '01_learning_scheme.png':'Что учит модель: причинные X, будущие пересечения и отдельное исполнение.',
        '02_bank_and_examples.png':'Распределение целей, зависимость вопросов через общий ID и полнота исходов.',
        '03_time_and_net_profiles.png':'Профили всех перекрывающихся вопросов. Это описание рынка, а не equity счёта. Цена задержки включает хвосты, не показанные в гистограмме.',
        '04_model_comparison.png':'Сравнение на январе–феврале, которые уже использовались для исследований. Правильная линия не тождественна прибыльному действию.',
        '05_temporal_and_gradient.png':'Смена классов остаётся дискретной. Равный вес события меняет вклад примеров в реальный градиент.',
        '06_candidate_equity.png':'Кандидатные счета с одним открытым положением. Крестик завершает только известный префикс; цензурированный остаток не дорисован.',
        '07_microevents_and_coverage.png':'Повторные пересечения одной линии могут различаться лишь миллисекундами. Готовность длинного масштаба после разрыва требует новой непрерывной истории.',
        '08_micro_crossing_ids.png':'Первые шесть ID в одном небольшом колебании около средней линии. Здесь показаны все публикации BBO, без пятисекундного прореживания; дальнейшие пересечения не подписаны.',
        '09_full_session_profiles.png':'Фактические первые crossings до конца поддержанной части сессии, без потолка 30 минут. Это отдельная будущая разметка; frozen модели и счета выше используют 30-минутный контракт.'}
    for date in bb.contract()['visual_dates']:
        figure_captions[date+'_crossings.png']=f'{date}: первый старт, для которого видны все четыре пересечения. Выбор сцены использует будущее только для иллюстрации, не для обучения или входа.'
        figure_captions[date+'_trade_profile.png']='Один и тот же исполненный вход, четыре альтернативных checkpoint. LONG и SHORT — отдельные ветви. Точки не суммируются и не изображают путь прибыли между пересечениями.'
    for date in ['2026-01-02','2026-02-02']:figure_captions[date+'_predictions.png']='Первый час Лондона: реальные причинные прогнозы и будущая разметка; countdown делает общий event ID видимым.'
    figures=''.join(f'<figure><a href="{name}"><img loading="lazy" src="{name}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>' for name,caption in figure_captions.items())
    censor_map={(r['stage'],r['scale']):r for r in read(REPORT/'execution_censor_diagnostics.json')['censors']}
    econ_rows=[]
    for s in bb.SCALES:
        account=summary['stages']['B']['economics'][str(s)]['candidate_account'];reason=censor_map.get(('B',s))
        detail='Полный период' if reason is None else reason['date']+' / '+('тайм-аут выхода '+f"{reason['first_exit_quote_delay_s']:.3f}s" if reason['reason']=='EXIT_TIMEOUT_WITH_SUPPORTED_QUOTE_PATH' else 'потеря поддержки геометрии')
        econ_rows.append(f'<tr><td>M{s}</td><td>{summary["admissions"][str(s)]}</td><td>{account["fills"]}</td><td>{account["observed_prefix_net"]:.2f}</td><td>{detail}</td></tr>')
    econ=''.join(econ_rows)
    binrows=''.join(f'<tr><td>{p["1"]["future_delay_bins"][i]["lo"]}–{p["1"]["future_delay_bins"][i]["hi"]} s</td>'+''.join(f'<td>{p[str(s)]["future_delay_bins"][i]["queries"]:,}</td>' for s in bb.SCALES)+'</tr>' for i in range(7))
    smoothrows=''.join(f'<tr><td>M{s}</td><td>{diag["same_pair_old_target"][str(s)]["pairs"]:,}</td><td>{diag["same_pair_old_target"][str(s)]["old_three_class_flip"]:.2%}</td><td>{diag["same_pair_old_target"][str(s)]["bb_class_flip"]:.2%}</td><td>{diag["same_pair_old_target"][str(s)]["bb_event_id_flip"]:.2%}</td><td>{diag["interarrival"][str(s)]["fraction_under_1s"]:.1%}</td></tr>' for s in bb.SCALES)
    text=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Bollinger: банк первых пересечений</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;background:#f3f5f7;color:#182b3a;margin:auto;max-width:1500px;padding:28px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:42px}}p{{max-width:1050px}}.card,figure{{background:white;border:1px solid #d5dfe6;border-radius:10px;padding:20px;margin:22px 0}}.badge{{color:#8b3a24;font-weight:bold}}img{{width:100%;height:auto}}figcaption{{font-size:15px;color:#415362}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.scroll{{overflow-x:auto}}a{{color:#176b9b}}code{{background:#e9eef2;padding:2px 4px}}</style>
<h1>Банк будущих первых пересечений Bollinger</h1><p class="badge">Исследовательское сравнение · GC midpoint · B уже исследован ранее · торговая готовность не заявляется</p>
<p>{d['days']} дней · {d['roots']:,} исходных запросов · {d['events']:,} событий · {d['quotes']:,} публикаций BBO.</p>
<div class="card"><strong>Результат.</strong> Класс будущей линии стал заметно устойчивее во времени. Равный вес события улучшил log-score по событиям. Дополнительные потоковые и дневные признаки устойчивого выигрыша над геометрией не дали. Денежный контроль не прошёл допуск: полные A2 счета M1 −$308.70, M3 −$193.20; M6/M9 цензурированы. Для всех четырёх масштабов рабочее действие остаётся WAIT.</div>
<div class="card"><strong>Что именно предсказывается.</strong> Для каждой точки и M1/M3/M6/M9: первая будущая линия lower/middle/upper, её ID и время. Отдельная общая голова выбирает первый масштаб и линию. TIE сохраняется как множество, отсутствие события до 30 минут отделено от неизвестного будущего. Это разворот строгой стороны относительно движущейся линии; равенство не создаёт новое событие. Полная фактическая проекция до session liquidation deadline также сохранена в <code>data/derived/bb_crossing_bank_v1/first_session</code>, включая crossings позже 30 минут. Её marks не подставляются в frozen 30m модель/счёт.</div>
<p>SMA20 ± 2σ, исходный ddof=1; developing содержит 19 завершённых native closes и текущий BBO midpoint. Шкала M9 означает 20 девятиминутных свечей, а не окно девять минут. Бары выровнены по UTC и доступному clock котировок. <a href="https://help.ctrader.com/ctrader-algo/references/Indicators/BollingerBands/">cTrader API</a> подтверждает SMA20/2; точный знаменатель встроенной дисперсии и совпадение с брокерским графиком не подтверждены. Контроль ddof=0 сохранён в data_receipt.json.</p>
<h2>Стало ли меньше шума?</h2><p>На общей поддержке старого T/L target (24/12 ticks, 300s) и нового BB target (первое событие до 1800s) класс следующей линии меняется заметно реже. Это разные вопросы; редкая смена метки не доказывает больше предсказательной информации в X. Банк сырых пересечений содержит много дребезга: десятки быстрых переходов через одну линию могут оставлять название следующей линии прежним.</p>
<div class="scroll card"><table><tr><th>Масштаб</th><th>Общих пар / 30s</th><th>Смена старого TAKE/STOP/TIME</th><th>Смена BB класса</th><th>Смена event / NONE-block ID</th><th>Повтор той же линии &lt;1s</th></tr>{smoothrows}</table></div>
<h2>Проверка предсказуемости B</h2><div class="scroll card"><table><tr><th>Голова</th><th>Ближайшая линия</th><th>Geometry</th><th>Rich</th><th>Rich + вес события</th><th>Event accuracy: rich / weighted</th><th>Уникальных событий B</th><th>Смена target /30s</th></tr>{''.join(rows)}</table></div>
<p>Event score сначала усредняет ошибки вопросов одного события, затем даёт событиям равный вес. NONE без события объединены по фиксированным 1800-секундным блокам origins; это отдельные группы, не вымышленные crossings. Здесь нет правила «хотя бы один удачный прогноз означает успех всего события». В X нет будущих ID, времени пересечения, equity или target. Rich использует 65 прежних скаляров потока/дня/STFT и 16 полей BB; полная последовательность текущего и вчерашнего дня в модель не подаётся.</p>
<h2>Исполнение и equity</h2><p>В этом сравнении вход T — первый поддержанный BBO строго после 100 ms, timeout 2 s. BUY по ask, SELL по bid, комиссия $0.30 за оборот одной условной унции. Пассивный P здесь повторно не исследован. Четыре checkpoint принадлежат одному гипотетическому входу; их нельзя суммировать. Счёт выбирает один масштаб заранее и выходит после первого пересечения с ещё 100 ms задержки, либо по 30 минутам/концу сессии. Фиксированного защитного stop в этой ветви нет.</p>
<div class="scroll card"><table><tr><th>Счёт</th><th>A2 admission</th><th>B завершённых сделок</th><th>B известный префикс, $</th><th>Причина остановки / цензура</th></tr>{econ}</table></div>
<p>При admission=false рабочее действие остаётся WAIT. Кандидатные префиксы приведены для диагностики. Отсутствие дополнительного cooldown не снимает ограничения одного открытого положения. Общий event ID исправляет повторный учёт будущего события, но не гарантирует независимость соседних событий или устойчивость модели.</p>
<p>Пять из шести остановок A2/B связаны с отсутствием новой публикации для выхода в пределах 2 секунд после запроса, хотя поддержка котировок сохранялась. Это ограничение объявленного рецепта исполнения; оно не доказывает потерю стакана. Один B счёт остановлен при потере поддержки геометрии. Первые доступные цены позже тайм-аута не подставлены задним числом. Причины сохранены в <a href="execution_censor_diagnostics.json">execution_censor_diagnostics.json</a>.</p>
<p>Доли положительных marks по фактическому delay — условное описание будущего, не доступное при входе правило ожидания. Количество перекрывающихся запросов в показанных delay bins:</p><div class="scroll card"><table><tr><th>Будущий delay</th><th>M1</th><th>M3</th><th>M6</th><th>M9</th></tr>{binrows}</table></div><p>На гистограмме задержки исполнения за пределами ±$2 остаются {p['1']['latency_plot_tail_fraction']:.2%}–{p['9']['latency_plot_tail_fraction']:.2%} наблюдений для M1–M9. Все хвостовые квантили сохранены в <a href="profile_summary.json">profile_summary.json</a>.</p>
<h2>Визуальные проверки</h2>{figures}
<h2>Воспроизводимые материалы</h2><p><a href="summary.json">Результаты моделей и градиентов</a> · <a href="data_receipt.json">Покрытие и ddof</a> · <a href="full_session_projection.json">Полная проекция первых crossings</a> · <a href="raw_quote_checks.json">Независимая проверка котировок</a> · <a href="account_replay.json">Replay счетов</a> · <a href="numerical_validation.json">Проверка моделей и хешей</a> · <a href="feature_context_audit.json">Доступ к признакам и контексту</a> · <a href="scene_receipt.json">IDs сцен и точные часы</a> · <a href="SOL61_VISUAL_AUDIT_RU.md">Визуальный аудит Sol 6.1</a></p>
</html>'''
    (REPORT/'index.html').write_text(text,encoding='utf-8',newline='\n')
    save(REPORT/'visual_receipt.json',dict(script_sha256=sha(__file__),summary_sha256=sha(REPORT/'summary.json'),
        images={name:sha(REPORT/name) for name in figure_captions},page_sha256=sha(REPORT/'index.html')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=['bank','models','page','all']);a=parser.parse_args()
    if a.part in ('bank','all'):bank_figures()
    if a.part in ('models','all'):model_figures()
    if a.part in ('page','all'):page()
