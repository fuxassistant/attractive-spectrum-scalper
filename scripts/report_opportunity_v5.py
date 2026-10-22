"""Unsmoothed target, gradient, alert and execution evidence for V5."""
from pathlib import Path
import gc
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from scripts.run_opportunity_v5 import read, save, PNAMES, QNAMES
from scripts.evaluate_opportunity_v5 import ARMS, score_channels
from spectrum.research_v3.learning import sha

ROOT=Path('reports/scientific_pipeline_v5')
COLORS=dict(base='#2563eb',semantic='#db8a12',risk='#14916c',hash_control='#a274ba')
HEADS=['event','chosen_correct_given_event','safe_cost2_given_correct','safe_cost4_given_correct','safe_cost6_given_correct']
SHORT=['Event','Correct side | E','Safe cost2 | correct','Safe cost4 | correct','Safe cost6 | correct']
FIGURES=[]


def finish(fig,name,title,caption):
    fig.savefig(ROOT/name,dpi=145,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    FIGURES.append(dict(file=name,title=title,caption=caption,sha256=sha(ROOT/name)))


def format_num(x,digits=3):
    return '—' if x is None else f'{x:.{digits}f}'


def load():
    c=read('configs/research/opportunity_pipeline_v5.json')
    e=read(ROOT/'evaluation.json'); suite=read(ROOT/'suite_summary.json')
    if e['status']!='COMPLETE' or suite['status']!='COMPLETE' or e['config_sha256']!=sha('configs/research/opportunity_pipeline_v5.json'):
        raise ValueError('Complete unchanged V5 evidence required')
    bundles={}
    for r in suite['runs']:
        path=Path(suite['runs_root'])/r['run_id']; receipt=read(path/'receipt.json')
        for name,digest in receipt['outputs'].items():
            if sha(path/name)!=digest:
                raise ValueError('Changed run output '+name)
        bundles[r['seed']]=dict(path=path,p=pd.read_parquet(path/'predictions.parquet'),
            fp=pd.read_parquet(path/'filter_predictions.parquet'),targets=pd.read_parquet(path/'targets.parquet'),
            history=read(path/'history.json'),summary=read(path/'summary.json'))
    return c,e,suite,bundles


def learning_figures(c,e,b):
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    for col,seed in enumerate((17,29)):
        h=b[seed]['history']; ax=axes[0,col]
        ax.plot([r['update'] for r in h],[r['train_probe_nll'] for r in h],'-o',label='TRAIN fixed probe')
        ax.plot([r['update'] for r in h],[r['validation_nll'] for r in h],'-o',label='VAL selection')
        ax.axvline(b[seed]['summary']['training']['best_update'],ls='--',color='black',label='Selected')
        ax.set(title=f'Seed {seed}: conditional BCE / E-known roots',xlabel='Updates',ylabel='Composite NLL'); ax.legend(); ax.grid(alpha=.2)
        ax=axes[1,col]
        r=next(r for r in e['runs'] if r['seed']==seed)
        x=np.arange(5)
        for i,role in enumerate(('train','validation','test')):
            vals=[r['diagnostics'][role]['filter_heads'][q]['nll'] for q in QNAMES]
            ax.bar(x+(i-1)*.24,vals,.24,label=role)
        ax.set(xticks=x,xticklabels=SHORT,ylabel='Conditional head NLL',title='Heads have different conditional populations')
        ax.tick_params(axis='x',rotation=20); ax.legend(); ax.grid(axis='y',alpha=.2)
    finish(fig,'01_learning.png','Обучение и выбор checkpoint','Основные TCN заморожены. Учится только MLP-фильтр; checkpoint выбран по VAL. Head NLL условны и имеют разные знаменатели.')

    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    for col,seed in enumerate((17,29)):
        p,fp=b[seed]['p'],b[seed]['fp']; use=p.feature_available & p.date.isin(c['splits']['train'])
        y=fp[['y_'+h for h in HEADS]].to_numpy()[use]
        grad=fp[['g_'+h for h in HEADS]].to_numpy()[use]
        x=np.arange(5); known=(y>=0).sum(0); positives=(y==1).sum(0)
        ax=axes[0,col]
        ax.bar(x,known,label='Known conditional rows',color='#c8d5e7'); ax.bar(x,positives,label='Positive',color='#14916c')
        ax.set(xticks=x,xticklabels=SHORT,ylabel='TRAIN roots',title=f'Seed {seed}: actual label populations'); ax.tick_params(axis='x',rotation=20); ax.legend()
        ax=axes[1,col]
        pos=(np.abs(grad)*(y==1)).sum(0); neg=(np.abs(grad)*(y==0)).sum(0)
        ax.bar(x,neg,label='Negative-label |dL/dz|',color='#c15f63'); ax.bar(x,pos,bottom=neg,label='Positive-label |dL/dz|',color='#14916c')
        ax.set(xticks=x,xticklabels=SHORT,ylabel='Sum of unreduced |dL/dz|',title='Actual BCE mass; each risk coefficient = 1/3'); ax.tick_params(axis='x',rotation=20); ax.legend()
    finish(fig,'02_targets_gradients.png','Распределение меток и градиентов','UNKNOWN и неактивные условные головы дают нулевой градиент. Это производная loss по logit; не производная временного ряда таргета.')


def ranking_figures(e):
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    for row,r in enumerate(e['runs']):
        for arm in ARMS:
            rows=[v for v in r['grid'] if v['arm']==arm and v['plan']=='cost2']
            rows.sort(key=lambda v:v['test_signal']['total_causal_alerts'])
            count=[v['test_signal']['total_causal_alerts'] for v in rows]
            for col,key,title in ((0,'precision','Correct-side precision'),(1,'safe_precision','Safe crossing precision'),(2,'safe_recall','Safe crossing recall')):
                axes[row,col].plot(count,[v['test_signal'][key] for v in rows],'-o',label=arm,color=COLORS[arm])
                axes[row,col].set(title=f'Seed {r["seed"]}: {title}',xlabel='TEST causal alerts, CAL-frozen thresholds',ylabel=title); axes[row,col].grid(alpha=.2)
        axes[row,0].legend()
    finish(fig,'03_retention_tradeoff.png','Отбор при разных бюджетах сигналов','Все пороги заморожены на CAL. Их перенос на TEST меняет число сигналов; кривые показывают это явно. Hash control разрежает сигналы без знания исходов.')

    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    for col,r in enumerate(e['runs']):
        x=np.arange(4); labels=list(ARMS)
        for row,kind in enumerate(('primary','exact_test_count_diagnostic')):
            m=[r['primary'][a]['test_signal'] for a in ARMS] if kind=='primary' else [v['metrics'] for v in r[kind]]
            ax=axes[row,col]
            bottom=np.zeros(4)
            for key,label,color in [('none_false_positives','NONE','#a7a9b0'),('wrong_side_alerts','Wrong side','#be5560'),('unsafe_correct_alerts','Correct, risk first','#e9a32c'),('safe_correct_alerts','Correct, safe','#14916c'),('unknown_alerts','UNKNOWN','#b496c7'),('unknown_risk_correct_alerts','Correct, risk unknown','#ddd2ba')]:
                vals=np.array([v[key] for v in m]); ax.bar(x,vals,bottom=bottom,label=label,color=color); bottom+=vals
            ax.set(xticks=x,xticklabels=labels,ylabel='Root alerts',title=f'Seed {r["seed"]}: '+('CAL 50% budget transferred' if row==0 else 'Exact TEST count; retrospective only'))
            if col==0: ax.legend(fontsize=8,ncol=2)
    finish(fig,'04_false_entries.png','NONE, неверная сторона и риск до crossing','Верх: применимые CAL-пороги. Низ: одинаковое число TEST-точек по ретроспективному ranking, только диагностика селективности; эти пороги не участвуют в торговом replay.')

    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for row,r in enumerate(e['runs']):
        for col,name in enumerate(QNAMES[:3]):
            ax=axes[row,col]
            for role,color in [('validation','#6b7280'),('calibration','#d97706'),('test','#2563eb')]:
                bins=r['diagnostics'][role]['filter_heads'][name]['reliability']
                populated=[v for v in bins if v['n']>=20]
                ax.plot([v['predicted'] for v in populated],[v['observed'] for v in populated],'-o',label=role,color=color)
            ax.plot([0,1],[0,1],ls='--',color='black',alpha=.4)
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted conditional probability',ylabel='Observed fraction',title=f'Seed {r["seed"]}: {SHORT[col]}'); ax.grid(alpha=.2)
            if col==0: ax.legend()
    finish(fig,'05_reliability.png','Калибровка условных вероятностей','Показаны только bins с ≥20 известными примерами. q_safe условна на правильную сторону; её нельзя читать как вероятность прибыльной сделки.')


def economics_figures(e,c):
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    for col,r in enumerate(e['runs']):
        ax=axes[0,col]
        for arm in ARMS:
            v=r['primary'][arm]['test_economics']; daily=v['daily']
            dates=[d for d in c['splits']['test'] if daily[d]['cumulative_closed_cash'] is not None]
            ax.plot(range(len(dates)),[daily[d]['cumulative_closed_cash'] for d in dates],'-o',color=COLORS[arm],
                label=f'{arm}: {v["status"]}, n={v["trades"]}')
        ax.axhline(0,color='black',lw=.8); ax.set(xticks=range(5),xticklabels=[d[5:] for d in c['splits']['test']],ylabel='Closed cash, USD/reference oz',title=f'Seed {r["seed"]}: primary cost2 / CAL retain 50%'); ax.legend(fontsize=8); ax.grid(alpha=.2)
        ax=axes[1,col]; x=np.arange(4); width=.22
        for i,plan in enumerate(('cost2','cost4','cost6')):
            vals=[]
            for arm in ARMS:
                rec=next(v for v in r['grid'] if v['arm']==arm and v['plan']==plan and v['fraction']==.5)['test_economics']
                vals.append(rec['total_net_if_complete'] if rec['completed'] else np.nan)
            ax.bar(x+(i-1)*width,vals,width,label=plan)
        ax.axhline(0,color='black',lw=.8); ax.set(xticks=x,xticklabels=ARMS,ylabel='COMPLETE total net only',title='Stop-budget sensitivity; fixed original event gain'); ax.legend(); ax.grid(axis='y',alpha=.2)
    finish(fig,'06_economics.png','Последовательная экономика','Однопозиционный счёт, taker BBO, комиссия 0.3 и задержка 100 ms. CENSORED останавливает счёт; частичный cash не выдаётся за итог. Разные фильтры меняют занятость позиции.')


def scenes(c,e,b):
    cmap=ListedColormap(['#9ca3af','#c9646c','#20a37e']); norm=BoundaryNorm([-1.5,-.5,.5,1.5],3)
    for seed in (17,29):
        bundle=b[seed]; p=bundle['p']; fp=bundle['fp']
        channels,_=score_channels(p,seed)
        for number,case in enumerate(c['experiment']['fixed_visual_cases']):
            date=case['date']; rows=np.flatnonzero(p.date.eq(date)); t0=int(p.t_ns.iloc[rows[0]])
            hours=(p.t_ns.to_numpy()-t0)/3.6e12
            ids=rows[(hours[rows]>=case['offset_hours'])&(hours[rows]<case['offset_hours']+case['duration_hours'])]
            scene=pd.read_parquet(Path(c['data']['root'])/(date+'_scene.parquet')).set_index('t_ns').loc[p.t_ns.iloc[ids]]
            xx=hours[ids]; fig,axes=plt.subplots(4,1,figsize=(17,12),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[3,2,1.2,2]})
            ax=axes[0]; ax.plot(xx,scene.mid,color='#111827',lw=1,label='Mid, 5s origins')
            for scale,color in [(1,'#3b82f6'),(3,'#d97706'),(6,'#8b5cf6')]:
                middle=scene[f'BB{scale}_middle'].to_numpy(); half=scene[f'BB{scale}_half'].to_numpy()
                ax.plot(xx,middle,color=color,lw=.9,label=f'BB M{scale} middle')
                if scale in (1,6): ax.fill_between(xx,middle-half,middle+half,color=color,alpha=.07)
            ax.set(title=f'Fixed window {date} / seed {seed}: unsmoothed 5s predictions and teacher',ylabel='Price'); ax.legend(ncol=4,fontsize=8)
            ax=axes[1]
            for arm in ('base','semantic','risk'):
                ax.plot(xx,channels['cost2'][arm][ids],color=COLORS[arm],lw=.8,label=arm+' score')
            ax.axhline(.35,color='#2563eb',ls=':',label='base candidate floor .35')
            ax.set(ylim=(0,1),ylabel='Scores'); ax.legend(ncol=4,fontsize=8)
            ys=p[['target_q_event','target_q_correct','target_q_safe_cost2']].to_numpy()[ids].T
            axes[2].imshow(ys,aspect='auto',interpolation='none',extent=[xx[0],xx[-1],2.5,-.5],cmap=cmap,norm=norm)
            axes[2].set(yticks=[0,1,2],yticklabels=['E','C | E','safe2 | C'],title='Teacher: red=0, green=1, grey=unknown/inactive')
            ax=axes[3]
            for head,color in zip(HEADS[:3],['#d97706','#14916c','#8b5cf6']):
                ax.plot(xx,fp['g_'+head].to_numpy()[ids],lw=.7,color=color,label='filter '+head.split('_')[0])
            ax.plot(xx,p.base_diagnostic_grad_0.to_numpy()[ids],lw=.7,color='#4b5563',alpha=.6,label='frozen base E diagnostic only')
            ax.axhline(0,color='black',lw=.6); ax.set(ylabel='Unreduced dL/dz',xlabel='Hours from first session decision'); ax.legend(ncol=4,fontsize=8)
            for ax in axes: ax.grid(alpha=.13)
            finish(fig,f'07_scene_{seed}_{number+1}.png',f'Сигнал, таргет и градиент: {date}, seed {seed}',
                'Окно выбрано до fit. Точки не сглажены. Серый градиент основной модели — диагностический: backbone не обновлялся. Цветные производные соответствуют сохранённому BCE фильтра.')


def trade_cases(c,e,b):
    from spectrum.cognition.execution import QuoteTape
    from spectrum.playground.data import MonthDay,QUOTE_CACHE
    r=next(r for r in e['runs'] if r['seed']==17)
    ledger=read(r['primary']['base']['test_ledger']); pred=b[17]['p']; target=b[17]['targets']
    index={int(t):i for i,t in enumerate(pred.t_ns)}; chosen={}
    for trade in ledger:
        if trade['status']!='COMPLETE': continue
        i=index[int(trade['t_ns'])]; y=int(pred.y_crossing.iloc[i]); side=int(pred.side.iloc[i]); safe=float(pred.target_q_safe_cost2.iloc[i])
        bucket='NONE' if y==0 else 'WRONG_SIDE' if y>0 and not(y&(1 if side==1 else 2)) else 'CORRECT_RISK_FIRST' if safe==0 else 'CORRECT_SAFE' if safe==1 else 'UNKNOWN'
        if bucket!='UNKNOWN' and bucket not in chosen: chosen[bucket]=(trade,i)
    receipts=[]
    for num,(bucket,(trade,i)) in enumerate(chosen.items()):
        date=trade['date']; day=MonthDay(Path(c['data']['market_root'])/date[:7]/date,verify=True); q=QuoteTape.from_day(day)
        row=target.iloc[i]; entry=int(row.entry_ordinal); origin=int(row.t_ns); side=int(row.chosen_side)
        end=min(int(row.deadline_ns)+2*10**9,int(q.quote_t[-1])); left=max(0,np.searchsorted(q.quote_t,origin-10*10**9)); right=np.searchsorted(q.quote_t,end,side='right')
        ids=np.arange(left,right); seconds=(q.quote_t[ids]-origin)/1e9
        net=((q.bid[ids]-q.ask[entry])/1e9-.3 if side==1 else (q.bid[entry]-q.ask[ids])/1e9-.3)
        # Every publication is plotted: do not hide a brief stop hit by sampling.
        fig,axes=plt.subplots(2,1,figsize=(16,8),layout='constrained',sharex=True,gridspec_kw={'height_ratios':[3,1]})
        ax=axes[0]; ax.plot(seconds,net,color='#1f2937',lw=.7,label='Full-publication hypothetical liquidation net')
        ax.axhline(-row.R_cost2,color='#c23d51',ls='--',label='Decision stop -R2'); ax.axhline(row.G_event,color='#14916c',ls='--',label='Fixed qualifying G')
        ax.axvline((int(row.entry_ns)-origin)/1e9,color='#2563eb',label='Actual entry')
        witness=int(row.witness_ns)
        if witness>0: ax.axvline((witness-origin)/1e9,color='#14916c',label='First qualifying chosen-side crossing')
        ax.axvline((int(trade['exit_ns'])-origin)/1e9,color='#d97706',ls=':',label=f'Actual {trade["reason"]} exit')
        ax.scatter([(int(trade['exit_ns'])-origin)/1e9],[trade['net']],color='#d97706',s=50,zorder=5)
        ax.set(title=f'Earliest seed17 primary base example: {bucket} | {date} | side {side:+d} | actual net {trade["net"]:.2f}',ylabel='Net USD/reference oz'); ax.legend(fontsize=8,ncol=3); ax.grid(alpha=.2)
        mask=pred.date.eq(date).to_numpy() & (pred.t_ns.to_numpy()>=origin-10*10**9)&(pred.t_ns.to_numpy()<=end)
        channels,_=score_channels(pred,17)
        for arm in ('base','semantic','risk'):
            axes[1].plot((pred.t_ns.to_numpy()[mask]-origin)/1e9,channels['cost2'][arm][mask],'-o',ms=2,label=arm,color=COLORS[arm])
        axes[1].axvline(0,color='black',lw=1); axes[1].set(ylim=(0,1),xlabel='Seconds from decision',ylabel='Causal scores'); axes[1].legend(ncol=3); axes[1].grid(alpha=.2)
        filename=f'08_trade_{num+1}_{bucket.lower()}.png'
        finish(fig,filename,'Путь сделки: '+bucket,'Первый по времени подходящий пример, без выбора по PnL. Линия после фактического выхода — контрфактический mark исходного входа; она показывает crossing, но не продолжение счёта.')
        receipts.append(dict(bucket=bucket,date=date,t_ns=str(origin),root_index=i,source_quote_parts=day.meta['quote_parts'],
            actual_exit_ns=trade['exit_ns'],witness_ns=str(witness),full_quote_rows=len(ids),file=filename))
        del q,day; QUOTE_CACHE.clear(); gc.collect()
    save(ROOT/'visual_case_receipt.json',dict(rule=c['experiment']['execution_case_rule'],cases=receipts,missing_buckets=sorted(set(['NONE','WRONG_SIDE','CORRECT_RISK_FIRST','CORRECT_SAFE'])-set(chosen))))


def candidate_funnel(c,e,b):
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    for col,r in enumerate(e['runs']):
        seed=r['seed']; p=b[seed]['p']; channels,_=score_channels(p,seed)
        score=channels['cost2']['base']; test=p.feature_available.to_numpy() & p.date.isin(c['splits']['test']).to_numpy()
        known=test & p.y_crossing.ge(0).to_numpy(); correct=known & p.target_q_correct.eq(1).to_numpy()
        safe=correct & p.target_q_safe_cost2.eq(1).to_numpy()
        ax=axes[0,col]
        for mask,label,color in [(known,'All known','#9ca3af'),(correct,'Correct side','#2563eb'),(safe,'Correct + safe2','#14916c')]:
            ax.hist(score[mask],bins=np.linspace(0,1,51),histtype='step',lw=1.5,label=label,color=color)
        ax.axvline(.35,color='#c23d51',ls='--',label='Frozen candidate floor .35')
        ax.set(title=f'Seed {seed}: base proposal bottleneck',xlabel='Frozen base max-side score',ylabel='TEST known roots'); ax.legend(fontsize=8); ax.grid(alpha=.2)
        candidate=score>=.35
        counts=np.array([[known.sum(),correct.sum(),safe.sum()],[(known&candidate).sum(),(correct&candidate).sum(),(safe&candidate).sum()]],float)
        ax=axes[1,col]; x=np.arange(3)
        ax.bar(x,100*counts[1]/np.maximum(counts[0],1),color=['#9ca3af','#2563eb','#14916c'])
        for k in range(3): ax.text(k,100*counts[1,k]/max(counts[0,k],1)+.2,f'{int(counts[1,k])}/{int(counts[0,k])}',ha='center',fontsize=10)
        ax.set(xticks=x,xticklabels=['All known','Correct side','Correct + safe2'],ylabel='% surviving base candidate floor',title='A downstream filter cannot recover rejected proposals'); ax.grid(axis='y',alpha=.2)
    finish(fig,'09_candidate_funnel.png','Что доходит до отдельного фильтра','Порог base ≥0.35 зафиксирован до опыта. График диагностический: показывает потерю доступных возможностей до фильтра; порог после просмотра TEST не менялся.')


def html_report(c,e):
    rows=[]
    for r in e['runs']:
        for arm in ARMS:
            v=r['primary'][arm]; s=v['test_signal']; z=v['test_economics']
            rows.append(f'<tr><td>{r["seed"]}</td><td>{arm}</td><td>{s["alerts"]}</td><td>{s["none_false_positives"]}</td><td>{s["wrong_side_alerts"]}</td><td>{s["unsafe_correct_alerts"]}</td><td>{format_num(s["precision"]*100,1)}%</td><td>{format_num(s["safe_precision"]*100,1)}%</td><td>{z["trades"]}</td><td>{format_num(z["total_net_if_complete"],2)}</td><td>{z["status"]}</td></tr>')
    policy=[]
    for r in e['runs']:
        for arm,v in r['selected'].items():
            policy.append(f'<li>Seed {r["seed"]}, {arm}: {html.escape(v["reason"])}; {v["plan"]}, threshold {v["threshold"]:.6f}; TEST {v["test"]["trades"]} trades, net {format_num(v["test"]["total_net_if_complete"],2)}.</li>')
    audit=read(ROOT/'learning_audit.json')
    audit_rows=[]
    for r in audit['runs']:
        candidate=r['candidate_diagnostics']['by_role']['test']['candidate_head_metrics']['heads']
        global_heads=r['by_role']['test']['model']['heads']
        for name in HEADS[:3]:
            audit_rows.append(f'<tr><td>{r["seed"]}</td><td>{name}</td><td>{global_heads[name]["n"]}</td><td>{global_heads[name]["auc"]:.3f}</td><td>{candidate[name]["n"]}</td><td>{candidate[name]["auc"]:.3f}</td></tr>')
    conclusion=f'''<section><h2>Вывод после независимых проверок</h2><p><strong>Фильтр не показал экономического преимущества.</strong> На заранее выбранном сравнении cost2/CAL50 оба фильтра ухудшили полный TEST net относительно базы в обоих seeds. Seed17 CAL выбрал одинаковый для всех вариант cost6/retain100% — фильтрация отключена; TEST −25.30. Seed29 выбрал WAIT.</p>
<p>При одинаковом числе TEST-сигналов семантический отбор улучшает seed17 и ухудшает seed29. Добавление risk-головы не даёт устойчивого повышения safe precision. Пять TEST-дней и смена контракта ограничивают переносимость выводов.</p>
<p>Глобальная метрика не описывает множество входов. Внутри base-кандидатов направление уже различается лучше случайного, но head риска cost2 заметно слабее, чем на всех правильных сторонах. Фильтр обучался на всех известных TRAIN-точках; сам генератор предложил только 400/2410 TRAIN-кандидатов и 514/4178 TEST-кандидатов.</p>
<div class="scroll"><table><tr><th>Seed</th><th>Head</th><th>Global conditional N</th><th>AUC</th><th>Candidate conditional N</th><th>AUC</th></tr>{''.join(audit_rows)}</table></div>
<p>Следующая проверяемая гипотеза: согласовать обучение и калибровку фильтра с распределением предложенных входов и отдельно проверить более широкий отбор кандидатов. Текущие результаты не обосновывают подбор focal loss. Этот опыт завершён без изменения TEST-порогов или повторного fit по увиденным результатам.</p></section>'''
    figures=''.join(f'<section><h2>{html.escape(v["title"])}</h2><p>{html.escape(v["caption"])}</p><a href="{v["file"]}"><img loading="lazy" src="{v["file"]}" alt="{html.escape(v["title"])}"></a></section>' for v in FIGURES)
    text=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Scientific pipeline V5 · crossing risk filter</title><style>
body{{margin:0;background:#f1f4f8;color:#172334;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1450px;margin:auto;padding:28px}}header,section{{background:white;padding:26px;margin:18px 0;border:1px solid #dce2ea;border-radius:12px}}h1{{font-size:32px;line-height:1.2}}h2{{font-size:23px}}p{{max-width:1100px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #dce2ea;text-align:right}}th{{background:#eef2f7}}td:nth-child(2){{text-align:left}}.scroll{{overflow:auto}}code{{background:#eef2f7;padding:2px 6px}}a{{color:#175ac7}}.tag{{color:#315b85;font-weight:600}}
</style><main><header><p class="tag">SCIENTIFIC PIPELINE V5 · FROZEN RESTORED + UNIFORM · 2 SEEDS</p><h1>Фильтр NONE, неверной стороны и риска до crossing</h1>
<p>20 сессий GCZ6, 201 480 точек: TRAIN 7, VAL 3, CAL 5, TEST 5. Исходный crossing E/D target и BCE основной модели сохранены. Замороженные TCN обучались на GCQ6; новые блоки проверяют и перенос на следующий контракт.</p>
<p>Отдельная MLP 32×2 получает 182 канала: текущие Micro values/masks, внутреннее состояние замороженной TCN, её вероятности и сторону, причинные отношения риска к G. Максимум подробной истории — 300 секунд. Полный банк Rich из 225 фич сохранён, но дневные/VWAP/STFT каналы в этот fit не входят.</p>
<p><code>L = BCE(E) + I(E)·BCE(C) + I(E∧C)·[BCE(S₂)+BCE(S₄)+BCE(S₆)]/3</code>, затем сумма по батчу делится на число известных E. C — правильность уже выбранной базовой стороны. S — отсутствие достижения −R до первого qualifying crossing. S не означает прибыль фактического выхода.</p>
<p>Во всех вариантах доступен только прежний кандидат base score ≥0.35. Фильтр не меняет сторону. Semantic rank = qE·qC; risk rank = qE·qC·qS. Порог берётся на CAL, будущее состояние crossing не участвует в replay.</p>
<p>Это новые даты для scientific V3/V4; исторически нетронутый holdout не заявляется: старый cognition код назначал август в TEST. Пять TEST-дней дают ограниченную оценку устойчивости.</p></header>
{conclusion}<section><h2>Заранее выбранное сравнение: cost2, 50% CAL-кандидатов</h2><p>Порог переносится на TEST как число; фактическое число TEST-сигналов может различаться. Суммы в USD на reference ounce.</p><div class="scroll"><table><tr><th>Seed</th><th>Arm</th><th>Known alerts</th><th>NONE</th><th>Wrong side</th><th>Correct/risk first</th><th>Correct precision</th><th>Safe precision</th><th>Trades</th><th>Full net</th><th>Status</th></tr>{''.join(rows)}</table></div></section>
<section><h2>Политика, выбранная только на CAL</h2><p>Для допуска требуется полный положительный CAL net и ≥50 сделок. Иначе WAIT. Hash — только контроль и не выбирается как политика.</p><ul>{''.join(policy)}</ul></section>
{figures}<section><h2>Проверяемые артефакты</h2><p><a href="evaluation.json">Все пороги, buckets и экономика</a> · <a href="suite_summary.json">Fits и checkpoint</a> · <a href="source_inventory.json">Покрытие и исключения</a> · <a href="data_admission.json">Контекст и маски</a> · <a href="causal_verification.json">Независимый replay источников</a> · <a href="learning_audit.json">Аудит обучения</a> · <a href="economic_audit.json">Аудит исполнения и риска</a> · <a href="sol61_visual_audit.json">Sol 6.1</a></p></section></main></html>'''
    (ROOT/'index.html').write_text(text,encoding='utf-8',newline='\n')


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    c,e,suite,b=load()
    learning_figures(c,e,b); ranking_figures(e); economics_figures(e,c); scenes(c,e,b); trade_cases(c,e,b); candidate_funnel(c,e,b); html_report(c,e)
    save(ROOT/'visual_manifest.json',dict(status='READY_FOR_PIXEL_AUDIT',renderer_sha256=sha(__file__),
        inputs={str(ROOT/n):sha(ROOT/n) for n in ('evaluation.json','suite_summary.json')},figures=FIGURES,html_sha256=sha(ROOT/'index.html')))
    print('Rendered',len(FIGURES),'figures',flush=True)


if __name__=='__main__': main()
