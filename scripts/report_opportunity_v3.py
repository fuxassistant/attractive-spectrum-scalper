"""Read-only figures for the complete matched context and ID-weighting suite.
All fixed time cuts are preregistered, probabilities and targets are unsmoothed.
"""
from pathlib import Path
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from spectrum.research_v3.learning import load_dataset, sha

ROOT=Path('reports/scientific_pipeline_v3')
COLORS=['#475569','#d97706','#2563eb','#059669']
ARMS=[('legacy','uniform'),('legacy','crossing_id'),('restored','uniform'),('restored','crossing_id')]
LABELS=['Legacy / uniform','Legacy / ID weights','Restored / uniform','Restored / ID weights']
PNAMES=['p_none','p_long','p_short','p_both']

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def finish(fig,name):
    fig.savefig(ROOT/name,dpi=135,bbox_inches='tight',facecolor='white'); plt.close(fig)

def main():
    suite=read(ROOT/'suite_summary.json')
    if suite['status']!='COMPLETE': raise ValueError('Complete eight-arm suite required')
    config=read('configs/research/opportunity_pipeline_v3.json')
    if sha('configs/research/opportunity_pipeline_v3.json')!=suite['config_sha256']: raise ValueError('Preregistered configuration changed')
    store=load_dataset(suite['dataset_root']); f=store.frame
    manifest=read(Path(suite['dataset_root'])/'manifest.json')
    if sha(Path(suite['dataset_root'])/'manifest.json')!=suite['dataset_manifest_sha256']: raise ValueError('Dataset changed')
    bundles={}; inputs={str(ROOT/'suite_summary.json'):sha(ROOT/'suite_summary.json'),str(Path(suite['dataset_root'])/'manifest.json'):suite['dataset_manifest_sha256'],
                        'configs/research/opportunity_pipeline_v3.json':suite['config_sha256']}
    for entry in suite['runs']:
        path=Path(suite['runs_root'])/entry['run_id']; rec=read(path/'receipt.json')
        if rec['status']!='COMPLETE' or rec['dataset_manifest_sha256']!=suite['dataset_manifest_sha256']: raise ValueError('Incomplete or unmatched run')
        for name,h in rec['outputs'].items():
            if sha(path/name)!=h: raise ValueError('Changed run output: '+name)
        for name,h in rec['sources'].items():
            if sha(name)!=h: raise ValueError('Changed source: '+name)
        inputs[str(path/'receipt.json')]=sha(path/'receipt.json')
        b=dict(entry=entry,path=path,summary=read(path/'summary.json'),p=pd.read_parquet(path/'predictions.parquet'),history=pd.DataFrame(read(path/'history.json')),economics=read(path/'economic_evaluation.json'))
        if (not np.array_equal(b['p'].t_ns,f.t_ns) or not np.array_equal(b['p'].y_crossing,f.y_crossing)
            or not np.array_equal(b['p'].root_index,np.arange(len(f)))
            or not np.array_equal(b['p'].feature_available,f.feature_available)
            or not np.array_equal(b['p'].date,f.date)): raise ValueError('Unmatched full-clock prediction')
        if b['summary']['context_mode']!=entry['context'] or b['summary']['weighting']!=entry['weighting']: raise ValueError('Run identity differs from suite arm')
        bundles[(entry['seed'],entry['context'],entry['weighting'])]=b
    expected={(seed,*arm) for seed in (17,29) for arm in ARMS}
    if len(suite['runs'])!=8 or set(bundles)!=expected: raise ValueError('Exactly eight distinct prescribed arms required')
    baseline=bundles[(17,'legacy','uniform')]['summary']
    for b in bundles.values():
        s=b['summary']
        if (s['cohorts']!=baseline['cohorts'] or s['scaler_sha256']!=baseline['scaler_sha256']
            or s['feature_names']!=baseline['feature_names'] or s['input_channels']!=baseline['input_channels']):
            raise ValueError('Unmatched cohort, scaler, or feature schema')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    # Context span is physical clock availability, not a claim every row was observed.
    fig,ax=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    roles=list(config['splits']); x=np.arange(4)
    for mode,color,shift in [('legacy',COLORS[0],-.18),('restored',COLORS[2],.18)]:
        c=bundles[(17,mode,'uniform')]['summary']['context']['by_split']
        ax[0].bar(x+shift,[c[r]['oldest_observed_feature_span_seconds']['q50'] for r in roles],.34,color=color,label=mode)
        ax[1].bar(x+shift,[100*c[r]['full_supported_clock_receptive_field_fraction'] for r in roles],.34,color=color)
        ax[2].bar(x+shift,[c[r]['observed_feature_rows']['mean'] for r in roles],.34,color=color)
    for a in ax: a.set_xticks(x,roles,rotation=15)
    ax[0].set(ylabel='Median oldest observed X age, seconds',title='Observed span, not a contiguous observed path (max 300 s)'); ax[0].legend()
    ax[1].set(ylabel='% known admitted roots',title='Full supported clock window supplied: 61 positions',ylim=(0,105))
    ax[2].set(ylabel='Mean observed X rows out of 61',title='Masks still remove unavailable observations',ylim=(0,64))
    fig.suptitle('Context repair: same current admission, same 70 values, different causal history')
    finish(fig,'01_context.png')
    fig,ax=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    keys=['joint_nll','event_auc','direction_nll_given_event']
    for row,seed in enumerate((17,29)):
        for col,key in enumerate(keys):
            vals=[bundles[(seed,*arm)]['summary']['evaluation']['test']['model'][key] for arm in ARMS]
            ax[row,col].bar(np.arange(4),vals,color=COLORS)
            ax[row,col].set_xticks(np.arange(4),['L/U','L/ID','R/U','R/ID'])
            ax[row,col].set_title(f'Seed {seed}: TEST {key}')
            for k,v in enumerate(vals): ax[row,col].text(k,v,f'{v:.4f}',ha='center',va='bottom')
            ax[row,col].set_ylim(0,max(vals)*1.14)
    fig.suptitle('Every prespecified arm shown; lower NLL is better, higher AUC is better. No TEST selection.')
    finish(fig,'02_comparison.png')
    fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for row,seed in enumerate((17,29)):
        for j,arm in enumerate(ARMS):
            b=bundles[(seed,*arm)]; h=b['history']; ax[row,0].plot(h['update'],h.validation_nll,'o-',label=LABELS[j],color=COLORS[j],ms=3)
            selected=h.loc[h['update'].eq(b['summary']['training']['best_update'])].iloc[0]
            ax[row,0].scatter([selected['update']],[selected.validation_nll],marker='*',s=105,color=COLORS[j],edgecolors='black',linewidths=.5,zorder=5)
            ax[row,1].plot(h['update'],h.gradient_clip_fraction,'o-',color=COLORS[j],ms=3)
        ax[row,0].set(title=f'Seed {seed}: unweighted VAL joint NLL',xlabel='Updates',ylabel='NLL')
        ax[row,1].set(title=f'Seed {seed}: optimizer gradient clipping',xlabel='Updates',ylabel='Fraction of batches clipped',ylim=(-.02,1.02))
    ax[0,0].legend(fontsize=9); fig.suptitle('Matched initialization and sampled-root prefixes; stars mark selected checkpoints by unweighted VAL NLL')
    finish(fig,'03_training.png')
    # Fixed test windows, never selected for performance or visual attractiveness.
    fixed_panels=[(seed,case) for seed in (17,29) for case in config['experiment']['fixed_visual_cases']]
    for case_index,(plot_seed,case) in enumerate(fixed_panels):
        date=case['date']; date_ids=np.flatnonzero(f.date.eq(date)); t0=int(f.t_ns.iloc[date_ids[0]])+int(case['offset_hours']*3600e9)
        ids=np.flatnonzero(f.date.eq(date)&f.t_ns.ge(t0)&f.t_ns.lt(t0+int(case['duration_hours']*3600e9)))
        x=(f.t_ns.iloc[ids].to_numpy()-t0)/60e9; y=f.y_crossing.iloc[ids].to_numpy(); admitted=f.feature_available.iloc[ids].to_numpy(); known=(y>=0)&admitted
        scene=pd.read_parquet(Path(suite['dataset_root'])/(date+'_scene.parquet')); local=ids-date_ids[0]; scene=scene.iloc[local]
        if not np.array_equal(scene.t_ns,f.t_ns.iloc[ids]): raise ValueError('Fixed scene is not aligned with origin clock')
        physical=f.physical_support_available.iloc[ids].to_numpy()
        fig,ax=plt.subplots(7,1,figsize=(17,17),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1.6,1.5,1.4,1.3,1,0.6]})
        ax[0].plot(x,np.where(physical,scene.mid,np.nan),color='#111827',lw=.9,label='mid (physical support)')
        for scale,color in [(1,'#f59e0b'),(3,'#8b5cf6'),(6,'#0ea5e9')]:
            valid=(scene[f'BB{scale}_support'].to_numpy()>=0)&physical
            mid=np.where(valid,scene[f'BB{scale}_middle'],np.nan); half=np.where(valid,scene[f'BB{scale}_half'],np.nan)
            ax[0].plot(x,mid,color=color,lw=.8,label=f'BB M{scale}'); ax[0].fill_between(x,mid-half,mid+half,color=color,alpha=.08)
        ax[0].set_ylabel('Price USD/oz'); ax[0].legend(ncol=4,loc='upper left')
        ax[1].step(x,np.where(known,(y>0).astype(float),np.nan),where='post',color='black',alpha=.5,lw=.7,label='observed E (unsmoothed)')
        for j,arm in enumerate(ARMS):
            p=bundles[(plot_seed,*arm)]['p'].iloc[ids]; probs=p[PNAMES].to_numpy(); score=np.where(admitted,1-probs[:,0],np.nan)
            ax[1].plot(x,score,color=COLORS[j],lw=.85,label=LABELS[j])
            ax[2].plot(x,np.where(admitted,probs[:,1]-probs[:,2],np.nan),color=COLORS[j],lw=.85)
            ax[3].plot(x,np.where(known,p.g_event,np.nan),color=COLORS[j],lw=.8)
            nll=-np.log(np.clip(probs[np.arange(len(y)),np.maximum(y,0)],1e-12,1)); ax[4].plot(x,np.where(known,nll,np.nan),color=COLORS[j],lw=.8)
        ax[1].set(ylabel='E and P(E)',ylim=(-.05,1.05)); ax[1].legend(fontsize=8,ncol=3,loc='upper left')
        ax[2].step(x,np.where(known,(np.isin(y,[1,3]).astype(float)-np.isin(y,[2,3]).astype(float)),np.nan),where='post',color='black',alpha=.35,lw=.6)
        ax[2].set(ylabel='Crossing P(L)-P(S) / L-S',ylim=(-1.05,1.05)); ax[3].set_ylabel('d BCE / dz (unweighted TEST)'); ax[3].axhline(0,color='black',lw=.5); ax[4].set_ylabel('Per-root joint NLL')
        ax[2].text(.01,.03,'L-S is 0 for both NONE and BOTH; see categorical ribbon below.',transform=ax[2].transAxes,fontsize=8,
                   bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        for mode,color in [('legacy',COLORS[0]),('restored',COLORS[2])]:
            starts=store.segment_starts(5,mode); age=np.where(physical,np.minimum(ids-starts[ids],60)*5,np.nan); ax[5].plot(x,age,color=color,label=mode,lw=1)
        ax[5].set_ylabel('Supported clock span, s'); ax[5].legend(ncol=2)
        ax[5].text(.99,.03,'Missing X is masked inside a supported clock span.',transform=ax[5].transAxes,ha='right',fontsize=8,
                   bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        ribbon=np.where(known,y,-1); cmap=ListedColormap(['#cbd5e1','#475569','#059669','#dc2626','#8b5cf6'])
        ax[6].imshow(ribbon[None,:],aspect='auto',extent=[x[0],x[-1]+5/60,0,1],cmap=cmap,norm=BoundaryNorm(np.arange(-1.5,4.5),5),interpolation='nearest')
        ax[6].set_yticks([]); ax[6].set_ylabel('Target'); ax[6].set_xlabel('Minutes from fixed window start; gray unknown/unavailable, slate NONE, green LONG, red SHORT, violet BOTH')
        fig.suptitle(f'{date}: fixed TEST window +{case["offset_hours"]} h from first decision, seed{plot_seed}. No target or score smoothing.')
        finish(fig,f'04{chr(97+case_index)}_signal_{date}_seed{plot_seed}.png')
    fig,ax=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for j,arm in enumerate(ARMS):
        b=bundles[(17,*arm)]; p=b['p']; mask=p['split'].eq('train')&p.feature_available&(p.y_crossing>=0); pp=p.loc[mask]; y=pp.y_crossing.to_numpy(); w=pp.optimization_weight_train_only.to_numpy()
        ge=np.abs(pp.g_event.to_numpy()); gd=pp.g_direction_norm.to_numpy(); a=ax.flat[j]; xx=np.arange(4)
        weighted_e=np.abs(pp.g_event_training_weighted.to_numpy()); weighted_d=pp.g_direction_norm_training_weighted.to_numpy()
        if not np.allclose(weighted_e,ge*w) or not np.allclose(weighted_d,gd*w): raise ValueError('Saved weighted derivatives differ from actual weight scaling')
        for offset,values,color,label in [(-.27,ge,'#93c5fd','|gE|'),(-.09,weighted_e,'#2563eb','w |gE|'),(.09,gd,'#fcd34d','||gD||'),(.27,weighted_d,'#d97706','w ||gD||')]:
            masses=np.array([values[y==k].sum() for k in range(4)]); masses/=max(masses.sum(),1e-12); a.bar(xx+offset,masses,.18,color=color,label=label)
        a.set_xticks(xx,['NONE','LONG','SHORT','BOTH']); a.set(title=LABELS[j],ylabel='Fraction of total head gradient mass',ylim=(0,1)); a.legend(ncol=4,fontsize=8)
    fig.suptitle('Seed17 selected-checkpoint TRAIN derivatives: gradient mass can change despite preserved sums of class weights')
    finish(fig,'05_gradient_mass.png')
    fig,ax=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    w=pd.read_parquet(bundles[(17,'restored','crossing_id')]['path']/'train_weights.parquet'); yy=w.y_crossing.to_numpy(); ww=w.weight.to_numpy()
    if not np.isfinite(ww).all() or np.any(ww<=0): raise ValueError('Invalid saved TRAIN weights')
    for cls,color in [(1,'#059669'),(2,'#dc2626'),(3,'#8b5cf6')]: ax[0].hist(ww[yy==cls],bins=np.geomspace(ww.min(),ww.max()+.01,40),histtype='step',color=color,label=['NONE','LONG','SHORT','BOTH'][cls])
    ax[0].set(xscale='log',yscale='log',xlabel='TRAIN crossing-ID weight',ylabel='Roots',title='Class-normalized inverse ID frequency'); ax[0].legend()
    for i,role in enumerate(('validation','test'),1):
        ax[i].plot([0,1],[0,1],'--',color='#94a3b8')
        for j,arm in enumerate(ARMS):
            bins=bundles[(17,*arm)]['summary']['evaluation'][role]['model']['reliability']; bins=[v for v in bins if v['count']]
            ax[i].plot([v['probability'] for v in bins],[v['frequency'] for v in bins],'o-',color=COLORS[j],label=LABELS[j],ms=3)
            for v in bins:
                if v['count']<50:
                    ax[i].plot(v['probability'],v['frequency'],'o',mfc='white',mec=COLORS[j],ms=5)
                    ax[i].annotate(f"n={v['count']}",(v['probability'],v['frequency']),xytext=(4,4+8*j),textcoords='offset points',fontsize=6,color=COLORS[j])
        ax[i].set(title=f'Seed17 {role}: unweighted E reliability',xlabel='Mean P(E), fixed 0.1 bins',ylabel='Observed E frequency',xlim=(0,1),ylim=(0,1))
    ax[2].legend(fontsize=8); fig.suptitle('Uniform-root E reliability. Open circles: n<50 correlated roots; sparse tails do not establish calibration.')
    finish(fig,'06_weights_calibration.png')
    fig,ax=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    for j,arm in enumerate(ARMS):
        b=bundles[(17,*arm)]; grid=b['economics']['calibration_grid']; rows=[r for r in grid if r['threshold']==.35]
        ax[0].plot([r['plan'] for r in rows],[r['net'] if r['completed'] else np.nan for r in rows],'o-',label=LABELS[j],color=COLORS[j])
        for k,r in enumerate(rows):
            if r['completed']: ax[0].annotate(str(r['trades'])+'t',(k,r['net']),fontsize=7,color=COLORS[j])
        incomplete=[r['plan'] for r in rows if not r['completed']]
        if incomplete: ax[0].text(.02,.03+.04*j,LABELS[j]+': censored '+', '.join(incomplete)+' (net omitted)',transform=ax[0].transAxes,fontsize=7,color=COLORS[j])
    ax[0].axhline(0,color='black',lw=.7); ax[0].legend(fontsize=8); ax[0].set(title='Complete CAL net at fixed threshold .35 (trade counts)',ylabel='USD per reference oz after fees',xlabel='Frozen risk plan; not a TEST choice')
    b=bundles[(17,'legacy','uniform')]; plans=b['economics']['paired_diagnostics']['plans']
    for side,color in [('LONG','#059669'),('SHORT','#dc2626')]:
        values=[v['by_split']['test']['conditional_crossing'][side]['fixed_gain_retention_fraction'] for v in plans.values()]
        ax[1].plot(list(plans),[np.nan if v is None else 100*v for v in values],'o-',label=side,color=color)
    ax[1].set(title='Oracle TEST: true crossing retains fixed gain',ylabel='% true side events, common fully observed trials',ylim=(0,105)); ax[1].legend()
    fig.suptitle('Execution is a separate experiment: wider stops can retain a target without creating profitable model entries')
    finish(fig,'07_execution.png')
    compact=[]
    for key,b in bundles.items():
        s=b['summary']; train_counts=s['cohorts']['train']['state_counts']; test_counts=s['cohorts']['test']['state_counts']
        compact.append(dict(seed=key[0],context=key[1],weighting=key[2],run_id=b['entry']['run_id'],
            test=s['evaluation']['test']['model'],validation=s['evaluation']['validation']['model'],
            train_prior_test=s['evaluation']['test']['prior'],evaluation=s['evaluation'],
            state_counts_train=train_counts,state_counts_test=test_counts,
            event_fraction_train=sum(train_counts[1:])/sum(train_counts),event_fraction_test=sum(test_counts[1:])/sum(test_counts),
            best_update=s['training']['best_update'],selected_checkpoint=s['training']['selected_checkpoint'],
            scaler_sha256=s['scaler_sha256'],cohorts=s['cohorts'],actual_context=s['context'],
            weight_diagnostics=s['weight_diagnostics'],selected_policy=b['economics']['summary']))
    save(ROOT/'report_metrics.json',dict(status='COMPLETE',runs=compact,inputs=inputs,source_sha256=sha(__file__),fixed_cases=config['experiment']['fixed_visual_cases']))
    rows=''.join('<tr><td>'+str(r['seed'])+'</td><td>'+r['context']+'</td><td>'+r['weighting']+'</td><td>'+f'{r["test"]["joint_nll"]:.6f}'+'</td><td>'+f'{r["train_prior_test"]["joint_nll"]:.6f}'+'</td><td>'+f'{r["test"]["event_auc"]:.6f}'+'</td><td>'+f'{r["test"]["direction_nll_given_event"]:.6f}'+'</td><td>'+str(r['best_update'])+'</td></tr>' for r in compact)
    prevalence=('<p>Общий TRAIN-prior оценён только по TRAIN и не меняется между условиями. '
                f'Доля event: TRAIN <b>{compact[0]["event_fraction_train"]:.2%}</b>, TEST <b>{compact[0]["event_fraction_test"]:.2%}</b>. '
                f'Числа NONE / LONG / SHORT / BOTH: TRAIN {compact[0]["state_counts_train"]}; TEST {compact[0]["state_counts_test"]}. '
                'Абсолютный NLL v3 нельзя интерпретировать как улучшение относительно v2: даты, контракт и распределение исходов изменились.</p>')
    economic_rows=[]
    for r in compact:
        selected=r['selected_policy']['selected']; replay=r['selected_policy']['test']
        status='Полный replay' if replay['completed'] else 'ЧАСТИЧНЫЙ: CENSORED'
        if replay['completed'] and replay['trades']==0:
            status+='; WAIT / нет завершённых сделок' if selected['threshold']>1 else '; нет завершённых сделок'
        economic_rows.append(f'<tr><td>{r["seed"]}</td><td>{r["context"]} / {r["weighting"]}</td>'
            f'<td>{html.escape(selected["plan"])}, p≥{selected["threshold"]:g}</td><td>{replay["trades"]}</td>'
            f'<td>{replay["net"]:+.2f}</td><td>{status}</td></tr>')
    economic_table=('<article><h2>Отдельный TEST-replay после выбора на CAL</h2><table><thead><tr>'
        '<th>Seed</th><th>Условие</th><th>План / порог CAL</th><th>Сделки</th><th>Closed net, USD/oz</th><th>Завершённость</th>'
        '</tr></thead><tbody>'+''.join(economic_rows)+'</tbody></table>'
        '<p>При CENSORED показан только закрытый cash до остановки, а не полный результат периода. '
        'Необработанные будущие дни не считаются нулевой доходностью. WAIT и отсутствие сделок не подтверждают прибыльный edge.</p></article>')
    figures=[p.name for p in sorted(ROOT.glob('0*.png'))]
    pictures=''.join('<figure><img src="'+p+'"><figcaption>'+p+'</figcaption></figure>' for p in figures)
    page='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Причинный контекст и crossing ID — v3</title><style>body{font:17px/1.55 system-ui;max-width:1450px;margin:auto;padding:40px;color:#162236;background:#f4f7fa}h1{font-size:38px;line-height:1.15}article,figure{background:white;padding:24px;border-radius:14px;margin:24px 0}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #dde4ed;text-align:left;padding:10px}a{color:#1656a1}.note{border-left:4px solid #d97706;padding-left:18px}code{font-size:14px}</style><h1>Контекст без искусственных обрывов<br>и веса по crossing ID</h1><p>Научный опыт v3 · GCQ6 · 1 июня — 24 июля 2026 · 36 сессий · TCN Micro · 2×2 условия × 2 seed</p><article><h2>Что именно сравнили</h2><p>Старый контекст и история, ограниченная физической поддержкой котировок. Пропуски X сохраняются на сетке 5 секунд с нулевыми значениями и масками. Прошлые наблюдения не исчезают из-за задержки отдельного признака. Никакого заполнения из будущего, сжатия времени или forward fill.</p><p>В обоих условиях — обычные веса либо обратная частота TRAIN crossing ID, с нормализацией внутри каждого совместного класса. Для BOTH берётся среднее двух обратных частот. Суммарные веса NONE/LONG/SHORT/BOTH сохранены. Идентификаторы никогда не входят в X.</p><p>Таргет BB M1/M3/M6 и BCE(E) + условная CE(D), правила входа, комиссии, риски и гиперпараметры сохранены из v2. Выбор checkpoint — только по обычному VAL joint NLL. Риск и порог — только CAL. Все восемь условий показаны без выбора победителя на TEST.</p><p class="note">Micro: 70 признаков, 141 входной канал, максимальный RF TCN — 61 точка, до 300 секунд между крайними токенами. Фактически поданные clock-позиции и наблюдённые X показаны отдельно; masked строки не считаются наблюдениями. Богатый набор 225 признаков сохранён, но в этом опыте не обучается. Подробный контекст «сегодня и вчера с Лондона» здесь не проверяется. Контракт GCQ6 внутри нового блока постоянен, однако отличается от GCM6 в v2. Это новые даты относительно v2, без заявления о никогда не просмотренном holdout.</p></article><article><h2>Полная таблица TEST</h2>'+prevalence+'<table><thead><tr><th>Seed</th><th>Контекст</th><th>Веса</th><th>Joint NLL ↓</th><th>TRAIN-prior NLL</th><th>E AUC ↑</th><th>D NLL ↓</th><th>Selected update</th></tr></thead><tbody>'+rows+'</tbody></table></article>'+economic_table+pictures+'<article><h2>Проверяемые артефакты</h2><p><a href="statistical_audit.json">Парный статистический аудит</a> · <a href="causal_verification.json">Проверка причинности и точного совпадения v2</a> · <a href="sol61_visual_audit.json">Визуальный аудит Sol 6.1</a> · <a href="report_metrics.json">Полные метрики</a> · <a href="source_inventory.json">Покрытие источников</a></p><p>На графике направлений L−S равен нулю и для NONE, и для BOTH; цветовая полоса различает эти состояния. Масса весов класса сохранена, но фактическая масса градиента зависит от текущей ошибки модели. Редкие reliability-bins помечены числом roots; коррелированные roots не являются независимыми наблюдениями и не подтверждают калибровку хвостов.</p><p>Пересечение — геометрический target. Оно не гарантирует исполнение или прибыль. Экономические цифры — taker BBO на эталонную унцию, без доказательства глубины, очереди и impact. Equity — закрытый cash, не внутрисделочная переоценка.</p></article></html>'
    for filename in ('statistical_audit.json','causal_verification.json','decision_findings.json','paired_figure_receipt.json','weight_semantics.json'):
        path=ROOT/filename
        if path.exists():
            evidence=read(path)
            if evidence.get('status') not in ('COMPLETE','PASS'):
                raise ValueError('Final report requires completed evidence: '+filename)
            inputs[str(path)]=sha(path)
    if (ROOT/'decision_findings.json').exists():
        conclusions=read(ROOT/'decision_findings.json')
        block='<article><h2>Результат и следующий шаг</h2>'+''.join('<p>'+html.escape(v)+'</p>' for v in conclusions['paragraphs'])+'</article>'
        page=page.replace('<article><h2>Полная таблица TEST</h2>',block+'<article><h2>Полная таблица TEST</h2>')
    (ROOT/'index.html').write_text(page,encoding='utf-8',newline='\n')
    save(ROOT/'visual_manifest.json',dict(status='COMPLETE',source_sha256=sha(__file__),inputs=inputs,figures={p:sha(ROOT/p) for p in figures},index_sha256=sha(ROOT/'index.html')))
    print(json.dumps(dict(status='COMPLETE',figures=figures)))

if __name__=='__main__': main()
