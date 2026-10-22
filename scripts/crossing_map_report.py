"""Visual atlas of time/equity states, event definitions, and factorized loss."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import FancyBboxPatch
from scripts.crossing_map_build import OUT,REPORT,DATES
from scripts.flow_context_coverage import save,sha


def main():
    summary=json.loads((REPORT/'summary.json').read_text())
    f=pd.concat([pd.read_parquet(OUT/(date+'_map.parquet')) for date in DATES],ignore_index=True)
    targets=pd.read_parquet(OUT/'targets.parquet')
    case_ns=pd.Timestamp('2026-02-04 12:00:00',tz='UTC').value
    case=f.loc[f.t_ns==case_ns].sort_values(['family','scale','role'])
    save(REPORT/'case_receipt.json',dict(date='2026-02-04',decision_ns=str(case_ns),selection='Fixed Wednesday noon, before plotting; not selected by net outcome',
        cells=[{k:(str(int(v)) if k.endswith('_ns') else None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()}
               for row in case[['family','scale','role','entry_ns','cross_ns','event_id','delay_s','mark_L','mark_S','mae_L','mae_S','risk_R']].to_dict('records')]))
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.15})
    images=[]
    def finish(fig,name):
        fig.savefig(REPORT/name,dpi=145,bbox_inches='tight');plt.close(fig);images.append(name)
    labels=[f'M{s} {role}' for s in [1,3,5,6,9] for role in ['lower','middle','upper']]
    fig,ax=plt.subplots(figsize=(16,7));ax.set(xlim=(0,16),ylim=(0,7));ax.axis('off')
    boxes=[(0.2,4,3.5,2.2,'CAUSAL ORIGIN X','Quotes, flow, BB/VWAP geometry\nKnown cost and price scale\nR fixed BEFORE future lookup'),
        (4.3,4,5.4,2.2,'FUTURE MAP / teacher only','2 families x 5 scales x 3 lines\nEach line has its own next event ID\ntime, net L/S, full-prefix MAE/MFE'),
        (10.3,4,5.4,2.2,'SEPARATE QUESTIONS','E: net >= G within D exists?\nDirection: L-only / S-only / BOTH\nTiming + magnitude + path risk'),
        (1.5,.4,6,2.4,'IMPLEMENTED LOSS CONTRACT','BCE(E) + E-positive conditional CE(direction)\nUnknown rows masked; NONE has no direction loss\nRoot averaging; physical IDs never enter X'),
        (8.5,.4,6,2.4,'PROPOSED NEXT MODEL','Crossing-time survival + net/MAE quantiles\nAverage map heads, do not multiply by line count\nExecution/risk policy evaluated separately')]
    for x,y,w,h,title,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor='#edf3f7',edgecolor='#6a8499'))
        ax.text(x+w/2,y+h-.35,title,ha='center',weight='bold',fontsize=12)
        ax.text(x+w/2,y+.8,text,ha='center',va='center',fontsize=10.5,linespacing=1.7)
    for a,b in [((3.85,5.1),(4.13,5.1)),((9.85,5.1),(10.1,5.1)),((12,3.8),(12,3.0)),((7,3.8),(5,3.0))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=2,color='#47758f'))
    fig.suptitle('Developer decision: preserve the future profile; factor event, direction, time and risk\nThis audit implements the map and E/direction loss; it does not train a new predictor',fontsize=15)
    finish(fig,'01_target_loss_scheme.png')
    # Rows are line identities, columns are successive origins, not one account path.
    roi=f.loc[(f.t_ns>=case_ns)&(f.t_ns<case_ns+7200*10**9)].copy()
    fig,axes=plt.subplots(3,2,figsize=(19,16),layout='constrained')
    mats={}
    for family in ['BB','VWAP']:
        a=roi.loc[roi.family==family].copy();a['line']=[f'M{s} {r}' for s,r in zip(a.scale,a.role.map({0:'lower',1:'middle',2:'upper'}))]
        for key in ['delay_s','mark_L','mark_S']:
            vals=a[key]/(60 if key=='delay_s' else a.risk_R)
            a['value']=np.log1p(vals) if key=='delay_s' else vals
            mats[(family,key)]=a.pivot(index='line',columns='t_ns',values='value').reindex(labels).to_numpy()
    vmax=max(np.nanmax(abs(mats[(fam,k)])) for fam in ['BB','VWAP'] for k in ['mark_L','mark_S'])
    for col,fam in enumerate(['BB','VWAP']):
        for row,key in enumerate(['delay_s','mark_L','mark_S']):
            ax=axes[row,col];matrix=mats[(fam,key)];cmap=plt.get_cmap('viridis' if key=='delay_s' else 'RdBu_r').copy();cmap.set_bad('#d8dce0')
            norm=colors.Normalize(0,max(np.nanmax(mats[(ff,key)]) for ff in ['BB','VWAP'])) if key=='delay_s' else colors.SymLogNorm(linthresh=1,vmin=-vmax,vmax=vmax)
            im=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',interpolation='nearest',origin='upper',cmap=cmap,norm=norm,extent=(0,120,14.5,-.5))
            ax.set(yticks=np.arange(15),yticklabels=labels,xticks=[0,30,60,90,120],xticklabels=['12:00','12:30','13:00','13:30','14:00'],xlabel='Decision time UTC')
            ax.set_title(f'{fam}: '+('time to next crossing of THIS line' if row==0 else f'{"LONG" if row==1 else "SHORT"} net checkpoint / decision R'))
            cb=fig.colorbar(im,ax=ax,pad=.01)
            if row==0:
                ticks=[0,1,5,15,30,60,120,240,480];ticks=[t for t in ticks if np.log1p(t)<=norm.vmax]
                cb.set_ticks(np.log1p(ticks),labels=[str(t) for t in ticks]);cb.set_label('Minutes, log(1+t) scale')
            else:cb.set_label('Net / R; symlog, no clipping')
    fig.suptitle('Wednesday 12:00-14:00: 30-cell future map from every 30-second origin\nGrey = unresolved/unsupported. Each column is a separate hypothetical entry; columns are NOT cumulative equity.',fontsize=15)
    finish(fig,'02_time_equity_map.png')
    fig,axes=plt.subplots(2,2,figsize=(17,12),layout='constrained')
    case_max=np.nanmax(abs(case[['mark_L','mark_S','mae_L','mae_S']].to_numpy()))
    for row,fam in enumerate(['BB','VWAP']):
        a=case.loc[case.family==fam].sort_values(['scale','role']);known=a.cross_ordinal>=0;ax=axes[row,0]
        for i,(_,r) in enumerate(a.iterrows()):
            if r.cross_ordinal>=0:
                ax.plot([0,r.delay_s/60],[i,i],color='#b3bec6',lw=1)
                ax.scatter(r.delay_s/60,i,c=r.mark_L,cmap='RdBu_r',norm=colors.SymLogNorm(linthresh=1,vmin=-case_max,vmax=case_max),s=60)
            else:ax.text(.02,i,'unknown',color='#8d949a',va='center')
        ax.set(yscale='linear',xscale='symlog',yticks=np.arange(15),yticklabels=labels,ylim=(14.7,-.7),xlabel='Minutes from same entry (symlog)',title=f'{fam}: chronological location of each first-line event')
        ax=axes[row,1];mat=a[['mark_L','mark_S','mae_L','mae_S']].to_numpy();cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d8dce0')
        im=ax.imshow(np.ma.masked_invalid(mat),aspect='auto',cmap=cmap,norm=colors.SymLogNorm(linthresh=1,vmin=-case_max,vmax=case_max))
        ax.set(yticks=np.arange(15),yticklabels=labels,xticks=range(4),xticklabels=['Net LONG','Net SHORT','MAE LONG','MAE SHORT'],title=f'{fam}: net liquidation marks / full-prefix risk, USD')
        for i in range(15):
            for j in range(4):
                if np.isfinite(mat[i,j]):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=9,color='black',bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=.3))
        fig.colorbar(im,ax=ax,pad=.01,label='USD per reference oz; no clipping')
    entry=pd.Timestamp(int(case.entry_ns.iloc[0]),tz='UTC');risk=float(case.risk_R.iloc[0])
    fig.suptitle(f'Fixed entry {entry.strftime("%Y-%m-%d %H:%M:%S.%f")} UTC; R=${risk:.2f}\nEach side starts from hypothetical $1500: equity at a cell = 1500 + net. Alternative cells are never summed.',fontsize=14)
    finish(fig,'03_one_entry_sequence.png')
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained');scales=[1,3,5,6,9]
    for role,ax in enumerate(axes.flat[:3]):
        for family,color in [('BB','#29679a'),('VWAP','#bd542d')]:
            vals=[100*summary['coverage'][family][str(s)][str(role)]['observed_by_seconds']['1800'] for s in scales]
            ax.plot(scales,vals,marker='o',color=color,label=family)
        ax.set(xticks=scales,ylim=(0,100),xlabel='Native scale (minutes)',ylabel='Observed by 30 minutes, % of ready origins',title=['Lower','Middle','Upper'][role]);ax.legend()
    ax=axes[1,1]
    for family,color in [('BB','#29679a'),('VWAP','#bd542d')]:
        vals=[100*summary['paired_fast5_fast6'][family][str(d)]['safe_event_agreement'] for d in [300,900,1800]]
        ax.plot([5,15,30],vals,marker='o',label=family,color=color)
    ax.set(xticks=[5,15,30],ylim=(90,100),xlabel='Deadline, minutes',ylabel='Agreement on same mature origins, %',title='M1/M3/M5 vs M1/M3/M6: safe event');ax.legend()
    fig.suptitle('Compressing the scales changes maturity and outcomes; it does not eliminate path risk\nCoverage denominators are scale-specific; set agreement uses paired common support.',fontsize=14)
    finish(fig,'04_scale_coverage.png')
    fig,axes=plt.subplots(2,1,figsize=(16,10),layout='constrained')
    for ax,family in zip(axes,['BB','VWAP']):
        cats=[];data=[];ns=[]
        for d in [300,900,1800]:
            for name in ['raw','amplitude','safe']:
                r=summary['labels'][family]['fast6'][str(d)][name]
                cats.append(f'{d//60}m\n{name}');data.append([r['counts'][k]/r['mature'] for k in ['NONE','LONG_ONLY','SHORT_ONLY','BOTH']]);ns.append(r['mature'])
        data=np.array(data);bottom=np.zeros(len(data))
        for i,(name,color) in enumerate(zip(['NONE','LONG only','SHORT only','BOTH'],['#b3bdc5','#286a9e','#cc6a32','#4a9b83'])):
            ax.bar(np.arange(len(data)),data[:,i]*100,bottom=bottom*100,label=name,color=color);bottom+=data[:,i]
        for i,n in enumerate(ns):ax.text(i,101,f'n={n}',ha='center',fontsize=8)
        ax.set(xticks=range(len(data)),xticklabels=cats,ylim=(0,111),ylabel='Mature roots, %',title=f'{family}, M1/M3/M6: raw >0; amplitude >=2R; safe adds prefix MAE > -R');ax.legend(ncol=4,loc='lower right')
    fig.suptitle('Favorable is only meaningful with an explicit magnitude, time and risk definition\nUnknown outcomes excluded from these distributions, never labelled NONE. Primary R is fixed at decision time.',fontsize=14)
    finish(fig,'05_favorable_distribution.png')
    fig,axes=plt.subplots(3,1,figsize=(16,11),layout='constrained',sharex=True)
    t=targets.loc[(targets.family=='BB')&(targets.scale_set=='fast6')&(targets.deadline==900)&(targets.t_ns>=case_ns)&(targets.t_ns<case_ns+7200*10**9)].copy()
    xx=(t.t_ns-case_ns)/60e9;known=t.all_known
    for offset,name,color in [(2,'raw','#778794'),(1,'amplitude','#286a9e'),(0,'safe','#bd542d')]:
        y=((t[name+'_state']>0).astype(float)).where(known)
        axes[0].step(xx,y*.7+offset,where='post',label=name,color=color,lw=1.2)
    axes[0].set(yticks=[.35,1.35,2.35],yticklabels=['safe','amplitude','raw'],title='Actual binary future event labels (levels vertically offset)');axes[0].legend(ncol=3)
    axes[1].plot(xx,t.quality_L.where(known),label='LONG quality',color='#286a9e');axes[1].plot(xx,t.quality_S.where(known),label='SHORT quality',color='#bd542d')
    axes[1].set(ylim=(0,1),ylabel='Bounded quality',title='Max discounted checkpoint quality; descriptive utility, NOT probability');axes[1].legend()
    for name,color in [('raw','#778794'),('amplitude','#286a9e'),('safe','#bd542d')]:
        prior=summary['event_gradient_at_fit_prior']['BB'][name]['fit_event_prior'];gradient=(prior-(t[name+'_state']>0).astype(float)).where(known)
        axes[2].plot(xx,gradient,label=f'{name}, fit prior={prior:.3f}',color=color,lw=.9)
    axes[2].set(ylim=(-1,1),ylabel='d BCE / d logit',xlabel='Minutes after Wednesday 12:00 UTC',title='Gradient at constant Mon-Tue prior; no learned model');axes[2].legend(ncol=3)
    fig.suptitle('The hard risk boundary makes event labels rough again\nChanging the factorization does not prove lower irreducible noise; soft quality changes the question being learned.',fontsize=14)
    finish(fig,'06_targets_and_gradients.png')
    # Per-line positivity prevents a line name from silently becoming a direction label.
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    for family,color in [('BB','#29679a'),('VWAP','#bd542d')]:
        axes[0].plot(scales,[100*summary['coverage'][family][str(s)]['2']['long_positive_rate'] for s in scales],marker='o',label=family,color=color)
        axes[1].plot(scales,[100*summary['interarrival'][family][str(s)]['under1_fraction'] for s in scales],marker='o',label=family,color=color)
    axes[0].set(xticks=scales,ylim=(0,100),xlabel='Scale',ylabel='Positive LONG checkpoint, %',title='Upper-line crossing does not imply profitable LONG')
    axes[1].set(xticks=scales,ylim=(0,100),xlabel='Scale',ylabel='Same-line intervals below one second, %',title='Full-publication IDs still contain micro-recrosses')
    for ax in axes:ax.legend()
    fig.suptitle('Separate physical event identity, side of the line, and economic trade direction\nRates are descriptive over observed crossings; adjacent same-line time intervals include support gaps.',fontsize=14)
    finish(fig,'07_direction_and_ids.png')
    rows=''
    for d in [300,900,1800]:
        for name in ['raw','amplitude','safe']:
            r=summary['labels']['BB']['fast6'][str(d)][name]
            rows+=f'<tr><td>{d//60} мин</td><td>{name}</td><td>{r["mature"]:,}</td><td>{r["event_rate"]:.2%}</td><td>{r["both_rate"]:.2%}</td><td>{r["event_flip"]:.2%}</td></tr>'
    captions=['Фактическая карта и реализованный event/direction loss. Временная survival и денежные квантили — схема следующего обучения, не уже обученная модель.',
        'Каждая колонка — новый вход; каждая строка — своя линия. Время и LONG/SHORT marks взяты из будущего и не подаются в X.',
        'Один заранее выбранный вход: последовательность первых событий каждой линии и состояния двух альтернативных позиций. MAE включает весь путь до crossing.',
        'M5/M6/M9: полнота первых пересечений каждой линии и сравнение наборов на общей поддержке.',
        'Разные определения favorable дают разные задачи. В распределениях сохранён BOTH; неизвестное не равно NONE.',
        'Реальные метки и мягкая оценка качества; градиент BCE при константном prior первых двух дней. Это не прогноз обученной модели.',
        'Название линии не определяет денежное направление. Новый ID не означает независимое движение.']
    page=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Полная карта пересечений — target/loss</title>
<style>body{{font:16px/1.6 system-ui;background:#f3f5f7;color:#182b3b;margin:auto;max-width:1550px;padding:28px}}h1{{line-height:1.2}}figure,.card{{background:white;border:1px solid #d3dce4;border-radius:10px;margin:25px 0;padding:22px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}a{{color:#176b99}}code{{background:#e8eef2;padding:2px 4px}}figcaption{{color:#506273}}</style>
<h1>Полная карта пересечений: время, магнитуда, equity и риск</h1><p>2–6 февраля 2026 · GCJ6 · {summary['origins']:,} исходных точек · {summary['cells']:,} ячеек · {summary['bank_events']:,} физических событий BB/VWAP в банке.</p>
<div class="card"><b>Решение для следующего опыта.</b> Сохранить карту исходов и разделить event, направление, время и риск. Для основной короткой задачи проверить magnitude-favorable на 5 минут с фиксированным при решении порогом G=2R. Для 15/30 минут оставить отдельные горизонты. Риск до достижения результата учить отдельно; safe-checkpoint и полный gain-before-stop использовать как строгие проверки. Упростить вопрос модели не означает объявить безопасным вход по одному favorable.</div>
<p>В каждой точке теперь 30 ячеек: BB/VWAP × M1/M3/M5/M6/M9 × lower/middle/upper. Каждая ячейка ждёт <b>своё</b> первое пересечение после исполненного входа. Более раннее пересечение другой линии её не завершает. Фактическая карта продолжается до конца поддержанной сессии, без потолка 30 минут; горизонты 5/15/30 минут применяются только к определениям целей.</p>
<p>Геометрия: native SMA20±2 sample SD и trade VWAP±2 population SD, 19 завершённых native свечей + текущая. Полный порядок публикаций сохранён, включая разные публикации с одинаковыми ns. Вход — первая поддержанная цена после 100 ms, timeout 2 s; BUY ask, SELL bid; комиссия $0.30 за одну условную унцию. Equity состояния = условные $1500 + net mark. Это альтернативные оценки ликвидации, не сумма сделок и не equity работающей стратегии. Доступный выход с задержкой сохранён отдельно как net_exit; определения favorable ниже используют mark на crossing, не обещанный fill.</p>
<h2>Что следует называть favorable</h2><p>R=max(2×(spread решения+$0.30), 0.10×полуширина BB1 решения, $0.50); G=2R. Медиана R=${summary['risk_query']['median_R']:.2f}; в {summary['risk_query']['cost_floor_fraction']:.1%} поддержанных точек R определяется нижним пределом по издержкам. Порог причинный и не подбирается по будущей прибыли.</p>
<div class="card"><table><tr><th>Горизонт</th><th>Определение</th><th>Полностью известных точек</th><th>Event</th><th>BOTH</th><th>Смена event за 30s</th></tr>{rows}</table></div>
<p>Таблица: BB, M1/M3/M6. <b>raw</b>: хотя бы одна отметка &gt;0. <b>amplitude</b>: хотя бы одна отметка ≥G. <b>safe</b>: отметка ≥G и весь предыдущий MAE&gt;−R. На горизонте 15 минут raw уже почти константа, а safe снова рваный. Разные сроки имеют разные maturity выборки. Здесь измерена стабильность целей, не их предсказуемость из X.</p>
<p>Мягкая вспомогательная оценка стороны: max по ячейкам до D от tanh(max(net,0)/R) × exp(−max(−MAE,0)/R) × exp(−delay/D). Она сохраняет магнитуду, риск и время. Это hindsight качество доступного пути, не вероятность выигрыша и не причинный выбор лучшего выхода. Точное значение max известно только при maturity всех включённых ячеек.</p>
<h2>Разделение event и направления</h2><p>E=есть amplitude-возможность в пределах D. При E=1 направление имеет три состояния: LONG_ONLY, SHORT_ONLY, BOTH. При E=0 направление не обучается. BOTH означает две возможности на разных будущих участках пути, а не два одновременно прибыльных исполнения. Имя upper/lower и направление пересечения не подменяют денежную сторону сделки.</p>
<p>Реализован и проверен loss: <code>BCE(E) + 1[E=1] CE(direction)</code>, с отдельными масками неизвестного event и неизвестного направления. Совместные вероятности согласованы: P(NONE)=1−pE; P(L/S/BOTH)=pE×p(direction|E). Это факторизация той же совместной вероятности, а не математическое доказательство, что задача стала легче. Максимальная ошибка численной проверки градиента {summary['factorized_gradient_max_error']:.2g}.</p>
<p>Для следующей модели к нему следует добавить условную survival времени crossings и квантили net/R и MAE/R. Усреднять головы карты по исходной точке, чтобы добавление соседнего масштаба не увеличивало её вес. Event ID применять для подсчёта повторных примеров только внутри FIT; сохранять временное разбиение и purge по maturity. Проверенные здесь physical IDs всё ещё дробят тиковые возвраты.</p>
<p><b>Отбор примеров.</b> Для первого BCE/CE сравнения использовать одну и ту же полностью mature выборку положительных и отрицательных точек, как в распределениях этого отчёта. Ранний положительный witness позволяет знать E=1, но смешивание всех таких positives только с mature negatives может сместить обучение из-за зависимости наблюдаемости от исхода. Частичные наблюдения сохранять для явно заданного censor-aware likelihood. Полностью mature выборка также задаёт условную популяцию: её coverage и исключения нужно показывать. Правила зафиксированы в <a href="../../contracts/crossing_map_training_proposal_v1.json">предложении для следующего обучения</a>.</p>
<h2>Нужен ли M9, стоит ли заменить его на M5</h2><p>M5 соответствует примерно 100 минутам памяти, M6 — 120, M9 — 180; это не сроки выхода. Крупные outer-линии часто не достигаются за 30 минут. M5 увеличивает покрытие относительно M6, но не устраняет длинное ожидание.</p>
<p>На одинаковых mature точках наборы M1/M3/M5 и M1/M3/M6 совпадают по safe event на 15 минут в {summary['paired_fast5_fast6']['BB']['900']['safe_event_agreement']:.1%} случаев BB и {summary['paired_fast5_fast6']['VWAP']['900']['safe_event_agreement']:.1%} VWAP. Их конкретные времена/marks различаются; это не тождественные карты. Добавление M9 к M1/M3/M6 даёт {summary['incremental_m9']['BB']['900']['safe_event_added_by_m9']} дополнительных safe-positive BB и {summary['incremental_m9']['VWAP']['900']['safe_event_added_by_m9']} VWAP на {summary['incremental_m9']['BB']['900']['common_mature']:,} общих точках.</p>
<p>Рабочий выбор для следующего замороженного сравнения: M1/M3/M6 как основной набор короткой цели, M5 как отдельная альтернатива M6, M9 как дальний контекст и продолжение. Неделя не доказывает превосходство M6 над M5; одновременное добавление M5 и M6 не должно считаться независимым удвоением информации.</p>
'''+''.join(f'<figure><a href="{name}"><img src="{name}" alt="{cap}" loading="lazy"></a><figcaption>{cap}</figcaption></figure>' for name,cap in zip(images,captions))+'''
<h2>Воспроизводимость и границы</h2><p><a href="summary.json">Распределения, coverage, сравнение масштабов</a> · <a href="build_receipt.json">Банк и независимые проверки</a> · <a href="validation.json">Full-quote first-passage и тесты</a> · <a href="case_receipt.json">Точные IDs и времена показанного входа</a> · <a href="SOL61_VISUAL_AUDIT_RU.md">Аудит Sol 6.1</a></p><p>Новая предсказывающая модель в этом опыте не обучалась. Все пять дней уже исследовались. Выводы — проектирование таргета и loss, а не подтверждение обучаемости, прибыли или допуска к торговле. Данные сохранены в data/derived/crossing_map_target_v1; исходные BB/VWAP опыты не изменены.</p></html>'''
    (REPORT/'index.html').write_text(page,encoding='utf-8',newline='\n')
    save(REPORT/'visual_receipt.json',dict(images={name:sha(REPORT/name) for name in images},page_sha256=sha(REPORT/'index.html'),
        summary_sha256=sha(REPORT/'summary.json'),script_sha256=sha(__file__)))
    print(json.dumps(dict(images=len(images))),flush=True)


if __name__=='__main__':main()
