"""Scientific plots and one-second crossing diagnostics for the fixed week."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from spectrum.cognition.bb_crossing import crossing_rows
from scripts.flow_context_coverage import save,sha
from scripts.vwap_week_source import OUT,REPORT
from scripts.vwap_week_compare import DATES

COL={'bb':'#285b98','vwap':'#bb482a','vwma':'#28936b','vwap_mid':'#9270b1'}
LABEL={'bb':'Bollinger SMA20','vwap':'Trade VWAP','vwma':'Volume-weighted closes','vwap_mid':'T-weighted midpoint'}


def targets(f,name):
    t=f.t_ns.to_numpy();m=f[name].to_numpy();h=f[name+'_half'].to_numpy()
    support=np.where(np.isfinite(m)&np.isfinite(h)&(h>0),f.segment,-1).astype(np.int64)
    row,role,*_=crossing_rows(t,f.mid.to_numpy(),m,h,support)
    event=np.unique(row);bits=np.zeros(len(f),np.int64)
    np.bitwise_or.at(bits,row,1<<role)
    end=np.r_[np.flatnonzero(support[1:]!=support[:-1])+1,len(f)]
    ends=np.repeat(end,np.diff(np.r_[0,end]))
    origins=np.arange(0,len(f),30);chosen=np.searchsorted(event,origins,side='right')
    has=chosen<len(event);eventrow=event[np.minimum(chosen,max(0,len(event)-1))] if len(event) else np.full(len(origins),len(f))
    resolved=has&(eventrow<ends[origins])&(eventrow-origins<=1800)&(support[origins]>=0)
    known_none=(origins+1800<ends[origins])&(support[origins]>=0)
    y=np.where(known_none,8,-1);y[resolved]=bits[eventrow[resolved]]
    delay=np.where(resolved,eventrow-origins,np.nan)
    return y,delay,dict(line_crossings=len(row),packets=len(event),supported_seconds=int((support>=0).sum()))


def main():
    summary=json.loads((REPORT/'summary.json').read_text())
    source=json.loads((REPORT/'source_receipt.json').read_text())
    f=pd.concat([pd.read_parquet(OUT/(d+'_comparison.parquet')) for d in DATES],ignore_index=True)
    invalid=f.mid<=0
    if (invalid&(f.bb.notna()|f.vwap.notna())).any():raise ValueError('Invalid midpoint entered supported geometry')
    diagnostics={}
    for s in [1,3,6,9]:
        pieces=[];cross=[];counts={'bb':[],'vwap':[]};deltas=[];slopes={str(h):[] for h in [60,300,900]}
        for date in DATES:
            x=f.loc[(f.scale==s)&(f.date==date)].reset_index(drop=True);pieces.append(x)
            y,b,cb=targets(x,'bb');z,v,cv=targets(x,'vwap');good=(y>=0)&(z>=0)
            cross.append(np.c_[y[good],z[good],b[good],v[good]])
            counts['bb'].append(cb);counts['vwap'].append(cv)
            ok=np.isfinite(x.bb)&np.isfinite(x.vwap)&np.isfinite(x.vwap_mid)&np.isfinite(x.vwma)
            deltas.append(np.c_[(x.vwma-x.bb)[ok],(x.vwap_mid-x.vwma)[ok],(x.vwap-x.vwap_mid)[ok]])
            for horizon in [60,300,900]:
                known=ok&ok.shift(horizon,fill_value=False)&(x.segment==x.segment.shift(horizon))
                a=x.bb-x.bb.shift(horizon);b=x.vwap-x.vwap.shift(horizon)
                slopes[str(horizon)].append(np.c_[a[known],b[known]])
        data=np.concatenate(cross);both=np.isfinite(data[:,2])&np.isfinite(data[:,3]);decomp=np.concatenate(deltas)
        diagnostics[str(s)]=dict(common_origins=len(data),first_line_agreement=float(np.mean(data[:,0]==data[:,1])),
            both_have_crossing=int(both.sum()),first_time_exact_on_1s_grid=float(np.mean(data[both,2]==data[both,3])),
            first_delay_abs_median_s=float(np.median(abs(data[both,2]-data[both,3]))),
            sample_counts={name:{k:sum(a[k] for a in arr) for k in arr[0]} for name,arr in counts.items()},
            decomposition_median_abs_usd=dict(volume_weighting_of_closes=float(np.median(abs(decomp[:,0]))),
                within_bar_sampling=float(np.median(abs(decomp[:,1]))),trade_vs_publication_mid=float(np.median(abs(decomp[:,2])))),
            slopes={})
        for horizon,arr in slopes.items():
            a=np.concatenate(arr);active=(abs(a[:,0])>.1)&(abs(a[:,1])>.1)
            signs=np.where(abs(a)>.1,np.sign(a),0).astype(int)
            confusion=np.zeros((3,3),int);np.add.at(confusion,(signs[:,0]+1,signs[:,1]+1),1)
            diagnostics[str(s)]['slopes'][horizon]=dict(pairs=len(a),both_active=int(active.sum()),
                opposite_fraction=float(np.mean(np.sign(a[active,0])!=np.sign(a[active,1]))),
                correlation=float(np.corrcoef(a.T)[0,1]),down_flat_up_confusion=confusion.tolist())
    save(REPORT/'comparison_diagnostics.json',dict(status='PASS',by_scale=diagnostics,
        invalid_midpoint_rows=int(invalid.sum()),invalid_midpoints_in_geometry_or_metrics=0,
        crossing_scope='Strict reversals observed on 1-second grid; 30-second origins, 30-minute future cap, common mature support, ties retained. Not full-publication crossing counts.',
        script_sha256=sha(__file__)))
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.18})
    images=[]
    def finish(fig,name):
        fig.savefig(REPORT/name,dpi=150,bbox_inches='tight');plt.close(fig);images.append(name)
    def times(frame):return pd.to_datetime(frame.t_ns,utc=True)
    def clock(ax):ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'));ax.set_xlabel('UTC')
    fig,axes=plt.subplots(2,5,figsize=(22,9),layout='constrained')
    for col,date in enumerate(DATES):
        for row,s in enumerate([1,9]):
            x=f.loc[(f.date==date)&(f.scale==s)].iloc[::15];ax=axes[row,col]
            ax.plot(times(x),x.mid.where(x.bb.notna()),color='#a5a9ae',lw=.6,label='Midpoint (BB-ready)')
            for name in ['bb','vwap']:ax.plot(times(x),x[name],color=COL[name],lw=1,label=LABEL[name])
            ax.set_title(f'{date} / M{s}');clock(ax)
            if col==0:ax.set_ylabel('Price, USD');ax.legend(fontsize=8)
    fig.suptitle('Fixed full week: calendar-equivalent developing BB vs trade VWAP\n15s display sampling; calculations at 1s; gaps are unsupported history',fontsize=15)
    finish(fig,'01_week_centers.png')
    fig,axes=plt.subplots(4,1,figsize=(16,15),layout='constrained')
    for ax,s in zip(axes,[1,3,6,9]):
        x=f.loc[(f.date=='2026-02-04')&(f.scale==s)].iloc[::5];t=times(x)
        ax.plot(t,x.mid.where(x.bb.notna()),color='#7c838b',lw=.6,label='Midpoint (BB-ready)')
        for name in ['bb','vwap']:
            ax.plot(t,x[name],color=COL[name],lw=1.2,label=LABEL[name])
            ax.fill_between(t,x[name]-x[name+'_half'],x[name]+x[name+'_half'],color=COL[name],alpha=.11)
            ax.plot(t,x[name]-x[name+'_half'],color=COL[name],lw=.45,alpha=.7)
            ax.plot(t,x[name]+x[name+'_half'],color=COL[name],lw=.45,alpha=.7)
        ax.set_title(f'M{s} / same native support: 19 completed {s}m bars + developing bar');ax.set_ylabel('Price, USD');clock(ax);ax.legend(loc='upper left',ncol=3)
    fig.suptitle('2026-02-04, fixed Wednesday: center and +/-2 price-dispersion bands\nBB: sample SD of 20 closes; VWAP: population SD of every T price, volume weighted',fontsize=15)
    finish(fig,'02_wednesday_bands.png')
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained');scales=[1,3,6,9];xx=np.arange(4)
    for n,name in enumerate(['vwma','vwap_mid','vwap']):
        r=[summary['results'][str(s)][name] for s in scales]
        for ax,key,title in [(axes[0,0],'center_abs_median_usd','Median absolute center displacement from BB, USD'),
                              (axes[0,1],'half_width_ratio_median','Median band half-width / BB half-width'),
                              (axes[1,0],'line_total_variation_ratio','Central-line total variation / BB (1-second steps)'),
                              (axes[1,1],'age_seconds_median','Median weighted age of input prices, minutes')]:
            ax.bar(xx+(n-1)*.25,[a[key]/(60 if key=='age_seconds_median' else 1) for a in r],width=.24,color=COL[name],label=LABEL[name]);ax.set_title(title);ax.set_xticks(xx,[f'M{s}' for s in scales])
    axes[0,1].axhline(1,color='black',ls='--');axes[1,0].axhline(1,color='black',ls='--')
    axes[1,1].plot(xx,[summary['results'][str(s)]['vwap']['bb_age_seconds_median']/60 for s in scales],color=COL['bb'],marker='o',label='BB mean age')
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.suptitle('Mechanism controls on common supported observations\nSmoother central-line motion does not establish predictability or economic value',fontsize=15)
    finish(fig,'03_mechanism_metrics.png')
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    for s in scales:
        a=summary['scale_matching'][str(s)];err=a['calibration_rmse'];factors=[int(k.split('_')[1])/(s*1200) for k in err]
        axes[0].plot(factors,list(err.values()),marker='o',label=f'M{s} / {20*s}min nominal')
    axes[0].set(xlabel='Rolling VWAP window / nominal calendar match',ylabel='Center RMSE to native BB, USD',title='Feb 2-3: same support for all candidate windows');axes[0].legend()
    for i,h in enumerate([60,300,900]):
        vals=[100*diagnostics[str(s)]['slopes'][str(h)]['opposite_fraction'] for s in scales]
        axes[1].plot(scales,vals,marker='o',label=f'{h//60}min line displacement')
    axes[1].set(xticks=scales,xlabel='Native scale, minutes',ylabel='Opposite direction, %',title='Both line moves > $0.10; horizon-specific populations');axes[1].legend()
    fig.suptitle('Scale correspondence and changes in line direction\nWindow selected on Mon-Tue only; Wed-Fri RMSE saved separately',fontsize=14)
    finish(fig,'04_scales_and_direction.png')
    fig,axes=plt.subplots(3,1,figsize=(16,11),layout='constrained',sharex=True)
    x=f.loc[(f.date=='2026-02-04')&(f.scale==3)].iloc[::5];t=times(x)
    axes[0].plot(t,x.mid.where(x.bb.notna()),color='#acb1b8',lw=.6,label='Midpoint (BB-ready)')
    for name in ['bb','vwma','vwap']:axes[0].plot(t,x[name],color=COL[name],lw=1.1,label=LABEL[name])
    axes[0].legend(ncol=4);axes[0].set_ylabel('Price, USD')
    for values,label,color in [(x.vwma-x.bb,'Volume weighting of same closes',COL['vwma']),
        (x.vwap_mid-x.vwma,'Within-bar price sampling',COL['vwap_mid']),(x.vwap-x.vwap_mid,'Trade price minus simultaneous midpoint',COL['vwap'])]:
        axes[1].plot(t,values,label=label,color=color,lw=1)
    axes[1].axhline(0,color='black',lw=.6);axes[1].legend();axes[1].set_ylabel('Center difference, USD')
    tr=pd.read_parquet(OUT/'2026-02-04_trades.parquet');tr=tr.loc[(tr.available_ns>=x.t_ns.min())&(tr.available_ns<=x.t_ns.max())]
    vol=tr.groupby(tr.available_ns//(60*10**9)).quantity.sum()
    axes[2].bar(pd.to_datetime(vol.index*60*10**9,utc=True),vol,width=1/1440,color='#6e8295');axes[2].set_ylabel('T contracts / minute');clock(axes[2])
    fig.suptitle('Fixed Wednesday, M3: additive decomposition of central-line displacement\nSigned components sum to VWAP minus BB; absolute medians do not add',fontsize=14)
    finish(fig,'05_volume_decomposition.png')
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for i,name in enumerate(['bb','vwap']):
        rates=[3600*diagnostics[str(s)]['sample_counts'][name]['line_crossings']/diagnostics[str(s)]['sample_counts'][name]['supported_seconds'] for s in scales]
        axes[0].bar(xx+(i-.5)*.32,rates,width=.3,label=LABEL[name],color=COL[name])
    axes[0].set(xticks=xx,xticklabels=[f'M{s}' for s in scales],ylabel='Observed crossings / supported hour',title='All three lines, 1-second sampled strict reversals');axes[0].legend()
    axes[1].bar(xx,[100*diagnostics[str(s)]['first_line_agreement'] for s in scales],color='#667995')
    axes[1].set(xticks=xx,xticklabels=[f'M{s}' for s in scales],ylim=(0,100),ylabel='Agreement, %',title='Next-line label agreement on common mature roots')
    for i,s in enumerate(scales):axes[1].text(i,3,f"n={diagnostics[str(s)]['common_origins']:,}",ha='center',color='white',fontsize=9)
    fig.suptitle('Crossing diagnostics at 1-second resolution\n30-second origins; <=30-minute future; ties and NONE retained; micro-recrosses are omitted',fontsize=14)
    finish(fig,'06_sampled_crossings.png')
    rows='';matchrows='';trends=''
    for s in scales:
        a=summary['results'][str(s)]['vwap'];m=summary['scale_matching'][str(s)];diag=diagnostics[str(s)]
        rows+=f'<tr><td>M{s}</td><td>{20*s} мин</td><td>{a["rows"]:,}</td><td>{a["center_abs_median_usd"]:.2f}</td><td>{a["center_abs_p90_usd"]:.2f}</td><td>{a["half_width_ratio_median"]:.3f}</td><td>{a["line_total_variation_ratio"]:.3f}</td><td>{a["price_side_disagreement"]:.1%}</td></tr>'
        matchrows+=f'<tr><td>M{s}</td><td>{m["selected_minutes"]} мин</td><td>{m["heldout_selected_rmse"]:.3f}</td><td>{m["common_evaluation_rows"]:,}</td></tr>'
        trends+=f'<tr><td>M{s}</td>'+''.join(f'<td>{diag["slopes"][str(h)]["opposite_fraction"]:.1%} / {diag["slopes"][str(h)]["both_active"]:,}</td>' for h in [60,300,900])+'</tr>'
    captions=['Центральные линии за всю выбранную неделю. Midpoint показан только при готовой BB-геометрии; пробелы включают прогрев после разрыва поддержки.',
        'Фиксированная среда: полосы разных масштабов. Все масштабы показаны на одном интервале.',
        'Контроли отделяют эффект весов объёма, распределения цены внутри свечи и цены самой сделки.',
        'Подбор длительности на первых двух днях и направление изменения средней линии на разных горизонтах.',
        'Вклад трёх механизмов в смещение средней линии; объём показан отдельно.',
        'Пересечения на секундной сетке. Быстрые возвраты между отсчётами не видны.']
    page=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VWAP bands vs Bollinger — неделя</title>
<style>body{{font:16px/1.6 system-ui;background:#f2f5f7;color:#172b3c;max-width:1500px;margin:auto;padding:28px}}h1{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #d4dce2;text-align:left}}figure,.card{{background:white;padding:22px;border:1px solid #d4dce2;border-radius:10px;margin:25px 0}}img{{width:100%;height:auto}}a{{color:#176e9b}}.scroll{{overflow:auto}}figcaption{{color:#526476}}</style>
<h1>VWAP и полосы: сравнение с Bollinger</h1><p>2–6 февраля 2026 · GCJ6 · пять полных торговых дней · {sum(r['rows'] for r in source['captures']):,} исходных T за неделю с подготовительной историей.</p>
<div class="card"><b>Главный результат.</b> Соответствия M1/M3/M6/M9 → 20/60/120/180 минут подтвердились среди шести проверенных длительностей. VWAP сохраняет общий масштаб средней линии, но меняет её положение и краткосрочную реакцию. В этом опыте линия VWAP двигается существенно плавнее, а её входные цены в среднем старше. Торговые результаты из этого не следуют.</div>
<p>BB = 20 native midpoint closes, включая формирующуюся цену, ±2 sample SD (ddof=1). Основной VWAP = Σ(цена T × объём T)/Σобъём T; полосы ±2√[Σобъём T × (цена T − VWAP)² / Σобъём T]. Это взвешенное распределение цен, не ошибка оценки среднего. Окно основного сравнения совпадает по времени: 19 завершённых свечей и текущая часть свечи. Роллинг VWAP на ровно 20/60/120/180 минут сохранён отдельно.</p>
<p>Данные доступны только после полной публикации. T с неизвестным агрессором включены; F/C не дублируют объём. Исходный месячный файл и шесть суточных частей перепроверены по SHA256; суммы T совпали с независимыми накопительными счётчиками. При разрывах и смене поддержки окно прогревается заново. Точного broker parity с cTrader не заявляется.</p>
<h2>Насколько меняется геометрия</h2><div class="card scroll"><table><tr><th>BB</th><th>VWAP, календарный аналог</th><th>Общих секунд</th><th>|Δ центра| медиана, $</th><th>|Δ центра| p90, $</th><th>Ширина / BB</th><th>Движение линии / BB</th><th>Цена по другую сторону средней</th></tr>{rows}</table></div>
<p>Движение линии — сумма абсолютных секундных изменений на общей непрерывной поддержке. Разница определений дисперсии тоже сохранена: BB sample SD на 2,60% больше собственного population SD. Ширины не подгонялись. Наблюдения перекрываются; число секунд не равно числу независимых примеров.</p>
<h2>Соответствующие масштабы</h2><p>Проверены множители 0,5 / 0,75 / 1 / 1,25 / 1,5 / 2 к календарному окну. Минимум RMSE средней линии выбирался на 2–3 февраля, проверка выполнена на 4–6 февраля. У всех кандидатов одна общая поддержка. Это подбор геометрического соответствия в данной сетке, не универсальная оптимизация.</p><div class="card"><table><tr><th>BB</th><th>Выбранный rolling VWAP</th><th>RMSE центра 4–6 февраля, $</th><th>Секунд проверки</th></tr>{matchrows}</table></div>
<h2>Меняется ли направление линии</h2><p>Доля противоположных знаков изменения, только когда обе линии сдвинулись больше чем на один тик ($0,10). После процента указано число таких пар. Для каждого горизонта выборка своя; изменение средней за минуту не тождественно направлению многочасового тренда.</p><div class="card"><table><tr><th>Масштаб</th><th>За 1 минуту / n</th><th>За 5 минут / n</th><th>За 15 минут / n</th></tr>{trends}</table></div>
<p>VWMA на тех же closes изолирует изменение весов. VWAP midpoint в моменты T добавляет внутрисвечную траекторию, а точный VWAP заменяет midpoint ценой сделки. Эти signed различия складываются; медианы абсолютных различий складывать нельзя.</p>
'''+''.join(f'<figure><a href="{name}"><img src="{name}" alt="{caption}" loading="lazy"></a><figcaption>{caption}</figcaption></figure>' for name,caption in zip(images,captions))+'''
<h2>Проверки и материалы</h2><p><a href="summary.json">Измерения и соответствие масштабов</a> · <a href="source_receipt.json">Исходные сделки и сверка объёмов</a> · <a href="direct_checks.json">400 прямых проверок VWAP и причинности</a> · <a href="comparison_diagnostics.json">Направления и секундные пересечения</a> · <a href="SOL61_VISUAL_AUDIT_RU.md">Аудит Sol 6.1</a></p><p>Опыт описывает одну неделю и геометрию линий. Он не устанавливает превосходство торговой стратегии. Результаты прежнего BB-опыта сохранены отдельно.</p></html>'''
    (REPORT/'index.html').write_text(page,encoding='utf-8',newline='\n')
    save(REPORT/'visual_receipt.json',dict(images={name:sha(REPORT/name) for name in images},summary_sha256=sha(REPORT/'summary.json'),
        diagnostics_sha256=sha(REPORT/'comparison_diagnostics.json'),page_sha256=sha(REPORT/'index.html'),script_sha256=sha(__file__)))
    print(json.dumps(dict(images=len(images),crossings=diagnostics)),flush=True)


if __name__=='__main__':main()
