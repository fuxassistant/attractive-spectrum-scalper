"""Inspectable native paths, decomposed scores, book masks and parameter tradeoffs."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.scalper_view_audit_v1.audit import dump
from .run import ROOT,HERE,read

METHODS=['legacy','rhythm_gate','book_gate']
LABELS={'legacy':'Прежняя формула','geometry':'Отдельные геометрические меры',
        'rhythm_soft':'Ритм с отдельным весом','rhythm_gate':'Форма + границы ритма',
        'book_gate':'Форма + ритм + книга'}
SHORT_BOOK=['Баланс: низ','Баланс: верх','Неизв. T','Активность T',
            'Fill bid','Fill ask','Снятие bid','Снятие ask','Refill bid','Refill ask','Add bid','Add ask','Ответ цены / A']


def trace(folder,case,name):
    stem=case['case_id']+'_'+name
    tokens=pd.read_parquet(folder/(stem+'_tokens.parquet'))
    q=pd.read_parquet(folder/(stem+'_quotes.parquet'))
    start=int(tokens.start_ns.iloc[0]);end=int(tokens.available_ns.iloc[-1])
    anchor=int(tokens.start_mid_nano.iloc[0]);amp=float(tokens.amplitude_ticks.iloc[0])*100_000_000
    p=(q.bid_nano.to_numpy(np.int64)+q.ask_nano.to_numpy(np.int64))//2
    t=q.available_ns.to_numpy(np.int64)
    # Exact held BBO at the observation boundary, without any future publication.
    x=np.r_[(t-start)/1e9,(end-start)/1e9]
    y=np.r_[(p-anchor)/amp,(p[-1]-anchor)/amp]
    boundaries=(tokens.start_ns.to_numpy(np.int64)[1:]-start)/1e9
    return x,y,boundaries,tokens


def atlas(folder,cases,clock):
    group=[c for c in cases if c['query_ns']==clock]
    group=sorted(group,key=lambda c:['fast','middle','slow'].index(c['scale']))
    fig,axes=plt.subplots(3,3,figsize=(17,10),squeeze=False)
    for row,case in enumerate(group):
        qx,qy,qb,qt=trace(folder,case,'query')
        traces={m:trace(folder,case,m) for m in METHODS if case['methods'][m] is not None}
        max_time=max([qx[-1]]+[v[0][-1] for v in traces.values()])
        all_prices=np.concatenate([qy]+[v[1] for v in traces.values()])
        lo=float(all_prices.min());hi=float(all_prices.max());padding=max(.12,(hi-lo)*.1)
        for col,method in enumerate(METHODS):
            ax=axes[row,col]
            ax.step(qx,qy,where='post',color='#126da0',lw=1.05,label=f'Запрос · {qx[-1]:.1f} с',rasterized=True)
            for b in qb:ax.axvline(b,color='#126da0',ls=':',lw=.7,alpha=.6)
            if method in traces:
                x,y,boundaries,tokens=traces[method]
                ax.step(x,y,where='post',color='#ae5d16',lw=.85,alpha=.9,label=f'Аналог · {x[-1]:.1f} с',rasterized=True)
                for b in boundaries:ax.axvline(b,color='#ae5d16',ls='--',lw=.65,alpha=.5)
                match=case['methods'][method]
                message=f'Время ×{match["total_duration_factor"]:.2f}  ·  форма {match["shape_rmse"]:.2f}'
                message+=f'  ·  score {match["score"]:.3f}'
                ax.text(.02,.025,message,transform=ax.transAxes,fontsize=8,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            else:
                status=case['statuses'][method]
                reason=('Недостаточно книги у запроса' if status['status']=='QUERY_BOOK_CONTEXT_INCOMPLETE'
                        else 'Не совпали время/направление/данные' if status['context_eligible']==0 else 'Ближайший не прошёл допуски')
                msg='Аналог не выдан\n'+reason
                ax.text(.5,.5,msg,ha='center',va='center',transform=ax.transAxes,fontsize=10,
                        bbox=dict(facecolor='#fff3da',edgecolor='none',alpha=.95))
            ax.set_xlim(0,max_time);ax.set_ylim(lo-padding,hi+padding)
            ax.grid(alpha=.15);ax.tick_params(labelsize=8)
            ax.set_title(f'{LABELS[method]} · {case["scale"]}',fontsize=11)
            ax.set_xlabel('Физическое время от начала тройки, с',fontsize=9)
            if col==0:ax.set_ylabel('Изменение цены / A',fontsize=10)
            ax.legend(loc='upper left',fontsize=7,framealpha=.8)
    stamp=pd.Timestamp(clock,tz='UTC').strftime('%d.%m.%Y %H:%M:%S UTC')
    fig.suptitle('Один запрос — три определения похожести\n'+stamp,fontsize=15)
    fig.text(.5,.013,'В пределах строки оси одинаковы. Вертикали — границы трёх отрезков. Все аналоги из предыдущей торговой сессии; будущего продолжения нет.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=[.01,.035,1,.94])
    name=pd.Timestamp(clock,tz='UTC').strftime('atlas_%H%M%S.png')
    fig.savefig(folder/name,dpi=150);plt.close(fig)
    return name


def mechanism_figure(folder,cases,clock):
    group=sorted([c for c in cases if c['query_ns']==clock],key=lambda c:['fast','middle','slow'].index(c['scale']))
    fig,axes=plt.subplots(3,1,figsize=(16,9))
    cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#dfe3e7')
    for ax,case in zip(axes,group):
        arrays=[];names=[]
        for method in ['query']+METHODS:
            name=case['case_id']+'_'+method+'_features.npz'
            if (folder/name).exists():
                f=np.load(folder/name);v=f['book'][-1].astype(float)
                v[~f['book_known'][-1]]=np.nan
                arrays.append(v)
            else:arrays.append(np.full(13,np.nan))
            names.append('Запрос' if method=='query' else LABELS[method]+(' · нет аналога' if case['methods'][method] is None else ''))
        a=np.asarray(arrays);ax.imshow(np.ma.masked_invalid(a),aspect='auto',cmap=cmap,norm=TwoSlopeNorm(vmin=-1.5,vcenter=0,vmax=1.5))
        ax.set_yticks(range(4),names,fontsize=9)
        ax.set_xticks(range(13),SHORT_BOOK,fontsize=8,rotation=20,ha='right')
        ax.set_title(f'{case["scale"]} · контекст последнего, формирующегося токена',fontsize=11)
        for i in range(4):
            for j in range(13):
                if np.isfinite(a[i,j]):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=7,color='#20232a')
    fig.suptitle('Цена может быть похожа при разном состоянии книги',fontsize=15)
    fig.text(.5,.012,'Серое — неизвестное или отсутствие аналога. Величины преобразованы согласно контракту; столбцы имеют разные физические смыслы. T — только наблюдавшиеся сделки.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.96])
    fig.savefig(folder/'mechanism_120000.png',dpi=150);plt.close(fig)
    return 'mechanism_120000.png'


def sensitivity_figure(folder,summary,config):
    table=pd.DataFrame(summary['parameter_sensitivity'])
    weights=config['sensitivity']['rhythm_weights'];factors=config['sensitivity']['total_duration_factors']
    arrays=[]
    for field in ('with_neighbor','median_total_duration_factor','median_shape_rmse'):
        a=np.full((len(weights),len(factors)),np.nan)
        for i,w in enumerate(weights):
            for j,f in enumerate(factors):
                g=table[table.method==f'time_w{w}_f{f}'].iloc[0]
                a[i,j]=g[field]/g.queries if field=='with_neighbor' else g[field]
        arrays.append(a)
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    for ax,a,title in zip(axes,arrays,['Доля запросов с аналогом','Медианный множитель длительности','Медианная ошибка формы']):
        ax.imshow(a,aspect='auto',cmap='YlGnBu')
        ax.set_xticks(range(len(factors)),[str(f) if f else 'без границы' for f in factors],fontsize=9)
        ax.set_yticks(range(len(weights)),[str(w) for w in weights],fontsize=9)
        ax.set_title(title,fontsize=11);ax.set_xlabel('Максимальный множитель времени');ax.set_ylabel('Вес ритма')
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',bbox=dict(facecolor='white',alpha=.75,edgecolor='none'))
    fig.suptitle('Параметры меняют строгость поиска и состав найденных примеров',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(folder/'parameter_sensitivity.png',dpi=150);plt.close(fig)
    return 'parameter_sensitivity.png'


def main(run):
    folder=ROOT/run;m=read(folder/'manifest.json');summary=read(folder/'summary.json');config=read(folder/'config.json')
    if (folder/'report_receipt.json').exists():raise FileExistsError('Sealed report already exists')
    if m['status']!='COMPLETE_DESCRIPTIVE_SIMILARITY':raise ValueError('Complete comparison required')
    for path,digest in m['artifacts'].items():
        if sha(folder/path)!=digest:raise ValueError('Artifact drift: '+path)
    cases=read(folder/'atlas_index.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    figures=[atlas(folder,cases,t) for t in sorted({c['query_ns'] for c in cases})]
    noon=int(pd.Timestamp(config['query_session']+'T12:00:00Z').value)
    figures.append(mechanism_figure(folder,cases,noon));figures.append(sensitivity_figure(folder,summary,config))
    rows=''
    for r in summary['methods']:
        if r['scope']!='previous':continue
        f=lambda x:'—' if x is None else f'{x:.3f}'
        rows+=f'<tr><td>{LABELS[r["method"]]}</td><td>{r["with_neighbor"]} / {r["queries"]}</td><td>{r["no_analogue"]}</td><td>{f(r["median_total_duration_factor"])}</td><td>{f(r["max_total_duration_factor"])}</td><td>{f(r["median_shape_rmse"])}</td><td>{f(r["median_mechanism_rmse"])}</td></tr>'
    details=''
    for case in cases:
        for method in METHODS:
            match=case['methods'][method];status=case['statuses'][method]
            values=' · '.join(f'{k}: {match[k]:.3f}' for k in ['shape','amplitude','texture','structure','rhythm','trajectory','mechanism'] if match and match[k] is not None)
            details+=f'<tr><td>{case["case_id"]}</td><td>{LABELS[method]}</td><td>{status["status"]}</td><td>{values or "—"}</td></tr>'
    gallery=''.join(f'<figure><a href="{name}"><img src="{name}" loading="lazy" alt="{html.escape(name)}"></a><figcaption>{name} · нажмите для полного разрешения</figcaption></figure>' for name in figures)
    doc=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Сходство историй — форма, ритм и книга</title>
<style>body{{margin:0;background:#f4f6f8;color:#213149;font:17px/1.6 system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:32px 24px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:40px}}p{{max-width:1100px}}.notice{{background:#fff0d7;padding:18px;border-radius:12px}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce3ec}}.scroll{{overflow:auto}}figure{{background:white;margin:24px 0;padding:8px;border-radius:10px}}img{{width:100%;height:auto}}figcaption{{font-size:13px;color:#596b7c}}a{{color:#086da3}}details{{margin:22px 0}}@media(max-width:600px){{main{{padding:18px 10px}}h1{{font-size:28px}}}}</style>
<main><p>STORY_SIMILARITY_COMPONENTS_V2 · описательный опыт на 24–25 августа 2026</p>
<h1>Что именно делает две истории похожими</h1>
<p>Сравниваются пять заранее заданных определений на одинаковом допустимом прошлом. Каждый запрос — две завершённые истории и доступная часть текущей.
36 состояний получены из трёх прежних контрольных времён и девяти времён, выбранных по хешу без будущих исходов. Поиск по текущей и предыдущей сессиям посчитан отдельно.</p>
<div class="notice">Это развитие измерений и проверка подбора аналогов. Порог «подходит» выражает заранее заданные допуски; его рыночная полезность не откалибрована.
Улучшение величины, входящей в формулу поиска, не доказывает улучшения смысла или торгового результата.</div>
<h2>Функции сходства</h2><p><b>Форма:</b> 16 точек на уже наблюдённом пути, с отдельной нормировкой размаха.
<b>Амплитуда:</b> реальное движение, диапазон и откаты относительно A. <b>Внутренние колебания:</b> эффективность, путь, смены направления и распределение пути по четырём частям времени.
<b>Структура:</b> порядок направлений и причины завершения двух предыдущих историй.
<b>Ритм:</b> логарифмы отношений длительностей каждого отрезка, с отдельным весом.
<b>Целая история:</b> части соединяются с накопленным смещением; крупное противоположное направление не может исчезнуть при нормировке отдельных частей.
<b>Книга:</b> доступный контекст сделок и bid/ask исполнений, снятий, пополнений, добавлений и ответа цены.</p>
<p>В строгом варианте длительность всей тройки отличается не более чем в 1.5 раза, каждой части — не более чем в 2 раза.
При модуле движения ≥0.5A запрещены противоположные направления всей тройки или её значимых частей; проверяется ошибка соединённого пути.
Для книги требуется ≥75% совместно известных полей и ≥60% в каждом из трёх отрезков. Недостающие данные не заменяются хорошим совпадением.
При отсутствии допустимого кандидата возвращается явное «аналога нет».</p>
<h2>Предыдущая сессия: одинаковые 36 запросов</h2><div class="scroll"><table><tr><th>Метод</th><th>С аналогом</th><th>Без аналога</th><th>Время, медиана ×</th><th>Время, максимум ×</th><th>Форма, RMSE</th><th>Книга, RMSE</th></tr>{rows}</table></div>
<p>Часть методов меняет состав запросов с найденными соседями. Медианы в таблице относятся к выданным первым соседям; это не равный по составу тест качества.
Сравнение с текущей сессией и все результаты находятся в JSON/Parquet. Ни один вариант не выбран как победитель по этой уже изученной ленте.</p>
<h2>Визуальные пары</h2><p>Синяя линия — текущая история, коричневая — найденный пример. В пределах строки сохранены одинаковые физические оси, поэтому растяжение времени видно.
Пунктир обозначает границы трёх частей. После правой границы каждой истории продолжение не показывается и не используется.</p>
{gallery}
<details><summary>Компоненты расстояния для показанных пар</summary><div class="scroll"><table><tr><th>Сцена</th><th>Метод</th><th>Результат</th><th>Состав расстояния</th></tr>{details}</table></div></details>
<h2>Проверки</h2><p>Прежние девять контрольных аналогов воспроизведены: {summary['checks']['legacy_anchor_matches']} / 9.
Проверены {summary['checks']['book_scalar_checks']} контекстных окон книги отдельной скалярной формулой и {summary['checks']['native_shape_samples']} отсчётов пути по native BBO.
Обрезка всей доступной текущей ленты и признаков в 12:00 сохранила все компоненты и подбор для пяти методов в обоих источниках истории.</p>
<p>Синтетические проверки различают перестановку отрезков, изменение ритма при неизменной форме, противоположный поток и отсутствие данных.
Отдельно реализован зеркальный оператор с перестановкой bid/ask и buy/sell; на реальных данных симметрия не предполагается и этот оператор в подборе выключен.</p>
<h2>Что остаётся проверить</h2><p>Человеческое согласие о похожести, чувствительность к исходной нарезке и сетке, границы применимости каждого механизма и добавочная информация для денежного таргета.
Время построения индекса и всего опыта приведено в сводке; это пакетный исследовательский поиск.</p>
<p>Предыдущая версия verified_v2 прошла арифметику, но визуально ошиблась в медленной сцене 12:00: локальное сходство частей скрывало противоположный общий ход.
После её просмотра добавлены целая траектория и ограничения направления. Это открытая правка по исследованному примеру, а не независимая проверка обобщения.</p>
<p><a href="summary.json">Все численные результаты</a> · <a href="matches.json">Соседи и компоненты</a> · <a href="config.json">Параметры и допуски</a> · <a href="manifest.json">Источники и хеши</a></p></main></html>'''
    (folder/'report.html').write_text(doc,encoding='utf-8',newline='\n')
    dump(folder/'report_receipt.json',dict(status='RENDERED_STATIC_COMPARISON',source_sha256=sha(__file__),
        input_manifest_sha256=sha(folder/'manifest.json'),artifacts={n:sha(folder/n) for n in figures+['report.html']},
        browser_rendered=False,visual_review='separate receipt'))
    print(json.dumps(dict(report=str(folder/'report.html'),figures=figures)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=read(HERE/'config.json')['output']);main(p.parse_args().run)
