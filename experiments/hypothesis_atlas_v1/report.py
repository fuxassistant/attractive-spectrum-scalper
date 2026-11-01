"""Readable before/after market scenes, frozen hypotheses and honest controls."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

from .engine import NS, CLASSES, FAMILIES
from .run import ROOT, HERE, dump, digest, checked

COLORS=['#117f77','#bd553e','#a57b2c']
LABELS=['Возврат к зоне','Удаление на 1R','Ни одной границы за 5 мин']
SHORT=['Возврат','Удаление','Без события']
ARM_NAMES={'geometry':'Геометрия и активность','context':'+ История пути','hypotheses':'+ Память и гипотезы',
           'hot_global':'+ Общее обновление','hot_families':'+ Обновление по гипотезам'}
RULES={
 'memory_return':dict(question='Может ли область, где цена задерживалась и чаще разворачивалась, вернуть цену после нового ухода?',
    rule='Медианное время в зоне не ниже медианы обучающего префикса; доля проходов в последних посещениях ≤ 37,5%.',
    expected='Ближняя граница прежней зоны будет достигнута раньше удаления ещё на 1R, в течение 300 секунд.',
    distinguish='Длительное удержание места отделяется от простого факта прежнего касания.',
    limit='Долгая жизнь уровня не устанавливает, что он продолжает действовать сейчас.'),
 'permeable_departure':dict(question='Продолжится ли уход от области, через которую цена часто проходила насквозь?',
    rule='Не менее 50% последних посещений закончились проходом; последнее тоже было проходом; движение за 30 секунд направлено от зоны или равно нулю.',
    expected='Цена удалится ещё на 1R раньше возврата к зоне, в течение 300 секунд.',
    distinguish='Старая занятость цены может быть следствием транзита, а не удержания.',
    limit='Проходимость в прошлых посещениях не доказывает продолжение именно последнего ухода.'),
 'failed_escape':dict(question='Начался ли возврат после очередной попытки покинуть знакомую область?',
    rule='За последние 30 секунд цена продвинулась к зоне минимум на 0,25R; число отмен попыток ухода на прошлое посещение не ниже обучающей медианы.',
    expected='Возврат к зоне наступит раньше удаления ещё на 1R, за 300 секунд.',
    distinguish='Учитывает текущую реакцию вместе с историей отмен, а не только возраст или силу уровня.',
    limit='Короткая обратная реакция может оказаться откатом внутри продолжающегося ухода.'),
 'place_migration':dict(question='Утрачил ли старый центр пребывания актуальность, несмотря на накопленную память?',
    rule='История длительного пребывания есть; средняя наблюдаемая цена второй половины последних 10 минут сместилась от зоны ≥ 0,5R; доля секунд в зоне упала более чем вдвое с положительного уровня.',
    expected='Дальнейшее удаление на 1R произойдёт раньше возврата, за 300 секунд.',
    distinguish='Отделяет накопленную память места от текущего перемещения ценовой активности.',
    limit='Сдвиг центра — описание уже состоявшегося пути; это ещё не прогноз его продолжения. Доли времени оценены секундной сеткой.'),
 'tight_range':dict(question='Сохранится ли задержка между целевой зоной и противоположной границей?',
    rule='Диапазон последних 120 секунд ≤ 0,75R; абсолютное смещение за них ≤ 0,25R.',
    expected='За следующие 300 секунд не будет достигнута ни целевая граница, ни граница удаления на 1R.',
    distinguish='Даёт проверяемую гипотезу отсутствия события и основание продолжать наблюдение.',
    limit='Прошлое сжатие может закончиться расширением; отсутствие события не означает сохранения той же волатильности.'),
}


def stamp(t):return pd.Timestamp(int(t),tz='UTC').strftime('%d.%m.%Y %H:%M:%S UTC')


def scene(q,r,destination,mobile=False,reveal=False):
    t=int(r['t_ns']);left=t-600*NS;right=t+300*NS
    qt=q.available_ns.to_numpy(np.int64)
    a=max(0,int(np.searchsorted(qt,left,'right'))-1)
    b=int(np.searchsorted(qt,right if reveal else t,'right'))
    z=q.iloc[a:b].copy();x=(z.available_ns.to_numpy(np.int64)-t)/(60*NS)
    p=(z.bid_nano.to_numpy(np.int64)+z.ask_nano.to_numpy(np.int64))/(2*NS)
    valid=(z.valid&z.epoch.ge(0)&z.ask_nano.gt(z.bid_nano)).to_numpy(bool)
    gap=np.r_[False,(np.diff(z.available_ns)>30*NS)|(np.diff(z.epoch)!=0)]
    p[~valid|gap]=np.nan
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13 if mobile else 12,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(4.0,6.8) if mobile else (11.8,6.3),layout='constrained')
    grid=fig.add_gridspec(2,1,height_ratios=[3.4,1.35]);ax=fig.add_subplot(grid[0]);bar=fig.add_subplot(grid[1])
    low=r['zone_low_nano']/NS;high=r['zone_high_nano']/NS
    origin=r['origin_mid_x2_nano']/(2*NS);risk=r['risk_per_oz'];adverse=r['adverse_x2_nano']/(2*NS)
    target=r['target_nano']/NS;_,event=FAMILIES[r['family']]
    ax.axhspan(low,high,color=COLORS[0],alpha=.12,label='Целевая зона')
    ax.axhline(target,color=COLORS[0],lw=1.3,ls='--')
    ax.axhline(adverse,color=COLORS[1],lw=1.3,ls='--',label='Удаление на 1R')
    ax.axvline(0,color='#202e3a',lw=1.4)
    ax.plot(x[x<=0],p[x<=0],color='#253b4a',lw=1.3,label='Наблюдаемая история')
    ax.scatter([0],[origin],color='#202e3a',s=35,zorder=5)
    ax.set_xlim(-10,5);ax.set_xticks([-10,-5,0,5])
    ax.set_xlabel('Минуты от решения' if mobile else 'Минуты относительно решения');ax.set_ylabel('Цена · USD/oz')
    ax.yaxis.set_major_locator(MaxNLocator(5));ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.grid(alpha=.14)
    if reveal:
        future=x>0;first=int(r['first_event_ns']) if r['first_event_ns']>=0 else right
        time=(first-t)/(60*NS);before=future&(x<=time);after=future&(x>=time)
        ax.plot(x[before],p[before],color=COLORS[CLASSES.index(r['state'])],lw=1.7)
        ax.plot(x[after],p[after],color='#77828b',lw=1,alpha=.65,ls=':')
        if r['first_event_ns']>=0:
            j=int(np.searchsorted(qt,first,'right'))-1
            value=(int(q.bid_nano.iloc[j])+int(q.ask_nano.iloc[j]))/(2*NS)
            ax.scatter([time],[value],marker='D',s=48,color=COLORS[CLASSES.index(r['state'])],zorder=6)
        verdict=LABELS[CLASSES.index(r['state'])]
        ax.set_title(('ПОСЛЕ · '+verdict) if not mobile else 'ПОСЛЕ\n'+'\n'.join(textwrap.wrap(verdict,24)),loc='left')
    else:
        # Future bounds do not enter any axis or annotation before reveal.
        ax.axvspan(0,5,color='#e9edf0',alpha=.8)
        ax.text(2.5,.5,'Будущее\nскрыто' if mobile else 'Следующие\n5 минут\nскрыты',ha='center',va='center',transform=ax.get_xaxis_transform(),color='#56646f')
        ax.set_title('ДО · история' if mobile else 'ДО · доступно при решении',loc='left')
    ax.text(.02,.98,f'R = {risk:.2f} USD/oz',va='top',transform=ax.transAxes,color='#52616e',fontsize=12 if mobile else 10,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=2))
    p0=np.array(r['probabilities']['context']);p1=np.array(r['probabilities']['hot_families'])
    positions=np.arange(3);bar.barh(positions-.16,p0,height=.28,color='#c8d1d7',label='Контекст')
    bar.barh(positions+.16,p1,height=.28,color=COLORS,label='Гипотезы + обновление')
    for i,(a0,b0) in enumerate(zip(p0,p1)):
        bar.text(max(a0,b0)+.025,i,f'{a0:.0%} → {b0:.0%}',va='center',fontsize=12 if mobile else 10)
    bar.set_yticks(positions,SHORT);bar.invert_yaxis();bar.set_xlim(0,1);bar.set_xticks([0,.5,1],['0%','50%','100%'])
    bar.set_xlabel('Контекст → гипотезы\nс обновлением' if mobile else 'Контекст → гипотезы + обновление',fontsize=12);bar.grid(axis='x',alpha=.12)
    if mobile:fig.suptitle(stamp(t),fontsize=12)
    else:fig.suptitle(f'{FAMILIES[r["family"]][0]} · {stamp(t)} · GCZ6',fontsize=13)
    fig.savefig(destination,dpi=140);plt.close(fig)


def overview(s,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
    for mobile in (False,True):
        fig,axes=plt.subplots(2,1,figsize=(4.8,8.3) if mobile else (11.8,7.8),layout='constrained')
        for ax,block,label in zip(axes,['evaluation','extension'],['24–28 августа · 557 решений','31 августа – 1 сентября · 223 решения']):
            rows=[r for r in s['metrics'] if r['block']==block]
            values=[r['logloss'] for r in rows]; names=[ARM_NAMES[r['arm']] for r in rows]
            if mobile:names=[n.replace('Геометрия и активность','Геометрия\nи активность').replace('+ Обновление по гипотезам','+ По гипотезам') for n in names]
            ax.barh(range(5),values,color=['#657a8a','#96adb6','#a57b2c','#a9785c','#117f77'],height=.6)
            for i,v in enumerate(values):ax.text(v+.006,i,f'{v:.4f}',va='center',fontsize=11)
            ax.set_yticks(range(5),names);ax.invert_yaxis();ax.set_xlim(0,1.25)
            ax.set_xlabel('Log loss · меньше лучше');ax.set_title(label,loc='left',fontsize=11 if mobile else 13)
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
        fig.suptitle('Добавочная информация пока не подтверждена',fontsize=11 if mobile else 15)
        fig.savefig(out/('overview_mobile.png' if mobile else 'overview.png'),dpi=140);plt.close(fig)


def picture(stem,alt):
    return f'<a href="{stem}.png" aria-label="Открыть график крупнее"><picture><source media="(max-width:620px)" srcset="{stem}_mobile.png"><img src="{stem}.png" alt="{html.escape(alt)}" loading="lazy"></picture></a>'


def example_html(r,index):
    key=r['family'];_,expected=FAMILIES[key];name=f'{key}_{index}'
    active=[FAMILIES[k][0] for k in FAMILIES if r['h_'+k]]
    kind='Пример по сходству наблюдаемых условий' if r['selection']=='prototype' else 'Контрпример · специально выбран по опровержению'
    target=r['target_nano']/NS;opposite=r['adverse_x2_nano']/(2*NS)
    p=r['probabilities']['hot_families'][expected]
    known=[f'Зона {r["zone_low_nano"]/NS:.2f}–{r["zone_high_nano"]/NS:.2f}',
           f'Последние проходы: {r["x_mem_recent_pass_fraction"]:.0%}',
           f'Медианное пребывание: {r["median_dwell_seconds"]:.1f} с',
           f'Движение к зоне / 30 с: {r["past_net30_r"]:+.2f}R',
           f'Сдвиг центра к зоне: {r["past_migration_r"]:+.2f}R',
           f'Диапазон / 120 с: {r["past_range120_r"]:.2f}R']
    outcome=LABELS[CLASSES.index(r['state'])]
    elapsed=(r['first_event_ns']-r['t_ns'])/NS if r['first_event_ns']>=0 else 300.
    truth='Ожидание подтвердилось' if r['state']==CLASSES[expected] else 'Ожидание не подтвердилось'
    history=r.get('history_tokens',[])
    history_rows=''.join(f'<tr><td>{html.escape(h["clock"])}</td><td>{html.escape(h["outcome"])}</td><td>{h["dwell"]:.1f} с</td></tr>' for h in history)
    return f'''<article id="{name}"><div class="eyebrow">{kind}</div><h3>{stamp(r['t_ns'])}</h3>
    <p class="contract">Цель <b>{target:.2f}</b> · противоположная граница <b>{opposite:.2f}</b> · срок <b>300 с</b></p>
    <p>Оценка ожидаемого гипотезой события после обновления: <b>{p:.1%}</b>. Это прогнозная оценка, не вероятность прибыльной сделки.</p>
    {picture(name+'_before','Наблюдаемая история и три вероятности; будущее скрыто')}
    <ul class="facts">{''.join('<li>'+html.escape(v)+'</li>' for v in known)}</ul>
    <p><b>Одновременно применимы:</b> {html.escape('; '.join(active))}. Совпадение нескольких описаний не считается независимым подтверждением.</p>
    <details class="memory"><summary>Последние доступные посещения этой зоны</summary><div class="table-wrap"><table><thead><tr><th>Завершение UTC</th><th>Наблюдаемый исход</th><th>Время внутри</th></tr></thead><tbody>{history_rows}</tbody></table></div></details>
    <details class="reveal"><summary>Показать следующие пять минут</summary><p><b>{truth}.</b> {outcome}; первое событие / срок: {elapsed:.1f} с. Пунктир после первого события показывает дальнейший путь и не меняет исход проверки.</p>
    {picture(name+'_after','Фактическое продолжение и первое достигнутое событие')}</details></article>'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();out=ROOT/args.run
    s=json.loads((out/'summary.json').read_text(encoding='utf-8'));examples=json.loads((out/'examples.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'research_results/zone_conditional_v1/verified_v1/manifest.json').read_text())
    receipts={r['date']:r for r in source['receipts']};receipts_images=[]
    for date in sorted({r['date'] for r in examples}):
        market=Path(s['source_paths'][date]);qpath=market/'quotes.parquet';checked(qpath,s['source_hashes'][str(qpath)])
        q=pd.read_parquet(qpath)
        tokenpath=ROOT/'research_results/zone_episode_lab/verified_v3/days'/date/'history_tokens.parquet'
        tokenhash=checked(tokenpath,receipts[date]['source_hashes'][str(tokenpath)])
        tokens=pd.read_parquet(tokenpath)
        for i,r in enumerate(examples):
            if r['date']!=date:continue
            h=tokens[tokens.candidate_id.eq(r['candidate_id'])].sort_values('position')
            if (h.available_ns>r['t_ns']).any():raise AssertionError('Future visit in displayed memory')
            r['history_tokens']=[dict(clock=pd.Timestamp(int(v.available_ns),tz='UTC').strftime('%H:%M:%S'),outcome=v.outcome,dwell=float(v.core_dwell_seconds)) for v in h.itertuples()]
            stem=f'{r["family"]}_{i}'
            for mobile in (False,True):
                for reveal in (False,True):
                    suffix=('_after' if reveal else '_before')+('_mobile' if mobile else '')+'.png'
                    scene(q,r,out/(stem+suffix),mobile,reveal)
            receipts_images.append(dict(candidate_id=r['candidate_id'],family=r['family'],selection=r['selection'],
                history_tokens_sha256=tokenhash,origin_ns=r['t_ns'],future_price_used_in_before_view=False))
        print(json.dumps(dict(stage='render',date=date)),flush=True)
    overview(s,out)
    sections=[]
    for key,(title,event) in FAMILIES.items():
        rule=RULES[key];stats=[x for x in s['families'] if x['family']==key and x['block'] in ['evaluation','extension']]
        table=''.join(f'<tr><td>{"24–28.08" if x["block"]=="evaluation" else "31.08–01.09"}</td><td>{x["n"]} / {x["days"]}</td><td>{x.get("row_success",0):.1%}</td><td>{x.get("day_success",0):.1%}</td><td>{x.get("context_expected",0):.1%}</td></tr>' for x in stats)
        cards=''.join(example_html(r,i) for i,r in enumerate(examples) if r['family']==key)
        sections.append(f'''<section id="{key}"><div class="eyebrow">Гипотеза {list(FAMILIES).index(key)+1}</div><h2>{title}</h2><p class="lead">{rule['question']}</p>
        <dl><dt>Когда появляется</dt><dd>{rule['rule']}</dd><dt>Что ожидает</dt><dd>{rule['expected']}</dd><dt>Что позволяет различить</dt><dd>{rule['distinguish']}</dd><dt>Граница смысла</dt><dd>{rule['limit']}</dd></dl>
        <p>Любой другой известный первый исход опровергает именно эту проверку. Неполная наблюдаемость остаётся неизвестностью.</p>
        <div class="table-wrap"><table><thead><tr><th>Период</th><th>Случаи / дни</th><th>Доля совпадений</th><th>Средняя по дням</th><th>Ожидание контекста*</th></tr></thead><tbody>{table}</tbody></table></div>
        <p class="small">* Прогноз контекстной модели на тех же применимых случаях, с равным весом дней. Это сравнение с моделью, не причинный эффект. Доля совпадений не является точностью торговой политики.</p>{cards}</section>''')
    navigation=''.join(f'<a href="#{key}">{i+1}. {title}</a>' for i,(key,(title,_)) in enumerate(FAMILIES.items()))
    report=f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Гипотезы о ценовом месте · GCZ6</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f4f2ed;color:#263741;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:48px 32px 96px}}h1{{font-size:clamp(30px,5vw,54px);line-height:1.08;max-width:850px;font-weight:600}}h2{{font-size:32px;line-height:1.2;margin:8px 0 18px}}h3{{font-size:21px;margin:8px 0}}.eyebrow{{text-transform:uppercase;letter-spacing:.12em;font-size:12px;color:#60717a}}.lead{{font-size:21px;max-width:850px}}nav{{display:flex;gap:12px 24px;flex-wrap:wrap;margin:30px 0}}a{{color:#136f68;text-underline-offset:4px}}section{{border-top:1px solid #cbd3d5;padding:42px 0;margin-top:32px}}article{{background:#fff;padding:28px;margin:30px 0 0;border:1px solid #d8dddf;border-radius:8px}}img{{width:100%;display:block}}.facts{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px 22px;padding-left:20px}}dl{{display:grid;grid-template-columns:180px 1fr;gap:12px 24px}}dt{{font-weight:600}}dd{{margin:0}}.small{{font-size:13px;color:#5c6a73}}summary{{cursor:pointer;padding:14px 0;color:#116d66;font-weight:600}}details{{border-top:1px solid #d7e0e2;margin-top:14px}}.reveal summary{{font-size:18px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #d8dddf}}th{{font-size:13px;color:#526771}}.table-wrap{{overflow-x:auto}}.scope{{border-left:3px solid #a57b2c;padding-left:20px}}.contract{{font-variant-numeric:tabular-nums}}@media(max-width:620px){{main{{padding:28px 16px 64px}}article{{padding:16px 10px;margin-top:22px}}h2{{font-size:26px}}.lead{{font-size:18px}}dl{{display:block}}dd{{margin:4px 0 18px}}.facts{{grid-template-columns:1fr;font-size:14px}}nav{{display:grid;gap:10px}}td,th{{padding:8px;font-size:12px}}}}
    </style></head><body><main><div class="eyebrow">Исследовательский атлас · GCZ6 · 22 дня</div><h1>Что мы вправе предположить о следующем движении?</h1><p class="lead">Пять гипотез о ценовом месте, реальные примеры и возможность открыть будущее после просмотра доступной истории.</p>
    <p>Из 10 865 исходных предложений выбрано 2 456 решений: ближайшая зона в момент наблюдения, затем пауза минимум 300 секунд. Прогнозы охватывают возврат, дальнейшее удаление и отсутствие обоих событий. У каждой проверки заморожены цена, противоположная граница и срок.</p>
    <nav>{navigation}</nav><div class="scope"><p><b>Смысл результата.</b> Эти правила выделяют различимые ситуации. Добавление памяти и гипотез пока ухудшило общий прогноз на обоих поздних блоках. Обновление по завершившимся эпизодам это не исправило. Примеры ниже показывают и содержательность описания, и его ошибки.</p></div>
    {picture('overview','Сравнение вероятностной ошибки пяти моделей на двух последующих блоках')}
    <p>Все модели обучены только на 3–14 августа. Фиксированы C=0,1, шаблоны, признаки и пороги. Обновление начинается 17 августа; исход поступает строго после общего пятиминутного срока. Память обновления затухает с периодом полураспада два часа. Состояние переносится по времени, без будущих исходов и без подбора победителя.</p>
    <p>«Геометрия и активность» использует прежние базовые признаки. «История пути» добавляет измерения последних 10 минут. «Память и гипотезы» добавляет четыре описания посещений и пять флагов применимости. Последние две версии корректируют вероятности по уже завершённым проверкам: общим либо также относящимся к активным гипотезам.</p>
    <p class="small">Первые два примера каждого семейства выбраны по близости наблюдаемых X к типичному состоянию, без исходов. Отдельный контрпример намеренно выбран по неудаче и служит объяснению, а не оценке частоты. На графиках «ДО» будущая цена не влияет даже на масштаб вертикальной оси.</p>
    {''.join(sections)}<section><h2>Что этот опыт позволяет получить</h2><p>Проверяемый словарь локальных предположений; различение памяти места и текущего ухода; явные конкурирующие ожидания; измеримую задержку обратной связи и механизм пересмотра вероятностей. Эти объекты можно использовать для поиска лучших определений гипотез и для последующего обучения генератора.</p><p>Сейчас 1 302 из 2 456 решений не покрыты ни одним правилом; 156 покрыты несколькими. Отсутствие гипотезы сохраняется, совпадение нескольких не превращается в умножение уверенности.</p><p>Популяция уже требует завершённых посещений зоны и решения снаружи. Первый контакт, поведение внутри области и заново обнаруженные уровни этим опытом не проверены. Память посещений здесь внутридневная; межсессионное обучение свойствам абсолютного места ещё не реализовано.</p><p>Секундные профили — приближение времени по цене; все нативные публикации проверены на разрывы, инвалидность и смену эпох. Ожидания геометрические. Исполнение, комиссия, последовательный капитал и реакция рынка на наши заявки в этом опыте не оценивались. Даты ранее исследовались, это не нетронутый holdout.</p><p><a href="summary.json">Все численные результаты</a> · <a href="config.json">Зафиксированный протокол</a> · <a href="examples.json">Исходные карточки</a></p></section></main></body></html>'''
    (out/'report.html').write_text(report,encoding='utf-8')
    dump(out/'display_examples.json',examples)
    dump(out/'render_receipt.json',dict(status='RENDERED_AWAITING_VISUAL_REVIEW',scenes=receipts_images,
         report_sha256=digest(out/'report.html'),images={p.name:digest(p) for p in out.glob('*.png')},
         implementation_sha256={str(p.relative_to(ROOT)):digest(p) for p in HERE.glob('*.py')}))
    print(json.dumps(dict(stage='report_complete',examples=len(examples),images=len(list(out.glob('*.png'))))),flush=True)


if __name__=='__main__':main()
