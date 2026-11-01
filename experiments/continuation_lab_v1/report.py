"""Inspect confirmed causal phrases; outcome region is explicitly separate."""
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

from spectrum.io import sha
from experiments.mechanical_lab.data import dump
from experiments.regime_language_v1.measures import TimeAtPrice

NS=1_000_000_000
ROOT=Path(__file__).resolve().parents[2]
NAMES={'pullback':'Импульс → откат → возобновление',
       'break_retest':'Выход из области → первый ретест → возобновление',
       'repeat_bounce':'Первый отскок → уход → возврат → повторение'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dt(t):return pd.to_datetime(t,utc=True)
def signed(value,unit,digits=3):return f'{value:+.{digits}f}{unit}' if np.isfinite(value) else 'неизвестно'


def scene(root,row,meta):
    q=pd.read_parquet(root/meta['path']);t=q.available_ns.to_numpy(np.int64)
    mid=(q.bid_nano.to_numpy(np.int64)+q.ask_nano.to_numpy(np.int64))/(2*NS)
    context=pd.read_parquet(root/'days'/row.date/'context.parquet')
    start=int(row.start_ns);decision=int(row.t_ns);end=min(int(t[-1]),decision+125*NS)
    mapper=TimeAtPrice(q)
    profile=mapper.profile(start,decision)
    future=pd.read_parquet(root/'days'/row.date/'outcomes.parquet')
    o=future[(future.origin_index==row.origin_index)&(future.side==row.side)&(future.scenario=='base')].iloc[0]
    fig=plt.figure(figsize=(16,10),layout='constrained')
    grid=fig.add_gridspec(3,4,height_ratios=[1,1.65,1.1])
    ax=fig.add_subplot(grid[0,:]);price=fig.add_subplot(grid[1,:3]);hist=fig.add_subplot(grid[1,3]);heat=fig.add_subplot(grid[2,:3]);info=fig.add_subplot(grid[2,3])
    left=decision-60*60*NS
    c=context[(context.available_ns>=left)&(context.available_ns<=decision)]
    ax.plot(dt(c.available_ns),c.close,color='#172b4d',lw=1,label='M1 midpoint close (контекст)')
    for key,color,label in [('vwap60','#197a67','VWAP60'),('vwap240','#97569d','VWAP240'),('day_vwap','#b7782a','сессионный VWAP')]:
        ax.plot(dt(c.available_ns),c[key],color=color,lw=1.3,label=label)
    ax.plot(dt(c.available_ns),c.bb1,color='#478dc0',lw=.8,label='BB M1: SMA20 ± 2s')
    ax.fill_between(dt(c.available_ns),c.bb1-c.bb1_half,c.bb1+c.bb1_half,color='#478dc0',alpha=.12)
    ax.set_title(f"{row.date} · {NAMES[row.family]} · {'LONG' if row.side==1 else 'SHORT'} · #{row.candidate_id}",loc='left',fontsize=15)
    ax.legend(loc='upper left',fontsize=8,ncol=5);ax.set_ylabel('Цена, USD/oz')
    # All native points determine the extrema; 1s envelope is only display.
    display=pd.DataFrame({'t':dt(t),'mid':mid}).set_index('t').resample('1s',closed='right',label='right').agg(['min','max','last'])['mid']
    price.fill_between(display.index,display['min'],display['max'],color='#294b68',alpha=.23,label='native диапазон внутри 1s')
    price.plot(display.index,display['last'],color='#294b68',lw=1,label='последний midpoint в 1s')
    price.axvspan(dt(decision),dt(end),color='#d8dce0',alpha=.5,label='будущее: только для исхода')
    low,high=float(np.nanmin(mid)),float(np.nanmax(mid));margin=max(.5,(high-low)*.1)
    price.set_ylim(low-margin,high+margin)
    if np.isfinite(row.level):
        price.axhline(row.level,color='#7c5d35',lw=1.5,label='замороженная область времени')
        price.axhspan(row.level-.25*row.risk_hint,row.level+.25*row.risk_hint,color='#b99868',alpha=.12)
    else:
        if low-margin<=row.frozen_bb1<=high+margin:
            price.axhline(row.frozen_bb1,color='#478dc0',ls='--',label='BB midpoint, заморожен')
        if row.nearest_rail!=99:
            v=row.frozen_vwap60+row.nearest_rail*row.frozen_sigma60
            if low-margin<=v<=high+margin:price.axhline(v,color='#197a67',ls='--',label='ближайшая frozen VWAP rail')
    for stamp,value,label,color in [(start,row.start_price,'начало','#7d8691'),(int(row.peak_ns),row.peak_price,'экстремум хода','#946c20'),
                                     (int(row.trough_ns),row.trough_price,'край отката','#a64a4a'),(decision,row.decision_price,'подтверждение','#008579')]:
        price.scatter([dt(stamp)],[value],color=color,s=36,zorder=5,label=label)
    price.axvline(dt(int(row.gate_ns)),color='#c98642',ls=':',lw=1,label='старт выдержки')
    price.axvline(dt(decision),color='#008579',lw=1.4)
    if o.status=='COMPLETE':
        price.scatter([dt(int(o.entry_ns)),dt(int(o.exit_ns))],[o.entry_price,o.exit_price],marker='x',s=70,color='#b54e53',zorder=6,label='фактические BBO исполнения')
    price.set_title(f"R={row.risk_hint:.2f}; откат={row.pullback_r:.2f}R; от края до подтверждения={row.rebound_r:.2f}R / {row.extreme_to_confirmation_seconds:.1f}s",loc='left',fontsize=10)
    price.legend(loc='best',ncol=3,fontsize=7);price.set_ylabel('Цена, USD/oz')
    bins=np.asarray(profile['bin_low_nano'])/NS+.25
    hist.barh(bins,profile['bin_seconds'],height=.45,color='#638fa4',alpha=.85)
    hist.set_ylim(price.get_ylim());hist.set_xlabel('секунды до решения');hist.set_title('Время по цене\nтолько история эпизода',fontsize=10)
    edges=start+(np.arange(25,dtype=np.int64)*(decision-start)//24)
    observed_bins=np.asarray(profile['bin_low_nano'],dtype=np.int64)
    allbins=np.arange(observed_bins[0],observed_bins[-1]+500_000_000,500_000_000,dtype=np.int64)
    matrix=np.zeros((len(allbins),24))
    for j in range(24):
        part=mapper.profile(int(edges[j]),int(edges[j+1]))
        matrix[np.searchsorted(allbins,part['bin_low_nano']),j]=part['bin_seconds']
    if len(allbins):
        heat.pcolormesh(md.date2num(dt(edges)),np.r_[allbins/NS,(allbins[-1]+500_000_000)/NS],matrix,cmap='YlGnBu',shading='flat')
    heat.set_title('Порядок накопления времени: 24 последовательных окна до решения',fontsize=10,loc='left')
    heat.set_ylabel('Цена, USD/oz')
    info.axis('off')
    info.text(0,1,f"Δ VWAP60 /15m: {signed(row.slope60_r,'R')}\nΔ VWAP240 /15m: {signed(row.slope240_r,'R')}\nКонтекст: {row.macro_alignment}\n\nПуть 1s: {row.path_1s:.2f} USD/oz\nNative путь: {row.native_path:.2f} USD/oz\nСдвиг центра 5m:\n{signed(row.time_migration,' USD/oz',2)}\nПерекрытие 5m: {row.time_overlap:.2f}\n\nBBO исход: {o.status}/{o.reason}\nNet: {o.net_per_oz:+.2f} USD/oz\nКомиссия: 0.30 USD/oz\nНе обучающий признак.",va='top',fontsize=9.5,linespacing=1.3)
    for a in (ax,price,heat):
        a.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S',tz='UTC'));a.grid(alpha=.15)
    fig.suptitle('Новый эпизодный слой · причины входа доступны слева от зелёной линии · шкалы UTC',fontsize=10,color='#536170')
    name=f'scene_{int(row.candidate_id)}.png';fig.savefig(root/name,dpi=130);plt.close(fig)
    return dict(candidate_id=int(row.candidate_id),date=row.date,family=row.family,file=name,sha256=sha(root/name),
                outcome_status=o.status,reason=o.reason,net_per_oz=float(o.net_per_oz) if np.isfinite(o.net_per_oz) else None,
                time_mass_seconds=float(matrix.sum()),episode_seconds=(decision-start)/NS,
                selection='Lowest pandas stable hash of time/side among admitted slope-aligned events per day/family; no outcomes')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();root=ROOT/args.run
    manifest=read(root/'manifest.json');assert manifest['status']=='COMPLETE'
    f=pd.read_parquet(root/'features.parquet');accounts=pd.read_parquet(root/'accounts.parquet')
    meta=sum([read(p) for p in sorted((root/'days').glob('*/scenes.json'))],[])
    atlas=[scene(root,f[f.candidate_id==m['candidate_id']].iloc[0],m) for m in meta]
    dump(root/'atlas.json',atlas)
    a=accounts[(accounts.scenario=='base')&(accounts.direction=='signal')].copy()
    stress=accounts[(accounts.scenario=='stress')&(accounts.direction=='signal')].set_index('recipe')
    a['stress_net']=a.recipe.map(stress.net_usd)
    a['stress_complete']=a.recipe.map(stress.capital_curve_complete)
    a.loc[a.coin_complete<199,['coin_median','coin_p95']]=np.nan
    a=a[['recipe','trades','active_days','net_usd','mean_usd','capital_curve_complete','stress_net','stress_complete','coin_complete','coin_median','coin_p95','max_drawdown_usd']]
    a.columns=['Вариант','Сделок','Дней','Реализовано, USD','Средняя, USD','Полный счёт','Стресс реализовано','Стресс полный','Полных монеток/199','Монетка p50','Монетка p95','MTM DD, USD']
    summary=read(root/'summary.json')
    cards=''.join(f'<article><h3>{html.escape(x["date"]+" · "+NAMES[x["family"]])}</h3><a href="{x["file"]}"><img loading="lazy" src="{x["file"]}" alt="Эпизод и его причинное подтверждение"></a><p>#{x["candidate_id"]} · {x["reason"]} · net {x["net_per_oz"]:+.2f} USD/oz. Выбран до знания исхода.</p></article>' for x in atlas)
    body=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Фразы продолжения · диагностическая неделя</title><style>
    body{{font:17px/1.55 system-ui,sans-serif;color:#172d3e;background:#f6f7f8;margin:0}}main{{max-width:1400px;margin:auto;padding:32px}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:36px}}.note{{padding:18px 22px;background:#fff2dc;border-left:5px solid #b77c26}}article{{background:white;margin:28px 0;padding:20px;border:1px solid #dce2e6;border-radius:10px}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:14px;white-space:nowrap;background:white}}td,th{{text-align:right;padding:10px;border:1px solid #dce2e6}}th:first-child,td:first-child{{text-align:left}}code{{background:#e8edf0;padding:2px 4px}}@media(max-width:600px){{main{{padding:14px}}h1{{font-size:27px}}article{{padding:8px}}}}
    </style><main><p>CONTINUATION_PHRASES_V1 · GCZ6 · 24–28 августа 2026</p><h1>Продолжение движения после наблюдаемой реакции</h1>
    <p>Вместо требования угадать разворот режима проверены три последовательности: импульс–откат–возобновление, выход–ретест и повторный отскок после ухода. События строятся на native midpoint. Предложение появляется только после обратного хода 0.3R и наблюдаемого удержания 2s.</p>
    <div class="note">Это диагностический опыт на уже исследовавшихся датах. Не слепой holdout, не обученная модель и не торговый допуск. R заморожен как масштаб геометрии; денежный стоп −R и тейк +1.5R заданы в net PnL, поэтому это не буквальные ценовые границы вокруг экстремума.</div>
    <h2>Что выяснилось</h2><p>Все10 полных базовых счетов дали убыток; ещё2 счёта TPO-ретеста закончились с неизвестным обязательством. Сочетание наклона VWAP и подтверждённой реакции не показало преимущества. В типичном предложении от края до признания реакции уже пройдено около0.42R за9s; текущая оценка спреда с комиссией составляет около0.28–0.31R. Уже пройденный ход и денежные издержки — разные величины.</p>
    <p>На сцене514 обе средние направлены вниз, но последний возврат к месту — быстрый подъём примерно9USD/oz. Это противоречие теряется в общей метке ALIGNED. Следующее описание должно отдельно хранить последнюю ногу подхода, силу/скорость реакции, текущий коридор и возраст памяти места. Данные ещё не доказали полезность этих новых мер как фильтров.</p>
    <h2>Что разделено по смыслу</h2><p>VWAP60: локальный объёмный ориентир и его изменение за15мин. VWAP240: более медленный контекст, сохранён как страта. Прошлая, текущая и объединённая сессии: крупные адреса, без обязательного совпадения направлений. BB M1: SMA20 и разброс завершённых midpoint закрытий. VWAP60±σ: разброс фактических принятых сделок за60мин. Это разные окна и распределения; разность линий нельзя целиком приписывать «умному объёму».</p>
    <p>Область времени — мода предыдущих30мин, замороженная на следующий30мин блок. Она проверяется рядом с округлённым уровнем и со сдвинутой на2.5тика сеткой. Повторный отскок требует ухода наR; дрожание вокруг линии не увеличивает число значимых визитов. На границе блока незавершённая память явно истекает.</p>
    <p>После денежного опыта добавлено точное разложение rolling-VWAP: ΔV = [Σ новых v(p−V₀) − Σ выбывших v(p−V₀)] / новый объём. В5079 известных наблюдениях на60/240min вклады1650 раз имели разные знаки;2651 раз модуль выбывающего вклада был больше. Это зависимые описания арифметики средней, а не сигналы намерений участников. Правила игры не менялись. <a href="anchor_diagnostics/receipt.json">Проверка разложения</a>.</p>
    <h2>Одинаковая механическая игра</h2><p>Один контракт, одна позиция, начальные100000USD. Вход/выход: первая публикация строго после100ms, bid/ask и комиссия0.30USD/oz за круг; максимум120s. Стресс250ms и ещё1тик на каждое исполнение. Очередь, глубина и market impact не моделируются. Каждый вариант начинает отдельный последовательный счёт; будущий результат не используется для отбора.</p>
    <h2>Результаты всех заранее заданных вариантов</h2><p>{summary['proposals']} предложений, {summary['admitted']} прошли проверку источника/стоимости. По199 монеток на тех же предложениях для каждого варианта. Монетки могут иметь другое число сделок и экспозицию из-за разной длительности; это контроль направления, не идеально совпадающий timing. Квантили не показаны, если хотя бы один счёт монетки неполон. Множественный поиск не превращается в доказательство через лучший p.</p><p class="note">Если «Полный счёт» = False, показана только реализованная сумма до потери наблюдения; итоговая стоимость открытой позиции неизвестна. Такой вариант нельзя ранжировать по конечному PnL. Ограничения дневного убытка и просадки останавливают новые входы, поэтому число предложений и сделок различается.</p><div class="scroll">{a.to_html(index=False,float_format=lambda x:f'{x:,.2f}')}</div>
    <h2>Атлас без выбора по прибыли</h2><p>По одному эпизоду каждой семьи в день выбран устойчивым хешем времени и стороны. Серый участок — будущее только для аудита. Верхний график контекста заканчивается в момент решения. Полосы на подробном графике заморожены до реакции; невместившиеся линии остаются в численных признаках.</p>{cards}
    <h2>Что получает будущее обучение</h2><p><code>ledger.parquet</code>: контакт, ход, откат, попытка подтверждения, срыв, отмена, истечение и подтверждение. <code>features.parquet</code>: только известные числа/маски подтверждённых предложений. <code>outcomes.parquet</code>: отдельные будущие LONG/SHORT исходы. Ledger хранит незавершённые и неудавшиеся сюжеты; готового модельного датасета всех фаз и learned encoder здесь ещё нет.</p>
    <p>Код и параметры сохранены в source_snapshot, исходные источники и проверки удаления будущего — в manifest.json. <a href="validation.json">Независимая проверка</a> · <a href="../astra_review/review_ru.json">Визуальный разбор старой карты Astra High</a> · <a href="../astra_review/visual_new_episodes_review.json">Разбор15 новых эпизодов Astra High</a> · <a href="../astra_review/code_review_ru.json">Разбор состояний Astra</a>.</p></main></html>'''
    (root/'report.html').write_text(body,encoding='utf-8')
    dump(root/'report_receipt.json',dict(status='GENERATED',reporter_sha256=sha(Path(__file__)),
         computational_manifest_sha256=sha(root/'manifest.json'),atlas_sha256=sha(root/'atlas.json'),
         html_sha256=sha(root/'report.html'),images=len(atlas),browser_rendered=False))
    print(json.dumps({'images':len(atlas),'report':str(root/'report.html')}))


if __name__=='__main__':main()
