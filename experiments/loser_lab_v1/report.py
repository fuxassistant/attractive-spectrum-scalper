"""Research figures and a local Russian viewer; numbers come from saved runs."""
from __future__ import annotations
import argparse,json,html,re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.hypothesis_atlas_v1.run import dump,digest

ROOT=Path(__file__).resolve().parents[2]
NAMES={'frozen_geometry':'Геометрия: старый опыт','frozen_context':'Контекст: старый опыт','pair_context':'Только последние два дня',
       'blend_context':'Смесь старого опыта и пары','expanding_context':'Вся доступная история','clean_blend':'Цель: чистый проигрыш',
       'expert_pair':'Выбор правил по паре','memory_return':'Память удержания','momentum':'По импульсу','anti_momentum':'Против импульса',
       'always_long':'Всегда LONG','always_short':'Всегда SHORT','random_all':'Случайная сторона',
       'continuation_blend':'Продолжение импульса: смесь','continuation_frozen':'Продолжение: старый опыт','anti_impulse_control':'Против сильного импульса'}
STYLES={'fixed':'Один контракт','after_loss':'×2 после убытка, максимум ×4','after_gain':'×2 после выигрыша, максимум ×4'}
COLORS={'fixed':'#087f8c','after_loss':'#cb7851','after_gain':'#7657a4'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def money(v):return f'{v:,.0f}'.replace(',',' ')
def num(v):return '—' if pd.isna(v) else f'{v:.3f}'
def pct(v):return '—' if pd.isna(v) else f'{100*v:.1f}%'
def esc(v):return html.escape(str(v))


def save(fig,out,name):
    fig.savefig(out/name,dpi=150,bbox_inches='tight');plt.close(fig)


def figures(out,main,second):
    dates=sorted(main['daily'].date.unique());daymap={v:i for i,v in enumerate(dates)}
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    for ax,data,policy,color,title in [(axes[0],main,'blend_context','#087f8c','Первый опыт: смесь старой модели и двух дней'),
                                      (axes[1],second,'continuation_blend','#7657a4','Второй опыт: выбрать продолжение и поставить против')]:
        g=data['daily'];g=g[(g.target=='usd2_300')&(g.policy==policy)].sort_values('date')
        ax.bar([daymap[v] for v in g.date],g.reward,color=[color if x>=0 else '#c7624d' for x in g.reward],width=.65)
        ax.axhline(0,color='#75828b',lw=.8);ax.axvspan(7.5,9.5,color='#e8edf1',zorder=-1)
        ax.set_xticks(range(len(dates)),[v[8:]+'.'+v[5:7] for v in dates]);ax.set_title(title,loc='left')
        ax.set_ylabel('(потери X − выигрыши X) / входы');ax.set_ylim(-.6,.65);ax.grid(axis='y',alpha=.2)
        for r in g.itertuples():ax.text(daymap[r.date],r.reward+(.035 if r.reward>=0 else -.06),f'n={r.n}',ha='center',fontsize=8)
    fig.suptitle('Качество проигрыша без стоимости и без размера ставки\nСерый фон — два следующих дня; выше нуля лучше для лузера',fontsize=14)
    save(fig,out,'quality_days.png')
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    for ax,data,policy,title in [(axes[0],main,'blend_context','Первый опыт'),(axes[1],second,'continuation_blend','Второй опыт')]:
        for style in STYLES:
            t=data['trades'];t=t[(t.target=='usd2_300')&(t.policy==policy)&(t['mode']=='bid')&(t['style']==style)&(~t.reverse)].sort_values('t_ns')
            x=[pd.Timestamp(dates[0],tz='UTC')]+list(pd.to_datetime(t.exit_ns,utc=True));y=[10000]+list(t.equity_usd)
            ax.step(x,y,where='post',label=STYLES[style],color=COLORS[style],lw=1.5)
        ax.axhline(0,color='#c7624d',lw=1);ax.axhline(2000,color='#aaa',ls=':',lw=.8)
        ax.set_title(title+' · все сигналы прежние, изменяется только ставка',loc='left');ax.set_ylabel('Бумажный депозит, USD');ax.grid(alpha=.15)
        ax.legend(fontsize=8,loc='upper left');ax.tick_params(axis='x',labelrotation=15)
    fig.suptitle('Bid → bid, комиссия = 0, спред исполнения = 0\nСтарт $10 000, один контракт = 100 oz, без пополнений и заимствования',fontsize=14)
    save(fig,out,'deposits.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,data,policy,title in [(axes[0],main,'blend_context','Первый опыт'),(axes[1],second,'continuation_blend','Второй опыт')]:
        for i,style in enumerate(STYLES):
            a=data['accounts'];a=a[(a.target=='usd2_300')&(a.policy==policy)&(a['mode']=='bid')&(a['style']==style)&(~a.reverse)].iloc[0]
            r=data['random'];r=r[(r.policy==policy)&(r['style']==style)]
            lo,med,hi=r.terminal_usd.quantile([.05,.5,.95]);ax.plot([i,i],[lo,hi],color='#a6b3bd',lw=6)
            ax.scatter([i],[med],marker='_',s=170,color='#43535c');ax.scatter([i],[a.terminal_usd],s=65,color=COLORS[style],zorder=5)
            ax.text(i,hi+650,f'80% потеряли\n{r.depleted80.mean():.0%} монеток',ha='center',fontsize=8)
        ax.set_xticks(range(3),['Фиксированно','После убытка','После выигрыша']);ax.set_title(title,loc='left');ax.axhline(10000,color='#aaa',ls=':')
        ax.axhline(0,color='#c7624d',lw=.8);ax.set_ylim(-800,37000);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Случайная сторона на тех же сигналах и с тем же правилом ставок\nСерый: 5–95-й процентили 99 счетов, риска: медиана, точка: модель',fontsize=12)
    save(fig,out,'risk_controls.png')


def scenes(out,second,source):
    dec=second['decisions'];dec=dec[(dec.policy=='continuation_blend')&dec.accepted].sort_values(['t_ns','candidate_id'])
    metadata=read(ROOT/'research_results/zone_conditional_v1/verified_v1/manifest.json');markets={r['date']:Path(r['source_market']) for r in metadata['receipts']}
    rows=[]
    for block in ('primary','extension'):
        pool=dec[dec.block==block]
        if not len(pool):continue
        r=pool.iloc[0];q=pd.read_parquet(markets[r.date]/'quotes.parquet',columns=['available_ns','bid_nano'])
        a=int(np.searchsorted(q.available_ns,r.t_ns,'right')-1);bid0=int(q.iloc[a].bid_nano)
        q=q[q.available_ns.between(int(r.t_ns)-120_000_000_000,int(r.exit_ns)+10_000_000_000)]
        x=(q.available_ns.to_numpy(np.int64)-int(r.t_ns))/1e9;y=int(r.side)*(q.bid_nano.to_numpy(np.int64)-bid0)/1e9
        past=x<=0;pastmin=min(-r.threshold_per_oz,float(y[past].min()));pastmax=max(r.threshold_per_oz,float(y[past].max()))
        for future in (False,True):
            fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');ax.plot(x[past],y[past],color='#425d6b',lw=1)
            if future:
                active=(x>0)&(q.available_ns.to_numpy()<=r.exit_ns);after=q.available_ns.to_numpy()>r.exit_ns
                ax.plot(x[active],y[active],color='#cb7851',lw=1.3);ax.plot(x[after],y[after],color='#aaa',ls=':')
                ax.scatter([(r.exit_ns-r.t_ns)/1e9],[r.side*r.bid_delta_per_oz],marker='X',color='#7657a4',zorder=5)
            ax.axhline(-r.threshold_per_oz,color='#087f8c',ls='--',label='Потеря X: награда лузеру +1')
            ax.axhline(r.threshold_per_oz,color='#c7624d',ls='--',label='Выигрыш X: награда −1');ax.axvline(0,color='#777',ls=':')
            ax.set_xlim(-120,max(30,(r.exit_ns-r.t_ns)/1e9+10) if future else 0)
            ax.set_ylim(pastmin-.3,pastmax+.3);ax.set_ylabel('Результат выбранной стороны, USD/oz');ax.set_xlabel('Секунды относительно входа; время публикации')
            direction='LONG' if r.side==1 else 'SHORT';title=f"{r.date} · {direction} · {'ПОСЛЕ' if future else 'ДО: будущее скрыто'}"
            ax.set_title(title,loc='left');ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.15)
            name=f'scene_{block}_{"after" if future else "before"}.png';save(fig,out,name)
        rows.append(dict(block=block,candidate_id=int(r.candidate_id),date=r.date,side=int(r.side),class_index=int(r.class_index),reward=int(r.reward),
                         bid_unit_pnl=float(r.bid_unit_pnl),selection='Earliest accepted continuation_blend signal in block, no outcome selection'))
    dump(out/'scene_receipt.json',dict(scenes=rows,source_run=str(source)))
    return rows


def load(folder):
    return {name:pd.read_parquet(folder/(file+'.parquet')) for name,file in [('quality','quality'),('daily','daily_quality'),('accounts','accounts'),('trades','trades'),('random','random_accounts'),('decisions','decisions')]}


def quality_table(data,filterable=True):
    rows=[]
    for r in data['quality'].sort_values(['target','block','policy']).itertuples():
        attribute=f' data-target="{r.target}"' if filterable else ''
        rows.append(f'<tr{attribute}><td>{esc(NAMES[r.policy])}</td><td>{"8 дней" if r.block=="primary" else "Следующие 2"}</td><td>{r.selected}</td><td>{r.losses} / {r.gains} / {r.timeouts}</td><td>{pct(r.decisive_accuracy)}</td><td>{num(r.day_reward)}</td><td>{num(r.random_p95)}</td><td>{num(r.mean_bid_per_oz)}</td></tr>')
    return '<div class="scroll"><table><thead><tr><th>Правило</th><th>Период</th><th>Входы</th><th>−X / +X / время</th><th>−X первым среди касаний</th><th>Баллы за вход, поровну по дням</th><th>95% монеток</th><th>Средний bid PnL, $/oz</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--second',required=True);a=ap.parse_args()
    out=ROOT/a.run;second_path=ROOT/a.second;main=load(out);second=load(second_path);figures(out,main,second);examples=scenes(out,second,second_path)
    account_rows=[]
    for stage,data,policy in [('Первый',main,'blend_context'),('Второй',second,'continuation_blend')]:
        for r in data['accounts'][(data['accounts'].target=='usd2_300')&(data['accounts'].policy==policy)&(~data['accounts'].reverse)&(data['accounts']['mode']=='bid')].itertuples():
            account_rows.append(f'<tr><td>{stage}</td><td>{STYLES[r.style]}</td><td>{r.trades}</td><td>{money(r.terminal_usd)}</td><td>{"да" if r.depleted80 else "нет"}</td><td>{"да" if r.bankrupt else "нет"}</td><td>{r.unaffordable}</td><td>{num(r.pnl_per_planned_risk)}</td></tr>')
    cards=''
    for r in examples:
        cards+=f'<article><h3>{r["date"]}: первый выбранный вход</h3><p>Сторона {"LONG" if r["side"]==1 else "SHORT"}; пример выбран по времени, без просмотра результата.</p><img src="scene_{r["block"]}_before.png"><details><summary>Раскрыть следующий путь и ответ таргета</summary><p>Награда: {r["reward"]:+d}; результат bid: {r["bid_unit_pnl"]:+.2f} $/oz.</p><img src="scene_{r["block"]}_after.png"></details></article>'
    text=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Лаборатория сильного лузера</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#eef3f5;color:#243840;font:17px/1.65 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:32px 20px 80px}}h1{{font-size:clamp(30px,5vw,52px);line-height:1.15}}h2{{font-size:28px}}h3{{font-size:21px}}article,section{{background:white;padding:26px;border-radius:12px;margin:24px 0;border:1px solid #dbe4e8}}.eyebrow{{font-size:12px;letter-spacing:.14em;color:#087f8c;font-weight:bold}}.callout{{border-left:4px solid #087f8c;padding:16px;background:#e3eff0}}.warning{{border-color:#cb7851;background:#f8eee8}}img{{max-width:100%;height:auto;display:block}}nav{{display:flex;gap:12px;flex-wrap:wrap}}a{{color:#087380}}table{{border-collapse:collapse;min-width:780px;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dde5e9;vertical-align:top}}th{{background:#edf3f4}}.scroll{{overflow:auto}}summary{{cursor:pointer;color:#087380;font-weight:600;padding:12px 0}}select{{font:inherit;padding:10px;max-width:100%}}.muted{{color:#60737d;font-size:14px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.stat{{background:#edf3f4;padding:16px}}.stat strong{{display:block;font-size:27px}}@media(max-width:600px){{main{{padding:22px 12px}}section,article{{padding:16px}}.grid{{grid-template-columns:1fr}}h2{{font-size:24px}}}}</style>
<main><div class="eyebrow">GCZ6 · ПРИЧИННЫЕ ТАРГЕТЫ · ИССЛЕДОВАТЕЛЬСКИЙ REPLAY</div><h1>Как научиться проигрывать движению цены</h1>
<p>Нулевая комиссия, единая цена bid для обеих сторон, симметричная гонка −X / +X. Сначала качество выбора входа. Затем размер ставки и депозит.</p>
<nav><a href="#quality">Качество</a><a href="#risk">Депозит и риск</a><a href="#trivial">Простые объяснения</a><a href="#scenes">Реальные пути</a><a href="#method">Устройство</a></nav>
<section><h2>Локальный лузер найден. Устойчивость не подтверждена.</h2><p>Первый опыт получил 258 потерь против 211 выигрышей на 516 входах за восемь следующих дней. После этого — 54 против 72 на двух днях продолжения: модель перестала быть лузером.</p>
<div class="grid"><div class="stat">Первый опыт, без стоимости<strong>$10 000 → $20</strong>Фиксированная ставка; капитал перестал позволять новый вход.</div><div class="stat">Второй опыт, фиксированно<strong>$10 000 → $4 870</strong>Обучение условиям продолжения импульса, ставка против него.</div><div class="stat">Второй опыт, ×2 после выигрыша<strong>$10 000 → −$20</strong>Банкротство в бумажном replay; максимум ×4, не критерий обучения.</div></div>
<p class="callout">Почти потерянный депозит первого опыта скрыл смену качества: оставшихся $20 уже не хватало на продолжение, когда сигналы стали приносить прибыль. Equity сама по себе не сообщает, сохраняется ли умение проигрывать.</p>
<p class="muted">Первый каталог вариантов зафиксирован до расчёта. Второй опыт создан после анализа первого. Все даты ранее исследовались; это разработка, а не независимое подтверждение.</p></section>
<section id="quality"><h2>За что модель получает награду</h2><p>−X раньше +X: +1. +X раньше −X: −1. Ни одна граница до срока: 0. WAIT: 0. Неизвестный путь не становится нулём. X = $2/oz, то есть $200 на один стандартный контракт. Вариант «чистого проигрыша» отдельно требует, чтобы благоприятный ход до потери был не больше 0.25X.</p><img src="quality_days.png"><p>Каждый столбец — средний ограниченный балл на выбранный вход этого дня. Количество входов показано отдельно. Крупный проскок цены не даёт дополнительной награды.</p>
<p class="muted">Нажмите на график, чтобы открыть его в полном размере.</p><label>Таргет в таблице первого опыта: <select id="target"><option value="usd2_300">±$2 за 300 секунд</option><option value="usd2_120">±$2 за 120 секунд</option><option value="scale_min2_300">±max($2, R), округление к тику, 300 секунд</option></select></label>
<details><summary>Все варианты первого опыта</summary>{quality_table(main)}</details><details><summary>Второй опыт: продолжение в координатах импульса</summary><p>Этот опыт использует ±$2, 300 секунд.</p>{quality_table(second,False)}</details>
<p class="muted">Баллы усредняются поровну по активным дням; суммарные частоты считаются по входам. На редких правилах эти показатели могут расходиться. 95-й процентиль 99 случайных сторон — описательное сравнение, не поправленный на множество опытов p-value.</p></section>
<section id="risk"><h2>Депозит — последующая проверка</h2><img src="deposits.png"><div class="scroll"><table><tr><th>Опыт</th><th>Размер ставки</th><th>Сделки</th><th>Осталось, USD</th><th>Терял ≥80%</th><th>Банкротство</th><th>Не хватило капитала</th><th>PnL / сумма планового риска</th></tr>{''.join(account_rows)}</table></div>
<p>После убытка или выигрыша ставка проходит 1 → 2 → 4 контракта, затем остаётся на 4 до сброса. Если денег не хватает на плановую потерю X при требуемом размере ставки, вход пропускается; автоматического уменьшения размера нет. Гэп может превысить плановую потерю. Просадка 80%, нехватка средств на требуемую ставку и отрицательный депозит — разные исходы.</p>
<img src="risk_controls.png"><p class="callout warning">На сигналах первого опыта случайная сторона с увеличением ставки после выигрыша теряла 80% депозита в 42 из 99 прогонов. На сигналах второго — в 18 из 99, один прогон обанкротился. Увеличение ставки умеет создавать сильный денежный проигрыш без сильной модели.</p></section>
<section id="trivial"><h2>Какие простые объяснения проверены</h2><p><b>Комиссия и spread.</b> Их нет в обучающих ответах. В отдельной оценке используются bid/ask, 100 мс и $0.30/oz за оборот; стресс — 250 мс и тик на каждый fill. Зеркальная сторона первого лузера на bid дала $15 980 конечного капитала, но в базовом BBO-исполнении осталось $100. Слабого направленного эффекта недостаточно для оплаты исполнения.</p>
<p><b>Просто SHORT.</b> На тех же 516 входах средний дневной балл постоянного SHORT равен 0.055, модели — 0.093. Но ставка против 30-секундного импульса дала 0.098: сложный выбор стороны не оказался сильнее этого простого объяснения.</p>
<p><b>Продолжение импульса.</b> Поэтому второй опыт прогнозирует, когда импульс продолжится, в координатах его текущего направления. При достаточной оценке он ставит против. На первом блоке — 57.3% потерь среди достигнутых границ, 193 входа. На двух следующих днях всего 14 входов и 50%: устойчивость не установлена.</p>
<p><b>Память удержания.</b> В новой игре bid и с новыми симметричными выходами фиксированный вариант закончил с $12 080. Старый денежный убыток этой идеи не превратился в сильного лузера автоматически.</p>
<p><b>Чистый проигрыш.</b> Более строгий таргет резко уменьшил поддержку; первый вариант выбрал 46 входов на основном блоке и 6 на продолжении. Улучшения, пережившего следующий период, не получено.</p></section>
<section id="scenes"><h2>Как выглядит попытка проиграть X</h2><p>Показаны первые принятые сигналы второго опыта в каждом блоке. Вертикальная ось — изменение bid в координатах выбранной позиции. Издержек нет.</p>{cards}</section>
<section id="method"><h2>Что реализовано</h2><p>Исходный префикс 3–14 августа обучает замороженные модели. Перед каждым следующим днём используются только завершившиеся ответы прошлых дней; раннее касание также ждёт общего срока созревания. Основной блок — 19–28 августа, продолжение — 31 августа и 1 сентября. Базовая популяция: 2 456 ранее сформированных возможностей около старых зон.</p>
<p>Первый опыт: три таргета × 13 политик. Регуляризованные модели геометрии и контекста, переобучение на паре дней, смесь 50/50, расширяющаяся история, чистый проигрыш, выбор экспертов и простые контроли. Второй: два прогноза продолжения и контроль против сильного импульса. Вход модели не содержит equity, ставки, будущего пути или денежного результата.</p>
<p>Цена bid на обеих сторонах — исследовательская общая координата. Она не означает реальное исполнение LONG по bid. Книга, время публикации, разрывы и эпохи проверяются. Первые касания, путь и стоимость исполнения разделены. Не проверяются реальные маржинальные требования, очередь или воздействие наших заявок.</p>
<p><a href="summary.json">Сводка</a> · <a href="verification.json">Независимая проверка</a> · <a href="quality.parquet">Качество всех вариантов</a> · <a href="accounts.parquet">Счета</a> · <a href="diagnostics.json">Контроли</a></p></section></main>
<script>document.querySelector('#target').addEventListener('change',e=>{{document.querySelectorAll('tr[data-target]').forEach(r=>r.hidden=r.dataset.target!==e.target.value)}});document.querySelector('#target').dispatchEvent(new Event('change'));</script></html>'''
    labels={'quality_days.png':'Дневное качество проигрыша без издержек','deposits.png':'Бумажный депозит при трёх правилах ставки','risk_controls.png':'Модели и случайная сторона при одинаковом риске'}
    text=re.sub(r'<img src="([^"]+)">',lambda m:f'<a href="{esc(m[1])}" target="_blank"><img src="{esc(m[1])}" alt="{labels.get(m[1],"Путь bid выбранного входа")}"></a>',text)
    (out/'report.html').write_text(text,encoding='utf-8',newline='\n')
    dump(out/'render_receipt.json',dict(report_sha256=digest(out/'report.html'),images={p.name:digest(p) for p in out.glob('*.png')},
         second_run=str(second_path),second_summary_sha256=digest(second_path/'summary.json'),implementation_sha256=digest(__file__)))
    print(out/'report.html')


if __name__=='__main__':main()
