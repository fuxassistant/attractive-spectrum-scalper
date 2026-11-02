"""Readable research report: parameter failures, changing beliefs, real scenes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.mechanical_lab.data import ROOT,digest
from .prepare import dump,clean

NAMES={'anti_impulse':'Против любого импульса','fade2_g10':'Против импульса · 2 дня','fade5_g10':'Против импульса · 5 дней',
       'fade10_g10':'Против импульса · 10 дней','switch5_g10':'Выбор ошибки · 5 дней','switch10_g20':'Выбор ошибки · 10 дней · 0,20',
       'guard_fade5_g10':'Против импульса с проверкой условий','guard_switch5_g10':'Выбор ошибки с проверкой условий',
       'weekly_selector':'Недельный выбор по прошлым прогнозам','online_local8h':'Коррекция внутри дня · 8 часов',
       'online_local2h':'Коррекция внутри дня · 2 часа','online_global2h':'Общая коррекция · 2 часа',
       'coarse5':'6 признаков движения','no_flow5':'13 признаков без потока/книги','no_memory5':'13 признаков без памяти места',
       'multiscale5':'23 признака с медленным контекстом','blend2_fade':'Первые 10 дней + последние 2 дня','frozen_fade':'Замороженная модель',
       'clean5':'Только чистый путь убытка','frequency5':'Только частоты прошлых 5 дней'}
NAMES.update(always_long='Всегда LONG',always_short='Всегда SHORT',follow_impulse='Следом за любым импульсом',toward_memory='В сторону старой зоны')
for window in (2,5,10):
    for gate in (10,20):
        for mode,title in [('fade','Против импульса'),('switch','Выбор ошибки')]:
            NAMES.setdefault(f'{mode}{window}_g{gate}',f'{title} · {window} дн. · 0,{gate}')


def save(fig,path):
    fig.savefig(path,dpi=155,bbox_inches='tight',facecolor=fig.get_facecolor()); plt.close(fig)


def plots(out,quality,daily,calibration,accounts,random):
    figdir=out/'figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.facecolor':'#fcfbf8','figure.facecolor':'#fcfbf8','axes.titleweight':'bold'})
    keep=['anti_impulse','fade2_g10','fade5_g10','fade10_g10','switch5_g10','switch10_g20','guard_fade5_g10','guard_switch5_g10','weekly_selector','online_local8h']
    q=quality[quality.target.eq('usd2_300')&quality.block.eq('all')].set_index('policy').loc[keep]
    fig,ax=plt.subplots(figsize=(11,5.8)); yy=np.arange(len(q))
    ax.errorbar(q.day_reward,yy,xerr=np.array([q.day_reward-q.day_reward_lo,q.day_reward_hi-q.day_reward]),fmt='o',color='#8b3655',capsize=3)
    ax.axvline(0,color='#555',lw=1); ax.set_yticks(yy,[NAMES[k] for k in keep]); ax.invert_yaxis()
    ax.set_xlabel('Среднее по активным дням: +1 потерян X, −1 заработан X, 0 таймаут')
    ax.set_title('Привлекательная точка ещё не означает устойчивый лузер')
    ax.text(0,-.17,'Полосы: описательный 95% bootstrap по дням. Повторные подборы не скорректированы.',transform=ax.transAxes,fontsize=9)
    save(fig,figdir/'quality.png')
    d=daily[daily.target.eq('usd2_300')&daily.policy.isin(keep)].copy(); d['week']=pd.to_datetime(d.date).dt.to_period('W-SUN').astype(str)
    matrix=d.groupby(['policy','week']).opportunity_reward.mean().unstack().reindex(keep)
    fig,ax=plt.subplots(figsize=(11,5.8)); im=ax.imshow(matrix,vmin=-.08,vmax=.08,cmap='PiYG',aspect='auto')
    ax.set_yticks(np.arange(len(keep)),[NAMES[k] for k in keep]); ax.set_xticks(np.arange(len(matrix.columns)),[k[5:10] for k in matrix.columns],rotation=40)
    ax.set_title('Та же идея меняет знак между неделями'); ax.set_xlabel('Понедельник недели; между июлем и августом есть перерыв')
    for i in range(len(keep)):
        for j in range(len(matrix.columns)):
            v=matrix.iloc[i,j]
            if np.isfinite(v): ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=8,color='#222')
    fig.colorbar(im,ax=ax,label='Награда на возможность входа, WAIT = 0'); save(fig,figdir/'weeks.png')
    keepcal=['coarse5','no_flow5','no_memory5','fade10_g10','fade5_g10','frequency5','multiscale5','fade2_g10']
    c=calibration[calibration.target.eq('usd2_300')&calibration.block.eq('all')].set_index('policy').loc[keepcal]
    fig,ax=plt.subplots(figsize=(10,4.7)); ax.barh([NAMES.get(k,k) for k in keepcal],c.day_logloss,color='#456e7a')
    ax.invert_yaxis(); ax.set_xlim(1.45,1.7); ax.set_xlabel('Log loss пяти исходов на общей выборке; меньше — лучше')
    ax.set_title('37 дней прогноза: больше признаков пока не помогает')
    for i,v in enumerate(c.day_logloss): ax.text(v+.002,i,f'{v:.4f}',va='center',fontsize=9)
    save(fig,figdir/'complexity.png')
    r=random[random.policy.eq('fade5_g10')]; a=accounts[accounts.policy.eq('fade5_g10')&~accounts.reverse]
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
    for ax,style,label in zip(axes,['fixed','after_loss','after_gain'],['Фиксированная ставка','Увеличение после убытка','Увеличение после прибыли']):
        z=r[r['style'].eq(style)]; actual=a[a['style'].eq(style)].terminal_usd.iloc[0]
        ax.hist(z.terminal_usd,bins=22,color='#b8c6c6'); ax.axvline(actual,color='#a53f5f',lw=2,label=f'Модель: ${actual:,.0f}')
        ax.axvline(10000,color='#555',ls=':',lw=1); ax.set_title(label,fontsize=10); ax.set_xlabel('Остаток бумажного депозита, $'); ax.legend(fontsize=8)
        ax.text(.04,.88,f'Случайные направления:\n−80%: {int(z.depleted80.sum())}/{len(z)}',transform=ax.transAxes,fontsize=9)
    axes[0].set_ylabel('Число случайных повторов'); fig.suptitle('Та же модель: почти ноль на счёте может создать размер ставки',fontweight='bold')
    save(fig,figdir/'risk.png')


def scenes(out,source,decisions,features,targets):
    chosen=decisions[decisions.target.eq('usd2_300')&decisions.policy.eq('weekly_selector')&decisions.accepted].sort_values('t_ns')
    chosen=chosen.groupby('symbol',sort=False).head(1); receipts=json.loads((source/'manifest.json').read_text())['receipts']; bydate={r['date']:r for r in receipts}
    rows=[]; f=features.set_index('candidate_id'); y=targets[targets.target.eq('usd2_300')].set_index('candidate_id')
    for i,r in enumerate(chosen.itertuples()):
        x=f.loc[r.candidate_id]; target=y.loc[r.candidate_id]; q=pd.read_parquet(Path(bydate[r.date]['source'])/'quotes.parquet')
        origin=float(q.bid_nano.iloc[int(target.origin_ordinal)])/1e9; qt=q.available_ns.to_numpy(np.int64)
        left=np.searchsorted(qt,r.t_ns-600*10**9); right=np.searchsorted(qt,r.t_ns+300*10**9,'right')
        piece=q.iloc[left:right]; momentum=1 if x.past_net30_r*x.side>=0 else -1
        for reveal in (False,True):
            visible=piece if reveal else piece[piece.available_ns<=r.t_ns]
            # Figure decimation is only visual; target marker uses native first hit.
            step=max(1,len(visible)//2200); sampled=visible.iloc[::step]
            fig,ax=plt.subplots(figsize=(10.5,3.5)); ax.plot((sampled.available_ns-r.t_ns)/1e9,sampled.bid_nano/1e9-origin,color='#315c70',lw=1)
            ax.axvline(0,color='#333',lw=1); ax.axhline(0,color='#aaa',lw=.7); ax.scatter(0,0,s=28,color='#333',zorder=5)
            for side in (-1,1): ax.axhline(side*target.threshold_per_oz,color='#a53f5f' if side==-r.side else '#547d62',ls='--',lw=1)
            if reveal: ax.scatter((target.exit_ns-r.t_ns)/1e9,target.bid_delta_per_oz,s=65,color='#a53f5f',zorder=5)
            else: ax.axvspan(0,300,color='#e4e1da'); ax.text(150,0,'Будущее скрыто',ha='center',va='center',color='#555')
            ax.set_xlim(-600,300); ax.set_ylim(min(-3,piece.bid_nano.min()/1e9-origin-.3) if reveal else min(-3,visible.bid_nano.min()/1e9-origin-.3),
                                             max(3,piece.bid_nano.max()/1e9-origin+.3) if reveal else max(3,visible.bid_nano.max()/1e9-origin+.3))
            ax.set_xlabel('Секунды относительно решения'); ax.set_ylabel('Bid − bid входа, $/oz')
            ax.set_title(f'{r.date} {pd.Timestamp(r.t_ns,tz="UTC").strftime("%H:%M UTC")} · {r.symbol} · '+('после решения' if reveal else 'известно в момент решения'))
            save(fig,out/'figures'/f'scene_{i}_{"after" if reveal else "before"}.png')
        up=r.p1+r.p2; down=r.p3+r.p4
        rows.append(dict(index=i,date=r.date,time=pd.Timestamp(r.t_ns,tz='UTC').strftime('%H:%M UTC'),symbol=r.symbol,candidate_id=r.candidate_id,
          recipe=r.selected_recipe,side='LONG' if r.side==1 else 'SHORT',mechanism='ставка против продолжения импульса' if r.side==-momentum else 'следование за импульсом перед ожидаемым разворотом',
          loss_probability=down if r.side==1 else up,gain_probability=up if r.side==1 else down,score=r.score,
          net30_r=x.past_net30_r*x.side,volume30=np.expm1(x.x_base_mbo_now_trade_total_log),flow=x.x_mbo_mbo_now_trade_imbalance*x.side,
          memory_pass=x.x_mem_recent_pass_fraction,distance_r=x.distance_r,unit_pnl=r.unit_bid_pnl,event_reward=r.reward,seconds=target.seconds))
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); ap.add_argument('--online',required=True); args=ap.parse_args()
    out=ROOT/args.run; online=ROOT/args.online; manifest=json.loads((out/'manifest.json').read_text()); source=Path(manifest['source'])
    tables={n:pd.read_parquet(out/(n+'.parquet')) for n in ('quality','daily_quality','calibration','accounts','random_accounts','decisions','selection_receipts','cost_metrics')}
    tables['calibration']=pd.concat([tables['calibration'],pd.read_parquet(out/'frequency_calibration.parquet')],ignore_index=True)
    for n in ('quality','daily_quality','calibration'):
        more=pd.read_parquet(online/(n+'.parquet')); more=more[more.policy.ne('anti_impulse')]
        tables[n]=pd.concat([tables[n],more],ignore_index=True)
    plots(out,tables['quality'],tables['daily_quality'],tables['calibration'],tables['accounts'],tables['random_accounts'])
    f=pd.read_parquet(source/'features.parquet'); y=pd.read_parquet(source/'targets.parquet')
    scene=scenes(out,source,tables['decisions'],f,y)
    sr=tables['selection_receipts']; selections=[]
    for week,g in sr[sr.target.eq('usd2_300')].groupby('week',sort=True):
        selected=g.selection.iloc[0]; r=g[g.chosen].iloc[0] if selected!='WAIT' else g.sort_values('score',ascending=False).iloc[0]
        d=tables['decisions']; d=d[d.policy.eq('weekly_selector')&d.target.eq('usd2_300')&(pd.to_datetime(d.date).dt.to_period('W-SUN').astype(str)==week)&d.accepted]
        selections.append(dict(week=week,choice=selected,past_score=r.score,past_rows=r.n,past_active_days=r.active_days,
            current_selected=len(d),current_losses=int(d.loss_first.sum()),current_gains=int(d.gain_first.sum()),current_pnl=float(d.unit_bid_pnl.mean()) if len(d) else None))
    payload=clean(dict(names=NAMES,quality=tables['quality'].to_dict('records'),weeks=selections,scenes=scene))
    dump(out/'report_data.json',payload)
    html=HTML.replace('__DATA__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</',r'<\/'))
    (out/'report.html').write_text(html,encoding='utf-8',newline='\n')
    dump(out/'report_receipt.json',dict(figures={p.name:digest(p) for p in (out/'figures').glob('*.png')},report_sha256=digest(out/'report.html'),
        source_manifest_sha256=digest(out/'manifest.json'),online_manifest_sha256=digest(online/'manifest.json'),
        scene_selection='Earliest accepted weekly-selector signal in each contract, independent of its future result.'))
    print(str(out/'report.html'))


HTML='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Лузер · ещё пять недель</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f4ef;color:#222d32;font:17px/1.55 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:40px 28px 80px}h1{font-size:clamp(32px,5vw,58px);line-height:1.08;letter-spacing:-.035em;max-width:900px}h2{font-size:28px;line-height:1.2;margin:45px 0 14px}p{max-width:900px}.eyebrow{font-size:13px;letter-spacing:.13em;color:#72574f;text-transform:uppercase}.lead{font-size:22px;max-width:950px}.card{background:#fffdfa;border:1px solid #dcd8cf;border-radius:14px;padding:22px;margin:20px 0}.numbers{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.numbers b{display:block;font-size:31px;color:#365d69}.muted{color:#657176;font-size:14px}img{display:block;width:100%;height:auto;border-radius:8px}a{color:#31677c}figure{margin:20px 0}figcaption{font-size:14px;color:#657176;margin:8px 3px}.controls{display:flex;flex-wrap:wrap;gap:18px;margin:14px 0}label{font-size:14px}select,button{display:block;max-width:100%;padding:10px 13px;font:inherit;background:#fff;border:1px solid #c6cbc8;border-radius:7px;color:#26353d}button{cursor:pointer;background:#edf2f2}.tablewrap{overflow:auto}table{border-collapse:collapse;white-space:nowrap;font-size:14px;width:100%}td,th{padding:10px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}th{color:#657176;font-weight:500}details{border:1px solid #ddd7cc;border-radius:8px;padding:14px;margin:15px 0}summary{cursor:pointer;font-weight:600}.formula{font:19px/1.65 ui-monospace,monospace;background:#e8eeed;padding:18px;border-radius:8px}.warning{border-left:4px solid #9f5264;padding:12px 18px;background:#f2e8e8}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.tag{font-size:13px;background:#e7eeef;border-radius:20px;padding:4px 10px;display:inline-block}.scene-info{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.scene-info strong{display:block}.footer{border-top:1px solid #ccc;margin-top:44px;padding-top:20px;font-size:14px}@media(max-width:680px){main{padding:25px 16px 50px}.numbers{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.scene-info{grid-template-columns:1fr 1fr}.card{padding:16px}h2{font-size:24px}.lead{font-size:19px}}
</style><main><div class="eyebrow">GC · исследование 02.11.2026 · только симуляция</div><h1>Сильный лузер должен переносить свою ошибку в следующий день</h1>
<p class="lead">Пять добавленных недель ослабили прежний результат. Плохое направление научиться выбирать оказалось почти так же трудно, как хорошее. Размер ставки легко создаёт разорение; устойчивой ошибки модели пока нет.</p>
<div class="numbers card"><div><b>+25</b>дней · 22 июня–24 июля</div><div><b>47</b>дней в общей выборке</div><div><b>5 229</b>причинных точек решения</div><div><b>37</b>дней последовательного прогноза</div></div>
<p>Первые 10 дней, 22 июня–3 июля, используются для начального обучения. Прогнозы идут с 6 июля по 1 сентября. В каждом дне модель получает только созревшие исходы прошлых дней. Новые недели относятся к GCQ6, август — к GCZ6. 3 июля сессия заканчивается в 17:00 UTC; 27–30 июля доступны в архиве, но не входят в эти пять недель; 31 июля отсутствует в источнике. Память локальной надёжности сбрасывается при смене контракта и длинном разрыве.</p>
<p class="warning">Это расширенная исследовательская проверка, не нетронутый holdout. Август уже изучался раньше; вторую стадию с адаптацией внутри дня добавили после просмотра первой. Все варианты показаны, включая неудачные.</p>
<h2>Какое будущее загадывает лузер</h2><div class="formula">Первое достижение bid ± X за H секунд.<br>+1: выбранная сторона потеряла X раньше.<br>−1: выбранная сторона заработала X раньше.<br>0: ни одна граница не достигнута; WAIT тоже 0.</div>
<p>Основной X = $2 за унцию, H = 300 секунд. Это $200 на один условный контракт 100 oz. Обе стороны используют одну цену bid, без комиссии и спреда. Перескок границы остаётся в денежном результате, но не увеличивает награду выше +1. Обратная связь раскрывается только после полного H, даже если граница достигнута раньше.</p>
<div class="grid"><div class="card"><span class="tag">Против продолжения</span><p>Импульс, вероятно, продолжится. Лузер открывается против него. Условия проверяются по движению за 30/120/600 секунд, памяти места и текущему потоку.</p></div><div class="card"><span class="tag">Следом перед разворотом</span><p>Импульс, вероятно, исчерпан. Лузер следует за ним. Режим «выбор ошибки» переключается между этим случаем и ставкой против продолжения.</p></div></div>
<h2>Насколько результат устойчив</h2><figure><a href="figures/quality.png"><img src="figures/quality.png" alt="Качество лузера с описательными интервалами по дням"></a><figcaption>Положительное значение полезно лузеру. Во всех показанных вариантах интервал пересекает ноль. Дни имеют одинаковый вес; это не совпадает с долей проигрышных сделок.</figcaption></figure>
<p><b>55,8% проигрышей</b> у варианта «выбор ошибки, 10 дней, порог 0,20» выглядят интересно: 247 входов, средний bid-результат −$0,158/oz. Но интервал дневной награды −0,021…+0,215 и контроль случайного выбора моментов не подтверждают устойчивость. После просмотра множества вариантов этот результат — кандидат для новой проверки.</p>
<h2>Параметры и все варианты</h2><div class="controls"><label>Какое событие<select id="target"><option value="usd2_300">±$2/oz · 300 секунд</option><option value="usd2_120">±$2/oz · 120 секунд</option><option value="usd3_300">±$3/oz · 300 секунд</option><option value="scale_min2_300">±max($2, текущий R) · 300 секунд</option></select></label><label>Период прогноза<select id="block"><option value="all">Все 37 дней</option><option value="added_forward">Новые данные: 6–24 июля · 15 дней</option><option value="original">Ранее изученные: 3 августа–1 сентября · 22 дня</option></select></label></div>
<div class="tablewrap"><table><thead><tr><th>Вариант</th><th>Входы</th><th>Проигрыш первым</th><th>Награда / день</th><th>95% диапазон</th><th>Bid $/oz</th><th>Награда / возможность</th></tr></thead><tbody id="results"></tbody></table></div>
<p class="muted">Проигрыш первым = потери X / (потери X + выигрыши X); таймауты исключены только из этой доли. Bid $/oz включает таймауты. Награда / день — среднее по активным дням; награда / возможность включает WAIT. Диапазон — описательный bootstrap по дням, не гарантия и не поправка на перебор.</p>
<h2>«Модель выбрала эту стратегию — что это говорит о рынке?»</h2><p>Это говорит о проверяемом ожидании модели. Разделяем её основания в прошлом и то, что рынок затем сделал. Недельный выбор оценивает только уже состоявшиеся прогнозы последних 10 дней, штрафует нестабильность между днями и может выбрать WAIT.</p>
<label>Неделя<select id="week"></select></label><div id="weekcard" class="card"></div>
<figure><a href="figures/weeks.png"><img src="figures/weeks.png" alt="Смена знака награды стратегий по неделям"></a><figcaption>Зелёный — положительная награда лузеру, розовый — отрицательная. Нулевая клетка у селектора может означать WAIT. Календарный разрыв не склеен в непрерывный ряд котировок.</figcaption></figure>
<h2>Объём, память и достаточная сложность</h2><figure><a href="figures/complexity.png"><img src="figures/complexity.png" alt="Сравнение ошибки вероятностного прогноза при разной сложности"></a></figure>
<p>Самая простая проверенная модель использует шесть величин: масштаб R, импульс 30 секунд, его эффективность, диапазон и движение 120 секунд, миграцию 600 секунд. Добавление памяти места и потока/книги пока ухудшает вероятность пяти исходов. Медленный контекст M1 также не помог. Это ограничение проверенного представления и модели на данной выборке; причинную бесполезность объёма отсюда вывести нельзя.</p>
<div class="card"><b>Что модель действительно различает: достигнет ли цена расстояния X, а не куда она пойдёт.</b><p>У модели из шести признаков ошибка прогноза достижения границы снизилась с 0,4652 до 0,4004 относительно частот прошлых пяти дней. Но условная ошибка направления выросла с 0,6939 до 0,7011; точность направления — 49,7%. Улучшение общего прогноза происходит за счёт различения активности. Для сильного лузера этого недостаточно.</p></div>
<p>Признаки MBO описывают глобальный поток за последние 30 секунд и текущую видимую книгу. Они не доказывают, что сделки происходили именно на запомненном уровне, и не раскрывают намерения участников. Удаление «потока/книги» убирает совместно шесть признаков, включая активность.</p>
<h2>Два реальных решения, без выбора удачных исходов</h2><p>Выбрано самое раннее принятое решение недельного селектора в каждом контракте. Сначала видны только прошлый bid и состояние модели. Исход можно раскрыть отдельно.</p><div id="scenes"></div>
<h2>Более быстрая обратная связь</h2><p>Вторая исследовательская стадия корректирует прогнозы внутри дня после завершения эпизодов. Проверены периоды полураспада 2 и 8 часов, общая и локальная коррекция. Требуются хотя бы 12 единиц взвешенной поддержки; при смене контракта история сбрасывается. Это не 12 независимых экспериментов.</p><p>Локальная коррекция на 8 часах дала 50,6% проигрышей на 1 323 входах, средний bid-результат −$0,004/oz. Ускорение само по себе не решило задачу.</p>
<h2>Разорение и качество решения — разные результаты</h2><figure><a href="figures/risk.png"><img src="figures/risk.png" alt="Распределение бумажного депозита при случайном направлении и разных ставках"></a></figure>
<p>Модель с памятью 5 дней и фиксированной ставкой оставила $9 700 из $10 000. Увеличение после убытка 1→2→4 оставило $600 уже после 24 сделок. Но на тех же моментах входа случайное направление с тем же правилом потеряло 80% депозита в 87 из 199 повторов. Такое разорение не доказывает сильного лузера.</p>
<p>Ставка ограничена четырьмя контрактами; если денег не хватает на запрошенную ставку, автоматического уменьшения нет. Остаток $600 при требовании $800 — остановка по доступному капиталу, а не банкротство. Это учёт закрытых сделок, без биржевой маржи и внутрипозиционного margin call. Комиссия/BBO/задержка и зеркальная сторона рассчитаны отдельно в данных отчёта.</p>
<p>Даже зеркальная сторона варианта «10 дней, порог 0,20» после BBO, комиссии и задержки дала в среднем −$0,503/oz на 246 поддержанных эпизодах из 247; один исход исполнения неизвестен. Его чистый bid-убыток пока нельзя превратить в подтверждённую прибыль простым переворотом стороны.</p>
<div class="footer"><p><b>Что проверено:</b> native bid-пути и порядок первого достижения границ, доступность признаков, коэффициенты моделей, отложенное обучение, недельный выбор и бумажные счета. Независимая проверка: <a href="verification.json">verification.json</a>. Данные интерфейса: <a href="report_data.json">report_data.json</a>.</p><p>Исследовательский статус: устойчивый переносимый лузер не установлен. Следующий содержательный объект — срок жизни конкретной локальной ошибки и доступность информации о её прекращении, а не минимальный остаток депозита.</p></div></main>
<script>const D=__DATA__;const fmt=(v,n=3)=>v==null?'—':Number(v).toFixed(n);const name=k=>D.names[k]||k;
function render(){let rows=D.quality.filter(r=>r.target===target.value&&r.block===block.value);results.innerHTML=rows.map(r=>`<tr><td>${name(r.policy)}</td><td>${r.selected}</td><td>${r.decisive_accuracy==null?'—':fmt(r.decisive_accuracy*100,1)+'%'}</td><td>${fmt(r.day_reward)}</td><td>${fmt(r.day_reward_lo)}…${fmt(r.day_reward_hi)}</td><td>${fmt(r.mean_bid_per_oz)}</td><td>${fmt(r.opportunity_reward,4)}</td></tr>`).join('')}
target.onchange=block.onchange=render;render();week.innerHTML=D.weeks.map((r,i)=>`<option value="${i}">${r.week}</option>`).join('');
function renderWeek(){let r=D.weeks[Number(week.value)];let claim=r.choice==='WAIT'?'Среди кандидатов нет достаточно устойчивого основания для входа.':r.choice.startsWith('switch')?'Ожидается, что направление ошибки различается по состоянию: иногда входить против продолжения, иногда следом перед разворотом.':'Ожидается продолжение импульса в выбранных состояниях; лузер будет вставать против него.';weekcard.innerHTML=`<span class="tag">Решение до начала недели</span><p><b>${r.choice==='WAIT'?'WAIT':name(r.choice)}</b></p><p>${claim}</p><p>Прошлые принятые прогнозы: ${r.past_rows}; активных дней: ${r.past_active_days}; оценка после штрафа за разброс: ${fmt(r.past_score,4)}.</p><details><summary>Раскрыть последующее поведение рынка</summary><p>Входов ${r.current_selected}, потеря X первой ${r.current_losses}, выигрыш X первым ${r.current_gains}. Средний bid-результат: ${fmt(r.current_pnl)} $/oz.</p><p class="muted">Это проверка ожидания после факта, не информация для выбора в начале недели.</p></details>`}week.onchange=renderWeek;renderWeek();
scenes.innerHTML=D.scenes.map(r=>`<div class="card"><span class="tag">${r.symbol} · ${r.date} · ${r.time}</span><h3>${r.side}: ${r.mechanism}</h3><p>Рецепт ${name(r.recipe)}. Вероятность потери X: ${fmt(r.loss_probability*100,1)}%; выигрыша X: ${fmt(r.gain_probability*100,1)}%; разность ${fmt(r.score)}.</p><div class="scene-info"><div><strong>${fmt(r.net30_r,2)} R</strong>движение 30 секунд</div><div><strong>${fmt(r.volume30,0)}</strong>объём 30 секунд</div><div><strong>${fmt(r.flow,2)}</strong>дисбаланс к росту цены</div><div><strong>${fmt(r.memory_pass*100,0)}%</strong>проходы прошлых визитов</div><div><strong>${fmt(r.distance_r,2)} R</strong>расстояние до зоны</div></div><figure><a href="figures/scene_${r.index}_before.png"><img src="figures/scene_${r.index}_before.png" alt="Рынок до решения ${r.date}"></a></figure><details><summary>Раскрыть следующие 300 секунд</summary><figure><a href="figures/scene_${r.index}_after.png"><img loading="lazy" src="figures/scene_${r.index}_after.png" alt="Будущий путь и точка выхода ${r.date}"></a></figure><p>Награда лузеру: ${fmt(r.event_reward,0)}. Bid-результат: ${fmt(r.unit_pnl,2)} $/oz; выход через ${fmt(r.seconds,1)} секунды. Граница убытка показана розовым; прибыли — зелёным.</p></details></div>`).join('');
</script></html>'''


if __name__=='__main__': main()
