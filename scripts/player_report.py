"""Render saved experiment only; never fit, select or modify a model."""
from pathlib import Path
import argparse,json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectrum.player.data import Day
from spectrum.player.contract import PLANS,NS
from spectrum.ingest import save_json

plt.rcParams.update({'figure.dpi':125,'axes.grid':True,'grid.alpha':.18,'font.size':10,'text.parse_math':False})

def read_trades(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()

def extrema(y,bins=1800):
    step=max(1,len(y)//bins);ids=[]
    for a in range(0,len(y),step):
        z=y[a:a+step]
        ids.extend([a,a+int(np.argmin(z)),a+int(np.argmax(z)),a+len(z)-1])
    return np.unique(ids)

def render(root,run):
    root=Path(root).resolve();src=root/'data/research/player_v1'/run;out=root/'reports/player_v1'/run
    summary=json.loads((src/'summary.json').read_text());epochs=pd.read_csv(out/'epochs.csv')
    days={p.name:Day(p,verify=False) for w in ('week1','week2') for p in sorted((root/'data/derived/player_v1'/w).glob('2026*'))}
    fig,ax=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    ax[0,0].plot(epochs.epoch,epochs.train_sampled_net,label='Sampled train: sum of 4 days')
    ax[0,0].plot(epochs.epoch,epochs.validation_net,label='Greedy validation: Friday')
    ax[0,0].axhline(0,color='grey',lw=1);ax[0,0].legend();ax[0,0].set_ylabel('Observed net USD (includes censored proxy)')
    ax[0,1].plot(epochs.epoch,epochs.train_trades,label='Sampled train trades')
    ax[0,1].plot(epochs.epoch,epochs.validation_trades,label='Greedy validation trades');ax[0,1].legend()
    ax[1,0].plot(epochs.epoch,epochs.entropy,label='Policy entropy (legal decisions)')
    ax[1,0].axhline(np.log(7),color='grey',ls='--',label='7-way uniform max');ax[1,0].legend()
    ax[1,1].plot(epochs.epoch,epochs.train_censored,label='Train censored trades')
    ax[1,1].plot(epochs.epoch,epochs.validation_censored,label='Validation censored trades');ax[1,1].legend()
    for a in ax.flat:a.set_xlabel('Epoch');a.axvline(summary['freeze']['selected_epoch'],color='#dd7025',ls=':')
    fig.suptitle('LSTM PPO: behaviour across epochs | orange = selected checkpoint | no week-two tuning')
    fig.savefig(out/'epochs.png');plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(14,8),layout='constrained')
    economic=[]
    for label,color in [('selected_test','#cb513a'),('untrained_test','#666aaa'),('wait_test','#339377')]:
        base=1000.;peak=base;maxdd=0.;alltr=[]
        for k,m in enumerate(summary['results'][label]['days']):
            date=m['date'];d=days[date];q=np.load(src/label/date/'quote_equity.npy',mmap_mode='r');ii=extrema(q)
            cumulative=q+base;peaks=np.maximum.accumulate(np.maximum(cumulative,peak));maxdd=max(maxdd,float(np.max(peaks-cumulative)));peak=max(peak,float(cumulative.max()))
            x=k+(d.quote_t[ii]-int(d.meta['start_ns']))/(23*3600*NS)
            ax[0].plot(x,cumulative[ii],color=color,label=label if k==0 else None)
            ax[1].plot(x,(peaks-cumulative)[ii],color=color,label=label if k==0 else None)
            base+=m['net'];tr=read_trades(src/label/date/'trades.csv')
            if len(tr):alltr.append(tr)
        trades=pd.concat(alltr,ignore_index=True) if alltr else pd.DataFrame()
        economic.append(dict(policy=label,observed_net=base-1000,ending_observed_equity=base,week_drawdown=maxdd,
                             trades=len(trades),censored=int(trades.censored.sum()) if len(trades) else 0,
                             gross=float(trades.gross.sum()) if len(trades) else 0.,fees=len(trades)*.30,
                             uncensored_trade_net=float(trades.loc[trades.censored==0,'net'].sum()) if len(trades) else 0.))
    for a in ax:a.set_xticks(np.arange(5)+.5,['May 11','May 12','May 13','May 14','May 15']);a.legend()
    ax[0].set_ylabel('Observed liquidation equity, USD');ax[1].set_ylabel('Continuous week drawdown, USD')
    fig.suptitle('Frozen week two | initial $1,000 | 1 oz | $0.30 roundtrip | censoring prevents economic admission')
    fig.savefig(out/'equity.png');plt.close(fig);save_json(out/'economic_metrics.json',economic)
    alltr=[]
    for parent in sorted(src.iterdir()):
        if not parent.is_dir():continue
        for path in sorted(parent.glob('2026*/trades.csv')):
            t=read_trades(path)
            if len(t):t['simulation']=parent.name;t['date']=path.parent.name;alltr.append(t)
    ledger=pd.concat(alltr,ignore_index=True);ledger['entry_utc']=pd.to_datetime(ledger.entry_ns,utc=True).dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    ledger.to_csv(out/'all_trades.csv',index=False)
    # Model choices remain distinct from action feasibility and confidence.
    fig,axs=plt.subplots(5,1,figsize=(14,12),layout='constrained',sharex=True)
    for ax,m in zip(axs,summary['results']['selected_test']['days']):
        date=m['date'];d=days[date];p=np.load(src/'selected_test'/date/'policy_trace.npz');mask=p['mask'][:,1:].any(axis=1)
        hours=(d.decision_t-int(d.meta['start_ns']))/NS/3600
        for col,label in [(0,'P(WAIT)'),(3,'P(LONG/wide)')]:
            value=p['probability'][:,col].copy();value[~mask]=np.nan;ax.plot(hours,value,label=label)
        dec=pd.read_parquet(src/'selected_test'/date/'decisions.parquet');entries=dec.action.to_numpy()>0
        ax.scatter(hours[entries],np.full(entries.sum(),.28),marker='^',color='#cc5533',s=20,label='Actual entry decision')
        ax.axvspan(0,9,color='grey',alpha=.1);ax.set_ylim(0,.5);ax.set_title(f'{date}: {m["trades"]} trades, {m["censored"]} censored');ax.legend(loc='upper right',ncol=3)
    axs[-1].set_xlabel('Hours since previous 22:00 UTC opening; blank = no legal entry')
    fig.suptitle('Frozen policy: small probability differences can change greedy actions; these probabilities are not calibrated profit forecasts')
    fig.savefig(out/'policy.png');plt.close(fig)
    pool=ledger[ledger.simulation.isin(['selected_test','train_sampled_epoch_001','train_sampled_epoch_030'])]
    rng=np.random.default_rng(61015);chosen=list(rng.choice(pool.index,size=min(6,len(pool)),replace=False))
    chosen=list(dict.fromkeys(chosen+[pool.net.idxmin(),pool.net.idxmax()]))
    receipts=[]
    for n,i in enumerate(chosen):
        tr=ledger.loc[i];d=days[tr.date];center=int(tr.entry_ns);lo=center-10*60*NS;hi=int(tr.exit_ns)+10*60*NS
        q0,q1=np.searchsorted(d.quote_t,[lo,hi]);indices=np.arange(q0,q1);mid=(d.bid[q0:q1].astype(float)+d.ask[q0:q1])/2e9
        bad=~d.valid[q0:q1];bad=bad.copy();bad[1:]|=np.diff(d.quote_t[q0:q1])>5*NS;mid[bad]=np.nan
        # Keep gaps in plotted support, even when quote density is reduced.
        step=max(1,len(indices)//8000);take=np.unique(np.r_[np.arange(0,len(indices),step),np.flatnonzero(bad),np.minimum(np.flatnonzero(bad)+1,len(indices)-1)])
        fig,ax=plt.subplots(3,1,figsize=(14,8),layout='constrained',sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
        x=(d.quote_t[indices[take]]-center)/NS/60
        ax[0].plot(x,mid[take],color='#34557c',lw=.8,label='Supported MBO midpoint (quote sampled)')
        side=int(tr.side);entry=float(tr.entry_nano)/1e9;exit=float(tr.exit_nano)/1e9;plan=PLANS[int(tr.plan)]
        ax[0].scatter([0], [entry],marker='^' if side>0 else 'v',color='#208657',s=70,label='Actual bid/ask entry')
        ax[0].scatter([(int(tr.exit_ns)-center)/NS/60],[exit],marker='x',color='#bb4444',s=70,label='Actual exit')
        ax[0].axhline(entry+side*plan[0]*.1,color='#208657',ls=':',label='Take threshold');ax[0].axhline(entry-side*plan[1]*.1,color='#bb4444',ls=':',label='Stop threshold')
        ax[0].set_ylabel('USD / oz');ax[0].legend(ncol=3,fontsize=8)
        b=(d.decision_t>=lo)&(d.decision_t<=hi);bx=(d.decision_t[b]-center)/NS/60
        for name in ('displacement_30s','spread_ticks'):
            j=d.meta['feature_names'].index(name);known=d.x[b,d.meta['feature_names'].index(name+'__known')].astype(bool)
            y=d.x[b,j].copy();y[~known]=np.nan;ax[1].plot(bx,y,'.-',label=name)
        for name in ('trade_buy_30s','trade_sell_30s'):
            j=d.meta['feature_names'].index(name);known=d.x[b,d.meta['feature_names'].index(name+'__known')].astype(bool)
            y=d.x[b,j].copy();y[~known]=np.nan;ax[2].plot(bx,y,'.-',label=name)
        ax[1].set_ylabel('Ticks');ax[2].set_ylabel('Volume');ax[2].set_xlabel('Minutes relative to actual entry; feature dots available only at decision proofs')
        for a in ax[1:]:a.legend(ncol=2)
        fig.suptitle(f'{tr.simulation} | {tr.date} | {"LONG" if side>0 else "SHORT"} plan {int(tr.plan)} | net ${tr.net:+.2f} | {tr.reason} | censored={int(tr.censored)}')
        filename=f'scene_{n:02d}.png';fig.savefig(out/filename);plt.close(fig)
        receipts.append(dict(image=filename,ledger_row=int(i),simulation=tr.simulation,date=tr.date,entry_ns=str(int(tr.entry_ns)),censored=int(tr.censored)))
    save_json(out/'scene_selection.json',dict(seed=61015,method='Six random trades from selected test and sampled first/final epochs, plus worst and best in same pool',scenes=receipts))
    # Table uses formatted UTC and decimal-string clocks; never JS float nanoseconds.
    rows=ledger[['simulation','date','entry_utc','side','plan','net','reason','censored']].round({'net':2})
    rows.to_csv(out/'trade_browser.csv',index=False)
    table=rows.to_html(index=False,classes='ledger',table_id='trades',border=0)
    cards=''.join(f'<option value="{html.escape(x)}">{html.escape(x)}</option>' for x in sorted(rows.simulation.unique()))
    scenes=''.join(f'<figure><img loading="lazy" src="{x["image"]}"><figcaption>{html.escape(x["simulation"])} — {x["date"]}</figcaption></figure>' for x in receipts)
    doc='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>LSTM player — исследовательский gym</title>
<style>body{font:16px system-ui;max-width:1450px;margin:auto;padding:30px;background:#f5f7fb;color:#192a3b}h1{font-size:30px}img{width:100%;background:white}figure{margin:25px 0}section{background:white;padding:22px;margin:20px 0;border-radius:12px}p{max-width:1050px;line-height:1.55}.warn{border-left:5px solid #cc633d}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:7px;text-align:left;border-bottom:1px solid #ddd}th{position:sticky;top:0;background:#e7edf6}.scroll{max-height:520px;overflow:auto}select,input{padding:10px;margin:8px}a{color:#245c9f}</style>
<h1>Первый игрок: LSTM → WAIT / LONG / SHORT + план выхода</h1>
<section class="warn"><b>Прибыльные повторяемые состояния пока не найдены.</b><p>Чекпойнт 28 выбран на пятнице первой недели: 0 сделок, $0. На второй неделе модель снова открывала позиции. Наблюдаемый результат −$46.40; 37 сделок, 17 с потерей достоверности исполнения. Это диагностический proxy, а не доказанный торговый PnL. WAIT сохранил капитал. Меньший убыток, чем у необученной сети, не доказывает преимущество.</p></section>
<section><b>Управляемые условия</b><p>Одна позиция 1 oz, старт $1,000, комиссия $0.30 один раз при входе, spread через bid/ask. Решение после M1; исполнение на первой допустимой публикации строго позже 100 ms. Контекст с 22:00 UTC предыдущего дня; входы с 08:00 Europe/London до 17:00 America/New_York. Ежедневный перерыв 21–22 UTC, выходной не заполняется свечами. Перед закрытием рынка обязательная ликвидация; дневной стоп по просадке $30. Планы: take/stop 12/8 ticks и 120 s; 24/12 и 300 s; 40/20 и 600 s. Tick=$0.10/oz.</p>
<p>4–7 мая: обновления PPO по собственным действиям; 8 мая: выбор эпохи; 11–15 мая: замороженная проверка. Нормализация только по 4–7 мая. Вторая неделя ранее была видна при проверке ядра, поэтому это не полностью слепой исследовательский holdout.</p>
<p><a href="protocol.json">Протокол</a> · <a href="frozen_before_evaluation.json">Фиксация до теста</a> · <a href="summary.json">Результаты по дням</a> · <a href="epochs.csv">Эпохи CSV</a> · <a href="economic_metrics.json">Equity и комиссии</a> · <a href="all_trades.csv">Все сделки CSV</a> · <a href="INDEPENDENT_AUDIT_SOL61_RU.md">Независимый аудит Sol 6.1</a></p></section>
<figure><img src="epochs.png"></figure><figure><img src="equity.png"></figure><figure><img src="policy.png"></figure>
<section><h2>Исследовать сделки</h2><p>Фильтры работают локально. Censored=1: денежный результат наблюдаемого закрытия сохранён, но заданный путь исполнения нельзя считать полностью проверенным.</p>
<select id="sim"><option value="">Все симуляции</option>CARDS</select><input id="query" placeholder="Дата, take, stop, support_loss…"><label><input type="checkbox" id="censor">Только censored</label><span id="count"></span><div class="scroll">TABLE</div></section>
<h2>Воспроизводимая выборка рынка, исполнения и признаков</h2><p>Шесть случайных сделок (seed 61015), плюс лучшая и худшая в указанном пуле. Эти изображения — диагностика, а не подбор сигналов.</p>SCENES
<script>const rows=[...document.querySelectorAll('#trades tbody tr')];function filter(){let n=0;for(const r of rows){const c=r.children;const yes=(!sim.value||c[0].textContent===sim.value)&&r.textContent.toLowerCase().includes(query.value.toLowerCase())&&(!censor.checked||c[7].textContent==='1');r.hidden=!yes;if(yes)n++;}count.textContent=n+' сделок';}for(const e of [sim,query,censor])e.addEventListener('input',filter);filter();</script></html>'''
    selected=summary['results']['selected_test'];untrained=summary['results']['untrained_test'];selected_epoch=summary['freeze']['selected_epoch']
    vp=np.load(src/f'validation_epoch_{selected_epoch:03d}'/'2026-05-08'/'policy_trace.npz');legal=vp['mask'][:,1:].any(axis=1)
    pw=float(vp['probability'][legal,0].mean())
    dynamic=f'<b>Исследовательский результат; прибыльные повторяемые состояния не установлены.</b><p>Эпоха {selected_epoch} выбрана по первой неделе; selection net ${summary["freeze"]["selection_net"]:+.2f}. Вторая неделя: {selected["trades"]} сделок, наблюдаемый net ${selected["net"]:+.2f}, censored {selected["censored"]}. Необученная сеть: {untrained["trades"]} сделок, ${untrained["net"]:+.2f}. WAIT $0. При наличии censor сумма является наблюдаемым proxy, а не подтверждённым торговым PnL.</p><p>На допустимых состояниях selection Friday средняя P(WAIT)={pw:.5f}; сумма шести торговых вероятностей={1-pw:.5f}. Отсутствие greedy сделок не означает уверенного распознавания отсутствия возможности.</p>'
    if run=='run_001':dynamic+='<p>Независимый аудит установил: все17 censored selected-test сделок вызваны паузами между публикациями более5 s; invalid/crossed book, source age и epoch не были причиной. Тишина и потеря source proof требуют отдельного исследования; правило этого опыта не менялось задним числом.</p>'
    import re
    doc=re.sub(r'<section class="warn">.*?</section>','<section class="warn">'+dynamic+'</section>',doc,flags=re.S)
    doc=doc.replace("const rows=[...", "const sim=document.getElementById('sim'),query=document.getElementById('query'),censor=document.getElementById('censor'),count=document.getElementById('count');const rows=[...")
    (out/'index.html').write_text(doc.replace('CARDS',cards).replace('TABLE',table).replace('SCENES',scenes),encoding='utf-8')
    print(json.dumps(dict(report=str(out/'index.html'),trades=len(ledger),scenes=len(receipts),economic=economic)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--run',default='run_001');a=p.parse_args();render(a.root,a.run)
