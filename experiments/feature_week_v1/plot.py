"""Standalone audit figures. Explicit UTC axes, unknown gaps and known masks."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from profile import OUTPUT, WEEK, DATES, NS, MINUTE
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap

RUN=OUTPUT/'serial_v2'
OUT=OUTPUT/'audit'/'figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.dpi':140,'savefig.dpi':160,'font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'font.family':'DejaVu Sans'})
BLUE='#1768ac';ORANGE='#d76425';GREEN='#24845b';RED='#b63c44'


def stamp(series): return pd.to_datetime(series,utc=True)
def save(fig,name):
    fig.savefig(OUT/name,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def format_time(ax):
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz='UTC'))
    ax.set_xlabel('UTC')
def frame(date,cols=None):return pd.read_parquet(RUN/'days'/date/'features.parquet',columns=cols)


def overview():
    m=pd.read_parquet(RUN/'minutes.parquet')
    m=m[(m.t_ns>=pd.Timestamp('2026-08-23 22:00',tz='UTC').value)&(m.t_ns<=pd.Timestamp('2026-08-28 21:00',tz='UTC').value)]
    index=pd.date_range('2026-08-23 22:00','2026-08-28 21:00',freq='min',tz='UTC')
    m=m.assign(time=stamp(m.t_ns)).set_index('time').reindex(index)
    fig,ax=plt.subplots(4,1,figsize=(15,10),sharex=True,gridspec_kw={'height_ratios':[2.1,1,1.1,1.5]},layout='constrained')
    ax[0].plot(index,m.close.where(m.known.eq(True)),color=BLUE,lw=.8)
    ax[0].set_ylabel('GCZ6, USD/oz')
    ax[0].set_title('Тестовая неделя 24–28 августа 2026 · воскресное открытие включено',loc='left',weight='bold')
    ax[1].plot(index,m.volume.where(m.known.eq(True)),color=BLUE,lw=.7)
    ax[1].set_ylabel('T-объём / минуту\nконтракты')
    for name,color in [('stft_2_8m',BLUE),('stft_8_32m',ORANGE),('stft_32_64m',GREEN)]:
        ax[2].plot(index,m[name],label=name,color=color,lw=.7)
    for fraction,color in ((48/64,BLUE),(12/64,ORANGE),(2/64,GREEN)):
        ax[2].axhline(fraction,color=color,lw=.6,ls='--',alpha=.7)
    ax[2].text(.99,.95,'Пунктир: доля FFT-бинов при равномерной мощности',
               ha='right',va='top',transform=ax[2].transAxes,fontsize=8)
    ax[2].set_ylabel('Доля энергии\n128 минут истории');ax[2].legend(ncol=3,loc='upper left',fontsize=8)
    names=['spread_ticks','trade_buy_30s','fft_mid_change_128s_power','vwap_exact_60s_center','vwap_exact_900s_center','vwap_exact_3600s_center']
    parts=[frame(d,['t_ns']+[n+'__known' for n in names]) for d in DATES[1:]]
    x=pd.concat(parts);x=x.assign(time=stamp(x.t_ns)).set_index('time')
    agg=x[[n+'__known' for n in names]].resample('5min').mean().reindex(pd.date_range(index[0],index[-1],freq='5min'))
    extent=[mdates.date2num(index[0]),mdates.date2num(index[-1]),len(names)-.5,-.5]
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#ddd')
    image=ax[3].imshow(agg.to_numpy().T,aspect='auto',interpolation='nearest',extent=extent,cmap=cmap,vmin=0,vmax=1)
    ax[3].set_yticks(range(len(names)),['BBO / spread','Trade flow 30s','FFT 128s','VWAP 60s','VWAP 900s','VWAP 3600s'],fontsize=8)
    fig.colorbar(image,ax=ax[3],label='Доля known в 5 мин',pad=.01)
    for a in ax:
        for d in WEEK[:-1]:
            a.axvspan(pd.Timestamp(d+' 21:00',tz='UTC'),pd.Timestamp(d+' 22:00',tz='UTC'),color='#ddd',alpha=.7)
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%a %d\n%H:%M',tz='UTC'))
    ax[-1].set_xlabel('UTC · серые вертикальные области — плановый перерыв · пустые значения не интерполированы')
    save(fig,'01_week.png')


def detail():
    date='2026-08-28'
    lo,hi=[pd.Timestamp(date+' '+t,tz='UTC').value for t in ('13:00','15:00')]
    x=frame(date);x=x[(x.t_ns>=lo)&(x.t_ns<=hi)]
    f=pd.read_parquet(RUN/'days'/date/'fast.parquet');f=f[(f.t_ns>=lo)&(f.t_ns<=hi)]
    t=stamp(f.t_ns);tx=stamp(x.t_ns)
    fig,ax=plt.subplots(5,1,figsize=(15,11),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2.2,1,1,1,1]})
    ax[0].plot(t,f.mid.where(f.known),color='#26374a',lw=.8,label='Mid, 1s')
    for name,label,color in [('vwap_exact_300s_center','VWAP exact 300s',BLUE),('vwap_proxy_300s_center','VWAP proxy 300s',ORANGE)]:
        ax[0].plot(tx,x[name].where(x[name+'__known']),color=color,lw=1.1,label=label)
    known=x.vwap_exact_300s_center__known
    ax[0].fill_between(tx,x.vwap_exact_300s_lower1.where(known),x.vwap_exact_300s_upper1.where(known),color=BLUE,alpha=.1,label='±1 σ цен сделок')
    ax[0].legend(loc='upper left',fontsize=8,ncol=2);ax[0].set_ylabel('USD/oz')
    ax[0].set_title('Содержательность: цена, точные сделки и изменения книги · фиксированное окно пятницы',loc='left',weight='bold')
    ax[1].plot(t,f.trade_buy30,color=GREEN,label='Покупки T, 30s',lw=.8)
    ax[1].plot(t,-f.trade_sell30,color=RED,label='Продажи T, 30s',lw=.8)
    ax[1].axhline(0,color='#888',lw=.5);ax[1].set_ylabel('Контракты');ax[1].legend(ncol=2,fontsize=8)
    ax[2].plot(t,f.depth_imbalance,color=BLUE,lw=.7,label='Баланс видимой глубины ±12 тиков')
    ax[2].set_ylim(-1.05,1.05);ax[2].set_ylabel('[-1; 1]');ax[2].legend(fontsize=8)
    ax[3].plot(t,f.bid_near_fill30+f.ask_near_fill30,label='Связанные исполнения',color=GREEN,lw=.8)
    ax[3].plot(t,f.bid_near_withdrawal30+f.ask_near_withdrawal30,label='Несвязанные снятия',color=ORANGE,lw=.8)
    ax[3].plot(t,f.bid_near_refill30+f.ask_near_refill30,label='Refill proxy',color=BLUE,lw=.8)
    ax[3].set_ylabel('Объём / 30s');ax[3].legend(ncol=3,fontsize=8)
    latency=(f.available_ns-f.t_ns)/1e6
    ax[4].plot(t,np.maximum(latency,.01),color=BLUE,lw=.6)
    ax[4].axhline(1000,color=RED,lw=.8,label='Порог свежего origin = 1s')
    ax[4].set_yscale('log');ax[4].set_ylabel('Публикация\nмс после конца 1s');ax[4].legend(fontsize=8)
    format_time(ax[-1]);save(fig,'02_market_detail.png')


def boundaries():
    names=['spread_ticks','vwap_exact_60s_center','vwap_exact_900s_center','vwap_exact_3600s_center']
    x=pd.concat([frame(d,['t_ns']+[n+'__known' for n in names]) for d in WEEK[:2]])
    fig,ax=plt.subplots(2,1,figsize=(15,7),layout='constrained')
    ranges=[('2026-08-24 23:40','2026-08-25 01:20','Граница UTC-файлов: исходный reset + незаписанный хвост'),
            ('2026-08-24 20:40','2026-08-24 23:20','Плановый часовой перерыв: короткая история прогревается заново')]
    for a,(lo,hi,title) in zip(ax,ranges):
        idx=pd.date_range(lo,hi,freq='min',tz='UTC')
        part=x[(x.t_ns>=idx[0].value)&(x.t_ns<=idx[-1].value)].copy()
        part=part.assign(time=stamp(part.t_ns)).set_index('time')
        values=part[[n+'__known' for n in names]].resample('1min').mean().reindex(idx)
        cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#ccc')
        im=a.imshow(values.to_numpy().T,aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=1,
             extent=[mdates.date2num(idx[0]),mdates.date2num(idx[-1]),3.5,-.5])
        a.set_yticks(range(4),['BBO / spread','VWAP 60s','VWAP 900s','VWAP 3600s'])
        a.set_title(title,loc='left');format_time(a)
        fig.colorbar(im,ax=a,label='Доля known в минуте',pad=.01)
    save(fig,'03_boundaries.png')


def cadence_and_speed():
    c=pd.read_parquet(OUTPUT/'audit'/'cadence.parquet').groupby('name')[['rows','known','pairs','changes']].sum()
    names=['spread_ticks','imbalance_l1','depth_balance_12','displacement_30s','trade_buy_1s','trade_buy_30s',
           'fft_mid_change_128s_power','fft_signed_trade_128s_power','vwap_exact_60s_center','vwap_exact_900s_center','vwap_exact_3600s_center']
    c=c.loc[names]
    fig,ax=plt.subplots(1,2,figsize=(15,7),layout='constrained',gridspec_kw={'width_ratios':[1.8,1]})
    y=np.arange(len(c));ax[0].barh(y-.18,100*c.known/c.rows,height=.32,label='Known / все строки',color=BLUE)
    ax[0].barh(y+.18,100*c.changes/c.pairs,height=.32,label='Изменился / пары known',color=ORANGE)
    ax[0].set_yticks(y,names,fontsize=9);ax[0].invert_yaxis();ax[0].set_xlim(0,102);ax[0].set_xlabel('%')
    ax[0].set_title('Частота публикации 1 Hz ≠ частота изменения',loc='left',weight='bold');ax[0].legend(fontsize=8)
    manifests=[json.loads((OUTPUT/r/'manifest.json').read_text()) for r in ('serial_v2','parallel2')]
    days=[r['days_elapsed_seconds'] for r in manifests];contexts=[r['context']['elapsed_seconds'] for r in manifests]
    ax[1].bar(['1 процесс','2 процесса'],days,color=BLUE,label='Дневные признаки и fast')
    ax[1].bar(['1 процесс','2 процесса'],contexts,bottom=days,color=ORANGE,label='Общий контекст и проверка loader')
    for i,r in enumerate(manifests):ax[1].text(i,r['total_elapsed_seconds']+2,f"{r['total_elapsed_seconds']:.1f} s",ha='center')
    ax[1].set_ylim(0,max(r['total_elapsed_seconds'] for r in manifests)*1.15)
    ax[1].set_ylabel('Секунды');ax[1].set_title('Неделя + 2 контекстных дня\nиз готовых исходных фактов',loc='left')
    ax[1].legend(fontsize=8,loc='upper right')
    save(fig,'04_cadence_speed.png')


if __name__=='__main__':
    overview();detail();boundaries();cadence_and_speed()
    print(str(OUT))
