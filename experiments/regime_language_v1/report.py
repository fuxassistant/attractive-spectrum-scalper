"""Native time-at-price maps alongside multiple published VWAP coordinates."""
import argparse
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from spectrum.io import sha
from experiments.scalper_view_audit_v1.audit import dump
from .run import ROOT,DEFAULT,read
from .measures import TimeAtPrice,VwapContext,NS

LABELS={'day':'Сессия','combined':'Две сессии','previous':'Прошлая сессия','rolling_60m':'60 минут','rolling_240m':'240 минут'}
COLORS={'day':'#1584b0','combined':'#ce7b28','previous':'#8050a0','rolling_60m':'#26975a','rolling_240m':'#c74762'}


def scene(folder,r,prior):
    case=r['case_id'];left=r['start_ns'];right=r['available_ns']
    q=pd.read_parquet(folder/(case+'_quotes.parquet'));c=pd.read_parquet(folder/(case+'_context.parquet'))
    clock=pd.Timestamp(right,tz='UTC');time_map=TimeAtPrice(q)
    profile=r['profile'];lows=np.asarray(profile['bin_low_nano'],np.int64);width=profile['width_nano']
    grid=np.arange(lows.min(),lows.max()+width,width,dtype=np.int64)
    boundaries=left+(right-left)*np.arange(25,dtype=np.int64)//24
    heat=np.zeros((len(grid),24))
    for i in range(24):
        p=time_map.profile(int(boundaries[i]),int(boundaries[i+1]))
        heat[np.searchsorted(grid,p['bin_low_nano']),i]=p['bin_seconds']/p['duration_seconds']
        if not p['known']:heat[:,i]=np.nan
    np.savez_compressed(folder/(case+'_heatmap.npz'),price_low_nano=grid,time_edges_ns=boundaries,
                        time_fraction=heat,width_nano=width)
    fig=plt.figure(figsize=(14.5,10));gs=fig.add_gridspec(2,2,height_ratios=[1,1.25],width_ratios=[2,1])
    macro=fig.add_subplot(gs[0,:]);mapax=fig.add_subplot(gs[1,0]);dist=fig.add_subplot(gs[1,1],sharey=mapax)
    # Physical endpoints appear only after their complete minute was published.
    ct=pd.to_datetime(c.t_ns,utc=True);macro.plot(ct,c.close.where(c.known),color='#273746',lw=1,label='Завершённая M1 цена')
    for name in ('day','combined','previous'):
        macro.plot(ct,c[name+'_vwap'].where(c[name+'_vwap__known']),color=COLORS[name],lw=1.3,
                   ls='--' if name=='previous' else '-',label='VWAP: '+LABELS[name])
    vc=VwapContext(c);rolling=[]
    for row in c.itertuples():rolling.append(vc.at(int(row.available_ns)+1)['anchors'].get('rolling_60m',{}).get('value'))
    macro.plot(ct,rolling,color=COLORS['rolling_60m'],lw=1.2,label='VWAP: 60 минут')
    a=r['anchor_relations']['rolling_240m']
    if a['known']:macro.scatter([clock],[a['value']],color=COLORS['rolling_240m'],marker='D',s=35,label='VWAP 240 минут: последний доступный')
    macro.axvspan(pd.Timestamp(left,tz='UTC'),clock,color='#ddcb89',alpha=.35,label='Исследуемая тройка')
    macro.set_ylabel('Цена');macro.grid(alpha=.15);macro.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=clock.tz))
    macro.legend(ncol=4,fontsize=8,loc='upper left');macro.set_title('Крупный контекст · опубликованные VWAP имеют собственные начало и горизонт',fontsize=11)
    x=(boundaries-left)/NS;y=np.r_[grid,grid[-1]+width]/NS
    cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#c7cbd0')
    im=mapax.pcolormesh(x,y,np.ma.masked_invalid(heat),cmap=cmap,vmin=0,vmax=1,shading='flat')
    mid=(q.bid_nano.to_numpy(np.int64)+q.ask_nano.to_numpy(np.int64))//2
    mapax.step((q.available_ns.to_numpy(np.int64)-left)/NS,mid/NS,where='post',color='#1c4667',lw=.65,alpha=.8)
    mapax.axvline((right-left)/NS/2,color='#34495e',ls='--',lw=.9)
    mapax.set_xlim(0,(right-left)/NS);mapax.set_ylim(y.min(),y.max())
    mapax.set_ylabel('Абсолютная цена');mapax.set_xlabel('Секунды от начала тройки')
    mapax.set_title('Время по цене · доля времени в каждой временной ячейке',fontsize=11)
    fig.colorbar(im,ax=mapax,fraction=.025,pad=.02,label='Доля времени')
    for part,label,color in [('first_half','Первая половина','#1584b0'),('second_half','Вторая половина','#c77726')]:
        h=profile[part];mass=np.asarray(h['bin_seconds'])/h['supported_seconds']
        dist.step(mass,(np.asarray(h['bin_low_nano'])+width/2)/NS,where='mid',color=color,lw=1.6,label=label)
        dist.axhline(h['time_mean'],color=color,ls=':',lw=1)
    pp=np.asarray(prior['bin_low_nano']);support=np.isin(pp,grid)
    dist.step(np.asarray(prior['bin_seconds'])[support]/prior['supported_seconds'],(pp[support]+width/2)/NS,
              where='mid',color='#808890',lw=1,label='Вчера: доля всей наблюдённой сессии')
    dist.set_xlabel('Доля времени');dist.grid(alpha=.15);dist.legend(fontsize=8,loc='best')
    dist.set_title('Области пребывания и их смещение',fontsize=11)
    summary=f'Ход {r["net_A"]:+.2f}A · сдвиг центра времени {profile["center_migration_A"]:+.2f}A · перекрытие половин {profile["half_profile_overlap"]:.2f}'
    fig.suptitle(clock.strftime('%d.%m.%Y %H:%M UTC')+' · медленная история\n'+summary,fontsize=14)
    fig.text(.5,.012,'Цвет означает время поддержанного midpoint, а не объём сделок. Линии VWAP взвешены объёмом принятых T. Продолжение после запроса не показано.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.93]);name=case+'.png';fig.savefig(folder/name,dpi=150);plt.close(fig)
    return name


def main(run):
    folder=ROOT/run;m=read(folder/'manifest.json');summary=read(folder/'summary.json')
    assert m['status']=='COMPLETE_DESCRIPTIVE_REGIME_PROBE'
    if (folder/'report_receipt.json').exists():raise FileExistsError('Sealed report exists')
    for name,value in m['artifacts'].items():assert sha(folder/name)==value,name
    rows=read(folder/'sentences.json');prior=read(folder/'previous_profile.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    figures=[scene(folder,r,prior) for r in rows if 'case_id' in r]
    table=''
    for r in rows:
        if 'case_id' not in r:continue
        state={'BELOW':'ниже','ABOVE':'выше','NEAR':'около','UNKNOWN':'неизвестно'}
        cells=''.join('<td>'+state[r['anchor_relations'][a]['state']]+'</td>' for a in LABELS)
        table+=f'<tr><td>{r["case_id"]}</td>{cells}<td>{r["profile"]["center_migration_A"]:+.3f}</td><td>{r["profile"]["half_profile_overlap"]:.3f}</td></tr>'
    gallery=''.join(f'<figure><a href="{n}"><img src="{n}" alt="Время по цене и VWAP"></a><figcaption>{n} · нажмите для полного размера</figcaption></figure>' for n in figures)
    doc=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Рыночные фразы: VWAP и время по цене</title>
<style>body{{margin:0;background:#f4f6f8;color:#213149;font:17px/1.6 system-ui}}main{{max-width:1350px;padding:30px 22px;margin:auto}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:36px}}p{{max-width:1150px}}.notice{{background:#fff0d7;padding:18px;border-radius:10px}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce3ec;text-align:left}}.scroll{{overflow:auto}}figure{{margin:25px 0;background:white;padding:7px}}img{{width:100%;height:auto}}figcaption{{font-size:13px}}a{{color:#086da3}}@media(max-width:600px){{main{{padding:18px 10px}}h1{{font-size:28px}}}}</style>
<main><p>REGIME_LANGUAGE_PROBE_V1 · описательный опыт</p><h1>Один участок имеет несколько одновременно наблюдаемых смыслов</h1>
<p>На 36 прежних тройках и 108 составляющих токенах посчитаны абсолютные профили времени, их изменение и положение относительно пяти VWAP. Сегментация движения сохранена. Это проверка состава будущего языка, а не уже обученный распознаватель режимов.</p>
<div class="notice">Первичный язык должен сохранять числа и неизвестность рядом с условными словами. Слово «ниже» относится к конкретному якорю. Слово «трение» не следует из одного большого пройденного пути. В этом опыте нет торговых исходов и новой проверки обобщения.</div>
<h2>Факты, отношения и последовательность</h2><p><b>Движение:</b> ход, путь и эффективность. <b>Место:</b> распределение секунд по абсолютной цене, центр времени, концентрация, профили первой и второй половин. <b>Контекст:</b> расстояние до VWAP сессии, прошлой сессии, двух сессий, 60 и 240 завершённых минут. <b>Изменение:</b> сдвиг центра пребывания и разложение Δ(цена−VWAP)=Δцена−ΔVWAP.</p>
<p>Карта не переезжает вслед за VWAP. Память относится к абсолютной цене; относительно линий вычисляются отдельные отношения. Временной профиль хранится отдельно от объёма сделок. Неполная поддержка остаётся неизвестностью.</p>
<h2>Три прежних участка под новым углом</h2><div class="scroll"><table><tr><th>Участок</th>{''.join('<th>VWAP '+html.escape(x)+'</th>' for x in LABELS.values())}<th>Сдвиг центра / A</th><th>Перекрытие половин</th></tr>{table}</table></div>
<p>«Около» означает расстояние не более 0.5A в этой начальной измерительной версии. A зависит от масштаба движения. В JSON сохранены непрерывные расстояния и расстояния в σ там, где дисперсия известна; слова не заменяют эти величины.</p>
{gallery}
<h2>Что обнаружено</h2><p>В 12:00 снижающийся участок находится ниже крупных сессионных средних, около часового VWAP и выше четырёхчасового. Один ярлык «медвежий режим» потерял бы различие горизонтов. Сдвиг центра времени отличается от изменения последней цены; профиль показывает, где рынок действительно провёл наблюдаемое время.</p>
<p>Прежнее правило большого пути при малом смещении сработало в {summary['term_counts'].get('HIGH_PATH_LOW_PROGRESS',0)} из 36 тройкок. На этих данных оно мало различает режимы. Native-путь включает все мелкие движения midpoint; называть такой результат доказанным трением или поглощением нельзя.</p>
<p>По начальному допуску 0.5A центр пребывания сместился вверх в {summary['term_counts'].get('OCCUPANCY_MIGRATES_UP',0)}, вниз в {summary['term_counts'].get('OCCUPANCY_MIGRATES_DOWN',0)}, остался в пределах допуска в {summary['term_counts'].get('OCCUPANCY_CENTER_STABLE',0)} случаях. Это описание этой подборки, не независимые наблюдения и не оценка точности классификации.</p>
<h2>Проверки и границы</h2><p>{summary['scalar_time_integrals']} отдельных скалярных интегралов по сырой ленте, {summary['context_prefix_checks']} проверок обрезки контекста и точное совпадение полуденной карты после удаления будущих котировок. Сдвиг сетки на 2.5 тика и ширина 10 вместо 5 сохранены как контроль: непрерывные центры не меняются, моды и концентрация могут меняться.</p>
<p>Построение заняло {summary['elapsed_seconds']:.1f} с. VWAP здесь обновляются по опубликованным полным минутам, возраст их данных сохранён. Совпадение нескольких якорей не означает несколько независимых свидетельств. Недельный и причинно выбранный событийный VWAP в этот опыт не включены.</p>
<p>Следующий этап первичной токенизации: отдельные события изменения профиля, ухода/возврата к адресуемой области и отношений к якорям; формирующиеся состояния и подтверждения должны сохранять разные часы. Словарь выше пока служит наблюдаемыми признаками, границы «предложений» остаются прежними.</p>
<p><a href="summary.json">Сводка</a> · <a href="tokens.json">108 токенов с числами</a> · <a href="sentences.json">36 троек и отношений</a> · <a href="previous_profile.json">Наблюдённый профиль прошлой сессии</a> · <a href="config.json">Контракт</a> · <a href="manifest.json">Источники и хеши</a></p></main></html>'''
    (folder/'report.html').write_text(doc,encoding='utf-8',newline='\n')
    outputs=figures+['report.html']+[r['case_id']+'_heatmap.npz' for r in rows if 'case_id' in r]
    dump(folder/'report_receipt.json',dict(status='RENDERED_STATIC_REGIME_PROBE',source_sha256=sha(__file__),
        manifest_sha256=sha(folder/'manifest.json'),artifacts={p:sha(folder/p) for p in outputs},browser_rendered=False))
    print(dict(report=str(folder/'report.html'),figures=figures))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=DEFAULT);main(p.parse_args().run)
