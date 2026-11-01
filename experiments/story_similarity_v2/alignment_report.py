"""Visual comparison of scheduled snapshots with exact-age historical prefixes."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from spectrum.io import sha
from experiments.scalper_view_audit_v1.audit import dump
from .run import ROOT,read
from .run_alignment import DEFAULT
from .report import trace

METHODS={'snapshot':'Периодические снимки','age_rhythm':'Одинаковый возраст текущей части',
         'age_book':'Одинаковый возраст + книга'}


def main(run):
    folder=ROOT/run;m=read(folder/'manifest.json');summary=read(folder/'summary.json')
    assert m['status']=='COMPLETE_DESCRIPTIVE_AGE_ALIGNMENT'
    if (folder/'report_receipt.json').exists():raise FileExistsError('Report already exists')
    for name,digest in m['artifacts'].items():assert sha(folder/name)==digest,name
    cases=read(folder/'atlas_index.json');figures=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    for clock in sorted({c['query_ns'] for c in cases}):
        fig,axes=plt.subplots(3,3,figsize=(18,10.5))
        group=sorted([c for c in cases if c['query_ns']==clock],key=lambda c:['fast','middle','slow'].index(c['scale']))
        for row,c in enumerate(group):
            qx,qy,qb,_=trace(folder,c,'query')
            others={k:trace(folder,c,k) for k in METHODS if c['methods'][k] is not None}
            span=max([qx[-1]]+[x[0][-1] for x in others.values()])
            prices=np.concatenate([qy]+[x[1] for x in others.values()]);lo=prices.min();hi=prices.max();pad=max(.15,(hi-lo)*.1)
            for col,(method,label) in enumerate(METHODS.items()):
                ax=axes[row,col]
                ax.step(qx,qy,where='post',color='#126da0',lw=1,label=f'Запрос: {qx[-1]:.1f} с')
                for b in qb:ax.axvline(b,color='#126da0',ls=':',lw=.7,alpha=.65)
                match=c['methods'][method]
                if match:
                    x,y,b,tokens=others[method]
                    ax.step(x,y,where='post',color='#ae5d16',lw=.85,label=f'Пример: {x[-1]:.1f} с')
                    for at in b:ax.axvline(at,color='#ae5d16',ls='--',lw=.7,alpha=.6)
                    age=float(tokens.elapsed_seconds.iloc[-1]);qage=qx[-1]-qb[-1]
                    text=f'Текущая часть {qage:.2f} / {age:.2f} с · форма {match["shape_rmse"]:.2f}'
                    ax.text(.02,.025,text,transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
                else:
                    reason='Недостаточно книги у запроса' if c['statuses'][method]['status']=='QUERY_BOOK_CONTEXT_INCOMPLETE' else 'Кандидаты не прошли допуски'
                    ax.text(.5,.5,'Аналог не выдан\n'+reason,transform=ax.transAxes,ha='center',va='center',fontsize=10,
                            bbox=dict(facecolor='#fff0d7',edgecolor='none'))
                ax.set_xlim(0,span);ax.set_ylim(lo-pad,hi+pad);ax.grid(alpha=.15)
                ax.set_title(label+' · '+c['scale'],fontsize=11)
                ax.set_xlabel('Физическое время от начала тройки, с',fontsize=9)
                if col==0:ax.set_ylabel('Изменение цены / A')
                ax.tick_params(labelsize=8);ax.legend(fontsize=7,loc='upper left')
        stamp=pd.Timestamp(clock,tz='UTC').strftime('%d.%m.%Y %H:%M UTC')
        fig.suptitle('Сходство после устранения сетки снимков\n'+stamp,fontsize=16)
        fig.text(.5,.012,'Синим — запрос, коричневым — прошлый пример. Оси одинаковы в строке. Изменилось только чтение незавершённого префикса; веса и допуски сохранены.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.035,1,.94]);name=pd.Timestamp(clock,tz='UTC').strftime('atlas_%H%M%S.png')
        fig.savefig(folder/name,dpi=150);plt.close(fig);figures.append(name)
    rows=''
    for r in summary['snapshot_baselines']+summary['methods']:
        if r['scope']!='previous' or r['method'] not in ('legacy','rhythm_gate','book_gate','age_rhythm','age_book'):continue
        label={'legacy':'Прежняя формула','rhythm_gate':'Ритм и целый путь: снимки','book_gate':'Ритм, путь и книга: снимки',**METHODS}[r['method']]
        rows+=f'<tr><td>{label}</td><td>{r["with_neighbor"]} / 36</td><td>{r["median_total_duration_factor"]:.3f}</td><td>{r["median_shape_rmse"]:.3f}</td><td>{r["median_mechanism_rmse"]:.3f}</td></tr>'
    gallery=''.join(f'<figure><a href="{n}"><img src="{n}" alt="Сравнение {n}"></a><figcaption>{n} · полный размер по нажатию</figcaption></figure>' for n in figures)
    doc=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Сходство историй: возраст и ритм</title>
<style>body{{margin:0;background:#f4f6f8;color:#213149;font:17px/1.6 system-ui}}main{{max-width:1350px;margin:auto;padding:30px 22px}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:35px}}p{{max-width:1150px}}.notice{{background:#fff0d7;padding:18px;border-radius:12px}}table{{border-collapse:collapse;background:white;font-size:15px;width:100%}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce3ec}}.scroll{{overflow:auto}}figure{{background:white;margin:24px 0;padding:8px}}img{{width:100%;height:auto}}figcaption{{font-size:13px}}a{{color:#086da3}}@media(max-width:600px){{main{{padding:18px 10px}}h1{{font-size:28px}}}}</style>
<main><p>STORY SIMILARITY · продолжение адаптера · 24–25 августа 2026</p><h1>Похожая история должна быть наблюдена в сопоставимый момент</h1>
<p>Сходство разложено на форму, амплитуду, внутренние колебания, структуру, ритм, целый путь и доступное состояние книги. Поиск читает две завершённые истории и текущую формирующуюся часть.</p>
<div class="notice">Исполнено и проверено как измерительный слой. Это открытая визуальная итерация на уже исследованных датах. Человеческая согласованность похожести, независимая выборка и информация о будущем денежном результате пока не установлены.</div>
<h2>Два дефекта, обнаруженных на графиках</h2><p>При нормировке каждой части отдельно терялось накопленное направление: в 12:00 slow к снижающейся тройке подбиралась растущая. Добавлены соединённый путь и ограничения существенного противоположного движения.</p>
<p>Следующая проверка обнаружила влияние расписания снимков. Хороший прежний пример длился 300 + 300 + 30 с, а запрос — 300 + 300 + примерно 14 с. Порог отношения длительностей каждой части отсекал его, хотя две завершённые части были сопоставимы.</p>
<p>Теперь отдельный вариант читает каждую прошлую историю ровно на возрасте текущего незавершённого отрезка. Проверяются все причинно доступные начала; история исключается, только если к тому историческому моменту уже опубликовано её завершение. Её последующее выживание до следующего снимка не требуется. Веса и допуски не менялись.</p>
<h2>Одинаковые 36 запросов к предыдущей сессии</h2><div class="scroll"><table><tr><th>Способ</th><th>С аналогом</th><th>Время, медиана ×</th><th>Форма, RMSE</th><th>Книга, RMSE</th></tr>{rows}</table></div>
<p>Эти величины входят в формулу поиска и служат диагностикой. Снижение ошибки не является независимой оценкой рыночного смысла. Новый вариант также меняет множество доступных префиксов; его эффект нельзя приписывать одним весам. Отсутствие книги у запроса остаётся отдельным состоянием.</p>
<h2>Визуальный аудит</h2>{gallery}
<h2>Проверки причинности и расчётов</h2><p>Проверено {summary['checks']['aligned_candidates']:,} доступных исторических префиксов: одинаковый возраст текущей части и окончание до начала всей запрашиваемой тройки. У {summary['checks']['scalar_prefixes']} примеров отдельно пересчитаны экстремумы, путь, смены направления и состояние завершённости по исходной ленте. Обрезка данных в полдень сохранила результаты в обоих источниках истории.</p>
<p>Синтетическая смена частоты публикации снимков не меняет набор аналогов и компоненты расстояния. Это проверка механики чтения истории, не доказательство независимости от самой событийной нарезки.</p>
<p>Весь дополнительный опыт: {summary['elapsed_seconds']:.1f} с. Это пакетный исследовательский поиск, не замер задержки торгового решения.</p>
<p><a href="../verified_v3/report.html">Первый опыт: пять определений и 20 настроек</a> · <a href="../verified_v3/diagnostics.json">Почему отсекался прежний аналог</a> · <a href="summary.json">Численные результаты</a> · <a href="matches.json">Все соседи</a> · <a href="config.json">Параметры</a> · <a href="manifest.json">Источники и хеши</a></p></main></html>'''
    (folder/'report.html').write_text(doc,encoding='utf-8',newline='\n')
    dump(folder/'report_receipt.json',dict(status='RENDERED_STATIC_AGE_COMPARISON',source_sha256=sha(__file__),
        input_manifest_sha256=sha(folder/'manifest.json'),artifacts={n:sha(folder/n) for n in figures+['report.html']},browser_rendered=False))
    print(dict(report=str(folder/'report.html'),figures=figures))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=DEFAULT);main(p.parse_args().run)
