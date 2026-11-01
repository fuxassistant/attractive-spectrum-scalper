"""Evidence-bound report and visual atlas for the bounded representation audit."""
from pathlib import Path
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .audit import ROOT,HERE,digest,dump,describe,NS
from spectrum.representation.session import session_bounds


def main():
    cfg=json.loads((HERE/'config.json').read_text());out=ROOT/cfg['output']
    source_manifest=json.loads((out/'manifest.json').read_text())
    assert source_manifest['status']=='COMPLETE_DESCRIPTIVE_ONLY'
    assert not (out/'report_receipt.json').exists(),'Render to a new audit run after report finalization'
    for name,value in source_manifest['artifacts'].items():assert digest(out/name)==value,name
    summary=json.loads((out/'summary.json').read_text());cases=json.loads((out/'atlas_index.json').read_text())
    inputs={str((out/'manifest.json').relative_to(ROOT)):digest(out/'manifest.json')}
    # Read the actual common model/policy population, separate from all zones.
    conditional=ROOT/'research_results/zone_conditional_v1/verified_v1'
    cm=json.loads((conditional/'manifest.json').read_text())
    for name in ('policy_clock.parquet','dataset.parquet'):
        assert digest(conditional/name)==cm['artifacts'][name]
        inputs[str((conditional/name).relative_to(ROOT))]=cm['artifacts'][name]
    inputs[str((conditional/'manifest.json').relative_to(ROOT))]=digest(conditional/'manifest.json')
    c=pd.read_parquet(conditional/'policy_clock.parquet').merge(pd.read_parquet(conditional/'dataset.parquet',columns=['candidate_id','episode_available_ns']),on='candidate_id',validate='one_to_one')
    common=[];stores=[]
    for date in cfg['dates']:
        g=c[c.date.eq(date)]
        common.append(dict(date=date,clocks=len(g),departure_to_common_policy_clock_s=describe((g.t_ns-g.episode_available_ns)/NS)))
        folder=ROOT/cfg['feature_week']/'stores'/date;m=json.loads((folder/'manifest.json').read_text())
        tables={}
        for name in ('events.parquet','context.parquet'):
            assert digest(folder/name)==m['files'][name]
            inputs[str((folder/name).relative_to(ROOT))]=m['files'][name];tables[name]=pd.read_parquet(folder/name)
        inputs[str((folder/'manifest.json').relative_to(ROOT))]=digest(folder/'manifest.json')
        start,end=session_bounds(date);ev=tables['events.parquet'];ctx=tables['context.parquet']
        stores.append(dict(date=date,previous_date=m['previous_date'],session_start_ns=start,
            previous_context_rows=int((ctx.t_ns<=start).sum()),previous_known_context_rows=int(((ctx.t_ns<=start)&ctx.known).sum()),
            stored_event_rows=len(ev),stored_previous_session_event_rows=int((ev.available_ns<start).sum()),
            event_kinds=ev.kind.value_counts().to_dict(),limitation='Session context is present, but this published event table contains only current-session events; prior-session price-level episode bank is not joined.'))
    archive=ROOT/'.local_archive/research-2026-10-26/reports/player_v3/sol61_audit/series_verdict.json'
    verdict=json.loads(archive.read_text(encoding='utf-8'));inputs[str(archive.relative_to(ROOT))]=digest(archive)
    architecture_files=['.local_archive/research-2026-10-26/docs/PLAYER_V3_DATA_SCHEMA.md',
        '.local_archive/research-2026-10-26/spectrum/player_v3/policy.py',
        'experiments/feature_week_v2/build.py','spectrum/representation/windows.py',
        'spectrum/representation/representation.py','spectrum/representation/zones.py',
        'experiments/zone_conditional_v1/features.py','experiments/zone_episode_lab/candidates.py',
        'spectrum/features/families/landmarks.py','experiments/scalper_view_audit_v1/observation_contract.json']
    for name in architecture_files:inputs[name]=digest(ROOT/name)
    supplemental=dict(common_clocks=common,session_store_content=stores,
        player_v3=dict(source='Historical saved series_001 review; not a new training rerun',
            selected_greedy_complete_wait=verdict['confirmed']['selected_greedy_complete_wait'],
            paired_memory_probes=verdict['confirmed']['paired_memory_probes'],
            paired_action_changes=verdict['confirmed']['paired_action_changes'],
            training_populations_complete=verdict['confirmed']['training_populations_complete'],
            training_populations_censored=verdict['confirmed']['training_populations_censored'],
            event_vocabulary='5s burst classes fresh_impulse/second_in_cluster/recurrent_*; not new price-level zone episodes',
            model='fast/slow LSTMs with attention retrieval; event values pooled by content+age attention, no event-to-event recurrent transition in this branch',
            conclusion='No demonstrated economic benefit in that bounded setup; does not prove no sequential or joint information in the new map.'),input_sha256=inputs)
    dump(out/'adapter_audit.json',supplemental)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    case_images=[]
    for page in range(2):
        fig,axes=plt.subplots(3,2,figsize=(14,11));fig.patch.set_facecolor('#fafbf7')
        for ax,case in zip(axes.flat,cases[page*6:page*6+6]):
            q=pd.read_parquet(out/(case['case_id']+'_quotes.parquet'));ev=pd.read_parquet(out/(case['case_id']+'_events.parquet'))
            at=case['decision_ns'];x=(q.available_ns.to_numpy(np.int64)-at)/NS;y=q.mid_nano.to_numpy(np.int64)/NS
            good=q.valid.to_numpy(bool)&(q.ask_nano>q.bid_nano).to_numpy()&q.epoch.ge(0).to_numpy()
            y=np.where(good,y,np.nan)
            # Remove identical adjacent quote values only for drawing; no time
            # aggregation is used by the measurements or stored scene.
            keep=np.r_[True,(np.diff(q.mid_nano)!=0)|(~good[1:])|(~good[:-1])|(np.diff(q.epoch)!=0)]
            keep[-1]=True
            ax.step(x[keep],y[keep],where='post',color='#176e7d',lw=1.0)
            low=case['zone_low_nano']/NS
            ax.axhspan(low,low+1,color='#78a486',alpha=.20)
            for z in (low-1,low+2):ax.axhline(z,color='#83918d',ls='--',lw=.7)
            ax.axvspan(0,30,color='#777777',alpha=.10);ax.axvline(0,color='#263e46',lw=1.2)
            for kind,color,marker in [('GATE_START','#126db2','^'),('GATE_ABORT','#c17b24','x'),('DEPARTURE_CONFIRMED','#b93f55','s')]:
                t=ev[(ev.event_type==kind)&(ev.episode_id==case['episode_id'])]
                ax.scatter((t.available_ns-at)/NS,t.mid_nano/NS,s=24,c=color,marker=marker,zorder=5)
            label=pd.Timestamp(at,tz='UTC').strftime('%d.%m %H:%M:%S')
            ax.set_title(f'{label} UTC · зона [{low:.1f}, {low+1:.1f})',loc='left',fontsize=10)
            ax.set_xlabel('Секунды от первого GATE_START');ax.set_ylabel('USD/oz');ax.set_xlim(-30,30);ax.grid(alpha=.12)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        fig.suptitle('Незавершённая попытка: что уже известно на вертикальной линии',fontsize=15,x=.04,ha='left')
        fig.text(.04,.015,'Зелёная полоса — ядро; пунктир — ворота ±$1. ▲ начало, × отмена, ■ подтверждение. Серое справа — последующая проверка, не X.',fontsize=10)
        fig.tight_layout(rect=[0,.04,1,.95]);name=f'atlas_{page+1}.png';fig.savefig(out/name,dpi=150);plt.close(fig);case_images.append(name)
    vr=[]
    for r in summary['variants']:
        vr.append(f'<tr><td>{r["date"]}</td><td>{r["variant"]}</td><td>{r["episodes_started"]}</td><td>{r["last_edge_to_confirmation_s"]["p50"]:.2f}</td><td>{r["signed_move_after_last_edge_per_oz"]["p50"]:.2f}</td></tr>')
    cr=[]
    for r in common:cr.append(f'<tr><td>{r["date"]}</td><td>{r["clocks"]}</td><td>{r["departure_to_common_policy_clock_s"]["p50"]:.2f}</td></tr>')
    intro='''<header><p class="eyebrow">30 октября 2026 · аудит представления</p><h1>Что должна видеть машина, чтобы работать с ходом рынка</h1>
<p class="lead">Предлагается совместить быстрый ответ цены и книги, упорядоченные эпизоды, память ценовых областей и контекст двух сессий.
Готовые паттерны полезны вместе с их измеримыми основаниями и текущей стадией формирования.</p>
<p>Выполнен ограниченный описательный опыт на 24–25 августа. Новые денежные исходы и торговые политики не рассчитывались. Данные уже исследованы ранее.</p></header>'''
    content=f'''<section><h2>Важные поправки к диагнозу</h2><ul>
<li>Ранние события уже сохраняются: EDGE_EXIT, GATE_START, GATE_ABORT. Последний ML-адаптер использует главным образом завершённую историю и её агрегаты.</li>
<li>Прежний опыт был построен для возврата к зоне на расстоянии1–3R с M1-clock. Его поздние запросы не являются неизбежной задержкой исходной ленты.</li>
<li>Сессионный контекст в CME_SESSION_STORE_V1 уже пересекает UTC-полночь. Но в проверенных stores сохранено <b>0 событий предыдущей сессии</b>, хотя её минутный контекст присутствует. Price-level episode memory между сессиями ещё не объединена.</li>
<li>player_v3 использовал другой словарь — всплески пятисекундных интервалов. В последнем series-run186/192 обучающих прогонов цензурированы. Нулевые изменения argmax в1280 пробах WAIT-политики не доказывают отсутствия информации в новой последовательности.</li>
<li>Неизвестная сторона встречается в92–93% известных128s окон этих двух дней, но медианная доля такого объёма —2,95–3,48%. Частота окон и потерянная доля объёма — разные величины.</li></ul></section>
<section><h2>Что классифицируется сейчас</h2><table><tr><th>Слой</th><th>Механизм</th><th>Ограничение</th></tr>
<tr><td>Микрокарта</td><td>Полосы5тиков, гистерезис2; native BBO и точное время</td><td>Микроприбытие не является независимым торговым касанием</td></tr>
<tr><td>Эпизоды</td><td>Автомат arrival/edge/gate/confirm; ядро10тиков, ворота10, выдержка3с</td><td>Фиксированная геометрия и физический порядок доступны; итог PASS/REJECT известен только при завершении</td></tr>
<tr><td>Механизмы книги</td><td>Пороговые30s дескрипторы объёма, движения, fill/remove/refill и качества стороны</td><td>Это теги наблюдаемой конфигурации. Например fill суммируется по обеим сторонам; одним названием причину движения установить нельзя</td></tr>
<tr><td>Всплески player_v3</td><td>5s интервалы;2,5×прошлый95-й процентиль за10мин, перевооружение после двух тихих интервалов</td><td>Это прежний словарь fresh/cluster/recurrent, а не текущая память взаимодействий с ценовой зоной</td></tr></table></section>
<section><h2>Измеренная задержка и чувствительность</h2><div class="scroll"><table><tr><th>Дата</th><th>Определение</th><th>Эпизоды07–20UTC</th><th>Медиана edge→confirm, с</th><th>Медианное движение, USD/oz</th></tr>{''.join(vr)}</table></div>
<p>Отсчёт от последнего наблюдаемого EDGE_EXIT; это не неизвестное «истинное начало тренда». Движение до подтверждения не объявляется упущенной прибылью.
Уменьшение выдержки до1с изменяет PASS/REJECT примерно у6% одинаковых исходных прибытий и добавляет15–17% новых эпизодов.
Ворота5вместо10тиков изменяют итог примерно у12–13% совпавших прибытий. Это чувствительность определения, не ошибка относительно истинной разметки.</p>
<p>Полусдвиг сетки изменил общее число эпизодов на−1,7% и+0,6%. Это не доказывает адресную устойчивость: соседние физические зоны всё равно отличаются.
Гистерезис микрокарты2→3тика уменьшил число прибытий примерно на24–25%, сохранив суммарное наблюдаемое время.</p>
<table><tr><th>Дата</th><th>Общие часы фактической политики</th><th>Медиана confirm→query, с</th></tr>{''.join(cr)}</table>
<p>Эта дополнительная задержка включает M1-clock, геометрию, качество зоны и cooldown. Это другой этап, чем распознавание события.
Полная таблица всех зональных кандидатов в summary.json даёт83–85с; реальная общая популяция политики —73–76с.</p></section>
<section><h2>Предлагаемый вход</h2><ol>
<li><b>Быстрый ответ:</b> недавний путь mid/spread/depth и направленные fill/remove/refill; известная и неизвестная часть потока отдельно.</li>
<li><b>Последовательность эпизодов:</b> начало, обновления формирования, отмена, подтверждение; parent/zone ID, время и источник. Новое подтверждение не переписывает прежнее наблюдение.</li>
<li><b>Память по цене:</b> занимаемость, визиты, проходы, возраст, поддержка на нескольких масштабах. Абсолютные адреса сохраняются, относительные координаты вычисляются при запросе.</li>
<li><b>Две сессии:</b> профиль, уровни и режим; отдельный бюджет для вчерашней памяти, чтобы очередь last256 не вытеснила её в активное утро.</li>
<li><b>Позиция:</b> inventory, цена входа, текущая ликвидация, время и остаточный риск — отдельная ветвь состояния.</li></ol>
<p>Начать с одной небольшой модели с отдельными кодировщиками этих представлений и общим решением. Несколько независимых голосующих моделей пока не нужны.
Измерители дают словарь и факты; модель учит отношения между ними и условные последствия действий.</p>
<p>Пограничные состояния передаются непрерывными величинами: дистанция до ворот, доля истекшей выдержки, скорость, интенсивность потока, доля неизвестного.
FORMING — известная стадия наблюдения, не автоматически заданная вероятность успеха. Пороговые теги дополняют эти величины.</p>
<p><a href="../../../experiments/scalper_view_audit_v1/observation_contract.json">Контракт входа</a> имеет статус PROPOSED_NOT_IMPLEMENTED.</p></section>
<section><h2>Как проверять именно пользу последовательности</h2>
<p>Сравнить на общей популяции непрерывные измерения, агрегаты завершённых эпизодов, упорядоченные события, затем формирующиеся состояния и память вчера.
Контроль порядка должен удалять временную информацию осмысленно: простая перестановка строк с сохранённым age не скрывает порядок от модели.</p>
<p>Нужны синтетические пары с одинаковыми средними и разным порядком. Например, refill после продаж и refill до нового удара должны различаться во входе.
Если отдельные типы событий имеют одинаковое среднее будущее, это ещё не исключает эффекта их последовательностей и взаимодействий с режимом.</p>
<p>Дополнительные задачи на переходы, реакцию цены и повторное посещение помогут проверить обучение представления. Экономический допуск остаётся отдельным:
денежные исходы, WAIT, стоимость исполнения и последовательный счёт. Сначала наблюдение → способность различать → польза прогноза → деньги.</p>
<p>Внешние ориентиры: <a href="https://arxiv.org/abs/1808.03668">DeepLOB</a> сочетает пространственную обработку книги с временной;
<a href="https://arxiv.org/abs/1011.6402">Cont–Kukanov–Stoikov</a> исследует совместное действие добавлений, отмен и исполнений.
Эти работы мотивируют измерения, но не доказывают прибыльность нашего GC-контракта или переноса архитектуры.</p></section>
<section><h2>12 сцен незавершённой попытки</h2><p>Выбор первых GATE_START по хешу, шесть на день, без выбора по исходу. Слева и на вертикальной линии — доступное прошлое.
Серое справа — аудит последующего развития. Это ограниченный технический атлас; оценка человеком «похоже ли это на значимый скальперский паттерн» остаётся отдельной задачей.</p>
{''.join(f'<img src="{name}" alt="Шесть сцен формирования ухода, страница{i+1}">' for i,name in enumerate(case_images))}</section>
<footer><a href="summary.json">Измерения</a> · <a href="adapter_audit.json">Адаптер и архив player_v3</a> · <a href="report_receipt.json">Источники отчёта</a><p>Нет нового обучения, PnL replay или экономического допуска. Исходные исследования и библиотека не менялись.</p></footer>'''
    css='''*{box-sizing:border-box}body{margin:0;background:#f5f6f2;color:#233b40;font:17px/1.65 system-ui}main{max-width:1120px;margin:auto;padding:40px 28px}h1{font-size:clamp(31px,5vw,50px);line-height:1.14;max-width:960px;letter-spacing:-1.2px}h2{font-size:26px;line-height:1.3}.lead{font-size:21px}.eyebrow{font-size:13px;text-transform:uppercase;color:#466e6a}section{border-top:1px solid #d6dfd7;padding:24px 0;margin-top:25px}li{margin-bottom:12px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px 9px;text-align:left;border-bottom:1px solid #d6dfd7;vertical-align:top}th{background:#e9eee6}.scroll{overflow:auto}a{color:#176e7d;text-underline-offset:3px}img{width:100%;height:auto;background:#fafbf7;border:1px solid #d6dfd7;border-radius:8px;margin:15px 0}footer{border-top:1px solid #d6dfd7;padding:25px 0;font-size:14px}@media(max-width:640px){main{padding:24px 14px}body{font-size:16px}h2{font-size:23px}table{font-size:12px}td,th{padding:8px 5px}ol,ul{padding-left:20px}}'''
    (out/'report.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Представление скальпера</title><style>'+css+'</style><main>'+intro+content+'</main></html>',encoding='utf-8',newline='\n')
    inputs[str(Path(__file__).relative_to(ROOT))]=digest(__file__)
    dump(out/'report_receipt.json',dict(status='COMPLETE_DESCRIPTIVE_REPORT',input_sha256=inputs,
        artifacts={name:digest(out/name) for name in ['adapter_audit.json','report.html']+case_images},
        visual_review='PENDING',training=False,economic_admission=False))
    print(json.dumps(dict(output=str(out/'report.html'),common_clock=common,store=stores),ensure_ascii=False))


if __name__=='__main__':main()
