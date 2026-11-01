"""Static scientific atlas: exact observed paths and causally retrieved neighbors."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectrum.io import sha
from experiments.scalper_view_audit_v1.audit import dump
from .adapter import ROOT, HERE
from .tokens import NS, TICK


def plot_case(ax, folder, case, role, color):
    name = case['case_id']+'_'+role
    q = pd.read_parquet(folder/(name+'_quotes.parquet'))
    tokens = pd.read_parquet(folder/(name+'_tokens.parquet'))
    start = int(tokens.start_ns.iloc[0]); end = int(tokens.available_ns.iloc[-1])
    anchor = int(tokens.start_mid_nano.iloc[0]); amplitude = float(tokens.amplitude_ticks.iloc[0])
    mid = (q.bid_nano.to_numpy(np.int64)+q.ask_nano.to_numpy(np.int64))//2
    # Full native trace, no candle resampling or interpolation over unavailable future.
    ax.plot((q.available_ns.to_numpy()-start)/NS, (mid-anchor)/(TICK*amplitude),
            color=color, lw=.85, rasterized=True)
    for i, row in enumerate(tokens.itertuples()):
        a = (int(row.start_ns)-start)/NS; b = (int(row.available_ns)-start)/NS
        ax.axvspan(a,b,color=['#e7edf6','#e0e9f2','#fce8bb'][i],alpha=.55)
        ax.axvline(a,color='#708295',lw=.6,ls='--')
        text_x = (end-start)/NS*.98 if i==2 else (a+b)/2
        ax.text(text_x,.98, ['до−2','до−1','формируется'][i], ha='right' if i==2 else 'center', va='top',
                transform=ax.get_xaxis_transform(), fontsize=8, color='#425368')
    timestamp = pd.Timestamp(end, tz='UTC').strftime('%d.%m %H:%M:%S')
    label = 'Текущее наблюдение' if role == 'query' else 'Прошлая сессия'
    ax.set_title(f'{label} · {timestamp} UTC\n{case["scale"]}, A = {amplitude:g} тиков', fontsize=10)
    ax.set_xlabel('Секунды от начала тройки', fontsize=9)
    ax.set_ylabel('Изменение цены / A', fontsize=9)
    ax.grid(alpha=.18); ax.tick_params(labelsize=8)
    ax.set_xlim(0,max(1,(end-start)/NS))


def main(run):
    folder = ROOT/run
    if (folder/'report_receipt.json').exists(): raise FileExistsError('Report already sealed')
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'COMPLETE_DESCRIPTIVE_ADAPTER': raise ValueError('Incomplete adapter run')
    for name, value in manifest['artifacts'].items():
        if sha(folder/name) != value: raise ValueError('Artifact drift: '+name)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    cases = json.loads((folder/'atlas_index.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    files = []
    for query in ('query_1','query_2','query_3'):
        subset = [c for c in cases if c['case_id'].startswith(query)]
        if not subset: continue
        fig, axes = plt.subplots(2,len(subset),figsize=(15,7),squeeze=False)
        for k, case in enumerate(subset):
            plot_case(axes[0,k],folder,case,'query','#116b9e')
            plot_case(axes[1,k],folder,case,'past','#9350a2')
            axes[1,k].text(.99,.04,f'd² = {case["distance"]:.4f}',ha='right', transform=axes[1,k].transAxes,
                           fontsize=9,bbox=dict(fc='white',ec='none',alpha=.85))
        fig.suptitle('Истории разных скоростей: две завершённые части и доступное продолжение',fontsize=14)
        fig.text(.5,.014,'Каждый график имеет собственную ось времени. Продолжение после правой границы не показано и не участвовало в поиске.',
                 ha='center', fontsize=9,color='#4d5966')
        fig.tight_layout(rect=[0,.04,1,.95])
        name = query+'_atlas.png'; fig.savefig(folder/name,dpi=150);plt.close(fig);files.append(name)
    # A direct view of where yesterday's dwell and rejected/passed visits survive.
    memory = pd.read_parquet(folder/'query_2_price_memory.parquet')
    fig, axes = plt.subplots(1,3,figsize=(15,5),sharey=False)
    for ax, scale in zip(axes, ('fast','middle','slow')):
        f = memory[memory.scale==scale]
        for k,(date,g) in enumerate(f.groupby('session_id')):
            ax.barh(g.offset_ticks+(k-.5)*.4, g.dwell_seconds, height=.35,
                    label=date,color=['#9350a2','#116b9e'][k],alpha=.8)
        ax.axhline(0,color='#d68c27',lw=1)
        ax.set_title(scale);ax.set_xlabel('Наблюдавшееся время в полосе, с');ax.set_ylabel('Середина полосы от текущей цены, тики')
        ax.grid(axis='x',alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Память мест в цене: текущая и предыдущая сессии в 12:00 UTC',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(folder/'price_memory.png',dpi=150);plt.close(fig)
    files.append('price_memory.png')
    build_rows = ''.join(f'<tr><td>{b["session_id"]}</td><td>{b["quote_rows"]:,}</td>'
        f'<td>{b["motion_updates"]:,}</td><td>{b["zone_updates"]:,}</td>'
        f'<td>{b["seconds_including_source_verification"]:.2f} с</td></tr>' for b in summary['builds'])
    obs_rows = ''.join(f'<tr><td>{o["timestamp"][11:19]}</td><td>{o["current_retained"]}</td>'
        f'<td>{o["previous_retained"]}</td><td>{o["active_retained"]}</td><td>{o["matches"]}</td>'
        f'<td>{o["elapsed_seconds"]:.2f} с</td></tr>' for o in summary['observations'])
    sections = ''.join(f'<figure><a href="{name}"><img src="{name}" alt="Атлас историй {name}"></a>'
        f'<figcaption>{html.escape(name)} · полное разрешение открывается по нажатию</figcaption></figure>' for name in files)
    document = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Адаптер историй рынка — причинный прототип</title><style>
body{{font:17px/1.6 system-ui,sans-serif;color:#203046;background:#f4f6f8;margin:0}}main{{max-width:1200px;margin:auto;padding:36px 24px}}
h1{{font-size:36px;line-height:1.2}}h2{{margin-top:40px}}p{{max-width:1000px}}.notice{{background:#fff3d9;padding:20px;border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:15px;background:white}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #dde4ed}}
.scroll{{overflow:auto}}figure{{margin:24px 0;background:white;padding:10px;border-radius:10px}}img{{width:100%;height:auto}}figcaption{{font-size:13px;color:#5e6c7a}}
code{{font-size:14px}}a{{color:#086ca1}}li{{margin:8px 0}}@media(max-width:600px){{main{{padding:20px 12px}}h1{{font-size:28px}}}}
</style><main><p>SCALPER_STORIES_ADAPTER_V1 · 24–25 августа 2026 · описательный опыт</p>
<h1>Машина получает истории нескольких скоростей</h1>
<p>Из одной native BBO ленты построены истории движения, обновления ценовых зон, карта времени в цене и причинный поиск похожих троек.
Текущая попытка доступна до подтверждения. Предыдущая сессия сохраняется отдельной квотой и через найденные исторические аналоги.</p>
<div class="notice">Готов причинный пакетный адаптер. Обученного энкодера и подтверждения денежной полезности здесь нет.
Ближайший по формуле аналог может оказаться плохим рыночным аналогом — атлас нужен, чтобы это увидеть.</div>
<h2>Как нарезана лента</h2><p>Три сочетания амплитуды и времени: <b>4 тика / 15 с; 12 тиков / 60 с; 36 тиков / 300 с</b>.
Разворот фиксируется после отката на соответствующую амплитуду; подтверждение не переносится назад к экстремуму.
Истечение времени отличается от разворота. Параллельно идут зоны шириной 5/10/30 тиков.</p>
<p>В измерениях остаются смещение, полный путь, диапазон, эффективность, изменения направления, форма в четвертях наблюдённого интервала,
откаты, время в зоне и повторы попыток. Книга/T добавляются как отдельные опубликованные окна 1/5/30 с с масками и неизвестной стороной сделки.</p>
<h2>Что сгенерировано</h2><div class="scroll"><table><tr><th>Сессия</th><th>Native BBO</th><th>Обновления движения</th><th>Обновления зон</th><th>Построение + проверка источников</th></tr>{build_rows}</table></div>
<p>Число обновлений не равно числу независимых событий или торговых возможностей. Повторные состояния принадлежат одной истории.</p>
<div class="scroll"><table><tr><th>Запрос UTC</th><th>Текущие токены</th><th>Прошлая сессия</th><th>Активные</th><th>Аналоги</th><th>Время запроса</th></tr>{obs_rows}</table></div>
<p>Численный вход: {summary['feature_count']} мер + 4 категориальных поля. Отдельно передаются карта и опубликованный контекст сессии.
Квоты последних обновлений, завершённых историй и прошлого дня независимы. Все активные состояния и найденные тройки добавляются сверх них.</p>
<h2>Как читать сходство</h2><p>Сравниваются две предшествующие завершённые истории и доступная часть текущей.
Положение в тройке сохраняется. Кандидат заканчивается до начала всей текущей тройки; продолжение справа от границы не участвует.
Формула сопоставляет геометрию, длительность и путь внутри одного масштаба и фазы. Сходство книги или торгового результата пока не установлено.</p>
{sections}
<h2>Что проверено</h2><ul><li>Пересборка всей текущей сессии только до 12:00: модельное наблюдение, маски, карта, контекст и найденные аналоги совпали.</li>
<li>{summary['scalar_native_interval_checks']} отдельных скалярных проверок native пути, размаха, смещения и времени в полосе.</li>
<li>Синтетика: разный порядок при одинаковых суммах; таймер до одновременной публикации; обрывы и invalid book; частые касания не вытесняют завершённые истории.</li>
<li>Хеши входов и исходники сохранены. Лента содержит уже исследованные даты; новый holdout не заявляется.</li></ul>
<h2>Что ещё ограничивает применение</h2><p>Запрос длится секунды, поэтому это эталон для проверки представления, а не готовый поток для торговли.
Пороговые определения и метрика сходства пока ручные. Карта сохраняет пространственную сетку и её известные ограничения.
Следующая содержательная проверка — насколько найденные аналоги совпадают по механизму и добавляет ли последовательность информацию сверх агрегатов.</p>
<p><a href="summary.json">Численные результаты</a> · <a href="manifest.json">Происхождение и хеши</a> · <a href="config.json">Настройки</a> ·
<a href="query_2_similarity.json">Сходства в 12:00</a> · <a href="query_2_receipt.json">Контракт модельного входа</a></p></main></html>'''
    (folder/'report.html').write_text(document,encoding='utf-8',newline='\n');files.append('report.html')
    dump(folder/'report_receipt.json', dict(status='RENDERED_STATIC_ATLAS',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},
        input_manifest_sha256=sha(folder/'manifest.json'),
        artifacts={name:sha(folder/name) for name in files},
        note='Browser rendering is not asserted; PNG visual review is a separate receipt'))
    print(json.dumps(dict(report=str(folder/'report.html'), figures=files[:-1])),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=json.loads((HERE/'config.json').read_text())['output'])
    main(p.parse_args().run)
