"""A compact visual explanation of hindsight opportunity versus action value."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spectrum.io import sha
from experiments.mechanical_lab.data import dump

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();root=ROOT/args.run
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    paired=pd.read_parquet(root/'paired_targets.parquet');p=paired[paired.future_both_filled_complete]
    z=summary['complete_unweighted'];g=summary['future_description']
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    ax=axes[0]
    colors=p.future_joint_profit_state.map({'LONG_ONLY':'#347d91','SHORT_ONLY':'#b98b43','NEITHER':'#aa5f69','BOTH':'#36875d'})
    ax.scatter(p.future_u_long_r,p.future_u_short_r,s=10,alpha=.22,c=colors)
    ax.axhline(0,color='#63727a',lw=1);ax.axvline(0,color='#63727a',lw=1)
    ax.set_xlabel('Фактический будущий LONG, net R');ax.set_ylabel('Фактический будущий SHORT, net R')
    ax.set_title(f"{len(p)} известных пар · точки зависимы\nЭти ответы недоступны в момент решения",loc='left',fontsize=11)
    ax.grid(alpha=.12)
    ax=axes[1]
    names=['Всегда LONG','Всегда SHORT','Всегда WAIT','Выбор с будущим\nLONG/SHORT/WAIT']
    values=[z['mean_long_r'],z['mean_short_r'],0,z['hindsight_with_wait_r']]
    bars=ax.bar(range(4),values,color=['#347d91','#b98b43','#687784','#ded8ee'],edgecolor=['#347d91','#b98b43','#687784','#806295'])
    bars[-1].set_hatch('///')
    ax.set_xticks(range(4),names,fontsize=9);ax.set_ylabel('Средний net R на отдельное предложение')
    ax.axhline(0,color='#63727a',lw=1);ax.set_ylim(-.46,.68)
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,value+(.015 if value>=0 else -.015),f'{value:+.3f}R',ha='center',va='bottom' if value>=0 else 'top',fontsize=12)
    ax.set_title('Наличие прибыльного будущего не равно\nвозможности выбрать его по истории',loc='left',fontsize=11)
    ax.text(.02,.97,'Справа — ретроспективная граница,\nне модель и не результат торговли.',transform=ax.transAxes,va='top',fontsize=9,color='#5f4c6f')
    fig.suptitle('Таргет возможности и ценность действия — разные величины',fontsize=16)
    fig.savefig(root/'opportunity_vs_value.png',dpi=140);plt.close(fig)
    recon=pd.DataFrame(json.loads((root/'colleague_reconciliation.json').read_text(encoding='utf-8')))
    recon=recon[['family','relation','complete_actions','win_fraction','net_mean_r','same_execution_mid_move_mean_r','actual_cost_mean_r']]
    recon.columns=['Фраза','Сторона','Полных исходов','Win rate','Net/R','Mid ход при тех же исполнениях/R','Реальные издержки/R']
    states=''.join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>' for k,v in z['joint_counts'].items())
    body=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Таргет действия: возможность, сторона и контроль</title>
    <style>body{{font:17px/1.6 system-ui;background:#f5f7f8;color:#203340;margin:0}}main{{max-width:1200px;margin:auto;padding:30px}}h1{{font-size:35px;line-height:1.2}}h2{{margin-top:30px}}img{{width:100%;height:auto}}.note{{background:#fff0d9;border-left:5px solid #b58441;padding:18px}}table{{border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:9px;border:1px solid #d5dfe4;text-align:right}}td:first-child{{text-align:left}}.scroll{{overflow:auto}}code{{background:#e5ecf0;padding:3px 5px}}@media(max-width:600px){{main{{padding:14px}}h1{{font-size:26px}}}}</style><main>
    <p>PAIRED_ACTION_TARGET_AUDIT_V1 · 24–28 августа 2026 · без обучения</p><h1>Прогноз должен отвечать за ценность выбранного действия</h1>
    <p>Проверена таблица коллеги и добавлен явный контракт: две стороны одного решения, их общая составляющая, преимущество стороны, сроки зрелости и неизвестные исходы. Предыдущие входы, выходы и комиссия не менялись.</p>
    <p class="note">{summary['admitted_origins']} предложений — это {summary['exact_clock_risk_contracts']} разных сочетаний точного времени и риска, а не независимые сделки. Полностью исполнены обе стороны у {summary['complete_filled_pairs']} пар; у {summary['nofill_pairs']} нет смоделированного fill, у {summary['incomplete_pairs']} неизвестен итог хотя бы одной стороны. Даты ранее изучались.</p>
    <img src="opportunity_vs_value.png" alt="Фактические будущие исходы и недоступный выбор с подсказкой будущего">
    <h2>Как связать opportunity и direction</h2><p>Пусть uL/uS — будущие net-исходы зафиксированного действия. C=(uL+uS)/2, D=(uL−uS)/2. По доступной истории X оцениваются c=E[C|X], d=E[D|X]. Тогда qL=c+d, qS=c−d, ожидаемая ценность входа c+|d|. Это одна согласованная пара денежных ответов.</p>
    <p><code>E[max(0,uL,uS)|X] ≥ max(0,E[uL|X],E[uS|X])</code></p><p>Левая часть требует подсказки будущего, правая — доступной предсказуемости. В синтетике (+1,−1.4) или (−1.4,+1) с вероятностями 1/2 каждая траектория односторонняя, но оба ожидания −0.2: WAIT. При условной вероятности 70% первой ветви LONG имеет +0.28. Тесты сохраняют это различие.</p>
    <p>{z['near_equal_pairs']} почти равных пар (разность ≤0.1R) здесь все NEITHER. Равенство не измеряет риск само по себе. При другом управлении два близких положительных результата могли бы разрешать действие. UNKNOWN остаётся маской, не классом плохого рынка.</p>
    <h2>Движение есть, доступная прибыль — другой вопрос</h2><p>На одинаковом будущем окне 120 с диапазон ≥1R у {g['range_ge_1r']} из {g['known']} поддержанных наблюдений ({100*g['range_ge_1r']/g['known']:.1f}%). Медиана {g['range_median_r']:.2f}R. При этом {g['neither_despite_range_ge_1r']} полных пар с диапазоном ≥1R имеют оба неприбыльных исхода. Поэтому volatility/range могут быть auxiliary-целями, но не самостоятельным разрешением входа.</p>
    <p>Также сохранены будущие концентрация времени, перекрытие половин профиля, миграция центра, секундная эффективность и время внутри полосы ±0.5R. Они измерены, но их предсказуемость и денежная польза ещё не проверены. Старые MFE/MAE ограничены выходом своей сделки и не описывают это общее окно.</p>
    <h2>Проверка арифметики издержек</h2><p>Net разложен на midpoint-ход между фактическими исполнениями минус половины спредов при входе/выходе и комиссию. Снятие издержек сохраняет те же net-барьеры и моменты исполнения: это не отдельный беззатратный backtest.</p><div class="scroll">{recon.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div>
    <h2>Единая задача нескольких голов</h2><p>Общий encoder получает текущую ногу/реакцию, память места, большие VWAP/BB, стоимость и маски. Основной loss проверяет оба денежных исхода одной точки; вспомогательные головы — диапазон, удержание места, риск и время до первого события. Сохраняется только то дополнение, которое улучшило общую калибровку и последовательную полезность сверх action-value baseline. Истинный будущий режим не передаётся нижним головам как вход.</p><p>Координаты c/d — обратимая запись qL/qS, а не источник нового преимущества. Попарный квадратичный loss равен сумме двух ошибок C/D. Общий encoder сам по себе не доказывает взаимную пользу голов: нужны сравнение с прямыми qL/qS и последовательные исключения вспомогательных задач.</p>
    <p>Совместная голова NEITHER/LONG_ONLY/SHORT_ONLY/BOTH может связать вероятности знаков. Размеры выигрыша/убытка всё равно нужны для q. Более высокая точность одной головы или произведение независимых confidence-score не доказывают преимущество входа.</p>
    <h2>От фиксированного выхода к управлению</h2><p>Сейчас исполнима supervised-задача WAIT/LONG/SHORT с одним executor. Затем можно сравнить несколько заранее заданных exit-планов. Следующий отдельный опыт — HOLD/CLOSE из одного причинно достижимого состояния позиции, с наградой приращения капитала и без идеального будущего максимума в метке.</p><p>В полной последовательной задаче WAIT сохраняет будущие возможности: его value не равен нулю. Занятость позиции, длительность и ограничения счёта требуют последовательного replay, затем optimal stopping/fitted value iteration или SMDP/RL. Сам переход к RL не устраняет ошибки исполнения и отсутствие прогнозируемого преимущества.</p>
    <h2>Что нужно до обучения</h2><ol><li>X и исходы формирующихся, отменённых и подтверждённых фаз плюс причинный clock-контроль; не только выжившие подтверждения.</li><li>Хронологические train/calibration/test, зрелость меток, группировка повторов и разрыв пересекающихся окон.</li><li>Калибровка денежных ожиданий и риска среди выбранных входов; полный счёт, WAIT, matched controls и стресс.</li><li>Отдельный договор по неизвестным исполненным позициям. Их нельзя убрать из replay по будущей маске.</li></ol>
    <p>Полный контракт: <a href="../../../experiments/paired_action_target_v1/README.md">README</a>. <a href="validation.json">Проверка</a> · <a href="summary.json">Числа</a>. Научные ориентиры: <a href="https://arxiv.org/abs/1710.08005">decision-focused learning</a>, <a href="https://proceedings.mlr.press/v70/guo17a.html">калибровка</a>, <a href="https://arxiv.org/abs/2005.01643">offline RL</a>.</p></main></html>'''
    (root/'report.html').write_text(body,encoding='utf-8')
    dump(root/'report_receipt.json',dict(status='GENERATED',reporter_sha256=sha(Path(__file__)),manifest_sha256=sha(root/'manifest.json'),
         image_sha256=sha(root/'opportunity_vs_value.png'),html_sha256=sha(root/'report.html'),browser_rendered=False))
    print(str(root/'report.html'))


if __name__=='__main__':main()
