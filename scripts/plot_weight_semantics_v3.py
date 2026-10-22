"""Plot frozen TRAIN teacher-population diagnostics without reading fit scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--report', default='reports/scientific_pipeline_v3/weight_semantics.json')
    parser.add_argument('--output', default='reports/scientific_pipeline_v3/09_weight_target_population.png')
    parser.add_argument('--receipt', default='reports/scientific_pipeline_v3/weight_figure_receipt.json')
    args = parser.parse_args()
    source = Path(args.report)
    report_hash = sha(source)
    data = json.loads(source.read_text(encoding='utf-8'))
    if data['status'] != 'PASS' or not data['reconstructed_weights_exact'] or not data['no_test_scores_read']:
        raise ValueError('A verified TRAIN-only teacher audit is required')
    classes = ('LONG_ONLY', 'SHORT_ONLY', 'BOTH')
    colors = ('#236491', '#c76a25')
    population = data['observed_comparison']
    values = {metric: np.asarray([population[c]['uniform_to_weighted_medians'][metric] for c in classes])
              for metric in ('origin_span_s', 'delay_s', 'gain_margin_R')}
    values['tail_percent'] = 100*np.asarray([[population[c]['high_weight_tail_root_fraction'],
        population[c]['high_weight_tail_optimization_mass_fraction']] for c in classes])
    if not all(np.isfinite(a).all() and a.shape == (3, 2) for a in values.values()):
        raise ValueError('Complete finite three-class comparison required')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False})
    fig, axes = plt.subplots(2, 2, figsize=(15.7, 10.7), facecolor='#f5f7fa')
    fig.subplots_adjust(left=.105, right=.965, top=.755, bottom=.27, hspace=.68, wspace=.30)
    specs = [
        ('origin_span_s', 'A  ·  Окно допустимых точек одного ID', 'Медиана span, секунды', (0, 260), '{:g} с'),
        ('delay_s', 'B  ·  От входа до первого qualifying crossing', 'Медиана задержки, секунды', (0, 190), '{:.0f} с'),
        ('gain_margin_R', 'C  ·  Запас над порогом среди первых пересечений', 'Медиана max-first-crossing margin / R', (0, 1.55), '{:.2f} R'),
        ('tail_percent', 'D  ·  Верхний хвост весов: точки и масса', 'Доля внутри соответствующего класса, %', (0, 55), '{:.1f}%'),
    ]
    for ax, (metric, title, xlabel, limits, fmt) in zip(axes.flat, specs):
        ax.set_facecolor('white')
        ax.grid(axis='x', color='#dce3eb', linewidth=.8)
        ax.set_axisbelow(True)
        y = np.arange(3)
        a = values[metric]
        for i in y:
            ax.plot(a[i], [i, i], color='#aab5c3', linewidth=2, zorder=1)
            ax.scatter(a[i, 0], i, s=72, color=colors[0], marker='o', zorder=3)
            ax.scatter(a[i, 1], i, s=82, color=colors[1], marker='D', zorder=3)
            ax.annotate(fmt.format(a[i, 0]), (a[i, 0], i), xytext=(0, 10),
                textcoords='offset points', ha='center', va='bottom', color=colors[0], fontsize=11, weight='bold')
            ax.annotate(fmt.format(a[i, 1]), (a[i, 1], i), xytext=(0, -13),
                textcoords='offset points', ha='center', va='top', color=colors[1], fontsize=11, weight='bold')
        ax.set_yticks(y, ['LONG', 'SHORT', 'BOTH'])
        ax.tick_params(axis='y', length=0, pad=12)
        ax.tick_params(axis='x', length=0, labelsize=10)
        ax.set_ylim(2.55, -.55)
        ax.set_xlim(*limits)
        ax.set_xlabel(xlabel, labelpad=9, color='#485568', fontsize=10)
        ax.set_title(title, loc='left', fontsize=12, weight='bold', pad=17, color='#1e3047')
    fig.text(.06, .954, 'TRAIN  /  TEACHER DIAGNOSTICS', fontsize=11, weight='bold', color='#537087')
    fig.text(.06, .915, 'Что усиливает обратная кратность crossing-ID', fontsize=22, weight='bold', color='#172c43')
    fig.text(.06, .875, 'Меньше точек на ID → больше веса каждой точки. Масса LONG, SHORT и BOTH сохраняется отдельно.',
             fontsize=12, color='#46576a')
    fig.legend(handles=[
        Line2D([0], [0], marker='o', linestyle='', color=colors[0], markersize=8, label='Равные точки TRAIN'),
        Line2D([0], [0], marker='D', linestyle='', color=colors[1], markersize=8, label='Масса inverse-ID')],
        loc='upper left', bbox_to_anchor=(.055, .849), ncol=2, frameon=False, columnspacing=3, fontsize=12)
    fig.text(.06, .199, 'Наблюдение: около 11% точек LONG/SHORT получают 47–48% массы класса; медиана запаса над порогом снижается.',
        fontsize=12, weight='bold', color='#273d55')
    notes = [
        'A–C: медианы распределения до и после взвешивания; BOTH — среднее двух положительных сторон каждой точки.',
        'Span = время между первой и последней допущенными TRAIN-точками ID. Это не длительность сделки; span ≠ count × 5 с.',
        'Margin = max(net на первых пересечениях 9 линий за 300 с) − G, в единицах R. Это не executed PnL и не margin первого qualifying crossing.',
        'D: хвост weight ≥ q90 внутри класса; равные веса сохранены, поэтому доля точек может превышать 10%. G/R = 2 во всех группах.',
        f"{data['train_roots']:,} TRAIN-точек; {data['id_distribution']['unique_ids']:,} положительных ID. NONE имеет вес 1. Fitted scores и TEST не использованы.",
    ]
    for i, note in enumerate(notes):
        fig.text(.06, .161-i*.026, note, fontsize=10, color='#536173')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    if sha(source) != report_hash:
        raise ValueError('Bound teacher audit changed while plotting')
    receipt = dict(status='GENERATED_REQUIRES_VISUAL_REVIEW',
        source_script=str(Path(__file__).resolve()), source_script_sha256=sha(__file__),
        input_report=str(source.resolve()), input_report_sha256=report_hash,
        png=str(output.resolve()), png_sha256=sha(output), figure_size_pixels=[2512, 1712],
        plotted_values={k: v.tolist() for k, v in values.items()}, class_order=list(classes),
        statistics='Medians of uniform-root and weight-mass empirical distributions, not medians of values multiplied by weights.',
        semantics={'population': 'Known feature-available TRAIN roots only',
            'span': 'Observed admitted origin span, not trade duration',
            'margin': 'Maximum net first-crossing mark across nine lines within H minus G; not executed PnL',
            'both': 'Side-averaged root metrics', 'tail': 'weight >= class q90 including all ties'},
        fitted_scores_read=False, frozen_training_files_modified=False)
    Path(args.receipt).write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(png=str(output), png_sha256=receipt['png_sha256'], receipt=args.receipt)), flush=True)


if __name__ == '__main__':
    main()
