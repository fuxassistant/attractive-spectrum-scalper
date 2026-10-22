"""Unsmoothed visual evidence for the preregistered bounded-ID comparison."""
from pathlib import Path
import html
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator

from spectrum.research_v3.learning import load_dataset, sha

ROOT = Path('reports/scientific_pipeline_v4')
ARMS = ('uniform', 'soft_crossing_id')
COLORS = ('#2563eb', '#d97706')
LABELS = ('Uniform', 'Bounded sqrt-ID')
PNAMES = ['p_none', 'p_long', 'p_short', 'p_both']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def finish(fig, name):
    fig.savefig(ROOT/name, dpi=135, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load_evidence():
    suite = read(ROOT/'suite_summary.json')
    config = read('configs/research/opportunity_pipeline_v4.json')
    if suite['status'] != 'COMPLETE' or len(suite['runs']) != 4:
        raise ValueError('Four COMPLETE runs required')
    if sha('configs/research/opportunity_pipeline_v4.json') != suite['config_sha256']:
        raise ValueError('Configuration changed after fit')
    store = load_dataset(suite['dataset_root'])
    f = store.frame
    inputs = {str(ROOT/'suite_summary.json'): sha(ROOT/'suite_summary.json'),
              'configs/research/opportunity_pipeline_v4.json': suite['config_sha256'],
              str(Path(suite['dataset_root'])/'manifest.json'): suite['dataset_manifest_sha256']}
    bundles = {}
    for entry in suite['runs']:
        path = Path(suite['runs_root'])/entry['run_id']
        receipt = read(path/'receipt.json')
        if receipt['status'] != 'COMPLETE' or receipt['dataset_manifest_sha256'] != suite['dataset_manifest_sha256']:
            raise ValueError('Incomplete or unmatched run')
        for name, digest in receipt['outputs'].items():
            if sha(path/name) != digest:
                raise ValueError('Changed run output: '+name)
        for name, digest in receipt['sources'].items():
            if sha(name) != digest:
                raise ValueError('Changed run source: '+name)
        inputs[str(path/'receipt.json')] = sha(path/'receipt.json')
        p = pd.read_parquet(path/'predictions.parquet')
        for column in ('t_ns', 'y_crossing', 'date', 'feature_available'):
            if not np.array_equal(p[column], f[column]):
                raise ValueError('Prediction clock/cohort mismatch: '+column)
        if not np.array_equal(p.root_index, np.arange(len(f))):
            raise ValueError('Prediction root order differs')
        bundles[(entry['seed'], entry['weighting'])] = dict(path=path, p=p,
            summary=read(path/'summary.json'), history=pd.DataFrame(read(path/'history.json')),
            economics=read(path/'economic_evaluation.json'))
    if set(bundles) != {(seed, arm) for seed in (17, 29) for arm in ARMS}:
        raise ValueError('Wrong experiment matrix')
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError('Input changed: '+path)
    return suite, config, store, bundles, inputs


def probability_and_training(bundles):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for row, seed in enumerate((17, 29)):
        for col, (key, title) in enumerate((('joint_nll', 'Joint NLL (lower better)'),
                ('event_auc', 'Event AUC (higher better)'),
                ('direction_nll_given_event', 'Conditional direction NLL (lower better)'))):
            values = [bundles[seed, arm]['summary']['evaluation']['test']['model'][key] for arm in ARMS]
            axes[row, col].bar(LABELS, values, color=COLORS)
            for i, value in enumerate(values):
                axes[row, col].text(i, value, f'{value:.5f}', ha='center', va='bottom')
            axes[row, col].set(title=f'Seed {seed}: {title}', ylim=(0, max(values)*1.13))
    fig.suptitle('All four TEST results; same restored context and frozen target/BCE. No TEST winner selection.')
    finish(fig, '01_probability.png')
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
    for row, seed in enumerate((17, 29)):
        for arm, color, label in zip(ARMS, COLORS, LABELS):
            b = bundles[seed, arm]
            h = b['history']
            axes[row, 0].plot(h['update'], h.validation_nll, 'o-', color=color, label=label, ms=3)
            chosen = h.loc[h['update'].eq(b['summary']['training']['best_update'])].iloc[0]
            axes[row, 0].scatter(chosen['update'], chosen.validation_nll, marker='*', s=110, color=color, edgecolor='black')
            axes[row, 1].plot(h['update'], h.gradient_clip_fraction, 'o-', color=color, ms=3)
            axes[row, 2].plot(h['update'], h.unweighted_batch_loss, 'o-', color=color, ms=3)
        axes[row, 0].set(title=f'Seed {seed}: unweighted VAL joint NLL', xlabel='Update', ylabel='NLL')
        axes[row, 1].set(title=f'Seed {seed}: gradient clipping', xlabel='Update', ylabel='Fraction of batches', ylim=(-.02, 1.02))
        axes[row, 2].set(title=f'Seed {seed}: unweighted batch loss', xlabel='Update', ylabel='BCE + conditional CE')
        axes[row, 0].legend()
    fig.suptitle('Identical initialization, sampler prefixes and optimizer. Stars: VAL-selected checkpoint.')
    finish(fig, '02_training.png')


def weights_and_gradients(bundles):
    wtable = pd.read_parquet(bundles[17, 'soft_crossing_id']['path']/'train_weights.parquet')
    y, w = wtable.y_crossing.to_numpy(), wtable.weight.to_numpy()
    if np.any(w < .5-1e-12) or np.any(w > 2+1e-12):
        raise ValueError('Weights outside registered bounds')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    names = ['NONE', 'LONG', 'SHORT', 'BOTH']
    for k, color in ((1, '#059669'), (2, '#dc2626'), (3, '#8b5cf6')):
        axes[0].hist(w[y == k], bins=np.linspace(.5, 2, 31), histtype='step', label=names[k], color=color, lw=1.5)
    axes[0].set(xlabel='Actual weight after bounded class normalization', ylabel='TRAIN roots', yscale='log', title='Bounds remain [0.5, 2.0] after normalization')
    axes[0].legend()
    for i in range(4):
        q = w[y == i]
        axes[1].bar(i, q.mean(), color='#2563eb')
        axes[1].text(i, q.mean(), f'{q.mean():.6f}\nn={len(q):,}', ha='center', va='bottom', fontsize=9)
        share = np.sort(q)[::-1][:max(1, int(np.ceil(.1*len(q))))].sum()/q.sum()
        axes[2].bar(i, 100*share, color='#d97706')
        axes[2].text(i, 100*share, f'{100*share:.1f}%', ha='center', va='bottom')
    axes[1].set(xticks=range(4), xticklabels=names, ylim=(0, 1.3), ylabel='Mean weight', title='Original mass of every joint class preserved')
    axes[2].axhline(10, color='#2563eb', ls='--', label='Uniform: 10%')
    axes[2].set(xticks=range(4), xticklabels=names, ylim=(0, 25), ylabel='% class weight mass', title='Weight mass in highest-weight 10% of roots')
    axes[2].legend()
    ess = w.sum()**2/np.square(w).sum()
    fig.suptitle(f'TRAIN only, inverse sqrt multiplicity; BOTH averages side raw weights. Kish concentration ESS {ess:,.0f}/{len(w):,}, not independent N.')
    finish(fig, '03_weights.png')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    for row, seed in enumerate((17, 29)):
        for col, arm in enumerate(ARMS):
            p = bundles[seed, arm]['p']
            pp = p.loc[p['split'].eq('train') & p.feature_available & p.y_crossing.ge(0)]
            y = pp.y_crossing.to_numpy()
            w = pp.optimization_weight_train_only.to_numpy()
            ge, gd = np.abs(pp.g_event.to_numpy()), pp.g_direction_norm.to_numpy()
            we, wd = np.abs(pp.g_event_training_weighted.to_numpy()), pp.g_direction_norm_training_weighted.to_numpy()
            if not np.allclose(we, ge*w) or not np.allclose(wd, gd*w):
                raise ValueError('Actual weighted gradients disagree')
            for offset, v, color, label in ((-.27, ge, '#93c5fd', '|gE|'), (-.09, we, '#2563eb', 'w|gE|'),
                                           (.09, gd, '#fcd34d', '||gD||'), (.27, wd, '#d97706', 'w||gD||')):
                mass = np.array([v[y == k].sum() for k in range(4)])
                mass /= max(mass.sum(), 1e-12)
                axes[row, col].bar(np.arange(4)+offset, mass, .18, label=label, color=color)
            axes[row, col].set(xticks=range(4), xticklabels=names, ylim=(0, 1), ylabel='Fraction of head gradient mass', title=f'Seed {seed}: {LABELS[col]}')
            axes[row, col].legend(ncol=4, fontsize=8)
    fig.suptitle('Selected checkpoint TRAIN gradients. Equal class weight mass does not mean equal gradient mass.')
    finish(fig, '04_gradients.png')


def fixed_signals(suite, config, store, bundles):
    f = store.frame
    cases = [(seed, case) for seed in (17, 29) for case in config['experiment']['fixed_visual_cases']]
    for number, (seed, case) in enumerate(cases):
        date = case['date']
        date_ids = np.flatnonzero(f.date.eq(date))
        t0 = int(f.t_ns.iloc[date_ids[0]])+int(case['offset_hours']*3600e9)
        ids = np.flatnonzero(f.date.eq(date) & f.t_ns.ge(t0) & f.t_ns.lt(t0+int(case['duration_hours']*3600e9)))
        scene = pd.read_parquet(Path(suite['dataset_root'])/(date+'_scene.parquet')).iloc[ids-date_ids[0]]
        if not np.array_equal(scene.t_ns, f.t_ns.iloc[ids]):
            raise ValueError('Scene clock mismatch')
        x = (f.t_ns.iloc[ids].to_numpy()-t0)/60e9
        y = f.y_crossing.iloc[ids].to_numpy()
        available = f.feature_available.iloc[ids].to_numpy()
        physical = f.physical_support_available.iloc[ids].to_numpy()
        known = available & (y >= 0)
        fig, axes = plt.subplots(7, 1, figsize=(17, 17), sharex=True, layout='constrained',
            gridspec_kw={'height_ratios': [2, 1.5, 1.5, 1, 1.3, 1.2, .5]})
        axes[0].plot(x, np.where(physical, scene.mid, np.nan), color='#111827', lw=.8, label='mid')
        for scale, color in ((1, '#f59e0b'), (3, '#8b5cf6'), (6, '#0ea5e9')):
            valid = physical & (scene[f'BB{scale}_support'].to_numpy() >= 0)
            mid = np.where(valid, scene[f'BB{scale}_middle'], np.nan)
            half = np.where(valid, scene[f'BB{scale}_half'], np.nan)
            axes[0].plot(x, mid, color=color, lw=.8, label=f'BB M{scale}')
            axes[0].fill_between(x, mid-half, mid+half, color=color, alpha=.08)
        axes[0].set_ylabel('USD / oz'); axes[0].legend(ncol=4, fontsize=8)
        axes[1].step(x, np.where(known, y > 0, np.nan), where='post', color='black', lw=.6, alpha=.5, label='Observed E')
        for arm, color, label in zip(ARMS, COLORS, LABELS):
            pp = bundles[seed, arm]['p'].iloc[ids]
            p = pp[PNAMES].to_numpy(dtype=float)
            score = 1-p[:, 0]
            marginal = p[:, 1:3]+p[:, 3, None]
            side = np.where(marginal[:, 1] > marginal[:, 0], 2, 1)
            alert = marginal.max(1) >= .35
            false = known & alert & ((y & side) == 0)
            axes[1].plot(x, np.where(available, score, np.nan), color=color, lw=.85, label=label)
            axes[2].plot(x, np.where(available, marginal.max(1), np.nan), color=color, lw=.85)
            axes[2].scatter(x[false], marginal.max(1)[false], color=color, marker='x', s=12, linewidths=.6)
            axes[3].plot(x, np.where(available, p[:, 1]-p[:, 2], np.nan), color=color, lw=.85)
            axes[4].plot(x, np.where(known, pp.g_event, np.nan), color=color, lw=.85)
            nll = -np.log(np.clip(p[np.arange(len(y)), np.maximum(y, 0)], 1e-12, 1))
            axes[5].plot(x, np.where(known, nll, np.nan), color=color, lw=.85)
        axes[1].legend(ncol=3, fontsize=8)
        axes[1].set(ylabel='E and P(E)', ylim=(-.05, 1.05))
        axes[2].axhline(.35, ls='--', color='#64748b', lw=.8)
        axes[2].set(ylabel='Max side probability', ylim=(-.05, 1.05))
        axes[2].text(.01, .92, 'x: false selected-side alert at fixed .35 (NONE or wrong direction); not necessarily an executed trade', transform=axes[2].transAxes, fontsize=8)
        axes[3].step(x, np.where(known, np.isin(y, [1, 3]).astype(float)-np.isin(y, [2, 3]).astype(float), np.nan),
                     where='post', color='black', alpha=.35, lw=.6)
        axes[3].set(ylabel='P(L)-P(S)', ylim=(-1.05, 1.05))
        axes[3].text(.01, .08, 'Target L-S = 0 for both NONE and BOTH; ribbon distinguishes them.', transform=axes[3].transAxes, fontsize=8)
        axes[4].set_ylabel('d BCE / dz, TEST'); axes[4].axhline(0, color='black', lw=.5)
        axes[5].set_ylabel('Unweighted root NLL')
        cmap = ListedColormap(['#cbd5e1', '#475569', '#059669', '#dc2626', '#8b5cf6'])
        axes[6].imshow(np.where(known, y, -1)[None, :], aspect='auto', extent=[x[0], x[-1]+5/60, 0, 1], cmap=cmap,
                       norm=BoundaryNorm(np.arange(-1.5, 4.5), 5), interpolation='nearest')
        axes[6].set_yticks([])
        axes[6].set_xlabel('Minutes from fixed start. Gray UNKNOWN/unavailable; slate NONE; green LONG; red SHORT; violet BOTH.')
        fig.suptitle(f'{date}, +{case["offset_hours"]} h from first root, seed {seed}: fixed scene; unsmoothed scores, target, gradients and loss.')
        finish(fig, f'05{chr(97+number)}_signal_{date}_seed{seed}.png')


def economic_panels(economic, bundles):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for row, seed in enumerate((17, 29)):
        for col, threshold in enumerate((.35, .5)):
            ax = axes[row, col]
            for arm, color, label in zip(ARMS, COLORS, LABELS):
                run_id = bundles[seed, arm]['summary']['run_id']
                all_rows = economic['runs'][run_id]['preregistered_fixed_threshold_test_stress']
                rows = [r for r in all_rows if r['threshold'] == threshold]
                values = [r['total_net_if_complete'] if r['completed'] else np.nan for r in rows]
                ax.plot(range(len(rows)), values, 'o-', color=color, label=label)
                for i, r in enumerate(rows):
                    if r['completed']:
                        ax.annotate(f"{r['trades']} trades", (i, r['total_net_if_complete']),
                            xytext=(3, 6 if arm == 'uniform' else -14), textcoords='offset points', color=color, fontsize=8)
                censored = [r['plan'] for r in rows if not r['completed']]
                if censored:
                    ax.text(.02, .96-.06*ARMS.index(arm), f"{label}: {', '.join(censored)} CENSORED; full-period net unavailable",
                            transform=ax.transAxes, color=color, fontsize=8, va='top', zorder=5,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.97, pad=2))
            ax.axhline(0, color='black', lw=.6)
            ax.margins(y=.15)
            ax.set(xticks=range(3), xticklabels=['cost2', 'cost4', 'cost6'],
                title=f'Seed {seed}: fixed TEST threshold {threshold:.2f}', ylabel='Complete-period net, USD / reference oz')
            ax.legend(fontsize=8)
    fig.suptitle('Prespecified execution stress only. No choice on TEST; censored partial cash is never a full-period net.')
    finish(fig, '09_execution.png')


def false_signal_panels(audit, bundles):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for row, seed in enumerate((17, 29)):
        for col, channel in enumerate(('event', 'selected_side')):
            for arm, color, label in zip(ARMS, COLORS, LABELS):
                run_id = bundles[seed, arm]['summary']['run_id']
                rows = audit['runs'][run_id]['false_signals']['by_split']['test'][channel+'_fixed']
                x = [100*r['none_fpr'] for r in rows]
                y = [100*r['event_opportunity_recall'] for r in rows]
                axes[row, col].plot(x, y, 'o-', color=color, label=label)
                for xx, yy, r in zip(x, y, rows):
                    if r['threshold'] > .5:
                        continue
                    axes[row, col].annotate(f"{r['threshold']:.2f}", (xx, yy), xytext=(4, 4 if arm == 'uniform' else -12),
                                           textcoords='offset points', fontsize=8, color=color)
            axes[row, col].set(title=f'Seed {seed}: {channel}', xlabel='False alerts / known NONE roots (%)',
                ylabel='Correct alerts / ALL known event roots (%)', xlim=(-.3, None), ylim=(-.3, None))
            axes[row, col].legend(fontsize=9)
            axes[row, col].text(.98, .03, 'Cuts .65 and .80 are near the origin; exact counts in audit JSON.',
                               transform=axes[row, col].transAxes, ha='right', fontsize=8)
    fig.suptitle('TEST trade-off: all four fixed cuts .35 / .50 / .65 / .80 shown; .35 / .50 labeled. No threshold selection.\nSide recall counts a wrong-side decision as a miss; BOTH permits either side.')
    finish(fig, '06_false_signal_tradeoff.png')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    for row, seed in enumerate((17, 29)):
        for col, (metric, title) in enumerate((('none_fpr', 'NONE false-positive rate'),
                 ('event_opportunity_recall', 'Correct opportunity recall'), ('precision', 'Precision among known alerts'))):
            for j, (arm, color, label) in enumerate(zip(ARMS, COLORS, LABELS)):
                run_id = bundles[seed, arm]['summary']['run_id']
                entries = audit['runs'][run_id]['false_signals']['calibrated_none_fpr']
                values = [entries[channel]['test'][metric] for channel in ('event', 'selected_side')]
                xx = np.arange(2)+(j-.5)*.32
                axes[row, col].bar(xx, [np.nan if v is None else 100*v for v in values], .3, color=color, label=label)
                for i, value in enumerate(values):
                    if value is not None:
                        axes[row, col].text(xx[i], 100*value, f'{100*value:.2f}%', ha='center', va='bottom', fontsize=8)
            axes[row, col].set(xticks=range(2), xticklabels=['Event', 'Selected side'], ylabel='TEST %', title=f'Seed {seed}: {title}')
            axes[row, col].set_ylim(0, max(6, axes[row, col].get_ylim()[1]*1.13))
            if col == 0:
                axes[row, col].axhline(5, color='black', ls='--', lw=.8, label='CAL budget 5%')
            axes[row, col].legend(fontsize=8)
    fig.suptitle('Frozen CAL threshold: at most 5% NONE-FPR on CAL, conservative ties; applied unchanged to TEST.\nTEST false-positive rate is measured, not constrained. E and side thresholds are selected separately.')
    finish(fig, '07_calibration_transfer.png')
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    for row, seed in enumerate((17, 29)):
        for col, role in enumerate(('validation', 'test')):
            axes[row, col].plot([0, 1], [0, 1], '--', color='#94a3b8')
            for arm, color, label in zip(ARMS, COLORS, LABELS):
                bins = bundles[seed, arm]['summary']['evaluation'][role]['model']['reliability']
                bins = [r for r in bins if r['count']]
                axes[row, col].plot([r['probability'] for r in bins], [r['frequency'] for r in bins], 'o-', color=color, label=label, ms=4)
                for r in bins:
                    if r['count'] < 50:
                        axes[row, col].plot(r['probability'], r['frequency'], 'o', mfc='white', mec=color, ms=6)
            axes[row, col].set(title=f'Seed {seed}: {role}', xlabel='Mean P(E), fixed 0.1 bins', ylabel='Observed E fraction', xlim=(0, 1), ylim=(0, 1))
            axes[row, col].legend()
    fig.suptitle('Uniform-root event reliability. Open markers: fewer than 50 correlated roots; sparse tails are uncertain.')
    finish(fig, '08_reliability.png')


def paired_panels(audit):
    metrics = [('joint_nll', 'Joint NLL'), ('event_auc', 'Event AUC'),
               ('direction_nll_given_event', 'Conditional direction NLL'),
               ('event_id_positive_side_nll', 'Positive-only side-ID NLL'),
               ('none_nll', 'NONE event NLL'), ('positive_event_nll', 'Positive event NLL')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
    for ax, (key, title) in zip(axes.flat, metrics):
        result = audit['paired']['by_split']['test'][key]
        sources = [(result['by_seed']['17'], 'Seed17', '#2563eb'),
                   (result['by_seed']['29'], 'Seed29', '#d97706'),
                   (result['mean_seed_effect_paired_by_date'], 'Mean seeds per day', '#111827')]
        for row, (v, label, color) in enumerate(sources):
            lo, hi = v['percentile_95_interval']
            ax.errorbar(v['mean'], row, xerr=[[max(0, v['mean']-lo)], [max(0, hi-v['mean'])]],
                        fmt='o', color=color, capsize=4)
            ax.text(.02, row+.22, f'{v["mean"]:+.5f} [{lo:+.5f}, {hi:+.5f}]',
                    transform=ax.get_yaxis_transform(), va='top', fontsize=8, color=color)
        ax.axvline(0, color='#64748b', ls='--', lw=.8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set(yticks=range(3), yticklabels=[r[1] for r in sources], ylim=(2.55, -.9),
               title=title, xlabel='Soft minus uniform utility; positive = improvement')
    fig.suptitle('Ten paired TEST days, descriptive 95% day-bootstrap intervals. Seeds averaged within date first.\nPreviously inspected blocks, dependent days and multiple metrics: intervals are not confirmatory significance tests.')
    finish(fig, '10_paired_effects.png')


def main():
    suite, config, store, bundles, inputs = load_evidence()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .17})
    probability_and_training(bundles)
    weights_and_gradients(bundles)
    fixed_signals(suite, config, store, bundles)
    economic = read(ROOT/'economic_audit.json')
    if economic['status'] not in ('PASS', 'COMPLETE'):
        raise ValueError('Independent economic audit required')
    inputs[str(ROOT/'economic_audit.json')] = sha(ROOT/'economic_audit.json')
    economic_panels(economic, bundles)
    audit = read(ROOT/'statistical_audit.json')
    if audit['status'] not in ('PASS', 'COMPLETE'):
        raise ValueError('Independent statistical audit required')
    inputs[str(ROOT/'statistical_audit.json')] = sha(ROOT/'statistical_audit.json')
    false_signal_panels(audit, bundles)
    paired_panels(audit)
    write_report(suite, config, store, bundles, inputs)


def write_report(suite, config, store, bundles, inputs):
    audit = read(ROOT/'statistical_audit.json')
    economic = read(ROOT/'economic_audit.json')
    findings = read(ROOT/'decision_findings.json')
    inputs[str(ROOT/'decision_findings.json')] = sha(ROOT/'decision_findings.json')
    rows, alarm_rows, econ_rows, stress_rows = [], [], [], []
    def number(value, digits=4):
        return '—' if value is None else f'{value:.{digits}f}'
    for seed in (17, 29):
        for arm, label in zip(ARMS, LABELS):
            b = bundles[seed, arm]
            s = b['summary']; run_id = s['run_id']
            m = s['evaluation']['test']['model']
            rows.append(f'<tr><td>{seed}</td><td>{label}</td><td>{m["joint_nll"]:.6f}</td><td>{m["event_auc"]:.6f}</td><td>{m["direction_nll_given_event"]:.6f}</td><td>{s["training"]["best_update"]}</td></tr>')
            for channel, a in audit['runs'][run_id]['false_signals']['calibrated_none_fpr'].items():
                v = a['test']
                alarm_rows.append(f'<tr><td>{seed} / {label}</td><td>{channel}</td><td>{a["selection"]["threshold"]:.17g}</td><td>{a["calibration"]["none_fpr"]:.2%}</td><td>{v["none_fpr"]:.2%}</td><td>{v["event_opportunity_recall"]:.2%}</td><td>{number(v["precision"])}</td><td>{v["alerts"]}</td><td>{v["none_false_positives"]}</td><td>{v["wrong_side_alerts"]}</td><td>{v["unknown_alerts"]}</td></tr>')
            e = economic['runs'][run_id]
            selected, replay = e['selected_from_calibration'], e['selected_replay']['test']
            econ_rows.append(f'<tr><td>{seed} / {label}</td><td>{selected["plan"]}, ≥{selected["threshold"]:.2f}</td><td>{replay["status"]}</td><td>{replay["trades"]}</td><td>{number(replay["total_net_if_complete"], 2)}</td></tr>')
            for v in e['preregistered_fixed_threshold_test_stress']:
                bk = v['teacher_buckets_after_execution']
                cell = lambda name: f'{bk[name]["complete_trades"]} / {bk[name]["observed_closed_net"]:+.2f}'
                stress_rows.append(f'<tr><td>{seed} / {label}</td><td>{v["plan"]} / {v["threshold"]:.2f}</td><td>{v["status"]}</td><td>{v["trades"]}</td><td>{number(v["total_net_if_complete"], 2)}</td><td>{cell("TEACHER_NONE")}</td><td>{cell("WRONG_SIDE")}</td><td>{cell("TARGET_POSITIVE_SIDE")}</td><td>{cell("UNKNOWN_TEACHER")}</td></tr>')
    images = sorted(ROOT.glob('*.png'))
    figures = ''.join(f'<article><h2>{html.escape(p.stem)}</h2><a href="{p.name}"><img src="{p.name}" loading="lazy"></a></article>' for p in images)
    conclusions = ''.join('<p>'+html.escape(v)+'</p>' for v in findings['findings_ru'])
    document = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V4 · Мягкие crossing-ID веса</title><style>body{{margin:0;background:#eef2f6;color:#17243a;font:16px/1.55 system-ui}}main{{max-width:1550px;margin:auto;padding:30px}}article,header{{background:white;border:1px solid #dbe3ee;border-radius:12px;padding:24px;margin-bottom:22px}}h1{{font-size:32px}}h2{{font-size:22px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #dbe3ee;text-align:left}}th{{background:#eef2f6}}.scroll{{overflow:auto}}a{{color:#1d4ed8}}code{{background:#eef2f6;padding:2px 5px}}</style><main>
<header><h1>Мягкие crossing-ID веса на восстановленном контексте</h1>
<p>Четыре свежих обучения: uniform / bounded sqrt-ID × seeds 17, 29. Общий dataset V3: 36 сессий, 362 664 roots; TRAIN 16 / VAL 5 / CAL 5 / TEST 10 дней. Таргет, BCE + условная CE, архитектура и исполнение сохранены.</p>
<p><code>r=m(ID)^(-0.5)</code>; BOTH — среднее двух r; NONE=1. По каждому положительному joint-классу решается <code>sum clip(a·r, 0.5, 2.0)=N</code>. Границы действуют после нормализации. ID не входят в X; веса вычислены только по TRAIN. Основные VAL/CAL/TEST root-метрики невзвешенные, отдельные ID-balanced диагностики явно выделены.</p>
<p>TEST уже исследовался в V3; это контролируемая проверка гипотезы на ранее просмотренных временных блоках. Выбор победителя по TEST не выполняется. Сигналы и таргеты на графиках не сглажены.</p></header>
<article><h2>Выводы</h2>{conclusions}</article>
<article><h2>Вероятностные метрики TEST</h2><div class="scroll"><table><tr><th>Seed</th><th>Веса</th><th>Joint NLL ↓</th><th>Event AUC ↑</th><th>D NLL ↓</th><th>VAL checkpoint</th></tr>{''.join(rows)}</table></div>
<p>Два seed — репликации, а не дополнительные независимые дни. Positive-ID score рассматривается отдельно от качества всей популяции, включая NONE.</p></article>
<article><h2>Порог CAL под 5% NONE-FPR → неизменённый TEST</h2><div class="scroll"><table><tr><th>Условие</th><th>Канал</th><th>Порог CAL</th><th>CAL FPR</th><th>TEST FPR</th><th>TEST recall</th><th>Precision</th><th>Alerts</th><th>NONE FP</th><th>Wrong side</th><th>UNKNOWN alerts</th></tr>{''.join(alarm_rows)}</table></div>
<p>Порог зависит только от отрицательных CAL scores; совпадения на границе обрабатываются консервативно. Recall стороны использует общий знаменатель всех известных положительных roots. UNKNOWN не включён в NONE и не объявлен ложным сигналом. Это диагностика сигналов, не торговая политика.</p></article>
<article><h2>Штатная политика: выбор по экономике CAL, исполнение TEST</h2><div class="scroll"><table><tr><th>Условие</th><th>Выбор CAL</th><th>Статус TEST</th><th>Сделки</th><th>Полный net</th></tr>{''.join(econ_rows)}</table></div></article>
<article><h2>Заранее заданный stress исполнения TEST</h2><div class="scroll"><table><tr><th>Условие</th><th>Risk / threshold</th><th>Статус</th><th>Сделки</th><th>Полный net</th><th>NONE n / cash</th><th>Wrong side n / cash</th><th>Target-positive n / cash</th><th>UNKNOWN n / cash</th></tr>{''.join(stress_rows)}</table></div>
<p>USD на reference ounce; taker BBO и прежние комиссии. Метки присоединяются после причинного однопозиционного replay. В CENSORED строках bucket cash описывает только наблюдённый префикс, полного net нет. STOP может предшествовать будущему благоприятному crossing. Значение таргета не равно прибыли сделки.</p></article>
{figures}<article><h2>Проверяемые артефакты и ограничения</h2><p><a href="statistical_audit.json">Статистический аудит</a> · <a href="economic_audit.json">Независимый replay</a> · <a href="decision_findings.json">Решения</a> · <a href="sol61_visual_audit.json">Sol 6.1</a> · <a href="completion_receipt.json">Итоговая проверка</a></p>
<p>TCN Micro использует 70 признаков, 141 входной канал и до 300 секунд подробной истории. Длинный контекст «сегодня и вчера с открытия Лондона» и богатые VWAP/STFT здесь не обучались. Наличие причинного контекста не доказывает использование всей его глубины моделью. Более низкий loss, рост positive-ID coverage или отдельный положительный cash не подтверждают экономический допуск.</p></article></main></html>'''
    (ROOT/'index.html').write_text(document, encoding='utf-8', newline='\n')
    save(ROOT/'visual_manifest.json', dict(status='COMPLETE', source_sha256=sha(__file__), inputs=inputs,
        figures={p.name: sha(p) for p in images}, index_sha256=sha(ROOT/'index.html'),
        unsmoothed=True, fixed_visual_cases=config['experiment']['fixed_visual_cases'],
        interpretation='All prespecified arms and scenes shown; statistical diagnostics and execution remain separate'))
    print(json.dumps(dict(status='COMPLETE', images=len(images), html=str(ROOT/'index.html'))))


if __name__ == '__main__':
    main()
