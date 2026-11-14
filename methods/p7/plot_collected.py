"""Plot only collected real metrics; version deltas are differences of means."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ['M1', 'M2', 'M3']
LABELS = {'M1': 'GBDT', 'M2': '1D-CNN', 'M3': 'Autoencoder'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-original', action='store_true')
    args = parser.parse_args()
    base = ROOT / ('results/p7_preview' if args.preview_original else 'results')
    data = json.loads((base / 'p7_figure_data.json').read_text(encoding='utf-8'))
    groups = {(g['method'], g['dataset_version']): g for g in data['groups']}
    versions = ['original'] if args.preview_original else ['original', 'corrected']
    if not args.preview_original and data['status'] != 'formal_corrected_integration':
        raise ValueError('Final figures require accepted corrected formal results')
    if not all((method, version) in groups for method in METHODS for version in versions):
        raise ValueError('Missing method/version; do not draw an incomplete final comparison')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out = base / 'figures' if args.preview_original else ROOT / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'figure.facecolor': 'white'})
    order = [(m, v) for m in METHODS for v in versions]
    labels = [f'{LABELS[m]} {groups[m,v]["configuration"]} / {v}' for m, v in order]
    colors = {'original': '#0072B2', 'corrected': '#D55E00'}
    values_table = []

    def save(fig, stem):
        fig.savefig(out / (stem + '.png'), dpi=300, bbox_inches='tight')
        fig.savefig(out / (stem + '.pdf'), bbox_inches='tight')
        plt.close(fig)

    def clean(ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)

    def pair_plot(specifications, stem, title):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
        for ax, (key, label) in zip(axes, specifications):
            for i, (method, version) in enumerate(order):
                metric = groups[method, version][key]
                ax.errorbar(metric['mean'], i, xerr=metric['sample_std'],
                            fmt='o' if version == 'original' else 's',
                            color=colors[version], capsize=3)
                values_table.append({'figure': stem, 'method': method, 'version': version,
                    'configuration': groups[method, version]['configuration'], 'metric': key,
                    'attack_family': '', 'support': '', 'mean': metric['mean'],
                    'sample_std': metric['sample_std']})
            ax.set(xlabel=label, xlim=(0, 1.025))
            clean(ax)
        axes[0].set_yticks(range(len(order)), labels)
        axes[0].invert_yaxis()
        fig.suptitle(title)
        fig.text(.5, .01, 'Test; seeds 42–46; error bars are sample SD (ddof=1).', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        save(fig, stem)

    pair_plot([('ap', 'Average precision'), ('roc_auc', 'ROC-AUC')],
              'fig_p7_overall_performance', 'Ranking performance by dataset version')
    pair_plot([('tpr_at_fpr_1pct', 'Empirical Test TPR at FPR ≤ 1%'),
               ('tpr_at_fpr_5pct', 'Empirical Test TPR at FPR ≤ 5%')],
              'fig_p7_tpr_at_fpr', 'Empirical ROC operating points; not validation-fixed thresholds')
    families = sorted(set().union(*(groups[key]['family'] for key in order)))
    matrix = np.full((len(families), len(order)), np.nan)
    supports = np.zeros_like(matrix, dtype=int)
    for i, family in enumerate(families):
        for j, key in enumerate(order):
            group = groups[key]['family'].get(family, {'mean': None, 'sample_std': None, 'support': 0})
            supports[i, j] = group['support']
            if group['mean'] is not None:
                matrix[i, j] = group['mean']
            values_table.append({'figure': 'fig_p7_family_recall', 'method': key[0], 'version': key[1],
                'configuration': groups[key]['configuration'], 'metric': 'family_recall',
                'attack_family': family, 'support': group['support'],
                'mean': group['mean'], 'sample_std': group['sample_std']})
    fig, ax = plt.subplots(figsize=(max(9, len(order) * 1.8), max(7, len(families) * .45)))
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#eeeeee')
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(order)), labels, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(families)), families, fontsize=9)
    for i in range(len(families)):
        for j in range(len(order)):
            value = matrix[i, j]
            text = '--\n(n=0)' if np.isnan(value) else f'{value:.2f}\n(n={supports[i,j]})'
            ax.text(j, i, text, ha='center', va='center', fontsize=7,
                    color='white' if not np.isnan(value) and value > .65 else '#222222')
    ax.set_title('Attack-family recall: frozen low-FPR rules; absent families remain undefined')
    fig.colorbar(im, ax=ax, label='Five-seed mean recall')
    fig.tight_layout()
    save(fig, 'fig_p7_family_recall')
    if not args.preview_original:
        deltas, delta_labels, bar_colors = [], [], []
        for method in METHODS:
            for metric in ['ap', 'roc_auc', 'tpr_at_fpr_1pct', 'tpr_at_fpr_5pct']:
                delta = groups[method, 'corrected'][metric]['mean'] - groups[method, 'original'][metric]['mean']
                deltas.append(delta)
                delta_labels.append(f'{LABELS[method]} / {metric}')
                bar_colors.append('#009E73' if delta >= 0 else '#D55E00')
                values_table.append({'figure': 'fig_p7_version_delta', 'method': method,
                    'version': 'corrected_minus_original', 'configuration': '', 'metric': metric,
                    'attack_family': '', 'support': '', 'mean': delta, 'sample_std': None})
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(len(deltas)), deltas, color=bar_colors)
        ax.set_yticks(range(len(deltas)), delta_labels)
        ax.invert_yaxis()
        ax.axvline(0, color='#333333', linewidth=.8)
        ax.set_xlabel('Corrected mean minus original mean')
        ax.set_title('Dataset-version sensitivity')
        clean(ax)
        fig.text(.5, .01, 'Independent evaluation populations; these deltas are not paired confidence intervals.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, 1))
        save(fig, 'fig_p7_version_delta')
    with (base / 'p7_plot_values.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values_table[0]))
        writer.writeheader()
        writer.writerows(values_table)
    print('Saved figures:', out, flush=True)
    print('Saved plot values:', base / 'p7_plot_values.csv', flush=True)


if __name__ == '__main__':
    main()
