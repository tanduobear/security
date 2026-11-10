"""Improve the error-distribution plot using saved scores only; no inference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = ROOT / 'results/m3_runs'
    record = json.loads((folder / 'records/Autoencoder_seed42_q95.json').read_text(encoding='utf-8'))
    record99 = json.loads((folder / 'records/Autoencoder_seed42_q99.json').read_text(encoding='utf-8'))
    frame = pd.read_parquet(folder / 'predictions/Autoencoder_seed42.parquet')
    values = np.log1p(frame.score.to_numpy())
    thresholds = [np.log1p(r['decision_rule']['threshold']) for r in [record, record99]]
    full_max = float(values.max())
    zoom_max = max(float(np.quantile(values, 0.99)), max(thresholds) * 1.25)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, upper, title in zip(axes, [full_max, zoom_max],
                                 ['Full range', 'Central range (tails omitted)']):
        bins = np.linspace(0, upper, 100)
        for label, name, color in [(0, 'Benign', '#4477aa'), (1, 'Attack', '#ee8844')]:
            group = values[frame.y_binary.to_numpy() == label]
            counts, _ = np.histogram(group, bins=bins)
            # Normalize by ALL class rows; do not renormalize the truncated view.
            density = counts / (len(group) * np.diff(bins))
            ax.stairs(density, bins, color=color, label=name, linewidth=1.5)
        for threshold, name, color, style in zip(thresholds, ['Validation q95', 'Validation q99'],
                                                 ['#228833', '#aa3377'], ['--', ':']):
            ax.axvline(threshold, color=color, linestyle=style, label=name)
        ax.set(xlabel='log(1 + reconstruction MSE)', ylabel='Density', title=title, xlim=(0, upper))
        ax.grid(alpha=0.15)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Density (log scale)')
    retained = [100 * float(np.mean(values[frame.y_binary.to_numpy() == label] <= zoom_max))
                for label in [0, 1]]
    axes[1].text(0.98, 0.98, f'Shown: {retained[0]:.1f}% benign; {retained[1]:.1f}% attack',
                 ha='right', va='top', transform=axes[1].transAxes, fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle('Selected autoencoder A06, seed 42: test reconstruction errors')
    fig.tight_layout()
    path = ROOT / 'figures/fig_m3_error_dist_v2.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print('Saved:', path)
    print('Original plot retained. Both panels use common bins across classes.')


if __name__ == '__main__':
    main()
