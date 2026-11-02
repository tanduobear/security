"""Synthetic layout acceptance for the requested single enlarged PR view."""
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from methods.m1 import plot_pr_curve_enlarged as enlarged


class EnlargedPRTests(unittest.TestCase):
    def test_single_local_range_axes_ticks_legend_caption_and_exports(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        curves = [{'seed': seed, 'average_precision': .9998,
                   'precision': np.array([.2, .99, 1.]), 'recall': np.array([1., .98, 0.])}
                  for seed in [42, 43, 44, 45, 46]]
        close, checked = plt.close, []
        def inspect(fig):
            if hasattr(fig, 'axes'):
                self.assertEqual(len(fig.axes), 1)
                ax = fig.axes[0]
                self.assertFalse(ax.child_axes)
                self.assertEqual(ax.get_xlim(), (.98, 1))
                self.assertEqual(ax.get_ylim(), (.99, 1))
                np.testing.assert_allclose(np.diff(ax.get_xticks()), .005)
                np.testing.assert_allclose(np.diff(ax.get_yticks()), .002)
                self.assertEqual(ax.get_title(), enlarged.TITLE)
                self.assertIn(enlarged.CAPTION, [text.get_text() for text in fig.texts])
                self.assertEqual(len(ax.lines), 5)
                self.assertEqual(len({line.get_color() for line in ax.lines}), 5)
                self.assertEqual(len({str(line._dash_pattern) for line in ax.lines}), 5)
                self.assertLess(ax.get_legend().get_window_extent().y1, ax.get_window_extent().y0)
                self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()],
                                 [f'Seed {seed}  (AP = 0.999800)' for seed in [42, 43, 44, 45, 46]])
                checked.append(True)
            close(fig)
        with patch.object(plt, 'close', side_effect=inspect):
            png, pdf = enlarged.render(curves)
        self.assertEqual(checked, [True])
        with Image.open(BytesIO(png)) as image:
            self.assertEqual(image.size, (2400, 1740))
            self.assertAlmostEqual(image.info['dpi'][0], 300, places=1)
        self.assertTrue(pdf.startswith(b'%PDF-'))
        self.assertIn(b'/FontFile2', pdf)
        # Exercise publication with a synthetic guarded-data substitute. This
        # protects the agreed delivery names without opening a real archive.
        @contextmanager
        def synthetic_data():
            yield curves, {'archive_files_verified': 302}
        with tempfile.TemporaryDirectory(dir=ROOT / '.tmp') as directory:
            project = Path(directory)
            (project / 'figures').mkdir()
            (project / 'docs').mkdir()
            with patch.object(enlarged, 'ROOT', project), patch.object(enlarged, 'verified_g05_curves', synthetic_data), \
                 patch.object(enlarged, 'render', return_value=(png, pdf)), \
                 patch.object(enlarged, 'report', return_value=b'SYNTHETIC REPORT'), \
                 patch.object(sys, 'argv', ['plot_pr_curve_enlarged']):
                enlarged.main()
            self.assertEqual((project / 'figures/fig_m1_pr.png').read_bytes(), png)
            self.assertEqual((project / 'figures/fig_m1_pr.pdf').read_bytes(), pdf)
            self.assertEqual((project / 'docs/m1_pr_curve_enlarged_report.md').read_bytes(), b'SYNTHETIC REPORT')
            self.assertFalse(list((project / 'figures').glob('*_enlarged.*')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
