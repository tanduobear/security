"""Single-panel enlarged G05 Validation PR; no full-range view or inset."""
import argparse
import hashlib
from io import BytesIO
import json
import os

from . import pr_curve_data as original
from .pr_curve_data import verified_g05_curves

ROOT = original.ROOT
OUTPUT_PNG = 'figures/fig_m1_pr.png'
OUTPUT_PDF = 'figures/fig_m1_pr.pdf'
OUTPUT_REPORT = 'docs/m1_pr_curve_enlarged_report.md'
TITLE = 'GBDT G05 — Validation PR Curve (Enlarged View)'
CAPTION = 'Axes show local ranges: Recall 0.98–1.00; Precision 0.99–1.00.'


def render(curves):
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.tmp/m1_pr_mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    styles = ['-', '--', '-.', ':', (0, (5, 1, 1, 1, 1, 1))]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'axes.labelsize': 12, 'axes.titlesize': 12.5, 'legend.fontsize': 9.5,
                         'axes.linewidth': .8, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'path.simplify': False, 'pdf.fonttype': 42, 'pdf.compression': 9}):
        fig, ax = plt.subplots(figsize=(8, 5.8))
        fig.subplots_adjust(left=.11, right=.97, bottom=.30, top=.90)
        try:
            for curve, color, style in zip(curves, colors, styles):
                ax.step(curve['recall'], curve['precision'], where='post', color=color, linestyle=style,
                        linewidth=1.5, label=f"Seed {curve['seed']}  (AP = {curve['average_precision']:.6f})")
            ax.set(xlim=(.98, 1), ylim=(.99, 1), xlabel='Recall', ylabel='Precision', title=TITLE)
            ax.set_xticks([.98, .985, .99, .995, 1])
            ax.set_yticks([.99, .992, .994, .996, .998, 1])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.grid(True, color='#D9DDE3', linewidth=.55, alpha=.65)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2, frameon=False,
                      columnspacing=1.8, handlelength=3.2, borderaxespad=0)
            fig.text(.5, .025, CAPTION, ha='center', fontsize=9, color='#444444')
            original.require(len(fig.axes) == 1 and not ax.child_axes and ax.get_xlim() == (.98, 1) and ax.get_ylim() == (.99, 1),
                             'Single enlarged panel required')
            png, pdf = BytesIO(), BytesIO()
            fig.savefig(png, format='png', dpi=300, metadata={'Title': TITLE, 'Description': CAPTION + ' Validation only; no Test evaluation.'})
            fig.savefig(pdf, format='pdf', metadata={'Title': TITLE, 'Author': 'P4', 'Subject': CAPTION,
                                                   'CreationDate': None, 'ModDate': None})
            return png.getvalue(), pdf.getvalue()
        finally:
            plt.close(fig)


def report(curves, metadata, png, pdf):
    import matplotlib
    import numpy
    import sklearn
    lines = ['# GBDT G05 Validation PR Curve — Enlarged View验收', '',
             '输出：[PNG](../figures/fig_m1_pr.png)、[PDF](../figures/fig_m1_pr.pdf)。仅为Validation，不代表Test评估。', '',
             '## 图形规范', '',
             '- 标题：GBDT G05 — Validation PR Curve (Enlarged View)。',
             '- 仅一个Enlarged View面板，无Full-range View、无inset、无双面板。',
             '- Recall 0.98–1.00，刻度间隔0.005；Precision 0.99–1.00，刻度间隔0.002；两轴三位小数。',
             '- 图内图注明确声明局部坐标范围：' + CAPTION,
             '- 五个种子独立曲线，不同颜色和线型；图例位于图外下方，AP保留六位小数。AP来自完整Validation预测，不是局部图范围内积分。',
             '- sklearn precision_recall_curve(y_true,y_score,pos_label=1)，完整阈值点，step(where="post")；不平滑、不平均、不从AP反推曲线，不简化路径。',
             '- 白底、DejaVu Sans、轻网格；PNG 2400×1740、300 DPI；PDF为同图矢量导出，嵌入字体。', '',
             '## 原始数据和独立验证', '',
             '- 本地封存归档定位名：`' + metadata['run_relative'] + '/`，仅为本地定位说明，非在线链接。',
             '- m1.json、原selection及任务确认G05；original / stratified / main、P2固定140特征，配置及数据绑定一致。',
             f'- 归档{metadata["archive_files_verified"]}/302文件在绘图前后通过SHA256；Windows只读共享句柄阻止归档并发写入或删除。',
             f'- 每种子{curves[0]["rows"]:,}行，良性{curves[0]["benign_rows"]:,}行、攻击{curves[0]["attack_rows"]:,}行；行ID唯一、五种子行序和标签逐项一致。',
             '- 行ID摘要：`' + curves[0]['row_ids_sha256'] + '`。',
             '- 标签uint8摘要：`' + curves[0]['labels_uint8_sha256'] + '`。',
             '- 独立调用average_precision_score，从真实y_true及连续攻击概率y_score复算，五次AP与封存任务及m1.json严格相等，差值均为0。', '',
             '| Seed | 独立复算AP | 本地归档内预测路径 | 预测SHA256 |', '|---|---:|---|---|',
             *[f"| {c['seed']} | {c['average_precision']!r} | `{c['prediction_path']}` | `{c['prediction_sha256']}` |" for c in curves], '',
             '参数：`' + json.dumps(metadata['parameters'], sort_keys=True) + '`。', '',
             'P2输入身份（只核对封存binding，不打开Parquet）：', '',
             *[f'- `{k}`：`{v}`。' for k, v in metadata['input_file_sha256'].items()], '',
             '## SHA256', '',
             '- PNG：`' + hashlib.sha256(png).hexdigest() + '`。',
             '- PDF：`' + hashlib.sha256(pdf).hexdigest() + '`。',
             '- 绘图脚本：`' + original.sha(__file__) + '`。',
             '- 数据核验脚本pr_curve_data.py：`' + original.sha(ROOT / 'methods/m1/pr_curve_data.py') + '`。',
             *[f'- `{k}`：`{metadata[k]}`。' for k in ['manifest_sha256', 'selection_sha256', 'config_sha256', 'm1_sha256']], '',
             '## 复现与边界', '',
             f'环境：sklearn {sklearn.__version__}、NumPy {numpy.__version__}、Matplotlib {matplotlib.__version__}。', '',
             '```powershell', '.\\.venv\\Scripts\\python.exe -B -X utf8 -m methods.m1.plot_pr_curve_enlarged', '```', '',
             '需要本地完整归档。仅解码五个G05 Validation预测，其他原归档文件只做字节哈希；不读取Test、不加载模型、不训练、不重新预测、不运行Bootstrap或阈值标定。原模型、预测、封存归档及P2/P3保持不变。旧绘图版本按用户要求已清理；数据核验逻辑独立保存在pr_curve_data.py。已有PNG/PDF字节相同则保持原样，不同则拒绝覆盖；--refresh-report仅允许在校验通过且图片不变时更新脚本依赖哈希报告。', '']
    return '\n'.join(lines).encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-report', action='store_true', help='Refresh audit hashes only; existing PNG/PDF must remain byte-identical')
    args = parser.parse_args()
    # Publish only after the context exits and the final archive hash pass succeeds.
    with verified_g05_curves() as (curves, metadata):
        png, pdf = render(curves)
        audit = report(curves, metadata, png, pdf)
    outputs = [(ROOT / OUTPUT_PNG, png), (ROOT / OUTPUT_PDF, pdf), (ROOT / OUTPUT_REPORT, audit)]
    for path, payload in outputs:
        original.require((args.refresh_report and path == outputs[2][0]) or not path.exists() or path.read_bytes() == payload,
                         'Refuse different existing output: ' + path.name)
    for path, payload in outputs:
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(payload)
        elif args.refresh_report and path == outputs[2][0] and path.read_bytes() != payload:
            path.write_bytes(payload)
        original.require(original.sha(path) == hashlib.sha256(payload).hexdigest(), 'Published hash mismatch')
    print(json.dumps({'png_sha256': original.sha(outputs[0][0]), 'pdf_sha256': original.sha(outputs[1][0]),
                      'AP_exact_matches': 5, 'archive_hashes_verified': metadata['archive_files_verified'], 'test_accessed': False}, indent=2))


if __name__ == '__main__':
    main()
