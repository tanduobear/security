"""Export G05 Test PR from sealed predictions; no model/Parquet loading or fitting.

Run from the repository root with the original .venv:
    python -B -X utf8 -m tools.plot_m1_test_pr
Existing outputs must be byte-identical; differing outputs are never replaced.
"""
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import statistics

from methods.m1.pr_curve_data import ROOT, SEEDS, digest, require, safe_path, sha, verified_curve
from methods.m1.seal_gbdt import artifact_paths, file_inventory, freeze_files
from methods.m1.search_gbdt import resource_slot

SUMMARY = 'results/m1_test_summary/summary.json'
SUMMARY_SHA = 'cb2a95db154792d5dc0d3fb5f2710aba3ed6cc04f12a964e958cf26f371c23e3'
LABEL_SHA = '7096835d7bca2b55d6dc5743f68046466cea0e6153a36eb20d8836b9bf0286c8'
PNG = 'figures/fig_m1_test_pr.png'
PDF = 'figures/fig_m1_test_pr.pdf'
REPORT = 'docs/m1_test_pr_curve_report.md'
TITLE = 'GBDT G05 — Test PR Curve (Enlarged View)'
CAPTION = 'Local axes: Recall 0.98–1.00; Precision 0.99–1.00. AP uses the complete Test set.'


def read(path):
    return json.loads(Path(path).read_bytes())


@contextmanager
def verified_test_curves():
    """Freeze sources, check identity/byte hashes, then verify real per-seed scores."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from threadpoolctl import threadpool_limits

    public = ROOT / SUMMARY
    require(sha(public) == SUMMARY_SHA, 'Public Test summary pin mismatch')
    summary = read(public)
    provenance = summary['provenance']
    run = safe_path(ROOT, provenance['local_archive'])
    train = safe_path(ROOT, provenance['original_archives']['GBDT']['path'])
    m1 = ROOT / provenance['validation_summary_path']
    config_path = ROOT / provenance['config_path']
    protocol_path = ROOT / 'methods/m1/protocol.unified.v1.json'
    selected = read(m1)['gbdt']['selected_candidate']
    critical = ['MANIFEST.json', 'selection.json', 'binding.json', 'config.json']
    for seed in SEEDS:
        relative = f'completed/G05_seed_{seed}.json'
        critical.append(relative)
        directory = read(train / relative)['artifact_directory']
        critical += [f'{directory}/task.json', f'{directory}/model.joblib']
    training_paths = [safe_path(train, name) for name in critical]
    protected = [*artifact_paths(run).values(), run / 'MANIFEST.json', public, m1,
                 config_path, protocol_path, *training_paths]
    with resource_slot(), freeze_files(protected):
        require(sha(public) == SUMMARY_SHA, 'Test summary changed before freeze')
        require(sha(run / 'MANIFEST.json') == provenance['manifest_sha256'], 'Test MANIFEST pin mismatch')
        manifest = read(run / 'MANIFEST.json')
        require(manifest['status'] == 'p4_test_evaluation_complete' and
                len(manifest['files']) == provenance['manifest_entries'] == 82 and
                file_inventory(run) == manifest['files'], 'Test archive hash/file-set mismatch')
        require(sha(m1) == provenance['validation_summary_sha256'] and
                read(m1)['gbdt']['selected_candidate_id'] == summary['selected_gbdt'] == 'G05', 'Winner/source mismatch')
        require(summary['dataset']['split'] == 'test' and summary['dataset']['n_samples'] == 346544 and
                summary['seeds'] == SEEDS and summary['scope']['thresholds_fitted_on'] == 'validation_only' and
                summary['scope']['models_retrained'] is False, 'Wrong Test evaluation scope')
        require(sha(config_path) == provenance['config_sha256'] == sha(run / 'config.json'), 'Test execution config mismatch')
        config, index = read(run / 'config.json'), read(run / 'evaluation_index.json')
        require(config['dataset'] == {'version': 'original', 'protocol': 'stratified', 'variant': 'main', 'features': 140} and
                summary['dataset']['version'] == 'original' and summary['dataset']['protocol'] == 'stratified' and
                summary['dataset']['variant'] == 'main', 'Dataset identity mismatch')
        require(index['five_seed_summaries'] == summary['five_seed_summaries'] and
                index['test_row_ids_sha256'] == summary['test_row_ids_sha256'] and
                index['test_rows'] == 346544 and index['training_fits'] == 0, 'Test index/public identity mismatch')
        ready = read(run / 'cache/test/READY.json')
        require(ready['split'] == 'test' and ready['rows'] == 346544 and len(ready['features']) == 140 and
                ready['source_manifest_output']['main']['sha256'] == summary['dataset']['sha256'], 'P2 Test source mismatch')

        spec = provenance['original_archives']['GBDT']
        require(sha(train / 'MANIFEST.json') == spec['manifest_sha256'] and
                sha(train / 'selection.json') == spec['selection_sha256'], 'Original training/selection pins differ')
        training_manifest = read(train / 'MANIFEST.json')
        for name in critical:
            if name == 'MANIFEST.json':
                continue
            item = training_manifest['files'][name]
            require(sha(train / name) == item['sha256'] and (train / name).stat().st_size == item['bytes'],
                    'G05 source hash mismatch: ' + name)
        binding, training_config, selection = (read(train / name) for name in ['binding.json', 'config.json', 'selection.json'])
        require(binding['config_sha256'] == digest(training_config) and
                binding['inputs'] == read(m1)['dataset']['input_file_sha256'] and
                training_config == read(m1)['gbdt']['execution_config'] and
                selection['binding_sha256'] == digest(binding) and selection['selected_candidate_id'] == 'G05' and
                sha(protocol_path) == binding['unified_file_sha256'], 'Training/config/P2 binding mismatch')

        curves, reference = [], None
        with threadpool_limits(limits=1):
            for seed in SEEDS:
                records = [r for r in summary['per_seed_threshold_records'] if r['method'] == 'M1_GBDT' and r['seed'] == seed]
                require(len(records) == 2 and {r['config']['target_validation_fpr'] for r in records} == {.01, .05}, 'Missing/duplicate Test records')
                job = read(train / f'completed/G05_seed_{seed}.json')
                expected_task = {'task_id': f'G05_seed_{seed}', 'candidate_id': 'G05', 'seed': seed,
                                 'algorithm': 'hist_gradient_boosting', 'params': selected['parameters']}
                require(job['task'] == expected_task and job['binding_sha256'] == digest(binding) and
                        read(safe_path(train, job['artifact_directory']) / 'task.json') == expected_task, 'Seed/task binding mismatch')
                for record in records:
                    raw_path = safe_path(run, record['record_path'])
                    require(sha(raw_path) == record['record_sha256'], 'Raw P3 record hash mismatch')
                    raw = read(raw_path)
                    require(raw['metrics']['curves']['average_precision'] == record['metrics']['average_precision'] and
                            raw['metrics']['curves']['roc_auc'] == record['metrics']['roc_auc'] and
                            raw['config'] == record['config'] and raw['dataset'] == record['dataset'] and
                            raw['method'] == record['method'] and raw['seed'] == record['seed'], 'Raw/public Test record differs')
                    require(record['config']['candidate_id'] == 'G05' and record['config']['parameters'] == selected['parameters'] and
                            record['config']['model_sha256'] == job['files']['model.joblib']['sha256'] and
                            record['config']['effective_parameters']['random_state'] == seed and
                            record['dataset'] == summary['dataset'], 'Prediction/model/data identity mismatch')
                    del raw
                relative = f'predictions/G05_seed_{seed}.npz'
                with np.load(run / relative, allow_pickle=False) as saved:
                    ids, y, scores = saved['row_ids'], saved['y_true'], saved['y_score']
                curve = verified_curve(ids, y, scores, expected_rows=346544,
                                       expected_row_digest=summary['test_row_ids_sha256'],
                                       expected_ap=records[0]['metrics']['average_precision'], reference=reference)
                require(curve['labels_uint8_sha256'] == LABEL_SHA and curve['benign_rows'] == 282506 and
                        curve['attack_rows'] == 64038, 'Test label/population mismatch')
                roc_auc = float(roc_auc_score(y, scores))
                require(all(r['metrics']['average_precision'] == curve['average_precision'] and
                            r['metrics']['roc_auc'] == roc_auc for r in records), 'Independent AP/ROC-AUC mismatch')
                if reference is None:
                    reference = (ids.copy(), y.copy())
                curves.append({**curve, 'seed': seed, 'roc_auc': roc_auc, 'prediction_path': relative,
                               'prediction_sha256': manifest['files'][relative]['sha256']})
                print(f"Seed {seed}: {len(y)} Test rows; AP={curve['average_precision']!r}; exact match", flush=True)
                del ids, y, scores
        primary = next(r for r in summary['five_seed_summaries'] if r['method'] == 'M1_GBDT' and r['target_validation_fpr'] == .01)
        for metric, field in [('ap', 'average_precision'), ('roc_auc', 'roc_auc')]:
            values = [c[field] for c in curves]
            require(math.isclose(statistics.mean(values), primary[metric]['mean'], abs_tol=1e-15, rel_tol=0) and
                    math.isclose(statistics.stdev(values), primary[metric]['sample_std'], abs_tol=1e-15, rel_tol=0), 'Five-seed summary mismatch')
        metadata = {'summary': summary, 'run_relative': provenance['local_archive'],
                    'test_archive_files_verified': 82, 'training_source_files_verified': len(critical) - 1,
                    'test_manifest_sha256': sha(run / 'MANIFEST.json'), 'test_summary_sha256': sha(public),
                    'training_manifest_sha256': sha(train / 'MANIFEST.json'), 'selection_sha256': sha(train / 'selection.json')}
        yield curves, metadata
        require(file_inventory(run) == manifest['files'] and sha(public) == SUMMARY_SHA, 'Source changed during export')


def render(curves):
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.tmp/m1_test_pr_mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    styles = ['-', '--', '-.', ':', (0, (5, 1, 1, 1, 1, 1))]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.labelsize': 12,
                         'axes.titlesize': 12.5, 'legend.fontsize': 9.5, 'axes.linewidth': .8,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'path.simplify': False,
                         'pdf.fonttype': 42, 'pdf.compression': 9}):
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
            fig.text(.5, .025, CAPTION, ha='center', fontsize=8.5, color='#444444')
            png, pdf = BytesIO(), BytesIO()
            fig.savefig(png, format='png', dpi=300, metadata={'Title': TITLE, 'Description': CAPTION})
            fig.savefig(pdf, format='pdf', metadata={'Title': TITLE, 'Author': 'P4', 'Subject': CAPTION,
                                                   'CreationDate': None, 'ModDate': None})
            return png.getvalue(), pdf.getvalue()
        finally:
            plt.close(fig)


def metric_table(summary):
    primary = next(r for r in summary['five_seed_summaries'] if r['method'] == 'M1_GBDT' and r['target_validation_fpr'] == .01)
    fields = [('AP', 'ap', 'Threshold-independent'), ('ROC-AUC', 'roc_auc', 'Threshold-independent'),
              ('Macro-F1', 'macro_f1', 'Frozen Validation 1% FPR threshold'),
              ('TPR', 'test_tpr', 'Frozen Validation 1% FPR threshold'),
              ('Actual FPR', 'test_fpr', 'Frozen Validation 1% FPR threshold')]
    lines = ['| Test metric | Mean ± sample SD | Operating rule |', '|---|---:|---|']
    lines += [f"| {label} | {primary[key]['mean']:.6f} ± {primary[key]['sample_std']:.6f} | {rule} |" for label, key, rule in fields]
    return '\n'.join(lines)


def audit_report(curves, metadata, png, pdf):
    import matplotlib
    import numpy
    import sklearn
    summary = metadata['summary']
    lines = ['# GBDT G05 Test PR Curve 验收报告', '',
             '输出：[PNG](../figures/fig_m1_test_pr.png)、[PDF](../figures/fig_m1_test_pr.pdf)。仅描述 G05 Test，不作方法比较。', '',
             '## 数据与完整性', '',
             f'- 来源：[正式 Test 摘要](../{SUMMARY})。原始封存目录 `{metadata["run_relative"]}/` 仅为本地定位名，GitHub 不包含完整归档。',
             '- Test 归档 82/82 个文件在计算前后通过大小和 SHA256 校验；Windows FILE_SHARE_READ 句柄阻止已有文件并发写入/删除。',
             f'- 另核对 GBDT 原 MANIFEST 固定哈希，以及 {metadata["training_source_files_verified"]} 个 G05 相关配置、输入绑定、selection、任务、模型文件的哈希；未宣称重新校验全部 302 个训练归档文件。模型仅字节哈希，不反序列化。',
             '- 五个种子各 346,544 行，良性 282,506 行，攻击 64,038 行；行 ID 唯一且顺序、标签逐项一致。',
             '- original / stratified / main；P2 固定 140 特征、训练参数、模型种子、数据及原始 P3 记录身份全部匹配。',
             '- 仅解码五份 G05 Test NPZ 的 row_ids / y_true / y_score，使用 sklearn average_precision_score 和 roc_auc_score 独立复算；与两套冻结阈值下的原记录严格一致。',
             '- Test 行 ID 摘要：`' + summary['test_row_ids_sha256'] + '`。',
             '- 标签 uint8 SHA256：`' + LABEL_SHA + '`。',
             '- 原 Test Parquet 身份（未打开）：`' + summary['dataset']['sha256'] + '`。', '',
             '| Seed | 独立 AP（与封存值差 0） | 独立 ROC-AUC（差 0） | Test NPZ SHA256 |', '|---|---:|---:|---|',
             *[f"| {c['seed']} | {c['average_precision']!r} | {c['roc_auc']!r} | `{c['prediction_sha256']}` |" for c in curves], '',
             '预测位置均为本地原始归档内 `predictions/G05_seed_<seed>.npz`。', '',
             '## 绘图与论文口径', '',
             '- sklearn precision_recall_curve(y_true, y_score, pos_label=1) 完整分数计算五条独立曲线；不平滑、不平均，不从 AP 反推曲线。',
             '- 单面板 Enlarged View：Recall 0.98–1.00（间隔 0.005），Precision 0.99–1.00（间隔 0.002）；图注明确局部范围，无 inset 或双面板。',
             '- 五种颜色与线型，图例六位 AP、位于图外；AP 是完整 Test AP，不是局部面积。白底轻网格，PNG 2400×1740、300 DPI，PDF 同图矢量导出并嵌入字体。',
             '- 原 figures/fig_m1_pr.png/.pdf 及其报告仍为 Validation；此次新增文件名明确包含 test，不覆盖旧图。',
             '- 论文图表使用 P4-1 临时编号，最终由 P8 全文统一编号。指标表仅列 G05；不提前排名、不使用 Bootstrap 比较，§4.4 另述。', '',
             metric_table(summary), '',
             '以上为五种子均值 ± 样本标准差（ddof=1），所有数值以 0–1 比例给出；AP/ROC-AUC 不依赖阈值，分类指标固定使用仅在 Validation 标定的 1% FPR 阈值。', '',
             '## SHA256 与复现', '',
             *[f'- {label}：`{value}`。' for label, value in [
                 ('PNG', hashlib.sha256(png).hexdigest()), ('PDF', hashlib.sha256(pdf).hexdigest()),
                 ('绘图脚本 tools/plot_m1_test_pr.py', sha(__file__)),
                 ('复用数据校验模块 methods/m1/pr_curve_data.py', sha(ROOT / 'methods/m1/pr_curve_data.py')),
                 *[(k, metadata[k]) for k in ['test_summary_sha256', 'test_manifest_sha256', 'training_manifest_sha256', 'selection_sha256']]]], '',
             f'环境：Python 原 .venv，NumPy {numpy.__version__}、sklearn {sklearn.__version__}、Matplotlib {matplotlib.__version__}。', '',
             '```powershell', '.\\.venv\\Scripts\\python.exe -B -X utf8 -m tools.plot_m1_test_pr', '```', '',
             '需要本地完整 Test 归档及 G05 训练元数据/模型文件。所有输出不同即拒绝覆盖；原始档案只读。没有读取 Parquet、加载训练矩阵、加载或重训模型，没有新选优、阈值标定或 Bootstrap，没有修改 P2/P3、配置、原始预测与封存结果。', '']
    return '\n'.join(lines).encode('utf-8')


def main():
    with verified_test_curves() as (curves, metadata):
        png, pdf = render(curves)
        report = audit_report(curves, metadata, png, pdf)
    outputs = [(PNG, png), (PDF, pdf), (REPORT, report)]
    for name, payload in outputs:
        path = ROOT / name
        require(not path.exists() or path.read_bytes() == payload, 'Refuse different existing output: ' + name)
    for name, payload in outputs:
        path = ROOT / name
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(payload)
        require(sha(path) == hashlib.sha256(payload).hexdigest(), 'Published output hash mismatch')
    print(json.dumps({'png_sha256': sha(ROOT / PNG), 'pdf_sha256': sha(ROOT / PDF),
                      'AP_exact_matches': 5, 'ROC_AUC_exact_matches': 5,
                      'test_archive_files_verified': 82, 'models_trained': 0}, indent=2))


if __name__ == '__main__':
    main()
