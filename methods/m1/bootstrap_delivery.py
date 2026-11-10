"""Independent CSV/RNG verification and non-overwriting Bootstrap delivery."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

from . import bootstrap as boot
from .search_gbdt import require
from .seal_gbdt import artifact_paths, file_inventory, freeze_files


def independent_summary(rows, task, y):
    """Independent from bootstrap_stats.summarize; consumes exported decimals."""
    require(len(rows) == task['count'], 'Wrong replicate row count')
    differences = []
    for iteration, row in enumerate(rows, 1):
        require(int(row['seed']) == task['seed'] and row['metric'] == task['metric'] and
                row['fpr_target'] == ('' if task['target'] is None else str(task['target'])) and
                int(row['bootstrap_iteration']) == iteration, 'Duplicate/misidentified iteration')
        a, b, delta = (float(row[key]) for key in ('gbdt_value', 'rf_value', 'delta'))
        require(np.isfinite([a, b, delta]).all() and 0 <= a <= 1 and 0 <= b <= 1 and delta == a - b, 'Replicate value/difference mismatch')
        differences.append(delta)
    values = np.asarray(differences, dtype=np.float64)
    tail = (1 - .95) / 2
    lo, hi = np.quantile(values, [tail, 1 - tail], method='linear')
    return {'ci_lower': float(lo), 'ci_upper': float(hi), 'bootstrap_standard_error': float(np.std(values, ddof=1)),
            'valid_replicates': len(values), 'n_negative': int(np.count_nonzero(y == 0)),
            'n_positive': int(np.count_nonzero(y == 1))}


def reference_value(labels, scores, task):
    if task['metric'] == 'ap':return float(average_precision_score(labels, scores))
    x, y, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    return float(max(y[x <= task['target']]))


def verify_rng_and_samples(directory, task, rows, y, a, b):
    """Fresh RNG, no adapter sampler. Regenerate all states, re-evaluate 3 rows."""
    rng = np.random.default_rng(2026)
    negative, positive = np.where(y == 0)[0], np.where(y == 1)[0]
    checkpoints = {boot.read(p)['end']: boot.read(p)['rng_state'] for p in (directory / 'checkpoints').glob('*.json')}
    checked = []
    for i, row in enumerate(rows, 1):
        neg = rng.choice(negative, size=len(negative), replace=True)
        pos = rng.choice(positive, size=len(positive), replace=True)
        if i in (1, 1000, 2000):
            indices = np.r_[neg, pos]
            va, vb = reference_value(y[indices], a[indices], task), reference_value(y[indices], b[indices], task)
            require(va == float(row['gbdt_value']) and vb == float(row['rf_value']) and va - vb == float(row['delta']), 'Independent metric/seed mismatch')
            checked.append(i)
        if i in checkpoints:
            require(rng.bit_generator.state == checkpoints[i], 'RNG checkpoint differs from seed 2026')
    require(checked == [1, 1000, 2000], 'Independent sample verification incomplete')
    return checked


def independent_verify(run, config, source, public_summary, phase):
    expected = boot.tasks(config)
    require(len(expected) == 20 and len(list((run / 'completed').glob('*.json'))) == 20, 'Twenty tasks required')
    evidence = []
    for seed in config['statistics']['model_seeds']:
        y, a, b, points = boot.load_pair(config, source, public_summary, seed)
        for task in [t for t in expected if t['seed'] == seed]:
            phase('independent_verify_' + task['id'])
            require(boot.verify_completed(run, config, task), 'Task completion failed')
            directory = run / 'tasks' / task['id']
            rows, _, _, _ = boot.completed_chunks(directory, config, task)
            result = boot.read(directory / 'result.json')['result']
            checked = independent_summary(rows, task, y)
            require(all(result[key] == value for key, value in checked.items()), 'Independent CI/SE mismatch')
            require(result['seed'] == 2026 and result['n_bootstrap'] == 2000 and result['confidence'] == .95 and
                    result['interval_method'] == 'percentile' and result['quantile_method'] == 'linear', 'Statistical settings differ')
            require(result['value_a'] == points[('M1_GBDT', task['metric'], task['target'])] and
                    result['value_b'] == points[('M1_RF_auxiliary', task['metric'], task['target'])] and
                    result['difference'] == result['value_a'] - result['value_b'], 'Original point difference mismatch')
            samples = verify_rng_and_samples(directory, task, rows, y, a, b)
            evidence.append({'task_id': task['id'], 'rows': len(rows), 'independent_metric_iterations': samples,
                'result_sha256': boot.sha(directory / 'result.json'),
                'completed_record_sha256': boot.sha(run / 'completed' / (task['id'] + '.json'))})
    require(sum(x['rows'] for x in evidence) == 40000, 'Forty thousand rows required')
    boot.write(run / 'independent_verification.json', {'status': 'passed', 'tasks': evidence,
        'task_count': 20, 'row_count': 40000, 'config_sha256': boot.sha(run / 'config.json'),
        'p3_equivalence': 'Synthetic strict field equivalence in engineering tests; production CSV independently verified.',
        'rng': 'All 40000 index draws regenerated from seed 2026; every chunk-end state matched.',
        'metric_rechecks': 60, 'all_CIs_and_SE_recomputed': True, 'created_at': boot.now(), 'training_fits': 0})


def plots(results, staging):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    def panel(ax, metric, target, title):
        selected = [r for r in results if r['task']['metric'] == metric and r['task']['target'] == target]
        for color, item in zip(colors, selected):
            x = item['task']['seed'];s = item['result']
            # Percentile intervals need not contain the original point estimate.
            ax.vlines(x, s['ci_lower'], s['ci_upper'], color=color, lw=1.8)
            ax.plot(x, s['difference'], 'o', color=color, ms=5)
        ax.axhline(0, color='0.3', ls='--', lw=1)
        ax.set(title=title, xlabel='Model seed', xticks=[42, 43, 44, 45, 46], ylabel='GBDT - RF')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3));ax.grid(alpha=.2)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    panel(ax, 'ap', None, 'Test AP difference: paired Bootstrap 95% CI')
    fig.tight_layout();fig.savefig(staging / 'fig_m1_bootstrap_ap.png', dpi=300, facecolor='white');plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, target in zip(axes, (.001, .01, .05)):
        panel(ax, 'tpr_at_fpr', target, f'Empirical TPR @ FPR {target:.1%}')
    fig.suptitle('Test TPR differences: paired Bootstrap 95% CIs', y=1.01)
    fig.tight_layout();fig.savefig(staging / 'fig_m1_bootstrap_tpr.png', dpi=300, bbox_inches='tight', facecolor='white');plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (metric, target) in zip(axes.flat, boot.METRICS):
        for color, item in zip(colors, [r for r in results if r['task']['metric'] == metric and r['task']['target'] == target]):
            ax.hist(item['deltas'], bins=40, histtype='step', lw=1.3, color=color, label=f"Seed {item['task']['seed']}")
        ax.axvline(0, color='0.3', ls='--', lw=1)
        ax.set(title='AP' if target is None else f'TPR @ FPR {target:.1%}', xlabel='GBDT - RF', ylabel='Replicate count')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3));ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Stratified paired Bootstrap distributions (2,000 per comparison)')
    fig.tight_layout();fig.savefig(staging / 'fig_m1_bootstrap_distribution.png', dpi=300, facecolor='white');plt.close(fig)


def report_text(results, verification, trial, totals, run):
    lines = ['# P4 Formal Paired Bootstrap 报告', '',
        'GBDT G05与RF_FIXED按模型种子42–46逐组配对；完整Test为346,544行（良性282,506、攻击64,038）。每项2,000次，95% percentile区间、linear分位数、Bootstrap seed=2026，差值为GBDT−RF。', '',
        'P4适配层仅为保留逐次值；使用与P3相同的NumPy抽样顺序和sklearn指标定义，合成数据严格字段一致性通过。P2/P3、模型、原预测、冻结阈值和既有归档均未修改。未重训、未读取Parquet、未加载模型，也未在Test上重标部署阈值。', '',
        '| Seed | Metric | FPR | GBDT | RF | Delta | 95% CI | Bootstrap SE | Valid B |',
        '|---:|---|---:|---:|---:|---:|---|---:|---:|']
    for item in results:
        t, s = item['task'], item['result']
        fpr = '—' if t['target'] is None else f"{t['target']:.1%}"
        lines.append(f"| {t['seed']} | {t['metric']} | {fpr} | {s['value_a']:.10f} | {s['value_b']:.10f} | {s['difference']:.10f} | [{s['ci_lower']:.10f}, {s['ci_upper']:.10f}] | {s['bootstrap_standard_error']:.10f} | {s['valid_replicates']} |")
    lines += ['', '## 独立验收', '',
        '20项全部完成，CSV共40,000行（不含表头）。独立读取CSV复算全部CI和标准误；重新生成全部40,000次抽样并核对每段RNG状态；每项独立重算第1、1000、2000次双模型指标，共60个配对值。没有将试跑结果计入正式结果。', '',
        '标准误为Bootstrap差值分布的样本标准差（ddof=1），不是除以sqrt(2000)后的Monte Carlo均值误差。索引矩阵不保存；由锁定的NumPy版本、seed与抽样规则复现。', '',
        '## 资源', '', f"试跑（seed42/AP/20次）受控工作进程耗时 {trial['elapsed_seconds']:.3f} 秒。正式20项工作进程累计耗时 {totals['elapsed_seconds']:.3f} 秒（含输入核验、加载和保存，不等于纯指标CPU时间）。",
        f"正式峰值Job提交 {totals['job_peak_commit_bytes']/boot.GIB:.3f} GiB；采样进程树峰值RSS {totals['sampled_peak_rss_bytes']/boot.GIB:.3f} GiB；最低系统空闲内存 {totals['min_available_physical_bytes']/boot.GIB:.3f} GiB；峰值CPU {totals['peak_cpu_core_equivalent']:.3f} 个逻辑核心等效。",
        '所有工作进程使用8 GiB Job提交硬限、4逻辑CPU亲和性、外层串行；启动内存门槛14 GiB、运行最低6 GiB、RSS上限8 GiB。资源峰值覆盖每项整个工作进程，不仅是计算循环。长任务仅临时阻止自动待机，结束后释放，不修改电源计划。', '',
        '## 统计解释与局限', '',
        '这些是固定已训练模型、给定观测类别比例下的逐种子行级区间，不包含训练随机性，也不处理流量组内依赖。未拼接五份预测，未平均区间端点。20项边际95%区间不是同时置信区间，不据此声称全部比较整体显著，也不额外产生未经规定的p值。', '',
        'TPR@FPR为重采样内的描述性ROC操作点，不是冻结Validation部署阈值下的TPR区间；不会回写或修改任何部署阈值。', '',
        '## Results §4.1 可用摘要', '']
    for metric, target in boot.METRICS:
        items = [r for r in results if r['task']['metric'] == metric and r['task']['target'] == target]
        values = np.array([r['result']['difference'] for r in items])
        positive = sum(r['result']['ci_lower'] > 0 for r in items)
        negative = sum(r['result']['ci_upper'] < 0 for r in items)
        crossing = 5 - positive - negative
        label = 'AP' if target is None else f'TPR@FPR={target:.1%}'
        lines.append(f"{label}：五种子原始差值的描述性均值为{values.mean():.10f}（种子间样本标准差{values.std(ddof=1):.10f}）；逐种子区间中{positive}个完全为正、{negative}个完全为负、{crossing}个包含零。该描述性均值没有被赋予一个通过平均端点得到的联合区间。")
    lines += ['', '## 归档', '',
        '本地原始Bootstrap运行目录：`' + run.relative_to(boot.ROOT).as_posix() + '`。保留逐任务分段CSV、RNG检查点、源码快照、环境、资源日志及独立验证；该目录是本地归档，不是GitHub可访问链接。',
        'results/m1_bootstrap/MANIFEST.json以仓库根目录为路径根，覆盖五个结果文件、本报告和三张图；MANIFEST不包含自身，SHA256另在本地PUBLISHED验收记录保存。',
        '未自动commit或push。原results/m1.json及Validation PR图保持其原阶段身份。']
    return '\n'.join(lines) + '\n'


def verify_public(manifest_path, expected_sha):
    require(boot.sha(manifest_path) == expected_sha, 'Published manifest pin differs')
    m = boot.read(manifest_path)
    require(m['status'] == 'complete' and len(m['files']) == 9, 'Incomplete publication')
    expected = {f'results/m1_bootstrap/{name}' for name in ('bootstrap_config.json', 'bootstrap_results.csv',
        'bootstrap_summary.json', 'confidence_intervals.json', 'execution_log.csv')}
    expected |= {'docs/m1_bootstrap_report.md', 'figures/fig_m1_bootstrap_ap.png',
                 'figures/fig_m1_bootstrap_tpr.png', 'figures/fig_m1_bootstrap_distribution.png'}
    require(set(m['files']) == expected and m['task_count'] == 20 and m['replicate_rows'] == 40000, 'Wrong formal delivery set')
    for name, entry in m['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe manifest path')
        path = boot.ROOT / relative
        require(not path.is_symlink() and path.stat().st_size == entry['bytes'] and boot.sha(path) == entry['sha256'], 'Published file hash differs')
    return len(m['files'])


def publish(run, config, phase):
    phase('final_acceptance_and_publication')
    verification = boot.read(run / 'independent_verification.json')
    require(verification['status'] == 'passed' and verification['row_count'] == 40000 and
            verification['config_sha256'] == boot.sha(run / 'config.json'), 'Independent acceptance missing')
    results, all_rows, logs = [], [], []
    for task in boot.tasks(config):
        require(boot.verify_completed(run, config, task), 'Task changed before publication')
        e = next(v for v in verification['tasks'] if v['task_id'] == task['id'])
        directory = run / 'tasks' / task['id']
        require(boot.sha(directory / 'result.json') == e['result_sha256'] and
                boot.sha(run / 'completed' / (task['id'] + '.json')) == e['completed_record_sha256'], 'Verified task receipt changed')
        raw = boot.read(directory / 'result.json');rows, _, _, _ = boot.completed_chunks(directory, config, task)
        results.append({**raw, 'deltas': [float(r['delta']) for r in rows]})
        all_rows.extend([[r[k] for k in boot.COLUMNS] for r in rows])
        outcome = boot.read(run / 'completed' / (task['id'] + '.json'))['outcome']
        logs.append({'task_id': task['id'], 'seed': task['seed'], 'metric': task['metric'],
            'fpr_target': '' if task['target'] is None else task['target'], 'started_at': raw['started_at'],
            'ended_at': raw['ended_at'], 'compute_seconds': raw['compute_seconds'],
            'guarded_elapsed_seconds': outcome['elapsed_seconds'], 'job_peak_commit_bytes': outcome['job_peak_commit_bytes'],
            'sampled_peak_rss_bytes': outcome['sampled_peak_rss_bytes'], 'min_free_memory_bytes': outcome['min_available_physical_bytes'],
            'peak_cpu_core_equivalent': outcome['peak_cpu_core_equivalent'], 'status': 'complete',
            'resource_scope': 'entire protected task worker, including input verification'})
    require(len(all_rows) == 40000 and len(results) == 20, 'Publication counts differ')
    staging = run / 'delivery';staging.mkdir(exist_ok=False)
    boot.publish_bytes(staging / 'bootstrap_results.csv', boot.csv_bytes(all_rows))
    boot.write(staging / 'bootstrap_config.json', config)
    summaries = [{k: v for k, v in item.items() if k != 'deltas'} for item in results]
    boot.write(staging / 'bootstrap_summary.json', {'status': 'complete', 'comparisons': summaries, 'task_count': 20,
        'row_count': 40000, 'mean_differences_descriptive_only': [
            {'metric': m, 'fpr_target': t, 'mean_delta': float(np.mean([r['result']['difference'] for r in results if r['task']['metric'] == m and r['task']['target'] == t]))}
            for m, t in boot.METRICS], 'independent_verification': verification,
        'joint_five_seed_CI': None, 'familywise_significance_claim': False})
    boot.write(staging / 'confidence_intervals.json', {'confidence': .95, 'interval_method': 'percentile', 'quantile_method': 'linear',
        'comparisons': [{'seed': r['task']['seed'], **r['result']} for r in results],
        'interpretation': 'Marginal fixed-model class-stratified row intervals; not simultaneous or training-variation intervals.'})
    stream = io.StringIO(newline='');writer = csv.DictWriter(stream, fieldnames=list(logs[0]), lineterminator='\n')
    writer.writeheader();writer.writerows(logs);boot.publish_bytes(staging / 'execution_log.csv', stream.getvalue().encode('utf-8'))
    totals = {'elapsed_seconds': sum(r['guarded_elapsed_seconds'] for r in logs),
        'job_peak_commit_bytes': max(r['job_peak_commit_bytes'] for r in logs),
        'sampled_peak_rss_bytes': max(r['sampled_peak_rss_bytes'] for r in logs),
        'min_available_physical_bytes': min(r['min_free_memory_bytes'] for r in logs),
        'peak_cpu_core_equivalent': max(r['peak_cpu_core_equivalent'] for r in logs)}
    trial_ref = boot.read(run / 'trial_reference.json')
    trial = boot.read(Path(trial_ref['directory']) / 'TRIAL_COMPLETE.json')['outcome']
    plots(results, staging)
    boot.publish_bytes(staging / 'm1_bootstrap_report.md', report_text(results, verification, trial, totals, run).encode('utf-8'))
    files = {f'results/m1_bootstrap/{name}': staging / name for name in ('bootstrap_config.json', 'bootstrap_results.csv',
        'bootstrap_summary.json', 'confidence_intervals.json', 'execution_log.csv')}
    files[config['delivery']['report']] = staging / 'm1_bootstrap_report.md'
    files.update({name: staging / Path(name).name for name in config['delivery']['figures']})
    require(not any((boot.ROOT / name).exists() for name in files) and
            not (boot.ROOT / 'results/m1_bootstrap/MANIFEST.json').exists(), 'Publication target already exists; no overwrite')
    manifest = {'schema_version': '1', 'status': 'complete', 'path_root': 'repository_root',
        'files': {name: {'bytes': path.stat().st_size, 'sha256': boot.sha(path)} for name, path in files.items()},
        'input_manifest_sha256': config['input']['manifest_sha256'], 'config_payload_sha256': boot.digest(boot.payload(config)),
        'local_run': run.relative_to(boot.ROOT).as_posix(), 'task_count': 20, 'replicate_rows': 40000,
        'runtime_files_excluded': {'RUN.lock': 'Runtime coordination only; raw logs and checkpoints are in the separate local run archive.'},
        'self_hash_policy': 'MANIFEST excluded from itself; pin externally in local PUBLISHED.json.'}
    for name, source in files.items():boot.publish_bytes(boot.ROOT / name, source.read_bytes())
    destination = boot.ROOT / 'results/m1_bootstrap/MANIFEST.json';boot.write(destination, manifest)
    pin = boot.sha(destination);count = verify_public(destination, pin)
    require(len(boot.read_rows(boot.ROOT / 'results/m1_bootstrap/bootstrap_results.csv')) == 40000, 'Final CSV row count differs')
    boot.write(run / 'PUBLISHED.json', {'status': 'complete', 'manifest_sha256': pin, 'files_verified': count,
        'task_count': 20, 'replicate_rows': 40000, 'resource_totals': totals, 'trial_outcome': trial,
        'training_fits': 0, 'parquet_accessed': False, 'published_at': boot.now()})
    print('FORMAL_BOOTSTRAP_MANIFEST_SHA256', pin, flush=True)
