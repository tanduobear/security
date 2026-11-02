"""Approval-gated formal preparation. No training/search/test evaluation CLI."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.m1.data import preparation_path
from methods.m1.storage import RunStore, verify_run


def protocol_digest(protocol):
    content = {k: v for k, v in protocol.items() if k != 'approval'}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def protocol_issues(protocol):
    """Check recorded approval/completeness, not the authenticity of signatures.

    No particular budget, seed list, ranking metric or threshold is supplied
    by this module. It never opens dataset files or timing identity files.
    """
    issues = []
    def need(condition, message):
        if not condition:
            issues.append(message)
    def positive(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    def seed(value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    def text(value):
        return isinstance(value, str) and bool(value.strip())
    need(protocol.get('schema_version') == '1', 'schema_version must be 1')
    approval = protocol.get('approval') or {}
    need(approval.get('status') == 'approved', 'P4/P5/P6 approval is pending')
    for role in ('P4', 'P5', 'P6'):
        need(text((approval.get('reviewers') or {}).get(role)), f'{role} approval identity is missing')
    need(text(approval.get('approved_at')), 'approval date is missing')
    need(approval.get('protocol_sha256') == protocol_digest(protocol), 'approved protocol digest does not match')
    data = protocol.get('dataset') or {}
    try:
        preparation_path(ROOT, data.get('version'), data.get('protocol'), data.get('variant'), 'train')
    except ValueError:
        issues.append('dataset version/protocol/variant must be agreed')
    search = protocol.get('search') or {}
    need(positive(search.get('configurations_per_algorithm')), 'search configuration count is missing')
    need(text(search.get('budget_scope')), 'RF/GBDT and P5/P6 budget scope is missing')
    need(text(search.get('strategy')) and seed(search.get('seed')), 'search strategy/seed is missing')
    configurations = search.get('configurations')
    need(isinstance(configurations, dict) and set(configurations) == {'random_forest', 'hist_gradient_boosting'}, 'explicit RF/GBDT candidate lists are missing')
    if isinstance(configurations, dict):
        for name, values in configurations.items():
            need(isinstance(values, list) and len(values) == search.get('configurations_per_algorithm') and all(isinstance(x, dict) for x in values), f'{name} candidate count/content disagrees')
            if isinstance(values, list):
                need(len({json.dumps(x, sort_keys=True) for x in values}) == len(values), f'{name} configurations contain duplicates')
    seeds = protocol.get('model_seeds')
    need(isinstance(seeds, list) and len(seeds) == 5 and all(seed(x) for x in seeds) and len(set(seeds)) == 5, 'five distinct model seeds must be approved')
    selection = protocol.get('selection') or {}
    for key in ('metric', 'aggregation', 'tie_rule'):
        need(text(selection.get(key)), 'selection.' + key + ' is missing')
    early = protocol.get('early_stopping')
    need(isinstance(early, dict) and text(early.get('rule')), 'common early-stopping rule is missing')
    threshold = protocol.get('threshold_policy')
    need(isinstance(threshold, dict) and text(threshold.get('rule')) and threshold.get('source_split') == 'val', 'explicit validation-only threshold policy is missing')
    inference = protocol.get('inference') or {}
    for key in ('batch_size', 'repeats', 'threads'):
        need(positive(inference.get(key)), 'inference.' + key + ' is missing')
    need(seed(inference.get('warmup')), 'inference warmup is missing')
    need(text(inference.get('row_ids_path')) and text(inference.get('population_identity')), 'shared timing row identity plan is missing')
    families = protocol.get('expected_attack_families')
    need(isinstance(families, list) and bool(families) and all(text(x) and x != 'benign' for x in families) and len(set(families)) == len(families), 'full attack-family list is missing')
    resources = protocol.get('resources') or {}
    need(positive(resources.get('fit_threads')) and positive(resources.get('duckdb_memory_limit_mb')), 'resource limits are missing')
    return issues


def prepare_run(protocol, output_root, *, run_id=None):
    issues = protocol_issues(protocol)
    if issues:
        raise ValueError('Protocol not approved/complete: ' + '; '.join(issues))
    store = RunStore(output_root, kind='prepared', run_id=run_id)
    try:
        store.write_json('config.json', protocol)
        data = protocol['dataset']
        store.write_json('plan.json', {
            'status': 'prepared_not_executed', 'protocol_sha256': protocol_digest(protocol),
            'train_path': str(preparation_path(ROOT, data['version'], data['protocol'], data['variant'], 'train').relative_to(ROOT)),
            'validation_path': str(preparation_path(ROOT, data['version'], data['protocol'], data['variant'], 'val').relative_to(ROOT)),
            'test_access': 'disabled_in_preparation',
            'execution': 'No training or search executor is enabled by this preparation entry point.'})
        store.finish()
        verify_run(store.path)
    except Exception as error:
        store.fail(error)
        raise
    return store.path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--prepare-run', action='store_true', help='Write a prepared manifest only after approval')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/m1_runs')
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    issues = protocol_issues(protocol)
    if issues:
        print('BLOCKED: no dataset has been opened and no training has been started.')
        for issue in issues:
            print('- ' + issue)
        return 2
    if args.prepare_run:
        print(prepare_run(protocol, args.output_root))
    else:
        print('Recorded protocol approval/completeness checks passed; no execution requested.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
