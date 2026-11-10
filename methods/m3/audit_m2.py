"""Read-only P5 delivery audit; never reads real train/val/test samples."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def audit():
    from jsonschema import Draft202012Validator
    result_path = ROOT / 'results/m2.json'
    result = json.loads(result_path.read_text(encoding='utf-8-sig'))
    schema = json.loads((ROOT / 'eval/schema.json').read_text(encoding='utf-8-sig'))
    errors = list(Draft202012Validator(schema).iter_errors(result))
    seeds = result.get('seeds', [])
    weights = [ROOT / 'results/preds_m2_corrected' / f'model_seed{s}.pth' for s in seeds]
    evidence = {
        'scope': 'static delivery audit only; no model rerun; no dataset opened',
        'm2_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'dataset_declared': result.get('dataset'),
        'seeds': seeds,
        'seed_policy_matches': seeds == [42, 43, 44, 45, 46],
        'checkpoint_presence': {p.relative_to(ROOT).as_posix(): p.exists() for p in weights},
        'p3_schema_passed': not errors,
        'p3_schema_errors': [{'path': '/'.join(map(str, e.path)), 'message': e.message}
                             for e in errors],
        'acceptance': 'pending_actual_rerun_and_protocol_reconciliation',
        'review_items': [
            'Published M2 files show one configuration; request eight-configuration search evidence or a team-approved budget amendment.',
            'Keep multi-seed summary but also export per-seed P3 records with full curves and provenance.',
            'P5 prose compares M1 validation AP against M2 test AP; align populations before comparison.',
            'Verify test execution agreement; M1 pending execution details apply to M1 and do not alone prove an M2 violation.'
        ]}
    out = ROOT / 'docs/p6_acceptance'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'm2_static_audit.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False)
                                           + '\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    print('Saved: docs/p6_acceptance/m2_static_audit.json')


if __name__ == '__main__':
    audit()
