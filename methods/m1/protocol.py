"""Frozen team terms are separate from as-yet-unapproved execution details."""
import hashlib
import json

from .data import integer, require


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def unified_digest(protocol):
    return digest({k: v for k, v in protocol.items() if k != 'sha256'})


def execution_digest(details):
    return digest({k: v for k, v in details.items() if k != 'approval'})


def validate_unified(protocol):
    require(protocol['name'] == 'NIDS Unified Experimental Protocol' and protocol['version'] == '1.0', 'Unknown unified protocol')
    require(protocol['approval']['status'] == 'approved' and set(protocol['approval']['roles']) == {'P4','P5','P6','P7'}, 'Unified approval missing')
    require(protocol['sha256'] == unified_digest(protocol), 'Unified protocol has changed')
    require(protocol['main_methods'] == ['GBDT','1D-CNN','Autoencoder'], 'Main method scope changed')
    require(protocol['auxiliary_methods'] == ['RF','Isolation Forest','OCSVM'], 'Auxiliary scope changed')
    require(protocol['dataset'] == {'version':'original','protocol':'stratified','variant':'main','features':140,'owner':'P2','refit_preprocessor':False}, 'Main data contract changed')
    require(protocol['search']['configurations_per_main_method'] == 8 and protocol['search']['seeds_per_configuration'] == 5, 'Main search budget changed')
    require(protocol['model_seeds'] == [42,43,44,45,46], 'Unified seeds changed')
    require(protocol['selection']['supervised'] == 'mean_full_validation_AP' and protocol['selection']['autoencoder'] == 'benign_unlabeled_validation', 'Selection rule changed')
    require(protocol['thresholds']['fpr_targets'] == [.01,.05] and protocol['thresholds']['source_split'] == 'val', 'Threshold protocol changed')
    require(protocol['thresholds']['ae_benign_quantiles'] == [.95,.99], 'AE threshold contract changed')
    require(protocol['inference'] == {'device':'cpu','batch_size':512,'warmup':5,'repeats':30,'shared_row_ids':True,'interface_owner':'P3'}, 'Inference contract changed')
    require(protocol['statistics']['std_ddof'] == 1 and protocol['statistics']['paired_bootstrap'] is True, 'Statistics contract changed')
    require(protocol['extension_order'] == ['main','temporal_and_port_ablation','corrected'], 'Extension order changed')
    return True


def execution_issues(protocol, details):
    """No I/O: reject pending details before creating outputs or loaders."""
    validate_unified(protocol)
    issues = []
    def check(value, message):
        if not value:issues.append(message)
    approval = details.get('approval') or {}
    check(approval.get('status') == 'approved', 'Execution details are not approved')
    check(bool(approval.get('source')) and bool(approval.get('approved_at')), 'Execution approval source/date missing')
    check(approval.get('execution_sha256') == execution_digest(details), 'Execution details differ from approved digest')
    check(details.get('unified_protocol_sha256') == protocol['sha256'], 'Execution/unified protocol mismatch')
    candidates = details.get('gbdt_candidates', [])
    check(len(candidates) == 8, 'Exactly eight predefined GBDT configurations are required')
    ids = [c.get('id') for c in candidates]
    check(all(isinstance(x,str) and x.isalnum() for x in ids) and len(set(ids)) == len(ids), 'Invalid candidate IDs')
    check(len({digest(c.get('params')) for c in candidates}) == len(candidates), 'Duplicate GBDT parameters')
    stop = details.get('stopping', {})
    check(stop.get('mode') == 'fixed_rounds' and stop.get('monitor') is None and stop.get('patience') is None, 'This executor supports explicit fixed-round caps, not an unspecified early-stopping variant')
    for candidate in candidates:
        params = candidate.get('params', {})
        check(params.get('early_stopping') is False, 'Internal random holdout/automatic early stopping is forbidden')
        check(params.get('class_weight') is None and 'random_state' not in params, 'Weights/seeds must come from P2/unified protocol')
        check(isinstance(params.get('max_iter'),int) and params['max_iter'] > 0, 'Explicit positive max_iter is required')
    rf = details.get('rf_auxiliary', {})
    check(rf.get('id') == 'RF_FIXED' and rf.get('seed_policy') == 'same_five' and rf.get('eligible_for_main_selection') is False, 'RF must be a single fixed auxiliary configuration')
    rp = rf.get('params', {})
    check(isinstance(rp.get('n_estimators'),int) and rp['n_estimators'] > 0 and rp.get('class_weight') is None and 'random_state' not in rp, 'Invalid fixed RF configuration')
    selection = details.get('tie_break', {})
    check(selection.get('rule') == 'lowest_candidate_id', 'An explicit implemented tie rule is required')
    check(isinstance(selection.get('ap_tolerance'),(int,float)) and 0 <= selection['ap_tolerance'] < 1, 'Invalid AP tie tolerance')
    check(details.get('threshold_boundary_rule') == 'strict_nextafter_benign_boundary', 'Threshold tie/boundary rule must be approved')
    boot = details.get('bootstrap', {})
    check(isinstance(boot.get('repetitions'),int) and boot['repetitions'] >= 2, 'Bootstrap repetitions missing')
    check(isinstance(boot.get('confidence'),(int,float)) and 0 < boot['confidence'] < 1, 'Invalid bootstrap confidence')
    check(boot.get('metrics') == ['ap','tpr_at_fpr'] and boot.get('fpr_targets') == [.001,.01,.05], 'Bootstrap metric/operating-point plan missing')
    check(boot.get('comparison') == 'selected_gbdt_minus_fixed_rf_same_seed', 'Bootstrap comparison must be predefined')
    timing = details.get('timing', {})
    check(timing.get('source_split') == 'val' and timing.get('selection') == 'sha256_row_id_rank', 'Shared timing population/selection rule must be approved')
    resources = details.get('resources', {})
    for name in ('fit_threads','inference_threads','prediction_batch_rows','duckdb_memory_limit_mb'):
        try:integer(resources.get(name), name, 1)
        except ValueError:issues.append('Invalid resource limit: ' + name)
    check(resources.get('outer_parallel_jobs') == 1, 'Concurrent large training tasks are forbidden')
    if isinstance(rp.get('n_jobs'),int) and isinstance(resources.get('fit_threads'),int):
        check(0 < rp['n_jobs'] <= resources['fit_threads'], 'RF workers exceed training limit')
    else:issues.append('Explicit RF worker count is required')
    for value, name in [(boot.get('seed'),'bootstrap seed'), (timing.get('seed'),'timing seed')]:
        try:integer(value,name)
        except ValueError:issues.append('Invalid ' + name)
    # Ask sklearn to validate candidate values without touching data or fitting.
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    for constructor, params in [(HistGradientBoostingClassifier,c.get('params',{})) for c in candidates] + [(RandomForestClassifier,rp)]:
        try:constructor(**params)._validate_params()
        except (ValueError,TypeError):issues.append('Invalid estimator hyperparameters')
    return issues


def require_execution(protocol, details):
    issues = execution_issues(protocol, details)
    require(not issues, 'Formal execution blocked: ' + '; '.join(issues))
