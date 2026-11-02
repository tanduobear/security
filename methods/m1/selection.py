"""Validation-only selection and frozen FPR calibration. No dataset file I/O."""
import numpy as np

from .data import require


def select_configuration(jobs, protocol, details):
    seeds = protocol['model_seeds']
    candidates = details['gbdt_candidates']
    summary = []
    for candidate in candidates:
        runs = [job for job in jobs if job['role'] == 'main' and job['candidate_id'] == candidate['id']]
        require(len(runs) == len(seeds) and sorted(job['seed'] for job in runs) == seeds, 'Incomplete or duplicated five-seed main results')
        require(all(job['algorithm'] == 'hist_gradient_boosting' and job['params'] == candidate['params'] and job['evaluation_split'] == 'val' for job in runs), 'Main job identity/configuration mismatch')
        ap = np.array([job['validation_ap'] for job in runs], dtype=float)
        require(np.isfinite(ap).all() and ((ap >= 0) & (ap <= 1)).all(), 'Invalid validation AP')
        summary.append({'candidate_id':candidate['id'],'mean_validation_ap':float(ap.mean()),'sample_std_ap':float(ap.std(ddof=1))})
    require(len([j for j in jobs if j['role'] == 'main']) == 40, 'Unexpected main search count')
    maximum = max(item['mean_validation_ap'] for item in summary)
    eligible = [item for item in summary if maximum - item['mean_validation_ap'] <= details['tie_break']['ap_tolerance']]
    chosen = min(eligible, key=lambda item:item['candidate_id'])
    return {'selected_candidate_id':chosen['candidate_id'], 'selection_metric':'mean_full_validation_AP',
            'candidate_summaries':summary,'tie_break':details['tie_break'],'rf_used_for_selection':False,
            'fitted_on':'train','threshold_calibration_split':'val','refit_on_train_plus_val':False}


def calibrate_fpr(y, scores, alpha):
    """At most floor(alpha*n_benign) false alarms; never split equal scores.

    nextafter moves strictly above the (k+1)-th descending benign score,
    including the zero-allowed-false-alarms and score=1 boundary cases.
    The resulting test FPR is not guaranteed to equal the validation target.
    """
    y, scores = np.asarray(y), np.asarray(scores,dtype=float)
    require(y.ndim == scores.ndim == 1 and len(y) == len(scores) > 0, 'Invalid calibration arrays')
    require(np.isin(y,[0,1]).all() and np.isfinite(scores).all(), 'Invalid calibration labels/scores')
    require(isinstance(alpha,(int,float)) and not isinstance(alpha,bool) and 0 <= alpha < 1, 'Invalid target FPR')
    benign = np.sort(scores[y == 0])[::-1]
    require(len(benign) > 0, 'No benign validation samples')
    allowed = int(np.floor(alpha * len(benign)))
    threshold = float(np.nextafter(benign[allowed], np.inf))
    false = int((benign >= threshold).sum())
    require(np.isfinite(threshold) and false <= allowed, 'Threshold calibration failed')
    return {'target_fpr':float(alpha),'threshold':threshold,'operator':'>=',
            'source':'validation_benign_quantile','boundary_rule':'strict_nextafter_benign_boundary',
            'validation_benign_support':len(benign),'allowed_false_positives':allowed,
            'validation_false_positives':false,'actual_validation_fpr':false / len(benign)}
