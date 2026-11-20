import json
import numpy as np
from scipy.stats import rankdata

METRIC_PROTOCOL = 'joint_gaussian_test_imputation_brier_v3'
CRPS_PROTOCOL = 'test_point_or_fair_samples_Jge2_v2'


def checked_cholesky(cov, label='covariance'):
    import warnings
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.isfinite(cov).all():
        raise ValueError(f'{label}: invalid/non-finite covariance')
    scale = max(float(np.max(np.abs(np.diag(cov)))), 1.)
    if not np.allclose(cov, cov.T, rtol=1e-6, atol=1e-8*scale):
        raise ValueError(f'{label}: asymmetric covariance')
    cov = (cov + cov.T) / 2
    try:
        return np.linalg.cholesky(cov), 0.
    except np.linalg.LinAlgError:
        if np.linalg.eigvalsh(cov).min() < -1e-7*scale:
            raise np.linalg.LinAlgError(f'{label}: materially non-positive-definite covariance')
        for relative in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
            jitter = relative*scale
            try:
                factor = np.linalg.cholesky(cov + jitter*np.eye(len(cov)))
                warnings.warn(f'{label}: covariance_jitter={jitter:.12g}', RuntimeWarning)
                return factor, jitter
            except np.linalg.LinAlgError:
                pass
        raise np.linalg.LinAlgError(f'{label}: bounded Cholesky repair failed')


def fair_crps(truth, samples, mask, std):
    truth, samples = np.asarray(truth, dtype=float), np.asarray(samples, dtype=float)
    mask, std = np.asarray(mask), np.asarray(std, dtype=float)
    if truth.ndim != 2 or mask.shape != truth.shape or mask.dtype != np.bool_:
        raise ValueError('Expected truth matrix and matching boolean evaluation mask')
    if std.shape != (truth.shape[1],) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Evaluation scales must be finite and strictly positive')
    if samples.ndim != 3 or samples.shape[1:] != truth.shape or samples.shape[0] < 1:
        raise ValueError('Expected J by N by D imputation samples')
    J = len(samples)
    reason = 'no_evaluation_positions' if not mask.any() else 'fair_crps_requires_J_ge_2' if J < 2 else None
    if reason:
        return dict(test_impute_crps=None, test_crps_type='unavailable', test_crps_reason=reason)
    values, target = (samples / std)[:, mask], (truth / std)[mask]
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError('Non-finite truth or imputation samples at evaluation positions')
    ordered = np.sort(values, axis=0)
    weights = (2 * np.arange(J) - J + 1)[:, None]
    correction = np.sum(weights * ordered, axis=0) / (J * (J - 1))
    score = float(np.mean(np.mean(np.abs(values - target), axis=0) - correction))
    return dict(test_impute_crps=score, test_crps_type='fair_samples', test_crps_reason=None)


def imputation_error_stats(truth, fill, mask, std, prefix, *, samples=None, independent=False):
    if prefix not in {"train", "test"}:
        raise ValueError("Expected train or test evaluation split")
    truth, fill = np.asarray(truth, dtype=float), np.asarray(fill, dtype=float)
    mask, std = np.asarray(mask), np.asarray(std, dtype=float)
    if truth.ndim != 2 or fill.shape != truth.shape or mask.shape != truth.shape or mask.dtype != np.bool_:
        raise ValueError('Expected matching truth/fill matrices and a boolean evaluation mask')
    if std.shape != (truth.shape[1],) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Evaluation scales must be finite and strictly positive')
    errors = (fill - truth) / std
    selected = errors[mask]
    if not np.isfinite(selected).all():
        raise ValueError('Non-finite truth or fill at evaluation positions')
    mae = float(np.mean(np.abs(selected))) if selected.size else None
    encode = lambda x: json.dumps(x.tolist(), separators=(',', ':'), allow_nan=False)
    result = {f'{prefix}_impute_rmse': float(np.sqrt(np.mean(selected**2))) if selected.size else None,
              f'{prefix}_impute_mae': mae,
              f'{prefix}_missing_count': encode(mask.sum(0)),
              f'{prefix}_squared_error_sum': encode(np.where(mask, errors**2, 0).sum(0)),
              f'{prefix}_absolute_error_sum': encode(np.where(mask, np.abs(errors), 0).sum(0))}
    if prefix == 'train':
        return result
    crps, kind, reason = mae, 'point', None
    if samples is not None and independent:
        result.update(fair_crps(truth, samples, mask, std))
        return result
    elif samples is not None:
        reason = 'point_output_used_samples_not_verified_iid'
    if not selected.size:
        crps, kind, reason = None, 'unavailable', 'no_evaluation_positions'
    result.update(test_impute_crps=crps, test_crps_type=kind, test_crps_reason=reason)
    return result


def posterior_components(m0, c0, m1, c1):
    m0, m1 = np.asarray(m0, dtype=float), np.asarray(m1, dtype=float)
    c0, c1 = np.asarray(c0, dtype=float), np.asarray(c1, dtype=float)
    if m0.ndim != 1 or m1.shape != m0.shape or m0.size == 0:
        raise ValueError('Posterior means must have the same nonzero parameter dimension')
    p = m0.size
    if c0.shape != (p,p) or c1.shape != (p,p):
        raise ValueError('Posterior covariance dimensions must match the means')
    if not np.isfinite(m0).all() or not np.isfinite(m1).all():
        raise ValueError('Non-finite posterior mean')
    l0, l1 = checked_cholesky(c0)[0], checked_cholesky(c1)[0]
    lb = np.linalg.cholesky((l0 @ l0.T + l1 @ l1.T) / 2)
    logdet = lambda l: 2 * np.log(np.diag(l)).sum()
    delta = np.linalg.solve(lb, m1 - m0)
    db_mu = float(delta @ delta / 8)
    db_cov = float(.5 * (logdet(lb) - .5 * logdet(l0) - .5 * logdet(l1)))
    db = db_mu + db_cov
    if db < -1e-10:
        raise ValueError('Negative Bhattacharyya distance')
    db = max(0., db)
    return dict(db_mu=db_mu, db_cov=db_cov, db_total=db, db_per_dim=db/p, overlap=float(np.exp(-db/p)))


def classification_metrics(y, p, *, auc_only=False):
    y, p = np.asarray(y).reshape(-1), np.asarray(p, dtype=float).reshape(-1)
    if not y.size or y.shape != p.shape or not np.isin(y,[0,1]).all():
        raise ValueError('Expected nonempty binary labels matching probabilities')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Predictive probabilities must be finite and in [0,1]')
    positive = y == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    auc = float((rankdata(p)[positive].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None
    result = dict(AUC=auc, AUC_reason=None if auc is not None else 'single_class_test_labels')
    if not auc_only:
        result.update(ACC=float(np.mean((p >= .5)==y)), Brier=float(np.mean((p-y)**2)),
                      best_thr=.5, p_test=p)
    return result
