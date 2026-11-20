from functools import lru_cache
import numpy as np
from scipy.special import expit, roots_hermitenorm
from scipy.optimize import brentq
from generate_miss.fixed_design import D, MU, SIGMA

from generate_miss.data_miss import MAR_SLOPE, mar_target_rate, mar_probabilities, sample_missing_mask
MNAR_SLOPE = 0.5
_NODES, _QUADRATURE_WEIGHTS = roots_hermitenorm(128)
_QUADRATURE_WEIGHTS /= np.sqrt(2 * np.pi)


def validate_missing_rate(mechanism, rate):
    if mechanism not in {"MCAR", "MAR", "MNAR"}:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    maximum = (D - 1) / D if mechanism == "MAR" else 1.
    if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (int, float, np.integer, np.floating)):
        raise ValueError("Missing rate must be numeric")
    if not np.isfinite(rate) or not 0 <= rate <= maximum:
        raise ValueError(f"{mechanism} rate must be in [0, {maximum}]")
    return float(rate)


@lru_cache(maxsize=256)
def normal_intercept(rate, variance):
    if not np.isfinite(rate) or not 0 < rate < 1:
        raise ValueError("Intercept requires an interior probability")
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("Invalid normal score variance")
    scores = np.sqrt(variance) * _NODES
    residual = lambda a: float(expit(a + scores) @ _QUADRATURE_WEIGHTS) - rate
    bound = 1.
    for _ in range(20):
        if residual(-bound) < 0 < residual(bound):
            return float(brentq(residual, -bound, bound, xtol=1e-13))
        bound *= 2
    raise RuntimeError("Could not bracket normal logistic intercept")


def missingness_parameters(mechanism, rate):
    rate = validate_missing_rate(mechanism, rate)
    target = rate / ((D - 1) / D) if mechanism == "MAR" else rate
    boundary = "all_observed" if target == 0 else "all_eligible_missing" if target == 1 else None
    info = dict(mechanism=mechanism, target_missing_rate=rate, boundary=boundary)
    if mechanism == "MCAR":
        return info
    if mechanism == "MAR":
        target = target if boundary else mar_target_rate(D, rate)
        info.update(D=D, driver_name="X1", driver_indices=[0], target_indices=list(range(1, D)),
                    slope=MAR_SLOPE, center=float(MU[0]), scale=float(np.sqrt(SIGMA[0, 0])),
                    calibration="standard_normal_population", target_rate=target,
                    intercept=None if boundary else normal_intercept(target, MAR_SLOPE**2))
    else:
        info.update(type="self_masking", driver="standardized_self_value",
                    slope=MNAR_SLOPE, center=MU.tolist(), scale=np.sqrt(np.diag(SIGMA)).tolist(),
                    calibration="normal_population",
                    intercept=None if boundary else normal_intercept(rate, MNAR_SLOPE**2))
    return info


def fixed_mask_probabilities(X, mechanism, rate):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != D or not np.isfinite(X).all():
        raise ValueError("Expected finite nonempty N by 10 covariates")
    info = missingness_parameters(mechanism, rate)
    if mechanism == "MCAR":
        return np.full(X.shape, rate, dtype=float)
    probs = np.zeros_like(X)
    eligible = tuple(range(1, X.shape[1])) if mechanism == "MAR" else tuple(range(D))
    if info['boundary']:
        probs[:, eligible] = float(info['boundary'] == 'all_eligible_missing')
        return probs
    if mechanism == "MAR":
        return mar_probabilities(X, info['center'], info['scale'], info['intercept'])
    z = (X - np.asarray(info['center'])) / np.asarray(info['scale'])
    return expit(info['intercept'] + MNAR_SLOPE * z)


def fixed_missing_mask(X, mechanism, rate, uniforms):
    probabilities = fixed_mask_probabilities(X, mechanism, rate)
    return sample_missing_mask(probabilities, uniforms)


def summarize_missingness(mask):
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        raise TypeError("mask must have boolean dtype")
    if mask.ndim != 2 or min(mask.shape) == 0:
        raise ValueError("mask must be a nonempty two-dimensional array")
    n_samples, n_features = mask.shape
    missing_counts = mask.sum(axis=0)
    total_missing = int(missing_counts.sum())
    return dict(n_samples=int(n_samples), n_features=int(n_features), total_missing=total_missing,
                overall_missing_rate=total_missing / (n_samples * n_features),
                per_feature=[dict(feature_index=j + 1, missing_count=int(count),
                                  observed_count=int(n_samples - count), missing_rate=float(count / n_samples))
                             for j, count in enumerate(missing_counts)])
