import numpy as np
from scipy.special import expit


def _validate_rate(rate, maximum=1.):
    if isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (int, float, np.integer, np.floating)):
        raise TypeError("Missing rate must be a real numeric scalar, not a boolean")
    if not np.isfinite(rate) or not 0 <= rate <= maximum:
        raise ValueError(f"Missing rate must be finite and in [0, {maximum}]")
    return float(rate)


MAR_SLOPE = 0.5


def mar_target_rate(d, rate):
    if isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 2:
        raise ValueError("MAR requires at least two covariates (D >= 2)")
    maximum = (d - 1) / d
    rate = _validate_rate(rate, maximum)
    if not 0 < rate < maximum:
        raise ValueError(f"MAR fitting requires 0 < rate < (D-1)/D = {maximum}")
    return d * rate / (d - 1)


def _mar_covariates(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError("MAR requires a nonempty covariate matrix with D >= 2")
    if not np.isfinite(X[:, 0]).all():
        raise ValueError("MAR first covariate must be completely observed and finite in train and test")
    return X


def mar_probabilities(X, center, scale, intercept):
    X = _mar_covariates(X)
    if not np.isfinite([center, scale, intercept]).all() or scale <= 0:
        raise ValueError("MAR requires finite center/intercept and a positive finite driver scale")
    score = MAR_SLOPE * ((X[:, 0] - center) / scale)
    if not np.isfinite(score).all():
        raise ValueError("Non-finite standardized MAR driver")
    probabilities = np.zeros(X.shape, dtype=np.float64)
    probabilities[:, 1:] = expit(intercept + score)[:, None]
    return probabilities


def sample_missing_mask(probabilities, uniforms):
    uniforms = np.asarray(uniforms)
    if uniforms.shape != probabilities.shape or not np.issubdtype(uniforms.dtype, np.number):
        raise ValueError("Uniforms must be numeric and match X shape")
    if not np.isfinite(uniforms).all() or np.any((uniforms < 0) | (uniforms >= 1)):
        raise ValueError("Uniforms must be finite and in [0, 1)")
    return uniforms < probabilities
