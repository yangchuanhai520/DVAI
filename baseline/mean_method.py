from untils.metrics import classification_metrics
from untils.inference import fit_vb, posterior_predict_proba, estimate_fields
from untils.time import timed_method, phase, in_phase, measure
from typing import Any, Dict
import time

import numpy as np
import torch


def observed_mean_fill(train, test, train_mask, test_mask):
    train, test = np.asarray(train).copy(), in_phase("test", lambda: np.asarray(test).copy())
    train_mask, test_mask = np.asarray(train_mask, bool), in_phase("test", lambda: np.asarray(test_mask, bool))
    counts = (~train_mask).sum(0)
    if np.any(counts == 0):
        raise ValueError("A training feature has no observed values")
    means = np.where(train_mask, 0., train).sum(0) / counts
    if not np.isfinite(means).all():
        raise ValueError("Non-finite observed training values")
    return np.where(train_mask, means, train), in_phase("test", lambda: np.where(test_mask, means, test))


@timed_method
def run_mean_baseline(
    X_obs_train: np.ndarray,
    y_train: np.ndarray,
    X_obs_test: np.ndarray,
    y_test: np.ndarray,
    prior_var: float,
    vb_cfg: Any,
    device: torch.device,
    theta_mc_samples: int = 50,
    regression_transform=None,
    raw_observed=None,
    train_mask_missing=None,
    test_mask_missing=None,
) -> Dict[str, Any]:

    if train_mask_missing is None or test_mask_missing is None:
        raise ValueError("Mean imputation requires explicit masks")
    X_obs_train, X_obs_test = observed_mean_fill(X_obs_train, X_obs_test, train_mask_missing, test_mask_missing)
    if regression_transform is not None:
        X_obs_train = regression_transform.unstandardize(X_obs_train, raw_observed[0], train_mask_missing)
        with phase("test"):
            X_obs_test = regression_transform.unstandardize(X_obs_test, raw_observed[1], test_mask_missing)

    with measure(device) as measured:
        mu, cov, info = fit_vb(
            X_obs_train=X_obs_train,
            y_train=y_train,
            prior_var=prior_var,
            vb_cfg=vb_cfg,
            device=device,
            return_info=True,
        )
    train_time = measured['seconds']

    with phase("test"):
        t1 = time.perf_counter()
        torch.manual_seed(int(getattr(vb_cfg, "prediction_seed", vb_cfg.seed)))
        p_test = posterior_predict_proba(X=X_obs_test, vb_mu=mu, vb_cov=cov, device=device, n_mc=theta_mc_samples, seed=int(getattr(vb_cfg, 'prediction_seed', vb_cfg.seed)))
        test_time = float(time.perf_counter() - t1)
    metrics = in_phase("evaluation", lambda: classification_metrics(y_test, p_test))

    with phase("evaluation"):
        metrics.update({
            **estimate_fields(mu, cov, True),
            "method": "mean0_vb",
            "inference_sec": train_time, "prediction_sec": test_time,
            "method_detail": "observed_training_mean_in_regression_coordinates",
            "mu": mu,
            "cov": cov,
            "X_train_imp": np.asarray(X_obs_train),
            "X_test_imp": in_phase("test", lambda: np.asarray(X_obs_test)),
            "supports_oos": True,
            "elbo_last": info["elbo_last"],
        })
    return metrics
