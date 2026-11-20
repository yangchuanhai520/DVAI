from __future__ import annotations

from untils.metrics import classification_metrics

from untils.time import timed_method, phase, in_phase, measure

from typing import Any, Dict
import time
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from untils.inference import (
    fit_vb, estimate_fields,
    posterior_predict_proba,
)


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


def _initial_mean_fill(X_obs: np.ndarray, mask_missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_obs = np.asarray(X_obs, dtype=np.float32)
    mask_missing = np.asarray(mask_missing, dtype=bool)
    obs = ~mask_missing
    denom = obs.sum(axis=0).astype(np.float64)
    if np.any(denom == 0): raise ValueError("MissForest requires observations in each training column")
    col_mean = np.where(obs, X_obs, 0.).sum(axis=0).astype(np.float64) / denom
    col_mean = col_mean.astype(np.float32)

    X_imp = X_obs.copy()
    if int(mask_missing.sum()) > 0:
        rows, cols = np.where(mask_missing)
        X_imp[rows, cols] = col_mean[cols]
    return X_imp.astype(np.float32), col_mean


def _missing_delta(X_old: np.ndarray, X_new: np.ndarray, mask_missing: np.ndarray) -> float:
    mask_missing = np.asarray(mask_missing, dtype=bool)
    if int(mask_missing.sum()) == 0:
        return 0.0
    old_vals = np.asarray(X_old, dtype=np.float64)[mask_missing]
    new_vals = np.asarray(X_new, dtype=np.float64)[mask_missing]
    denom = float(np.sum(np.asarray(X_new, dtype=float) ** 2))
    if denom == 0:
        if np.array_equal(new_vals,old_vals): return 0.
        raise ValueError("Undefined MissForest relative change")
    return float(np.sum((new_vals - old_vals) ** 2) / denom)


def _fit_missforest(
    X_obs_train: np.ndarray,
    X_obs_test: np.ndarray,
    train_mask_missing: np.ndarray,
    test_mask_missing: np.ndarray,
    cfg: Any,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    n_estimators = int(_cfg_get(cfg, "n_estimators", 100))
    max_iter = int(_cfg_get(cfg, "max_iter", 10))
    max_depth = _cfg_get(cfg, "max_depth", None)
    min_samples_leaf = int(_cfg_get(cfg, "min_samples_leaf", 1))
    max_features = _cfg_get(cfg, "max_features", "sqrt")
    random_state = int(_cfg_get(cfg, "random_state", 20260426))
    n_jobs = 1
    transform_iter = max(1, int(_cfg_get(cfg, "transform_iter", 5)))

    X_train_obs = np.asarray(X_obs_train, dtype=np.float32)
    X_test_obs = in_phase("test", lambda: np.asarray(X_obs_test, dtype=np.float32))
    train_mask_missing = np.asarray(train_mask_missing, dtype=bool)
    test_mask_missing = in_phase("test", lambda: np.asarray(test_mask_missing, dtype=bool))

    X_train_imp, col_mean = _initial_mean_fill(X_train_obs, train_mask_missing)
    with phase("test"):
        X_test_imp = X_test_obs.copy()
        if int(test_mask_missing.sum()) > 0:
            rows, cols = np.where(test_mask_missing)
            X_test_imp[rows, cols] = col_mean[cols]

    D = int(X_train_imp.shape[1])
    if D < 2: raise ValueError("MissForest requires at least two features")
    if max_features == "sqrt": max_features = max(1,int(np.sqrt(D)))
    target_cols = np.flatnonzero(train_mask_missing.any(axis=0) | test_mask_missing.any(axis=0))
    if target_cols.size == 0:
        return X_train_imp.astype(np.float32), in_phase("test", lambda: X_test_imp.astype(np.float32)), 0, 0.0

    miss_rate = train_mask_missing[:, target_cols].mean(axis=0)
    order = target_cols[np.argsort(miss_rate, kind="stable")]
    predictors = {int(j): np.array([c for c in range(D) if c != int(j)], dtype=int) for j in order}

    prev_delta = np.inf
    accepted_iter = 0
    accepted_delta = 0.0
    accepted_models: dict[int, RandomForestRegressor] = {}

    for it in range(1, max_iter + 1):
        X_before = X_train_imp.copy()
        models_iter: dict[int, RandomForestRegressor] = {}

        for j_raw in order:
            j = int(j_raw)
            obs_rows = ~train_mask_missing[:, j]
            if int(obs_rows.sum()) < 2:
                raise ValueError("MissForest needs at least two observed targets")

            pred_cols = predictors[j]
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state + 997 * it + j,
                n_jobs=n_jobs,
            )
            rf.fit(X_train_imp[obs_rows][:, pred_cols], X_train_obs[obs_rows, j])
            models_iter[j] = rf

            miss_rows = train_mask_missing[:, j]
            if int(miss_rows.sum()) > 0:
                X_train_imp[miss_rows, j] = rf.predict(X_train_imp[miss_rows][:, pred_cols]).astype(np.float32)

        delta = _missing_delta(X_before, X_train_imp, train_mask_missing)
        if it > 1 and it < max_iter and delta >= prev_delta:
            X_train_imp = X_before
            break

        prev_delta = delta
        accepted_delta = delta
        accepted_iter = it
        accepted_models = models_iter

    with phase("test"):
        for _ in range(transform_iter):
            for j_raw in order:
                j = int(j_raw)
                miss_rows = test_mask_missing[:, j]
                if int(miss_rows.sum()) == 0 or j not in accepted_models:
                    continue
                pred_cols = predictors[j]
                X_test_imp[miss_rows, j] = accepted_models[j].predict(X_test_imp[miss_rows][:, pred_cols]).astype(np.float32)
            X_test_imp[~test_mask_missing] = X_test_obs[~test_mask_missing]

    X_train_imp[~train_mask_missing] = X_train_obs[~train_mask_missing]
    with phase("test"):
        X_test_imp[~test_mask_missing] = X_test_obs[~test_mask_missing]
    return X_train_imp.astype(np.float32), in_phase("test", lambda: X_test_imp.astype(np.float32)), int(accepted_iter), float(accepted_delta)


@timed_method
def run_missforest_baseline(
    X_obs_train: np.ndarray,
    y_train: np.ndarray,
    X_obs_test: np.ndarray,
    y_test: np.ndarray,
    train_mask_missing: np.ndarray,
    test_mask_missing: np.ndarray,
    prior_var: float,
    vb_cfg: Any,
    missforest_cfg: Any,
    device: torch.device,
    theta_mc_samples: int = 50,
    regression_transform=None,
    raw_observed=None,
) -> Dict[str, Any]:

    t_train0 = time.perf_counter()
    X_train_imp, X_test_imp, mf_iter, mf_delta = _fit_missforest(
        X_obs_train=X_obs_train,
        X_obs_test=X_obs_test,
        train_mask_missing=train_mask_missing,
        test_mask_missing=test_mask_missing,
        cfg=missforest_cfg,
    )

    imputation_sec = time.perf_counter() - t_train0
    if regression_transform is not None:
        X_train_imp = regression_transform.unstandardize(X_train_imp, raw_observed[0], train_mask_missing)
        with phase("test"):
            X_test_imp = regression_transform.unstandardize(X_test_imp, raw_observed[1], test_mask_missing)
    originals = raw_observed if raw_observed is not None else (X_obs_train,X_obs_test)
    X_train_imp, X_test_imp = np.asarray(X_train_imp,dtype=float),in_phase("test", lambda: np.asarray(X_test_imp,dtype=float))
    for i,(completed,original,mask) in enumerate(zip((X_train_imp,X_test_imp),originals,(train_mask_missing,test_mask_missing))):
        with phase("test" if i else "train"):
            completed[~mask] = np.asarray(original)[~mask]
    with measure(device) as measured:
        mu, cov, info = fit_vb(
            X_obs_train=X_train_imp,
            y_train=y_train,
            prior_var=prior_var,
            vb_cfg=vb_cfg,
            device=device,
            return_info=True,
        )
    inference_sec = measured['seconds']

    with phase("test"):
        t_test0 = time.perf_counter()
        torch.manual_seed(int(getattr(vb_cfg, "prediction_seed", vb_cfg.seed)))
        p_test = posterior_predict_proba(X=X_test_imp, vb_mu=mu, vb_cov=cov, device=device, n_mc=int(theta_mc_samples), seed=int(getattr(vb_cfg, 'prediction_seed', vb_cfg.seed)))
        test_time = float(time.perf_counter() - t_test0)
    metrics = in_phase("evaluation", lambda: classification_metrics(y_test, p_test))

    with phase("evaluation"):
        metrics.update({
            **estimate_fields(mu, cov, True),
            "method": "missforest_vb",
            "imputation_sec": imputation_sec, "inference_sec": inference_sec, "prediction_sec": test_time,
            "mu": mu,
            "cov": cov,
            "X_train_imp": X_train_imp,
            "X_test_imp": X_test_imp,
            "supports_oos": True,
            "elbo_last": info["elbo_last"],
            "missforest_iter": int(mf_iter),
            "missforest_delta": float(mf_delta),
        })
    return metrics
