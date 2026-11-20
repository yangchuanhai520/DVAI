from typing import Any
import numpy as np
import torch
from configs import dvai_config
from untils.metrics import checked_cholesky
from untils.cholesky_gvb import CholeskyGaussianVB
from untils.logistic_regression import BayesianLogisticRegression, BayesianLogisticRegressionMC


@torch.no_grad()
def posterior_predict_proba(
    X: np.ndarray,
    vb_mu: np.ndarray,
    vb_cov: np.ndarray,
    device: torch.device,
    n_mc: int = 50,
    seed=None,
) -> np.ndarray:
    if seed is not None:
        return checked_predict(X,vb_mu,vb_cov,seed,n_mc)
    X_np = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X_np).all():
        raise FloatingPointError("Non-finite prediction input")

    mu_np = np.asarray(vb_mu, dtype=np.float32).reshape(-1)
    if not np.isfinite(mu_np).all():
        raise FloatingPointError("Non-finite prediction input")

    cov_np = np.asarray(vb_cov, dtype=np.float32)
    if not np.isfinite(cov_np).all():
        raise FloatingPointError("Non-finite prediction input")
    factor, _ = checked_cholesky(cov_np, 'baseline posterior')
    X_t = torch.as_tensor(X_np, device=device, dtype=torch.float32)
    mu = torch.as_tensor(mu_np, device=device, dtype=torch.float32).reshape(1, -1)
    L = torch.as_tensor(factor, device=device, dtype=torch.float32)
    d = len(mu_np)
    n_mc = int(max(1, n_mc))
    eps = torch.randn((n_mc, d), device=device, dtype=torch.float32)
    theta = mu + eps @ L.T
    logits = X_t @ theta.T
    if not torch.isfinite(logits).all(): raise FloatingPointError("Non-finite predictive logits")
    p_hat = torch.sigmoid(logits).mean(dim=1).detach().cpu().numpy()
    if not np.isfinite(p_hat).all():
        raise FloatingPointError("Non-finite posterior predictive probabilities")
    return p_hat


def fit_vb(
    X_obs_train: np.ndarray,
    y_train: np.ndarray,
    prior_var: float,
    vb_cfg: Any,
    device: torch.device,
    return_info: bool = False,
):
    X_t = torch.as_tensor(X_obs_train, device=device, dtype=torch.float32)
    y_t = torch.as_tensor(np.asarray(y_train, dtype=np.float32), device=device, dtype=torch.float32)

    model = BayesianLogisticRegression(X_t, y_t, prior_var=float(prior_var))
    vb = CholeskyGaussianVB(dim=X_obs_train.shape[1], cfg=vb_cfg, device=device)
    out = vb.fit(model)

    mu = np.asarray(out["mu"]).reshape(-1)
    cov = np.asarray(out["cov"])
    if return_info:
        hist = out.get("elbo_history", [])
        info = {"elbo_last": float(hist[-1]) if len(hist) > 0 else ""}
        return mu, cov, info
    return mu, cov


def estimate_fields(theta, cov, posterior_available):
    theta = np.asarray(theta)
    if posterior_available and cov is None:
        raise ValueError("Posterior covariance required")
    return dict(theta_estimate=theta, posterior_available=posterior_available,
                posterior_mean=theta if posterior_available else None,
                posterior_cov=cov if posterior_available else None,
                missingness_model="ignored")


def fit_vb_mc(
    X_samples: np.ndarray,
    y_train: np.ndarray,
    prior_var: float,
    vb_cfg: Any,
    device: torch.device,
):
    task = BayesianLogisticRegressionMC(
        torch.tensor(np.asarray(X_samples, dtype=np.float32), device=device, dtype=torch.float32),
        torch.tensor(np.asarray(y_train, dtype=np.float32), device=device, dtype=torch.float32),
        prior_var=float(prior_var),
    )
    vb = CholeskyGaussianVB(dim=int(X_samples.shape[2]), cfg=vb_cfg, device=device)
    out = vb.fit(task)
    hist = out.get("elbo_history", [])
    return (
        np.asarray(out["mu"], dtype=np.float32).reshape(-1),
        np.asarray(out["cov"], dtype=np.float32),
        float(hist[-1]) if len(hist) > 0 else "",
    )


@torch.no_grad()
def predict_proba_mc(X_samples: np.ndarray, mu: np.ndarray, cov: np.ndarray, n_mc: int, seed) -> np.ndarray:
    return checked_predict(X_samples, mu, cov, seed, n_mc)


def checked_predict(samples, mean, cov, seed, n_mc=50):
    from scipy.special import expit
    samples, mean = np.asarray(samples, dtype=float), np.asarray(mean, dtype=float).reshape(-1)
    if not np.isfinite(samples).all() or not np.isfinite(mean).all():
        raise ValueError('Non-finite completed samples or posterior mean')
    factor, _ = checked_cholesky(cov, 'predictive posterior')
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    theta = mean + rng.standard_normal((int(n_mc), len(mean))) @ factor.T
    logits = samples @ theta.T
    if not np.isfinite(logits).all(): raise FloatingPointError('Non-finite predictive logits')
    probabilities = expit(logits).mean(-1)
    if samples.ndim == 3:
        probabilities = probabilities.mean(0)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError('Non-finite predictive probabilities')
    return probabilities


class StopTracker:
    def __init__(self, tolerance=dvai_config.posterior_tol, patience=dvai_config.patience, min_iter=dvai_config.min_iter):
        self.tolerance, self.patience, self.min_iter = tolerance, patience, min_iter
        self.streak = 0
        self.first_stop_k = None

    def update(self, k, distance):
        if distance is not None and not np.isfinite(distance):
            raise ValueError("Invalid convergence distance")
        self.streak = self.streak + 1 if distance is not None and distance < self.tolerance else 0
        stop = k >= self.min_iter and self.streak >= self.patience
        if stop and self.first_stop_k is None:
            self.first_stop_k = k
        return stop
