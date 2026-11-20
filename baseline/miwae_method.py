from __future__ import annotations


from untils.metrics import classification_metrics

from untils.time import timed_method, phase, in_phase, measure

from typing import Any, Dict, Tuple
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from untils.inference import estimate_fields, fit_vb_mc, predict_proba_mc


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


class _MIWAE(nn.Module):
    def __init__(self, dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.dim = int(dim)
        self.latent_dim = int(latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * dim),
        )

    def encode(self, x_filled, m_obs):
        mu, raw_scale = self.encoder(x_filled * m_obs).chunk(2, dim=-1)
        return mu, torch.nn.functional.softplus(raw_scale).log()

    def decode(self, z):
        mu, scale, df = self.decoder(z).chunk(3, dim=-1)
        return mu, torch.nn.functional.softplus(scale) + .001, torch.nn.functional.softplus(df) + 3.


def _log_normal_diag(x: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor, dim: int = -1):
    return (-0.5 * (((x - mu) / torch.exp(log_std)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))).sum(dim=dim)


def _miwae_loss(model: _MIWAE, x_filled: torch.Tensor, m_obs: torch.Tensor, K: int):

    q_mu, q_log_std = model.encode(x_filled, m_obs)
    B, D = x_filled.shape
    L = q_mu.shape[1]

    eps = torch.randn((int(K), B, L), device=x_filled.device, dtype=x_filled.dtype)
    z = q_mu.unsqueeze(0) + torch.exp(q_log_std).unsqueeze(0) * eps

    dec_mu, dec_scale, dec_df = model.decode(z.reshape(int(K) * B, L))
    dec_mu = dec_mu.reshape(int(K), B, D)
    dec_scale, dec_df = dec_scale.reshape(int(K), B, D), dec_df.reshape(int(K), B, D)

    x3 = x_filled.unsqueeze(0).expand(int(K), B, D)
    m3 = m_obs.unsqueeze(0).expand(int(K), B, D)

    log_px_obs = (torch.distributions.StudentT(dec_df, dec_mu, dec_scale).log_prob(x3) * m3).sum(2)
    log_pz = _log_normal_diag(z, torch.zeros_like(z), torch.zeros_like(z), dim=2)
    log_qz = _log_normal_diag(z, q_mu.unsqueeze(0), q_log_std.unsqueeze(0), dim=2)

    log_w = log_px_obs + log_pz - log_qz
    loss = -(torch.logsumexp(log_w, dim=0) - np.log(float(K))).mean()
    return loss


def _train_miwae(
    X_obs_train: np.ndarray,
    train_mask_missing: np.ndarray,
    cfg: Any,
    device: torch.device,
):
    seed = int(_cfg_get(cfg, "seed", 20260426))
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_filled = np.where(train_mask_missing, 0., X_obs_train).astype(np.float32)
    m_obs = (~np.asarray(train_mask_missing, dtype=bool)).astype(np.float32)

    X_t = torch.as_tensor(X_filled, device=device, dtype=torch.float32)
    M_t = torch.as_tensor(m_obs, device=device, dtype=torch.float32)

    N, D = X_t.shape
    torch.manual_seed(int(_cfg_get(cfg, "init_seed", _cfg_get(cfg, "seed", 2026))))
    model = _MIWAE(
        dim=D,
        latent_dim=int(_cfg_get(cfg, "latent_dim", 8)),
        hidden_dim=int(_cfg_get(cfg, "hidden_dim", 128)),
    ).to(device)

    for layer in model.modules():
        if isinstance(layer, nn.Linear): torch.nn.init.orthogonal_(layer.weight)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(_cfg_get(cfg, "lr", 1e-3)),
        weight_decay=float(_cfg_get(cfg, "weight_decay", 0.0)),
    )

    batch_size = min(int(_cfg_get(cfg, "batch_size", 256)), N)
    epochs = int(_cfg_get(cfg, "epochs", 3000))
    K = int(_cfg_get(cfg, "num_importance_samples", 20))
    monitor_every = max(int(_cfg_get(cfg, "monitor_every", 100)), 1)
    patience = int(_cfg_get(cfg, "patience", 20))

    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    bad = 0
    last_loss = float("inf")

    torch.manual_seed(int(_cfg_get(cfg, "seed", 2026)))
    from untils.seed_device import isolated_rng, check_parameters, record_training
    monitor_seed = int(_cfg_get(cfg, 'monitor_seed', seed ^ 0x714CE))
    best_step, stop_reason = 0, 'max_iter'
    if epochs < 1: raise ValueError('Positive MIWAE budget required')
    model.train()
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(N)
        for offset in range(0, N, batch_size):
            ids = permutation[offset:offset+batch_size]
            loss = _miwae_loss(model, X_t[ids], M_t[ids], K)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MIWAE loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            check_parameters(model.parameters(), gradients=True)
            opt.step()
            check_parameters(model.parameters())

        if epoch % monitor_every == 0 or epoch == epochs:
            with phase("evaluation"), isolated_rng(monitor_seed), torch.no_grad():
                val = float(_miwae_loss(model, X_t, M_t, K).detach().cpu().item())
            if not np.isfinite(val): raise FloatingPointError("Non-finite MIWAE monitor")
            last_loss = val
            if val < best_loss:
                best_loss = val
                best_step = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
            if patience > 0 and bad >= patience:
                stop_reason = "patience"
                break

    model.load_state_dict(best_state)
    model.eval()
    with phase("evaluation"):
        info = dict(actual_steps=epoch, best_step=best_step, best_monitor_value=best_loss,
                    last_monitor_value=last_loss, inner_stop_reason=stop_reason, restored_best=True, monitor_seed=monitor_seed)
        record_training('miwae', info)
        return model, {
            **info,
            "miwae_loss": float(best_loss if np.isfinite(best_loss) else last_loss),
            "trained_epochs": int(epoch),
        }


@torch.no_grad()
def _sample_miwae(
    model: _MIWAE,
    X_obs: np.ndarray,
    mask_missing: np.ndarray,
    cfg: Any,
    device: torch.device,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_filled = np.where(mask_missing, 0., X_obs).astype(np.float32)
    mask_missing = np.asarray(mask_missing, dtype=bool)
    m_obs = (~mask_missing).astype(np.float32)

    X_t = torch.as_tensor(X_filled, device=device, dtype=torch.float32)
    M_t = torch.as_tensor(m_obs, device=device, dtype=torch.float32)

    K = max(1, int(_cfg_get(cfg, "num_importance_samples", 20)))

    q_mu, q_log_std = model.encode(X_t, M_t)
    N, D = X_t.shape
    L = q_mu.shape[1]
    J = int(max(1, num_samples))

    eps_z = torch.randn((K, N, L), device=device, dtype=torch.float32)
    z = q_mu.unsqueeze(0) + torch.exp(q_log_std).unsqueeze(0) * eps_z

    dec_mu, dec_scale, dec_df = model.decode(z.reshape(K * N, L))
    dec_mu = dec_mu.reshape(K, N, D)
    dec_scale, dec_df = dec_scale.reshape(K, N, D), dec_df.reshape(K, N, D)

    x3 = X_t.unsqueeze(0).expand(K, N, D)
    m3 = M_t.unsqueeze(0).expand(K, N, D)
    log_px_obs = (torch.distributions.StudentT(dec_df, dec_mu, dec_scale).log_prob(x3) * m3).sum(2)
    log_pz = _log_normal_diag(z, torch.zeros_like(z), torch.zeros_like(z), dim=2)
    log_qz = _log_normal_diag(z, q_mu.unsqueeze(0), q_log_std.unsqueeze(0), dim=2)
    log_w = log_px_obs + log_pz - log_qz
    weights = torch.softmax(log_w, dim=0)

    idx_jn = torch.multinomial(weights.T, num_samples=J, replacement=True).T
    n_idx = torch.arange(N, device=device).reshape(1, N).expand(J, N)
    candidates = torch.distributions.StudentT(dec_df, dec_mu, dec_scale).sample()
    x_gen = candidates[idx_jn, n_idx]

    samples = x_gen.detach().cpu().numpy().astype(np.float32)
    X_obs_np = np.asarray(X_obs, dtype=np.float32)
    samples[:, ~mask_missing] = X_obs_np[~mask_missing][None, :]

    X_mean = samples.mean(axis=0).astype(np.float32)
    X_mean[~mask_missing] = X_obs_np[~mask_missing]
    return X_mean, samples


@timed_method
def run_miwae_baseline(
    X_obs_train: np.ndarray,
    y_train: np.ndarray,
    X_obs_test: np.ndarray,
    y_test: np.ndarray,
    train_mask_missing: np.ndarray,
    test_mask_missing: np.ndarray,
    prior_var: float,
    vb_cfg: Any,
    miwae_cfg: Any,
    device: torch.device,
    theta_mc_samples: int = 50,
    regression_transform=None,
    raw_observed=None,
) -> Dict[str, Any]:

    from untils.data_process import ObservedTransform
    if regression_transform is None:
        transform = ObservedTransform(X_obs_train, train_mask_missing)
        raw_train, raw_test = X_obs_train.copy(), in_phase("test", lambda: X_obs_test.copy())
        X_obs_train = transform.standardize(raw_train, train_mask_missing)
        X_obs_test = in_phase("test", lambda: transform.standardize(raw_test, test_mask_missing))
        to_regression = transform.unstandardize
    else:
        raw_train, raw_test = raw_observed
        to_regression = regression_transform.unstandardize
    with measure(device) as measured:
        model, stats = _train_miwae(X_obs_train, train_mask_missing, miwae_cfg, device)
    training_sec = measured['seconds']
    impute_start = time.perf_counter()

    Jtr = int(_cfg_get(miwae_cfg, "n_imputations_train", 20))
    Jte = int(_cfg_get(miwae_cfg, "n_imputations_test", 20))

    torch.manual_seed(int(_cfg_get(miwae_cfg, "sampling_seed", miwae_cfg.seed)))
    X_train_imp, X_train_samples = _sample_miwae(
        model=model,
        X_obs=X_obs_train,
        mask_missing=train_mask_missing,
        cfg=miwae_cfg,
        device=device,
        num_samples=Jtr,
    )

    X_train_samples = to_regression(X_train_samples, raw_train, train_mask_missing)
    X_train_imp = X_train_samples.mean(0)
    X_train_imp[~train_mask_missing] = raw_train[~train_mask_missing]
    imputation_sec = time.perf_counter() - impute_start
    with measure(device) as measured:
        mu, cov, elbo_last = fit_vb_mc(
            X_samples=X_train_samples,
            y_train=y_train,
            prior_var=prior_var,
            vb_cfg=vb_cfg,
            device=device,
        )
    inference_sec = measured['seconds']

    with phase("test"):
        t_test0 = time.perf_counter()
        torch.manual_seed(int(_cfg_get(miwae_cfg, "sampling_seed", miwae_cfg.seed)) ^ 0xA5A5A5A5)
        X_test_imp, X_test_samples = _sample_miwae(
            model=model,
            X_obs=X_obs_test,
            mask_missing=test_mask_missing,
            cfg=miwae_cfg,
            device=device,
            num_samples=Jte,
        )

        X_test_samples = to_regression(X_test_samples, raw_test, test_mask_missing)
        X_test_imp = X_test_samples.mean(0)
        X_test_imp[~test_mask_missing] = raw_test[~test_mask_missing]
        imputation_sec += time.perf_counter() - t_test0
        prediction_start = time.perf_counter()
        torch.manual_seed(int(getattr(miwae_cfg, "prediction_seed", miwae_cfg.seed)))
        p_test = predict_proba_mc(X_test_samples, mu, cov, int(theta_mc_samples), seed=int(getattr(miwae_cfg, "prediction_seed", miwae_cfg.seed)))

    prediction_sec = time.perf_counter() - prediction_start
    with phase("evaluation"):
        return {
            **in_phase("evaluation", lambda: classification_metrics(y_test, p_test)),
            **estimate_fields(mu, cov, True),
            "X_train_samples": X_train_samples, "X_test_samples": X_test_samples,
            "gvb_fits": 1, "regression_input_shape": list(X_train_samples.shape),
            "method": "miwae_mc_vb",
            "prediction_seed": int(getattr(miwae_cfg, "prediction_seed", miwae_cfg.seed)),
            "training_diagnostics": {k:stats[k] for k in ["actual_steps","best_step","best_monitor_value","last_monitor_value","inner_stop_reason","restored_best"]},
            "training_sec": training_sec, "imputation_sec": imputation_sec,
            "inference_sec": inference_sec, "prediction_sec": prediction_sec,
            "p_test": p_test,
            "X_train_imp": X_train_imp,
            "X_test_imp": X_test_imp,
            "mu": mu,
            "cov": cov,
            "elbo_last": elbo_last,
            "miwae_loss": float(stats["miwae_loss"]),
            "miwae_epochs": int(stats["trained_epochs"]),
            "miwae_latent_dim": int(_cfg_get(miwae_cfg, "latent_dim", 8)),
            "miwae_hidden_dim": int(_cfg_get(miwae_cfg, "hidden_dim", 128)),
            "miwae_epochs_config": int(_cfg_get(miwae_cfg, "epochs", 3000)),
            "miwae_batch_size": int(_cfg_get(miwae_cfg, "batch_size", 256)),
            "miwae_lr": float(_cfg_get(miwae_cfg, "lr", 1e-3)),
            "miwae_num_importance_samples": int(_cfg_get(miwae_cfg, "num_importance_samples", 20)),
            "miwae_n_imputations_train": int(_cfg_get(miwae_cfg, "n_imputations_train", 20)),
            "miwae_n_imputations_test": int(_cfg_get(miwae_cfg, "n_imputations_test", 20)),
            "miwae_importance_samples": int(_cfg_get(miwae_cfg, "num_importance_samples", 20)),
            "miwae_importance_resampling": True,
            "miwae_fill_source": "zero_encoder_input",
        }
