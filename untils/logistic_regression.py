import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_data(x, y, ndim):
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Data must be torch tensors")
    if x.ndim != ndim or y.ndim != 1 or x.shape[-2] != len(y) or min(x.shape) < 1:
        raise ValueError("Incompatible data shapes")
    if not x.is_floating_point() or x.dtype != y.dtype or x.device != y.device:
        raise ValueError("Data must share floating dtype/device")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not ((y == 0) | (y == 1)).all():
        raise ValueError("Data must be finite with binary responses")


class BayesianLogisticRegression(nn.Module):

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        prior_var: float = 5.0,
        prior_mean: torch.Tensor | None = None,
        prior_cov: torch.Tensor | None = None,
        prior_jitter: float = 1e-4,
    ):
        super().__init__()
        assert X.ndim == 2, "X must be [N,D]"
        assert y.ndim == 1, "y must be [N]"
        self.register_buffer("X", X)
        self.register_buffer("y", y)
        self.N, self.D = X.shape
        _validate_data(X, y, 2)

        if not math.isfinite(prior_var) or prior_var <= 0:
            raise ValueError("prior_var must be finite and positive")
        self.prior_var = float(prior_var)

        for name in ("prior_mean", "prior_L", "prior_logdet"):
            self.register_buffer(name, None)

        if (prior_mean is None) != (prior_cov is None):
            raise ValueError("Provide both prior_mean and prior_cov")
        self.use_full_prior = (prior_mean is not None) and (prior_cov is not None)
        if self.use_full_prior:
            self._set_full_prior(prior_mean, prior_cov, prior_jitter)
        else:
            self.prior_mean = None
            self.prior_L = None
            self.prior_logdet = None


    def _set_full_prior(self, prior_mean: torch.Tensor, prior_cov: torch.Tensor, prior_jitter: float):
        prior_mean = torch.as_tensor(prior_mean, device=self.X.device, dtype=self.X.dtype).reshape(-1)
        prior_cov = torch.as_tensor(prior_cov, device=self.X.device, dtype=self.X.dtype)

        assert prior_mean.ndim == 1 and prior_mean.shape[0] == self.D
        assert prior_cov.ndim == 2 and prior_cov.shape[0] == self.D and prior_cov.shape[1] == self.D

        if not torch.isfinite(prior_mean).all() or not torch.isfinite(prior_cov).all():
            raise ValueError("Nonfinite prior")
        if not math.isfinite(prior_jitter) or prior_jitter < 0:
            raise ValueError("Invalid prior jitter")
        if not torch.allclose(prior_cov, prior_cov.T):
            raise ValueError("Prior covariance must be symmetric")
        cov = prior_cov + float(prior_jitter) * torch.eye(self.D, device=prior_cov.device, dtype=prior_cov.dtype)
        L = torch.linalg.cholesky(cov)

        self.prior_mean = prior_mean.detach()
        self.prior_L = L.detach()
        self.prior_logdet = (2.0 * torch.log(torch.diag(L))).sum().detach()

    def log_joint(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert theta.ndim == 2 and theta.shape[1] == self.D
        if theta.dtype != self.y.dtype or theta.device != self.y.device:
            raise ValueError("theta and data must share dtype/device")

        logits = self.X @ theta.t()
        y = self.y.unsqueeze(1)

        loglik = y * F.logsigmoid(logits) + (1.0 - y) * F.logsigmoid(-logits)
        loglik = loglik.sum(dim=0)

        if self.use_full_prior:
            diff = theta - self.prior_mean.unsqueeze(0)
            sol = torch.cholesky_solve(diff.t(), self.prior_L)
            quad = (diff * sol.t()).sum(dim=1)
            const = self.D * torch.log(theta.new_tensor(2.0 * torch.pi))
            logprior = -0.5 * (quad + self.prior_logdet + const)
            return loglik + logprior


        quad = (theta ** 2).sum(dim=1)
        const = self.D * torch.log(theta.new_tensor(2.0 * torch.pi * self.prior_var))
        logprior = -0.5 * (quad / self.prior_var + const)

        return loglik + logprior


class BayesianLogisticRegressionMC(nn.Module):

    def __init__(
        self,
        X_samples: torch.Tensor,
        y: torch.Tensor,
        prior_var: float = 5.0,
        prior_mean: torch.Tensor | None = None,
        prior_cov: torch.Tensor | None = None,
        prior_jitter: float = 1e-4,
    ):
        super().__init__()
        assert X_samples.ndim == 3, "X_samples must be [J,N,D]"
        assert y.ndim == 1, "y must be [N]"

        self.register_buffer("X_samples", X_samples)
        self.register_buffer("y", y)
        self.J, self.N, self.D = X_samples.shape
        _validate_data(X_samples, y, 3)

        assert y.shape[0] == self.N, "y length must match N in X_samples"

        if not math.isfinite(prior_var) or prior_var <= 0:
            raise ValueError("prior_var must be finite and positive")
        self.prior_var = float(prior_var)
        for name in ("prior_mean", "prior_L", "prior_logdet"):
            self.register_buffer(name, None)

        if (prior_mean is None) != (prior_cov is None):
            raise ValueError("Provide both prior_mean and prior_cov")
        self.use_full_prior = (prior_mean is not None) and (prior_cov is not None)
        if self.use_full_prior:
            self._set_full_prior(prior_mean, prior_cov, prior_jitter)
        else:
            self.prior_mean = None
            self.prior_L = None
            self.prior_logdet = None


    def _set_full_prior(self, prior_mean: torch.Tensor, prior_cov: torch.Tensor, prior_jitter: float):
        prior_mean = torch.as_tensor(prior_mean, device=self.X_samples.device, dtype=self.X_samples.dtype).reshape(-1)
        prior_cov = torch.as_tensor(prior_cov, device=self.X_samples.device, dtype=self.X_samples.dtype)

        assert prior_mean.ndim == 1 and prior_mean.shape[0] == self.D
        assert prior_cov.ndim == 2 and prior_cov.shape[0] == self.D and prior_cov.shape[1] == self.D

        if not torch.isfinite(prior_mean).all() or not torch.isfinite(prior_cov).all():
            raise ValueError("Nonfinite prior")
        if not math.isfinite(prior_jitter) or prior_jitter < 0:
            raise ValueError("Invalid prior jitter")
        if not torch.allclose(prior_cov, prior_cov.T):
            raise ValueError("Prior covariance must be symmetric")
        cov = prior_cov + float(prior_jitter) * torch.eye(self.D, device=prior_cov.device, dtype=prior_cov.dtype)
        L = torch.linalg.cholesky(cov)

        self.prior_mean = prior_mean.detach()
        self.prior_L = L.detach()
        self.prior_logdet = (2.0 * torch.log(torch.diag(L))).sum().detach()

    def _log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        if self.use_full_prior:
            diff = theta - self.prior_mean.unsqueeze(0)
            sol = torch.cholesky_solve(diff.t(), self.prior_L)
            quad = (diff * sol.t()).sum(dim=1)
            const = self.D * torch.log(theta.new_tensor(2.0 * torch.pi))
            return -0.5 * (quad + self.prior_logdet + const)


        quad = (theta ** 2).sum(dim=1)
        const = self.D * torch.log(theta.new_tensor(2.0 * torch.pi * self.prior_var))
        return -0.5 * (quad / self.prior_var + const)

    def log_joint(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert theta.ndim == 2 and theta.shape[1] == self.D
        if theta.dtype != self.y.dtype or theta.device != self.y.device:
            raise ValueError("theta and data must share dtype/device")

        logits = torch.einsum("jnd,sd->jns", self.X_samples, theta)
        y = self.y.reshape(1, self.N, 1)

        logp = y * F.logsigmoid(logits) + (1.0 - y) * F.logsigmoid(-logits)

        loglik_per_obs = torch.logsumexp(logp, dim=0) - math.log(float(self.J))
        loglik = loglik_per_obs.sum(dim=0)

        return loglik + self._log_prior(theta)
