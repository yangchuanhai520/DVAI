from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import math
import warnings
from typing import Any, Optional
import torch
import torch.nn as nn
from untils.time import phase


@dataclass
class CholeskyGVBConfig:
    lr: float = 0.002
    max_iter: int = 10000
    grad_clip: float = 0.0
    num_mc: int = 20
    window: int = 50
    track_history: bool = True
    seed: int = 42
    jitter: float = 1e-6
    restore_best: bool = False
    min_improve: float = 0.0
    patience: Optional[int] = 500
    beta1: float = 0.9
    beta2: float = 0.9
    step_adaptive: int = 500
    init_scale: float = 0.1
    denominator_floor: float = 1e-12
    min_abs_diag: float = 1e-12
    check_every: int = 1
    eigen_diagnostic_every: int = 0
    final_elbo_samples: int = 2000
    diagnostic_seed: int = 920260


class CholeskyGaussianVB(nn.Module):
    def __init__(self, dim: int, cfg: Any, device="cpu"):
        super().__init__()
        self.dim, self.cfg, self.device = int(dim), cfg, torch.device(device)
        if self.dim < 1:
            raise ValueError("dim must be positive")
        self.mu = nn.Parameter(torch.zeros(self.dim, device=self.device))
        self.C = nn.Parameter(torch.eye(self.dim, device=self.device) *
                              float(getattr(cfg, 'init_scale', 0.1)))
        cols, rows = torch.triu_indices(self.dim, self.dim, device=self.device)
        self.register_buffer('_rows', rows)
        self.register_buffer('_cols', cols)
        self._rng = torch.Generator(device=self.mu.device)
        self._rng.manual_seed(int(getattr(cfg, 'seed', 42)))

    def _apply(self, fn, recurse=True):
        old_device = self.mu.device
        result = super()._apply(fn, recurse=recurse)
        self.device = self.mu.device
        if self.mu.device != old_device:
            self._rng = torch.Generator(device=self.mu.device)
            self._rng.manual_seed(int(getattr(self.cfg, 'seed', 42)))
        return result

    def _build_L(self):
        return torch.tril(self.C)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any('L_unconstrained' in key for key in state_dict):
            raise ValueError('Legacy softplus checkpoint: convert to mu/cov and use set_init_from_mu_cov')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _check_factor(self, C):
        if not torch.isfinite(C).all() or (C.diag().abs() <=
                float(getattr(self.cfg, 'min_abs_diag', 1e-12))).any():
            raise FloatingPointError('Singular/nonfinite triangular factor; reduce lr or change initialization')

    def set_init_from_mu_cov(self, mu_np, cov_np):
        mu = torch.as_tensor(mu_np, device=self.mu.device, dtype=self.mu.dtype).reshape(-1)
        cov = torch.as_tensor(cov_np, device=self.mu.device, dtype=self.mu.dtype)
        if mu.shape != self.mu.shape or cov.shape != self.C.shape:
            raise ValueError('Initialization dimensions do not match')
        if not torch.isfinite(mu).all() or not torch.isfinite(cov).all():
            raise ValueError('Nonfinite initialization')
        if not torch.allclose(cov, cov.T, atol=1e-6, rtol=1e-5):
            raise ValueError('Covariance must be symmetric')
        C, info = torch.linalg.cholesky_ex(cov)
        if info.item():
            jitter = float(getattr(self.cfg, 'jitter', 1e-6))
            if jitter <= 0:
                raise ValueError('Initialization covariance is not positive definite')
            C = torch.linalg.cholesky(cov + jitter * torch.eye(self.dim, device=cov.device, dtype=cov.dtype))
            warnings.warn('Added configured jitter to initialization covariance')
        with torch.no_grad():
            self.mu.copy_(mu)
            self.C.copy_(C)

    def _draw(self, S):
        rng_device = torch.device(self._rng.device)
        if rng_device.type == 'cuda' and rng_device.index is None:
            rng_device = torch.device('cuda', torch.cuda.current_device())
        if rng_device != self.mu.device:
            raise RuntimeError('RNG/device mismatch; move the whole module with .to(...) so _apply() rebuilds the device RNG')
        return torch.randn(S, self.dim, device=self.mu.device,
                           dtype=self.mu.dtype, generator=self._rng)

    def estimate(self, task, epsilon):
        C, mu = self._build_L().detach(), self.mu.detach()
        self._check_factor(C)
        epsilon = epsilon.to(device=mu.device, dtype=mu.dtype)
        if epsilon.ndim != 2 or epsilon.shape[1] != self.dim or len(epsilon) < 1:
            raise ValueError('epsilon must have shape [S,D]')
        with torch.enable_grad():
            theta = (mu + epsilon @ C.T).detach().requires_grad_(True)
            joint = task.log_joint(theta)
            if joint.shape != (len(epsilon),):
                raise ValueError('task.log_joint must return [S]')
            joint_score, = torch.autograd.grad(joint.sum(), theta)
        v = joint_score.detach() + torch.linalg.solve_triangular(
            C.T, epsilon.T, upper=True).T
        gmu = v.mean(0)
        gC = v.T @ epsilon / len(epsilon)
        gradient = torch.cat((gmu, gC[self._rows, self._cols]))
        logq = (-0.5*self.dim*math.log(2*math.pi)
                -C.diag().abs().log().sum()-0.5*epsilon.square().sum(1))
        lb = (joint.detach()-logq).mean()
        if not torch.isfinite(gradient).all() or not torch.isfinite(lb):
            raise FloatingPointError('Nonfinite gradient or ELBO')
        return gradient, lb

    def elbo(self, task, n_samples):
        C = self._build_L().detach()
        self._check_factor(C)
        rng = torch.Generator(device=self.mu.device).manual_seed(
            int(getattr(self.cfg, 'diagnostic_seed', 920260)))
        eps = torch.randn(int(n_samples), self.dim, device=self.mu.device,
                          dtype=self.mu.dtype, generator=rng)
        with torch.no_grad():
            joint = task.log_joint(self.mu.detach() + eps @ C.T)
            logq = (-0.5*self.dim*math.log(2*math.pi)
                    - C.diag().abs().log().sum()-0.5*eps.square().sum(1))
            return (joint-logq).mean()

    def final_elbo(self, task):
        C = self._build_L().detach()
        self._check_factor(C)
        count = int(getattr(self.cfg, 'final_elbo_samples', 2000))
        if count < 2:
            raise ValueError('final_elbo_samples must be >= 2')
        seed = int(getattr(self.cfg, 'diagnostic_seed', 920260))
        rng = torch.Generator(device=self.mu.device).manual_seed(seed)
        values = []
        with torch.no_grad():
            for start in range(0, count, 256):
                eps = torch.randn(min(256, count-start), self.dim, generator=rng,
                                  dtype=self.mu.dtype, device=self.mu.device)
                logq = (-.5*self.dim*math.log(2*math.pi)
                        -C.diag().abs().log().sum()-.5*eps.square().sum(1))
                values.append(task.log_joint(self.mu.detach()+eps@C.T)-logq)
        values = torch.cat(values)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite final ELBO')
        return dict(final_elbo=float(values.mean()),
                    final_elbo_mcse=float(values.std()/math.sqrt(count)),
                    final_elbo_samples=count, diagnostic_seed=seed)

    def fit(self, task):
        cfg = self.cfg
        T, S, W = int(cfg.max_iter), int(cfg.num_mc), int(cfg.window)
        P = getattr(cfg, 'patience', 500)
        P = None if P is None or P <= 0 else int(P)
        beta1, beta2 = float(getattr(cfg,'beta1',0.9)), float(getattr(cfg,'beta2',0.9))
        tau = float(getattr(cfg,'step_adaptive',500))
        lr, floor = float(cfg.lr), float(getattr(cfg,'denominator_floor',1e-12))
        if min(T,S,W) < 1 or not (0 < beta1 < 1 and 0 < beta2 < 1) or min(tau,lr,floor) <= 0:
            raise ValueError('Invalid Algorithm 7 configuration')
        if getattr(cfg,'min_improve',0) != 0 or getattr(cfg,'check_every',1) != 1:
            raise ValueError('Algorithm 7 requires min_improve=0 and check_every=1')
        if getattr(cfg, 'monitor_seed', None) is not None:
            raise ValueError('monitor_seed is obsolete: Algorithm 7 monitors fresh training samples')
        clip = float(getattr(cfg,'grad_clip',0))
        eigen_every = int(getattr(cfg, 'eigen_diagnostic_every', 0))
        if eigen_every < 0:
            raise ValueError('eigen_diagnostic_every must be nonnegative')
        def clipped(g):
            if clip > 0:
                g = g * torch.clamp(clip / g.norm().clamp_min(floor), max=1.)
            return g
        g0, _ = self.estimate(task, self._draw(S))
        m = clipped(g0)
        v = m.square()
        window, raw, smooth, rates = deque(maxlen=W), [], [], []
        diag_history, eigen_history, wait_history = [], [], []
        floor_count = 0
        best, best_step, best_state, wait = -math.inf, None, None, 0
        save_best = bool(getattr(cfg, 'restore_best', False))
        stop = 'max_iter'
        for step in range(1,T+1):
            g, lb = self.estimate(task, self._draw(S))
            value = float(lb)
            window.append(value)
            raw.append(value)
            if len(window) == W:
                average = sum(window)/W
                smooth.append(average)
                if average >= best:
                    best, best_step, wait = average, step-1, 0
                    if save_best:
                        best_state = (self.mu.detach().clone(), self.C.detach().clone())
                else:
                    wait += 1
            g = clipped(g)
            m = beta1*m+(1-beta1)*g
            v = beta2*v+(1-beta2)*g.square()
            alpha = lr*min(1.,tau/step)
            rates.append(alpha)
            floor_count += int((v.sqrt() < floor).sum())
            delta = alpha*m/v.sqrt().clamp_min(floor)
            with torch.no_grad():
                new_mu = self.mu + delta[:self.dim]
                new_C = self._build_L().clone()
                new_C[self._rows,self._cols] += delta[self.dim:]
                self._check_factor(new_C)
                if not torch.isfinite(new_mu).all():
                    raise FloatingPointError('Nonfinite mean update')
                self.mu.copy_(new_mu)
                self.C.copy_(new_C)
                diag_history.append(float(new_C.diag().abs().min()))
                if eigen_every and step % eigen_every == 0:
                    eigen_history.append(float(torch.linalg.eigvalsh(new_C @ new_C.T).min()))
                wait_history.append(wait)
            if P is not None and len(window)==W and wait >= P:
                stop = 'patience'
                break
        restore = best_state is not None
        if restore:
            with torch.no_grad():
                self.mu.copy_(best_state[0]); self.C.copy_(best_state[1])
        with phase("evaluation"):
            info = dict(actual_steps=step, best_step=best_step,
                        best_monitor_value=best if best_step is not None else None,
                        last_monitor_value=raw[-1], inner_stop_reason=stop,
                        restored_best=restore, algorithm='Cholesky GVB Algorithm 7',
                        monitor_objective='rolling mean of fresh-sample pre-update ELBO',
                        gradient_estimator='sampled theta score; manuscript formula',
                        factor_parameterization='direct signed triangular',
                        clipping_enabled=clip>0, beta1=beta1,beta2=beta2,step_adaptive=tau,
                        denominator_floor_activations=floor_count,
                        seed=int(getattr(cfg, 'seed', 2026)), dtype=str(self.mu.dtype), device=str(self.mu.device),
                        min_abs_diag_seen=min(diag_history),
                        min_cov_eigenvalue_seen=min(eigen_history) if eigen_history else None,
                        eigen_diagnostic_every=eigen_every,
                        last_monitor_completed_updates=step-1,
                        returned_completed_updates=best_step if restore else step,
                        best_state_definition='state at end of best rolling window; not maximum single-state ELBO')
            info.update(self.final_elbo(task))
            self.training_info = info
            try:
                from untils.seed_device import record_training
            except ModuleNotFoundError as exc:
                if exc.name not in ('untils','untils.seed_device'):
                    raise
            else:
                record_training('vb', info)
        C = self._build_L().detach().cpu().numpy()
        track = bool(getattr(cfg,'track_history',True))
        return dict(mu=self.mu.detach().cpu().numpy().copy(),cov=C@C.T,
                    elbo_history=raw if track else [],
                    elbo_smooth_history=smooth if track else [],
                    min_abs_diag_history=diag_history if track else [],
                    min_cov_eigenvalue_history=eigen_history if track else [],
                    patience_history=wait_history if track else [],
                    learning_rate_history=rates if track else [], **info)


__all__ = ['CholeskyGaussianVB','CholeskyGVBConfig']
