import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class EDMLoss:

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, denoise_fn, data):
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma)
        target = y
        loss = weight.unsqueeze(1) * (D_yn - target) ** 2
        return loss

def edm_loss(model, x_clean):
    return EDMLoss(model.P_mean, model.P_std, model.sigma_data)(model, x_clean).mean(-1).mean()

@torch.no_grad()
def impute_mask_conditional_diffputer_refine(score_net, observed, mask, num_steps):
    return impute_mask(score_net, observed, mask, len(observed), observed.shape[1],
                       num_steps, observed.device, 1)


def _anchor_and_clamp(
    x: torch.Tensor,
    x_obs_clean: torch.Tensor,
    mask_missing: torch.Tensor,
    clamp_val=None,
) -> torch.Tensor:
    if clamp_val is not None:
        x = x.clamp(min=-float(clamp_val), max=float(clamp_val))
    x[~mask_missing] = x_obs_clean[~mask_missing]
    return x


@torch.no_grad()
def masked_langevin_refine(
    score_net,
    x_start: torch.Tensor,
    x_obs_clean: torch.Tensor,
    mask_missing: torch.Tensor,
    num_steps: int,
    sigma: float,
    step_scale: float,
    clamp_val=None,
    *,
    langevin_generator: torch.Generator,
    diagnostic_callback=None,
    diagnostic_timings=None,
) -> torch.Tensor:
    x = x_start.clone().float()
    x_obs_clean = x_obs_clean.float()
    mask_missing = mask_missing.bool()

    def emit(step, drift=None, noise=None):
        import time
        if x.is_cuda: torch.cuda.synchronize(x.device)
        started = time.perf_counter()
        diagnostic_callback(step, x.detach().clone(),
                            None if drift is None else (eta * drift).detach(),
                            None if noise is None else (noise_std * noise).detach())
        if x.is_cuda: torch.cuda.synchronize(x.device)
        if diagnostic_timings is not None:
            diagnostic_timings['seconds'] = diagnostic_timings.get('seconds', 0.) + time.perf_counter() - started

    if diagnostic_callback is not None:
        emit(0)

    n_steps = int(max(0, num_steps))
    if n_steps == 0 or float(step_scale) <= 0.0:
        return _anchor_and_clamp(
            x,
            x_obs_clean,
            mask_missing,
            clamp_val=clamp_val,
        )

    sigma_val = max(float(sigma), 1e-4)
    eta = float(step_scale)
    noise_std = (2.0 * eta) ** 0.5
    sigma_batch = torch.full(
        (x.shape[0],),
        sigma_val,
        device=x.device,
        dtype=x.dtype,
    )

    for step in range(n_steps):
        denoised = score_net(x, sigma_batch)
        score = (denoised - x) / (sigma_val ** 2)

        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=langevin_generator)
        x[mask_missing] = (
            x[mask_missing]
            + eta * score[mask_missing]
            + noise_std * noise[mask_missing]
        )
        x = _anchor_and_clamp(
            x,
            x_obs_clean,
            mask_missing,
            clamp_val=clamp_val,
        )
        if diagnostic_callback is not None:
            emit(step + 1, score, noise)

    return x


def dvai_sampling_metadata(cfg):
    return dict(sampling_mode='single_reverse_masked_langevin_v1', refine_steps=1,
                sample_steps=int(cfg.sample_steps), reverse_updates=int(cfg.sample_steps),
                return_noise_updates=0, L=int(cfg.L),
                langevin_updates=int(cfg.L) if cfg.langevin_step_scale > 0 else 0,
                langevin_sigma=float(cfg.langevin_sigma),
                langevin_step_scale=float(cfg.langevin_step_scale),
                langevin_enabled=cfg.L > 0 and cfg.langevin_step_scale > 0,
                langevin_step_rule='eta=step_scale; sigma=max(sigma,1e-4)',
                langevin_rng='derive_seed:langevin_train/langevin_test; generator_per_sample_call')


randn_like = torch.randn_like
SIGMA_MIN, SIGMA_MAX, rho = .002, 80., 7.
S_churn, S_min, S_max, S_noise = 1., 0., float("inf"), 1.

def validate_budget(value, name, minimum=1):
    from numbers import Integral
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def impute_mask(net, x, mask, num_samples, dim, num_steps=50, device='cuda:0', refine_steps=20):
    validate_budget(num_steps, "sample_steps", 2)
    validate_budget(refine_steps, "refine_steps")
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    x_t = torch.randn([num_samples, dim], device=device)
    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    mask = mask.to(torch.int).to(device)
    x_t = x_t.to(torch.float32) * t_steps[0]
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            N = refine_steps
            if i < num_steps - 1:
                for j in range(N):
                    n_curr = torch.randn_like(x_t).to(device) * t_cur
                    n_prev = torch.randn_like(x_t).to(device) * t_next
                    x_known_t_prev = x + n_prev
                    x_unknown_t_prev = sample_step(net, num_steps, i, t_cur, t_next, x_t)
                    x_t_prev = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask
                    n = torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt()
                    if j == N - 1:
                        x_t = x_t_prev
                    else:
                        x_t = x_t_prev + n
            else:
                n_curr = torch.randn_like(x_t).to(device) * t_cur
                n_prev = torch.randn_like(x_t).to(device) * t_next
                x_known_t_prev = x + n_prev
                x_unknown_t_prev = sample_step(net, num_steps, i, t_cur, t_next, x_t)
                x_t_prev = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask
                x_t = x_t_prev
    return x_t

def sample_step(net, num_steps, i, t_cur, t_next, x_next):
    x_cur = x_next
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

