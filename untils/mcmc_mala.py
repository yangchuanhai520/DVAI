import math
import time
import numpy as np
import torch


def run_mala(model, num_mcmc, burnin, thin=1, step_init=1e-3,
             target_acc=.57, adapt_until=0, seed=42, verbose=0,
             initial=None, return_info=False):
    for name, value in [('num_mcmc', num_mcmc), ('thin', thin), ('adapt_until', adapt_until)]:
        if not isinstance(value, (int, np.integer)) or value < (0 if name == 'adapt_until' else 1):
            raise ValueError(f'Invalid {name}')
    if not math.isfinite(burnin) or not 0 <= burnin < 1:
        raise ValueError('burnin must be in [0,1)')
    burn_start = int(num_mcmc * burnin)
    if adapt_until > burn_start:
        raise ValueError('adapt_until must be <= burn_start')
    if not math.isfinite(step_init) or step_init <= 0 or not 0 < target_acc < 1:
        raise ValueError('Invalid step_init or target_acc')
    data = model.X if hasattr(model, 'X') else model.X_samples
    if not data.is_floating_point():
        raise ValueError('Floating data required')
    started = time.perf_counter()
    rng = torch.Generator(device=data.device).manual_seed(int(seed))
    theta = data.new_zeros(model.D) if initial is None else torch.as_tensor(
        initial, dtype=data.dtype, device=data.device).clone()
    if theta.shape != (model.D,) or not torch.isfinite(theta).all():
        raise ValueError('Invalid initial state')

    def value_gradient(th):
        with torch.enable_grad():
            th = th.detach().requires_grad_(True)
            lp = model.log_joint(th).sum()
            grad, = torch.autograd.grad(lp, th)
        return lp.detach(), grad.detach()

    logp, grad = value_gradient(theta)
    if not torch.isfinite(logp) or not torch.isfinite(grad).all():
        raise ValueError('Nonfinite initial target')
    step = float(step_init)
    kept, warm_accept, sample_accept, failures = [], 0, 0, 0
    for it in range(num_mcmc):
        noise = torch.randn(theta.shape, generator=rng, device=data.device, dtype=data.dtype)
        forward = theta + .5 * step**2 * grad
        proposal = forward + step * noise
        accepted = False
        finite = bool(torch.isfinite(proposal).all())
        if finite:
            lp_new, grad_new = value_gradient(proposal)
            finite = bool(torch.isfinite(lp_new) and torch.isfinite(grad_new).all())
        if finite:
            reverse = proposal + .5 * step**2 * grad_new
            log_alpha = lp_new-logp + ((proposal-forward).square().sum()
                        -(theta-reverse).square().sum())/(2*step**2)
            finite = bool(torch.isfinite(log_alpha))
        uniform = torch.rand((), generator=rng, device=data.device, dtype=data.dtype)
        if finite:
            accepted = bool(uniform.log() < log_alpha)
        else:
            failures += 1
        if accepted:
            theta, logp, grad = proposal, lp_new, grad_new
        if it < burn_start:
            warm_accept += int(accepted)
        else:
            sample_accept += int(accepted)
        if it < adapt_until:
            gain = (it + 10.) ** -.6
            step = math.exp(np.clip(math.log(step) + gain*(int(accepted)-target_acc), -20, 5))
        if it >= burn_start and (it-burn_start) % thin == 0:
            kept.append(theta.detach().cpu().numpy().copy())
        if verbose and (it+1) % verbose == 0:
            print(f'MALA {it+1}/{num_mcmc}, step={step:.6g}', flush=True)
    samples = np.stack(kept)
    info = dict(seed=int(seed), initial=theta.new_tensor(initial).cpu().tolist() if initial is not None else [0.]*model.D,
                num_mcmc=num_mcmc, burn_start=burn_start, adapt_until=adapt_until, thin=thin,
                warmup_acceptance=warm_accept/burn_start if burn_start else None,
                sampling_acceptance=sample_accept/(num_mcmc-burn_start), final_step=step,
                nonfinite_proposals=failures, runtime_seconds=time.perf_counter()-started,
                dtype=str(data.dtype), device=str(data.device))
    return (samples, info) if return_info else samples


def run_mala_chains(model, initials, seeds, **kwargs):
    if len(initials) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('One distinct seed per initial state required')
    results = [run_mala(model, initial=x, seed=s, return_info=True, **kwargs)
               for x, s in zip(initials, seeds)]
    return np.stack([r[0] for r in results]), [r[1] for r in results]
