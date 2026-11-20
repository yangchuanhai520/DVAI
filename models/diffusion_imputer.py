import time
from typing import Any, Dict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.diffusion_model import EDMModel, MLPDiffusion
from models.diffusion_utils import edm_loss, set_seed
from untils.time import phase

DIFFUSION_EARLY_STOP_PATIENCE = 500
DIFFUSION_TRAINING_PROTOCOL = "shared_edm_patience500_v2"


def _clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class ConditionalDiffusionImputer:
    @torch.no_grad()
    def sample(self, observed, mask, J, *, langevin_seed, timings=None, diagnostic_callback=None):
        from models.diffusion_utils import impute_mask_conditional_diffputer_refine, masked_langevin_refine
        cfg = self.cfg
        if J < 1:
            raise ValueError("J must be positive")
        x = torch.as_tensor(observed, dtype=torch.float32, device=self.device)
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        from models.diffusion_utils import validate_budget
        validate_budget(cfg.L, "L", 0)
        if not np.isfinite(cfg.langevin_sigma) or not np.isfinite(cfg.langevin_step_scale) or cfg.langevin_sigma < 0 or cfg.langevin_step_scale < 0:
            raise ValueError("Invalid Langevin parameters")
        langevin_generator = torch.Generator(device=x.device).manual_seed(int(langevin_seed))
        langevin_sec = 0.
        diagnostic_timings = {}
        samples = []
        def clock():
            if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
            return time.perf_counter()
        started = clock()
        for j in range(J):
            draw = impute_mask_conditional_diffputer_refine(
                self.model, x, m, cfg.sample_steps)
            t_langevin = clock()
            draw = masked_langevin_refine(self.model, draw, x, m, cfg.L,
                                         cfg.langevin_sigma, cfg.langevin_step_scale,
                                         langevin_generator=langevin_generator,
                                         diagnostic_callback=(None if diagnostic_callback is None else
                                             lambda step, value, drift, noise: diagnostic_callback(j, step, value, drift, noise)),
                                         diagnostic_timings=diagnostic_timings)
            langevin_sec += clock() - t_langevin

            if not torch.isfinite(draw).all():
                raise FloatingPointError("Non-finite diffusion sample")
            samples.append(draw.cpu().numpy())
        if timings is not None:
            excluded = diagnostic_timings.get('seconds', 0.)
            timings.update(conditional_sampling_sec=clock()-started-excluded,
                           langevin_sec=langevin_sec-excluded, diagnostic_sec=excluded)
        return np.stack(samples)

    def __init__(self, dim: int, cfg: Any, device: torch.device):
        self.cfg = cfg
        self.device = device
        if getattr(cfg, "init_seed", getattr(cfg, "train_seed", None)) is not None:
            set_seed(int(getattr(cfg, "init_seed", cfg.train_seed)))
        self.model = EDMModel(
            net=MLPDiffusion(int(dim), int(cfg.hidden)),
            sigma_min=float(cfg.sigma_min),
            sigma_max=float(cfg.sigma_max),
            sigma_data=float(cfg.sigma_data),
            P_mean=float(cfg.edm_P_mean),
            P_std=float(cfg.edm_P_std),
        ).to(self.device)

    def fit(self, x_train_complete):
        return train_diffusion(self.model, x_train_complete, self.cfg, self.device,
                                             getattr(self.cfg, 'train_seed', None) or 2026)


def train_diffusion(model, completed, cfg, device, seed):
    from untils.seed_device import isolated_rng, check_parameters, record_training
    x = torch.as_tensor(completed, dtype=torch.float32, device=device)
    if len(x) < 1 or cfg.epochs < 1 or not torch.isfinite(x).all():
        raise ValueError('Invalid diffusion training data or budget')
    generator = torch.Generator().manual_seed(int(seed) ^ 0x714CE)
    loader = torch.utils.data.DataLoader(x, batch_size=cfg.batch_size, shuffle=True,
                                       drop_last=False, num_workers=0, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.9, patience=50, min_lr=0.)
    best, bad, steps, best_epoch = float('inf'), 0, 0, 0
    state = None
    model.train()
    with isolated_rng(seed):
        for epoch in range(1, cfg.epochs + 1):
            total = 0.
            for batch in loader:
                loss = edm_loss(model, batch)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite diffusion loss')
                total += float(loss.detach()) * len(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                check_parameters(model.parameters(), gradients=True)
                optimizer.step()
                check_parameters(model.parameters())
                steps += 1
            value = total / len(x)
            scheduler.step(value)
            if value < best:
                best, bad, state, best_epoch = value, 0, _clone_state_dict(model), epoch
            else:
                bad += 1
            if bad >= DIFFUSION_EARLY_STOP_PATIENCE: break
    model.load_state_dict(state)
    model.eval()
    with phase("evaluation"):
        info = dict(actual_epochs=epoch, actual_optimizer_steps=steps, actual_steps=steps,
                    best_step=best_epoch, best_epoch=best_epoch, best_loss=best, diff_loss=best,
                    last_loss=value, best_monitor_value=best, last_monitor_value=value,
                    inner_stop_reason='patience' if bad >= DIFFUSION_EARLY_STOP_PATIENCE else 'max_iter', restored_best=True,
                    consecutive_no_improvement=bad, early_stop_patience=DIFFUSION_EARLY_STOP_PATIENCE,
                    training_protocol=DIFFUSION_TRAINING_PROTOCOL,
                    steps=steps, loader_seed=int(seed)^0x714CE)
        record_training('diffusion', info)
    return info
