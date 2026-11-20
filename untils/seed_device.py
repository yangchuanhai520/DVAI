import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42, enforce_determinism: bool = True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if enforce_determinism:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(bool(enforce_determinism))


def derive_seed(master: int, stage: str, rep: int, purpose: str, *keys) -> int:
    import hashlib
    import json
    if stage not in {"smoke", "screen", "check", "formal", "gvb_examine"}:
        raise ValueError(f"Unknown stage: {stage}")
    payload = json.dumps([int(master), stage, int(rep), purpose, *keys], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


from contextlib import contextmanager


@contextmanager
def isolated_rng(seed):
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(int(seed))
        yield


def check_parameters(parameters, gradients=False):
    for parameter in parameters:
        value = parameter.grad if gradients else parameter
        if value is not None and not torch.isfinite(value).all():
            raise FloatingPointError('Non-finite gradient' if gradients else 'Non-finite updated parameter')


_THREAD_LIMITER = None

def configure_single_thread():
    global _THREAD_LIMITER
    from threadpoolctl import threadpool_limits, threadpool_info
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError('Set Torch interop threads before starting experiment work') from exc
    _THREAD_LIMITER = threadpool_limits(limits=1)
    return dict(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                blas=[dict(api=p['internal_api'], threads=p['num_threads']) for p in threadpool_info()])


from contextvars import ContextVar
_TRAINING_EVENTS = ContextVar('training_events', default=None)

@contextmanager
def collect_training():
    events = []
    token = _TRAINING_EVENTS.set(events)
    try: yield events
    finally: _TRAINING_EVENTS.reset(token)


def record_training(component, info):
    events = _TRAINING_EVENTS.get()
    if events is not None: events.append(dict(component=component, **info))
