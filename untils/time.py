TIMING_PROTOCOL = "method_v4_excludes_reporting_outer_stop_postfit_diagnostics"
SENSITIVITY_TIMING_PROTOCOL = "dvai_final_imputation_prediction_v3_excludes_reporting_outer_stop_postfit_diagnostics"
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from time import perf_counter
import torch

_active = ContextVar("baseline_timer", default=None)


class _Timer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.stage = "train"
        self.elapsed = {"train": 0., "test": 0., "evaluation": 0.}
        self.sync()
        self.started = self.last = perf_counter()

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def switch(self, stage):
        self.sync()
        now = perf_counter()
        self.elapsed[self.stage] += now - self.last
        self.last, self.stage = now, stage


@contextmanager
def phase(stage):
    timer = _active.get()
    if timer is None or timer.stage == stage:
        yield
        return
    previous = timer.stage
    timer.switch(stage)
    try:
        yield
    finally:
        timer.switch(previous)


def in_phase(stage, operation):
    with phase(stage):
        return operation()


@contextmanager
def measure(device):
    timer = _active.get()
    token = None
    if timer is None:
        timer = _Timer(device)
        token = _active.set(timer)
    else:
        timer.switch(timer.stage)
    before = timer.elapsed.copy()
    result = {}
    try:
        yield result
    finally:
        try:
            timer.switch(timer.stage)
            durations = {key: timer.elapsed[key] - before[key] for key in before}
            result.update(seconds=durations['train'] + durations['test'],
                          train_sec=durations['train'], test_sec=durations['test'],
                          excluded_sec=durations['evaluation'])
        finally:
            if token is not None:
                _active.reset(token)


def timed_method(function):
    sig = signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _active.get() is not None:
            return function(*args, **kwargs)
        arguments = sig.bind(*args, **kwargs)
        timer = _Timer(arguments.arguments.get("device", "cpu"))
        token = _active.set(timer)
        try:
            result = function(*args, **kwargs)
            timer.switch("train")
            result.update(train_time_sec=timer.elapsed["train"],
                          test_time_sec=timer.elapsed["test"],
                          total_time_sec=timer.elapsed["train"]+timer.elapsed["test"],
                          evaluation_sec=timer.elapsed["evaluation"],
                          timing_scope=TIMING_PROTOCOL, timing_protocol=TIMING_PROTOCOL,
                          diagnostic_stage="train")
            return result
        finally:
            _active.reset(token)
    return wrapped
