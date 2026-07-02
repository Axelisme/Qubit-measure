from __future__ import annotations

import time
from collections.abc import Callable
from typing import Generic, ParamSpec, overload

P = ParamSpec("P")


class MinIntervalFunc(Generic[P]):
    def __init__(self, func: Callable[P, None], min_interval: float) -> None:
        self.func = func
        self.duty_cycle_ratio = min_interval

        if not 0.0 < self.duty_cycle_ratio <= 1.0:
            raise ValueError(
                "min_interval duty-cycle ratio must be between 0.0 and 1.0"
            )

        self.last_call_time = 0.0
        self.last_exec_end = 0.0

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        cur_call_time = time.time()
        last_exec_period = self.last_exec_end - self.last_call_time
        cur_call_period = cur_call_time - self.last_call_time

        if last_exec_period <= self.duty_cycle_ratio * cur_call_period:
            self.func(*args, **kwargs)

            self.last_call_time = cur_call_time
            self.last_exec_end = time.time()


@overload
def min_interval(func: None, min_interval: float | None = None) -> None: ...


@overload
def min_interval(
    func: Callable[P, None], min_interval: float | None = None
) -> Callable[P, None]: ...


def min_interval(
    func: Callable[P, None] | None, min_interval: float | None = None
) -> Callable[P, None] | None:
    """Throttle a callback by duty-cycle ratio.

    ``min_interval`` is historical naming: the value is a ratio in ``(0, 1]``.
    A callback is allowed when the previous execution duration is no greater
    than ``min_interval * elapsed_since_previous_call``.
    """
    if func is None or min_interval is None or min_interval == 1.0:
        return func

    return MinIntervalFunc(func, min_interval)
