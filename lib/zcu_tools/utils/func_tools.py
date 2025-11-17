import time
from functools import wraps
from typing import Callable, Optional

from typing_extensions import ParamSpec

P = ParamSpec("P")


def min_interval(
    func: Optional[Callable[P, None]], min_interval: float = 0.1
) -> Optional[Callable[P, None]]:
    """ensures min_interval time ratio between function calls"""
    assert min_interval >= 0.0

    if func is None or min_interval >= 1.0:
        return func

    last_call_time = 0.0
    last_exec_end = 0.0

    @wraps(func)
    def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> None:
        nonlocal last_call_time, last_exec_end
        cur_call_time = time.time()

        last_exec_period = last_exec_end - last_call_time
        cur_call_period = cur_call_time - last_call_time

        # check if the function can be called
        if last_exec_period <= min_interval * cur_call_period:
            func(*args, **kwargs)

            last_call_time = cur_call_time
            last_exec_end = time.time()

    return wrapped_func
