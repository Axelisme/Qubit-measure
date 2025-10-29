import time
from functools import wraps
from typing import Callable, Optional

from typing_extensions import ParamSpec

P = ParamSpec("P")


def min_interval(
    func: Optional[Callable[P, None]], min_interval: float = 0.1
) -> Optional[Callable[P, None]]:
    """ensures min_interval seconds between function calls"""
    if func is None:
        return None

    last_time = 0.0

    @wraps(func)
    def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> None:
        nonlocal last_time
        cur_time = time.time()

        # check if the function can be called
        if cur_time >= last_time + min_interval:
            func(*args, **kwargs)
            last_time = cur_time

    return wrapped_func
