import time
from contextlib import contextmanager
from typing import Callable, ClassVar, Generator, Optional, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")


class MinIntervalFunc:
    FORCE_EXECUTE: ClassVar[bool] = False

    def __init__(self, func: Callable[P, None], min_interval: float) -> None:
        self.func = func
        self.min_interval = min_interval

        if not 0.0 < self.min_interval <= 1.0:
            raise ValueError("min_interval must be between 0.0 and 1.0")

        self.last_call_time = 0.0
        self.last_exec_end = 0.0

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        cur_call_time = time.time()
        last_exec_period = self.last_exec_end - self.last_call_time
        cur_call_period = cur_call_time - self.last_call_time

        if (
            self.FORCE_EXECUTE
            or last_exec_period <= self.min_interval * cur_call_period
        ):
            self.func(*args, **kwargs)

            self.last_call_time = cur_call_time
            self.last_exec_end = time.time()

    @classmethod
    @contextmanager
    def force_execute(cls) -> Generator[None, None, None]:
        cls.FORCE_EXECUTE = True
        yield
        cls.FORCE_EXECUTE = False


T_Func = TypeVar("T_Func", bound=Optional[Callable[..., None]])


def min_interval(func: T_Func, min_interval: Optional[float] = None) -> T_Func:
    """ensures min_interval time ratio between function calls"""
    if func is None or min_interval is None or min_interval == 1.0:
        return func

    return MinIntervalFunc(func, min_interval)
