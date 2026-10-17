"""Per-evolve execution and centrally coordinated process-wide native limits."""

from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from threading import Lock
from typing import Any, TypeVar

from joblib import Parallel
from threadpoolctl import threadpool_limits

_T = TypeVar("_T")


class _Serial:
    def __call__(
        self,
        jobs: Iterable[tuple[Callable[..., _T], tuple[Any, ...], dict[str, Any]]],
    ) -> list[_T]:
        return [function(*args, **kwargs) for function, args, kwargs in jobs]


Executor = Parallel | _Serial


@contextmanager
def execution_pool(workers: int) -> Generator[Executor]:
    """Avoid creating joblib resources when the evolve cannot use parallelism."""
    if workers == 1:
        yield _Serial()
    else:
        with Parallel(n_jobs=workers, backend="threading") as pool:
            yield pool


class _NativeLimit:
    """First evolve sets BLAS=1; the last restores the caller's native state.

    Threadpoolctl modifies process-global state. Keep entry/exit transitions under
    one lock, but release it while evolves work so independent engines can overlap.
    This coordinates evolution calls, not external threadpoolctl users: callers
    must not independently change native limits while evolution is active.
    """

    def __init__(self) -> None:
        self.lock = Lock()
        self.active = 0
        self.limiter: threadpool_limits | None = None

    @contextmanager
    def acquire(self) -> Generator[None]:
        with self.lock:
            if self.active == 0:
                self.limiter = threadpool_limits(limits=1, user_api="blas")
            self.active += 1
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1
                if self.active == 0 and self.limiter is not None:
                    self.limiter.restore_original_limits()
                    self.limiter = None


native_limit = _NativeLimit()
