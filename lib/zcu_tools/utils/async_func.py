import asyncio
import contextvars
import sys
import threading
import time
from copy import deepcopy
from functools import wraps
from typing import Callable, Dict, Generic, Optional, Tuple

from typing_extensions import ParamSpec

from zcu_tools.utils.debug import print_traceback

"""
AsyncFunc
=========
Wrap a **synchronous** function that returns ``None`` so that:

1. Calls from the main-thread **return immediately** – execution is
   delegated to a background thread that runs a single, global
   ``asyncio`` event-loop shared by **all** ``AsyncFunc`` instances.
2. For each ``AsyncFunc`` instance, *only the most-recent call* is ever
   executed (subsequent calls overwrite the pending one).
3. Leaving the ``with`` block:
   • clears any pending, not-yet-executed job from *this* instance;
   • waits for a running job of *this* instance to finish, then returns.
4. The background thread is created lazily on the *first* instance and
   marked as *daemon* – it won’t block interpreter shutdown.  Remaining
   jobs are silently ignored at process exit.

Typical usage
-------------

>>> def task(x):
>>>     print("working", x)
>>>     time.sleep(0.2)
>>>
>>> with AsyncFunc(task) as atask:
>>>     for i in range(5):
>>>         atask(i)   # returns instantly
>>>         time.sleep(0.05)
>>> # on exiting the context the last task is allowed to complete
"""

# -------------------------------------------------------------
# Global background event-loop (singleton)
# -------------------------------------------------------------
_LOOP_READY = threading.Event()
_BG_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _run_loop() -> None:
    """Target for the background daemon thread: create & run loop."""
    global _BG_LOOP
    _BG_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_BG_LOOP)
    _LOOP_READY.set()
    _BG_LOOP.run_forever()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background loop; create it on first use."""
    if _BG_LOOP is None:
        t = threading.Thread(target=_run_loop, daemon=True, name="AsyncFuncLoop")
        t.start()
        _LOOP_READY.wait()
    return _BG_LOOP  # type: ignore[return-value]


# -------------------------------------------------------------
# AsyncFunc implementation
# -------------------------------------------------------------
P = ParamSpec("P")


class AsyncFunc(Generic[P]):
    """See module-level docstring for behaviour details."""

    def __init__(
        self, func: Optional[Callable[P, None]], min_interval: float = 0.0, deepcopy: bool = False
    ) -> None:
        """
        Args:
            func: The function to wrap.
            min_interval: The minimum interval between function calls in seconds.
        """
        if func is not None and not callable(func):
            raise TypeError("func must be callable or None")
        self.func = func
        self.min_interval = min_interval
        self.deepcopy = deepcopy

        # Runtime state
        self._last_job: Optional[Tuple[contextvars.Context, Tuple, Dict]] = None
        self._have_new_job: Optional[asyncio.Event] = None
        self._closed: bool = False
        self._worker_done = threading.Event()

    # -----------------------------------------------------
    # Context-manager protocol
    # -----------------------------------------------------
    def __enter__(self) -> Optional[Callable[P, None]]:
        # Allow the same instance to be entered multiple times (non-nested).
        # If the wrapped function is `None`, we simply behave as a no-op
        # wrapper and return early.
        if self.func is None:
            return None

        loop = _get_loop()

        # Reset state in case this instance was used before.
        # `_closed` marks that the *previous* context was exited; clear it so
        # the new context becomes active again.
        self._closed = False
        self._last_job = None
        self._worker_done = threading.Event()

        # for python >=3.10, no loop argument is needed
        if sys.version_info >= (3, 10):
            self._have_new_job = asyncio.Event()
        else:
            self._have_new_job = asyncio.Event(loop=loop)

        # Register a dedicated worker coroutine for *this* context instance.
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(
                self._worker(), name=f"AsyncFuncWorker-{id(self)}"
            )
        )

        @wraps(self.func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:  # type: ignore[override]
            """Main-thread call: save latest args & wake worker; return immediately."""
            if self._closed:
                return

            ctx = contextvars.copy_context()
            if self.deepcopy:
                self._last_job = (ctx, deepcopy(args), deepcopy(kwargs))
            else:
                self._last_job = (ctx, args, kwargs)
            # Must set the asyncio.Event from the correct thread
            loop.call_soon_threadsafe(self._have_new_job.set)  # type: ignore[arg-type]

        return wrapper

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # If wrapper is a no-op or the context has already been closed, do
        # nothing. The extra guard ensures that nested exit calls do not break
        # repeated usage.
        if self.func is None or self._closed:
            return
        self._closed = True
        # Discard pending job and wake the worker so it can terminate
        self._last_job = None
        _get_loop().call_soon_threadsafe(self._have_new_job.set)  # type: ignore[arg-type]
        # Wait until the worker signals completion (incl. running job)
        self._worker_done.wait()

    # -----------------------------------------------------
    # Worker coroutine (runs inside the shared loop)
    # -----------------------------------------------------
    async def _worker(self) -> None:
        assert self.func is not None
        assert self._have_new_job is not None

        # this ensure the first call is executed immediately
        last_time = time.time() - 2 * self.min_interval

        try:
            while True:
                await self._have_new_job.wait()

                # If context exited and nothing left to do => quit
                if self._closed or self._last_job is None:
                    break

                time_left = self.min_interval - (time.time() - last_time)
                if time_left > 0:
                    await asyncio.sleep(time_left)
                last_time = time.time()

                job = self._last_job
                self._last_job = None
                self._have_new_job.clear()

                if job is None:
                    break # may update to None in __exit__

                ctx, args, kwargs = job
                try:
                    # execute function in the main thread, block the loop
                    ctx.run(self.func, *args, **kwargs)
                except Exception as e:
                    print(f"Error in async func: {e}")
                    print_traceback()
        finally:
            self._worker_done.set()
