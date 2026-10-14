"""Controlled executor for app-owner and UI completion contracts."""

from collections.abc import Callable


class ControlledBackground:
    """Capture executor callbacks; deliver an outcome without running the kernel."""

    def __init__(self) -> None:
        self.work: Callable[[], object] | None = None
        self.done: Callable[[object], None] | None = None
        self.error: Callable[[Exception], None] | None = None
        self.fail_submit = False

    def submit(
        self,
        work: Callable[[], object],
        *,
        run_in_pool: bool,
        on_done: Callable[[object], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if self.fail_submit:
            raise RuntimeError("submit failed")
        self.work, self.done, self.error = work, on_done, on_error

    def deliver(self, result: object) -> None:
        assert self.done is not None
        self.done(result)

    def fail(self, exc: Exception) -> None:
        assert self.error is not None
        self.error(exc)
