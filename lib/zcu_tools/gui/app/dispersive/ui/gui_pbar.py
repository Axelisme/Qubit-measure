"""GuiProgressBar — a BaseProgressBar that forwards progress to the Qt main thread.

Installed (via ``use_pbar_factory``) inside a worker thread so a notebook-style
progress loop (the preprocessing edelay fit, the auto-fit optimizer) feeds a Qt
progress bar. The bar is created on the worker thread, so it must NOT touch widgets
directly — it emits a Qt signal (queued connection) whose main-thread slot updates
the QProgressBar.

A ``GuiProgressBarChannel`` (a QObject on the main thread) owns the signal; the
factory it hands out builds bars bound to that channel. Updates are throttled to
avoid flooding the event loop on a fast inner loop. (Copied verbatim from fluxdep —
pure mechanism.)
"""

from __future__ import annotations

import time

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.progress_bar import BaseProgressBar
from zcu_tools.progress_bar.base import ProgressTotal, ProgressValue

_PUBLISH_INTERVAL = 0.05  # seconds — throttle worker→main updates


class GuiProgressBarChannel(QObject):
    """Main-thread owner of the progress signal; hands out a worker-side factory.

    ``progress`` carries ``(n, total, description)``; total is -1 when unknown
    (the Qt slot then shows a busy/indeterminate bar). Connect it with a queued
    connection so the worker thread's emit is delivered on the main thread.
    """

    progress = Signal(float, float, str)

    def factory(self):
        """Return a ``make_pbar``-compatible factory bound to this channel."""

        def _make(*_args, **kwargs) -> "GuiProgressBar":
            return GuiProgressBar(
                self,
                total=kwargs.get("total"),
                desc=str(kwargs.get("desc", "")),
            )

        return _make


class GuiProgressBar(BaseProgressBar):
    """Worker-side bar: forwards throttled (n, total, desc) to its channel."""

    def __init__(
        self,
        channel: GuiProgressBarChannel,
        total: ProgressTotal = None,
        desc: str = "",
    ) -> None:
        self._channel = channel
        self._total = total
        self._desc = desc
        self._n: ProgressValue = 0
        self._last_publish = 0.0
        self._publish(force=True)

    def _publish(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_publish < _PUBLISH_INTERVAL:
            return
        self._last_publish = now
        total = float(self._total) if self._total is not None else -1.0
        self._channel.progress.emit(float(self._n), total, self._desc)

    def update(self, value: ProgressValue = 1) -> None:
        self._n += value
        self._publish()

    def set_description(self, description: str) -> None:
        self._desc = description
        self._publish(force=True)

    def reset(self) -> None:
        self._n = 0
        self._publish(force=True)

    def refresh(self) -> None:
        self._publish(force=True)

    def close(self) -> None:
        self._publish(force=True)

    @property
    def total(self) -> ProgressTotal:
        return self._total

    @total.setter
    def total(self, value: ProgressTotal) -> None:
        self._total = value
        self._publish(force=True)

    @property
    def n(self) -> ProgressValue:
        return self._n

    @property
    def desc(self) -> str:
        return self._desc
