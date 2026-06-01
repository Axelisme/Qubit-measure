"""Progress-bar host: main-thread bridge for worker-thread progress bars.

Sibling of ``plot_host.py`` — both are main-thread hosts that receive
cross-thread signals from worker QThreads and own a class of GUI resource (here,
progress bars; there, matplotlib figures). Nothing here is device-specific; it
backs both run progress and device-setup progress.

Roles:
  - ``ProgressBar`` (worker side): a ``BaseProgressBar`` the experiment code
    drives via the ``make_pbar`` factory. It is **pure transport + throttle** —
    it forwards raw (token, label, total, n) over Qt signals and computes
    nothing (no format, no timing).
  - ``ProgressModel`` (main thread): receives those signals on the main thread
    and owns ``dict[token, ProgressBarModel]``. Optionally drives a
    ``ProgressStack`` widget via ``attach_stack``.
  - ``ProgressBarModel`` (main thread, the SSOT): one mutable object per live
    bar holding raw state (label / total / n / start_time). Derived values
    (format string, percent, elapsed/remaining) are **methods computed live on
    query** — every reader (widget render, agent ``run.progress``) asks this
    object, so there is a single source of truth and no stale formatted string.
"""

from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.progress_bar.base import BaseProgressBar, ProgressTotal, ProgressValue

if TYPE_CHECKING:
    from zcu_tools.gui.ui.progress_stack import ProgressStack

# Float totals are rendered on an integer Qt bar by scaling to this resolution.
_FLOAT_SCALE = 10000
_PUBLISH_INTERVAL = 0.033  # ~30 fps cap for cross-thread Qt signal updates


def _fmt_seconds(secs: float) -> str:
    minutes, seconds = divmod(int(max(0.0, secs)), 60)
    return f"{minutes}:{seconds:02d}"


class ProgressBarModel:
    """Main-thread SSOT for one live progress bar.

    Holds raw state; format / percent / timing are computed live on each query
    (so elapsed/remaining reflect wall-clock at read time, not a frozen string).
    ``start_time`` is stamped by the main thread when the bar is first pushed.
    """

    def __init__(self, label: str, total: ProgressTotal, start_time: float) -> None:
        self.label = label
        self.total = total
        self.n: ProgressValue = 0
        self.start_time = start_time

    def set_n(self, n: ProgressValue) -> None:
        self.n = n

    def set_label(self, label: str) -> None:
        self.label = label

    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.start_time)

    def remaining(self) -> Optional[float]:
        total, n, elapsed = self.total, self.n, self.elapsed()
        if total is None or total <= 0 or n <= 0 or elapsed <= 0:
            return None
        rate = float(n) / elapsed
        return (float(total) - float(n)) / rate if rate > 0 else None

    def qt_maximum(self) -> int:
        if self.total is None or self.total == 0:
            return 0
        if isinstance(self.total, int):
            return self.total
        return _FLOAT_SCALE

    def qt_value(self) -> int:
        total = self.total
        if isinstance(total, int):
            return int(round(self.n))
        if isinstance(total, float) and total > 0:
            return int(round(float(self.n) / total * _FLOAT_SCALE))
        return 0

    def percent(self) -> Optional[float]:
        maximum = self.qt_maximum()
        if maximum == 0:
            return None
        return round(self.qt_value() / maximum * 100, 1)

    def format(self) -> str:
        """Human-readable bar string, e.g. ``Rounds 23/100 [0:25<1:15]``."""
        elapsed = self.elapsed()
        remaining = self.remaining()
        if remaining is not None:
            time_part = f"[{_fmt_seconds(elapsed)}<{_fmt_seconds(remaining)}]"
        else:
            time_part = f"[{_fmt_seconds(elapsed)}]"
        prefix = f"{self.label} " if self.label else ""
        if isinstance(self.total, int):
            return f"{prefix}%v/%m {time_part}"
        return f"{prefix}{time_part}"


class ProgressModel(QObject):
    """Thread-safe Qt bridge: worker threads emit signals; the main thread owns
    a ``ProgressBarModel`` per token and updates it.

    Optionally connected to a ProgressStack widget via ``attach_stack()`` — when
    attached, the stack re-renders on every ``changed`` emit.
    """

    changed: Signal = Signal()
    # token, label, total  (push = create the bar; main thread stamps start_time)
    _push_requested: Signal = Signal(int, str, object)
    # token, label, n      (update = advance an existing bar)
    _update_requested: Signal = Signal(int, str, object)
    _pop_requested: Signal = Signal(int)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._entries: dict[int, ProgressBarModel] = {}
        self._push_requested.connect(self._on_push)
        self._update_requested.connect(self._on_update)
        self._pop_requested.connect(self._on_pop)
        self._stack: Optional[ProgressStack] = None

    def attach_stack(self, stack: ProgressStack) -> None:
        """Connect this model to a ProgressStack widget (main thread only)."""
        if self._stack is stack:
            return
        if self._stack is not None:
            self.changed.disconnect(self._render_stack)
        self._stack = stack
        self.changed.connect(self._render_stack)

    def detach_stack(self) -> None:
        """Disconnect the attached ProgressStack (main thread only)."""
        if self._stack is not None:
            self.changed.disconnect(self._render_stack)
            self._stack = None

    def _render_stack(self) -> None:
        if self._stack is not None:
            self._stack.render_models(self.models())

    def models(self) -> tuple[ProgressBarModel, ...]:
        """Live bar models (main-thread read). Readers call their methods."""
        return tuple(self._entries.values())

    def model_items(self) -> tuple[tuple[int, ProgressBarModel], ...]:
        """Live (token, model) pairs — for readers that need the token (e.g. the
        wire projection keys each bar by token)."""
        return tuple(self._entries.items())

    def clear(self) -> None:
        if self._entries:
            self._entries.clear()
            self.changed.emit()

    def _on_push(self, token: int, label: str, total: ProgressTotal) -> None:
        # The main thread stamps start_time the moment the bar is created, so
        # elapsed reflects the bar's real lifetime regardless of update timing.
        self._entries[token] = ProgressBarModel(label, total, time.monotonic())
        self.changed.emit()

    def _on_update(self, token: int, label: str, n: ProgressValue) -> None:
        entry = self._entries.get(token)
        if entry is None:
            # Update before push (e.g. a throttled-away initial): create it now.
            entry = ProgressBarModel(label, None, time.monotonic())
            self._entries[token] = entry
        entry.set_label(label)
        entry.set_n(n)
        self.changed.emit()

    def _on_pop(self, token: int) -> None:
        if self._entries.pop(token, None) is not None:
            self.changed.emit()


class ProgressBar(BaseProgressBar):
    """Worker-side bar: pure raw-forward + throttle, no format/timing here."""

    def __init__(
        self,
        model: ProgressModel,
        token: int,
        label: str = "",
        total: ProgressTotal = None,
        leave: bool = True,
        disabled: bool = False,
    ) -> None:
        self._model = model
        self._token = token
        self._label = label
        self._total = total
        self._leave = leave
        self._disabled = disabled
        self._n: ProgressValue = 0
        self._last_publish: float = 0.0
        if not disabled:
            # Push creates the bar (and stamps start_time) on the main thread.
            self._model._push_requested.emit(self._token, self._label, self._total)

    def _publish(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_publish < _PUBLISH_INTERVAL:
            return
        self._last_publish = now
        self._model._update_requested.emit(self._token, self._label, self._n)

    def set_description(self, description: str) -> None:
        self._label = description
        if not self._disabled:
            self._publish(force=True)

    def update(self, value: ProgressValue = 1) -> None:
        self._n += value
        if not self._disabled:
            self._publish()

    def reset(self) -> None:
        self._n = 0
        if not self._disabled:
            # Re-push so the main thread re-stamps start_time.
            self._model._push_requested.emit(self._token, self._label, self._total)

    def refresh(self) -> None:
        if not self._disabled:
            self._publish(force=True)

    def close(self) -> None:
        if not self._disabled and not self._leave:
            self._model._pop_requested.emit(self._token)

    @property
    def total(self) -> ProgressTotal:
        return self._total

    @total.setter
    def total(self, value: ProgressTotal) -> None:
        self._total = value
        if not self._disabled:
            # total change must re-create the bar entry (push carries total).
            self._model._push_requested.emit(self._token, self._label, self._total)

    @property
    def n(self) -> ProgressValue:
        return self._n

    @property
    def desc(self) -> str:
        return self._label


class ProgressFactory:
    def __init__(self, model: ProgressModel) -> None:
        self._model = model
        self._tokens = itertools.count()

    def live_models(self) -> tuple[tuple[int, ProgressBarModel], ...]:
        """Live (token, model) pairs of this factory's model (the SSOT)."""
        return self._model.model_items()

    def __call__(self, *args: Any, **kwargs: Any) -> ProgressBar:
        label = kwargs.pop("desc", "") or (args[1] if len(args) > 1 else "")
        total = kwargs.pop("total", None) or (args[2] if len(args) > 2 else None)
        leave = kwargs.pop("leave", True)
        disabled = bool(kwargs.pop("disable", False))
        return ProgressBar(
            self._model,
            next(self._tokens),
            label=str(label) if label else "",
            total=total,
            leave=leave,
            disabled=disabled,
        )
