from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.progress_bar.base import BaseProgressBar, ProgressTotal, ProgressValue

if TYPE_CHECKING:
    from zcu_tools.gui.ui.progress_stack import ProgressStack

_FLOAT_SCALE = 10000
_PUBLISH_INTERVAL = 0.033  # ~30 fps max for cross-thread Qt signal updates


@dataclass(frozen=True)
class ProgressEntrySnapshot:
    token: int
    format: str
    maximum: int
    value: int


def _qt_maximum(total: ProgressTotal) -> int:
    if total is None or total == 0:
        return 0
    if isinstance(total, int):
        return total
    return _FLOAT_SCALE


def _qt_value(value: ProgressValue, total: ProgressTotal) -> int:
    if isinstance(total, int):
        return int(round(value))
    if isinstance(total, float) and total > 0:
        return int(round(float(value) / total * _FLOAT_SCALE))
    return 0


def _fmt_seconds(secs: float) -> str:
    minutes, seconds = divmod(int(max(0.0, secs)), 60)
    return f"{minutes}:{seconds:02d}"


class ProgressModel(QObject):
    """Thread-safe Qt bridge: worker threads emit signals; main thread updates entries.

    Optionally connected to a ProgressStack widget via attach_stack() — when
    attached, the stack renders itself on every ``changed`` emit.
    """

    changed: Signal = Signal()
    _push_requested: Signal = Signal(int, str, object, object)
    _update_requested: Signal = Signal(int, str, object, object)
    _pop_requested: Signal = Signal(int)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._entries: dict[int, ProgressEntrySnapshot] = {}
        self._push_requested.connect(self._on_update)
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
            self._stack.render_snapshot(self.snapshot())

    def snapshot(self) -> tuple[ProgressEntrySnapshot, ...]:
        return tuple(self._entries.values())

    def clear(self) -> None:
        if self._entries:
            self._entries.clear()
            self.changed.emit()

    def _on_update(
        self, token: int, fmt: str, total: ProgressTotal, value: ProgressValue
    ) -> None:
        self._entries[token] = ProgressEntrySnapshot(
            token=token,
            format=fmt,
            maximum=_qt_maximum(total),
            value=_qt_value(value, total),
        )
        self.changed.emit()

    def _on_pop(self, token: int) -> None:
        if self._entries.pop(token, None) is not None:
            self.changed.emit()


# Backward-compat alias used by device.py and tests
DeviceSetupProgressModel = ProgressModel


class ProgressBar(BaseProgressBar):
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
        self._start_time = time.monotonic()
        self._last_publish: float = 0.0
        if not disabled:
            self._publish(initial=True)

    def _build_format(self) -> str:
        elapsed = time.monotonic() - self._start_time
        time_part = f"[{_fmt_seconds(elapsed)}]"
        if self._total is not None and self._total > 0 and self._n > 0:
            rate = float(self._n) / elapsed if elapsed > 0 else 0.0
            remaining = (float(self._total) - float(self._n)) / rate if rate > 0 else 0
            time_part = f"[{_fmt_seconds(elapsed)}<{_fmt_seconds(remaining)}]"
        prefix = f"{self._label} " if self._label else ""
        if isinstance(self._total, int):
            return f"{prefix}%v/%m {time_part}"
        return f"{prefix}{time_part}"

    def _publish(self, *, initial: bool = False, force: bool = False) -> None:
        now = time.monotonic()
        if not initial and not force and now - self._last_publish < _PUBLISH_INTERVAL:
            return
        self._last_publish = now
        signal = (
            self._model._push_requested if initial else self._model._update_requested
        )
        signal.emit(self._token, self._build_format(), self._total, self._n)

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
        self._start_time = time.monotonic()
        if not self._disabled:
            self._publish(force=True)

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
            self._publish(force=True)

    @property
    def n(self) -> ProgressValue:
        return self._n

    @property
    def desc(self) -> str:
        return self._label


# Backward-compat alias
DeviceSetupProgressBar = ProgressBar


class ProgressFactory:
    def __init__(self, model: ProgressModel) -> None:
        self._model = model
        self._tokens = itertools.count()

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


# Backward-compat alias
DeviceSetupProgressFactory = ProgressFactory
