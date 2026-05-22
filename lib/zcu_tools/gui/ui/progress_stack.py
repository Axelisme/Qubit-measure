"""Reusable compact progress bar stack widget."""

from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class ProgressStack(QWidget):
    """Compact progress bar panel that only occupies space for active bars.

    Bars are added to the layout on push() and removed on pop()/reset_all(),
    so the widget has zero height when idle and grows only as bars are pushed.
    The anti-jitter strategy: bars are reused from a pool so Qt does not
    repeatedly allocate/free widgets; only the layout insertion/removal happens.
    """

    MAX_LAYERS = 4

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._pool: list[QProgressBar] = [
            QProgressBar() for _ in range(self.MAX_LAYERS)
        ]
        self._active: list[QProgressBar] = []

    def push(self, label: str = "", total: int = 0) -> QProgressBar:
        if self._pool:
            bar = self._pool.pop()
        else:
            bar = self._active[-1]  # reuse innermost when all slots busy
            return bar
        bar.setFormat(f"{label} %v/%m" if label else "%v/%m")
        bar.setMaximum(total)
        bar.setValue(0)
        self._layout.insertWidget(0, bar)
        self._active.append(bar)
        return bar

    def pop(self, bar: QProgressBar) -> None:
        if bar in self._active:
            self._active.remove(bar)
            self._layout.removeWidget(bar)
            bar.setParent(None)  # type: ignore[call-overload]
            bar.setValue(0)
            bar.setFormat("%v/%m")
            self._pool.append(bar)

    def reset_all(self) -> None:
        """Remove all active bars (called when a run ends)."""
        for bar in list(self._active):
            self._layout.removeWidget(bar)
            bar.setParent(None)  # type: ignore[call-overload]
            bar.setValue(0)
            bar.setFormat("%v/%m")
        self._pool.extend(self._active)
        self._active.clear()
