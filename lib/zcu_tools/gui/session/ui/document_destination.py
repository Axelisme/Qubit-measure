"""Destination selection shared by the three Inspect pages."""

from qtpy.QtWidgets import QComboBox


class DocumentDestination(QComboBox):
    """List Global and the focused point, retaining selection within that focus.

    refresh(point) follows a changed focus and selects it initially. Global is
    always listed first and is the only choice when point is None. point()
    returns the selected point label, or None for Global. Neither method reads
    documents or changes the session focus.
    """

    def __init__(self) -> None:
        super().__init__()
        self._focus: str | None = None
        self.addItem("Global")

    def refresh(self, point: str | None) -> None:
        """Update available destinations after a session focus change."""
        if self._focus == point:
            return
        self._focus = point
        blocked = self.blockSignals(True)
        try:
            self.clear()
            self.addItem("Global")
            if point is not None:
                self.addItem(point)
                self.setCurrentIndex(1)
        finally:
            self.blockSignals(blocked)

    def point(self) -> str | None:
        """Return None for Global or the selected focused point label."""
        return None if self.currentIndex() == 0 else self._focus
