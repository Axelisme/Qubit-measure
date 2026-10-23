"""Shared page controls for live figure panes and Data raster groups."""

from qtpy.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget

from .figure_pane_layout import FigurePaneLayout


class FigurePager(QWidget):
    """Control a borrowed tiled stack on the Qt owner thread.

    pane is an object-name suffix such as run, analysis, post_analysis or
    data.run. The stack owns page selection; this widget hides for zero/one page
    and disables navigation at each end. It never attaches or owns Figures.
    """

    def __init__(self, stack: FigurePaneLayout, pane: str) -> None:
        super().__init__()
        self.setObjectName(f"figures.pager.{pane}")
        self._stack = stack
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        self._previous = QToolButton()
        self._previous.setText("Previous page")
        self._previous.setObjectName(f"{self.objectName()}.previous")
        self._previous.clicked.connect(lambda: stack.set_page(stack.current_page() - 1))
        layout.addWidget(self._previous)
        self._label = QLabel()
        self._label.setObjectName(f"{self.objectName()}.label")
        layout.addWidget(self._label)
        self._next = QToolButton()
        self._next.setText("Next page")
        self._next.setObjectName(f"{self.objectName()}.next")
        self._next.clicked.connect(lambda: stack.set_page(stack.current_page() + 1))
        layout.addWidget(self._next)
        layout.addStretch()
        stack.presentationChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        count = self._stack.page_count()
        current = self._stack.current_page()
        self.setVisible(count > 1 and self._stack.currentIndex() > 0)
        self._label.setText(f"{current + 1} / {count}")
        self._previous.setEnabled(current > 0)
        self._next.setEnabled(current + 1 < count)
