"""Transient diagnostic presentation owned by the measure main window."""

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.widgets import error_colour, set_foreground
from zcu_tools.gui.widgets.selectable_text import make_label_selectable


class DiagnosticNotice(QFrame):
    """Show the latest diagnostic summary without blocking other GUI controls.

    The parent main window owns this view and must call dispose before deleting
    its children. All methods run on the Qt owner thread.
    """

    def __init__(self, parent: QWidget) -> None:
        """Create an initially hidden notice under parent."""
        super().__init__(parent)
        self.setObjectName("diagnostic.notice")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._scope = LifetimeScope(self)
        layout = QVBoxLayout(self)
        self._title = QLabel(self)
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        self._summary = QLabel(self)
        for label in (self._title, self._summary):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            make_label_selectable(label)
            layout.addWidget(label)
        set_foreground(self._title, error_colour(self))
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._scope.add_cleanup(self._timer.stop)
        self._scope.connect(self._timer.timeout, self.hide)
        self.hide()

    def present(self, title: str, message: str) -> None:
        """Display title and message's first line for eight seconds.

        A later diagnostic replaces the summary and restarts its display time.
        Full content is retained by the caller's diagnostic log.
        """
        self._title.setText(title)
        self._summary.setText(message.partition("\n")[0])
        self.show()
        self._timer.start(8000)

    def dispose(self) -> None:
        """Stop the timer and retire callbacks before the parent deletes children."""
        self._scope.close()
