"""Transient diagnostic presentation owned by the measure main window."""

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.widgets import DialogPresenter, error_colour, set_foreground
from zcu_tools.gui.widgets.selectable_text import make_label_selectable


class DiagnosticNotice(QFrame):
    """Show the latest diagnostic summary without blocking other GUI controls.

    The parent main window owns this view and must call dispose before deleting
    its children. All methods run on the Qt owner thread.
    """

    def __init__(self, parent: QWidget, *, dialog_presenter: DialogPresenter) -> None:
        """Create a hidden notice under parent, using dialog_presenter for details.

        Details open as a report owned by parent, so dismissing this summary
        does not dismiss the full content the user is inspecting.
        """
        super().__init__(parent)
        self.setObjectName("diagnostic.notice")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._scope = LifetimeScope(self)
        self._report_parent = parent
        self._dialog_presenter = dialog_presenter
        self._message = ""
        layout = QVBoxLayout(self)
        heading = QHBoxLayout()
        layout.addLayout(heading)
        self._title = QLabel(self)
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        self._summary = QLabel(self)
        for label in (self._title, self._summary):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            make_label_selectable(label)
        heading.addWidget(self._title, stretch=1)
        details = QPushButton("Details…", self)
        self._scope.connect(details.clicked, self._show_details)
        heading.addWidget(details)
        layout.addWidget(self._summary)
        set_foreground(self._title, error_colour(self))
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._scope.add_cleanup(self._timer.stop)
        self._scope.connect(self._timer.timeout, lambda: self.hide())
        self.hide()

    def present(self, title: str, message: str) -> None:
        """Display title and message's first line for eight seconds.

        A later diagnostic replaces the summary and restarts its display time.
        Details opens the complete message through the injected presenter;
        full content is also retained by the caller's diagnostic log.
        """
        self._message = message
        self._title.setText(title)
        self._summary.setText(message.partition("\n")[0])
        self.show()
        self._timer.start(8000)

    def _show_details(self) -> None:
        self._dialog_presenter.critical(
            self._report_parent, self._title.text(), self._message
        )

    def dispose(self) -> None:
        """Stop the timer and retire callbacks before the parent deletes children."""
        self._scope.close()
