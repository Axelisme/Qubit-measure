"""Qt projection of point-file synchronization and explicit reload decisions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from qtpy.QtCore import QRect, Qt
from qtpy.QtGui import QResizeEvent
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import PointFileRevision


class PointFileBanner(QWidget):
    """Show one point document's pending/reloaded/error projection.

    control owns revision checks and reload validation. document selects the
    owner file; children are named 'inspect.<document>.sync.message',
    'inspect.<document>.sync.reload' and 'inspect.<document>.sync.keep'.
    on_reloaded discards the host's draft only after
    reload_file succeeds. Keep hides the notice, never changes a draft revision.
    Call refresh on source facts; failed checks can call show_conflict before
    displaying their usual validation error. Mutating Reload follows set_read_only.
    """

    def __init__(
        self,
        control: ContextControlPort,
        document: Literal["parameters", "library", "aliases"],
        on_reloaded: Callable[[], None],
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._control = control
        self._document = document
        self._on_reloaded = on_reloaded
        self._notice_file: PointFileRevision | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._message = QLabel()
        self._message.setObjectName(f"inspect.{document}.sync.message")
        self._message.setWordWrap(True)
        self._message.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._message)
        actions = QHBoxLayout()
        self._reload = QPushButton("Reload")
        self._reload.setObjectName(f"inspect.{document}.sync.reload")
        self._reload.clicked.connect(self._reload_file)
        keep = QPushButton("Keep")
        keep.setObjectName(f"inspect.{document}.sync.keep")
        keep.clicked.connect(self.hide)
        actions.addWidget(self._reload)
        actions.addWidget(keep)
        actions.addStretch()
        layout.addLayout(actions)
        self.hide()

    def capture(self) -> PointFileRevision | None:
        """Return the installed selected-point file, or None without a point."""
        return next(
            (
                f
                for f in self._control.get_file_revisions()
                if f.document == self._document
            ),
            None,
        )

    def check(self, expected: PointFileRevision | None) -> None:
        """Check a loaded draft's disk revision; show and rethrow owner rejection.

        None means no point was loaded, not a missing aliases revision. The
        normal writer still owns the no-context precondition in that case.
        """
        if expected is None:
            return
        try:
            self._control.check_file_revision(expected)
        except ExpectedError as error:
            self.show_conflict(str(error))
            raise

    def refresh(self, *, has_draft: bool) -> None:
        """Project pending/errors; successful reload conflicts require a draft."""
        file = self.capture()
        statuses = self._control.get_file_sync_status()
        for status in statuses:
            if file is not None and status.file.path == file.path:
                if status.state == "reloaded" and not has_draft:
                    self.hide()
                    return
                self.show_conflict(
                    status.error
                    or f"{file.path.name}: source changed ({status.state}). "
                    "Reload to discard the draft, or Keep to continue editing."
                )
                return
        if (
            file is None
            or self._notice_file is None
            or file.path != self._notice_file.path
        ):
            self.hide()

    def show_conflict(self, message: str) -> None:
        """Show a source/check failure without discarding or rebasing the draft."""
        self._notice_file = self.capture()
        self._message.setText(message)
        self.show()
        self._fit_message()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        """Reserve all wrapped message lines when the host changes width."""
        super().resizeEvent(a0)
        self._fit_message()

    def _fit_message(self) -> None:
        # A hidden notice can retain its previous one-line height. Reserve the
        # new wrapped height immediately, including the first error publication.
        bounds = self._message.fontMetrics().boundingRect(
            QRect(0, 0, self.width(), 0),
            Qt.TextFlag.TextWordWrap,
            self._message.text(),
        )
        self._message.setFixedHeight(bounds.height())
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            self.setFixedHeight(layout.sizeHint().height())
            layout.activate()
        parent = self.parentWidget()
        if parent is not None and (parent_layout := parent.layout()) is not None:
            parent_layout.activate()

    def set_read_only(self, *, read_only: bool) -> None:
        """Disable mutating Reload while keeping status and Keep available."""
        self._reload.setEnabled(not read_only)

    def _reload_file(self) -> None:
        file = self.capture()
        if file is None:
            return
        try:
            self._control.reload_file(file.path, file.revision)
        except (ExpectedError, OSError, ValueError) as error:
            self.show_conflict(str(error))
            return
        self._on_reloaded()
        self.hide()
