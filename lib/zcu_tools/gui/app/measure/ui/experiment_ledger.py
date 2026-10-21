"""Measure's existing foldable analysis ledger and its matching forms/actions."""

from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .analyze_form import AnalyzeFormWidget

# Approved prototype blue primary treatment (A3)
BLUE_PRIMARY_STYLESHEET = (
    "QPushButton#primaryButton { background-color: #286ac7; color: white; "
    "font-weight: 600; border: 1px solid #205aa9; border-radius: 4px; }"
    "QPushButton#primaryButton:disabled { background-color: #a0b8d9; color: #e6edf7; border-color: #8da6c9; }"
    "QPushButton#primaryButton:hover:!disabled { background-color: #2f76dc; }"
)


class LedgerSection(QWidget):
    """App-local single-column ledger section with whole-row folding.

    Header row (blank or text area) toggles the body; the action bar stays
    fixed outside the scroll area. Presentation is app-local; the shared tree
    adapter remains the only cross-module structural seam (S1).
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        collapsed: bool = False,
    ) -> None:
        """Create a title header and body_layout, optionally collapsed.

        parent takes normal Qt ownership. Header clicks toggle the body only;
        callers add their controls to body_layout and retain their action policy.
        """
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header — whole row is clickable (S1/A2).
        self._header = QWidget()
        # tolerated: QtPy promotes the binding enum members onto Qt at runtime.
        self._header.setCursor(Qt.PointingHandCursor)  # pyright: ignore[reportAttributeAccessIssue]
        self._header.setObjectName("ledgerHeader")
        header_row = QHBoxLayout(self._header)
        header_row.setContentsMargins(4, 4, 4, 4)
        header_row.setSpacing(6)

        self._toggle_btn = QPushButton("▼" if not collapsed else "▶")
        self._toggle_btn.setFixedWidth(16)
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(not collapsed)
        header_row.addWidget(self._toggle_btn)

        self._title_label = QLabel(f"<b>{title}</b>")
        font = self._title_label.font()
        font.setPixelSize(13)
        self._title_label.setFont(font)
        header_row.addWidget(self._title_label, stretch=1)

        # Body
        self._body = QWidget()
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(8, 2, 0, 2)
        self.body_layout.setSpacing(2)

        outer.addWidget(self._header)
        outer.addWidget(self._body)

        self._collapsed = collapsed
        self._body.setVisible(not collapsed)

        # Whole header row toggles; button also toggles (no double toggle because
        # click on button goes to button widget, not header).
        self._toggle_btn.clicked.connect(self._toggle)
        # Install header click handler via event filter on header.
        self._header.mouseReleaseEvent = self._on_header_mouse_release

    def _on_header_mouse_release(self, a0: QMouseEvent | None) -> None:
        if a0 is None:
            QWidget.mouseReleaseEvent(self._header, a0)
            return
        try:
            # tolerated: QtPy promotes the binding enum members onto Qt at runtime.
            if a0.button() == Qt.LeftButton:  # pyright: ignore[reportAttributeAccessIssue]
                self._toggle()
                a0.accept()
                return
        # tolerated: Event accessors/presentation may fail arbitrarily; delegate default mouse handling.
        except Exception:  # noqa: BLE001
            pass
        # Fallback to default
        QWidget.mouseReleaseEvent(self._header, a0)

    def _toggle(self, *_: object) -> None:
        self._collapsed = not self._collapsed
        self._body.setVisible(not self._collapsed)
        self._toggle_btn.setText("▶" if self._collapsed else "▼")
        self._toggle_btn.setChecked(not self._collapsed)

    def set_collapsed(self, *, collapsed: bool) -> None:
        """Show or hide the body and update the header, without changing its controls."""
        if bool(collapsed) == self._collapsed:
            return
        self._toggle()

    def is_collapsed(self) -> bool:
        """Return whether the header currently hides the section's body."""
        return self._collapsed


def ledger_column() -> tuple[QWidget, QVBoxLayout]:
    """Return a new pane and its top-aligned scrolled column for caller-owned controls.

    The caller reparents the pane into its tab; the pane owns the scroll and layout.
    """
    container = QWidget()
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    inner = QWidget()
    layout = QVBoxLayout(inner)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(4)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    scroll.setWidget(inner)
    outer.addWidget(scroll, stretch=1)
    return container, layout


def ledger_form() -> AnalyzeFormWidget:
    """Return a new unpopulated 13 px parameter form for the caller to parent."""
    form = AnalyzeFormWidget()
    font = form.font()
    font.setPixelSize(13)
    form.setFont(font)
    return form


def primary_action_button(text: str) -> QPushButton:
    """Return an unbound blue 30 px button with text for the caller to parent/connect."""
    button = QPushButton(text)
    button.setObjectName("primaryButton")
    button.setFixedHeight(30)
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    button.setStyleSheet(BLUE_PRIMARY_STYLESHEET)
    return button
