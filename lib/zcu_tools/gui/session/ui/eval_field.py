"""EvalNumericField — numeric input widget with right-click direct/eval toggle.

Standalone; no LiveField / LiveModel dependency. Designed for the device dialog
where eval expressions are resolved once at apply time against the current MetaDict
(Design 1 in ADR for device-dialog eval). The ghost label is informational only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from zcu_tools.gui.session.expression import (
    EvalRef,
    coerce_eval_result,
    evaluate_numeric_expr,
)
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict


class EvalNumericField(QWidget):
    """Standalone numeric input with right-click direct↔eval toggle.

    direct mode: TrimDoubleSpinBox (configurable range/decimals).
    eval mode: QLineEdit (MetaDict expression) + ghost '= <resolved>' label (live preview).

    Ghost is informational only; authoritative resolve happens at dialog apply time via
    the EvalRef marker returned by read_raw(). The md_provider callback is used solely
    for the ghost preview and for carrying the resolved value back when switching from
    eval to direct mode.

    R3 compliance: load_direct() in eval mode updates the stored backing value without
    clobbering the user's expression, so 1-second poll repaints do not reset the field.
    """

    def __init__(
        self,
        *,
        minimum: float,
        maximum: float,
        decimals: int,
        md_provider: Callable[[], MetaDict],
        type_: type = float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum
        self._decimals = decimals
        self._md_provider = md_provider
        self._type = type_
        self._mode: str = "direct"  # "direct" | "eval"
        # Backing store for the last known direct value. Updated by load_direct()
        # regardless of mode, and consulted when switching eval→direct.
        self._direct_value: float = 0.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._layout = layout

        # Direct-mode spinbox
        self._spin = TrimDoubleSpinBox()
        self._spin.setRange(minimum, maximum)
        self._spin.setDecimals(decimals)
        self._spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)  # type: ignore[attr-defined]
        layout.addWidget(self._spin, stretch=1)
        self._spin.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)  # type: ignore[attr-defined]
        self._spin.customContextMenuRequested.connect(  # type: ignore[attr-defined]
            lambda pos: self._show_context_menu(self._spin, self._spin.mapToGlobal(pos))
        )

        # Eval-mode widgets (hidden until mode switch)
        self._line_edit = QLineEdit()
        self._line_edit.setVisible(False)
        layout.addWidget(self._line_edit, stretch=1)
        self._ghost = QLabel()
        self._ghost.setVisible(False)
        layout.addWidget(self._ghost)
        self._line_edit.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)  # type: ignore[attr-defined]
        self._line_edit.customContextMenuRequested.connect(  # type: ignore[attr-defined]
            lambda pos: self._show_context_menu(
                self._line_edit, self._line_edit.mapToGlobal(pos)
            )
        )
        self._line_edit.textChanged.connect(self._sync_ghost)  # type: ignore[attr-defined]

    def load_direct(self, value: float) -> None:
        """Load a direct float value.

        In direct mode: updates the spinbox immediately.
        In eval mode: stores the backing value without touching the expression
        (R3 guard — prevents 1-second poll repaints from clobbering an expression
        the user is actively composing).
        """
        self._direct_value = value
        if self._mode == "direct":
            self._spin.setValue(value)

    def reset_to_direct(self) -> None:
        """Switch unconditionally to direct mode with the stored backing value.

        Called when a different device is loaded into this panel to prevent an
        expression from one device persisting into a different device's context.
        """
        self._mode = "direct"
        self._spin.setValue(self._direct_value)
        self._spin.setVisible(True)
        self._line_edit.setVisible(False)
        self._ghost.setVisible(False)

    def load_expression(
        self, expr: str, *, direct_fallback: float | None = None
    ) -> None:
        """Switch to eval mode with ``expr`` without requiring a context-menu action."""
        if direct_fallback is not None:
            self._direct_value = float(direct_fallback)
        elif self._mode == "direct":
            self._direct_value = self._spin.value()
        self._mode = "eval"
        self._line_edit.setText(expr)
        self._spin.setVisible(False)
        self._line_edit.setVisible(True)
        self._ghost.setVisible(True)
        self._sync_ghost()

    def read_raw(self) -> float | EvalRef:
        """Return the current value.

        direct mode: float (spinbox value).
        eval mode: EvalRef(expr, type_) — the dialog resolves this at apply time.
        """
        if self._mode == "eval":
            return EvalRef(
                expr=self._line_edit.text(),
                type_=self._type,
                minimum=self._minimum,
                maximum=self._maximum,
            )
        return self._spin.value()

    # ------------------------------------------------------------------
    # Internal mode switching
    # ------------------------------------------------------------------

    def _switch_to_eval(self) -> None:
        self._direct_value = self._spin.value()
        self._mode = "eval"
        # Pre-fill the expression with the current direct value string so the user
        # has a starting point and can see the numeric value before editing.
        self._line_edit.setText(str(self._direct_value))
        self._spin.setVisible(False)
        self._line_edit.setVisible(True)
        self._ghost.setVisible(True)
        self._sync_ghost()

    def _switch_to_direct(self) -> None:
        # Best-effort: carry the live-resolved ghost value into the spinbox.
        # Falls back to the stored direct value if resolution fails (no context,
        # invalid expression, etc.) — never surfaces an error here.
        try:
            md = self._md_provider()
            resolved = coerce_eval_result(
                evaluate_numeric_expr(self._line_edit.text(), md), self._type
            )
            if isinstance(resolved, (int, float)):
                self._direct_value = float(resolved)
        except Exception:
            pass
        self._mode = "direct"
        self._spin.setValue(self._direct_value)
        self._spin.setVisible(True)
        self._line_edit.setVisible(False)
        self._ghost.setVisible(False)

    def _sync_ghost(self) -> None:
        """Update the ghost label with a live resolve attempt against md_provider.

        Ghost format mirrors ScalarWidget: '= <value>' (gray italic) on success,
        '= ?' (red italic) with error tooltip on failure. Ghost is purely informational.
        """
        if self._mode != "eval":
            return
        expr = self._line_edit.text()
        try:
            md = self._md_provider()
            resolved = coerce_eval_result(evaluate_numeric_expr(expr, md), self._type)
            # Float trailing-zero stripping: mirrors ScalarWidget ghost formatting.
            if self._type is float and isinstance(resolved, (int, float)):
                raw = f"{resolved:.{self._decimals}f}"
                if "." in raw:
                    raw = raw.rstrip("0")
                    if raw.endswith("."):
                        raw += "0"
                text = f"= {raw}"
            else:
                text = f"= {resolved}"
            # Range check mirrors the inclusive bounds of the direct spinbox.
            if not (self._minimum <= float(resolved) <= self._maximum):
                self._ghost.setText(text)
                self._ghost.setToolTip(
                    f"out of range [{self._minimum}, {self._maximum}]"
                )
                self._ghost.setStyleSheet("color: red; font-style: italic;")
            else:
                self._ghost.setText(text)
                self._ghost.setToolTip("")
                self._ghost.setStyleSheet("color: gray; font-style: italic;")
        except Exception as exc:
            self._ghost.setText("= ?")
            self._ghost.setToolTip(str(exc))
            self._ghost.setStyleSheet("color: red; font-style: italic;")

    # ------------------------------------------------------------------
    # Context menu (right-click toggle)
    # ------------------------------------------------------------------

    def _build_context_menu(self, line_edit: QLineEdit) -> Any:
        """Build the standard context menu with a mode-toggle action appended.

        The toggle action's ``triggered`` signal is connected to the appropriate
        switch callback so callers (production: exec_; tests: action.trigger())
        both work without modal blocking.

        Returns the QMenu (with the action already wired), or None if
        createStandardContextMenu returned None.
        """
        menu = line_edit.createStandardContextMenu()
        if menu is None:
            return None
        if menu.actions():
            menu.addSeparator()
        if self._mode == "eval":
            action = menu.addAction("Use direct value")
            action.triggered.connect(self._switch_to_direct)  # type: ignore[attr-defined]
        else:
            action = menu.addAction("Use expression")
            action.triggered.connect(self._switch_to_eval)  # type: ignore[attr-defined]
        return menu

    def _show_context_menu(self, widget: QWidget, global_pos: Any) -> None:
        # Get the underlying QLineEdit for context menu creation (required for
        # spinbox: QAbstractSpinBox embeds a QLineEdit that owns createStandardContextMenu).
        if isinstance(widget, QDoubleSpinBox):
            line_edit = widget.lineEdit()
            if not isinstance(line_edit, QLineEdit):
                return
        elif isinstance(widget, QLineEdit):
            line_edit = widget
        else:
            return

        menu = self._build_context_menu(line_edit)
        if menu is None:
            return
        cast(Any, menu).exec_(global_pos)
