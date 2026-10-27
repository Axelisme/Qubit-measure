"""EvalNumericField — numeric input widget with right-click direct/eval toggle.

Standalone; no LiveField / LiveModel dependency. Designed for the device dialog
where eval expressions are resolved once at apply time against the current ParameterContext
(Design 1 in ADR for device-dialog eval). The ghost label is informational only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# tolerated: qtpy exposes Signal at runtime but its QtCore stub does not export it.
from qtpy.QtCore import Qt, Signal  # pyright: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import (
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
from zcu_tools.gui.widgets import error_colour, secondary_colour, set_foreground
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext


class EvalNumericField(QWidget):
    """Standalone numeric input with right-click direct↔eval toggle.

    direct mode: TrimDoubleSpinBox (configurable range/decimals).
    eval mode: QLineEdit (captured parameter paths) + ghost '= <resolved>' label (live preview).

    Ghost is informational only; authoritative resolve happens at dialog apply time via
    the EvalRef marker returned by read_raw(). The parameter_provider callback is used solely
    to supply a fixed capture (or None for pure calculation) for the ghost preview
    and for carrying the resolved value back when switching from
    eval to direct mode.

    R3 compliance: load_direct() in eval mode updates the stored backing value without
    clobbering the user's expression, so 1-second poll repaints do not reset the field.
    raw_changed emits when the direct value, expression text, or input mode changes.
    """

    raw_changed = Signal()

    def __init__(
        self,
        *,
        minimum: float,
        maximum: float,
        decimals: int,
        parameter_provider: Callable[[], ParameterContext | None],
        type_: type = float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum
        self._decimals = decimals
        self._parameter_provider = parameter_provider
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
        self._spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        layout.addWidget(self._spin, stretch=1)
        self._spin.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._spin.customContextMenuRequested.connect(
            lambda pos: self._show_context_menu(self._spin, self._spin.mapToGlobal(pos))
        )

        # Eval-mode widgets (hidden until mode switch)
        self._line_edit = QLineEdit()
        self._line_edit.setVisible(False)
        layout.addWidget(self._line_edit, stretch=1)
        self._ghost = QLabel()
        font = self._ghost.font()
        font.setItalic(True)
        self._ghost.setFont(font)
        make_label_selectable(self._ghost)
        self._ghost.setVisible(False)
        layout.addWidget(self._ghost)
        self._line_edit.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._line_edit.customContextMenuRequested.connect(
            lambda pos: self._show_context_menu(
                self._line_edit, self._line_edit.mapToGlobal(pos)
            )
        )
        self._spin.valueChanged.connect(lambda _value: self.raw_changed.emit())
        self._line_edit.textChanged.connect(self._sync_ghost)
        self._line_edit.textChanged.connect(lambda _text: self.raw_changed.emit())

    def set_input_name(self, prefix: str) -> None:
        """Name the direct and expression inputs for accessible component selection.

        The line edits receive prefix_direct and prefix_expression respectively.
        Hidden-mode inputs keep their identity when the user switches modes.
        """
        self._line_edit.setObjectName(f"{prefix}_expression")
        line_edit = self._spin.lineEdit()
        if line_edit is not None:
            line_edit.setObjectName(f"{prefix}_direct")

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
        self.raw_changed.emit()

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
        self.raw_changed.emit()

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
        self.raw_changed.emit()

    def _switch_to_direct(self) -> None:
        # Best-effort: carry the live-resolved ghost value into the spinbox.
        # Falls back to the stored direct value if resolution fails (no context,
        # invalid expression, etc.) — never surfaces an error here.
        try:
            parameters = self._parameter_provider()
            resolved = coerce_eval_result(
                evaluate_numeric_expr(self._line_edit.text(), parameters), self._type
            )
            self._direct_value = float(resolved)
        # tolerated: Any capture/source failure must retain the stored direct value when leaving eval mode.
        except Exception:  # noqa: BLE001
            pass
        self._mode = "direct"
        self._spin.setValue(self._direct_value)
        self._spin.setVisible(True)
        self._line_edit.setVisible(False)
        self._ghost.setVisible(False)
        self.raw_changed.emit()

    def _sync_ghost(self) -> None:
        """Update the ghost label with a live resolve attempt against parameter_provider.

        Ghost format mirrors ScalarWidget: '= <value>' (gray italic) on success,
        '= ?' (red italic) with error tooltip on failure. Ghost is purely informational.
        """
        if self._mode != "eval":
            return
        expr = self._line_edit.text()
        try:
            parameters = self._parameter_provider()
            resolved = coerce_eval_result(
                evaluate_numeric_expr(expr, parameters), self._type
            )
            # Float trailing-zero stripping: mirrors ScalarWidget ghost formatting.
            if self._type is float:
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
                set_foreground(self._ghost, error_colour(self))
            else:
                self._ghost.setText(text)
                self._ghost.setToolTip("")
                set_foreground(self._ghost, secondary_colour(self))
        # tolerated: Any capture/source failure becomes a preview diagnostic instead of escaping a Qt callback.
        except Exception as exc:  # noqa: BLE001
            self._ghost.setText("= ?")
            self._ghost.setToolTip(str(exc))
            set_foreground(self._ghost, error_colour(self))

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
            assert action is not None
            action.triggered.connect(self._switch_to_direct)
        else:
            action = menu.addAction("Use expression")
            assert action is not None
            action.triggered.connect(self._switch_to_eval)
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
        menu.exec_(global_pos)
