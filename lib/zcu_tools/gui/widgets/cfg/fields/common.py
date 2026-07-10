"""Common widgets for shared cfg binding fields."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

logger = logging.getLogger(__name__)

from qtpy.QtCore import QSize, Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QDoubleValidator, QIntValidator  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from zcu_tools.gui.cfg import (
    DirectValue,
    EvalValue,
    ScalarSpec,
    default_value_for_type,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    CfgField,
    LiteralField,
    ScalarField,
    SweepField,
)
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox

from ..decoration import FieldDecorationProtocol
from ..registry import TextInputEnhancer
from ._decoration import (
    apply_decoration,
    decorated_label_text,
    decoration_enabled,
)

FIELD_INPUT_MIN_WIDTH = 20
FIELD_LABEL_MAX_WIDTH = 80


class ElidedLabel(QLabel):
    """QLabel that elides text when it exceeds the configured label width.

    The full text is always shown in the tooltip so the user can read the
    complete field name on hover.
    """

    def __init__(
        self,
        text: str,
        parent: QWidget | None = None,
        *,
        max_width: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setMaximumWidth(FIELD_LABEL_MAX_WIDTH if max_width is None else max_width)
        self.setToolTip(text)
        self._update_elided()

    def _update_elided(self) -> None:
        fm = self.fontMetrics()
        elided = fm.elidedText(
            self._full_text,
            Qt.ElideRight,  # type: ignore[attr-defined]
            self.maximumWidth(),
        )
        super().setText(elided)

    def resizeEvent(self, event: Any) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_elided()


def make_value_widget(
    type_: type,
    default: Any,
    choices: list | None,
    editable: bool = True,
    decimals: int | None = None,
    optional: bool = False,
) -> QWidget:
    """Build an input widget from raw field attributes."""
    if optional:
        # An optional scalar may be empty (= None), which a spinbox cannot show.
        # Render a QLineEdit: empty text = None, a numeric validator keeps input
        # well-formed. choices/bool optionals are not supported (fast-fail).
        if choices or type_ is bool:
            raise RuntimeError(
                "optional ScalarSpec does not support choices/bool widgets"
            )
        w = QLineEdit("" if default in (None, "") else str(default))
        w.setPlaceholderText("(none)")
        if type_ is int:
            w.setValidator(QIntValidator())
        elif type_ is float:
            w.setValidator(QDoubleValidator())
        w.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
        w.setEnabled(editable)
        return w
    if choices is not None:
        w = QComboBox()
        w.addItems([str(c) for c in choices])
        idx = w.findText(str(default))
        if idx >= 0:
            w.setCurrentIndex(idx)
        else:
            w.setCurrentIndex(-1)
            if hasattr(w, "setPlaceholderText"):
                w.setPlaceholderText("Select...")
        w.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
        w.setEnabled(editable)
        return w
    if type_ is bool:
        w = QCheckBox()
        w.setChecked(bool(default))
        w.setEnabled(editable)
        return w
    if type_ is int:
        w = QSpinBox()
        w.setRange(-(2**31), 2**31 - 1)
        w.setValue(int(default))
        w.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        w.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
        w.setEnabled(editable)
        return w
    if type_ is float:
        w = TrimDoubleSpinBox()
        w.setRange(-1e12, 1e12)
        w.setDecimals(decimals if decimals is not None else 6)
        w.setValue(float(default))
        w.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        w.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
        w.setEnabled(editable)
        return w
    w = QLineEdit(str(default))
    w.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
    w.setEnabled(editable)
    return w


def read_value_widget(w: QWidget, type_: type, fallback: Any = None) -> Any:
    """Read the current value from a widget created by make_value_widget."""
    if isinstance(w, QComboBox):
        if w.currentIndex() < 0:
            return fallback
        txt = w.currentText()
        return type_(txt) if type_ is not str else txt
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    if isinstance(w, TrimDoubleSpinBox):
        return w.value()
    if isinstance(w, QLineEdit):
        return type_(w.text())
    return fallback


def make_scalar_widget(spec: ScalarSpec, value: Any) -> QWidget:
    """Build an input widget from a ScalarSpec and initial value."""
    return make_value_widget(
        spec.type, value, spec.choices, spec.editable, spec.decimals, spec.optional
    )


def _sweep_cell(label: QLabel, widget: QWidget) -> QWidget:
    cell = QWidget()
    cell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    row = QHBoxLayout(cell)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    row.addWidget(label)
    row.addWidget(widget, stretch=1)
    return cell


class _SweepPairRow(QWidget):
    """Two sweep cells whose outer widths are always split 50/50."""

    _SPACING = 4

    def __init__(self, left: QWidget, right: QWidget) -> None:
        super().__init__()
        self._left = left
        self._right = right
        self._left.setParent(self)
        self._right.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        left_hint = self._left.sizeHint()
        right_hint = self._right.sizeHint()
        return QSize(
            2 * max(left_hint.width(), right_hint.width()) + self._SPACING,
            max(left_hint.height(), right_hint.height()),
        )

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        left_hint = self._left.minimumSizeHint()
        right_hint = self._right.minimumSizeHint()
        return QSize(
            2 * max(left_hint.width(), right_hint.width()) + self._SPACING,
            max(left_hint.height(), right_hint.height()),
        )

    def resizeEvent(self, event: Any) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        available = max(0, self.width() - self._SPACING)
        left_width = available // 2
        right_width = available - left_width
        self._left.setGeometry(0, 0, left_width, self.height())
        self._right.setGeometry(
            left_width + self._SPACING, 0, right_width, self.height()
        )


def _sweep_pair(
    left_label: QLabel, left_widget: QWidget, right_label: QLabel, right_widget: QWidget
) -> QWidget:
    return _SweepPairRow(
        _sweep_cell(left_label, left_widget),
        _sweep_cell(right_label, right_widget),
    )


def _edge_decoration(
    path: str,
    edge: str,
    edge_field: CfgField,
    decoration_for_path: Callable[[str, Any], FieldDecorationProtocol] | None,
) -> FieldDecorationProtocol | None:
    if not path or decoration_for_path is None:
        return None
    return decoration_for_path(f"{path}.{edge}", edge_field)


def _dynamic_choices_for_scalar(field: ScalarField, current: Any) -> list | None:
    options = field.available_options()
    if options is None:
        return None
    choices = list(options)
    if current not in (None, "") and current not in choices:
        choices.insert(0, current)
    return choices


def read_scalar_widget(w: QWidget, spec: ScalarSpec) -> Any:
    """Read the current value from a widget created by make_scalar_widget."""
    if spec.optional and isinstance(w, QLineEdit):
        # Empty optional field = None; a partial/invalid entry also reads as None.
        txt = w.text().strip()
        if txt == "":
            return None
        try:
            return spec.type(txt)
        except (ValueError, TypeError):
            return None
    return read_value_widget(w, spec.type, fallback=None)


def _widget_default_for_direct_value(value: DirectValue, spec: ScalarSpec) -> Any:
    if value.value is None:
        # An optional unset scalar shows as an empty field (the "(none)" state),
        # not the type's zero default.
        if spec.optional:
            return ""
        default = default_value_for_type(spec.type)
        return "" if default is None else default
    return value.value


class BaseLiveWidget(QWidget):
    """Base class implementing FieldWidgetProtocol."""

    def __init__(self, field: CfgField, parent: QWidget | None = None):
        super().__init__(parent)
        self._field = field

    @property
    def field(self) -> CfgField:
        return self._field

    def teardown(self) -> None:
        pass

    def refresh_section(self, path: str) -> bool:
        del path
        return False


class LiteralWidget(QLineEdit):
    """Read-only display for fixed literal values when a view reveals them."""

    def __init__(self, field: LiteralField, parent: QWidget | None = None):
        super().__init__(parent)
        self._field = field
        self.setText(str(field.spec.value))
        self.setReadOnly(True)
        self.setFocusPolicy(Qt.NoFocus)  # type: ignore[attr-defined]
        self.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)

    @property
    def field(self) -> CfgField:
        return self._field

    def teardown(self) -> None:
        pass

    def refresh_section(self, path: str) -> bool:
        del path
        return False


class ScalarWidget(BaseLiveWidget):
    """Generic input widget for ScalarField."""

    def __init__(
        self,
        field: ScalarField,
        parent: QWidget | None = None,
        *,
        text_input_enhancer: TextInputEnhancer | None = None,
    ) -> None:
        super().__init__(field, parent)
        self._updating = False
        self._input: QWidget | None = None
        self._ghost: QLabel | None = None
        self._text_input_enhancer = text_input_enhancer
        self._input_enhancement: object | None = None
        self._mode = ""
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._rebuild_ui()
        field.on_change.connect(self._on_model_changed)

    def teardown(self) -> None:
        self._field.on_change.disconnect(self._on_model_changed)

    def _on_ui_changed(self, *_: Any) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            field = cast(ScalarField, self._field)
            inp = self._input
            assert inp is not None
            if isinstance(field.get_value(), EvalValue):
                assert isinstance(inp, QLineEdit)
                field.set_value(EvalValue(expr=inp.text().strip()))
                self._sync_eval_ghost(field.get_value())
            elif field.spec.optional and isinstance(inp, QLineEdit):
                # Optional direct input: empty = None (unset). A partial/invalid
                # entry (e.g. "-", "1e") is held until it parses — don't clobber.
                txt = inp.text().strip()
                if txt == "":
                    field.set_value(None)
                else:
                    try:
                        field.set_value(field.spec.type(txt))
                    except (ValueError, TypeError):
                        return
            else:
                val = read_value_widget(inp, field.spec.type)
                field.set_value(val)
        finally:
            self._updating = False

    def _on_model_changed(self, val: Any) -> None:
        next_mode = "eval" if isinstance(val, EvalValue) else "direct"
        if next_mode != self._mode:
            self._rebuild_ui()
            return
        if self._updating:
            return
        self._updating = True
        try:
            inp = self._input
            assert inp is not None
            if isinstance(val, EvalValue):
                assert isinstance(inp, QLineEdit)
                inp.setText(val.expr)
                self._sync_eval_ghost(val)
                return

            if not isinstance(val, DirectValue):
                return
            field = cast(ScalarField, self._field)
            raw = _widget_default_for_direct_value(val, field.spec)
            if isinstance(inp, QComboBox):
                choices = _dynamic_choices_for_scalar(field, raw) or []
                current_choices = [inp.itemText(i) for i in range(inp.count())]
                if current_choices != [str(choice) for choice in choices]:
                    self._rebuild_ui()
                    return
                idx = inp.findText(str(raw))
                if idx >= 0:
                    inp.setCurrentIndex(idx)
            elif isinstance(inp, QCheckBox):
                inp.setChecked(bool(raw))
            elif isinstance(inp, QSpinBox):
                inp.setValue(int(raw))
            elif isinstance(inp, TrimDoubleSpinBox):
                inp.setValue(float(raw))
            elif isinstance(inp, QLineEdit):
                inp.setText(str(raw))
        finally:
            self._updating = False

    def _rebuild_ui(self) -> None:
        self._clear_layout()
        self._input_enhancement = None
        field = cast(ScalarField, self._field)
        value = field.get_value()
        self._mode = "eval" if isinstance(value, EvalValue) else "direct"
        self._ghost = None

        if isinstance(value, EvalValue):
            inp = QLineEdit(value.expr)
            self._input = inp
            inp.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
            inp.setEnabled(field.spec.editable)
            inp.textChanged.connect(self._on_ui_changed)
            if self._text_input_enhancer is not None:
                self._input_enhancement = self._text_input_enhancer(inp)
            self._layout.addWidget(inp, stretch=1)

            self._ghost = QLabel()
            self._layout.addWidget(self._ghost)
            self._sync_eval_ghost(value)
        else:
            raw = _widget_default_for_direct_value(value, field.spec)
            self._input = make_value_widget(
                field.spec.type,
                raw,
                _dynamic_choices_for_scalar(field, raw),
                field.spec.editable,
                field.spec.decimals,
                field.spec.optional,
            )
            self._layout.addWidget(self._input, stretch=1)
            self._connect_direct_input()

        self._install_context_menu(self._input)

    def _clear_layout(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _connect_direct_input(self) -> None:
        inp = self._input
        assert inp is not None
        if isinstance(inp, QComboBox):
            inp.currentIndexChanged.connect(self._on_ui_changed)
        elif isinstance(inp, QCheckBox):
            inp.toggled.connect(self._on_ui_changed)
        elif isinstance(inp, (QSpinBox, TrimDoubleSpinBox)):
            inp.valueChanged.connect(self._on_ui_changed)
        elif isinstance(inp, QLineEdit):
            inp.textChanged.connect(self._on_ui_changed)

    def _sync_eval_ghost(self, value: object) -> None:
        if self._ghost is None or not isinstance(value, EvalValue):
            return
        if value.resolved is None:
            self._ghost.setText("= ?")
            self._ghost.setToolTip(value.error or "Expression is unresolved")
            self._ghost.setStyleSheet("color: red; font-style: italic;")
            return
        spec = cast(ScalarField, self._field).spec
        if spec.type is float and isinstance(value.resolved, (int, float)):
            decimals = spec.decimals if spec.decimals is not None else 6
            raw = f"{value.resolved:.{decimals}f}"
            if "." in raw:
                raw = raw.rstrip("0")
                if raw.endswith("."):
                    raw += "0"
            text = f"= {raw}"
        else:
            text = f"= {value.resolved}"
        self._ghost.setText(text)
        self._ghost.setToolTip("")
        self._ghost.setStyleSheet("color: gray; font-style: italic;")

    def _install_context_menu(self, widget: QWidget | None) -> None:
        if widget is None:
            return
        if not isinstance(widget, (QAbstractSpinBox, QLineEdit)):
            return
        widget.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        widget.customContextMenuRequested.connect(  # type: ignore[attr-defined]
            lambda pos, w=widget: self._show_context_menu(w, w.mapToGlobal(pos))
        )

    def _show_context_menu(
        self, widget: QAbstractSpinBox | QLineEdit, global_pos: Any
    ) -> None:
        if isinstance(widget, QAbstractSpinBox):
            line_edit = widget.lineEdit()
            if not isinstance(line_edit, QLineEdit):
                return
        else:
            line_edit = widget
        menu, mode_action = self._build_context_menu(line_edit)
        if mode_action is None:
            return
        field = cast(ScalarField, self._field)
        chosen = cast(Any, menu).exec_(global_pos)
        if chosen is not mode_action:
            return
        value = field.get_value()
        if isinstance(value, EvalValue):
            if value.resolved is None:
                field.set_value(None)
            else:
                field.set_value(DirectValue(value=value.resolved))
            return

        if isinstance(value, DirectValue):
            expr = "" if value.value is None else str(value.value)
            field.set_value(EvalValue(expr=expr))

    def _build_context_menu(self, widget: QLineEdit) -> tuple[QMenu, Any]:
        menu = widget.createStandardContextMenu()
        if menu is None:
            raise RuntimeError("QLineEdit.createStandardContextMenu() returned None")
        if not self._supports_eval_mode():
            return menu, None
        if menu.actions():
            menu.addSeparator()
        value = cast(ScalarField, self._field).get_value()
        if isinstance(value, EvalValue):
            return menu, menu.addAction("Use direct value")
        return menu, menu.addAction("Use expression")

    def _supports_eval_mode(self) -> bool:
        field = cast(ScalarField, self._field)
        spec = field.spec
        return (
            spec.editable
            and field.available_options() is None
            and spec.type in {int, float}
        )


class SweepWidget(BaseLiveWidget):
    """Inline 2x2 input for start/stop/expts/step with synchronized updates."""

    def __init__(
        self,
        field: SweepField,
        parent: QWidget | None = None,
        *,
        path: str = "",
        decoration_for_path: Callable[[str, Any], FieldDecorationProtocol]
        | None = None,
        text_input_enhancer: TextInputEnhancer | None = None,
    ) -> None:
        super().__init__(field, parent)
        self._updating = False

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        sv = field.get_value()

        decimals = field.spec.decimals

        self._start_widget = ScalarWidget(
            field.start_field,
            self,
            text_input_enhancer=text_input_enhancer,
        )
        self._stop_widget = ScalarWidget(
            field.stop_field,
            self,
            text_input_enhancer=text_input_enhancer,
        )

        self._expts = QSpinBox()
        self._expts.setRange(1, 2**31 - 1)
        self._expts.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._expts.setValue(sv.expts)
        self._expts.valueChanged.connect(self._on_expts_changed)

        self._step = TrimDoubleSpinBox()
        self._step.setRange(-1e12, 1e12)
        self._step.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._step.setDecimals(decimals if decimals is not None else 6)
        self._step.setValue(sv.step)
        self._step.valueChanged.connect(self._on_step_changed)

        enabled = field.spec.editable
        start_decoration = _edge_decoration(
            path, "start", field.start_field, decoration_for_path
        )
        stop_decoration = _edge_decoration(
            path, "stop", field.stop_field, decoration_for_path
        )
        self._start_widget.setEnabled(enabled and decoration_enabled(start_decoration))
        self._stop_widget.setEnabled(enabled and decoration_enabled(stop_decoration))
        self._expts.setEnabled(enabled)
        self._step.setEnabled(enabled)

        start_label = QLabel(decorated_label_text("start", start_decoration))
        stop_label = QLabel(decorated_label_text("stop", stop_decoration))
        apply_decoration(start_label, self._start_widget, start_decoration)
        apply_decoration(stop_label, self._stop_widget, stop_decoration)

        layout.addWidget(
            _sweep_pair(start_label, self._start_widget, stop_label, self._stop_widget),
            0,
            0,
        )
        layout.addWidget(
            _sweep_pair(QLabel("expts"), self._expts, QLabel("step"), self._step),
            1,
            0,
        )

        field.on_change.connect(self._on_model_changed)

    def teardown(self) -> None:
        field = cast(SweepField, self._field)
        field.on_change.disconnect(self._on_model_changed)
        self._start_widget.teardown()
        self._stop_widget.teardown()

    def _on_expts_changed(self, expts: int) -> None:
        if self._updating:
            return
        cast(SweepField, self._field).update_expts(expts)

    def _on_step_changed(self, step: float) -> None:
        if self._updating:
            return
        cast(SweepField, self._field).update_step(step)

    def _on_model_changed(self, val: Any) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            if not (
                self._expts.minimum() <= val.expts <= self._expts.maximum()
                and self._step.minimum() <= val.step <= self._step.maximum()
            ):
                raise RuntimeError("SweepValue is outside widget range")
            self._expts.setValue(val.expts)
            self._step.setValue(val.step)
        finally:
            self._updating = False


class CenteredSweepWidget(BaseLiveWidget):
    """Inline 2x2 input for center/span/expts/step with synchronized updates."""

    def __init__(
        self,
        field: CenteredSweepField,
        parent: QWidget | None = None,
        *,
        text_input_enhancer: TextInputEnhancer | None = None,
    ) -> None:
        super().__init__(field, parent)
        self._updating = False

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        sv = field.get_value()
        decimals = field.spec.decimals

        self._center_widget = ScalarWidget(
            field.center_field,
            self,
            text_input_enhancer=text_input_enhancer,
        )

        self._span = TrimDoubleSpinBox()
        self._span.setRange(0.0, 1e12)
        self._span.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._span.setDecimals(decimals if decimals is not None else 6)
        self._span.setValue(sv.span)
        self._span.valueChanged.connect(self._on_span_changed)

        self._expts = QSpinBox()
        self._expts.setRange(1, 2**31 - 1)
        self._expts.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._expts.setValue(sv.expts)
        self._expts.valueChanged.connect(self._on_expts_changed)

        self._step = TrimDoubleSpinBox()
        self._step.setRange(0.0, 1e12)
        self._step.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._step.setDecimals(decimals if decimals is not None else 6)
        self._step.setValue(sv.step)
        self._step.valueChanged.connect(self._on_step_changed)

        enabled = field.spec.editable
        self._center_widget.setEnabled(enabled and field.spec.center_editable)
        self._span.setEnabled(enabled)
        self._expts.setEnabled(enabled)
        self._step.setEnabled(enabled)

        center_label = QLabel(_centered_sweep_label("center", field.spec.center_badge))
        center_tooltip = field.spec.center_tooltip or field.spec.tooltip
        if center_tooltip:
            center_label.setToolTip(center_tooltip)
            self._center_widget.setToolTip(center_tooltip)

        layout.addWidget(
            _sweep_pair(center_label, self._center_widget, QLabel("span"), self._span),
            0,
            0,
        )
        layout.addWidget(
            _sweep_pair(QLabel("expts"), self._expts, QLabel("step"), self._step),
            1,
            0,
        )

        field.on_change.connect(self._on_model_changed)

    def teardown(self) -> None:
        field = cast(CenteredSweepField, self._field)
        field.on_change.disconnect(self._on_model_changed)
        self._center_widget.teardown()

    def _on_span_changed(self, span: float) -> None:
        if self._updating:
            return
        self._try_update(
            lambda: cast(CenteredSweepField, self._field).update_span(span)
        )

    def _on_expts_changed(self, expts: int) -> None:
        if self._updating:
            return
        self._try_update(
            lambda: cast(CenteredSweepField, self._field).update_expts(expts)
        )

    def _on_step_changed(self, step: float) -> None:
        if self._updating:
            return
        self._try_update(
            lambda: cast(CenteredSweepField, self._field).update_step(step)
        )

    def _try_update(self, update: Callable[[], None]) -> None:
        try:
            update()
        except ValueError:
            self._on_model_changed(cast(CenteredSweepField, self._field).get_value())

    def _on_model_changed(self, val: Any) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            if not (
                self._span.minimum() <= val.span <= self._span.maximum()
                and self._expts.minimum() <= val.expts <= self._expts.maximum()
                and self._step.minimum() <= val.step <= self._step.maximum()
            ):
                raise RuntimeError("CenteredSweepValue is outside widget range")
            self._span.setValue(val.span)
            self._expts.setValue(val.expts)
            self._step.setValue(val.step)
        finally:
            self._updating = False


def _centered_sweep_label(text: str, badge: str) -> str:
    return f"{text} [{badge}]" if badge else text
