"""Common widgets for shared cfg binding fields."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

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
    CenteredSweepSpec,
    CenteredSweepValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
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
    choices: Sequence[object] | None,
    *,
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
    if isinstance(w, (QSpinBox, TrimDoubleSpinBox)):
        return w.value()
    if isinstance(w, QLineEdit):
        return type_(w.text())
    return fallback


def write_value_widget(widget: QWidget, value: object) -> None:
    """Write a raw value to a supported scalar input widget."""
    if isinstance(widget, QComboBox):
        index = widget.findText(str(value))
        if index >= 0:
            widget.setCurrentIndex(index)
        return
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
        return
    if isinstance(widget, QSpinBox):
        widget.setValue(int(cast(Any, value)))
        return
    if isinstance(widget, TrimDoubleSpinBox):
        widget.setValue(float(cast(Any, value)))
        return
    if isinstance(widget, QLineEdit):
        widget.setText("" if value is None else str(value))
        return
    raise TypeError(f"Unsupported value widget {type(widget).__name__}")


def connect_value_widget(widget: QWidget, callback: Callable[..., object]) -> None:
    """Connect the value-change signal of a supported scalar input widget."""
    if isinstance(widget, QComboBox):
        widget.currentIndexChanged.connect(callback)
        return
    if isinstance(widget, QCheckBox):
        widget.toggled.connect(callback)
        return
    if isinstance(widget, (QSpinBox, TrimDoubleSpinBox)):
        widget.valueChanged.connect(callback)
        return
    if isinstance(widget, QLineEdit):
        widget.textChanged.connect(callback)
        return
    raise TypeError(f"Unsupported value widget {type(widget).__name__}")


def connect_committed_value_widget(
    widget: QWidget, callback: Callable[..., object]
) -> None:
    """Connect committed text edits and immediate non-text value changes."""
    if isinstance(widget, QLineEdit):
        widget.editingFinished.connect(callback)
        return
    connect_value_widget(widget, callback)


def make_scalar_widget(spec: ScalarSpec, value: Any) -> QWidget:
    """Build an input widget from a ScalarSpec and initial value."""
    return make_value_widget(
        spec.type,
        value,
        spec.choices,
        editable=spec.editable,
        decimals=spec.decimals,
        optional=spec.optional,
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


def _scalar_choices(
    options: tuple[object, ...] | None, current: object
) -> list[object] | None:
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


def _make_range_input(name: str) -> QLineEdit:
    entry = QLineEdit()
    entry.setObjectName(name)
    entry.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
    return entry


def _render_range_input(
    entry: QLineEdit, label: QLabel, name: str, value: int | float | DirectValue
) -> None:
    if isinstance(value, DirectValue):
        entry.setText(_direct_input_text(value))
        resolved = "?" if value.value is None else str(value.value)
        label.setText(f"{name} = {resolved}" if value.raw is not None else name)
        entry.setToolTip(value.error or "")
    else:
        entry.setText(str(value))
        label.setText(name)
        entry.setToolTip("")


def _direct_input_text(value: DirectValue) -> str:
    if value.raw is not None:
        return value.raw
    return "" if value.value is None else str(value.value)


def _widget_default_for_direct_value(value: DirectValue, spec: ScalarSpec) -> Any:
    if (spec.optional or spec.type is complex) and value.raw is not None:
        return value.raw
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


class ScalarInputWidget(QWidget):
    """Render published scalar state and submit input without parsing it."""

    def __init__(
        self,
        spec: ScalarSpec,
        value: DirectValue | EvalValue,
        parent: QWidget | None = None,
        *,
        options: tuple[object, ...] | None,
        submit: Callable[[DirectValue | EvalValue], None],
        text_input_enhancer: TextInputEnhancer | None = None,
    ) -> None:
        super().__init__(parent)
        self._spec = spec
        self._value = value
        self._options = options
        self._submit = submit
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

    def _on_ui_changed(self, *_: Any) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            inp = self._input
            assert inp is not None
            if isinstance(self._value, EvalValue):
                assert isinstance(inp, QLineEdit)
                self._submit(EvalValue(expr=inp.text().strip()))
                self._sync_eval_ghost(self._value)
            elif isinstance(inp, QLineEdit) and (
                self._spec.optional or self._spec.type in (int, float, complex)
            ):
                self._submit(DirectValue(None, raw=inp.text()))
            elif isinstance(inp, QComboBox):
                choices = _scalar_choices(
                    self._options,
                    _widget_default_for_direct_value(self._value, self._spec),
                )
                assert choices is not None
                index = inp.currentIndex()
                self._submit(DirectValue(choices[index] if index >= 0 else None))
            else:
                val = read_value_widget(inp, self._spec.type)
                self._submit(DirectValue(val))
        finally:
            self._updating = False

    def display(
        self, val: DirectValue | EvalValue, *, options: tuple[object, ...] | None
    ) -> None:
        """Apply a publication without submitting another edit."""
        self._value = val
        self._options = options
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

            raw = _widget_default_for_direct_value(val, self._spec)
            if isinstance(inp, QComboBox):
                choices = _scalar_choices(self._options, raw) or []
                current_choices = [inp.itemText(i) for i in range(inp.count())]
                if current_choices != [str(choice) for choice in choices]:
                    self._rebuild_ui()
                    return
                idx = inp.findText(str(raw))
                if idx >= 0:
                    inp.setCurrentIndex(idx)
            elif isinstance(inp, QCheckBox):
                inp.setChecked(bool(raw))
            elif isinstance(inp, QLineEdit):
                inp.setText(_direct_input_text(val))
        finally:
            self._updating = False

    def _rebuild_ui(self) -> None:
        self._clear_layout()
        self._input_enhancement = None
        value = self._value
        self._mode = "eval" if isinstance(value, EvalValue) else "direct"
        self._ghost = None

        if isinstance(value, EvalValue):
            inp = QLineEdit(value.expr)
            self._input = inp
            inp.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
            inp.setEnabled(self._spec.editable)
            inp.textChanged.connect(self._on_ui_changed)
            if self._text_input_enhancer is not None:
                self._input_enhancement = self._text_input_enhancer(inp)
            self._layout.addWidget(inp, stretch=1)

            self._ghost = QLabel()
            self._layout.addWidget(self._ghost)
            self._sync_eval_ghost(value)
        else:
            raw = _widget_default_for_direct_value(value, self._spec)
            choices = _scalar_choices(self._options, raw)
            if self._spec.type in (int, float, complex) and choices is None:
                # Parsing and incomplete state belong to the cfg owner. Spinbox
                # validation would silently restore an earlier value on blur.
                inp = QLineEdit(_direct_input_text(value))
                inp.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
                inp.setEnabled(self._spec.editable)
                if self._spec.optional:
                    inp.setPlaceholderText("(none)")
                self._input = inp
            else:
                self._input = make_value_widget(
                    self._spec.type,
                    raw,
                    choices,
                    editable=self._spec.editable,
                    decimals=self._spec.decimals,
                    optional=self._spec.optional,
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
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _connect_direct_input(self) -> None:
        inp = self._input
        assert inp is not None
        connect_value_widget(inp, self._on_ui_changed)

    def _sync_eval_ghost(self, value: object) -> None:
        if self._ghost is None or not isinstance(value, EvalValue):
            return
        if value.resolved is None:
            self._ghost.setText("= ?")
            self._ghost.setToolTip(value.error or "Expression is unresolved")
            self._ghost.setStyleSheet("color: red; font-style: italic;")
            return
        spec = self._spec
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
        chosen = cast(Any, menu).exec_(global_pos)
        if chosen is not mode_action:
            return
        value = self._value
        if isinstance(value, EvalValue):
            if value.resolved is None:
                self._submit(DirectValue(None))
            else:
                self._submit(DirectValue(value=value.resolved))
            return

        expr = "" if value.value is None else str(value.value)
        self._submit(EvalValue(expr=expr))

    def _build_context_menu(self, widget: QLineEdit) -> tuple[QMenu, Any]:
        menu = widget.createStandardContextMenu()
        if menu is None:
            raise RuntimeError("QLineEdit.createStandardContextMenu() returned None")
        if not self._supports_eval_mode():
            return menu, None
        if menu.actions():
            menu.addSeparator()
        value = self._value
        if isinstance(value, EvalValue):
            return menu, menu.addAction("Use direct value")
        return menu, menu.addAction("Use expression")

    def _supports_eval_mode(self) -> bool:
        spec = self._spec
        return (
            spec.editable
            and self._options is None
            and spec.type in {int, float, complex}
        )


class ScalarWidget(ScalarInputWidget):
    """Connect the shared scalar control to existing binding-owned forms."""

    def __init__(
        self,
        field: ScalarField,
        parent: QWidget | None = None,
        *,
        text_input_enhancer: TextInputEnhancer | None = None,
    ) -> None:
        self._field = field
        super().__init__(
            field.spec,
            field.get_value(),
            parent,
            options=field.available_options(),
            submit=self._write_input,
            text_input_enhancer=text_input_enhancer,
        )
        field.on_change.connect(self._on_model_changed)

    @property
    def field(self) -> CfgField:
        return self._field

    def teardown(self) -> None:
        self._field.on_change.disconnect(self._on_model_changed)

    def refresh_section(self, path: str) -> bool:
        del path
        return False

    def _on_model_changed(self, value: DirectValue | EvalValue) -> None:
        self.display(value, options=self._field.available_options())

    def _write_input(self, value: DirectValue | EvalValue) -> None:
        _write_scalar_input(self._field, value)


def _range_scalar_value(
    value: float | DirectValue | EvalValue,
) -> DirectValue | EvalValue:
    return value if isinstance(value, (DirectValue, EvalValue)) else DirectValue(value)


def _write_scalar_input(field: ScalarField, value: DirectValue | EvalValue) -> None:
    if isinstance(value, DirectValue) and value.raw is not None:
        field.set_text(value.raw)
    else:
        field.set_value(value)


def _range_input_text(value: DirectValue | EvalValue) -> str:
    if not isinstance(value, DirectValue) or value.raw is None:
        raise TypeError("Range sampling input requires raw direct text")
    return value.raw


class SweepInputWidget(QWidget):
    """Inline 2x2 input for start/stop/points/step with synchronized updates."""

    def __init__(
        self,
        spec: SweepSpec,
        value: SweepValue,
        parent: QWidget | None = None,
        *,
        submit: Callable[
            [Literal["start", "stop", "expts", "step"], DirectValue | EvalValue], None
        ],
        edge_decorations: tuple[
            FieldDecorationProtocol | None, FieldDecorationProtocol | None
        ] = (None, None),
        text_input_enhancer: TextInputEnhancer | None = None,
    ) -> None:
        super().__init__(parent)
        self._submit = submit
        self._updating = False

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        edge_spec = ScalarSpec(
            spec.label, float, editable=spec.editable, decimals=spec.decimals
        )

        self._start_widget = ScalarInputWidget(
            edge_spec,
            _range_scalar_value(value.start),
            self,
            options=None,
            submit=lambda value: self._submit("start", value),
            text_input_enhancer=text_input_enhancer,
        )
        self._stop_widget = ScalarInputWidget(
            edge_spec,
            _range_scalar_value(value.stop),
            self,
            options=None,
            submit=lambda value: self._submit("stop", value),
            text_input_enhancer=text_input_enhancer,
        )

        self._expts = _make_range_input("expts")
        self._expts.textChanged.connect(self._on_expts_changed)
        self._points_label = QLabel("points")
        self._step = _make_range_input("step")
        self._step.textChanged.connect(self._on_step_changed)
        self._step_label = QLabel("step")

        enabled = spec.editable
        start_decoration, stop_decoration = edge_decorations
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
            _sweep_pair(self._points_label, self._expts, self._step_label, self._step),
            1,
            0,
        )

        self.display(value)

    def _on_expts_changed(self, text: str) -> None:
        if not self._updating:
            self._submit("expts", DirectValue(None, raw=text))

    def _on_step_changed(self, text: str) -> None:
        if not self._updating:
            self._submit("step", DirectValue(None, raw=text))

    def display(self, val: SweepValue) -> None:
        """Render the published range without normalizing or resubmitting it."""
        if self._updating:
            return
        self._updating = True
        try:
            self._start_widget.display(_range_scalar_value(val.start), options=None)
            self._stop_widget.display(_range_scalar_value(val.stop), options=None)
            _render_range_input(self._expts, self._points_label, "points", val.expts)
            _render_range_input(self._step, self._step_label, "step", val.step)
        finally:
            self._updating = False


class CenteredSweepInputWidget(QWidget):
    """Inline 2x2 input for center/span/points/step with synchronized updates."""

    def __init__(
        self,
        spec: CenteredSweepSpec,
        value: CenteredSweepValue,
        parent: QWidget | None = None,
        *,
        submit: Callable[
            [Literal["center", "span", "expts", "step"], DirectValue | EvalValue], None
        ],
        text_input_enhancer: TextInputEnhancer | None = None,
    ) -> None:
        super().__init__(parent)
        self._submit = submit
        self._updating = False

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        center_spec = ScalarSpec(
            spec.label,
            float,
            editable=spec.editable and spec.center_editable,
            decimals=spec.decimals,
        )

        self._center_widget = ScalarInputWidget(
            center_spec,
            _range_scalar_value(value.center),
            self,
            options=None,
            submit=lambda value: self._submit("center", value),
            text_input_enhancer=text_input_enhancer,
        )

        self._span = _make_range_input("span")
        self._span.textChanged.connect(self._on_span_changed)
        self._span_label = QLabel("span")
        self._expts = _make_range_input("expts")
        self._expts.textChanged.connect(self._on_expts_changed)
        self._points_label = QLabel("points")
        self._step = _make_range_input("step")
        self._step.textChanged.connect(self._on_step_changed)
        self._step_label = QLabel("step")

        enabled = spec.editable
        self._center_widget.setEnabled(enabled and spec.center_editable)
        self._span.setEnabled(enabled)
        self._expts.setEnabled(enabled)
        self._step.setEnabled(enabled)

        center_label = QLabel(_centered_sweep_label("center", spec.center_badge))
        center_tooltip = spec.center_tooltip or spec.tooltip
        if center_tooltip:
            center_label.setToolTip(center_tooltip)
            self._center_widget.setToolTip(center_tooltip)

        layout.addWidget(
            _sweep_pair(
                center_label, self._center_widget, self._span_label, self._span
            ),
            0,
            0,
        )
        layout.addWidget(
            _sweep_pair(self._points_label, self._expts, self._step_label, self._step),
            1,
            0,
        )

        self.display(value)

    def _on_span_changed(self, text: str) -> None:
        if not self._updating:
            self._submit("span", DirectValue(None, raw=text))

    def _on_expts_changed(self, text: str) -> None:
        if not self._updating:
            self._submit("expts", DirectValue(None, raw=text))

    def _on_step_changed(self, text: str) -> None:
        if not self._updating:
            self._submit("step", DirectValue(None, raw=text))

    def display(self, val: CenteredSweepValue) -> None:
        """Render the published range without normalizing or resubmitting it."""
        if self._updating:
            return
        self._updating = True
        try:
            self._center_widget.display(_range_scalar_value(val.center), options=None)
            _render_range_input(self._span, self._span_label, "span", val.span)
            _render_range_input(self._expts, self._points_label, "points", val.expts)
            _render_range_input(self._step, self._step_label, "step", val.step)
        finally:
            self._updating = False


class SweepWidget(SweepInputWidget):
    """Connect a caller-owned binding field to the range presentation."""

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
        self._field = field
        super().__init__(
            field.spec,
            field.get_value(),
            parent,
            submit=self._write_input,
            edge_decorations=(
                _edge_decoration(path, "start", field.start_field, decoration_for_path),
                _edge_decoration(path, "stop", field.stop_field, decoration_for_path),
            ),
            text_input_enhancer=text_input_enhancer,
        )
        field.on_change.connect(self.display)

    @property
    def field(self) -> CfgField:
        return self._field

    def teardown(self) -> None:
        self._field.on_change.disconnect(self.display)

    def refresh_section(self, path: str) -> bool:
        del path
        return False

    def _write_input(
        self,
        edge: Literal["start", "stop", "expts", "step"],
        value: DirectValue | EvalValue,
    ) -> None:
        if edge == "start":
            _write_scalar_input(self._field.start_field, value)
        elif edge == "stop":
            _write_scalar_input(self._field.stop_field, value)
        else:
            self._field.set_text(edge, _range_input_text(value))


class CenteredSweepWidget(CenteredSweepInputWidget):
    """Connect a caller-owned binding field to the centered range presentation."""

    def __init__(
        self,
        field: CenteredSweepField,
        parent: QWidget | None = None,
        *,
        text_input_enhancer: TextInputEnhancer | None = None,
    ) -> None:
        self._field = field
        super().__init__(
            field.spec,
            field.get_value(),
            parent,
            submit=self._write_input,
            text_input_enhancer=text_input_enhancer,
        )
        field.on_change.connect(self.display)

    @property
    def field(self) -> CfgField:
        return self._field

    def teardown(self) -> None:
        self._field.on_change.disconnect(self.display)

    def refresh_section(self, path: str) -> bool:
        del path
        return False

    def _write_input(
        self,
        edge: Literal["center", "span", "expts", "step"],
        value: DirectValue | EvalValue,
    ) -> None:
        if edge == "center":
            _write_scalar_input(self._field.center_field, value)
        else:
            self._field.set_text(edge, _range_input_text(value))


def _centered_sweep_label(text: str, badge: str) -> str:
    return f"{text} [{badge}]" if badge else text
