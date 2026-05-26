"""Common widgets for LiveFields."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

logger = logging.getLogger(__name__)

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSpinBox,
    QWidget,
)

from ...adapter import DirectValue, EvalValue, default_value_for_type
from ...live_model import (
    LiteralLiveField,
    LiveField,
    MultiSweepLiveField,
    ScalarLiveField,
    SweepLiveField,
)
from ..widgets import TrimDoubleSpinBox
from .registry import register_widget

if TYPE_CHECKING:
    from ...adapter import ScalarSpec


FIELD_INPUT_MIN_WIDTH = 20
FIELD_LABEL_MAX_WIDTH = 80


class ElidedLabel(QLabel):
    """QLabel that elides text when it exceeds FIELD_LABEL_MAX_WIDTH pixels.

    The full text is always shown in the tooltip so the user can read the
    complete field name on hover.
    """

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setMaximumWidth(FIELD_LABEL_MAX_WIDTH)
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
    choices: Optional[list],
    editable: bool = True,
    decimals: Optional[int] = None,
) -> QWidget:
    """Build an input widget from raw field attributes."""
    if choices:
        w = QComboBox()
        w.addItems([str(c) for c in choices])
        idx = w.findText(str(default))
        if idx >= 0:
            w.setCurrentIndex(idx)
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


def make_scalar_widget(spec: "ScalarSpec", value: Any) -> QWidget:
    """Build an input widget from a ScalarSpec and initial value."""
    return make_value_widget(
        spec.type, value, spec.choices, spec.editable, spec.decimals
    )


def read_scalar_widget(w: QWidget, spec: "ScalarSpec") -> Any:
    """Read the current value from a widget created by make_scalar_widget."""
    return read_value_widget(w, spec.type, fallback=None)


def _widget_default_for_direct_value(value: DirectValue, spec: "ScalarSpec") -> Any:
    if value.value is None:
        default = default_value_for_type(spec.type)
        return "" if default is None else default
    return value.value


def _sweep_edge_to_float(value: object, edge_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, DirectValue):
        if isinstance(value.value, (int, float)):
            return float(value.value)
        raise RuntimeError(f"Sweep {edge_name} must be numeric")
    if isinstance(value, EvalValue):
        if isinstance(value.resolved, (int, float)):
            return float(value.resolved)
        raise RuntimeError(f"Sweep {edge_name} expression is unresolved")
    raise RuntimeError(f"Sweep {edge_name} must be numeric")


class BaseLiveWidget(QWidget):
    """Base class implementing FieldWidgetProtocol."""

    def __init__(self, field: LiveField, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._field = field

    @property
    def field(self) -> LiveField:
        return self._field

    def teardown(self) -> None:
        pass


@register_widget(LiteralLiveField)
class LiteralWidget(QLabel):
    """Hidden widget for Literal values."""

    def __init__(self, field: LiteralLiveField, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._field = field
        self.setVisible(False)

    @property
    def field(self) -> LiveField:
        return self._field

    def teardown(self) -> None:
        pass


@register_widget(ScalarLiveField)
class ScalarWidget(BaseLiveWidget):
    """Generic input widget for ScalarLiveField."""

    def __init__(self, field: ScalarLiveField, parent: Optional[QWidget] = None):
        super().__init__(field, parent)
        self._updating = False
        self._input: Optional[QWidget] = None
        self._ghost: Optional[QLabel] = None
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
            field = cast(ScalarLiveField, self._field)
            inp = self._input
            assert inp is not None
            if isinstance(field.get_value(), EvalValue):
                assert isinstance(inp, QLineEdit)
                field.set_value(EvalValue(expr=inp.text().strip()))
                self._sync_eval_ghost(field.get_value())
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
            field = cast(ScalarLiveField, self._field)
            raw = _widget_default_for_direct_value(val, field.spec)
            if isinstance(inp, QComboBox):
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
        field = cast(ScalarLiveField, self._field)
        value = field.get_value()
        self._mode = "eval" if isinstance(value, EvalValue) else "direct"
        self._ghost = None

        if isinstance(value, EvalValue):
            inp = QLineEdit(value.expr)
            self._input = inp
            inp.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
            inp.setEnabled(field.spec.editable)
            inp.textChanged.connect(self._on_ui_changed)
            self._layout.addWidget(inp, stretch=1)

            self._ghost = QLabel()
            self._layout.addWidget(self._ghost)
            self._sync_eval_ghost(value)
        else:
            self._input = make_value_widget(
                field.spec.type,
                _widget_default_for_direct_value(value, field.spec),
                field.spec.choices,
                field.spec.editable,
                field.spec.decimals,
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
        self._ghost.setText(f"= {value.resolved}")
        self._ghost.setToolTip("")
        self._ghost.setStyleSheet("color: gray; font-style: italic;")

    def _install_context_menu(self, widget: Optional[QWidget]) -> None:
        if widget is None:
            return
        if not isinstance(widget, (QAbstractSpinBox, QLineEdit)):
            return
        widget.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        widget.customContextMenuRequested.connect(  # type: ignore[attr-defined]
            lambda pos, w=widget: self._show_context_menu(w, w.mapToGlobal(pos))
        )

    def _show_context_menu(
        self, widget: "QAbstractSpinBox | QLineEdit", global_pos: Any
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
        field = cast(ScalarLiveField, self._field)
        chosen = cast(Any, menu).exec_(global_pos)
        if chosen is not mode_action:
            return
        value = field.get_value()
        if isinstance(value, EvalValue):
            if value.resolved is None:
                field.set_value(None)
            else:
                field.set_value(DirectValue(value=value.resolved, is_unset=False))
            return

        if isinstance(value, DirectValue):
            expr = "" if value.is_unset else str(value.value)
            field.set_value(EvalValue(expr=expr))

    def _build_context_menu(self, widget: QLineEdit) -> tuple[QMenu, Any]:
        menu = widget.createStandardContextMenu()
        if menu is None:
            raise RuntimeError("QLineEdit.createStandardContextMenu() returned None")
        if not self._supports_eval_mode():
            return menu, None
        if menu.actions():
            menu.addSeparator()
        value = cast(ScalarLiveField, self._field).get_value()
        if isinstance(value, EvalValue):
            return menu, menu.addAction("Use direct value")
        return menu, menu.addAction("Use expression")

    def _supports_eval_mode(self) -> bool:
        spec = cast(ScalarLiveField, self._field).spec
        return spec.editable and spec.choices is None and spec.type in {int, float}


@register_widget(SweepLiveField)
class SweepWidget(BaseLiveWidget):
    """Inline 2x2 input for start/stop/expts/step with synchronized updates."""

    def __init__(self, field: SweepLiveField, parent: Optional[QWidget] = None):
        super().__init__(field, parent)
        self._updating = False

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        sv = field.get_value()

        decimals = field.spec.decimals

        self._start_widget = ScalarWidget(field.start_field, self)
        self._stop_widget = ScalarWidget(field.stop_field, self)
        field.start_field.on_change.connect(self._on_start_changed)
        field.stop_field.on_change.connect(self._on_stop_changed)

        self._expts = QSpinBox()
        self._expts.setRange(1, 2**31 - 1)
        self._expts.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._expts.setValue(sv.expts)
        self._expts.valueChanged.connect(self._on_ui_changed)

        self._step = TrimDoubleSpinBox()
        self._step.setRange(-1e12, 1e12)
        self._step.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._step.setDecimals(decimals if decimals is not None else 6)
        self._step.setValue(sv.step)
        self._step.valueChanged.connect(self._on_ui_changed)

        enabled = field.spec.editable
        self._start_widget.setEnabled(enabled)
        self._stop_widget.setEnabled(enabled)
        self._expts.setEnabled(enabled)
        self._step.setEnabled(enabled)

        layout.addWidget(QLabel("start"), 0, 0)
        layout.addWidget(self._start_widget, 0, 1)
        layout.addWidget(QLabel("stop"), 0, 2)
        layout.addWidget(self._stop_widget, 0, 3)
        layout.addWidget(QLabel("expts"), 1, 0)
        layout.addWidget(self._expts, 1, 1)
        layout.addWidget(QLabel("step"), 1, 2)
        layout.addWidget(self._step, 1, 3)

        field.on_change.connect(self._on_model_changed)

    def teardown(self) -> None:
        field = cast(SweepLiveField, self._field)
        field.on_change.disconnect(self._on_model_changed)
        field.start_field.on_change.disconnect(self._on_start_changed)
        field.stop_field.on_change.disconnect(self._on_stop_changed)
        self._start_widget.teardown()
        self._stop_widget.teardown()

    def _on_start_changed(self, *_: Any) -> None:
        self._on_ui_changed()

    def _on_stop_changed(self, *_: Any) -> None:
        self._on_ui_changed()

    def _on_ui_changed(self, *_: Any) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            from ...adapter import SweepValue

            field = cast(SweepLiveField, self._field)
            source = self.sender()
            try:
                start = _sweep_edge_to_float(field.start_field.get_value(), "start")
                stop = _sweep_edge_to_float(field.stop_field.get_value(), "stop")
            except RuntimeError:
                return
            expts = self._expts.value()
            step = self._step.value()

            if source is self._step:
                # step changed → only recompute expts, then back-calculate step
                # to match the integer expts value
                if step == 0.0:
                    expts = 1
                else:
                    expts = max(1, round((stop - start) / step + 1))
                step = 0.0 if expts == 1 else (stop - start) / (expts - 1)
                self._expts.setValue(expts)
                self._step.setValue(step)
            else:
                # start / stop / expts changed → only recompute step
                step = 0.0 if expts == 1 else (stop - start) / (expts - 1)
                self._step.setValue(step)

            # start and stop are never touched here; take them directly from the field
            current = field.get_value()
            nv = SweepValue(
                start=current.start,
                stop=current.stop,
                expts=expts,
                step=step,
            )
            self._field.set_value(nv)
        finally:
            self._updating = False

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


@register_widget(MultiSweepLiveField)
class MultiSweepWidget(BaseLiveWidget):
    """Container for multiple SweepWidgets."""

    def __init__(self, field: MultiSweepLiveField, parent: Optional[QWidget] = None):
        super().__init__(field, parent)

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._widgets: Dict[str, SweepWidget] = {}

        self._build_axes()

    def teardown(self) -> None:
        for w in self._widgets.values():
            w.teardown()

    def _build_axes(self) -> None:
        layout = cast(QFormLayout, self.layout())
        field = cast(MultiSweepLiveField, self._field)

        for axis, axis_field in field.fields.items():
            w = SweepWidget(axis_field)
            layout.addRow(f"  {axis}:", w)
            self._widgets[axis] = w
