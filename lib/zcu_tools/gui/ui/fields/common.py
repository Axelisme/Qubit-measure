"""Common widgets for LiveFields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from ...live_model import (
    ChannelLiveField,
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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        spec = field.spec
        val = field.get_value().value

        self._input = make_value_widget(
            spec.type, val, spec.choices, spec.editable, spec.decimals
        )
        layout.addWidget(self._input)

        # Connect signals based on widget type
        if isinstance(self._input, QComboBox):
            self._input.currentIndexChanged.connect(self._on_ui_changed)
        elif isinstance(self._input, QCheckBox):
            self._input.toggled.connect(self._on_ui_changed)
        elif isinstance(self._input, (QSpinBox, TrimDoubleSpinBox)):
            self._input.valueChanged.connect(self._on_ui_changed)
        elif isinstance(self._input, QLineEdit):
            self._input.textChanged.connect(self._on_ui_changed)

        # Reactive sync: Model -> UI
        field.on_change.connect(self._on_model_changed)

    def teardown(self) -> None:
        self._field.on_change.disconnect(self._on_model_changed)

    def _on_ui_changed(self, *_: Any) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            val = read_value_widget(
                self._input, cast(ScalarLiveField, self._field).spec.type
            )
            self._field.set_value(val)
        finally:
            self._updating = False

    def _on_model_changed(self, val: Any) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            raw = val.value
            if isinstance(self._input, QComboBox):
                idx = self._input.findText(str(raw))
                if idx >= 0:
                    self._input.setCurrentIndex(idx)
            elif isinstance(self._input, QCheckBox):
                self._input.setChecked(bool(raw))
            elif isinstance(self._input, (QSpinBox, TrimDoubleSpinBox)):
                self._input.setValue(raw)
            elif isinstance(self._input, QLineEdit):
                self._input.setText(str(raw))
        finally:
            self._updating = False


@register_widget(SweepLiveField)
class SweepWidget(BaseLiveWidget):
    """Inline input for start/stop/pts."""

    def __init__(self, field: SweepLiveField, parent: Optional[QWidget] = None):
        super().__init__(field, parent)
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        sv = field.get_value()

        self._start = TrimDoubleSpinBox()
        self._start.setRange(-1e12, 1e12)
        self._start.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._start.setValue(sv.start)
        self._start.valueChanged.connect(self._on_ui_changed)

        self._stop = TrimDoubleSpinBox()
        self._stop.setRange(-1e12, 1e12)
        self._stop.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._stop.setValue(sv.stop)
        self._stop.valueChanged.connect(self._on_ui_changed)

        self._expts = QSpinBox()
        self._expts.setRange(1, 2**31 - 1)
        self._expts.setButtonSymbols(QAbstractSpinBox.NoButtons)  # type: ignore[attr-defined]
        self._expts.setValue(sv.expts)
        self._expts.valueChanged.connect(self._on_ui_changed)

        enabled = field.spec.editable
        self._start.setEnabled(enabled)
        self._stop.setEnabled(enabled)
        self._expts.setEnabled(enabled)

        layout.addWidget(QLabel("start"))
        layout.addWidget(self._start, stretch=1)
        layout.addWidget(QLabel("stop"))
        layout.addWidget(self._stop, stretch=1)
        layout.addWidget(QLabel("pts"))
        layout.addWidget(self._expts)

        field.on_change.connect(self._on_model_changed)

    def teardown(self) -> None:
        self._field.on_change.disconnect(self._on_model_changed)

    def _on_ui_changed(self, *_: Any) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            from ...adapter import SweepValue

            nv = SweepValue(
                start=self._start.value(),
                stop=self._stop.value(),
                expts=self._expts.value(),
                step=cast(SweepLiveField, self._field).get_value().step,
            )
            self._field.set_value(nv)
        finally:
            self._updating = False

    def _on_model_changed(self, val: Any) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self._start.setValue(val.start)
            self._stop.setValue(val.stop)
            self._expts.setValue(val.expts)
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
        field.on_change.connect(self._on_model_changed)

    def teardown(self) -> None:
        self._field.on_change.disconnect(self._on_model_changed)
        for w in self._widgets.values():
            w.teardown()

    def _build_axes(self) -> None:
        layout = cast(QFormLayout, self.layout())
        field = cast(MultiSweepLiveField, self._field)

        for axis, axis_field in field.fields.items():
            w = SweepWidget(axis_field)
            layout.addRow(f"  {axis}:", w)
            self._widgets[axis] = w

    def _on_model_changed(self, val: Any) -> None:
        # MultiSweepLiveField.set_value already updates child fields.
        # SweepWidget already listens to child field changes.
        pass


@register_widget(ChannelLiveField)
class ChannelWidget(BaseLiveWidget):
    """Input with resolution ghost label."""

    def __init__(self, field: ChannelLiveField, parent: Optional[QWidget] = None):
        super().__init__(field, parent)
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        cv = field.get_value()
        self._edit = QLineEdit(str(cv.chosen))
        self._edit.setMinimumWidth(FIELD_INPUT_MIN_WIDTH)
        self._edit.textChanged.connect(self._on_ui_changed)
        layout.addWidget(self._edit, stretch=1)

        self._ghost = QLabel()
        self._ghost.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._ghost)

        field.on_change.connect(self._on_model_changed)
        self._update_ghost(cv.resolved)

    def teardown(self) -> None:
        self._field.on_change.disconnect(self._on_model_changed)

    def _on_ui_changed(self, text: str) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self._field.set_value(text.strip())
        finally:
            self._updating = False

    def _on_model_changed(self, val: Any) -> None:
        self._update_ghost(val.resolved)
        if self._updating:
            return
        self._updating = True
        try:
            self._edit.setText(str(val.chosen))
        finally:
            self._updating = False

    def _update_ghost(self, resolved: Optional[int]) -> None:
        if resolved is not None:
            try:
                int(self._edit.text().strip())
                self._ghost.setText("")
            except ValueError:
                self._ghost.setText(f"= {resolved}")
            self._ghost.setStyleSheet("color: gray; font-style: italic;")
        else:
            self._ghost.setText("= ?")
            self._ghost.setStyleSheet("color: red; font-style: italic;")
