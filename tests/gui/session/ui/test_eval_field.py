"""Expression preview and direct/eval transitions through the public Qt widget seam."""

from __future__ import annotations

import pytest
from qtpy.QtCore import QPoint, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QMenu,
    QWidget,
)
from zcu_tools.gui.session.expression import EvalRef
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.ui.eval_field import EvalNumericField

from tests.gui.session.ui._parameter_fakes import make_parameters


def _field(
    parameters: ParameterContext | None = None,
    *,
    decimals: int = 6,
    minimum: float = -1e9,
    maximum: float = 1e9,
) -> EvalNumericField:
    return EvalNumericField(
        minimum=minimum,
        maximum=maximum,
        decimals=decimals,
        parameter_provider=lambda: parameters,
        type_=float,
    )


def _spin(field: EvalNumericField) -> QDoubleSpinBox:
    spin = field.findChild(QDoubleSpinBox)
    assert spin is not None
    return spin


def _edit(field: EvalNumericField) -> QLineEdit:
    return next(
        edit for edit in field.findChildren(QLineEdit) if edit.parent() is field
    )


def _ghost(field: EvalNumericField) -> QLabel:
    ghost = field.findChild(QLabel)
    assert ghost is not None
    return ghost


def _choose_mode(widget: QWidget, text: str) -> None:
    """Choose a real context-menu action; callbacks collect facts instead of raising."""
    chosen: list[str] = []

    def choose() -> None:
        menu = QApplication.activePopupWidget()
        if isinstance(menu, QMenu):
            for action in menu.actions():
                if action.text() == text:
                    chosen.append(action.text())
                    action.trigger()
                    break
            menu.close()

    def close_popup() -> None:
        popup = QApplication.activePopupWidget()
        if popup is not None:
            popup.close()

    timeout = QTimer(widget)
    timeout.setSingleShot(True)
    timeout.timeout.connect(close_popup)
    timeout.start(1000)
    QTimer.singleShot(0, choose)
    widget.customContextMenuRequested.emit(QPoint(0, 0))
    timeout.stop()
    assert chosen == [text]


def test_direct_default_read_raw_returns_float(qapp):
    field = _field()
    field.load_direct(1.5)
    result = field.read_raw()
    assert isinstance(result, float)
    assert result == pytest.approx(1.5)


def test_direct_default_mode_is_direct(qapp):
    field = _field()
    assert field.read_raw() == 0.0
    assert not _spin(field).isHidden()
    assert _edit(field).isHidden()
    assert _ghost(field).isHidden()


def test_eval_mode_read_raw_returns_eval_ref(qapp):
    field = _field(make_parameters({"flx_int": 0.5}))
    field.load_expression("general.ext.flx_int")
    result = field.read_raw()
    assert isinstance(result, EvalRef)
    assert result.expr == "general.ext.flx_int"
    assert result.type_ is float


def test_eval_mode_ghost_shows_resolved_value(qapp):
    field = _field(make_parameters({"flx_int": 0.5}))
    field.load_expression("0")
    _edit(field).setText("general.ext.flx_int")
    assert _ghost(field).text() == "= 0.5"
    assert "gray" in _ghost(field).styleSheet()
    assert "red" not in _ghost(field).styleSheet()


def test_undefined_name_shows_red_ghost(qapp):
    field = _field(make_parameters({}))
    field.load_expression("general.ext.missing_var")
    assert _ghost(field).text() == "= ?"
    assert "red" in _ghost(field).styleSheet()
    assert _ghost(field).toolTip()


def test_syntax_error_shows_red_ghost(qapp):
    field = _field(make_parameters({}))
    field.load_expression("general.ext.r_f[")
    assert _ghost(field).text() == "= ?"
    assert "red" in _ghost(field).styleSheet()


def test_eval_to_direct_carries_resolved_value(qapp):
    field = _field(make_parameters({"flx_int": 0.5}))
    field.load_expression("general.ext.flx_int")
    _choose_mode(_edit(field), "Use direct value")
    result = field.read_raw()
    assert isinstance(result, float)
    assert result == pytest.approx(0.5)


def test_eval_to_direct_falls_back_to_stored_value_on_unresolved(qapp):
    field = _field(make_parameters({}))
    field.load_direct(7.0)
    field.load_expression("general.ext.missing_var")
    _choose_mode(_edit(field), "Use direct value")
    assert field.read_raw() == pytest.approx(7.0)


def test_load_direct_in_eval_mode_preserves_expression(qapp):
    field = _field(make_parameters({"flx_int": 0.5}))
    field.load_expression("general.ext.flx_int")
    field.load_direct(9.0)
    result = field.read_raw()
    assert isinstance(result, EvalRef)
    assert result.expr == "general.ext.flx_int"
    assert _edit(field).text() == "general.ext.flx_int"
    field.reset_to_direct()
    assert field.read_raw() == pytest.approx(9.0)


def test_reset_to_direct_from_eval_mode(qapp):
    field = _field(make_parameters({"flx_int": 0.5}))
    field.load_direct(3.0)
    field.load_expression("general.ext.flx_int")
    field.reset_to_direct()
    assert field.read_raw() == pytest.approx(3.0)
    assert not _spin(field).isHidden()
    assert _edit(field).isHidden()
    assert _ghost(field).isHidden()


def test_load_expression_switches_to_eval_mode(qapp):
    field = _field(make_parameters({"frequency_mhz": 4567.0}))
    field.load_expression("general.ext.frequency_mhz", direct_fallback=1.25)
    assert not _edit(field).isHidden()
    assert _spin(field).isHidden()
    result = field.read_raw()
    assert isinstance(result, EvalRef)
    assert result.expr == "general.ext.frequency_mhz"


def test_no_context_parameter_provider_shows_red_ghost(qapp):
    def provider() -> ParameterContext:
        raise RuntimeError("no active context")

    field = EvalNumericField(
        minimum=-1e9, maximum=1e9, decimals=6, parameter_provider=provider
    )
    field.load_expression("general.ext.flx_int")
    assert _ghost(field).text() == "= ?"
    assert "red" in _ghost(field).styleSheet()
    assert "no active context" in _ghost(field).toolTip()


def test_ghost_strips_trailing_zeros(qapp):
    field = _field(make_parameters({"r_f": 6000.0}), decimals=3)
    field.load_expression("general.ext.r_f")
    assert _ghost(field).text() == "= 6000.0"


def test_context_menu_action_direct_to_eval(qapp):
    field = _field()
    field.load_direct(2.5)
    _choose_mode(_spin(field), "Use expression")
    result = field.read_raw()
    assert isinstance(result, EvalRef)
    assert result.expr == "2.5"
    assert not _edit(field).isHidden()
    assert _spin(field).isHidden()


def test_read_raw_eval_carries_min_max(qapp):
    field = _field(minimum=1e-9, maximum=1e9, decimals=9)
    field.load_expression("1.0")
    result = field.read_raw()
    assert isinstance(result, EvalRef)
    assert result.minimum == 1e-9
    assert result.maximum == 1e9


def test_ghost_above_max_shows_red(qapp):
    field = _field(make_parameters({"x": 200.0}), minimum=0.0, maximum=100.0)
    field.load_expression("general.ext.x")
    assert "red" in _ghost(field).styleSheet()
    assert "out of range" in _ghost(field).toolTip()


def test_ghost_below_min_shows_red(qapp):
    field = _field(make_parameters({"x": 0.0}), minimum=1e-9)
    field.load_expression("general.ext.x")
    assert "red" in _ghost(field).styleSheet()
    assert "out of range" in _ghost(field).toolTip()


def test_ghost_in_range_shows_gray(qapp):
    field = _field(make_parameters({"x": 50.0}), minimum=0.0, maximum=100.0)
    field.load_expression("general.ext.x")
    assert "gray" in _ghost(field).styleSheet()
    assert "red" not in _ghost(field).styleSheet()
    assert _ghost(field).toolTip() == ""


def test_context_menu_action_eval_to_direct(qapp):
    field = _field(make_parameters({"flx_int": 0.5}))
    field.load_expression("general.ext.flx_int")
    _choose_mode(_edit(field), "Use direct value")
    assert field.read_raw() == pytest.approx(0.5)
    assert not _spin(field).isHidden()
    assert _edit(field).isHidden()


def test_context_free_calculation_resolves_but_parameter_path_is_unresolved(qapp):
    field = _field()
    field.load_expression("sqrt(4) + 1")
    assert _ghost(field).text() == "= 3.0"
    field.load_expression("qubit.freq")
    assert _ghost(field).text() == "= ?"
    assert _ghost(field).toolTip()
