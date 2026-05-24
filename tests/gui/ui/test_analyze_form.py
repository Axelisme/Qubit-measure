from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from zcu_tools.gui.adapter import ParamMeta
from zcu_tools.gui.ui.analyze_form import AnalyzeFormWidget
from zcu_tools.gui.ui.widgets import TrimDoubleSpinBox


@dataclass
class _TestParams:
    threshold: Annotated[float, ParamMeta(label="Threshold", decimals=2)]
    model: Annotated[Literal["a", "b"], ParamMeta(label="Model")]


def test_analyze_form_round_trips_values(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    form.populate(_TestParams(threshold=0.5, model="a"))

    result = form.read_params()

    assert result == _TestParams(threshold=0.5, model="a")
    assert form.is_valid() is True


def test_analyze_form_has_params_after_populate(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    assert form.has_params() is False

    form.populate(_TestParams(threshold=0.5, model="a"))

    assert form.has_params() is True


def test_analyze_form_populate_values_restores_state(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    form.populate(_TestParams(threshold=0.5, model="a"))
    form.populate_values(_TestParams(threshold=0.9, model="b"))
    assert form.read_params() == _TestParams(threshold=0.9, model="b")


def test_analyze_form_hydration_does_not_emit_params_changed(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    emitted: list[object] = []
    form.params_changed.connect(lambda values: emitted.append(values))

    form.populate(_TestParams(threshold=0.5, model="a"))
    form.populate_values(_TestParams(threshold=0.9, model="b"))

    assert emitted == []


def test_analyze_form_user_edit_emits_params_changed(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    emitted: list[object] = []
    form.populate(_TestParams(threshold=0.5, model="a"))
    form.params_changed.connect(lambda values: emitted.append(values))

    spin = form.findChild(TrimDoubleSpinBox)
    assert spin is not None
    spin.setValue(0.9)

    assert emitted[-1] == _TestParams(threshold=0.9, model="a")
