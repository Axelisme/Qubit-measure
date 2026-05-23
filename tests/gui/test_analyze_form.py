from __future__ import annotations

from zcu_tools.gui.adapter import AnalyzeParam
from zcu_tools.gui.ui.analyze_form import AnalyzeFormWidget
from zcu_tools.gui.ui.widgets import TrimDoubleSpinBox


def test_analyze_form_round_trips_values(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    params = [
        AnalyzeParam(key="threshold", label="Threshold", type=float, default=0.5),
        AnalyzeParam(
            key="model_type",
            label="Model type",
            type=str,
            default="hm",
            choices=["hm", "t", "auto"],
        ),
    ]
    form.populate(params)

    raw = form.read_params()

    assert raw == {"threshold": 0.5, "model_type": "hm"}
    assert form.is_valid() is True


def test_analyze_form_has_params_after_populate(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    assert form.has_params() is False

    form.populate([AnalyzeParam(key="flag", label="Flag", type=bool, default=True)])

    assert form.has_params() is True


def test_analyze_form_populate_values_restores_state(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    form.populate(
        [AnalyzeParam(key="threshold", label="Threshold", type=float, default=0.5)]
    )
    form.populate_values({"threshold": 0.9})
    assert form.read_params() == {"threshold": 0.9}


def test_analyze_form_hydration_does_not_emit_params_changed(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    emitted: list[dict[str, object]] = []
    form.params_changed.connect(lambda values: emitted.append(values))

    form.populate(
        [AnalyzeParam(key="threshold", label="Threshold", type=float, default=0.5)]
    )
    form.populate_values({"threshold": 0.9})

    assert emitted == []


def test_analyze_form_user_edit_emits_params_changed(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    emitted: list[dict[str, object]] = []
    form.populate(
        [AnalyzeParam(key="threshold", label="Threshold", type=float, default=0.5)]
    )
    form.params_changed.connect(lambda values: emitted.append(values))

    spin = form.findChild(TrimDoubleSpinBox)
    assert spin is not None
    spin.setValue(0.9)

    assert emitted[-1] == {"threshold": 0.9}
