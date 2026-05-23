from __future__ import annotations

from zcu_tools.gui.adapter import AnalyzeParam
from zcu_tools.gui.ui.analyze_form import AnalyzeFormWidget


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
