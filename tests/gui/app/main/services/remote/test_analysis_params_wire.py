from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.adapters.onetone.freq import (
    OneToneFreqAnalyzeParams,
)
from zcu_tools.gui.app.main.services.remote.handlers.analysis import (
    h_tab_analyze,
    h_tab_get_analyze_params,
    h_tab_post_analyze,
)
from zcu_tools.gui.remote.errors import RemoteError


def _adapter_with_params(params: OneToneFreqAnalyzeParams) -> MagicMock:
    control = MagicMock()
    control.has_tab.return_value = True
    control.get_tab_snapshot.return_value = SimpleNamespace(
        analysis=SimpleNamespace(params=params),
        post_analysis=SimpleNamespace(params=params),
        interaction=None,
    )
    control.analyze.return_value = "op-1"
    control.start_post_analyze.return_value = "op-1"
    adapter = MagicMock()
    adapter.run_analyze_control = control
    return adapter


def test_remote_analyze_params_exposes_and_accepts_amplitude_slope_key() -> None:
    adapter = _adapter_with_params(OneToneFreqAnalyzeParams())

    reply = h_tab_get_analyze_params(adapter, {"tab_id": "t"})
    assert reply["analyze_params"] == {
        "model_type": "hm",
        "fit_bg_amp_slope": True,
        "fit_bg_phase_curvature": False,
        "edelay_mode": "auto",
        "manual_edelay": None,
        "max_edelay_search_radius": 100.0,
    }

    started = h_tab_analyze(
        adapter,
        {
            "tab_id": "t",
            "updates": {
                "fit_bg_amp_slope": False,
                "fit_bg_phase_curvature": True,
                "edelay_mode": "manual",
                "manual_edelay": 11.3,
                "max_edelay_search_radius": 150.0,
            },
        },
    )
    assert started == {"operation_id": "op-1"}
    forwarded = adapter.run_analyze_control.analyze.call_args.args[1]
    assert forwarded == OneToneFreqAnalyzeParams(
        fit_bg_amp_slope=False,
        fit_bg_phase_curvature=True,
        edelay_mode="manual",
        manual_edelay=11.3,
        max_edelay_search_radius=150.0,
    )


@pytest.mark.parametrize("handler", [h_tab_analyze, h_tab_post_analyze])
@pytest.mark.parametrize(
    "updates",
    [
        {"fit_bg_amp_slope": "false"},
        {"manual_edelay": True},
        {"max_edelay_search_radius": "150"},
        {"model_type": "unknown"},
        {"unknown_param": 1},
    ],
)
def test_analysis_parameter_errors_do_not_start_operation(handler, updates) -> None:
    original = OneToneFreqAnalyzeParams()
    adapter = _adapter_with_params(original)
    with pytest.raises(RemoteError) as caught:
        handler(adapter, {"tab_id": "t", "updates": updates})
    assert caught.value.code.value == "invalid_params"
    assert caught.value.data is not None
    definitions = {item["name"]: item for item in caught.value.data["definitions"]}
    assert definitions["fit_bg_amp_slope"]["type"] == "bool"
    assert definitions["model_type"]["choices"] == ["hm", "t", "auto"]
    assert definitions["manual_edelay"]["optional"] is True
    adapter.run_analyze_control.analyze.assert_not_called()
    adapter.run_analyze_control.start_post_analyze.assert_not_called()
    assert original == OneToneFreqAnalyzeParams()


@pytest.mark.parametrize(
    ("handler", "method"),
    [(h_tab_analyze, "analyze"), (h_tab_post_analyze, "start_post_analyze")],
)
def test_analysis_parameters_preserve_omitted_values_and_accept_null(handler, method):
    adapter = _adapter_with_params(OneToneFreqAnalyzeParams(model_type="t"))
    assert handler(adapter, {"tab_id": "t", "updates": {"manual_edelay": None}}) == {
        "operation_id": "op-1"
    }
    operation = getattr(adapter.run_analyze_control, method)
    operation.assert_called_once_with("t", OneToneFreqAnalyzeParams(model_type="t"))


def test_remote_analyze_params_rejects_removed_key() -> None:
    adapter = _adapter_with_params(OneToneFreqAnalyzeParams())
    removed_key = "_".join(("fit", "bg", "slope"))

    with pytest.raises(RemoteError):
        h_tab_analyze(
            adapter,
            {"tab_id": "t", "updates": {removed_key: True}},
        )
