from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.measure.adapters.onetone.freq import (
    OneToneFreqAnalyzeParams,
)
from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.app.measure.remote.handlers.analysis import (
    h_tab_analyze,
    h_tab_get_analyze_params,
    h_tab_post_analyze,
)
from zcu_tools.gui.remote.errors import RemoteError


def _adapter_with_params(params: OneToneFreqAnalyzeParams) -> MagicMock:
    control = MagicMock()
    control.has_tab.return_value = True
    control.get_tab_snapshot.return_value = SimpleNamespace(
        analysis=SimpleNamespace(params=params, has_writeback_draft=False),
        post_analysis=SimpleNamespace(
            params=params, has_writeback_draft=False, result=None
        ),
        capabilities=SimpleNamespace(analysis=AnalysisMode.FIT),
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
    assert started["operation_id"] == "op-1"
    assert started["interactive"] is False
    assert started["invalidated_on_success"] == []
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
    # MCP forwards the message even when structured wire error data is omitted.
    message = str(caught.value)
    for field in definitions:
        assert field in message
    assert "bool" in message
    assert all(choice in message for choice in ("hm", "auto"))
    adapter.run_analyze_control.analyze.assert_not_called()
    adapter.run_analyze_control.start_post_analyze.assert_not_called()
    assert original == OneToneFreqAnalyzeParams()


@pytest.mark.parametrize(
    ("handler", "method"),
    [(h_tab_analyze, "analyze"), (h_tab_post_analyze, "start_post_analyze")],
)
def test_analysis_parameters_preserve_omitted_values_and_accept_null(handler, method):
    original = OneToneFreqAnalyzeParams(model_type="t")
    adapter = _adapter_with_params(original)
    reply = handler(adapter, {"tab_id": "t", "updates": {"manual_edelay": None}})
    assert reply["operation_id"] == "op-1"
    assert reply["params"] == asdict(original)
    operation = getattr(adapter.run_analyze_control, method)
    operation.assert_called_once_with("t", OneToneFreqAnalyzeParams(model_type="t"))


@pytest.mark.parametrize(
    "handler, expected",
    [
        (h_tab_analyze, ["analysis.writeback", "post.result", "post.writeback"]),
        (h_tab_post_analyze, ["post.writeback"]),
    ],
)
def test_start_projects_only_content_replaced_on_success(handler, expected):
    original = OneToneFreqAnalyzeParams()
    adapter = _adapter_with_params(original)
    snapshot = adapter.run_analyze_control.get_tab_snapshot.return_value
    snapshot.analysis.has_writeback_draft = True
    snapshot.post_analysis.has_writeback_draft = True
    snapshot.post_analysis.result = object()
    snapshot.capabilities.analysis = AnalysisMode.INTERACTIVE

    reply = handler(adapter, {"tab_id": "t", "updates": {}})

    assert reply["invalidated_on_success"] == expected
    assert reply["params"] == asdict(original)
    assert reply["interactive"] is (handler is h_tab_analyze)


def test_remote_analyze_params_rejects_removed_key() -> None:
    adapter = _adapter_with_params(OneToneFreqAnalyzeParams())
    removed_key = "_".join(("fit", "bg", "slope"))

    with pytest.raises(RemoteError):
        h_tab_analyze(
            adapter,
            {"tab_id": "t", "updates": {removed_key: True}},
        )
