from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.app.measure.remote.handlers.analysis import (
    h_tab_analyze,
    h_tab_get_analyze_params,
    h_tab_post_analyze,
)
from zcu_tools.gui.remote.errors import RemoteError

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._synthetic import ChoiceParams

from ._helpers import Fixture, call, open_client


def test_analyze_params_wire_generates_definitions_after_run(qapp, tmp_path):
    fx = Fixture(project_root=str(tmp_path / "project"))
    fx.start()
    tab_id = fx.ctrl.new_tab("fake")
    sock = open_client(fx.service.port)
    try:
        primary = call(sock, "tab.get_analyze_params", {"tab_id": tab_id})
        assert primary["ok"] is True
        assert primary["result"]["analyze_params"] is None
        assert primary["result"]["definitions"] == []
        fx.ctrl.start_run(tab_id)
        qt_until(
            lambda: fx.ctrl.active_operation_count() == 0, label="sample Run terminal"
        )
        assert fx.state.get_tab(tab_id).run.result is not None
        generated = call(sock, "tab.get_analyze_params", {"tab_id": tab_id})
        assert generated["ok"] is True
        assert generated["result"]["definitions"] == [
            {
                "name": "threshold",
                "type": "float",
                "label": "Threshold",
                "decimals": 2,
                "default": 0.5,
            }
        ]
        post = call(sock, "tab.get_post_analyze_params", {"tab_id": tab_id})
        assert post["ok"] is True
        assert post["result"]["post_analyze_params"] is None
        assert isinstance(post["result"]["definitions"], list)
    finally:
        sock.close()
        fx.stop()


def _adapter_with_params(params: ChoiceParams) -> MagicMock:
    control = MagicMock()
    control.has_tab.return_value = True
    control.get_tab_snapshot.return_value = SimpleNamespace(
        analysis=SimpleNamespace(params=params, has_writeback_draft=False),
        post_analysis=SimpleNamespace(
            params=params, has_writeback_draft=False, result=None
        ),
        capabilities=SimpleNamespace(analysis=AnalysisMode.FIT),
        interaction=None,
        artifacts=(),
        run=None,
    )
    control.analyze.return_value = "op-1"
    control.start_post_analyze.return_value = "op-1"
    adapter = MagicMock()
    adapter.run_analyze_control = control
    return adapter


def test_remote_analyze_params_exposes_and_accepts_amplitude_slope_key() -> None:
    adapter = _adapter_with_params(ChoiceParams())

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
    assert forwarded == ChoiceParams(
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
    original = ChoiceParams()
    adapter = _adapter_with_params(original)
    with pytest.raises(RemoteError) as caught:
        handler(adapter, {"tab_id": "t", "updates": updates})
    assert caught.value.code.value == "invalid_params"
    assert caught.value.data is not None
    wire_definitions = caught.value.data["definitions"]
    assert isinstance(wire_definitions, list)
    definitions: dict[str, dict[str, object]] = {}
    for item in wire_definitions:
        assert isinstance(item, dict)
        name = item["name"]
        assert isinstance(name, str)
        definitions[name] = item
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
    assert original == ChoiceParams()


@pytest.mark.parametrize(
    ("handler", "method"),
    [(h_tab_analyze, "analyze"), (h_tab_post_analyze, "start_post_analyze")],
)
def test_analysis_parameters_preserve_omitted_values_and_accept_null(handler, method):
    original = ChoiceParams(model_type="t")
    adapter = _adapter_with_params(original)
    reply = handler(adapter, {"tab_id": "t", "updates": {"manual_edelay": None}})
    assert reply["operation_id"] == "op-1"
    assert reply["params"] == asdict(original)
    operation = getattr(adapter.run_analyze_control, method)
    kwargs = {"run_operation_id": None}
    if method == "start_post_analyze":
        kwargs["operation_id"] = None
    operation.assert_called_once_with("t", ChoiceParams(model_type="t"), **kwargs)


@pytest.mark.parametrize(
    "handler, expected",
    [
        (h_tab_analyze, ["analysis.writeback", "post.result", "post.writeback"]),
        (h_tab_post_analyze, ["post.writeback"]),
    ],
)
def test_start_projects_only_content_replaced_on_success(handler, expected):
    original = ChoiceParams()
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
    adapter = _adapter_with_params(ChoiceParams())
    removed_key = "_".join(("fit", "bg", "slope"))

    with pytest.raises(RemoteError):
        h_tab_analyze(
            adapter,
            {"tab_id": "t", "updates": {removed_key: True}},
        )
