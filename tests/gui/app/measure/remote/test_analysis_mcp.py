"""Real MCP/socket analysis uses GUI parameters and pane-owned writeback."""

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from ._helpers import Fixture, call, mcp_client, open_client


@dataclass
class ScalarParams:
    threshold: float


@dataclass
class ScalarResult:
    value: float

    def to_summary_dict(self) -> dict[str, float]:
        return {"value": self.value}


def _install_result(fx, tab: str, stage: str, value: float, operation: int) -> None:
    result = ScalarResult(value)
    plots = Plots(NonPresentingHost())
    plots.adopt("fit", Figure())
    plots.finish()
    if stage == "analysis":
        fx.state.update_tab_analyze(tab, result, plots, source_operation_id=operation)
    else:
        adapter = fx.state.get_tab(tab).adapter
        adapter.capabilities = replace(adapter.capabilities, post_analysis=True)
        if fx.state.get_tab(tab).analysis.result is None:
            _install_result(fx, tab, "analysis", 1.0, 100)
        fx.state.update_tab_post_analyze(
            tab, result, plots, source_operation_id=operation
        )


def _result_method(stage: str) -> str:
    return (
        "tab.get_analyze_result"
        if stage == "analysis"
        else "tab.get_post_analyze_result"
    )


@pytest.fixture()
def fx(qapp):
    fixture = Fixture(active_label="ctx001")
    fixture.state.set_context(
        replace(fixture.state.session_env, md=MetaDict(), ml=ModuleLibrary())
    )
    fixture.start()
    yield fixture
    fixture.stop()


@pytest.mark.parametrize("stage", ["analysis", "post_analysis"])
def test_writeback_preview_accepts_matching_analysis_operation(fx, stage):
    tab = fx.ctrl.new_tab("fake")
    fx.state.update_tab_result(tab, object())
    _install_result(fx, tab, stage, 3.0, 101)
    with open_client(fx.service.port) as sock:
        result = call(
            sock,
            "tab.writeback_preview",
            {
                "tab_id": tab,
                "subtab_id": stage,
                "operation_id": 101,
            },
        )
        assert result["ok"] is True, result
        assert result["result"]["has_draft"] is False


def test_mcp_analysis_returns_actual_params_and_replaces_old_draft(fx, tmp_path):
    tab = fx.ctrl.new_tab("fake")
    run = fx.ctrl.start_run(tab, fx.ctrl.cfg_resources.lookup(tab).observe().ref)
    sock = open_client(fx.service.port)
    try:
        assert (
            call(sock, "operation.await", {"operation_id": run, "timeout": 2})[
                "result"
            ]["status"]
            == "finished"
        )
    finally:
        sock.close()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("rpc_call", {"method": "context.snapshot"})
        invoke("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab}})
        with pytest.raises(GuiRpcError, match="threshold") as error:
            invoke("tab_analyze", {"tab": tab, "params": {"threshold": "bad"}})
        assert error.value.code == "invalid_params"
        assert "float" in str(error.value)
        first = invoke("tab_analyze", {"tab": tab, "params": {"threshold": 0.3}})
        assert first["status"] == "finished"
        assert first["params"] == {"threshold": 0.3}
        assert first["invalidated"] == []
        assert Path(first["figure"]).is_file()
        previous = fx.state.get_tab(tab).analysis.writeback_draft
        assert previous is not None and previous.is_active
        second = invoke("tab_analyze", {"tab": tab})
        assert second["status"] == "finished"
        assert second["params"] == {"threshold": 0.3}
        assert second["invalidated"] == ["analysis.writeback"]
        assert not previous.is_active
        current = fx.state.get_tab(tab).analysis
        assert current.writeback_draft is not previous
        assert second["result"]["summary"] == current.result.to_summary_dict()
        assert second["save_status"] == "saved"
        assert second["saved_images"]
        assert all(
            Path(item["image_path"]).is_file() for item in second["saved_images"]
        )
    finally:
        bridge.disconnect()


@pytest.mark.parametrize("stage", ["analysis", "post_analysis"])
def test_operation_result_retains_inputs_after_parameter_edits_and_replacement(
    fx, stage
):
    tab = fx.ctrl.new_tab("fake")
    fx.state.update_tab_result(tab, object())
    if stage == "post_analysis":
        _install_result(fx, tab, "analysis", 1.0, 100)
    original = ScalarParams(0.3)
    update_params = (
        fx.state.update_tab_analyze_param_instance
        if stage == "analysis"
        else fx.state.update_tab_post_analyze_param_instance
    )
    update_params(tab, original)
    _install_result(fx, tab, stage, 3.0, 101)
    # The installed result owns a detached input, not the next-edit draft.
    original.threshold = 0.5
    update_params(tab, ScalarParams(0.7))
    with open_client(fx.service.port) as sock:
        observed = call(
            sock, _result_method(stage), {"tab_id": tab, "operation_id": 101}
        )["result"]
        assert observed["params"] == {"threshold": 0.3}
        assert observed["summary"] == {"value": 3.0}
        assert call(sock, _result_method(stage), {"tab_id": tab})["result"] == {
            "summary": {"value": 3.0}
        }
        _install_result(fx, tab, stage, 7.0, 102)
        replaced = call(
            sock, _result_method(stage), {"tab_id": tab, "operation_id": 101}
        )
        assert replaced["error"]["reason"] == "result_superseded"
        current = call(
            sock, _result_method(stage), {"tab_id": tab, "operation_id": 102}
        )["result"]
        assert current["params"] == {"threshold": 0.7}
        assert current["summary"] == {"value": 7.0}


@pytest.mark.parametrize("stage", ["analysis", "post_analysis"])
def test_operation_result_observation_unlocks_only_the_observed_image(
    fx, tmp_path, stage
):
    tab = fx.ctrl.new_tab("fake")
    fx.state.update_tab_result(tab, object())
    _install_result(fx, tab, stage, 3.0, 101)
    destination = tmp_path / f"{stage}.png"
    save = {
        "tab_id": tab,
        "subtab_id": stage,
        "figure_name": "fit",
        "operation_id": 101,
        "image_path": str(destination),
    }
    with open_client(fx.service.port) as sock:
        summary = call(sock, _result_method(stage), {"tab_id": tab})
        assert summary["result"] == {"summary": {"value": 3.0}}
        assert call(sock, "tab.get_figure", save)["ok"] is True
        assert call(sock, "tab.save_image", save)["error"]["reason"] == "stale_version"
        rejected = call(
            sock, _result_method(stage), {"tab_id": tab, "operation_id": 99}
        )
        assert rejected["error"]["reason"] == "result_superseded"
        assert call(sock, "tab.save_image", save)["error"]["reason"] == "stale_version"
        observed = call(
            sock, _result_method(stage), {"tab_id": tab, "operation_id": 101}
        )["result"]
        assert observed["summary"] == {"value": 3.0}
        assert observed["operation_id"] == 101
        snapshot = observed["operation_state"]
        assert snapshot["tab_id"] == tab
        assert snapshot["result_state"]["available"] is True
        assert snapshot[f"{stage}_state"]["has_figure"] is True
        assert snapshot["save_paths"]
        saved = call(sock, "tab.save_image", save)
        assert saved["result"]["image_path"] == str(destination)
        assert destination.read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("stage", ["analysis", "post_analysis"])
@pytest.mark.parametrize("replacement", ["same_pane", "run", "primary"])
def test_replaced_operation_cannot_read_or_save_another_result(
    fx, tmp_path, stage, replacement
):
    tab = fx.ctrl.new_tab("fake")
    fx.state.update_tab_result(tab, object())
    _install_result(fx, tab, stage, 3.0, 101)
    with open_client(fx.service.port) as sock:
        assert (
            call(sock, _result_method(stage), {"tab_id": tab, "operation_id": 101})[
                "ok"
            ]
            is True
        )
        if replacement == "same_pane":
            _install_result(fx, tab, stage, 7.0, 102)
        elif replacement == "run":
            fx.state.update_tab_result(tab, object())
        else:
            _install_result(fx, tab, "analysis", 8.0, 103)
        for method, params in (
            (_result_method(stage), {"tab_id": tab}),
            ("tab.get_figure", {"tab_id": tab, "subtab_id": stage}),
            ("tab.writeback_preview", {"tab_id": tab, "subtab_id": stage}),
        ):
            rejected = call(sock, method, {**params, "operation_id": 101})
            assert rejected["error"]["reason"] == "result_superseded"
        # Even a fresh full observation cannot turn the replaced pane into the old result.
        before = call(sock, "tab.snapshot", {"tab_id": tab})["result"]["tabs"][0]
        destination = tmp_path / "must-not-export.png"
        rejected = call(
            sock,
            "tab.save_image",
            {
                "tab_id": tab,
                "subtab_id": stage,
                "figure_name": "fit",
                "operation_id": 101,
                "image_path": str(destination),
            },
        )
        assert rejected["error"]["reason"] == "result_superseded"
        after = call(sock, "tab.snapshot", {"tab_id": tab})["result"]["tabs"][0]
        assert after["save_paths"] == before["save_paths"]
        assert after["artifacts"] == before["artifacts"]
        assert not destination.exists()


@pytest.mark.parametrize("stage", ["analysis", "post_analysis"])
def test_other_caller_save_path_change_invalidates_operation_observation(
    fx, tmp_path, stage
):
    tab = fx.ctrl.new_tab("fake")
    fx.state.update_tab_result(tab, object())
    _install_result(fx, tab, stage, 3.0, 101)
    bound = {"tab_id": tab, "operation_id": 101}
    with open_client(fx.service.port) as first, open_client(fx.service.port) as other:
        assert call(first, _result_method(stage), bound)["ok"] is True
        assert call(other, _result_method(stage), bound)["ok"] is True
        assert (
            call(
                other,
                "tab.save_image",
                {
                    **bound,
                    "subtab_id": stage,
                    "figure_name": "fit",
                    "image_path": str(tmp_path / "other.png"),
                },
            )["ok"]
            is True
        )
        rejected = call(
            first, "tab.save_image", {**bound, "subtab_id": stage, "figure_name": "fit"}
        )
        assert rejected["error"]["reason"] == "stale_version"


def test_operation_bound_run_figure_is_rejected(fx):
    tab = fx.ctrl.new_tab("fake")
    with open_client(fx.service.port) as sock:
        rejected = call(
            sock,
            "tab.get_figure",
            {"tab_id": tab, "subtab_id": "run", "operation_id": 101},
        )
        assert rejected["error"]["code"] == "invalid_params"


def test_result_without_figure_does_not_report_a_saved_image(fx, tmp_path):
    tab = fx.ctrl.new_tab("fake")
    fx.state.update_tab_result(tab, object())
    fx.state.update_tab_analyze(tab, ScalarResult(3.0), None, source_operation_id=101)
    fx.service.render_view = None
    destination = tmp_path / "no-figure.png"
    with open_client(fx.service.port) as sock:
        observed = call(
            sock, "tab.get_analyze_result", {"tab_id": tab, "operation_id": 101}
        )["result"]
        assert observed["operation_state"]["analysis_state"]["has_figure"] is False
        assert observed["summary"] == {"value": 3.0}
        params = {
            "tab_id": tab,
            "subtab_id": "analysis",
            "figure_name": "fit",
            "operation_id": 101,
        }
        assert call(sock, "tab.get_figure", params)["ok"] is False
        rejected = call(
            sock, "tab.save_image", {**params, "image_path": str(destination)}
        )
        assert rejected["ok"] is False
        assert "result" not in rejected
        assert not destination.exists()
