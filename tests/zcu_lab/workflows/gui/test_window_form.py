"""Window form contracts through public native/GUI seams."""

import pytest
from pydantic import JsonValue
from qtpy.QtWidgets import QLineEdit
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab.workflows.gui.window import T1WorkflowWindow

from ._window_support import RpcRequest


def test_form_inspection_rpc_reads_constructor_drafts(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, _harness, _entry = attachment
    for form, panel in (
        ("plan", window.plan_panel),
        ("tunables", window.tunables_panel),
    ):
        before = panel.state.input.data, panel.state.base.data
        reply = rpc("workflow.form", {"form": form})
        assert reply["ok"] is True
        result = reply["result"]
        assert isinstance(result, dict)
        assert result["input"] == before[0]
        assert result["revision"] is None
        assert isinstance(result["schema"], dict)
        assert isinstance(result["values"], list)
        assert isinstance(result["errors"], list)
        assert (panel.state.input.data, panel.state.base.data) == before
    assert window.controller.binding is None


@pytest.mark.parametrize(
    "value", [{"expr": "2 * 7"}, {"expr": "missing_knob"}, {"expr": "1e308 * 1e308"}]
)
def test_form_inspection_rpc_candidate_does_not_write_start_draft(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    value: JsonValue,
) -> None:
    window, _harness, _entry = attachment
    state = window.tunables_panel.state
    before = state.input.data, state.base.data
    reply = rpc(
        "workflow.form",
        {"form": "tunables", "batch": [{"path": ["guess_us"], "value": value}]},
    )
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    candidate = result["input"]
    assert isinstance(candidate, dict) and candidate["guess_us"] == value
    assert (state.input.data, state.base.data) == before
    if value == {"expr": "2 * 7"}:
        assert {"path": ["guess_us"], "value": 14.0} in result["values"]
        assert result["errors"] == []
    else:
        assert any(e["path"] == ["guess_us"] for e in result["errors"])
        assert not any(v["path"] == ["guess_us"] for v in result["values"])


def test_form_inspection_rpc_preserves_complex_tag_on_invalid_numeric_candidate(
    rpc: RpcRequest,
) -> None:
    tagged = {"__complex__": [1.0, 2.0]}
    reply = rpc(
        "workflow.form",
        {
            "form": "tunables",
            "batch": [{"path": ["guess_us"], "value": tagged}],
        },
    )
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    assert {"path": ["guess_us"], "value": tagged} in result["values"]
    assert any(e["path"] == ["guess_us"] for e in result["errors"])


def test_form_inspection_rpc_plan_branch_batch_does_not_change_gui(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, _harness, _entry = attachment
    before = window.plan_panel.state.input.data, window.plan_panel.state.base.data
    reply = rpc(
        "workflow.form",
        {
            "form": "plan",
            "batch": [
                {"path": ["fluxes", "kind"], "value": "linrange"},
                {"path": ["fluxes", "start"], "value": {"expr": "1 / 2"}},
                {"path": ["fluxes", "stop"], "value": 1.0},
                {"path": ["fluxes", "n"], "value": 3},
            ],
        },
    )
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    assert result["errors"] == []
    assert {"path": ["fluxes", "start"], "value": 0.5} in result["values"]
    assert (
        window.plan_panel.state.input.data,
        window.plan_panel.state.base.data,
    ) == before


def test_form_inspection_rpc_uses_accepted_host_and_preserves_local_text(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
) -> None:
    del paused_tuning
    window, harness, _entry = attachment
    panel = window.tuning_panel
    engine = window.controller.engine
    assert panel is not None and engine is not None
    state = panel.host.state
    state.write({("guess_us",): {"expr": "3 * 11"}}, panel.host.env)
    local = panel.panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    assert isinstance(local, QLineEdit)
    local.setText("-")
    before = (
        state.input.data,
        state.base.data,
        panel.host.accepted_input.data,
        engine.tunables("tuning"),
        window.controller.operation_id,
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()),
    )
    read = rpc("workflow.form", {"form": "tunables", "run_id": "tuning"})
    assert read["ok"] is True
    result = read["result"]
    assert isinstance(result, dict)
    assert result["input"] == before[0]
    assert result["revision"] == before[3].revision
    assert {"path": ["guess_us"], "value": 33.0} in result["values"]
    assert any(e["path"] == ["sweep_factor"] for e in result["errors"])
    for batch in (
        [],
        [{"path": ["guess_us"], "value": {"expr": "2 * 20"}}],
        [{"path": ["guess_us"], "value": -1}],
    ):
        reply = rpc(
            "workflow.form",
            {
                "form": "tunables",
                "run_id": "tuning",
                "batch": batch,
            },
        )
        assert reply["ok"] is True
        preview = reply["result"]
        assert isinstance(preview, dict)
        assert not any(e["path"] == ["sweep_factor"] for e in preview["errors"])
        if batch == []:
            assert preview["input"] == before[0]
        elif batch[0]["value"] == -1:
            assert any(e["path"] == ["guess_us"] for e in preview["errors"])
        else:
            assert {"path": ["guess_us"], "value": 40.0} in preview["values"]
    assert (
        state.input.data,
        state.base.data,
        panel.host.accepted_input.data,
        engine.tunables("tuning"),
        window.controller.operation_id,
        tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot()),
    ) == before
    assert local.text() == "-"
    assert panel.panel.editor.local_parse_errors() is not None


@pytest.mark.parametrize(
    "params",
    [
        {"form": "tunables", "batch": {}},
        {"form": "tunables", "batch": [{"path": ["unknown"], "value": 2}]},
        {"form": "tunables", "batch": [{"path": [], "value": 2}]},
        {
            "form": "tunables",
            "batch": [
                {"path": ["reps"], "value": 2},
                {"path": ["reps"], "value": 3},
            ],
        },
    ],
)
def test_form_inspection_rpc_rejects_bad_form_or_batch_without_writes(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    params: dict[str, object],
) -> None:
    window, _harness, _entry = attachment
    before = window.tunables_panel.state.input.data
    reply = rpc("workflow.form", params)
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict) and error["code"] == "invalid_params"
    assert window.tunables_panel.state.input.data == before


def test_form_inspection_rpc_rejects_unknown_form_at_parameter_seam(
    rpc: RpcRequest,
) -> None:
    with pytest.raises(RemoteError) as raised:
        rpc("workflow.form", {"form": "other"})
    assert raised.value.code is ErrorCode.INVALID_PARAMS


def test_form_inspection_rpc_requires_accepted_explicit_target(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, harness, _entry = run_attachment
    absent = rpc("workflow.form", {"form": "plan", "run_id": "absent"})
    assert absent["ok"] is False
    token = window.start("pending-form")
    assert isinstance(token, int)
    for run_id, expected in (
        ("pending-form", "controller_error"),
        ("wrong", "precondition_failed"),
        (None, "precondition_failed"),
    ):
        reply = rpc("workflow.form", {"form": "tunables", "run_id": run_id})
        assert reply["ok"] is False
        error = reply["error"]
        assert isinstance(error, dict) and error["code"] == expected
    window.stop("pending-form")
    harness.complete_segment()
