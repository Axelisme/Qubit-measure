"""Window form contracts through public native/GUI seams."""

import pytest
from pydantic import JsonValue
from qtpy.QtWidgets import QLineEdit
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from tests.gui.widgets.form._text_input import commit_text

from ._sample_input import sample_input
from ._synthetic import Context
from ._window_support import RpcRequest


def test_form_inspection_rpc_reads_constructor_drafts(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
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
        assert isinstance(result["values"], list)
        assert isinstance(result["errors"], list)
        assert (panel.state.input.data, panel.state.base.data) == before
    assert window.controller.binding is None


@pytest.mark.parametrize(
    "value", [{"expr": "2 * 7"}, {"expr": "missing_knob"}, {"expr": "1e308 * 1e308"}]
)
def test_form_inspection_rpc_candidate_does_not_write_start_draft(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    value: JsonValue,
) -> None:
    window, _harness, _entry = attachment
    state = window.tunables_panel.state
    before = state.input.data, state.base.data
    reply = rpc(
        "workflow.form",
        {"form": "tunables", "batch": [{"path": ["sample", "value"], "value": value}]},
    )
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    candidate = result["input"]
    assert isinstance(candidate, dict) and sample_input(candidate)["value"] == value
    assert (state.input.data, state.base.data) == before
    if value == {"expr": "2 * 7"}:
        assert {"path": ["sample", "value"], "value": 14.0} in result["values"]
        assert result["errors"] == []
    else:
        assert any(e["path"] == ["sample", "value"] for e in result["errors"])
        assert not any(v["path"] == ["sample", "value"] for v in result["values"])


def test_form_inspection_rpc_preserves_complex_tag_on_invalid_numeric_candidate(
    rpc: RpcRequest,
) -> None:
    tagged = {"__complex__": [1.0, 2.0]}
    reply = rpc(
        "workflow.form",
        {
            "form": "tunables",
            "batch": [{"path": ["sample", "value"], "value": tagged}],
        },
    )
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    assert {"path": ["sample", "value"], "value": tagged} in result["values"]
    assert any(e["path"] == ["sample", "value"] for e in result["errors"])


def test_form_inspection_rpc_plan_branch_batch_does_not_change_gui(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    window, _harness, _entry = attachment
    before = window.plan_panel.state.input.data, window.plan_panel.state.base.data
    reply = rpc(
        "workflow.form",
        {
            "form": "plan",
            "batch": [
                {"path": ["samples", "kind"], "value": "linrange"},
                {"path": ["samples", "start"], "value": {"expr": "1 / 2"}},
                {"path": ["samples", "stop"], "value": 1.0},
                {"path": ["samples", "n"], "value": 3},
            ],
        },
    )
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    assert result["errors"] == []
    assert {"path": ["samples", "start"], "value": 0.5} in result["values"]
    assert (
        window.plan_panel.state.input.data,
        window.plan_panel.state.base.data,
    ) == before


def test_form_inspection_rpc_uses_accepted_host_and_preserves_local_text(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
) -> None:
    del paused_tuning
    window, harness, _entry = attachment
    panel = window.tuning_panel
    engine = window.controller.engine
    assert panel is not None and engine is not None
    state = panel.host.state
    state.write({("sample", "value"): {"expr": "3 * 11"}}, panel.host.env)
    local = panel.panel.editor.findChild(QLineEdit, "form.input.sample.factor")
    assert isinstance(local, QLineEdit)
    commit_text(local, "-")
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
    assert {"path": ["sample", "value"], "value": 33.0} in result["values"]
    assert any(e["path"] == ["sample", "factor"] for e in result["errors"])
    for batch in (
        [],
        [{"path": ["sample", "value"], "value": {"expr": "2 * 20"}}],
        [{"path": ["sample", "value"], "value": -1}],
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
        assert not any(e["path"] == ["sample", "factor"] for e in preview["errors"])
        if batch == []:
            assert preview["input"] == before[0]
        elif batch[0]["value"] == -1:
            assert any(e["path"] == ["sample", "value"] for e in preview["errors"])
        else:
            assert {"path": ["sample", "value"], "value": 40.0} in preview["values"]
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
                {"path": ["sample", "count"], "value": 2},
                {"path": ["sample", "count"], "value": 3},
            ],
        },
    ],
)
def test_form_inspection_rpc_rejects_bad_form_or_batch_without_writes(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
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
    run_attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
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


@pytest.mark.parametrize("path", [[], ["samples"], ["samples", "start"]])
def test_describe_form_rpc_uses_call_local_branch(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    path: list[str],
) -> None:
    window, _harness, _entry = attachment
    state = window.plan_panel.state
    before = state.input.data, state.base.data
    reply = rpc(
        "workflow.describe_form",
        {
            "form": "plan",
            "path": path,
            "batch": [
                {"path": ["samples", "kind"], "value": "linrange"},
                {"path": ["samples", "start"], "value": {"expr": "1 / 2"}},
                {"path": ["samples", "n"], "value": 0},
            ],
        },
    )
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    assert result["revision"] is None
    assert {"path": ["samples", "start"], "value": 0.5} in result["values"]
    assert any(e["path"][-1:] == ["n"] for e in result["errors"])
    description = result["description"]
    assert description["path"] == path
    if not path:
        assert description["target"] is None
        assert [child["path"] for child in description["children"]] == [
            ["scale"], ["points"], ["samples"]
        ]
    elif path == ["samples"]:
        target = description["target"]
        assert target["options"][target["selected"]]["value"] == {"kind": "linrange"}
        assert {tuple(child["path"]) for child in description["children"]} == {
            ("samples", "kind"), ("samples", "start"),
            ("samples", "stop"), ("samples", "n"),
        }
    else:
        assert description["target"]["expression"] is True
        assert description["target"]["kind"] == "number"
        assert description["children"] == []
    assert (state.input.data, state.base.data) == before
    assert window.controller.binding is None


def test_describe_form_rpc_keeps_run_capture_and_local_errors_read_only(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
) -> None:
    del paused_tuning
    window, harness, _entry = attachment
    panel = window.tuning_panel
    engine = window.controller.engine
    assert panel is not None and engine is not None
    host = panel.host
    host.state.write({("sample", "value"): {"expr": "general.ext.value * 2"}}, host.env)
    local = panel.panel.editor.findChild(QLineEdit, "form.input.sample.factor")
    assert isinstance(local, QLineEdit)
    commit_text(local, "-")
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.value", 99.0)
    before = host.state.input.data, host.state.base.data, host.accepted_input.data, engine.tunables("tuning")
    for batch in (None, []):
        reply = rpc("workflow.describe_form", {
            "form": "tunables", "run_id": "tuning", "path": ["sample", "value"],
            "batch": batch,
        })
        assert reply["ok"] is True
        result = reply["result"]
        assert isinstance(result, dict)
        assert result["description"]["target"]["constraints"]["gt"] == 0
        assert result["revision"] == before[3].revision
        assert {"path": ["sample", "value"], "value": 40.0} in result["values"]
        assert any(e["path"] == ["sample", "factor"] for e in result["errors"]) == (batch is None)
    assert (host.state.input.data, host.state.base.data, host.accepted_input.data, engine.tunables("tuning")) == before
    assert local.text() == "-"


@pytest.mark.parametrize("run_id", [None, "stale"])
def test_describe_form_rpc_requires_explicit_attached_target(
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    run_id: str | None,
) -> None:
    del paused_tuning
    reply = rpc("workflow.describe_form", {"form": "plan", "path": [], "run_id": run_id})
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict) and error["code"] == "precondition_failed"
