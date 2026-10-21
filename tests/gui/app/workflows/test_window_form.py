"""Window form contracts through public native/GUI seams."""

from collections.abc import Mapping
from typing import Annotated, Literal

import pytest
from pydantic import Field, JsonValue
from qtpy.QtWidgets import QLineEdit
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import FormEnv, FormModel, FormPath, FormPathError, Option
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.mcp.core.rpc import GuiInvocationError
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from tests.gui.widgets.form._text_input import commit_text

from ._reload_source import InjectedCatalog
from ._sample_input import sample_input
from ._synthetic import Context, PlanForm, TunablesForm, context_seed, sample_step
from ._window_support import RpcRequest, show_sample_section


def test_fixed_element_form_rpc_keeps_indexes_in_descriptions_and_preview(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    rpc: RpcRequest,
) -> None:
    from pydantic import create_model

    from tests.gui.form_adapter._fixed_models import named_form, named_input

    window, _harness, _entry = attachment
    model = named_form()
    plan = create_model(
        "FixedElementPlan",
        __base__=PlanForm,
        tones=(
            model.model_fields["tones"].rebuild_annotation(),
            named_input(model).data["tones"],
        ),
        participants=(model.model_fields["participants"].rebuild_annotation(), ["B"]),
    )
    catalog_source.catalog = WorkflowCatalog(
        (
            WorkflowAttachment(
                step=sample_step,
                plan_model=plan,
                knobs_model=TunablesForm,
                plan_to_cfg=plan.to_cfg,
                knobs_to_cfg=TunablesForm.to_cfg,
                seed=context_seed,
            ),
        )
    )
    window.reload_source()
    reply = rpc("workflow.describe_form", {"form": "plan", "path": ["tones"]})
    assert reply["ok"], reply
    result = reply["result"]
    assert isinstance(result, dict)
    target = result["description"]["target"]
    assert "elements" in target, (
        "Fixed-list description must expose original positions and names"
    )
    assert target["elements"] == [
        {"index": 0, "name": "A", "path": ["tones", 0]},
        {"index": 1, "name": "B", "path": ["tones", 1]},
    ]
    reply = rpc(
        "workflow.describe_form",
        {
            "form": "plan",
            "path": ["tones", 1, "gain"],
            "batch": [{"path": ["tones", 1, "gain"], "value": {"expr": "3 * 2"}}],
        },
    )
    assert reply["ok"], reply
    result = reply["result"]
    assert isinstance(result, dict)
    assert result["description"]["target"]["path"] == ["tones", 1, "gain"]
    assert {tuple(item["path"]): item["value"] for item in result["values"]}[
        ("tones", 1, "gain")
    ] == 6
    assert (
        window.plan_panel.state.input.data["tones"] == named_input(model).data["tones"]
    )


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


def test_form_inspection_rpc_plan_batch_does_not_change_gui(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
) -> None:
    from zcu_tools.gui.form import FormInput

    window, _harness, _entry = attachment
    state = window.plan_panel.state
    data = state.input.data
    data["samples"] = {"kind": "linrange", "start": 0.0, "stop": 1.0, "n": 3}
    state.reset(FormInput(PlanForm, data))
    before = state.input.data, state.base.data
    reply = rpc(
        "workflow.form",
        {
            "form": "plan",
            "batch": [
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
    show_sample_section(window, "Tunables (live)")
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
def test_describe_form_rpc_uses_call_local_edits(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    rpc: RpcRequest,
    path: list[str],
) -> None:
    from zcu_tools.gui.form import FormInput

    window, _harness, _entry = attachment
    state = window.plan_panel.state
    data = state.input.data
    data["samples"] = {"kind": "linrange", "start": 0.0, "stop": 1.0, "n": 3}
    state.reset(FormInput(PlanForm, data))
    before = state.input.data, state.base.data
    reply = rpc(
        "workflow.describe_form",
        {
            "form": "plan",
            "path": path,
            "batch": [
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
            ["scale"],
            ["points"],
            ["samples"],
        ]
    elif path == ["samples"]:
        target = description["target"]
        assert target["options"][target["selected"]]["value"] == {"kind": "linrange"}
        assert {tuple(child["path"]) for child in description["children"]} == {
            ("samples", "start"),
            ("samples", "stop"),
            ("samples", "n"),
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
    show_sample_section(window, "Tunables (live)")
    local = panel.panel.editor.findChild(QLineEdit, "form.input.sample.factor")
    assert isinstance(local, QLineEdit)
    commit_text(local, "-")
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.value", 99.0)
    before = (
        host.state.input.data,
        host.state.base.data,
        host.accepted_input.data,
        engine.tunables("tuning"),
    )
    for batch in (None, []):
        reply = rpc(
            "workflow.describe_form",
            {
                "form": "tunables",
                "run_id": "tuning",
                "path": ["sample", "value"],
                "batch": batch,
            },
        )
        assert reply["ok"] is True
        result = reply["result"]
        assert isinstance(result, dict)
        assert result["description"]["target"]["constraints"]["gt"] == 0
        assert result["revision"] == before[3].revision
        assert {"path": ["sample", "value"], "value": 40.0} in result["values"]
        assert any(e["path"] == ["sample", "factor"] for e in result["errors"]) == (
            batch is None
        )
    assert (
        host.state.input.data,
        host.state.base.data,
        host.accepted_input.data,
        engine.tunables("tuning"),
    ) == before
    assert local.text() == "-"


@pytest.mark.parametrize("run_id", [None, "stale"])
def test_describe_form_rpc_requires_explicit_attached_target(
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    run_id: str | None,
) -> None:
    del paused_tuning
    reply = rpc(
        "workflow.describe_form", {"form": "plan", "path": [], "run_id": run_id}
    )
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict) and error["code"] == "precondition_failed"


def test_describe_form_rpc_forwards_structural_expandability(
    rpc: RpcRequest,
) -> None:
    reply = rpc("workflow.describe_form", {"form": "plan", "path": []})
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    children = result["description"]["children"]
    assert [(child["name"], child.get("can_expand")) for child in children] == [
        ("scale", False),
        ("points", False),
        ("samples", True),
    ]


@pytest.mark.parametrize(
    "path", [["unknown"], ["samples", "start"], ["samples", "values", "item"]]
)
@pytest.mark.parametrize("caller", ["rpc", "mcp"])
def test_describe_form_separates_path_rejection_from_declaration_failure(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    rpc: RpcRequest,
    mcp_tools: ToolTable,
    path: list[str],
    caller: str,
) -> None:
    window, _harness, _entry = attachment
    fail_declaration: list[bool] = []

    class FaultyPlan(PlanForm):
        @classmethod
        def choices(cls, env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            if fail_declaration:
                raise ValueError("Synthetic declaration failure")
            return super().choices(env)

    catalog_source.catalog = WorkflowCatalog(
        (
            WorkflowAttachment(
                step=sample_step,
                plan_model=FaultyPlan,
                knobs_model=TunablesForm,
                plan_to_cfg=FaultyPlan.to_cfg,
                knobs_to_cfg=TunablesForm.to_cfg,
                seed=context_seed,
            ),
        )
    )
    window.reload_source()
    before = window.plan_panel.state.input.data, window.plan_panel.state.base.data
    params: dict[str, object] = {"form": "plan", "path": path}
    with pytest.raises(FormPathError) as core_error:
        window.plan_panel.state.describe(tuple(path), window.plan_panel.env)
    expected = (
        "invalid_params",
        str(core_error.value),
        "form_path",
        {"form_path": path},
    )
    if caller == "rpc":
        reply = rpc("workflow.describe_form", params)
        assert reply["ok"] is False
        error = reply["error"]
        assert isinstance(error, dict)
        assert (
            error["code"],
            error["message"],
            error.get("reason"),
            error.get("data"),
        ) == expected
    else:
        with pytest.raises(GuiInvocationError) as mcp_error:
            mcp_tools["workflows_workflow_describe_form"]["handler"](params)
        invocation = mcp_error.value
        assert (
            invocation.code,
            invocation.message,
            invocation.reason,
            invocation.data,
        ) == expected
    fail_declaration.append(True)
    try:
        if caller == "rpc":
            reply = rpc("workflow.describe_form", {"form": "plan", "path": []})
            assert reply["ok"] is False
            error = reply["error"]
            assert isinstance(error, dict)
            assert error["code"] == "controller_error"
            assert "Synthetic declaration failure" in error["message"]
            assert error.get("reason") is None
            assert error.get("data") is None
        else:
            with pytest.raises(GuiInvocationError) as mcp_error:
                mcp_tools["workflows_workflow_describe_form"]["handler"](
                    {"form": "plan", "path": []}
                )
            invocation = mcp_error.value
            assert invocation.code == "controller_error"
            assert "Synthetic declaration failure" in invocation.message
            assert invocation.reason is None
            assert invocation.data is None
    finally:
        fail_declaration.clear()
    assert (
        window.plan_panel.state.input.data,
        window.plan_panel.state.base.data,
    ) == before
    assert window.controller.binding is None


def test_describe_form_projects_fixed_branch_without_freezing_inner_selection(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    rpc: RpcRequest,
) -> None:
    window, _harness, _entry = attachment

    class First(FormModel):
        kind: Literal["first"] = "first"
        value: float = 1.0

    class Second(FormModel):
        kind: Literal["second"] = "second"
        value: float = 2.0

    class Inline(FormModel):
        kind: Literal["inline"] = "inline"
        module: Annotated[First | Second, Field(discriminator="kind")] = Field(
            default_factory=First
        )

    class FixedPlan(PlanForm):
        source: Inline = Field(default_factory=Inline)
        tag: Literal["only"] = "only"

        @classmethod
        def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
            return {
                ("source",): (
                    Option(
                        "First",
                        {"kind": "inline", "module": {"kind": "first"}},
                        fields=Inline,
                    ),
                    Option(
                        "Second",
                        {"kind": "inline", "module": {"kind": "second"}},
                        fields=Inline,
                    ),
                )
            }

    catalog_source.catalog = WorkflowCatalog(
        (
            WorkflowAttachment(
                step=sample_step,
                plan_model=FixedPlan,
                knobs_model=TunablesForm,
                plan_to_cfg=FixedPlan.to_cfg,
                knobs_to_cfg=TunablesForm.to_cfg,
                seed=context_seed,
            ),
        )
    )
    window.reload_source()
    reply = rpc("workflow.describe_form", {"form": "plan", "path": []})
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    rows = {row["name"]: row for row in result["description"]["children"]}
    assert (
        rows["source"].get("selection_fixed"),
        rows["source"]["fixed"],
        rows["source"]["fixed_value"],
    ) == (True, False, None)
    assert (
        rows["tag"].get("selection_fixed"),
        rows["tag"]["fixed"],
        rows["tag"]["fixed_value"],
    ) == (True, True, "only")
    reply = rpc("workflow.describe_form", {"form": "plan", "path": ["source"]})
    assert reply["ok"] is True
    result = reply["result"]
    assert isinstance(result, dict)
    description = result["description"]
    assert description["target"]["selection_fixed"] is True
    assert description["target"]["fixed"] is False
    assert [
        (row["name"], row["selection_fixed"], row["fixed"])
        for row in description["children"]
    ] == [("module", False, False)]
