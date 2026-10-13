"""Different registered workflows share one owner and retain independent drafts."""

import pytest
from pydantic import BaseModel, ConfigDict
from qtpy.QtWidgets import QComboBox
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import Done, Next, Step, WorkflowEnv, workflow
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import FormEnv, FormModel
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.resources.entry import ResultEntry

from ._reload_source import InjectedCatalog
from ._start_fakes import StartHarness
from ._synthetic import Context, build_registry


class AlternatePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    batches: int = 2


class AlternateTunables(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float = 12.0


class AlternatePlanForm(FormModel):
    batches: int = 2

    @staticmethod
    def to_cfg(
        form: "AlternatePlanForm", _env: FormEnv, _snapshot: RunSnapshot
    ) -> AlternatePlan:
        return AlternatePlan(batches=form.batches)


class AlternateTunablesForm(FormModel):
    value: float = 12.0

    def to_cfg(self) -> AlternateTunables:
        return AlternateTunables(value=self.value)


@workflow(
    "alternate",
    plan=AlternatePlan,
    tunables=AlternateTunables,
    init=lambda _env, _plan: 0,
)
def alternate(
    env: WorkflowEnv[Context], plan: AlternatePlan, tun: AlternateTunables, state: int
) -> Step[int, float]:
    del env
    yield from ()
    if state == plan.batches:
        return Done()
    return Next(tun.value, state + 1)


@pytest.fixture
def selections(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
) -> WorkflowWindow[Context]:
    from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog

    window, _harness, _entry = attachment
    catalog_source.catalog = WorkflowCatalog(
        (
            *build_registry(),
            WorkflowAttachment(
                step=alternate,
                plan_model=AlternatePlanForm,
                knobs_model=AlternateTunablesForm,
                plan_to_cfg=AlternatePlanForm.to_cfg,
                knobs_to_cfg=AlternateTunablesForm.to_cfg,
                seed=lambda _plan, _form: None,
            ),
        )
    )
    window.reload_source()
    return window


@pytest.mark.parametrize("starter", ["gui", "mcp"])
def test_selected_workflow_gui_and_mcp_commit_on_the_fixed_controller(
    selections: WorkflowWindow[Context],
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    mcp_tools: ToolTable,
    starter: str,
) -> None:
    window = selections
    _window, harness, _entry = attachment
    controller = window.controller
    window.select_workflow("alternate")
    selector = window.findChild(QComboBox, "workflow.select")
    assert selector is not None and selector.currentText() == "alternate"
    if starter == "gui":
        token = window.start("selected")
    else:
        token = mcp_tools["workflows_workflow_start"]["handler"](
            {"run_id": "selected", "name": "alternate", "plan": [], "tunables": []}
        )["operation_id"]
    assert isinstance(token, int)
    harness.complete_segment()
    assert window.controller is controller
    status = controller.status()
    assert status is not None and status.lifecycle == "done"
    assert status.workflow == "alternate" and status.committed_seq == 2
    engine = controller.engine
    assert engine is not None
    assert [record.encoded_record.value for record in engine.records("selected")] == [
        12.0,
        12.0,
    ]


def test_explicit_unknown_mcp_start_keeps_selection_and_drafts(
    selections: WorkflowWindow[Context], mcp_tools: ToolTable
) -> None:
    window = selections
    window.select_workflow("alternate")
    before = window.capture_persisted_state()
    with pytest.raises(RuntimeError, match="invalid_params.*unknown_workflow"):
        mcp_tools["workflows_workflow_start"]["handler"](
            {
                "run_id": "unknown",
                "name": "missing-workflow",
                "plan": [],
                "tunables": [],
            }
        )
    assert window.workflow_name == "alternate"
    assert window.capture_persisted_state() == before
    assert window.controller.run_id is None


def test_switching_workflows_preserves_separate_parsed_drafts(
    selections: WorkflowWindow[Context],
) -> None:
    window = selections
    window.plan_panel.state.write(
        {("scale",): {"expr": "1 / 3"}}, window.plan_panel.env
    )
    window.select_workflow("alternate")
    window.plan_panel.state.write({("batches",): 7}, window.plan_panel.env)
    window.tunables_panel.state.write(
        {("value",): {"expr": "3 * 4"}}, window.tunables_panel.env
    )
    window.select_workflow("sample")
    assert window.capture_persisted_state().plan["scale"] == {"expr": "1 / 3"}
    window.select_workflow("alternate")
    saved = window.capture_persisted_state()
    assert saved.plan["batches"] == 7
    assert saved.knobs["value"] == {"expr": "3 * 4"}


def test_switching_after_terminal_keeps_the_edited_start_tunables(
    selections: WorkflowWindow[Context],
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
) -> None:
    window = selections
    _window, harness, _entry = attachment
    window.select_workflow("alternate")
    assert isinstance(window.start("finished"), int)
    harness.complete_segment()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "done"

    window.tunables_panel.state.write(
        {("value",): {"expr": "3 * 5"}}, window.tunables_panel.env
    )
    window.select_workflow("sample")
    window.select_workflow("alternate")
    assert window.tunables_panel.state.input.data["value"] == {"expr": "3 * 5"}
