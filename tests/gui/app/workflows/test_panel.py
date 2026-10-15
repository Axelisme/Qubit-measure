"""Registered Window panels retain drafts, parse errors and native previews."""

from qtpy.QtWidgets import QLineEdit
from zcu_tools.datafile import RunSnapshot
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import FormErrors
from zcu_tools.resources.entry import ResultEntry

from tests.gui.widgets.form._text_input import commit_text

from ._start_fakes import StartHarness
from ._synthetic import Context, Plan, PlanForm, TunablesForm, inline_input


def test_panel_presents_host_state_without_replacing_draft(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
) -> None:
    window, _harness, _entry = attachment
    panel = window.tunables_panel
    state, env = panel.state, panel.env
    state.write({("sample", "value"): {"expr": "general.ext.value + 2"}}, env)
    assert panel.state is state
    assert state.input.data["sample"] == {
        "value": {"expr": "general.ext.value + 2"},
        "count": 100,
        "factor": 4.0,
    }
    result = panel.state.result(env)
    assert isinstance(result, TunablesForm) and result.sample.value == 22.0
    assert panel.editor.local_parse_errors() is None
    assert panel.editor.errors() is None


def test_panel_exposes_generic_local_parse_errors_and_host_reset(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
) -> None:
    window, harness, _entry = attachment
    panel = window.tunables_panel
    field = panel.editor.findChild(QLineEdit, "form.input.sample.factor")
    assert isinstance(field, QLineEdit)
    commit_text(field, "-")
    assert isinstance(panel.state.result(panel.env), TunablesForm)
    errors = panel.editor.local_parse_errors()
    assert isinstance(errors, FormErrors) and ("sample", "factor") in errors.messages
    commit_text(field, "8")
    result = panel.state.result(panel.env)
    assert isinstance(result, TunablesForm) and result.sample.factor == 8.0
    assert panel.editor.local_parse_errors() is None
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.value", 40.0)
    window.prepare_forms()
    updated = window.tunables_panel
    result = updated.state.result(updated.env)
    assert isinstance(result, TunablesForm) and result.sample.value == 40.0
    field = updated.editor.findChild(QLineEdit, "form.input.sample.factor")
    assert isinstance(field, QLineEdit) and float(field.text()) == result.sample.factor


def test_plan_panel_previews_native_plan_and_blocks_incomplete_text(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
) -> None:
    window, harness, _entry = attachment
    storage = harness.services.context.capture_storage(
        harness.services.context.read_context()
    )
    snapshot = RunSnapshot(
        entry_id=storage.entry_id,
        entry_name=storage.entry_name,
        point=storage.point,
        description=storage.description,
        roles=storage.roles,
    )
    panel, env = window.plan_panel, window.plan_panel.env
    panel.state.reset(PlanForm.default_input(env))
    missing = panel.preview(snapshot)
    assert isinstance(missing, FormErrors) and ("scale",) in missing.messages
    panel.state.reset(inline_input())
    panel.editor.refresh(env)
    preview = panel.preview(snapshot)
    assert isinstance(preview, Plan)
    assert preview.points == 21 and preview.samples == (1.0, 1.1)
    field = panel.editor.findChild(QLineEdit, "form.input.points")
    assert isinstance(field, QLineEdit)
    commit_text(field, "-")
    errors = panel.preview(snapshot)
    assert isinstance(errors, FormErrors) and ("points",) in errors.messages
    commit_text(field, "31")
    repaired = panel.preview(snapshot)
    assert isinstance(repaired, Plan) and repaired.points == 31
