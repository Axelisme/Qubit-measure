"""The T1 host attachment uses the shared draft and parse-error interface."""

from collections.abc import Iterator

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormState
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from tests.zcu_lab.workflows._sim_inputs import Inputs, make_plan
from zcu_lab.components import QubitSchema
from zcu_lab.workflows.gui.panel import T1PlanPanel, T1TunablesPanel
from zcu_lab.workflows.gui.plan import T1PlanForm
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.t1 import T1Plan

from ._plan_inputs import inline_input


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Use the process QApplication or create one for this Qt attachment."""
    app = QApplication.instance()
    if app is None:
        return QApplication([])
    if not isinstance(app, QApplication):
        raise RuntimeError("Panel tests require a QApplication")
    return app


@pytest.fixture
def panels(qapp: QApplication) -> Iterator[list[T1TunablesPanel]]:
    """Delete each panel on the existing QApplication after this test."""
    owned: list[T1TunablesPanel] = []
    yield owned
    for panel in owned:
        panel.deleteLater()
    qapp.processEvents()


def environment(guess_us: float) -> FormEnv:
    """Capture the lifetime guess in a real point's extension namespace."""
    return FormEnv(
        parameters=ParameterContext(
            point=PointDocument(
                format="zcu.parameter-container",
                format_version="1.0",
                general=PointGeneral(
                    created_at="2026-10-08T00:00:00Z", ext={"guess_us": guess_us}
                ),
            ),
            library=LibrarySnapshot(
                ModuleLibraryDocument(
                    format="zcu.module-library",
                    format_version="1.0",
                )
            ),
            target=EntryDraft(entry_id="entry"),
            roles=None,
        )
    )


def test_t1_panel_presents_host_state_without_replacing_draft(
    panels: list[T1TunablesPanel],
) -> None:
    env = environment(30.0)
    state = FormState(T1TunablesForm.default_input(env))
    state.write({("guess_us",): {"expr": "general.ext.guess_us + 2"}}, env)
    panel = T1TunablesPanel(state, env)
    panels.append(panel)
    assert panel.state is state
    assert state.input.data["guess_us"] == {"expr": "general.ext.guess_us + 2"}
    result = panel.state.result(env)
    assert isinstance(result, T1TunablesForm)
    assert result.guess_us == 32.0
    assert panel.editor.local_parse_errors() is None
    assert panel.editor.errors() is None


def test_t1_panel_exposes_generic_local_parse_errors_and_host_reset(
    panels: list[T1TunablesPanel],
) -> None:
    env = environment(30.0)
    state = FormState(T1TunablesForm.default_input(env))
    panel = T1TunablesPanel(state, env)
    panels.append(panel)
    field = panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    assert isinstance(field, QLineEdit)
    field.setText("-")
    result = panel.state.result(env)
    assert isinstance(result, T1TunablesForm)
    errors = panel.editor.local_parse_errors()
    assert isinstance(errors, FormErrors)
    assert ("sweep_factor",) in errors.messages

    field.setText("8")
    result = panel.state.result(env)
    assert isinstance(result, T1TunablesForm)
    assert result.sweep_factor == 8.0
    host_result = state.result(env)
    assert isinstance(host_result, T1TunablesForm)
    assert host_result.sweep_factor == 8.0
    assert panel.editor.local_parse_errors() is None
    updated_env = environment(40.0)
    panel.state.reset(T1TunablesForm.default_input(updated_env))
    panel.editor.refresh(updated_env)
    result = panel.state.result(updated_env)
    assert isinstance(result, T1TunablesForm)
    field = panel.editor.findChild(QLineEdit, "form.input.sweep_factor")
    assert isinstance(field, QLineEdit)
    assert result.guess_us == 40.0
    assert float(field.text()) == result.sweep_factor


def test_plan_panel_previews_native_plan_and_blocks_incomplete_text(
    qapp: QApplication,
    inputs: Inputs,
) -> None:
    env = FormEnv()
    template = make_plan(inputs)
    panel = T1PlanPanel(env)
    try:
        panel.state.write({("flux_dev",): inline_input(inputs).data["flux_dev"]}, env)
        missing = panel.preview(template.snapshot)
        assert isinstance(missing, FormErrors)
        assert ("pi_length_us",) in missing.messages
        panel.state.reset(inline_input(inputs))
        panel.editor.refresh(env)
        preview = panel.preview(template.snapshot)
        assert isinstance(preview, T1Plan)
        assert preview == template
        field = panel.editor.findChild(QLineEdit, "form.input.points")
        assert isinstance(field, QLineEdit)
        field.setText("-")
        errors = panel.preview(template.snapshot)
        assert isinstance(errors, FormErrors)
        assert ("points",) in errors.messages
        field.setText("31")
        repaired = panel.preview(template.snapshot)
        assert isinstance(repaired, T1Plan)
        assert repaired.points == 31
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("softlink", [False, True])
def test_plan_panel_resolves_default_device_alias_in_fixed_preview(
    qapp: QApplication, inputs: Inputs, softlink: bool
) -> None:
    parameters = environment(30.0).parameters
    assert parameters is not None
    point = parameters.document()
    point.components["Q1"] = QubitSchema(
        kind="qubit/test", flux_source="offline_device"
    )
    parameters = ParameterContext(
        point=point, library=parameters.library, target=parameters.target, roles=None
    )
    rows = (
        (AliasRow("qubit", "Q1"), AliasRow("flux_dev", "$qubit.flux_source"))
        if softlink
        else (AliasRow("flux_dev", "offline_device"),)
    )
    captured = parameters.bind_aliases(rows, {})
    env = FormEnv(parameters=captured)
    panel = T1PlanPanel(env)
    try:
        # Supply acquisition choices only; the device comes from form defaults.
        choices = inline_input(inputs).data
        del choices["flux_dev"]
        data = panel.state.input.data
        data.update(choices)
        panel.state.reset(FormInput(T1PlanForm, data))
        panel.editor.refresh(env)
        preview = panel.preview(make_plan(inputs).snapshot)
        assert isinstance(preview, T1Plan), preview
        assert preview.flux_dev == "offline_device"  # Not in any device catalog.
        assert panel.state.input.data["flux_dev"] == {"expr": "$flux_dev"}
        shown = panel.editor.findChild(QLabel, "form.value.flux_dev")
        assert shown is not None and "offline_device" in shown.text()
        later = parameters.bind_aliases((AliasRow("flux_dev", "later_device"),), {})
        changed = panel.state.result(FormEnv(parameters=later))
        assert isinstance(changed, T1PlanForm) and changed.flux_dev == "later_device"
        again = panel.preview(make_plan(inputs).snapshot)
        assert isinstance(again, T1Plan) and again.flux_dev == "offline_device"
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("binding", ["missing", "unbound", "source-free"])
def test_plan_default_device_alias_is_invalid_without_a_binding(
    inputs: Inputs, binding: str
) -> None:
    parameters = environment(30.0).parameters
    assert parameters is not None
    env = (
        FormEnv()
        if binding == "source-free"
        else FormEnv(
            parameters=parameters.bind_aliases(
                () if binding == "missing" else (AliasRow("flux_dev", ""),), {}
            )
        )
    )
    choices = inline_input(inputs).data
    del choices["flux_dev"]
    data = T1PlanForm.default_input(env).data
    data.update(choices)
    state = FormState(FormInput(T1PlanForm, data))
    preview = state.preview(env)
    assert preview.errors is not None
    assert ("flux_dev",) not in preview.values
    assert "alias" in preview.errors.messages[("flux_dev",)].lower()
    assert "flux_dev" in preview.errors.messages[("flux_dev",)]
    assert state.input.data["flux_dev"] == {"expr": "$flux_dev"}
