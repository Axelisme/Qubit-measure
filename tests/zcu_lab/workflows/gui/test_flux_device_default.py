"""Coherence direct device defaults are captured once, with explicit invalid aliases."""

import pytest
from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit
from zcu_tools.gui.form import FormEnv, FormInput, FormState
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from zcu_lab.workflows.gui.plan import CoherencePlanForm


def captured_device(name: str) -> FormEnv:
    context = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="1.0",
            general=PointGeneral(created_at="2026-10-11T00:00:00Z"),
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        ),
        target=SavedPoint(entry_id="entry", point="synthetic"),
        roles=None,
        storage=None,
    ).bind_aliases((AliasRow("flux_dev", name),), {})
    return FormEnv(parameters=context)


def test_direct_default_keeps_capture_and_expression_remains_editable(
    qapp: QApplication,
) -> None:
    env = captured_device("yoko_history")
    state = FormState(CoherencePlanForm.default_input(env))
    editor = FormEditor(state, env)
    try:
        mode = editor.findChild(QComboBox, "form.mode.flux_dev")
        text = editor.findChild(QLineEdit, "form.input.flux_dev")
        assert isinstance(mode, QComboBox) and isinstance(text, QLineEdit)
        assert mode.currentData() == "direct" and text.text() == "yoko_history"
        assert state.preview(env).values[("flux_dev",)] == "yoko_history"
        new_env = captured_device("yoko_new")
        assert state.preview(new_env).values[("flux_dev",)] == "yoko_history"
        fresh = FormState(CoherencePlanForm.default_input(new_env))
        assert fresh.preview(new_env).values[("flux_dev",)] == "yoko_new"
        mode.setCurrentIndex(mode.findData("expr"))
        text.setText("$flux_dev")
        assert state.preview(env).values[("flux_dev",)] == "yoko_history"
        assert state.input.data["flux_dev"] == {"expr": "$flux_dev"}
        restored = FormState(CoherencePlanForm.default_input(new_env))
        restored.write_compatible(
            FormInput.from_json(CoherencePlanForm, state.input.to_json()).data, new_env
        )
        assert restored.preview(new_env).values[("flux_dev",)] == "yoko_new"
        assert restored.input.data["flux_dev"] == {"expr": "$flux_dev"}
    finally:
        editor.close()
        editor.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("missing_capture", [True, False])
def test_missing_alias_builds_editor_with_named_invalid_field(
    qapp: QApplication, missing_capture: bool
) -> None:
    env = FormEnv() if missing_capture else captured_device("")
    state = FormState(CoherencePlanForm.default_input(env))
    editor = FormEditor(state, env)
    try:
        errors = editor.errors()
        assert errors is not None and "flux_dev" in errors.messages[("flux_dev",)]
        error = editor.findChild(QLabel, "form.error.flux_dev")
        assert isinstance(error, QLabel) and "flux_dev" in error.text()
        assert ("flux_dev",) not in state.preview(env).values
    finally:
        editor.close()
        editor.deleteLater()
        qapp.processEvents()
