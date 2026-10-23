"""Inspect drafts reject external file revisions through real owner seams.

All scenario interactions and observations use ComponentDriver. External edits
model an editor writing the temporary point files, not a replacement owner.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


class _NoArb:
    def list_data_keys(self) -> list[str]:
        return []


@pytest.fixture
def registered_components() -> Generator[None]:
    before = registry_state()
    try:
        register_fakes()
        yield
    finally:
        restore_registry(before)


@pytest.fixture
def conflict_ui(
    qapp: QApplication,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    registered_components: None,
) -> Generator[
    tuple[ComponentDriver, Path, Path, ContextService, InspectDialog]
]:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "Database"
    )
    entry.new_point("point")
    with entry.module_library("point").edit() as draft:
        draft.set_waveform("Q1.waveforms.flat", ConstWaveformCfg(length=0.1))
    bus = BaseEventBus()
    service = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus,
        bind_assets=ArbWaveformDatabase.info_at,
    )
    service.install_entry(entry)
    service.use_point(PointUseRequest("point"))
    control = ContextControlFacet(context=service)
    control.add_component("Q1", "fake/drive/a", {"rate": 4300.0})
    control.add_component("Q1", "fake/drive/a", {"rate": 1200.0}, setup=True)
    entry.new_point("second")
    service.use_point(PointUseRequest("second"))
    control.set_parameter(control.read_context().target, "Q1.rate", 5300.0)
    service.use_point(PointUseRequest("point"))
    # The missing-file case deliberately leaves aliases revision=None.
    document = request.param
    if document != "aliases-missing":
        control.set_alias_rows((AliasRow("qubit", "Q1"),))
    path = next(
        file.path
        for file in control.get_file_revisions()
        if file.document == document.removesuffix("-missing")
    )
    dialog = InspectDialog(
        control,
        bus,
        templates={},
        arb_waveforms=_NoArb(),
        open_arb_waveforms=lambda: None,
        dialog_presenter=RecordingDialogPresenter(),
    )
    dialog.resize(1000, 760)
    dialog.show()
    dialog.activateWindow()
    qapp.processEvents()
    evidence = Path(os.environ.get("INSPECT_SYNC_EVIDENCE", str(tmp_path))) / document
    try:
        yield ComponentDriver(dialog), path, evidence, service, dialog
    finally:
        dialog.hide()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("conflict_ui", ["library"], indirect=True)
def test_library_keep_cannot_rebase_stale_draft(conflict_ui) -> None:
    ui, path, evidence, *_ = conflict_ui
    ui.select_tab("Modules")
    ui.select_tree_row(
        ("waveforms", "Q1", "waveforms", "flat"), object_name="inspect.library"
    )
    ui.enter_text(
        object_name="inspect.entry.name", value="Q1.waveforms.draft", commit=False
    )
    assert ui.text_input(object_name="inspect.entry.name").text == "Q1.waveforms.draft"
    assert ui.observe(object_name="inspect.entry.status").text == "Unsaved"
    ui.screenshot(evidence / "01-draft.png")
    path.write_text(path.read_text().replace("0.1", "0.7"))
    ui.click(object_name="inspect.entry.apply")
    assert ui.observe(object_name="inspect.entry.status").text == "Unsaved"
    assert ui.observe(object_name="inspect.library.sync.message").visible
    ui.screenshot(evidence / "02-rejected.png")
    ui.click(object_name="inspect.library.sync.keep")
    ui.click(object_name="inspect.entry.apply")
    assert ui.observe(object_name="inspect.library.sync.message").visible
    assert ui.text_input(object_name="inspect.entry.name").text == "Q1.waveforms.draft"
    ui.screenshot(evidence / "03-keep-rejected.png")
    ui.click(object_name="inspect.library.sync.reload")
    assert ui.observe(object_name="inspect.entry.status").text == "Saved"
    assert ui.text_input(object_name="inspect.entry.name").text == "Q1.waveforms.flat"
    assert "0.7" in ui.text_input(object_name="form.input.length").text
    ui.screenshot(evidence / "04-reloaded.png")


@pytest.mark.parametrize("conflict_ui", ["aliases", "aliases-missing"], indirect=True)
def test_alias_draft_revision_includes_missing_file(conflict_ui) -> None:
    ui, path, evidence, *_ = conflict_ui
    ui.select_tab("Aliases")
    initial_count = len(ui.table_rows(object_name="inspect.aliases.table"))
    ui.click(object_name="inspect.aliases.add")
    ui.type_text("x")
    ui.press_key(Qt.Key.Key_Return)
    ui.screenshot(evidence / "01-draft.png")
    path.write_text(
        "format: zcu.aliases\nformat_version: '1.0'\naliases:\n  external: Q2\n"
    )
    ui.click(object_name="inspect.aliases.apply")
    assert ui.observe(object_name="inspect.aliases.error").text
    assert len(ui.table_rows(object_name="inspect.aliases.table")) == initial_count + 1
    ui.screenshot(evidence / "02-rejected.png")
    ui.click(object_name="inspect.aliases.sync.keep")
    ui.click(object_name="inspect.aliases.apply")
    assert ui.observe(object_name="inspect.aliases.sync.message").visible
    ui.screenshot(evidence / "03-keep-rejected.png")
    ui.click(object_name="inspect.aliases.sync.reload")
    assert ui.table_rows(object_name="inspect.aliases.table")[0].columns[:2] == (
        "external",
        "Q2",
    )
    ui.screenshot(evidence / "04-reloaded.png")


@pytest.mark.parametrize("conflict_ui", ["parameters"], indirect=True)
@pytest.mark.parametrize("submit", ["enter", "focus-loss"])
def test_parameter_rejection_retains_visible_draft(conflict_ui, submit: str) -> None:
    ui, path, evidence, *_ = conflict_ui
    evidence = evidence / submit
    ui.enter_text(
        object_name="inspect.parameters.filter", value="Q1.rate", commit=False
    )
    ui.edit_tree_value(
        ("Q1", "rate"), object_name="component_tree", value="9", commit=False
    )
    assert ui.focused_input().text == "9"
    ui.screenshot(evidence / "01-draft.png")
    path.write_text(path.read_text().replace("4300.0", "4700.0"))
    if submit == "enter":
        ui.press_key(Qt.Key.Key_Return)
    else:
        ui.choose_option("Current point", value="Current point")
    assert ui.observe(object_name="component_error").text
    rows = ui.visible().tree_rows
    assert any(row.path == ("Q1", "rate") and row.columns[1] == "9" for row in rows)
    ui.screenshot(evidence / "02-rejected.png")
    ui.click(object_name="inspect.parameters.sync.keep")
    ui.edit_tree_value(("Q1", "rate"), object_name="component_tree", value="8")
    assert ui.observe(object_name="inspect.parameters.sync.message").visible
    ui.screenshot(evidence / "03-keep-rejected.png")
    ui.click(object_name="inspect.parameters.sync.reload")
    rows = ui.visible().tree_rows
    assert any(
        row.path == ("Q1", "rate") and row.columns[1] == "4700.0" for row in rows
    )
    ui.screenshot(evidence / "04-reloaded.png")


@pytest.mark.parametrize("conflict_ui", ["parameters"], indirect=True)
def test_parameter_draft_cannot_follow_destination(conflict_ui) -> None:
    ui, _, evidence, *_ = conflict_ui
    evidence = evidence / "destination"
    ui.enter_text(
        object_name="inspect.parameters.filter", value="Q1.rate", commit=False
    )
    ui.edit_tree_value(("Q1", "rate"), object_name="component_tree", value="oops")
    assert ui.observe(object_name="component_error").text
    ui.screenshot(evidence / "01-rejected.png")
    ui.choose_option("Current point", value="Setup template")
    ui.edit_tree_value(("Q1", "rate"), object_name="component_tree", value="8")
    assert ui.observe(object_name="component_error").text
    assert any(
        row.path == ("Q1", "rate") and row.columns[1] == "8"
        for row in ui.visible().tree_rows
    )
    ui.screenshot(evidence / "02-target-rejected.png")
    ui.choose_option("Setup template", value="Current point")
    ui.click(object_name="inspect.parameters.sync.reload")
    assert any(
        row.path == ("Q1", "rate") and row.columns[1] == "4300.0"
        for row in ui.visible().tree_rows
    )
    ui.choose_option("Current point", value="Setup template")
    assert any(
        row.path == ("Q1", "rate") and row.columns[1] == "1200.0"
        for row in ui.visible().tree_rows
    )
    ui.screenshot(evidence / "03-setup-unchanged.png")


@pytest.mark.parametrize("conflict_ui", ["parameters"], indirect=True)
def test_parameter_draft_cannot_follow_context(conflict_ui) -> None:
    ui, _, evidence, source, _ = conflict_ui
    evidence = evidence / "context"
    ui.enter_text(
        object_name="inspect.parameters.filter", value="Q1.rate", commit=False
    )
    ui.edit_tree_value(
        ("Q1", "rate"), object_name="component_tree", value="9", commit=False
    )
    ui.screenshot(evidence / "01-draft.png")
    # An external context switch is not a UI action in this scenario.
    source.use_point(PointUseRequest("second"))
    assert ui.focused_input().text == "9"
    ui.press_key(Qt.Key.Key_Return)
    assert ui.observe(object_name="component_error").text
    assert any(
        row.path == ("Q1", "rate") and row.columns[1] == "9"
        for row in ui.visible().tree_rows
    )
    ui.screenshot(evidence / "02-context-rejected.png")
    ui.click(object_name="inspect.parameters.sync.reload")
    assert any(
        row.path == ("Q1", "rate") and row.columns[1] == "5300.0"
        for row in ui.visible().tree_rows
    )
    ui.screenshot(evidence / "03-second-unchanged.png")


@pytest.mark.parametrize("conflict_ui", ["parameters"], indirect=True)
def test_parameter_escape_and_multiple_retained_cells(conflict_ui) -> None:
    ui, path, evidence, source, _ = conflict_ui
    evidence = evidence / "multiple"
    ui.enter_text(object_name="inspect.parameters.filter", value="Q1", commit=False)
    ui.edit_tree_value(
        ("Q1", "rate"), object_name="component_tree", value="9", commit=False
    )
    ui.press_key(Qt.Key.Key_Escape)
    assert any(
        row.path == ("Q1", "rate") and row.columns[1] == "4300.0"
        for row in ui.visible().tree_rows
    )
    ui.screenshot(evidence / "01-escape.png")
    ui.edit_tree_value(
        ("Q1", "rate"), object_name="component_tree", value="9", commit=False
    )
    path.write_text(path.read_text().replace("4300.0", "4700.0"))
    source.synchronize_files(admit_reload=False)
    assert ui.focused_input().text == "9"
    ui.screenshot(evidence / "02-pending-active.png")
    ui.press_key(Qt.Key.Key_Return)
    ui.edit_tree_value(("Q1", "energy_a"), object_name="component_tree", value="3")
    rows = {row.path: row.columns for row in ui.visible().tree_rows}
    assert rows[("Q1", "rate")][1] == "9"
    assert rows[("Q1", "energy_a")][1] == "3"
    ui.screenshot(evidence / "03-two-retained.png")
    ui.edit_tree_value(
        ("Q1", "rate"), object_name="component_tree", value="8", commit=False
    )
    ui.press_key(Qt.Key.Key_Escape)
    rows = {row.path: row.columns for row in ui.visible().tree_rows}
    assert rows[("Q1", "rate")][1] == "9"
    assert rows[("Q1", "energy_a")][1] == "3"
    ui.screenshot(evidence / "04-escape-retained.png")
    ui.click(object_name="inspect.parameters.sync.reload")
    rows = {row.path: row.columns for row in ui.visible().tree_rows}
    assert rows[("Q1", "rate")][1] == "4700.0"
    assert rows[("Q1", "energy_a")][1] == "(unset)"
    ui.screenshot(evidence / "05-reloaded.png")


@pytest.mark.parametrize("conflict_ui", ["parameters"], indirect=True)
def test_read_only_locks_active_parameter_draft(conflict_ui) -> None:
    ui, _, evidence, _, dialog = conflict_ui
    evidence = evidence / "read-only"
    ui.enter_text(
        object_name="inspect.parameters.filter", value="Q1.rate", commit=False
    )
    ui.edit_tree_value(
        ("Q1", "rate"), object_name="component_tree", value="9", commit=False
    )
    ui.screenshot(evidence / "01-draft.png")
    # The host changes authority through its public dialog port.
    dialog.set_read_only(read_only=True)
    ui.type_text("8")
    assert ui.focused_input().text == "9"
    ui.press_key(Qt.Key.Key_Return)
    assert any(
        row.path == ("Q1", "rate") and row.columns[1] == "9"
        for row in ui.visible().tree_rows
    )
    ui.screenshot(evidence / "02-locked-retained.png")
    dialog.set_read_only(read_only=False)
    ui.edit_tree_value(("Q1", "rate"), object_name="component_tree", value="8")
    assert not ui.observe(object_name="component_error").text
    ui.click("Refresh")
    assert any(
        row.path == ("Q1", "rate") and row.columns[1] == "8.0"
        for row in ui.visible().tree_rows
    )
    ui.screenshot(evidence / "03-unlocked-applied.png")


@pytest.mark.parametrize("conflict_ui", ["parameters", "library", "aliases"], indirect=True)
def test_clean_pages_show_human_pending_and_invalid_source(conflict_ui) -> None:
    ui, path, evidence, source, dialog = conflict_ui
    document = next(file.document for file in source.get_file_revisions() if file.path == path)
    evidence = evidence / "clean-status"
    ui.select_tab({"parameters": "Parameters", "library": "Modules", "aliases": "Aliases"}[document])
    original = path.read_text()
    changed = {
        "parameters": original.replace("4300.0", "4700.0"),
        "library": original.replace("0.1", "0.7"),
        "aliases": "format: zcu.aliases\nformat_version: '1.0'\naliases:\n  external: Q2\n",
    }[document]
    path.write_text(changed)
    source.synchronize_files(admit_reload=False)
    message = ui.observe(object_name=f"inspect.{document}.sync.message")
    assert message.visible and "pending" in message.text
    ui.screenshot(evidence / "01-human-pending.png")
    dialog.set_read_only(read_only=True)
    assert not ui.observe(object_name=f"inspect.{document}.sync.reload").enabled
    ui.click(object_name=f"inspect.{document}.sync.reload")
    assert "pending" in ui.observe(object_name=f"inspect.{document}.sync.message").text
    ui.screenshot(evidence / "02-reload-locked.png")
    dialog.set_read_only(read_only=False)
    path.write_text("invalid: true\n")
    source.synchronize_files(admit_reload=True)
    message = ui.observe(object_name=f"inspect.{document}.sync.message")
    assert message.visible and str(path) in message.text
    ui.screenshot(evidence / "03-invalid-error.png")
    ui.click(object_name=f"inspect.{document}.sync.reload")
    assert str(path) in ui.observe(object_name=f"inspect.{document}.sync.message").text
    ui.screenshot(evidence / "04-invalid-reload-rejected.png")
    path.write_text(changed)
    ui.click(object_name=f"inspect.{document}.sync.reload")
    assert not ui.observe(object_name=f"inspect.{document}.sync.message").visible
    ui.screenshot(evidence / "05-valid-reloaded.png")


@pytest.mark.parametrize("conflict_ui", ["parameters", "library", "aliases"], indirect=True)
def test_source_publications_keep_visible_drafts_until_valid_reload(conflict_ui) -> None:
    ui, path, evidence, source, _ = conflict_ui
    document = next(file.document for file in source.get_file_revisions() if file.path == path)
    evidence = evidence / "published-draft"
    if document == "parameters":
        ui.enter_text(object_name="inspect.parameters.filter", value="Q1.rate", commit=False)
        ui.edit_tree_value(("Q1", "rate"), object_name="component_tree", value="9", commit=False)
    elif document == "library":
        ui.select_tab("Modules")
        ui.select_tree_row(("waveforms", "Q1", "waveforms", "flat"), object_name="inspect.library")
        ui.enter_text(object_name="inspect.entry.name", value="Q1.waveforms.draft", commit=False)
    else:
        ui.select_tab("Aliases")
        ui.click(object_name="inspect.aliases.add")
        ui.type_text("draft")
        ui.press_key(Qt.Key.Key_Return)
    ui.screenshot(evidence / "01-draft.png")
    original = path.read_text()
    changed = {
        "parameters": original.replace("4300.0", "4700.0"),
        "library": original.replace("0.1", "0.7"),
        "aliases": "format: zcu.aliases\nformat_version: '1.0'\naliases:\n  external: Q2\n",
    }[document]
    path.write_text(changed)
    source.synchronize_files(admit_reload=True)
    assert ui.observe(object_name=f"inspect.{document}.sync.message").visible
    if document == "parameters":
        assert ui.focused_input().text == "9"
        ui.press_key(Qt.Key.Key_Return)
        assert ui.observe(object_name="component_error").text
    elif document == "library":
        assert ui.text_input(object_name="inspect.entry.name").text == "Q1.waveforms.draft"
        ui.click(object_name="inspect.entry.apply")
        assert ui.observe(object_name="inspect.entry.status").text == "Unsaved"
    else:
        assert any(row.columns[0] == "draft" for row in ui.table_rows(object_name="inspect.aliases.table"))
        ui.click(object_name="inspect.aliases.apply")
        assert ui.observe(object_name="inspect.aliases.error").text
    ui.screenshot(evidence / "02-reloaded-source-rejects-draft.png")
    path.write_text("invalid: true\n")
    source.synchronize_files(admit_reload=True)
    ui.click(object_name=f"inspect.{document}.sync.reload")
    assert str(path) in ui.observe(object_name=f"inspect.{document}.sync.message").text
    ui.screenshot(evidence / "03-invalid-keeps-draft.png")
    path.write_text(changed)
    ui.click(object_name=f"inspect.{document}.sync.reload")
    assert not ui.observe(object_name=f"inspect.{document}.sync.message").visible
    if document == "parameters":
        assert any(row.path == ("Q1", "rate") and row.columns[1] == "4700.0" for row in ui.visible().tree_rows)
    elif document == "library":
        assert ui.text_input(object_name="inspect.entry.name").text == "Q1.waveforms.flat"
        assert ui.observe(object_name="inspect.entry.status").text == "Saved"
    else:
        assert ui.table_rows(object_name="inspect.aliases.table")[0].columns[:2] == ("external", "Q2")
    ui.screenshot(evidence / "04-valid-reload-discards-draft.png")
