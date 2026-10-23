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
def conflict_ui(
    qapp: QApplication, tmp_path: Path, request: pytest.FixtureRequest
) -> Generator[tuple[ComponentDriver, Path, Path]]:
    before = registry_state()
    register_fakes()
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
    qapp.processEvents()
    evidence = Path(os.environ.get("INSPECT_SYNC_EVIDENCE", str(tmp_path))) / document
    try:
        yield ComponentDriver(dialog), path, evidence
    finally:
        dialog.hide()
        dialog.deleteLater()
        qapp.processEvents()
        restore_registry(before)


@pytest.mark.parametrize("conflict_ui", ["library"], indirect=True)
def test_library_keep_cannot_rebase_stale_draft(conflict_ui) -> None:
    ui, path, evidence = conflict_ui
    ui.select_tab("Modules")
    ui.press_key(Qt.Key.Key_End, object_name="inspect.library")
    ui.enter_text(
        object_name="inspect.entry.name", value="Q1.waveforms.draft", commit=False
    )
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
    ui, path, evidence = conflict_ui
    ui.select_tab("Aliases")
    initial_count = len(ui.table_rows(object_name="inspect.aliases.table"))
    ui.click(object_name="inspect.aliases.add")
    ui.type_text("x")
    ui.press_key(Qt.Key.Key_Return)
    ui.screenshot(evidence / "01-draft.png")
    path.write_text("aliases:\n  external: Q2\n")
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
    ui, path, evidence = conflict_ui
    evidence = evidence / submit
    ui.enter_text(
        object_name="inspect.parameters.filter", value="Q1.rate", commit=False
    )
    ui.press_key(Qt.Key.Key_End, object_name="component_tree")
    ui.press_key(Qt.Key.Key_F2, object_name="component_tree")
    ui.press_key(Qt.Key.Key_A, modifiers=Qt.KeyboardModifier.ControlModifier)
    ui.type_text("9")
    ui.screenshot(evidence / "01-draft.png")
    path.write_text(path.read_text().replace("4300.0", "4700.0"))
    if submit == "enter":
        ui.press_key(Qt.Key.Key_Return)
    else:
        ui.click(object_name="inspect.parameters.filter")
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
