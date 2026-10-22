"""The shared Inspect composes the real parameter owner and component frontend."""

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QComboBox, QTabWidget, QTreeWidget
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.gui.session.ui.inspect_dialog import InspectDialog
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


class _NoArbNames:
    def list_data_keys(self) -> list[str]:
        return []


@pytest.fixture
def inspector(
    qapp, tmp_path: Path
) -> Generator[tuple[InspectDialog, ContextControlFacet]]:
    before = registry_state()
    dialog = None
    try:
        register_fakes()
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "results",
            database_root=tmp_path / "Database",
        )
        bus = BaseEventBus()
        service = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            bus,
            bind_assets=ArbWaveformDatabase.info_at,
        )
        entry.new_point("working")
        service.install_entry(entry)
        service.use_point(PointUseRequest("working"))
        control = ContextControlFacet(context=service)
        dialog = InspectDialog(
            control,
            bus,
            templates={},
            arb_waveforms=_NoArbNames(),
            open_arb_waveforms=lambda: None,
            dialog_presenter=RecordingDialogPresenter(),
        )
        yield dialog, control
        assert registry_state() == installed
    finally:
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
            qapp.processEvents()
        restore_registry(before)


def test_shipped_inspect_refresh_keeps_component_frontend_and_modules_tab(
    inspector,
) -> None:
    dialog, control = inspector
    editor = dialog.findChild(ComponentTreeWidget)
    assert editor is not None
    assert editor.add_component("Q1", "fake/drive/a", {"rate": 4300.0})
    dialog.refresh()
    assert dialog.findChild(ComponentTreeWidget) is editor
    tree = editor.findChild(QTreeWidget, "component_tree")
    assert tree is not None
    captured = control.read_context()
    assert captured.lookup("Q1.rate").value == 4300.0
    assert editor.edit_parameter("Q1.rate", 4400.0)
    assert control.read_context().lookup("Q1.rate").value == 4400.0
    assert captured.lookup("Q1.rate").value == 4300.0
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Parameters",
        "Modules",
        "Aliases",
    ]


@pytest.fixture
def general_ext_ui(inspector, read_only: bool) -> ComponentDriver:
    dialog, control = inspector
    control.set_setup_parameter(
        "general.ext",
        {
            "note": "setup note",
            "calibration": {"offset": 3.5},
            "flags": [True, None],
            "legacy.key": "literal key",
        },
    )
    control.set_parameter(
        control.read_context().target,
        "general.ext",
        {
            "note": "point note",
            "calibration": {"offset": 1.25},
            "flags": [False, None],
            "legacy.key": "literal key",
        },
    )
    dialog.set_read_only(read_only=read_only)
    dialog.show()
    return ComponentDriver(dialog)


def _general_values(ui: ComponentDriver) -> dict[tuple[str, ...], str]:
    return {
        row.path[2:]: row.columns[1]
        for row in ui.visible().tree_rows
        if row.tree_name == "component_tree" and row.path[:2] == ("general", "ext")
    }


def _expand_general(ui: ComponentDriver) -> None:
    for path in (("general",), ("general", "ext")):
        if any(row.path == path for row in ui.visible().tree_rows):
            ui.expand_tree(path, object_name="component_tree")


@pytest.mark.parametrize("read_only", [False, True], ids=["editable", "readonly"])
@pytest.mark.parametrize("destination", ["Current point", "Setup template"])
def test_general_ext_values_follow_destination_and_edit_policy(
    general_ext_ui: ComponentDriver,
    read_only: bool,
    destination: str,
    tmp_path: Path,
) -> None:
    ui = general_ext_ui
    evidence = Path(os.environ.get("INSPECT_GENERAL_EXT_EVIDENCE", str(tmp_path)))
    evidence /= (
        f"{destination.replace(' ', '-')}-{'readonly' if read_only else 'editable'}"
    )
    ui.choose_option("Current point", value=destination)
    _expand_general(ui)
    ui.screenshot(evidence / "01-values.png")
    expected_note = "point note" if destination == "Current point" else "setup note"
    assert _general_values(ui).get(("note",)) == expected_note
    assert _general_values(ui).get(("legacy.key",)) == "literal key"
    assert _general_values(ui).get(("flags",)) == (
        "[false, null]" if destination == "Current point" else "[true, null]"
    )

    ui.expand_tree(("general", "ext", "calibration"), object_name="component_tree")
    ui.screenshot(evidence / "02-nested.png")
    before = "1.25" if destination == "Current point" else "3.5"
    assert _general_values(ui).get(("calibration", "offset")) == before
    ui.edit_tree_value(
        ("general", "ext", "calibration", "offset"),
        object_name="component_tree",
        value="7.75",
    )
    ui.screenshot(evidence / "03-edited.png")
    expected = before if read_only else "7.75"
    assert _general_values(ui).get(("calibration", "offset")) == expected

    other = "Setup template" if destination == "Current point" else "Current point"
    ui.choose_option(destination, value=other)
    _expand_general(ui)
    ui.screenshot(evidence / "04-other-destination.png")
    assert _general_values(ui).get(("calibration", "offset")) == (
        "3.5" if destination == "Current point" else "1.25"
    )
    ui.choose_option(other, value=destination)
    ui.click("Refresh")
    ui.screenshot(evidence / "05-refreshed.png")
    assert _general_values(ui).get(("calibration", "offset")) == expected


def test_inspect_template_edits_do_not_change_current_point(inspector) -> None:
    dialog, control = inspector
    editor = dialog.findChild(ComponentTreeWidget)
    assert editor is not None
    assert editor.add_component("Q1", "fake/drive/a", {"rate": 4300.0})
    before = control.inspect_document()
    destination = next(
        combo
        for combo in editor.findChildren(QComboBox)
        if combo.findText("Setup template") >= 0
    )
    destination.setCurrentText("Setup template")
    assert editor.add_component(
        "Template", "fake/drive/a", {"rate": 5000.0}, setup=True
    )
    dialog.refresh()
    assert "Template" in control.inspect_document(setup=True).components
    assert control.inspect_document() == before
