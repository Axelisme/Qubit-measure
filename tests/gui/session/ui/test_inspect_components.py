"""The shared Inspect composes the real parameter owner and component frontend."""

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
    assert [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())] == [
        "Q1"
    ]
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
