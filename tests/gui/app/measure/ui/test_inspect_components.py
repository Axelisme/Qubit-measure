"""Shipped Inspect composes the real parameter owner and component frontend."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QComboBox, QTabWidget, QTreeWidget
from zcu_tools.gui.app.measure.ui.inspect_dialog import InspectDialog
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


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
        service.install_entry(entry)
        control = ContextControlFacet(context=service)
        # Only app-specific Modules/editor collaborators are outside this seam.
        app = MagicMock()
        app.context_control = control
        dialog = InspectDialog(app, bus)
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
    editor.set_focus("Q1")
    assert control.get_component_focus().name == "Q1"
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Parameters", "Modules"]


def test_inspect_template_edits_do_not_change_current_point_or_focus(inspector) -> None:
    dialog, control = inspector
    editor = dialog.findChild(ComponentTreeWidget)
    assert editor is not None
    assert editor.add_component("Q1", "fake/drive/a", {"rate": 4300.0})
    editor.set_focus("Q1")
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
    assert control.get_component_focus().name == "Q1"
