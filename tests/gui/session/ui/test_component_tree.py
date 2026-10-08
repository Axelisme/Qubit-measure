"""The component frontend submits owner commands and exposes failures."""

from collections.abc import Generator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QLabel, QTreeWidget
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.resources.entry import ComponentSchema, ResultEntry, component_registry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


class RequiredComponent(ComponentSchema):
    reference: str


@pytest.fixture
def component_widget(
    qapp, tmp_path: Path
) -> Generator[tuple[ComponentTreeWidget, ContextControlFacet]]:
    before = registry_state()
    widget = None
    try:
        register_fakes()
        component_registry.register("test/required", RequiredComponent)
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "results",
            database_root=tmp_path / "Database",
        )
        service = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            BaseEventBus(),
            bind_assets=ArbWaveformDatabase.info_at,
        )
        service.install_entry(entry)
        control = ContextControlFacet(context=service)
        widget = ComponentTreeWidget(control)
        yield widget, control
        assert registry_state() == installed
    finally:
        if widget is not None:
            widget.close()
            widget.deleteLater()
            qapp.processEvents()
        restore_registry(before)


def test_add_component_renders_tree_and_can_select_focus(component_widget):
    widget, control = component_widget
    assert widget.add_component("Q1", "fake/drive/a", {"rate": 4300.0})
    tree = widget.findChild(QTreeWidget, "component_tree")
    assert tree is not None
    assert any(
        tree.topLevelItem(i).text(0) == "Q1" for i in range(tree.topLevelItemCount())
    )
    widget.set_focus("Q1")
    assert control.get_component_focus().name == "Q1"


def test_missing_required_field_displays_error_without_mutation(component_widget):
    widget, control = component_widget
    assert not widget.add_component("Ref", "test/required", {})
    error = widget.findChild(QLabel, "component_error")
    assert error is not None and "reference" in error.text()
    color = error.palette().color(error.foregroundRole())
    assert color.red() > color.green() and color.red() > color.blue()
    assert "Ref" not in control.inspect_document().components
    assert widget.add_component(
        "Ref", "test/required", {"reference": "unresolved-name"}
    )
    assert widget.edit_parameter("Ref.reference", "ordinary-string")
    assert control.read_context().value("Ref.reference") == "ordinary-string"
