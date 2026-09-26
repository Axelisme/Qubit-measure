"""The shipped measure GUI composition works without its optional control socket."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.gui.app.main.app import MeasureGuiBehavior
from zcu_tools.gui.app.main.catalog import (
    ExperimentCatalogLoader,
    PreparedCatalogReload,
)
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.role_catalog import RoleCatalog
from zcu_tools.gui.app.main.ui.main_window import MainWindow
from zcu_tools.gui.cfg import DirectValue
from zcu_tools.gui.cfg.binding import ScalarField


@pytest.fixture
def measure_gui(qapp: QApplication, tmp_path: Path) -> Iterator[Controller]:
    class UnusedLoader:
        def prepare(self) -> PreparedCatalogReload:
            raise RuntimeError("reload is not part of this composition test")

        def load(self, plan: PreparedCatalogReload) -> Registry:
            raise RuntimeError("reload is not part of this composition test")

    def registry_factory() -> tuple[Registry, RoleCatalog, ExperimentCatalogLoader]:
        registry = Registry()
        registry.register("fake", FakeAdapter)
        return registry, RoleCatalog(), UnusedLoader()

    previous_hook = sys.excepthook
    ctrl: Controller | None = None
    window: MainWindow | None = None
    try:
        behavior = MeasureGuiBehavior(
            registry_factory, clean=True, project_root=str(tmp_path)
        )
        assembly = behavior.assemble(None)
        assert assembly.control_adapter is None
        assert isinstance(assembly.controller, Controller)
        assert isinstance(assembly.window, MainWindow)
        ctrl, window = assembly.controller, assembly.window
        behavior.before_show(assembly)
        yield ctrl
    finally:
        sys.excepthook = previous_hook
        if ctrl is not None and window is not None:
            ctrl._background_svc.quiesce()  # pyright: ignore[reportPrivateUsage] - fixture teardown
            window.deleteLater()
            qapp.processEvents()


def test_measure_gui_can_open_a_fake_tab_without_remote_adapter(
    qapp: QApplication, measure_gui: Controller
) -> None:
    tab_id = measure_gui.new_tab("fake")
    qapp.processEvents()
    assert measure_gui.get_tab_snapshot(tab_id).tab_id == tab_id


def test_tab_cfg_and_revision_publish_without_viewer_timer(
    qapp: QApplication, measure_gui: Controller
) -> None:
    ctrl = measure_gui
    tab_id = ctrl.new_tab("fake")
    other_tab = ctrl.new_tab("fake")
    qapp.processEvents()
    editor_id = ctrl.editor_id_for_owner(tab_id)
    assert editor_id is not None
    key = f"tab:{tab_id}:cfg"
    other_key = f"tab:{other_tab}:cfg"
    before = ctrl.resources_versions()

    ctrl.cfg_editor_set_field(editor_id, "gain", 0.25)

    saved = ctrl.get_tab_snapshot(tab_id).cfg_schema
    assert saved.value.fields["gain"] == DirectValue(0.25)
    assert ctrl.resources_versions()[key] == before[key] + 1
    assert ctrl.resources_versions()[other_key] == before[other_key]

    field = ctrl.get_cfg_editor_draft(editor_id).root.fields["gain"]
    assert isinstance(field, ScalarField)
    field.set_text("1e")

    invalid = ctrl.get_tab_snapshot(tab_id).cfg_schema.value.fields["gain"]
    assert isinstance(invalid, DirectValue)
    assert invalid.raw == "1e"
    assert invalid.error is not None
    assert invalid.value is None
    assert saved.value.fields["gain"] == DirectValue(0.25)
    assert ctrl.resources_versions()[key] == before[key] + 2
    qapp.processEvents()
    assert ctrl.resources_versions()[key] == before[key] + 2
    assert ctrl.resources_versions()[other_key] == before[other_key]


def test_non_tab_editor_does_not_publish_into_tab_state(
    qapp: QApplication, measure_gui: Controller
) -> None:
    ctrl = measure_gui
    tab_id = ctrl.new_tab("fake")
    qapp.processEvents()
    original = ctrl.get_tab_snapshot(tab_id).cfg_schema
    key = f"tab:{tab_id}:cfg"
    before = ctrl.resources_versions()[key]
    editor_id, _ = ctrl.open_seeded_cfg_editor(original, owner_key="writeback-test")
    try:
        ctrl.cfg_editor_set_field(editor_id, "gain", 0.75)
        assert ctrl.get_tab_snapshot(tab_id).cfg_schema == original
        assert ctrl.resources_versions()[key] == before
    finally:
        ctrl.teardown_cfg_editor(editor_id)
