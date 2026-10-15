"""The shipped measure GUI composition works without its optional control socket."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.catalog import (
    ExperimentCatalogLoader,
    PreparedCatalogReload,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.form import FormErrors

from tests.gui.app.measure._synthetic import SampleAdapter


@pytest.fixture
def measure_gui(qapp: QApplication, tmp_path: Path) -> Iterator[Controller]:
    class UnusedLoader:
        def prepare(self) -> PreparedCatalogReload:
            raise RuntimeError("reload is not part of this composition test")

        def load(self, plan: PreparedCatalogReload) -> Registry:
            raise RuntimeError("reload is not part of this composition test")

    def registry_factory() -> tuple[Registry, ExperimentCatalogLoader]:
        registry = Registry()
        registry.register("fake", SampleAdapter)
        return registry, UnusedLoader()

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
            ctrl.close()
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

    cfg = ctrl.cfg_resources.lookup(tab_id)
    key = f"tab:{tab_id}:cfg"
    other_key = f"tab:{other_tab}:cfg"
    before = ctrl.resources_versions()

    cfg.write({("gain",): 0.25})

    saved = ctrl.get_tab_snapshot(tab_id).cfg_input
    assert saved["gain"] == 0.25
    assert ctrl.resources_versions()[key] == before.get(key, 0) + 1
    assert ctrl.resources_versions().get(other_key, 0) == before.get(other_key, 0)

    cfg.write({("gain",): "1e"})

    invalid = ctrl.get_tab_snapshot(tab_id).cfg_input["gain"]
    assert invalid == "1e"
    assert cfg.input.data["gain"] == "1e"
    assert isinstance(cfg.result(), FormErrors)
    assert saved["gain"] == 0.25
    assert ctrl.resources_versions()[key] == before.get(key, 0) + 2
    qapp.processEvents()
    assert ctrl.resources_versions()[key] == before.get(key, 0) + 2
    assert ctrl.resources_versions().get(other_key, 0) == before.get(other_key, 0)
