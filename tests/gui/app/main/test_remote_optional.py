"""The shipped measure GUI composition works without its optional control socket."""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_measure_gui_can_open_a_fake_tab_without_remote_adapter(
    qapp: QApplication, tmp_path: Path
) -> None:
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
        tab_id = ctrl.new_tab("fake")
        qapp.processEvents()
        assert ctrl.get_tab_snapshot(tab_id).tab_id == tab_id
    finally:
        sys.excepthook = previous_hook
        if ctrl is not None and window is not None:
            ctrl._background_svc.quiesce()  # pyright: ignore[reportPrivateUsage] - fixture teardown
            window.deleteLater()
            qapp.processEvents()
