"""Isolated trusted source revisions for public workflow reload contracts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QApplication
from zcu_tools.gui.catalog_loader import SourceCatalogLoader, SourcePackage

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab.workflows.gui.window import T1WorkflowWindow

if TYPE_CHECKING:
    from zcu_lab.workflows.gui.catalog import WorkflowCatalog


class ReloadSource:
    """Own temporary source modules and all views opened against their loader."""

    def __init__(self, root: Path) -> None:
        self.root = root / "workflow_reload_fixture"
        self.root.mkdir()
        (self.root / "__init__.py").write_text("")
        (self.root / "fixed.py").write_text("VERSION = 1\n")
        self.write(1)
        sys.path.insert(0, str(root))
        self.path = str(root)
        self.views: list[tuple[T1WorkflowWindow, StartHarness]] = []
        self.loader = SourceCatalogLoader(
            sources=(SourcePackage("workflow_reload_fixture", self.root),),
            reload_modules=("workflow_reload_fixture.declaration",),
            preserved_modules=(),
            catalog_module="workflow_reload_fixture.declaration",
            catalog_builder=self.build,
        )

    def build(self) -> WorkflowCatalog:
        """Import the app builder at call time, never retain a prior declaration."""
        return importlib.import_module("workflow_reload_fixture.declaration").build()

    def write(self, version: int, *, failure: str = "") -> None:
        """Replace one trusted declaration; failure is executable import failure."""
        fields = "legacy: int = 4" if version == 1 else "new_field: int = 9"
        knobs = "legacy_knob: int = 5" if version == 1 else "new_knob: int = 10"
        (self.root / "declaration.py").write_text(
            f"""from dataclasses import replace
from zcu_tools.experiment.workflows import Done, workflow
from zcu_lab.workflows.t1 import T1Plan, T1Tunables, init_t1
from zcu_lab.workflows.gui.catalog import build_catalog
from zcu_lab.workflows.gui.plan import T1PlanForm
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.panel import T1PlanPanel, T1TunablesPanel

class Plan(T1PlanForm):
    {fields}

class Knobs(T1TunablesForm):
    {knobs}

class PlanPanel(T1PlanPanel):
    def __init__(self, env, parent=None):
        super().__init__(env, parent, model=Plan)

class KnobsPanel(T1TunablesPanel):
    pass

@workflow("reload-v{version}", plan=T1Plan, tunables=T1Tunables, init=init_t1)
def step(env, plan, tun, state):
    yield from ()
    return Done()

def prepare(*, plan, data_root, records):
    raise ValueError("export-v{version}:" + str(plan.points))

def build():
    return replace(build_catalog(), step=step, plan_model=Plan,
                   knobs_model=Knobs, plan_panel=PlanPanel,
                   knobs_panel=KnobsPanel, prepare_export=prepare)
{failure}
"""
        )

    def open(self, harness: StartHarness) -> T1WorkflowWindow:
        """Construct a view with the same real session and an isolated catalog."""
        catalog = self.loader.load(self.loader.prepare())
        view = T1WorkflowWindow(
            harness.state,
            harness.services,
            harness.owner,
            catalog=catalog,
            loader=self.loader,
        )
        self.views.append((view, harness))
        return view

    def close(self, qapp: QApplication) -> None:
        """Release views, retire all owned modules and restore the path list."""
        for view, harness in self.views:
            view.close()
            if harness.background.work is not None:
                harness.complete_segment()
                view.close()
            view.deleteLater()
        qapp.processEvents()
        sys.path.remove(self.path)
        for name in tuple(sys.modules):
            if name == "workflow_reload_fixture" or name.startswith(
                "workflow_reload_fixture."
            ):
                del sys.modules[name]
        importlib.invalidate_caches()
