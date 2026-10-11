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
    from zcu_tools.experiment.workflows.declaration import WorkflowStep

    from zcu_lab.workflows.calibration import Calibrated
    from zcu_lab.workflows.gui.catalog import WorkflowCatalog
    from zcu_lab.workflows.t1 import T1Plan, T1Tunables


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

    def write(self, version: int, *, failure: str = "", wait: bool = False) -> None:
        """Replace one trusted declaration; failure is executable import failure."""
        wait_step = (
            "yield from env.wait_until(env.started_at + timedelta(seconds=60))"
            if wait
            else "yield from ()"
        )
        fields = "legacy: int = 4" if version == 1 else "new_field: int = 9"
        knobs = "legacy_knob: int = 5" if version == 1 else "new_knob: int = 10"
        (self.root / "declaration.py").write_text(
            f"""from datetime import timedelta
from zcu_tools.experiment.workflows import Done, workflow
from zcu_lab.workflows.t1 import T1Plan, T1Tunables, init_t1
from zcu_lab.workflows.gui.catalog import WorkflowCatalog, calibrated_seed
from zcu_lab.workflows.gui.attachment import WorkflowAttachment
from zcu_lab.workflows.gui.plan import T1PlanForm
from zcu_lab.workflows.gui.tunables import T1TunablesForm

class Plan(T1PlanForm):
    {fields}

class Knobs(T1TunablesForm):
    {knobs}

@workflow("reload-v{version}", plan=T1Plan, tunables=T1Tunables, init=init_t1)
def step(env, plan, tun, state):
    {wait_step}
    return Done()

def prepare(*, plan, data_root, records):
    raise ValueError("export-v{version}:" + str(plan.points))

def build():
    return WorkflowCatalog((WorkflowAttachment(
        step=step, plan_model=Plan, knobs_model=Knobs,
        plan_to_cfg=Plan.to_cfg, knobs_to_cfg=Knobs.to_cfg,
        seed=calibrated_seed, prepare_export=prepare,
    ),))
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


class InjectedCatalog:
    """Public constructor/loader injection for declared Window scenario collaborators.

    The fixture owns a tiny isolated source namespace and current app catalog.
    Scenario installation reloads the real Window through its configured public
    SourceCatalogLoader; it never patches production callables or Controller.
    """

    def __init__(self, root: Path) -> None:
        from zcu_lab.workflows.gui.catalog import build_catalog

        self.root = root / "injected_workflow_source"
        self.root.mkdir()
        (self.root / "__init__.py").write_text("")
        self.path = str(root)
        sys.path.insert(0, self.path)
        self.catalog = build_catalog()
        self.loader = SourceCatalogLoader(
            sources=(SourcePackage("injected_workflow_source", self.root),),
            reload_modules=("injected_workflow_source",),
            preserved_modules=(),
            catalog_module="injected_workflow_source",
            catalog_builder=self.build,
        )

    def build(self) -> WorkflowCatalog:
        """Return the supplied typed pairing; native declarations remain real."""
        return self.catalog

    def install[S, R](
        self,
        step: WorkflowStep[T1Plan, T1Tunables, S, R, Calibrated],
        window: T1WorkflowWindow,
    ) -> None:
        """Select a real declared scenario via public current-catalog reload."""
        from zcu_lab.workflows.export import prepare_t1_fluxdep_export
        from zcu_lab.workflows.gui.attachment import WorkflowAttachment
        from zcu_lab.workflows.gui.catalog import WorkflowCatalog, calibrated_seed
        from zcu_lab.workflows.gui.plan import T1PlanForm
        from zcu_lab.workflows.gui.tunables import T1TunablesForm

        self.catalog = WorkflowCatalog(
            (
                WorkflowAttachment(
                    step=step,
                    plan_model=T1PlanForm,
                    knobs_model=T1TunablesForm,
                    plan_to_cfg=T1PlanForm.to_cfg,
                    knobs_to_cfg=T1TunablesForm.to_cfg,
                    seed=calibrated_seed,
                    prepare_export=prepare_t1_fluxdep_export,
                ),
            )
        )
        window.reload_source()

    def close(self) -> None:
        """Retire only owned source modules/path after the Window has drained."""
        sys.path.remove(self.path)
        for name in tuple(sys.modules):
            if name == "injected_workflow_source" or name.startswith(
                "injected_workflow_source."
            ):
                del sys.modules[name]
        importlib.invalidate_caches()
