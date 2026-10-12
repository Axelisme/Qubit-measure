"""Isolated trusted source revisions for public workflow reload contracts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.catalog_loader import SourceCatalogLoader, SourcePackage

from tests.gui.app.workflows._start_fakes import StartHarness

if TYPE_CHECKING:
    from zcu_tools.experiment.workflows.declaration import WorkflowStep
    from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog

    from zcu_lab.workflows.calibration import Calibrated
    from zcu_lab.workflows.coherence import CoherencePlan, CoherenceTunables


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
        self.views: list[tuple[WorkflowWindow[Calibrated], StartHarness]] = []
        self.loader = SourceCatalogLoader(
            sources=(SourcePackage("workflow_reload_fixture", self.root),),
            reload_modules=("workflow_reload_fixture.declaration",),
            preserved_modules=(),
            catalog_module="workflow_reload_fixture.declaration",
            catalog_builder=self.build,
        )

    def build(self) -> WorkflowCatalog[Calibrated]:
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
from zcu_lab.workflows.coherence import CoherencePlan, CoherenceTunables, init_coherence
from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog
from zcu_lab.workflows.registry import calibrated_seed
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_lab.workflows.gui.plan import CoherencePlanForm
from zcu_lab.workflows.gui.coherence import CoherenceTunablesForm

class Plan(CoherencePlanForm):
    {fields}

class Knobs(CoherenceTunablesForm):
    {knobs}

@workflow("reload-v{version}", plan=CoherencePlan, tunables=CoherenceTunables, init=init_coherence)
def step(env, plan, tun, state):
    {wait_step}
    return Done()

def build():
    return WorkflowCatalog((WorkflowAttachment(
        step=step, plan_model=Plan, knobs_model=Knobs,
        plan_to_cfg=Plan.to_cfg, knobs_to_cfg=Knobs.to_cfg,
        seed=calibrated_seed,
    ),))
{failure}
"""
        )

    def open(self, harness: StartHarness) -> WorkflowWindow[Calibrated]:
        """Construct a view with the same real session and an isolated catalog."""
        catalog = self.loader.load(self.loader.prepare())
        view = WorkflowWindow(
            harness.state,
            harness.services,
            harness.owner,
            registrations=catalog.attachments,
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
        from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog

        from zcu_lab.workflows.registry import build_registry

        self.root = root / "injected_workflow_source"
        self.root.mkdir()
        (self.root / "__init__.py").write_text("")
        self.path = str(root)
        sys.path.insert(0, self.path)
        self.catalog = WorkflowCatalog(build_registry())
        self.loader = SourceCatalogLoader(
            sources=(SourcePackage("injected_workflow_source", self.root),),
            reload_modules=("injected_workflow_source",),
            preserved_modules=(),
            catalog_module="injected_workflow_source",
            catalog_builder=self.build,
        )

    def build(self) -> WorkflowCatalog[Calibrated]:
        """Return the supplied typed pairing; native declarations remain real."""
        return self.catalog

    def install[S, R](
        self,
        step: WorkflowStep[CoherencePlan, CoherenceTunables, S, R, Calibrated],
        window: WorkflowWindow[Calibrated],
    ) -> None:
        """Select a real declared scenario via public current-catalog reload."""
        from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
        from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog

        from zcu_lab.workflows.gui.coherence import CoherenceTunablesForm
        from zcu_lab.workflows.gui.plan import CoherencePlanForm
        from zcu_lab.workflows.registry import calibrated_seed

        self.catalog = WorkflowCatalog(
            (
                WorkflowAttachment(
                    step=step,
                    plan_model=CoherencePlanForm,
                    knobs_model=CoherenceTunablesForm,
                    plan_to_cfg=CoherencePlanForm.to_cfg,
                    knobs_to_cfg=CoherenceTunablesForm.to_cfg,
                    seed=calibrated_seed,
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
