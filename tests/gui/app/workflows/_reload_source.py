"""Isolated trusted source revisions for public workflow reload contracts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog
from zcu_tools.gui.app.workflows.composition import WorkflowComposition
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.catalog_loader import SourceCatalogLoader, SourcePackage

from tests.gui.app.workflows._start_fakes import StartHarness

from ._synthetic import (
    Context,
    Plan,
    PlanForm,
    Tunables,
    TunablesForm,
    build_registry,
    context_seed,
)

if TYPE_CHECKING:
    from zcu_tools.experiment.workflows.declaration import WorkflowStep


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
        self.views: list[tuple[WorkflowWindow[Context], StartHarness]] = []
        self.loader = SourceCatalogLoader(
            sources=(SourcePackage("workflow_reload_fixture", self.root),),
            reload_modules=("workflow_reload_fixture.declaration",),
            preserved_modules=(),
            catalog_module="workflow_reload_fixture.declaration",
            catalog_builder=self.build,
        )

    def build(self) -> WorkflowCatalog[Context]:
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
from tests.gui.app.workflows._synthetic import Plan as NativePlan, Tunables, init_sample
from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog
from tests.gui.app.workflows._synthetic import context_seed
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from tests.gui.app.workflows._synthetic import PlanForm
from tests.gui.app.workflows._synthetic import TunablesForm

class Plan(PlanForm):
    {fields}

class Knobs(TunablesForm):
    {knobs}

@workflow("reload-v{version}", plan=NativePlan, tunables=Tunables, init=init_sample)
def step(env, plan, tun, state):
    {wait_step}
    return Done()

def build():
    return WorkflowCatalog((WorkflowAttachment(
        step=step, plan_model=Plan, knobs_model=Knobs,
        plan_to_cfg=Plan.to_cfg, knobs_to_cfg=Knobs.to_cfg,
        seed=context_seed,
    ),))
{failure}
"""
        )

    def open(self, harness: StartHarness) -> WorkflowWindow[Context]:
        """Construct a view with the same real session and an isolated catalog."""
        catalog = self.loader.load(self.loader.prepare())
        view = WorkflowWindow(
            harness.state,
            harness.services,
            harness.owner,
            composition=WorkflowComposition(
                templates={},
                alias_presets=None,
                registrations=catalog.attachments,
                loader=self.loader,
            ),
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


class SourceNamespace[C]:
    """Own one importable source package whose loader publishes ``catalog``.

    package names the empty package created under root; root goes on sys.path.
    catalog is what the next load or Window reload publishes, so a test may
    replace it before calling the Window's public reload. close retires the
    package's modules and the sys.path entry; call it after every Window over
    loader has closed.
    """

    def __init__(self, root: Path, package: str, catalog: WorkflowCatalog[C]) -> None:
        self.root = root / package
        self.root.mkdir()
        (self.root / "__init__.py").write_text("")
        self.package = package
        self.path = str(root)
        sys.path.insert(0, self.path)
        self.catalog = catalog
        self.loader = SourceCatalogLoader(
            sources=(SourcePackage(package, self.root),),
            reload_modules=(package,),
            preserved_modules=(),
            catalog_module=package,
            catalog_builder=self.build,
        )

    def build(self) -> WorkflowCatalog[C]:
        """Return the supplied typed pairing; native declarations remain real."""
        return self.catalog

    def close(self) -> None:
        """Retire only owned source modules/path after the Window has drained."""
        sys.path.remove(self.path)
        for name in tuple(sys.modules):
            if name == self.package or name.startswith(f"{self.package}."):
                del sys.modules[name]
        importlib.invalidate_caches()


class InjectedCatalog(SourceNamespace[Context]):
    """Public constructor/loader injection for declared Window scenario collaborators.

    The fixture owns a tiny isolated source namespace whose catalog starts as
    the synthetic sample registry. Scenario installation reloads the real Window
    through its configured public SourceCatalogLoader; it never patches
    production callables or Controller.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(
            root, "injected_workflow_source", WorkflowCatalog(build_registry())
        )

    def install[S, R](
        self,
        step: WorkflowStep[Plan, Tunables, S, R, Context],
        window: WorkflowWindow[Context],
    ) -> None:
        """Select a real declared scenario via public current-catalog reload."""
        self.catalog = WorkflowCatalog(
            (
                WorkflowAttachment(
                    step=step,
                    plan_model=PlanForm,
                    knobs_model=TunablesForm,
                    plan_to_cfg=PlanForm.to_cfg,
                    knobs_to_cfg=TunablesForm.to_cfg,
                    seed=context_seed,
                ),
            )
        )
        window.reload_source()
