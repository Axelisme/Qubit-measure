"""Synthetic lab descriptions shared by framework composition tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from zcu_tools.analysis.flux import ComponentFluxContext
from zcu_tools.cfg_model import TemplateDef
from zcu_tools.gui.app.fluxdep.export_plan import ExportPlanner
from zcu_tools.gui.app.measure.lab import CatalogReloadPolicy
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.workflow_attachment import Attachment
from zcu_tools.gui.catalog_loader import SourcePackage
from zcu_tools.mcp.measure.recipe_api import RecipeDeclaration
from zcu_tools.resources.entry.registry import ComponentRegistry
from zcu_tools.resources.entry.schema import ComponentSchema


def empty_components(registry: ComponentRegistry) -> None:
    """Leave the component registry unchanged."""


def empty_experiments(registry: Registry) -> None:
    """Leave the experiment catalog empty."""


def synthetic_reload_policy() -> CatalogReloadPolicy:
    """Monitor the synthetic measure catalog without importing lab definitions."""
    return CatalogReloadPolicy(
        sources=(
            SourcePackage(
                "tests.gui.app.measure",
                Path(__file__).parent / "gui" / "app" / "measure",
            ),
        ),
        reload_modules=("tests.gui.app.measure._synthetic",),
        preserved_modules=(),
        catalog_module="tests.gui.app.measure._synthetic",
    )


@dataclass(frozen=True)
class SyntheticLab[C]:
    """Empty lab catalogs, with explicit per-test registration replacements.

    Fields follow the corresponding framework entry protocols. Export and flux
    context providers default to None because empty labs have no component policy.
    """

    register_components: Callable[[ComponentRegistry], None] = empty_components
    register_experiments: Callable[[Registry], None] = empty_experiments
    templates: Mapping[str, TemplateDef] = field(default_factory=dict)
    workflow_attachments: tuple[Attachment[C], ...] = ()
    recipes: tuple[RecipeDeclaration, ...] = ()
    catalog_reload_policy: CatalogReloadPolicy = field(
        default_factory=synthetic_reload_policy
    )
    logging_namespaces: tuple[str, ...] = ()
    fluxdep_export_planner: ExportPlanner | None = None
    component_flux_context_provider: (
        Callable[[ComponentSchema], ComponentFluxContext] | None
    ) = None


def synthetic_lab(registry: Registry | None = None) -> SyntheticLab[None]:
    """Copy an optional caller catalog into each newly composed behavior."""
    if registry is None:
        return SyntheticLab()

    def register(target: Registry) -> None:
        for name in registry.list_names():
            target.register(name, registry.create(name))

    return SyntheticLab(register_experiments=register, templates=registry.templates)
