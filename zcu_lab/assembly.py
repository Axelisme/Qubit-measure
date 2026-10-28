"""Assemble lab declarations for framework entry points without registering them."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import zcu_tools
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

import zcu_lab
from zcu_lab.workflows.common.calibration_modules import Calibrated


@dataclass(frozen=True)
class LabDefinitions[C]:
    """Lab-owned catalogs and policies shared by GUI and MCP composition roots.

    register_components installs component models/roles into the supplied process
    registry. register_experiments fills a supplied measure Registry without
    registering components again. Both are definition-only operations; duplicate
    registration follows the receiving registry's error contract.
    templates maps persistent template IDs to module/waveform definitions; empty
    is valid. workflow_attachments contains ordered native/form pairs sharing
    opaque context C; empty means no workflows, otherwise first is the default.
    recipes contains ordered MCP recipe declarations; empty means no recipes.
    catalog_reload_policy declares monitored package paths and reload boundaries.
    logging_namespaces lists extra logger namespace prefixes; empty adds none.
    fluxdep_export_planner produces component writeback proposals from fitted
    alignment points and raises ValueError for unsupported component definitions.
    component_flux_context_provider projects component flux frames/default devices
    and raises ValueError for unsupported kinds. Neither callback reads hardware.

    Construction only retains declarations. Entry behaviors perform registration;
    no caller should mutate template contents after publishing this description.
    """

    register_components: Callable[[ComponentRegistry], None]
    register_experiments: Callable[[Registry], None]
    templates: Mapping[str, TemplateDef]
    workflow_attachments: tuple[Attachment[C], ...]
    recipes: tuple[RecipeDeclaration, ...]
    catalog_reload_policy: CatalogReloadPolicy
    logging_namespaces: tuple[str, ...]
    fluxdep_export_planner: ExportPlanner
    component_flux_context_provider: Callable[[ComponentSchema], ComponentFluxContext]


def lab_definitions() -> LabDefinitions[Calibrated]:
    """Return frozen lab declarations using the installed framework/lab roots.

    Imports and declaration errors propagate. This does not register components,
    open forms, acquire hardware, or start any GUI/MCP runtime.
    """
    from zcu_lab.component_flux import component_flux_context
    from zcu_lab.components import register_all as register_components
    from zcu_lab.definitions import register_all as register_experiments
    from zcu_lab.fluxdep_export import plan_fluxdep_export
    from zcu_lab.recipes import RECIPES
    from zcu_lab.templates import TEMPLATES
    from zcu_lab.workflows.registry import build_registry

    return LabDefinitions(
        register_components=register_components,
        register_experiments=register_experiments,
        templates=MappingProxyType(TEMPLATES),
        workflow_attachments=build_registry(),
        recipes=RECIPES,
        catalog_reload_policy=CatalogReloadPolicy(
            sources=(
                SourcePackage("zcu_tools", Path(zcu_tools.__file__).resolve().parent),
                SourcePackage("zcu_lab", Path(zcu_lab.__file__).resolve().parent),
            ),
            reload_modules=("zcu_lab.v2", "zcu_lab.definitions"),
            preserved_modules=(
                "zcu_lab.templates",
                "zcu_lab.v2.support.measure",
                "zcu_lab.v2.support.singleshot",
            ),
            catalog_module="zcu_lab.definitions",
        ),
        logging_namespaces=("zcu_lab",),
        fluxdep_export_planner=plan_fluxdep_export,
        component_flux_context_provider=component_flux_context,
    )
