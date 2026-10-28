"""Lab-supplied definitions required to compose the measure GUI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from zcu_tools.cfg_model import TemplateDef
from zcu_tools.gui.catalog_loader import SourcePackage
from zcu_tools.resources.entry.registry import ComponentRegistry

from .registry import Registry
from .services.workflow_attachment import Attachment


@dataclass(frozen=True)
class CatalogReloadPolicy:
    """Bounded source declarations for SourceExperimentCatalogLoader.

    sources lists monitored package namespaces and their installed directories.
    reload_modules lists nonempty module prefixes eligible for reload.
    preserved_modules lists fixed prefixes, overriding reload_modules; empty is
    allowed. catalog_module names the importable register_all(registry) module
    inside the effective reload scope. The loader validates names, directories
    and scope at construction and reports integrity failures at prepare/load.
    """

    sources: tuple[SourcePackage, ...]
    reload_modules: tuple[str, ...]
    preserved_modules: tuple[str, ...]
    catalog_module: str


class MeasureGuiLab[C](Protocol):
    """Measure definitions supplied by a lab or a synthetic test description.

    C is the opaque context shared by workflow attachments. Registrars install
    definitions only, without constructing forms or acquiring hardware. Their
    failures propagate from behavior construction. Empty templates and workflow
    attachments are valid and leave those catalogs empty.
    """

    @property
    def register_components(self) -> Callable[[ComponentRegistry], None]:
        """Install component models/roles into the process component registry."""
        ...

    @property
    def register_experiments(self) -> Callable[[Registry], None]:
        """Install adapters into a fresh registry already holding templates."""
        ...

    @property
    def templates(self) -> Mapping[str, TemplateDef]:
        """Return named persistent module/waveform templates; empty is valid."""
        ...

    @property
    def workflow_attachments(self) -> tuple[Attachment[C], ...]:
        """Return ordered native/form pairs; the first is the default workflow."""
        ...

    @property
    def catalog_reload_policy(self) -> CatalogReloadPolicy:
        """Return source/reload boundaries for the experiment catalog loader."""
        ...
