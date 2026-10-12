"""Measure adapter for shared bounded source loading; recovery stays app-owned."""

import importlib

from zcu_tools.gui.catalog_loader import SourceCatalogLoader
from zcu_tools.gui.catalog_loader import SourcePackage as SourcePackage

from .registry import Registry


class SourceExperimentCatalogLoader(SourceCatalogLoader[Registry]):
    """Build validated measure registries using the original loader signature.

    sources, reload_modules, preserved_modules and catalog_module obey
    SourceCatalogLoader's owner-thread, fixed-source and single-use contracts.
    catalog_module must export register_all(registry); registration/validation
    failures become CatalogReloadError. App recovery and tab publication remain
    with the measure Controller, not this adapter.
    """

    def __init__(
        self,
        *,
        sources: tuple[SourcePackage, ...],
        reload_modules: tuple[str, ...],
        preserved_modules: tuple[str, ...],
        catalog_module: str,
    ) -> None:
        """Capture source integrity and configure fresh measure registration."""

        def build() -> Registry:
            catalog = importlib.import_module(catalog_module)
            register = getattr(catalog, "register_all", None)
            if not callable(register):
                raise TypeError("Catalog must export register_all(registry)")
            candidate = Registry()
            register(candidate)
            candidate.validate()
            return candidate

        super().__init__(
            sources=sources,
            reload_modules=reload_modules,
            preserved_modules=preserved_modules,
            catalog_module=catalog_module,
            catalog_builder=build,
        )
