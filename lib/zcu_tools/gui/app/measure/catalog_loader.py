"""Measure adapter for shared bounded source loading; recovery stays app-owned."""

import importlib

from zcu_tools.gui.catalog_loader import SourceCatalogLoader
from zcu_tools.gui.catalog_loader import SourcePackage as SourcePackage

from .registry import Registry


class SourceExperimentCatalogLoader(SourceCatalogLoader[Registry]):
    """Build detached validated registries using the current injected templates.

    sources, reload_modules, preserved_modules and catalog_module obey
    SourceCatalogLoader's owner-thread, fixed-source and single-use contracts.
    current_registry is the app's stable registry identity. Each load copies its
    templates into a fresh candidate; clearing adapters does not lose templates.
    The loader never publishes or mutates current_registry. catalog_module must
    export register_all(registry) and explicitly register each adapter's form.
    Registration/validation failures become CatalogReloadError. App recovery and
    tab publication remain with the measure Controller, not this adapter.
    """

    def __init__(
        self,
        *,
        sources: tuple[SourcePackage, ...],
        reload_modules: tuple[str, ...],
        preserved_modules: tuple[str, ...],
        catalog_module: str,
        current_registry: Registry,
    ) -> None:
        """Monitor explicit sources and borrow current_registry's template catalog.

        Construction and all loads require the owner thread. Source declaration
        errors follow SourceCatalogLoader; the registry is only read during load.
        """

        def build() -> Registry:
            catalog = importlib.import_module(catalog_module)
            register = getattr(catalog, "register_all", None)
            if not callable(register):
                raise TypeError("Catalog must export register_all(registry)")
            candidate = Registry(templates=current_registry.templates)
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
