"""Measure's original signature adapts fresh shared loading into Registry."""

import importlib
import sys
from pathlib import Path

from zcu_tools.gui.app.measure.catalog_loader import (
    SourceExperimentCatalogLoader,
    SourcePackage,
)


def test_measure_adapter_keeps_registration_and_validation_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "measure_loader_adapter_fixture"
    root.mkdir()
    (root / "__init__.py").write_text("")
    path = root / "catalog.py"
    source = """from zcu_tools.gui.app.measure.adapter import (
    ExpAdapterProtocol, AdapterCapabilities, AdapterGuide,
)
class Adapter(ExpAdapterProtocol):
    capabilities = AdapterCapabilities()
    @classmethod
    def guide(cls):
        return AdapterGuide("VERSION", "", "", "", "")
def register_all(registry):
    registry.register("demo", Adapter)
"""
    path.write_text(source.replace("VERSION", "v1"))
    sys.path.insert(0, str(tmp_path))
    try:
        loader = SourceExperimentCatalogLoader(
            sources=(SourcePackage("measure_loader_adapter_fixture", root),),
            reload_modules=("measure_loader_adapter_fixture.catalog",),
            preserved_modules=(),
            catalog_module="measure_loader_adapter_fixture.catalog",
        )
        first = loader.load(loader.prepare())
        assert first.create("demo").guide().behavior == "v1"
        path.write_text(source.replace("VERSION", "v2"))
        second = loader.load(loader.prepare())
        assert second.create("demo").guide().behavior == "v2"
        assert first.create("demo").guide().behavior == "v1"
    finally:
        sys.path.remove(str(tmp_path))
        for name in tuple(sys.modules):
            if name == "measure_loader_adapter_fixture" or name.startswith(
                "measure_loader_adapter_fixture."
            ):
                del sys.modules[name]
        importlib.invalidate_caches()
