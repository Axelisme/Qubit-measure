"""Measure reload creates detached registries with the injected template catalog."""

import importlib
import sys
from pathlib import Path

import pytest
from zcu_tools.cfg_model import PulseSpec, TemplateDef
from zcu_tools.gui.app.measure.catalog_loader import (
    SourceExperimentCatalogLoader,
    SourcePackage,
)
from zcu_tools.gui.app.measure.registry import Registry


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
            current_registry=Registry(),
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


def test_reload_binds_forms_to_current_registry_templates(tmp_path: Path) -> None:
    root = tmp_path / "measure_template_loader_fixture"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "catalog.py").write_text(
        """from zcu_tools.cfg_model import ConfigBase, Template
from zcu_tools.program.v2.modules import PulseCfg
from zcu_tools.gui.app.measure.adapter import (
    ExpAdapterProtocol, AdapterCapabilities, AdapterGuide, Form,
)
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
class Cfg(ConfigBase):
    probe: PulseCfg
class Adapter(ExpAdapterProtocol):
    capabilities = AdapterCapabilities()
    form = Form(Cfg, {"probe": Template("probe")})
    @classmethod
    def guide(cls):
        env = FormEnv()
        result = FormState(cls.form.default_input(env)).result(env)
        if isinstance(result, FormErrors):
            raise ValueError(str(result.messages))
        cfg = cls.form.to_cfg(result, env)
        if isinstance(cfg, FormErrors):
            raise ValueError(str(cfg.messages))
        return AdapterGuide(str(cfg.probe.gain), "", "", "", "")
def register_all(registry):
    registry.register("demo", Adapter, form=Adapter.form)
"""
    )
    template = TemplateDef(
        PulseSpec,
        "Probe",
        {
            "ch": 1,
            "nqz": 1,
            "freq": 123.0,
            "gain": 0.25,
            "waveform.style": "const",
            "waveform.length": 0.1,
        },
    )
    current = Registry(templates={"probe": template})
    sys.path.insert(0, str(tmp_path))
    try:
        try:
            loader = SourceExperimentCatalogLoader(
                sources=(SourcePackage("measure_template_loader_fixture", root),),
                reload_modules=("measure_template_loader_fixture.catalog",),
                preserved_modules=(),
                catalog_module="measure_template_loader_fixture.catalog",
                current_registry=current,
            )
        except TypeError as exc:
            pytest.fail(
                f"Reload must accept the current registry's template catalog: {exc}"
            )
        current.clear()
        candidate = loader.load(loader.prepare())
        assert candidate is not current
        assert candidate.templates == {"probe": template}
        assert current.list_names() == []
        assert candidate.list_names() == ["demo"]
        # The adapter's guide observes a cfg materialized through its bound form.
        assert candidate.create("demo").guide().behavior == "0.25"
        current.replace_from(candidate)
        second = loader.load(loader.prepare())
        assert second is not candidate
        assert second.templates == current.templates
        assert second.create("demo").guide().behavior == "0.25"
    finally:
        sys.path.remove(str(tmp_path))
        for name in tuple(sys.modules):
            if name == "measure_template_loader_fixture" or name.startswith(
                "measure_template_loader_fixture."
            ):
                del sys.modules[name]
        importlib.invalidate_caches()
