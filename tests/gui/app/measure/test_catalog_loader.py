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

from tests.gui.app.measure.remote._helpers import Fixture


def test_measure_adapter_keeps_registration_and_validation_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "measure_loader_adapter_fixture"
    root.mkdir()
    (root / "__init__.py").write_text("")
    path = root / "catalog.py"
    source = """from pathlib import Path
from zcu_tools.gui.app.measure.adapter import Adapter as Definition
from tests.gui._adapter_fakes import DummyAdapter
guide = Path(__file__).with_name("guide-VERSION.md")
guide.write_text("VERSION")
Adapter = Definition(spec=DummyAdapter.spec, make_form=DummyAdapter.make_form,
                     run=DummyAdapter.run, guide=guide)
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
        assert first.create("demo").guide().markdown == "v1"
        path.write_text(source.replace("VERSION", "v2"))
        second = loader.load(loader.prepare())
        assert second.create("demo").guide().markdown == "v2"
        assert first.create("demo").guide().markdown == "v1"
    finally:
        sys.path.remove(str(tmp_path))
        for name in tuple(sys.modules):
            if name == "measure_loader_adapter_fixture" or name.startswith(
                "measure_loader_adapter_fixture."
            ):
                del sys.modules[name]
        importlib.invalidate_caches()


def test_reload_keeps_templates_for_open_and_reset(tmp_path: Path, qapp) -> None:
    root = tmp_path / "measure_template_loader_fixture"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "catalog.py").write_text(
        """from zcu_tools.cfg_model import ConfigBase, Template, PulseCfg
from zcu_tools.experiment import AxesSpec, ZSpec
from zcu_tools.gui.app.measure.adapter import Adapter as Definition
from zcu_tools.gui.form_adapter import Form, FormContext
from tests.gui._adapter_fakes import DummyResult
class Cfg(ConfigBase):
    probe: PulseCfg
def make_form(ctx: FormContext) -> Form[Cfg]:
    return Form(Cfg, {"probe": Template("probe")})
Adapter = Definition(spec=AxesSpec(axes=(), z=ZSpec("value", "value", ""),
    result_type=DummyResult, cfg_type=Cfg, tag="test/template"),
    make_form=make_form, run=lambda cfg, *, context: DummyResult(), guide=None)
def register_all(registry):
    registry.register("demo", Adapter)
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
        application = Fixture(headless=True, project_root=str(tmp_path / "project"))
        try:
            application.registry.replace_from(candidate)
            tab_id = application.ctrl.new_tab("demo")
            resource = application.ctrl.cfg_resources.lookup(tab_id)
            assert resource.read_module(("probe",)).value["gain"] == 0.25
            current.replace_from(candidate)
            second = loader.load(loader.prepare())
            assert second is not candidate
            assert second.templates == current.templates
            application.registry.replace_from(second)
            resource.reset()
            assert resource.read_module(("probe",)).value["gain"] == 0.25
        finally:
            application.stop()
            for tab_id in tuple(application.ctrl.list_tab_ids()):
                application.ctrl.close_tab(tab_id)
    finally:
        sys.path.remove(str(tmp_path))
        for name in tuple(sys.modules):
            if name == "measure_template_loader_fixture" or name.startswith(
                "measure_template_loader_fixture."
            ):
                del sys.modules[name]
        importlib.invalidate_caches()
