"""The registry owns injected template data and validates explicit form bindings."""

from inspect import signature

import pytest
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    Template,
    TemplateDef,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2.modules import PulseCfg

from tests.gui.app.measure.adapter._cfg_fakes import CalibrationAdapter


class TemplateCore(ConfigBase):
    probe: PulseCfg


@pytest.mark.parametrize("publish", [False, True])
def test_registry_templates_are_detached_and_preserved_across_publication(
    publish: bool,
) -> None:
    assert "templates" in signature(Registry).parameters, (
        "Missing registry template injection"
    )
    templates = {"square": TemplateDef(ConstWaveformCfg, "Square", {"length": 2})}
    registry = Registry(templates=templates)
    templates.clear()
    assert set(registry.templates) == {"square"}
    if publish:
        receiver = Registry()
        receiver.replace_from(registry)
        registry = receiver
    registry.clear()
    assert set(registry.templates) == {"square"}


def test_invalid_form_templates_reject_registration_without_publishing() -> None:
    form = Form(TemplateCore, {"probe": Template("missing")})
    registry = Registry(templates={})
    with pytest.raises(ValueError, match="synthetic.*probe"):
        registry.register("synthetic", CalibrationAdapter, form=form)
    assert not registry.has("synthetic")
