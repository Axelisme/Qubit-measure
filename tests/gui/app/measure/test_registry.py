"""The registry owns injected template data across detached publication."""

from inspect import signature

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, TemplateDef
from zcu_tools.gui.app.measure.registry import Registry


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
