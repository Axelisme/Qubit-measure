"""The registry owns injected template data across detached publication."""

from inspect import signature

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, TemplateDef
from zcu_tools.gui.app.measure.registry import Registry

from tests.gui.app.measure._native_save_support import AnalysisSavingAdapter, SavingAdapter


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


def test_resolve_experiment_prefers_exact_name_over_loadable_tag() -> None:
    registry = Registry()
    registry.register("saving", AnalysisSavingAdapter)
    registry.register("test/saving", SavingAdapter)
    resolution = registry.resolve_experiment("test/saving")
    assert resolution.adapter_name == "test/saving"
    assert resolution.reason is None


def test_resolve_experiment_selects_unique_load_capable_tag_owner() -> None:
    registry = Registry()
    registry.register("saving", AnalysisSavingAdapter)
    registry.register("non-loading", SavingAdapter)
    resolution = registry.resolve_experiment("test/saving")
    assert resolution.adapter_name == "saving"
    assert resolution.reason is None


@pytest.mark.parametrize("experiment", ["old_tag", None])
def test_resolve_experiment_explains_unknown_or_missing_identity(
    experiment: str | None,
) -> None:
    registry = Registry()
    registry.register("non-loading", SavingAdapter)
    resolution = registry.resolve_experiment(experiment)
    assert resolution.adapter_name is None
    assert resolution.reason
    if experiment:
        assert experiment in resolution.reason
    else:
        assert "Missing experiment" in resolution.reason


def test_resolve_experiment_rejects_ambiguous_tag_without_guessing() -> None:
    registry = Registry()
    registry.register("saving", AnalysisSavingAdapter)
    registry.register("alternative", AnalysisSavingAdapter)
    resolution = registry.resolve_experiment("test/saving")
    assert resolution.adapter_name is None
    assert resolution.reason
    assert all(name in resolution.reason for name in ("saving", "alternative", "test/saving"))

