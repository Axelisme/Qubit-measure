"""Validate explicit registration through the framework catalog seams."""

from collections.abc import Generator
from copy import deepcopy
from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog
from zcu_tools.resources.entry import (
    ComponentRegistry,
    ResultEntry,
    UnknownKindError,
    component_registry,
    role_registry,
)

from zcu_lab.definitions import register_all


def shared_registry_state() -> tuple[dict[str, object], dict[str, object]]:
    """Capture all shared kind and role settings for fixture pollution checks."""
    return deepcopy(
        (
            {
                name: value
                for name, value in vars(component_registry).items()
                if name != "roles"
            },
            vars(role_registry),
        )
    )


@pytest.fixture(scope="module", autouse=True)
def shared_registry_module_guard() -> Generator[None]:
    before = shared_registry_state()
    yield
    assert shared_registry_state() == before, "definitions tests polluted registries"


@pytest.fixture
def empty_shared_components(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Generator[ComponentRegistry]:
    before = shared_registry_state()
    empty = ComponentRegistry()
    with monkeypatch.context() as patch:
        for name, value in vars(empty).items():
            if name != "roles":
                patch.setattr(component_registry, name, value)
        for name, value in vars(empty.roles).items():
            patch.setattr(role_registry, name, value)
        yield component_registry
    assert shared_registry_state() == before, (
        f"registry polluter: {request.node.nodeid}"
    )


@pytest.mark.parametrize("startup", [False, True], ids=["reload", "startup"])
def test_registration_produces_valid_catalog(startup: bool) -> None:
    registry = Registry()
    templates = TemplateCatalog() if startup else None

    register_all(registry, templates=templates)

    assert registry.list_names()
    registry.validate()


def test_omitted_components_leave_shared_container_registry_unregistered(
    empty_shared_components: ComponentRegistry,
) -> None:
    registry = Registry()
    register_all(registry)
    assert registry.list_names()
    with pytest.raises(UnknownKindError):
        empty_shared_components.get("qubit/transmon")
    assert role_registry.shorthand == ()
    assert role_registry.focus_kinds == ()


@pytest.mark.parametrize("templates", [False, True])
def test_explicit_shared_components_are_available_to_entry_and_role_resolution(
    empty_shared_components: ComponentRegistry, tmp_path: Path, templates: bool
) -> None:
    registry = Registry()
    register_all(
        registry,
        templates=TemplateCatalog() if templates else None,
        components=empty_shared_components,
    )
    entry = ResultEntry.create(
        "bootstrap", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    entry.setup.add_component("C1", kind="qubit/transmon")
    point = entry.new_point("working")
    roles = point.resolve(["qubit"])
    assert roles.components == {"qubit": "C1"}
    assert roles.qubit.kind == point.C1.kind
    assert registry.list_names()
    registry.validate()


def test_reload_omits_component_bootstrap_and_preserves_registered_identities(
    empty_shared_components: ComponentRegistry,
) -> None:
    registry = Registry()
    register_all(registry, components=empty_shared_components)
    model = empty_shared_components.get("qubit/transmon")
    role = role_registry.get("qubit")
    shorthand = role_registry.shorthand
    focus = role_registry.focus_kinds
    registry.clear()

    register_all(registry)

    assert empty_shared_components.get("qubit/transmon") is model
    assert role_registry.get("qubit") is role
    assert role_registry.shorthand == shorthand
    assert role_registry.focus_kinds == focus
    assert registry.list_names()
    registry.validate()
    with pytest.raises(ValueError, match="already registered"):
        register_all(Registry(), components=empty_shared_components)
