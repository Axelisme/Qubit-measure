"""Registered notebook models in partial setup documents, with registry custody."""

from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from ruamel.yaml import YAML
from zcu_tools.resources.document_store import UnitSpec
from zcu_tools.resources.entry import ComponentSchema, ResultEntry, component_registry


class RequiredPhysicalSchema(ComponentSchema):
    freq: Annotated[float, UnitSpec("Hz", "MHz")]
    title: str


class RequiredTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: Annotated[float, UnitSpec("s", "us")]
    label: str


class RequiredNestedSchema(ComponentSchema):
    timing: RequiredTiming


@pytest.fixture(scope="module", autouse=True)
def registry_module_guard() -> Generator[None]:
    before = deepcopy(vars(component_registry))
    yield
    assert vars(component_registry) == before, (
        "component registry polluted by this module"
    )


@pytest.fixture(autouse=True)
def registry_state_guard(
    request: pytest.FixtureRequest,
) -> Generator[None]:
    before = deepcopy(vars(component_registry))
    yield
    assert vars(component_registry) == before, (
        f"registry polluter: {request.node.nodeid}"
    )


@pytest.fixture
def required_kind(registry_state_guard: None) -> Generator[str]:
    kind = "notebook/required"
    component_registry.register(kind, RequiredPhysicalSchema)
    try:
        yield kind
    finally:
        component_registry.unregister(kind)


@pytest.fixture
def nested_kind(registry_state_guard: None) -> Generator[str]:
    kind = "notebook/nested"
    component_registry.register(kind, RequiredNestedSchema)
    try:
        yield kind
    finally:
        component_registry.unregister(kind)


def create_entry(tmp_path: Path) -> tuple[ResultEntry, Path, Path]:
    results, database = tmp_path / "results", tmp_path / "Database"
    entry = ResultEntry.create("entry", result_root=results, database_root=database)
    return entry, results, database


@pytest.mark.parametrize("value", [None, "not a frequency"])
def test_partial_setup_still_validates_supplied_required_values(
    tmp_path: Path,
    required_kind: str,
    value: str | None,
) -> None:
    entry, results, _ = create_entry(tmp_path)
    entry.setup.add_component("N1", kind=required_kind)
    setup_file = results / "entry" / "setup.yaml"
    before = setup_file.read_bytes()
    with pytest.raises(ValidationError, match="freq"):
        entry.setup.N1.freq = value
    assert setup_file.read_bytes() == before
    with pytest.raises(AttributeError, match="not set"):
        _ = entry.setup.N1.freq


def test_nested_required_fields_are_deferred_and_preserve_nested_unit_metadata(
    tmp_path: Path,
    nested_kind: str,
) -> None:
    entry, results, database = create_entry(tmp_path)
    entry.setup.add_component("N1", kind=nested_kind, timing={"label": "prepared"})
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert "width" not in document["components"]["N1"]["timing"]

    entry.setup.N1.timing = {"width": 10.0, "label": "prepared"}
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.N1.kind == nested_kind
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert document["components"]["N1"]["timing"]["width"] == pytest.approx(1e-5)


def test_required_registered_fields_can_be_filled_incrementally_in_setup(
    tmp_path: Path,
    required_kind: str,
) -> None:
    entry, results, database = create_entry(tmp_path)
    entry.setup.add_component("N1", kind=required_kind)
    with pytest.raises(AttributeError, match="not set"):
        _ = entry.setup.N1.freq
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert "freq" not in document["components"]["N1"]
    assert "title" not in document["components"]["N1"]

    entry.setup.N1.freq = 5000.0
    entry.setup.N1.title = "prepared later"
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.N1.freq == pytest.approx(5000.0)
    assert reopened.setup.N1.title == "prepared later"
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert document["components"]["N1"]["freq"] == pytest.approx(5e9)
    assert component_registry.get(required_kind) is RequiredPhysicalSchema
