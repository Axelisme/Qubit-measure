"""Registered notebook models in partial setup documents, with registry custody."""

from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import Annotated

import pytest
from ruamel.yaml import YAML
from zcu_tools.resources.document_store import UnitSpec
from zcu_tools.resources.entry import ComponentSchema, ResultEntry, component_registry


class RequiredPhysicalSchema(ComponentSchema):
    freq: Annotated[float, UnitSpec("Hz", "MHz")]
    title: str


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


def test_required_registered_fields_can_be_filled_incrementally_in_setup(
    tmp_path: Path,
    required_kind: str,
) -> None:
    results, database = tmp_path / "results", tmp_path / "Database"
    entry = ResultEntry.create("entry", result_root=results, database_root=database)
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
