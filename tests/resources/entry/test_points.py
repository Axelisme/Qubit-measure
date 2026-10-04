"""Working-point lifecycle and layered access through ResultEntry public views."""

from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Self

import pytest
from pydantic import ValidationError, model_validator
from ruamel.yaml import YAML
from zcu_tools.resources.document_store import UnitSpec
from zcu_tools.resources.entry import ComponentSchema, ResultEntry, component_registry


class RangeSchema(ComponentSchema):
    low: Annotated[float, UnitSpec("1", "1")]
    high: Annotated[float, UnitSpec("1", "1")]

    @model_validator(mode="after")
    def check_range(self) -> Self:
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        return self


@pytest.fixture(scope="module", autouse=True)
def registry_module_guard() -> Generator[None]:
    before = deepcopy(vars(component_registry))
    yield
    assert vars(component_registry) == before, (
        "component registry polluted by points module"
    )


@pytest.fixture(autouse=True)
def registry_state_guard(request: pytest.FixtureRequest) -> Generator[None]:
    before = deepcopy(vars(component_registry))
    yield
    assert vars(component_registry) == before, (
        f"registry polluter: {request.node.nodeid}"
    )


@pytest.fixture
def range_kind(registry_state_guard: None) -> Generator[str]:
    kind = "test/point-range"
    component_registry.register(kind, RangeSchema)
    try:
        yield kind
    finally:
        component_registry.unregister(kind)


@pytest.fixture
def entry_roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "results", tmp_path / "Database"


@pytest.fixture
def entry(entry_roots: tuple[Path, Path]) -> ResultEntry:
    results, database = entry_roots
    return ResultEntry.create("entry", result_root=results, database_root=database)


def test_new_point_inherits_setup_without_copying_values(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    entry.setup.add_component("Q1", kind="qubit/transmon", freq=5000.0)
    point = entry.new_point("stable (cold)")
    assert point.Q1.freq == 5000.0
    assert point.description is None
    assert entry.list_points() == ["stable (cold)"]
    assert entry.use_point("stable (cold)").Q1.freq == 5000.0

    source = entry_roots[0] / "entry/points/stable (cold)/point.yaml"
    document = YAML(typ="safe").load(source.read_text())
    assert document["components"] == {}
    assert document["general"]["created_at"].endswith("Z")
    source.unlink()
    assert point.Q1.freq == 5000.0


def test_point_edit_routes_existing_and_new_fields_to_their_layers(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    entry.setup.add_component("Q1", kind="qubit/transmon", freq=5000.0)
    point = entry.new_point("a")
    with point.edit() as draft:
        draft.Q1.freq = 5100.0
        draft.Q1.t1 = 12.0
        draft.description = "point note"
    assert point.Q1.freq == 5100.0
    assert point.Q1.t1 == 12.0
    assert entry.setup.Q1.freq == 5100.0
    assert point.description == "point note"
    assert entry.setup.description is None

    results, database = entry_roots
    reloaded = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reloaded.use_point("a").Q1.t1 == 12.0
    assert reloaded.use_point("a").Q1.freq == 5100.0
    setup = YAML(typ="safe").load((results / "entry/setup.yaml").read_text())
    stored = YAML(typ="safe").load((results / "entry/points/a/point.yaml").read_text())
    assert setup["components"]["Q1"]["freq"] == 5.1e9
    assert stored["components"]["Q1"] == {"t1": 12e-6}


def test_setup_commit_validates_complete_views_of_existing_points(
    entry: ResultEntry, entry_roots: tuple[Path, Path], range_kind: str
) -> None:
    entry.setup.add_component("R1", kind=range_kind, low=1.0, high=2.0)
    entry.new_point("a")
    results, _ = entry_roots
    setup_source = results / "entry/setup.yaml"
    point_source = results / "entry/points/a/point.yaml"
    yaml = YAML(typ="rt")
    setup = yaml.load(setup_source.read_text())
    stored = yaml.load(point_source.read_text())
    stored["components"]["R1"] = {"high": setup["components"]["R1"].pop("high")}
    with setup_source.open("w") as stream:
        yaml.dump(setup, stream)
    with point_source.open("w") as stream:
        yaml.dump(stored, stream)
    entry.setup.refresh()
    point = entry.use_point("a")
    assert point.R1.high == 2.0
    before = setup_source.read_bytes(), point_source.read_bytes()

    with (
        pytest.raises(ValidationError, match="low must not exceed high"),
        entry.setup.edit() as draft,
    ):
        draft.R1.low = 3.0
    assert (setup_source.read_bytes(), point_source.read_bytes()) == before
    assert point.R1.low == 1.0
    assert point.R1.high == 2.0
