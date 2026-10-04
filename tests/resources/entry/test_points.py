"""Working-point lifecycle and layered access through ResultEntry public views."""

from collections.abc import Generator
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal, Self

import pytest
from pydantic import BaseModel, ValidationError, ValidationInfo, model_validator
from ruamel.yaml import YAML
from zcu_tools.resources.document_store import UnitSpec
from zcu_tools.resources.entry import (
    ComponentSchema,
    LayerConflictError,
    ResultEntry,
    component_registry,
)


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


def write_yaml(source: Path, document: object) -> None:
    with source.open("w") as stream:
        YAML(typ="rt").dump(document, stream)


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


@pytest.mark.parametrize("operation", ["refresh", "use_point"])
def test_point_reload_uses_latest_layers_and_keeps_snapshots_on_failure(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    operation: Literal["refresh", "use_point"],
) -> None:
    entry.setup.add_component("Q1", kind="qubit/transmon", freq=5000.0)
    point = entry.new_point("a")
    point.Q1.t1 = 10.0
    results, database = entry_roots
    other = ResultEntry.open("entry", result_root=results, database_root=database)
    with other.use_point("a").edit() as draft:
        draft.Q1.freq = 5100.0
        draft.Q1.t1 = 12.0
    assert point.Q1.freq == 5000.0
    assert point.Q1.t1 == 10.0
    if operation == "refresh":
        point.refresh()
    else:
        point = entry.use_point("a")
    assert point.Q1.freq == 5100.0
    assert point.Q1.t1 == 12.0
    assert entry.setup.Q1.freq == 5100.0

    other.setup.Q1.freq = 5200.0
    source = results / "entry/points/a/point.yaml"
    yaml = YAML(typ="rt")
    stored = yaml.load(source.read_text())
    stored["components"]["Q1"]["freq"] = 5.2e9
    with source.open("w") as stream:
        yaml.dump(stored, stream)
    reload_point = (
        point.refresh if operation == "refresh" else lambda: entry.use_point("a")
    )
    with pytest.raises(LayerConflictError, match="Q1.freq"):
        reload_point()
    assert point.Q1.freq == 5100.0
    assert point.Q1.t1 == 12.0
    assert entry.setup.Q1.freq == 5100.0


def test_complete_validation_rejects_after_validator_mutating_unset_default(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    registry_state_guard: None,
) -> None:
    class MutatingSchema(ComponentSchema):
        @model_validator(mode="after")
        def mutate_default(self) -> Self:
            self.ext["note"] = "changed by validator"
            return self

    kind = "test/point-mutating-default"
    component_registry.register(kind, MutatingSchema)
    try:
        entry.setup.add_component("C1", kind=kind)
        source = entry_roots[0] / "entry/setup.yaml"
        before = source.read_bytes()
        with pytest.raises(ValidationError, match="C1.ext"):
            entry.new_point("invalid")
        assert entry.list_points() == []
        assert not (entry_roots[0] / "entry/points/invalid").exists()
        assert source.read_bytes() == before
        with pytest.raises(KeyError, match="note"):
            entry.setup.C1.ext["note"]
    finally:
        component_registry.unregister(kind)


def test_move_transfers_value_and_provenance_between_layers(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
) -> None:
    entry.setup.add_component("Q1", kind="qubit/transmon", freq=5000.0)
    results, database = entry_roots
    setup_source = results / "entry/setup.yaml"
    point_source = results / "entry/points/a/point.yaml"
    yaml = YAML(typ="rt")
    stored = yaml.load(setup_source.read_text())
    provenance = {
        "source": "manual",
        "kind": None,
        "run_id": None,
        "at": "2026-10-04T00:00:00Z",
        "stderr": None,
    }
    stored["provenance"]["Q1.freq"] = provenance
    with setup_source.open("w") as stream:
        yaml.dump(stored, stream)
    entry.setup.refresh()
    point = entry.new_point("a")

    point.move("Q1.freq", to="point")
    assert point.Q1.freq == 5000.0
    with pytest.raises(AttributeError, match="Q1.freq"):
        _ = entry.setup.Q1.freq
    setup = yaml.load(setup_source.read_text())
    stored = yaml.load(point_source.read_text())
    assert "freq" not in setup["components"]["Q1"]
    assert "Q1.freq" not in setup["provenance"]
    assert stored["components"]["Q1"]["freq"] == 5e9
    assert stored["provenance"]["Q1.freq"] == provenance
    before = setup_source.read_bytes(), point_source.read_bytes()
    with pytest.raises(ValueError, match="Q1.freq"):
        point.move("Q1.freq", to="point")
    assert (setup_source.read_bytes(), point_source.read_bytes()) == before

    point.move("Q1.freq", to="setup")
    reloaded = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reloaded.setup.Q1.freq == 5000.0
    assert reloaded.use_point("a").Q1.freq == 5000.0
    stored = yaml.load(point_source.read_text())
    setup = yaml.load(setup_source.read_text())
    assert "freq" not in stored["components"].get("Q1", {})
    assert "Q1.freq" not in stored["provenance"]
    assert setup["provenance"]["Q1.freq"] == provenance


def test_point_edit_rejects_invalid_reload_before_exposing_draft(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    range_kind: str,
) -> None:
    entry.setup.add_component("R1", kind=range_kind, low=1.0, high=2.0)
    point = entry.new_point("a")
    setup_source = entry_roots[0] / "entry/setup.yaml"
    point_source = entry_roots[0] / "entry/points/a/point.yaml"
    yaml = YAML(typ="rt")
    stored = yaml.load(setup_source.read_text())
    stored["components"]["R1"]["low"] = 3.0
    with setup_source.open("w") as stream:
        yaml.dump(stored, stream)
    before = setup_source.read_bytes(), point_source.read_bytes()
    entered: list[bool] = []
    with (
        pytest.raises(ValidationError, match="low must not exceed high"),
        point.edit(),
    ):
        entered.append(True)
    assert entered == []
    assert point.R1.low == 1.0
    assert point.R1.high == 2.0
    assert (setup_source.read_bytes(), point_source.read_bytes()) == before


@pytest.mark.parametrize("source_mode", ["label", "view"])
def test_clone_copies_only_point_and_module_and_rejects_existing_destination(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    source_mode: Literal["label", "view"],
) -> None:
    entry.setup.add_component("Q1", kind="qubit/transmon", freq=5000.0)
    point = entry.new_point("source")
    with point.edit() as draft:
        draft.Q1.t1 = 12.0
        draft.description = "source environment"
    root = entry_roots[0] / "entry/points"
    source = root / "source"
    (source / "module_cfg.yaml").write_text(
        "format: zcu.module-library\nformat_version: '1.0'\nQ1: {pulse: arbitrary}\n"
    )
    (source / "data.h5").write_bytes(b"not cloned")
    (source / "image.png").write_bytes(b"not cloned")
    source_arg = "source" if source_mode == "label" else point
    cloned = entry.new_point("target", clone_from=source_arg)
    assert cloned.Q1.freq == 5000.0
    assert cloned.Q1.t1 == 12.0
    assert cloned.description == "source environment"
    destination = root / "target"
    assert {path.name for path in destination.iterdir()} == {
        "point.yaml",
        "module_cfg.yaml",
    }
    assert (destination / "module_cfg.yaml").read_bytes() == (
        source / "module_cfg.yaml"
    ).read_bytes()
    stored = YAML(typ="safe").load((destination / "point.yaml").read_text())
    assert stored["components"]["Q1"] == {"t1": 12e-6}
    before = (destination / "point.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        entry.new_point("target", clone_from=source_arg)
    assert (destination / "point.yaml").read_bytes() == before
    assert entry.list_points() == ["source", "target"]


def test_clone_marks_only_point_provenance_with_its_direct_source(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
) -> None:
    entry.setup.add_component("Q1", kind="qubit/transmon", freq=5000.0)
    point = entry.new_point("a")
    point.Q1.t1 = 12.0
    root = entry_roots[0] / "entry"
    yaml = YAML(typ="rt")
    source = root / "points/a/point.yaml"
    stored = yaml.load(source.read_text())
    original = {
        "source": "manual",
        "kind": None,
        "run_id": None,
        "at": "2026-10-04T00:00:00Z",
        "stderr": 1e-6,
    }
    stored["provenance"]["Q1.t1"] = original
    write_yaml(source, stored)
    setup = yaml.load((root / "setup.yaml").read_text())
    setup["provenance"]["Q1.freq"] = {**original, "stderr": None}
    write_yaml(root / "setup.yaml", setup)

    entry.new_point("b", clone_from=point)
    cloned = yaml.load((root / "points/b/point.yaml").read_text())
    assert cloned["provenance"]["Q1.t1"] == {
        **original,
        "cloned_from": {"entry_id": entry.entry_id, "point": "a"},
    }
    assert "Q1.freq" not in cloned["provenance"]
    entry.new_point("c", clone_from="b")
    cloned_again = yaml.load((root / "points/c/point.yaml").read_text())
    assert cloned_again["provenance"]["Q1.t1"] == {
        **original,
        "cloned_from": {"entry_id": entry.entry_id, "point": "b"},
    }
    assert (
        "cloned_from"
        not in yaml.load((root / "setup.yaml").read_text())["provenance"]["Q1.freq"]
    )


@pytest.mark.parametrize("nullable", [False, True])
def test_complete_nested_after_validator_reports_field_and_keeps_partial_setup(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    registry_state_guard: None,
    nullable: bool,
) -> None:
    class Details(BaseModel):
        freq: Annotated[float, UnitSpec("Hz", "MHz")]

        @model_validator(mode="after")
        def mutate(self, info: ValidationInfo) -> Self:
            self.freq += 1.0
            return self

    class DirectSchema(ComponentSchema):
        details: Details

    class NullableSchema(ComponentSchema):
        details: Details | None

    kind = "test/point-nested-after"
    component_registry.register(kind, NullableSchema if nullable else DirectSchema)
    try:
        entry.setup.add_component("C1", kind=kind, details={"freq": 12.0})
        source = entry_roots[0] / "entry/setup.yaml"
        before = source.read_bytes()
        with pytest.raises(ValidationError, match="C1.details.freq") as error:
            entry.new_point("invalid")
        assert "point.yaml" in str(error.value)
        assert entry.setup.C1.details == {"freq": 12.0}
        assert source.read_bytes() == before
        assert entry.list_points() == []
    finally:
        component_registry.unregister(kind)


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
