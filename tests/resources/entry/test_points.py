"""Working-point lifecycle and layered access through ResultEntry public views."""

import shutil
from collections.abc import Generator
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal, Self

import pytest
from pydantic import BaseModel, ValidationError, ValidationInfo, model_validator
from ruamel.yaml import YAML
from zcu_tools.resources.document_store import ConflictError, UnitSpec
from zcu_tools.resources.entry import (
    ComponentSchema,
    LayerConflictError,
    PartialCommitError,
    PointView,
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


@pytest.fixture
def working_point(entry: ResultEntry) -> PointView:
    entry.setup.add_component("Q1", kind="qubit/transmon", freq=5000.0)
    point = entry.new_point("a")
    point.Q1.t1 = 12.0
    return point


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


@pytest.mark.parametrize("nested", [False, True])
def test_move_transfers_value_and_provenance_between_layers(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    nested: bool,
) -> None:
    entry.setup.add_component(
        "Q1",
        kind="qubit/transmon",
        freq=5000.0,
        wiring={"time_of_flight": 2.0, "flux_ch": 3},
    )
    path = "Q1.wiring.time_of_flight" if nested else "Q1.freq"
    leaf = "time_of_flight" if nested else "freq"
    value = 2.0 if nested else 5000.0
    si_value = 2e-6 if nested else 5e9
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
    stored["provenance"][path] = provenance
    with setup_source.open("w") as stream:
        yaml.dump(stored, stream)
    entry.setup.refresh()
    point = entry.new_point("a")

    point.move(path, to="point")
    assert (point.Q1.wiring.time_of_flight if nested else point.Q1.freq) == value
    with pytest.raises(AttributeError, match=path):
        _ = entry.setup.Q1.wiring.time_of_flight if nested else entry.setup.Q1.freq
    setup = yaml.load(setup_source.read_text())
    stored = yaml.load(point_source.read_text())
    setup_fields = setup["components"]["Q1"]
    point_fields = stored["components"]["Q1"]
    assert leaf not in (setup_fields.get("wiring", {}) if nested else setup_fields)
    assert path not in setup["provenance"]
    assert (point_fields["wiring"] if nested else point_fields)[leaf] == si_value
    assert stored["provenance"][path] == provenance
    assert entry.setup.Q1.wiring.flux_ch == 3
    before = setup_source.read_bytes(), point_source.read_bytes()
    with pytest.raises(ValueError, match=path):
        point.move(path, to="point")
    assert (setup_source.read_bytes(), point_source.read_bytes()) == before

    point.move(path, to="setup")
    reloaded = ResultEntry.open("entry", result_root=results, database_root=database)
    reopened_point = reloaded.use_point("a")
    assert (
        reloaded.setup.Q1.wiring.time_of_flight if nested else reloaded.setup.Q1.freq
    ) == value
    assert (
        reopened_point.Q1.wiring.time_of_flight if nested else reopened_point.Q1.freq
    ) == value
    assert reopened_point.Q1.wiring.flux_ch == 3
    stored = yaml.load(point_source.read_text())
    setup = yaml.load(setup_source.read_text())
    point_fields = stored["components"].get("Q1", {})
    assert leaf not in (point_fields.get("wiring", {}) if nested else point_fields)
    assert path not in stored["provenance"]
    assert setup["provenance"][path] == provenance


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


@pytest.mark.parametrize(
    ("write", "path"),
    [
        ("attribute", "Q1.t1"),
        ("edit", "Q1.t1"),
        ("set", "Q1.t1"),
        ("set", "Q1.wiring.flux_ch"),
        ("set", "Q1.ext.nested.note"),
    ],
)
def test_rewriting_same_cloned_value_clears_only_accepted_field_origin(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    write: str,
    path: str,
) -> None:
    entry.setup.add_component("Q1", kind="qubit/transmon", freq=5000.0)
    original = entry.new_point("a")
    original.Q1.t1 = 12.0
    original.Q1.t2 = 13.0
    original.Q1.wiring.flux_ch = 3
    original.Q1.ext.nested = {"note": "accepted"}
    root = entry_roots[0] / "entry"
    source = root / "points/a/point.yaml"
    yaml = YAML(typ="rt")
    stored = yaml.load(source.read_text())
    provenance = {
        "source": "manual",
        "kind": None,
        "run_id": None,
        "at": "2026-10-04T00:00:00Z",
        "stderr": None,
    }
    origin_paths = ["Q1.t1", "Q1.t2", "Q1.wiring.flux_ch", "Q1.ext.nested.note"]
    stored["provenance"] = {
        origin_path: deepcopy(provenance) for origin_path in origin_paths
    }
    write_yaml(source, stored)
    point = entry.new_point("b", clone_from="a")
    setup_before = (root / "setup.yaml").read_bytes()

    if write == "attribute":
        with pytest.raises(ValidationError):
            point.Q1.t2 = "not-a-number"
        point.Q1.t1 = 12.0
    else:
        with point.edit() as draft:
            if write == "edit":
                with pytest.raises(ValidationError):
                    draft.Q1.t2 = "not-a-number"
                draft.Q1.t1 = 12.0
            else:
                with pytest.raises(ValidationError):
                    draft.set("Q1.t2", "not-a-number")
                value = (
                    "accepted"
                    if path == "Q1.ext.nested.note"
                    else 3
                    if path == "Q1.wiring.flux_ch"
                    else 12.0
                )
                draft.set(path, value)

    rewritten = yaml.load((root / "points/b/point.yaml").read_text())
    assert rewritten["provenance"][path] == provenance
    for untouched in set(origin_paths) - {path}:
        assert rewritten["provenance"][untouched] == {
            **provenance,
            "cloned_from": {"entry_id": entry.entry_id, "point": "a"},
        }
    assert (root / "setup.yaml").read_bytes() == setup_before
    for view in (point, entry.use_point("b")):
        assert view.Q1.t1 == 12.0
        assert view.Q1.t2 == 13.0
        assert view.Q1.wiring.flux_ch == 3
        assert view.Q1.ext.nested == {"note": "accepted"}


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


def test_clone_copy_failure_cleans_new_directory_and_keeps_source(
    working_point: PointView,
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = entry_roots[0] / "entry"
    source = root / "points/a"
    before = {
        name: (source / name).read_bytes() for name in ("point.yaml", "module_cfg.yaml")
    }
    copyfile = shutil.copyfile

    def fail_module_copy(
        source: str | Path, target: str | Path, *, follow_symlinks: bool = True
    ) -> str | Path:
        if Path(target) == root / "points/b/module_cfg.yaml":
            raise OSError("module copy failed")
        return copyfile(source, target, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", fail_module_copy)
        with pytest.raises(OSError, match="module copy failed"):
            entry.new_point("b", clone_from=working_point)
    assert not (root / "points/b").exists()
    assert entry.list_points() == ["a"]
    assert {
        name: (source / name).read_bytes() for name in ("point.yaml", "module_cfg.yaml")
    } == before
    assert working_point.Q1.t1 == 12.0
    assert entry.new_point("b", clone_from="a").Q1.t1 == 12.0


@pytest.mark.parametrize("source_mode", ["label", "view"])
def test_cross_entry_clone_is_rejected_and_removes_new_destination(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    source_mode: Literal["label", "view"],
) -> None:
    results, database = entry_roots
    other = ResultEntry.create("other", result_root=results, database_root=database)
    foreign_point = other.new_point("source")
    foreign_source = results / "other/points/source/point.yaml"
    before = foreign_source.read_bytes()
    source_arg = "other/source" if source_mode == "label" else foreign_point
    message = (
        "single path component" if source_mode == "label" else "Cross-entry cloning"
    )
    with pytest.raises(ValueError, match=message):
        entry.new_point("rejected", clone_from=source_arg)
    assert entry.list_points() == []
    assert not (results / "entry/points/rejected").exists()
    assert foreign_source.read_bytes() == before


def test_independent_entry_edits_merge_different_leaves_across_layers(
    working_point: PointView,
    entry_roots: tuple[Path, Path],
) -> None:
    results, database = entry_roots
    other = ResultEntry.open("entry", result_root=results, database_root=database)
    with working_point.edit() as draft:
        draft.Q1.freq = 5100.0
        other.use_point("a").Q1.t1 = 14.0
    assert working_point.Q1.freq == 5100.0
    assert working_point.Q1.t1 == 14.0
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    point = reopened.use_point("a")
    assert point.Q1.freq == 5100.0
    assert point.Q1.t1 == 14.0


@pytest.mark.parametrize("conflict", ["setup", "point"])
def test_multilayer_conflict_rejects_all_changes_and_keeps_snapshot(
    working_point: PointView,
    entry_roots: tuple[Path, Path],
    conflict: str,
) -> None:
    results, database = entry_roots
    other = ResultEntry.open("entry", result_root=results, database_root=database)
    other_point = other.use_point("a")
    setup_source = results / "entry/setup.yaml"
    point_source = results / "entry/points/a/point.yaml"
    with ExitStack() as transaction:
        draft = transaction.enter_context(working_point.edit())
        draft.Q1.freq = 5100.0
        draft.Q1.t1 = 13.0
        if conflict == "setup":
            other_point.Q1.freq = 5200.0
        else:
            other_point.Q1.t1 = 14.0
        before = setup_source.read_bytes(), point_source.read_bytes()
        with pytest.raises(ConflictError) as raised:
            transaction.close()
    assert raised.value.source == (
        setup_source if conflict == "setup" else point_source
    )
    assert (setup_source.read_bytes(), point_source.read_bytes()) == before
    assert working_point.Q1.freq == 5000.0
    assert working_point.Q1.t1 == 12.0

    working_point.refresh()
    assert working_point.Q1.freq == (5200.0 if conflict == "setup" else 5000.0)
    assert working_point.Q1.t1 == (12.0 if conflict == "setup" else 14.0)


@pytest.mark.parametrize("restore_failure", [False, True])
def test_second_layer_replace_failure_restores_or_reports_actual_partial_commit(
    working_point: PointView,
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    restore_failure: bool,
) -> None:
    results, database = entry_roots
    setup_source = results / "entry/setup.yaml"
    point_source = results / "entry/points/a/point.yaml"
    before = setup_source.read_bytes(), point_source.read_bytes()
    replace = Path.replace
    replaced_source: Path | None = None
    write_failed = False
    write_error = OSError("second replace failed")
    recovery_error = OSError("restore failed")

    def fail_second_replace(source: Path, target: str | Path) -> Path:
        nonlocal replaced_source, write_failed
        destination = Path(target)
        if destination not in (setup_source, point_source):
            return replace(source, target)
        if replaced_source is None:
            result = replace(source, target)
            replaced_source = destination
            return result
        if not write_failed:
            write_failed = True
            raise write_error
        if restore_failure:
            raise recovery_error
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_replace)
    with ExitStack() as transaction:
        draft = transaction.enter_context(working_point.edit())
        draft.Q1.freq = 5100.0
        draft.Q1.t1 = 13.0
        if restore_failure:
            with pytest.raises(PartialCommitError) as raised:
                transaction.close()
            assert raised.value.completed == (replaced_source,)
            assert raised.value.recovery_failed == (replaced_source,)
            assert raised.value.pending == (
                point_source if replaced_source == setup_source else setup_source,
            )
            assert raised.value.cause is write_error
            assert raised.value.recovery_cause is recovery_error
        else:
            with pytest.raises(OSError, match="second replace failed"):
                transaction.close()

    assert working_point.Q1.freq == 5000.0
    assert working_point.Q1.t1 == 12.0
    assert entry.setup.Q1.freq == 5000.0
    if restore_failure:
        yaml = YAML(typ="safe")
        assert yaml.load(setup_source.read_text())["components"]["Q1"]["freq"] == (
            5.1e9 if replaced_source == setup_source else 5e9
        )
        assert yaml.load(point_source.read_text())["components"]["Q1"]["t1"] == (
            13e-6 if replaced_source == point_source else 12e-6
        )
    else:
        assert (setup_source.read_bytes(), point_source.read_bytes()) == before
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    observed = reopened.use_point("a")
    assert observed.Q1.freq == (
        5100.0 if restore_failure and replaced_source == setup_source else 5000.0
    )
    assert observed.Q1.t1 == (
        13.0 if restore_failure and replaced_source == point_source else 12.0
    )


def test_failed_complete_point_validation_preserves_clone_origin_and_snapshots(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    range_kind: str,
) -> None:
    entry.setup.add_component("R1", kind=range_kind, low=1.0, high=2.0)
    original = entry.new_point("a")
    original.move("R1.high", to="point")
    root = entry_roots[0] / "entry"
    source = root / "points/a/point.yaml"
    yaml = YAML(typ="rt")
    stored = yaml.load(source.read_text())
    provenance = {
        "source": "manual",
        "kind": None,
        "run_id": None,
        "at": "2026-10-04T00:00:00Z",
        "stderr": None,
    }
    stored["provenance"]["R1.high"] = provenance
    write_yaml(source, stored)
    point = entry.new_point("b", clone_from="a")
    setup_source = root / "setup.yaml"
    point_source = root / "points/b/point.yaml"
    before = setup_source.read_bytes(), point_source.read_bytes()

    with (
        pytest.raises(ValidationError, match="low must not exceed high"),
        point.edit() as draft,
    ):
        draft.R1.high = 0.5
    assert (setup_source.read_bytes(), point_source.read_bytes()) == before
    assert point.R1.low == 1.0
    assert point.R1.high == 2.0
    assert entry.setup.R1.low == 1.0
    assert yaml.load(point_source.read_text())["provenance"]["R1.high"] == {
        **provenance,
        "cloned_from": {"entry_id": entry.entry_id, "point": "a"},
    }


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
