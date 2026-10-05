"""Complete seeded points through ResultEntry and its public document views."""

import shutil
from collections.abc import Generator
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.document_store import ConflictError
from zcu_tools.resources.entry import (
    ComponentSchema,
    PointView,
    Provenance,
    ResultEntry,
    UnknownKindError,
    component_registry,
)
from zcu_tools.resources.entry.views import FieldView


def field_view(value: object) -> FieldView:
    """Require an editable container returned by a public view."""
    assert isinstance(value, FieldView)
    return value


class RequiredRangeSchema(ComponentSchema):
    low: float
    high: float


class DefaultTiming(BaseModel):
    width: float
    note: str


class NotebookSchema(ComponentSchema):
    model_config = ConfigDict(extra="forbid")
    gain: Annotated[float, Field(gt=0)]
    title: str
    token: str = Field(default_factory=lambda: str(uuid4()))
    timing: DefaultTiming = Field(
        default_factory=lambda: DefaultTiming(width=2.0, note="factory")
    )

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return value.strip().lower()


@pytest.fixture
def range_kind(registry_state_guard: None) -> Generator[str]:
    kind = "test/point-range"
    component_registry.register(kind, RequiredRangeSchema)
    try:
        yield kind
    finally:
        component_registry.unregister(kind)


@pytest.fixture
def notebook_kind(registry_state_guard: None) -> Generator[str]:
    kind = "test/point-notebook"
    component_registry.register(kind, NotebookSchema)
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
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    point = entry.new_point("a")
    point.Q1.duration = 12.0
    return point


def write_yaml(source: Path, document: object) -> None:
    with source.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(document, stream)


def test_new_point_copies_complete_components_and_keeps_independent_values(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    entry.setup.add_component("R1", kind="fake/sensor", rate=6000.0)
    entry.setup.add_component(
        "Q1",
        kind="fake/drive/a",
        rate=5000.0,
        sense="R1",
        wiring={"bias_ch": 3},
        ext={"nested": {"note": "seed"}},
    )
    point = entry.new_point("stable (cold)")
    assert point.Q1.rate == 5000.0
    assert point.Q1.sense == "R1"
    assert point.description is None
    assert entry.list_points() == ["stable (cold)"]
    root = entry_roots[0] / "entry"
    source = root / "points/stable (cold)/point.yaml"
    document = YAML(typ="safe").load(source.read_text())
    assert document["components"]["Q1"]["kind"] == "fake/drive/a"
    assert document["components"]["Q1"]["rate"] == 5000.0
    assert document["components"]["R1"]["kind"] == "fake/sensor"
    assert document["general"]["created_at"].endswith("Z")

    entry.setup.Q1.rate = 5100.0
    field_view(entry.setup.Q1.wiring).bias_ch = 4
    field_view(entry.setup.Q1.ext).nested = {"note": "new template"}
    assert point.Q1.rate == 5000.0
    assert field_view(point.Q1.wiring).bias_ch == 3
    assert field_view(field_view(point.Q1.ext).nested).note == "seed"
    point.Q1.rate = 5200.0
    field_view(point.Q1.ext).nested = {"note": "point"}
    assert entry.setup.Q1.rate == 5100.0
    assert field_view(field_view(entry.setup.Q1.ext).nested).note == "new template"
    assert entry.new_point("later").Q1.rate == 5100.0
    assert entry.use_point("stable (cold)").Q1.rate == 5200.0
    source.unlink()
    assert point.Q1.rate == 5200.0


def test_seed_accepts_omitted_components_without_rewriting_setup(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    results, database = entry_roots
    source = results / "entry/setup.yaml"
    template = YAML(typ="rt").load(source.read_text())
    template.pop("components")
    write_yaml(source, template)
    before = source.read_bytes()

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    reopened.setup.refresh()
    point = reopened.new_point("empty")

    assert point.description is None
    assert reopened.use_point("empty").description is None
    assert reopened.list_points() == ["empty"]
    point_root = results / "entry/points/empty"
    stored = YAML(typ="safe").load((point_root / "point.yaml").read_text())
    assert stored["components"] == {}
    module = YAML(typ="safe").load((point_root / "module_cfg.yaml").read_text())
    assert module["format"] == "zcu.module-library"
    assert module["format_version"] == "1.0"
    assert source.read_bytes() == before


def test_list_points_sorts_complete_points_and_ignores_incomplete_directories(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    entry.new_point("z (cold)")
    entry.new_point("a (warm)")
    partial = entry_roots[0] / "entry/points/partial"
    partial.mkdir()
    (partial / "point.yaml").touch()
    assert entry.list_points() == ["a (warm)", "z (cold)"]
    with pytest.raises(FileNotFoundError):
        entry.use_point("partial")
    assert entry.list_points() == ["a (warm)", "z (cold)"]


@pytest.mark.parametrize("label", ["", ".", "..", "../outside", "nested/point", "x\\y"])
@pytest.mark.parametrize("operation", ["new_point", "use_point"])
def test_point_labels_reject_unsafe_path_segments(
    entry: ResultEntry, label: str, operation: Literal["new_point", "use_point"]
) -> None:
    create_or_open = entry.new_point if operation == "new_point" else entry.use_point
    with pytest.raises(ValueError, match="single path component"):
        create_or_open(label)
    assert entry.list_points() == []


@pytest.mark.parametrize("target", ["setup", "point"])
def test_add_component_requires_complete_values_and_runs_original_model(
    entry: ResultEntry, entry_roots: tuple[Path, Path], notebook_kind: str, target: str
) -> None:
    view = entry.setup if target == "setup" else entry.new_point("a")
    source = (
        entry_roots[0]
        / "entry"
        / ("setup.yaml" if target == "setup" else "points/a/point.yaml")
    )
    before = source.read_bytes()
    with pytest.raises(ValidationError, match="gain"):
        view.add_component("C1", kind=notebook_kind, title=" Title ")
    with pytest.raises(ValidationError, match="greater than"):
        view.add_component("C1", kind=notebook_kind, gain=-1.0, title=" Title ")
    assert source.read_bytes() == before
    view.add_component("C1", kind=notebook_kind, gain=1.0, title=" Title ")
    assert view.C1.gain == 1.0
    assert view.C1.title == "title"
    timing = field_view(view.C1.timing)
    assert timing.width == 2.0
    assert timing.note == "factory"
    with pytest.raises(ValueError, match="already exists"):
        view.add_component("C1", kind=notebook_kind, gain=2.0, title="other")
    with view.edit() as draft:
        draft.set("C1.timing.width", 3.0)
    assert timing.width == 3.0
    assert timing.note == "factory"
    stored = YAML(typ="safe").load(source.read_text())
    assert stored["components"]["C1"]["timing"]["width"] == 3.0


@pytest.mark.parametrize("stored_defaults", [True, False])
def test_seed_keeps_generated_default_values_across_independent_edits_and_reload(
    entry: ResultEntry,
    notebook_kind: str,
    entry_roots: tuple[Path, Path],
    stored_defaults: bool,
) -> None:
    entry.setup.add_component("C1", kind=notebook_kind, gain=1.0, title="template")
    if not stored_defaults:
        source = entry_roots[0] / "entry/setup.yaml"
        template = YAML(typ="rt").load(source.read_text())
        template["components"]["C1"].pop("token")
        template["components"]["C1"].pop("timing")
        write_yaml(source, template)
    point = entry.new_point("a")
    token = point.C1.token
    assert token == entry.setup.C1.token
    entry.setup.C1.title = "new template"
    point.C1.gain = 2.0
    point.refresh()
    assert point.C1.token == token
    assert point.C1.title == "template"
    assert entry.setup.C1.gain == 1.0
    assert entry.use_point("a").C1.token == token
    cloned = entry.new_point("b", clone_from=point)
    assert cloned.C1.token == token
    assert field_view(cloned.C1.timing).width == field_view(point.C1.timing).width
    assert field_view(cloned.C1.timing).note == field_view(point.C1.timing).note
    cloned.C1.gain = 3.0
    cloned.refresh()
    assert cloned.C1.token == token
    assert point.C1.gain == 2.0


def test_point_add_component_only_changes_its_own_document(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    point = entry.new_point("a")
    entry.setup.add_component("R1", kind="fake/sensor")
    source = entry_roots[0] / "entry/points/a/point.yaml"
    setup_source = entry_roots[0] / "entry/setup.yaml"
    before = setup_source.read_bytes()
    point.add_component("Q1", kind="fake/drive/a", sense="R1", rate=5000.0)
    assert YAML(typ="safe").load(source)["components"]["Q1"]["sense"] == "R1"
    assert setup_source.read_bytes() == before
    assert point.Q1.sense == "R1"
    assert entry.use_point("a").Q1.rate == 5000.0
    with pytest.raises(AttributeError):
        _ = entry.setup.Q1


def test_point_edit_only_changes_its_document(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    point = entry.new_point("a")
    setup_source = entry_roots[0] / "entry/setup.yaml"
    before = setup_source.read_bytes()
    with point.edit() as draft:
        draft.Q1.rate = 5100.0
        draft.Q1.duration = 12.0
        draft.description = "point note"
    assert point.Q1.rate == 5100.0
    assert point.Q1.duration == 12.0
    assert entry.setup.Q1.rate == 5000.0
    assert point.description == "point note"
    assert entry.setup.description is None
    assert setup_source.read_bytes() == before
    results, database = entry_roots
    reloaded = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reloaded.use_point("a").Q1.duration == 12.0
    stored = YAML(typ="safe").load((results / "entry/points/a/point.yaml").read_text())
    assert stored["components"]["Q1"]["rate"] == 5100.0
    assert stored["components"]["Q1"]["duration"] == 12.0


@pytest.mark.parametrize("operation", ["refresh", "use_point"])
def test_point_reload_uses_only_its_file_and_keeps_snapshot_on_failure(
    entry: ResultEntry, entry_roots: tuple[Path, Path], operation: str
) -> None:
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    point = entry.new_point("a")
    results, database = entry_roots
    other = ResultEntry.open("entry", result_root=results, database_root=database)
    other.use_point("a").Q1.rate = 5100.0
    assert point.Q1.rate == 5000.0
    root = results / "entry"
    (root / "setup.yaml").write_text("invalid: [", encoding="utf-8")
    reload_point = (
        point.refresh if operation == "refresh" else lambda: entry.use_point("a")
    )
    loaded = reload_point()
    if loaded is not None:
        point = loaded
    assert point.Q1.rate == 5100.0
    assert entry.setup.Q1.rate == 5000.0
    point.Q1.duration = 12.0
    assert point.Q1.duration == 12.0
    source = root / "points/a/point.yaml"
    stored = YAML(typ="rt").load(source.read_text())
    stored["components"]["Q1"]["rate"] = "invalid"
    write_yaml(source, stored)
    with pytest.raises(ValueError, match="rate"):
        reload_point()
    assert point.Q1.rate == 5100.0
    assert point.Q1.duration == 12.0


def test_point_edit_rejects_invalid_reload_before_exposing_draft(
    entry: ResultEntry, entry_roots: tuple[Path, Path], range_kind: str
) -> None:
    entry.setup.add_component("R1", kind=range_kind, low=1.0, high=2.0)
    point = entry.new_point("a")
    point_source = entry_roots[0] / "entry/points/a/point.yaml"
    stored = YAML(typ="rt").load(point_source.read_text())
    stored["components"]["R1"].pop("low")
    write_yaml(point_source, stored)
    before = point_source.read_bytes()
    entered: list[bool] = []
    with pytest.raises(ValidationError, match="R1.low"), point.edit():
        entered.append(True)
    assert entered == []
    assert point.R1.low == 1.0
    assert point.R1.high == 2.0
    assert point_source.read_bytes() == before


@pytest.mark.parametrize("operation", ["use", "refresh", "edit"])
def test_point_unknown_kind_reports_own_source_without_publishing(
    entry: ResultEntry,
    working_point: PointView,
    entry_roots: tuple[Path, Path],
    operation: str,
) -> None:
    source = entry_roots[0] / "entry/points/a/point.yaml"
    setup_source = entry_roots[0] / "entry/setup.yaml"
    stored = YAML(typ="rt").load(source.read_text())
    stored["components"]["Q1"]["kind"] = "missing/point-kind"
    write_yaml(source, stored)
    before = source.read_bytes()
    setup_before = setup_source.read_bytes()
    entered: list[bool] = []

    if operation == "edit":
        with pytest.raises(UnknownKindError) as error, working_point.edit():
            entered.append(True)
    else:
        reload_point = (
            working_point.refresh
            if operation == "refresh"
            else lambda: entry.use_point("a")
        )
        with pytest.raises(UnknownKindError) as error:
            reload_point()

    assert error.value.source == source
    assert error.value.component == "Q1"
    assert error.value.kind == "missing/point-kind"
    assert str(source) in str(error.value)
    assert str(setup_source) not in str(error.value)
    assert entered == []
    assert working_point.Q1.rate == 5000.0
    assert working_point.Q1.duration == 12.0
    assert source.read_bytes() == before
    assert setup_source.read_bytes() == setup_before


@pytest.mark.parametrize("operation", ["use", "refresh", "edit"])
def test_point_invalid_component_name_reports_own_source_without_publishing(
    entry: ResultEntry,
    working_point: PointView,
    entry_roots: tuple[Path, Path],
    operation: str,
) -> None:
    source = entry_roots[0] / "entry/points/a/point.yaml"
    setup_source = entry_roots[0] / "entry/setup.yaml"
    stored = YAML(typ="rt").load(source.read_text())
    stored["components"]["edit"] = stored["components"].pop("Q1")
    write_yaml(source, stored)
    before = source.read_bytes()
    setup_before = setup_source.read_bytes()
    entered: list[bool] = []

    if operation == "edit":
        with pytest.raises(ValueError, match="edit") as error, working_point.edit():
            entered.append(True)
    else:
        reload_point = (
            working_point.refresh
            if operation == "refresh"
            else lambda: entry.use_point("a")
        )
        with pytest.raises(ValueError, match="edit") as error:
            reload_point()

    assert str(source) in str(error.value)
    assert str(setup_source) not in str(error.value)
    assert entered == []
    assert working_point.Q1.rate == 5000.0
    assert working_point.Q1.duration == 12.0
    assert source.read_bytes() == before
    assert setup_source.read_bytes() == setup_before


def test_setup_commit_does_not_validate_or_modify_existing_points(
    working_point: PointView, entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    point_source = entry_roots[0] / "entry/points/a/point.yaml"
    point_source.write_text("invalid: [", encoding="utf-8")
    before = point_source.read_bytes()
    with entry.setup.edit() as draft:
        draft.Q1.rate = 5300.0
    assert point_source.read_bytes() == before
    assert working_point.Q1.rate == 5000.0
    assert entry.new_point("b").Q1.rate == 5300.0


def test_seed_copies_provenance_without_clone_origin_and_keeps_it_independent(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    root = entry_roots[0] / "entry"
    source = root / "setup.yaml"
    yaml = YAML(typ="rt")
    stored = yaml.load(source.read_text())
    provenance = {
        "source": "manual",
        "kind": None,
        "run_id": None,
        "at": "2026-10-04T00:00:00Z",
        "stderr": 1e6,
    }
    stored["provenance"]["Q1.rate"] = provenance
    write_yaml(source, stored)
    point = entry.new_point("a")
    destination = root / "points/a/point.yaml"
    seeded = yaml.load(destination.read_text())
    assert seeded["provenance"]["Q1.rate"] == provenance
    point.Q1.rate = 5100.0
    assert yaml.load(source.read_text())["provenance"]["Q1.rate"] == provenance
    entry.setup.Q1.rate = 5200.0
    assert point.Q1.rate == 5100.0
    assert point.Q1.kind == "fake/drive/a"


@pytest.mark.parametrize("source_mode", ["label", "view"])
def test_clone_copies_only_complete_point_and_module_without_reading_setup(
    entry: ResultEntry, entry_roots: tuple[Path, Path], source_mode: str
) -> None:
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    point = entry.new_point("source")
    with point.edit() as draft:
        draft.Q1.duration = 12.0
        draft.description = "source environment"
    root = entry_roots[0] / "entry"
    source = root / "points/source"
    (source / "module_cfg.yaml").write_text(
        "format: zcu.module-library\nformat_version: '1.0'\nQ1: {pulse: arbitrary}\n"
    )
    (source / "data.h5").write_bytes(b"not cloned")
    (source / "image.png").write_bytes(b"not cloned")
    (root / "setup.yaml").write_text("invalid: [", encoding="utf-8")
    source_arg = "source" if source_mode == "label" else point
    cloned = entry.new_point("target", clone_from=source_arg)
    assert cloned.Q1.rate == 5000.0
    assert cloned.Q1.duration == 12.0
    assert cloned.description == "source environment"
    destination = root / "points/target"
    assert {path.name for path in destination.iterdir() if path.suffix != ".lock"} == {
        "point.yaml",
        "module_cfg.yaml",
    }
    assert (destination / "module_cfg.yaml").read_bytes() == (
        source / "module_cfg.yaml"
    ).read_bytes()
    stored = YAML(typ="safe").load((destination / "point.yaml").read_text())
    assert stored["components"]["Q1"]["kind"] == "fake/drive/a"
    assert stored["components"]["Q1"]["duration"] == 12.0
    before = (destination / "point.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        entry.new_point("target", clone_from=source_arg)
    assert (destination / "point.yaml").read_bytes() == before
    assert entry.list_points() == ["source", "target"]


def test_yaml_extensions_survive_seed_clone_reload_and_keep_source_units(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    payload: YamlMap = {
        "empty-value": None,
        "slash/key": [True, "opaque", {"rate": "not a physical field"}, 1.25],
    }
    entry.setup.add_component(
        "R1", kind="fake/sensor", ext={"blob": payload, "noise": 11.0}
    )
    point = entry.new_point("a")
    point.general.ext["_misc-key"] = payload
    payload["new"] = "caller mutation"
    blob = field_view(field_view(point.R1.ext).blob)
    assert blob["empty-value"] is None
    expected = [True, "opaque", {"rate": "not a physical field"}, 1.25]
    assert blob["slash/key"] == expected
    general_blob = field_view(point.general.ext["_misc-key"])
    assert general_blob["empty-value"] is None
    assert general_blob["slash/key"] == expected
    with pytest.raises(KeyError):
        _ = blob["new"]
    source = Provenance("manual", None, None, "2026-10-04T00:00:00Z", 2.0)
    with point.edit() as draft:
        draft.set("R1.ext.noise", 12.0, provenance=source)
    clone = entry.new_point("b", clone_from="a")
    field_view(clone.R1.ext).blob = {"independent": False}
    clone.general.ext["_misc-key"] = None
    reopened = ResultEntry.open(
        "entry", result_root=entry_roots[0], database_root=entry_roots[1]
    )
    original = reopened.use_point("a")
    copied = reopened.use_point("b")
    assert field_view(field_view(original.R1.ext).blob)["slash/key"] == expected
    assert field_view(original.general.ext["_misc-key"])["slash/key"] == expected
    assert field_view(field_view(copied.R1.ext).blob).independent is False
    assert copied.general.ext["_misc-key"] is None
    metadata = copied.meta("R1.ext.noise")
    assert metadata is not None
    assert metadata.stderr == 2.0
    assert metadata.cloned_from is not None
    assert metadata.cloned_from["point"] == "a"
    field_view(copied.R1.ext).noise = 12.0
    accepted = copied.meta("R1.ext.noise")
    assert accepted is not None
    assert accepted.cloned_from is None
    assert field_view(entry.setup.R1.ext).noise == 11.0


def test_clone_marks_each_copied_source_with_its_direct_point(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    entry.new_point("a").Q1.duration = 12.0
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
    stored["provenance"]["Q1.duration"] = original
    write_yaml(source, stored)
    entry.new_point("b", clone_from="a")
    cloned = yaml.load((root / "points/b/point.yaml").read_text())
    assert cloned["provenance"]["Q1.duration"] == {
        **original,
        "cloned_from": {"entry_id": entry.entry_id, "point": "a"},
    }
    entry.new_point("c", clone_from="b")
    cloned_again = yaml.load((root / "points/c/point.yaml").read_text())
    assert cloned_again["provenance"]["Q1.duration"] == {
        **original,
        "cloned_from": {"entry_id": entry.entry_id, "point": "b"},
    }


@pytest.mark.parametrize(
    ("write", "path"),
    [
        ("attribute", "Q1.duration"),
        ("edit", "Q1.duration"),
        ("set", "Q1.duration"),
        ("set", "Q1.wiring.bias_ch"),
        ("set", "Q1.ext.nested.note"),
    ],
)
def test_rewriting_same_cloned_value_clears_only_accepted_field_origin(
    entry: ResultEntry, entry_roots: tuple[Path, Path], write: str, path: str
) -> None:
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    original = entry.new_point("a")
    original.Q1.duration = 12.0
    original.Q1.coherence = 13.0
    field_view(original.Q1.wiring).bias_ch = 3
    field_view(original.Q1.ext).nested = {"note": "accepted"}
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
    paths = ["Q1.duration", "Q1.coherence", "Q1.wiring.bias_ch", "Q1.ext.nested.note"]
    stored["provenance"] = {key: deepcopy(provenance) for key in paths}
    write_yaml(source, stored)
    point = entry.new_point("b", clone_from="a")
    setup_before = (root / "setup.yaml").read_bytes()
    if write == "attribute":
        with pytest.raises(ValidationError):
            point.Q1.coherence = "not-a-number"
        point.Q1.duration = 12.0
    else:
        with point.edit() as draft:
            with pytest.raises(ValidationError):
                draft.set("Q1.coherence", "not-a-number")
            if write == "edit":
                draft.Q1.duration = 12.0
            else:
                value = (
                    "accepted"
                    if path == "Q1.ext.nested.note"
                    else 3
                    if path == "Q1.wiring.bias_ch"
                    else 12.0
                )
                draft.set(path, value)
    rewritten = yaml.load((root / "points/b/point.yaml").read_text())
    accepted = rewritten["provenance"][path]
    assert accepted == {**provenance, "at": accepted["at"]}
    assert accepted["at"] != provenance["at"]
    for untouched in set(paths) - {path}:
        assert rewritten["provenance"][untouched] == {
            **provenance,
            "cloned_from": {"entry_id": entry.entry_id, "point": "a"},
        }
    assert (root / "setup.yaml").read_bytes() == setup_before
    for view in (point, entry.use_point("b")):
        assert view.Q1.duration == 12.0
        assert view.Q1.coherence == 13.0
        assert field_view(view.Q1.wiring).bias_ch == 3
        assert field_view(field_view(view.Q1.ext).nested).note == "accepted"


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
    assert {name: (source / name).read_bytes() for name in before} == before
    assert working_point.Q1.duration == 12.0
    assert entry.new_point("b", clone_from="a").Q1.duration == 12.0


@pytest.mark.parametrize("source_mode", ["label", "view"])
def test_cross_entry_clone_is_rejected_and_removes_new_destination(
    entry: ResultEntry, entry_roots: tuple[Path, Path], source_mode: str
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


def test_independent_handles_merge_different_leaves_in_one_point(
    working_point: PointView, entry_roots: tuple[Path, Path]
) -> None:
    results, database = entry_roots
    other = ResultEntry.open("entry", result_root=results, database_root=database)
    with working_point.edit() as draft:
        draft.Q1.rate = 5100.0
        other.use_point("a").Q1.duration = 14.0
    assert working_point.Q1.rate == 5100.0
    assert working_point.Q1.duration == 14.0
    reopened = ResultEntry.open(
        "entry", result_root=results, database_root=database
    ).use_point("a")
    assert reopened.Q1.rate == 5100.0
    assert reopened.Q1.duration == 14.0


@pytest.mark.parametrize("conflict", ["rate", "duration"])
def test_same_leaf_conflict_rejects_whole_point_transaction(
    working_point: PointView, entry_roots: tuple[Path, Path], conflict: str
) -> None:
    results, database = entry_roots
    other = ResultEntry.open(
        "entry", result_root=results, database_root=database
    ).use_point("a")
    source = results / "entry/points/a/point.yaml"
    with ExitStack() as transaction:
        draft = transaction.enter_context(working_point.edit())
        draft.Q1.rate = 5100.0
        draft.Q1.duration = 13.0
        if conflict == "rate":
            other.Q1.rate = 5200.0
        else:
            other.Q1.duration = 14.0
        before = source.read_bytes()
        with pytest.raises(ConflictError) as error:
            transaction.close()
    assert error.value.source == source
    assert source.read_bytes() == before
    assert working_point.Q1.rate == 5000.0
    assert working_point.Q1.duration == 12.0
    working_point.refresh()
    assert working_point.Q1.rate == (5200.0 if conflict == "rate" else 5000.0)
    assert working_point.Q1.duration == (12.0 if conflict == "rate" else 14.0)
