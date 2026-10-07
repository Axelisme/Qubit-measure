"""Recursive component value/path editing, sources, and isolated snapshots."""

from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_serializer
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    AcceptedPayload,
    AcceptedWrite,
    ComponentSchema,
    LedgerEvent,
    Provenance,
    ResultEntry,
    accept_source,
    document_fields,
    document_meta,
    document_value,
    manual_update,
    snapshot,
    source_path,
)
from zcu_tools.resources.entry.schema import PointDocument, SetupDocument
from zcu_tools.resources.entry.views import FieldView

from .view_support import create_entry, field_view, registered_model


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("version", ["1.0", "1.2"])
def test_allow_extras_support_child_views_set_meta_seed_and_reload(
    tmp_path: Path,
    registry_state_guard: None,
    owner: str,
    version: str,
) -> None:
    class Extensible(ComponentSchema):
        model_config = ConfigDict(extra="allow")
        title: str

    with registered_model("notebook/extensible", Extensible) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, title="ready", gain=2)
        view = entry.setup if owner == "setup" else entry.new_point("working")
        source = (
            results / "entry/setup.yaml"
            if owner == "setup"
            else results / "entry/points/working/point.yaml"
        )
        yaml = YAML(typ="rt")
        stored = yaml.load(source)
        stored["format_version"] = version
        stored["components"]["N1"]["future"] = {"level": 3, "untouched": 4}
        yaml.dump(stored, source)
        before = source.read_bytes()
        view.refresh()
        assert source.read_bytes() == before
        assert view.N1.gain == 2
        assert view.meta("N1.future.level") is None
        future = field_view(view.N1.future)
        assert future["level"] == 3
        view.N1.gain = 5
        assert view.N1.gain == 5
        future["level"] = 6
        with view.edit() as draft:
            draft.set("N1.future.level", 7)
            draft.set("N1.gain", 8)
        accepted = view.meta("N1.future.level")
        assert accepted is not None and accepted.source == "manual"
        gain_source = view.meta("N1.gain")
        assert gain_source is not None and gain_source.source == "manual"
        assert future.level == 7 and future.untouched == 4
        if owner == "setup":
            seeded = entry.new_point("seeded")
            assert seeded.N1.gain == 8
            assert seeded.meta("N1.future.level") == accepted
            field_view(seeded.N1.future).level = 9
            assert future.level == 7
        else:
            assert entry.setup.N1.gain == 2
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        loaded = reopened.setup if owner == "setup" else reopened.use_point("working")
        assert loaded.N1.gain == 8
        assert field_view(loaded.N1.future).level == 7
        assert loaded.meta("N1.future.level") == accepted
        assert YAML(typ="safe").load(source)["format_version"] == version


def test_nested_model_views_follow_commit_and_refresh_without_leaking_drafts(
    tmp_path: Path,
) -> None:
    class Pin(BaseModel):
        model_config = ConfigDict(extra="forbid")
        index: int = Field(strict=True, ge=0)

    class Pins(BaseModel):
        model_config = ConfigDict(extra="forbid")
        link: Pin

    class Wired(ComponentSchema):
        wiring: Pins

    with registered_model("notebook/wired", Wired) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, wiring={"link": {"index": 2}})
        wiring = entry.setup.N1.wiring
        assert isinstance(wiring, FieldView)
        pin = field_view(wiring.link)
        assert pin.index == 2
        with entry.setup.edit() as draft:
            draft_wiring = field_view(draft.N1.wiring)
            draft_pin = field_view(draft_wiring.link)
            draft_pin.index = 5
            assert draft_pin.index == 5
            assert pin.index == 2
        assert pin.index == 5
        source = entry.setup.meta("N1.wiring.link.index")
        assert source is not None and source.source == "manual"
        other = ResultEntry.open("entry", result_root=results, database_root=database)
        with other.setup.edit() as draft:
            draft.set("N1.wiring.link.index", 8)
        assert pin.index == 5
        entry.setup.refresh()
        assert pin.index == 8


def test_plain_typed_dict_views_edit_seed_and_reload_without_resolving_paths(
    tmp_path: Path,
) -> None:
    class Programmed(ComponentSchema):
        programs: dict[str, str] = Field(default_factory=dict)

    with registered_model("notebook/programmed", Programmed) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1", kind=kind, programs={"drive": "unresolved.drive"}
        )
        slots = entry.setup.N1.programs
        assert isinstance(slots, FieldView)
        assert slots.drive == "unresolved.drive"
        slots.drive = "another.path"
        point = entry.new_point("working")
        with point.edit() as draft:
            draft.set("N1.programs.sense", "unresolved.sense")
        point_slots = point.N1.programs
        assert isinstance(point_slots, FieldView)
        assert point_slots.drive == "another.path"
        assert point_slots.sense == "unresolved.sense"
        source = point.meta("N1.programs.drive")
        before = point_slots.drive
        with pytest.raises(ValidationError):
            point_slots.drive = 3
        assert point_slots.drive == before
        assert point.meta("N1.programs.drive") == source
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        reopened_slots = reopened.use_point("working").N1.programs
        assert isinstance(reopened_slots, FieldView)
        assert reopened_slots.sense == "unresolved.sense"
        assert slots.drive == "another.path"


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("invalid", [-1, None, True, "3", 3.5])
def test_recursive_dict_views_preserve_models_constraints_and_sources(
    tmp_path: Path,
    owner: str,
    invalid: object,
) -> None:
    class Pin(BaseModel):
        model_config = ConfigDict(extra="forbid")
        index: int = Field(strict=True, ge=0)

    class Route(BaseModel):
        model_config = ConfigDict(extra="forbid")
        pin: Pin

    class Routed(ComponentSchema):
        routes: dict[str, Route]
        levels: dict[str, dict[str, int]]

    with registered_model("notebook/routed", Routed) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1",
            kind=kind,
            routes={"drive": {"pin": {"index": 2}}},
            levels={"gain": {"left": 1, "right": 2}},
        )
        view = entry.setup if owner == "setup" else entry.new_point("working")
        routes = field_view(view.N1.routes)
        route = field_view(routes["drive"])
        pin = field_view(route.pin)
        pin["index"] = 3
        source = view.meta("N1.routes.drive.pin.index")
        assert source is not None and source.source == "manual"
        with view.edit() as draft:
            draft_routes = field_view(draft.N1.routes)
            draft_route = field_view(draft_routes.drive)
            draft_pin = field_view(draft_route.pin)
            with pytest.raises(ValidationError):
                draft_pin.index = invalid
            assert draft_pin.index == 3
            levels = field_view(draft.N1.levels)
            gain = field_view(levels["gain"])
            gain["left"] = 5
            assert gain.right == 2
            assert pin.index == 3
            assert view.meta("N1.routes.drive.pin.index") == source
        assert pin.index == 3
        assert view.meta("N1.routes.drive.pin.index") == source
        gain_source = view.meta("N1.levels.gain.left")
        assert gain_source is not None and gain_source.source == "manual"
        if owner == "point":
            setup_routes = field_view(entry.setup.N1.routes)
            setup_route = field_view(setup_routes.drive)
            setup_pin = field_view(setup_route.pin)
            assert setup_pin.index == 2
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        loaded = reopened.setup if owner == "setup" else reopened.use_point("working")
        with loaded.edit() as draft:
            draft.set("N1.routes.drive.pin.index", 6)
        loaded_routes = field_view(loaded.N1.routes)
        loaded_route = field_view(loaded_routes.drive)
        loaded_pin = field_view(loaded_route.pin)
        assert loaded_pin.index == 6
        loaded_levels = field_view(loaded.N1.levels)
        loaded_gain = field_view(loaded_levels.gain)
        assert loaded_gain.left == 5


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("version", ["1.0", "1.2"])
def test_collision_extra_edit_abort_preserves_both_components(
    tmp_path: Path, registry_state_guard: None, owner: str, version: str
) -> None:
    class Colliding(ComponentSchema):
        model_config = ConfigDict(extra="allow")

    original_config = Colliding.model_config.copy()
    try:
        with registered_model("notebook/collision", Colliding) as kind:
            entry, results, database = create_entry(tmp_path)
            entry.setup.add_component("N1", kind=kind, model_config={"marker": "first"})
            entry.setup.add_component(
                "N2", kind=kind, model_config={"marker": "second"}
            )
            view = entry.setup if owner == "setup" else entry.new_point("working")
            source = (
                results / "entry/setup.yaml"
                if owner == "setup"
                else results / "entry/points/working/point.yaml"
            )
            yaml = YAML(typ="rt")
            stored = yaml.load(source)
            stored["format_version"] = version
            yaml.dump(stored, source)
            view.refresh()
            before = source.read_bytes()

            def abort_edit() -> None:
                with view.edit() as draft:
                    draft.set("N1.model_config.marker", "changed")
                    raise RuntimeError("abort requested")

            with pytest.raises(RuntimeError, match="abort requested"):
                abort_edit()
            assert field_view(view.N1.model_config).marker == "first"
            assert field_view(view.N2.model_config).marker == "second"
            assert source.read_bytes() == before
            reopened = ResultEntry.open(
                "entry", result_root=results, database_root=database
            )
            loaded = (
                reopened.setup if owner == "setup" else reopened.use_point("working")
            )
            assert field_view(loaded.N1.model_config).marker == "first"
            assert field_view(loaded.N2.model_config).marker == "second"
    finally:
        # Restore third-party class state even when the original bug mutates it.
        Colliding.model_config = original_config


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("write", ["set", "property"])
def test_model_mapping_replacement_accepts_values_and_leaf_sources(
    tmp_path: Path, registry_state_guard: None, owner: str, write: str
) -> None:
    class Pin(BaseModel):
        model_config = ConfigDict(extra="forbid")
        index: int = Field(strict=True, ge=0)

    class Route(BaseModel):
        model_config = ConfigDict(extra="forbid")
        pin: Pin

    class Routed(ComponentSchema):
        routes: dict[str, Route]

    with registered_model("notebook/container", Routed) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component(
            "N1", kind=kind, routes={"drive": {"pin": {"index": 2}}}
        )
        view = entry.setup if owner == "setup" else entry.new_point("working")
        if write == "set":
            with view.edit() as draft:
                draft.set("N1.routes", {"drive": {"pin": {"index": 6}}})
        else:
            view.N1.routes = {"drive": {"pin": {"index": 6}}}
        route = field_view(field_view(view.N1.routes).drive)
        assert field_view(route.pin).index == 6
        accepted = view.meta("N1.routes.drive.pin.index")
        assert accepted is not None and accepted.source == "manual"
        with view.edit() as draft:
            if write == "set":
                with pytest.raises(ValidationError):
                    draft.set("N1.routes", {"drive": {"pin": {"index": -1}}})
            else:
                with pytest.raises(ValidationError):
                    draft.N1.routes = {"drive": {"pin": {"index": -1}}}
            draft_route = field_view(field_view(draft.N1.routes).drive)
            assert field_view(draft_route.pin).index == 6
        assert view.meta("N1.routes.drive.pin.index") == accepted

        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        loaded = reopened.setup if owner == "setup" else reopened.use_point("working")
        loaded_route = field_view(field_view(loaded.N1.routes).drive)
        assert field_view(loaded_route.pin).index == 6
        assert loaded.meta("N1.routes.drive.pin.index") == accepted


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("write", ["set", "property"])
def test_model_list_replacement_accepts_values_and_source(
    tmp_path: Path, registry_state_guard: None, owner: str, write: str
) -> None:
    class Pin(BaseModel):
        index: int = Field(strict=True, ge=0)

    class Route(BaseModel):
        pin: Pin

    class Routed(ComponentSchema):
        routes: list[Route]

    with registered_model("notebook/list-container", Routed) as kind:
        entry, results, database = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, routes=[{"pin": {"index": 2}}])
        view = entry.setup if owner == "setup" else entry.new_point("working")
        if write == "set":
            with view.edit() as draft:
                draft.set("N1.routes", [{"pin": {"index": 6}}])
        else:
            view.N1.routes = [{"pin": {"index": 6}}]
        assert view.N1.routes == [{"pin": {"index": 6}}]
        accepted = view.meta("N1.routes")
        assert accepted is not None and accepted.source == "manual"
        with view.edit() as draft:
            if write == "set":
                with pytest.raises(ValidationError):
                    draft.set("N1.routes", [{"pin": {"index": -1}}])
            else:
                with pytest.raises(ValidationError):
                    draft.N1.routes = [{"pin": {"index": -1}}]
            assert draft.N1.routes == [{"pin": {"index": 6}}]
        assert view.meta("N1.routes") == accepted
        reopened = ResultEntry.open(
            "entry", result_root=results, database_root=database
        )
        loaded = reopened.setup if owner == "setup" else reopened.use_point("working")
        assert loaded.N1.routes == [{"pin": {"index": 6}}]
        assert loaded.meta("N1.routes") == accepted


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize("version", ["1.0", "1.2"])
def test_collision_extras_read_write_meta_and_reload(
    tmp_path: Path, registry_state_guard: None, owner: str, version: str
) -> None:
    class Colliding(ComponentSchema):
        model_config = ConfigDict(extra="allow")

    original_config = Colliding.model_config.copy()
    try:
        with registered_model("notebook/collision-values", Colliding) as kind:
            entry, results, database = create_entry(tmp_path)
            entry.setup.add_component(
                "N1", kind=kind, model_config={"marker": "first"}, model_dump="first"
            )
            entry.setup.add_component(
                "N2", kind=kind, model_config={"marker": "second"}
            )
            view = entry.setup if owner == "setup" else entry.new_point("working")
            source = (
                results / "entry/setup.yaml"
                if owner == "setup"
                else results / "entry/points/working/point.yaml"
            )
            yaml = YAML(typ="rt")
            stored = yaml.load(source)
            stored["format_version"] = version
            yaml.dump(stored, source)
            before = source.read_bytes()
            view.refresh()
            assert source.read_bytes() == before
            config = field_view(view.N1.model_config)
            assert config.marker == "first"
            assert view.N1.model_dump == "first"
            untouched = view.meta("N2.model_config.marker")
            config.marker = "child"
            assert config.marker == "child"
            view.N1.model_config = {"marker": "whole"}
            with view.edit() as draft:
                assert field_view(draft.N1.model_config).marker == "whole"
                draft.set("N1.model_config.marker", "accepted")
                draft.N1.model_dump = "accepted"
            accepted = view.meta("N1.model_config.marker")
            assert accepted is not None and accepted.source == "manual"
            method_source = view.meta("N1.model_dump")
            assert method_source is not None and method_source.source == "manual"
            assert config.marker == "accepted"
            assert view.N1.model_dump == "accepted"
            assert field_view(view.N2.model_config).marker == "second"
            assert view.meta("N2.model_config.marker") == untouched
            reopened = ResultEntry.open(
                "entry", result_root=results, database_root=database
            )
            loaded = (
                reopened.setup if owner == "setup" else reopened.use_point("working")
            )
            assert field_view(loaded.N1.model_config).marker == "accepted"
            assert loaded.N1.model_dump == "accepted"
            assert loaded.meta("N1.model_config.marker") == accepted
            assert loaded.meta("N1.model_dump") == method_source
            assert YAML(typ="safe").load(source)["format_version"] == version
    finally:
        Colliding.model_config = original_config


@pytest.mark.parametrize("owner", ["setup", "point"])
def test_public_snapshot_and_source_path_preserve_cached_binding_and_isolation(
    tmp_path: Path, owner: str
) -> None:
    class Measured(ComponentSchema):
        values: list[int]

    with registered_model("notebook/read-snapshot", Measured) as kind:
        entry, results, database = create_entry(tmp_path)
        for name in ("N1", "snapshot", "source_path", "path", "accept_source"):
            entry.setup.add_component(name, kind=kind, values=[2, 3])
        view = entry.setup if owner == "setup" else entry.new_point("working")
        expected = (
            results
            / "entry"
            / ("setup.yaml" if owner == "setup" else "points/working/point.yaml")
        )
        assert source_path(view) == expected
        captured = snapshot(view)
        assert isinstance(
            captured, SetupDocument if owner == "setup" else PointDocument
        )
        assert view.snapshot.values == [2, 3]
        assert view.source_path.values == [2, 3]
        assert view.path.values == [2, 3]
        assert view.accept_source.values == [2, 3]
        component = captured.components["N1"]
        assert isinstance(component, Measured)
        component.values.append(9)
        captured.provenance["N1.values"]["source"] = "changed"
        accepted = view.meta("N1.values")
        assert accepted is not None and accepted.source == "manual"
        assert document_value(snapshot(view), "N1.values") == [2, 3]
        captured.components.clear()
        captured.general.ext["new"] = [1]
        assert document_value(snapshot(view), "N1.values") == [2, 3]
        assert snapshot(view).general.ext == {}
        other = ResultEntry.open("entry", result_root=results, database_root=database)
        other_view = other.setup if owner == "setup" else other.use_point("working")
        other_view.N1.values = [7]
        assert document_value(snapshot(view), "N1.values") == [2, 3]
        view.refresh()
        assert document_value(snapshot(view), "N1.values") == [7]
        expected.write_text("invalid: [", encoding="utf-8")
        assert source_path(view) == expected
        assert document_value(snapshot(view), "N1.values") == [7]
        assert expected.read_text(encoding="utf-8") == "invalid: ["


@pytest.mark.parametrize("owner", ["setup", "point"])
def test_public_edit_snapshot_isolates_uncommitted_values_and_sources(
    tmp_path: Path, owner: str
) -> None:
    class Measured(ComponentSchema):
        freq: float

    with registered_model("notebook/read-draft", Measured) as kind:
        entry, _, _ = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, freq=10.0)
        view = entry.setup if owner == "setup" else entry.new_point("working")
        before = source_path(view).read_bytes()

        def abort_edit() -> None:
            with view.edit() as draft:
                draft.set("N1.freq", 12.0)
                captured = snapshot(draft)
                assert document_value(captured, "N1.freq") == 12.0
                assert document_value(snapshot(view), "N1.freq") == 10.0
                accepted = document_meta(captured, "N1.freq")
                assert accepted is not None and accepted.source == "manual"
                captured.components.clear()
                captured.provenance.clear()
                assert document_value(snapshot(draft), "N1.freq") == 12.0
                assert document_meta(snapshot(draft), "N1.freq") is not None
                raise RuntimeError("abort")

        with pytest.raises(RuntimeError, match="abort"):
            abort_edit()
        assert document_value(snapshot(view), "N1.freq") == 10.0
        assert source_path(view).read_bytes() == before


@pytest.mark.parametrize("owner", ["setup", "point"])
def test_document_value_reads_logical_fields_extras_and_containers_without_hooks(
    tmp_path: Path, owner: str
) -> None:
    calls: list[str] = []

    class Timing(BaseModel):
        duration: float

    class Measured(ComponentSchema):
        model_config = ConfigDict(extra="allow")
        freq: float
        timing: Timing
        values: list[int]
        nullable: float | None = None
        absent: float | None = None

        def model_post_init(self, context: object) -> None:
            calls.append("validate")

        @field_serializer("freq")
        def serialize_freq(self, value: float) -> str:
            calls.append("serialize")
            return str(value)

    with registered_model("notebook/read-logical", Measured) as kind:
        entry, _, _ = create_entry(tmp_path)
        entry.setup.add_component(
            "N1",
            kind=kind,
            freq=10.0,
            timing={"duration": 0.2},
            values=[1, 2],
            nullable=None,
            model_config={"marker": "data"},
            model_dump="extra",
        )
        view = entry.setup if owner == "setup" else entry.new_point("working")
        view.general.ext["temperature"] = 0.03
        document = snapshot(view)
        before_calls = calls.copy()
        assert document_value(document, "N1.freq") == 10.0
        assert document_value(document, "N1.timing.duration") == 0.2
        assert document_value(document, "N1.timing") == {"duration": 0.2}
        assert document_value(document, "N1.model_config.marker") == "data"
        assert document_value(document, "N1.model_dump") == "extra"
        assert document_value(document, "general.ext.temperature") == 0.03
        assert document_value(document, "N1.nullable") is None
        with pytest.raises(AttributeError, match="absent"):
            document_value(document, "N1.absent")
        projected = document_value(document, "N1")
        assert isinstance(projected, dict)
        assert projected["freq"] == 10.0
        assert "absent" not in projected
        values = projected["values"]
        assert isinstance(values, list)
        values.append(9)
        projected["model_config"] = {"marker": "changed"}
        assert document_value(document, "N1.values") == [1, 2]
        assert document_value(document, "N1.model_config.marker") == "data"
        assert calls == before_calls


@pytest.mark.parametrize("owner", ["setup", "point"])
def test_document_fields_preserve_logical_rows_sources_and_literal_keys(
    tmp_path: Path, owner: str
) -> None:
    calls: list[str] = []

    class Timing(BaseModel):
        duration: float

    class Measured(ComponentSchema):
        freq: float
        timing: Timing
        values: list[int]
        empty: dict[str, int]
        nullable: float | None = None
        absent: float | None = None

        @field_serializer("freq")
        def serialize_freq(self, value: float) -> str:
            calls.append("serialize")
            return str(value)

    with registered_model("notebook/fields", Measured) as kind:
        entry, _, _ = create_entry(tmp_path)
        entry.setup.add_component(
            "N1",
            kind=kind,
            freq=10.0,
            timing={"duration": 0.2},
            values=[1, 2],
            empty={},
            nullable=None,
        )
        if owner == "setup":
            view = entry.setup
        else:
            entry.new_point("seed")
            view = entry.new_point("working", clone_from="seed")
        view.general.ext["mapping"] = {"a.b": 1, "": 2, "safe": 3}
        document = snapshot(view)
        before_calls = calls.copy()
        rows = document_fields(document)
        components = [row for row in rows if row.segments[0] == "N1"]
        assert [(row.segments, row.value) for row in components] == [
            (("N1", "kind"), kind),
            (("N1", "freq"), 10.0),
            (("N1", "timing", "duration"), 0.2),
            (("N1", "values"), [1, 2]),
            (("N1", "empty"), {}),
            (("N1", "nullable"), None),
        ]
        extension = [
            row for row in rows if row.segments[:3] == ("general", "ext", "mapping")
        ]
        assert [
            (row.segments[-1], row.editable_path, row.value) for row in extension
        ] == [
            ("a.b", None, 1),
            ("", None, 2),
            ("safe", "general.ext.mapping.safe", 3),
        ]
        frequency = next(row for row in rows if row.editable_path == "N1.freq")
        source = frequency.provenance
        assert source == document_meta(document, "N1.freq")
        assert source is not None and source.source == "manual"
        if owner == "point":
            assert source.cloned_from == {"entry_id": entry.entry_id, "point": "seed"}
        values = next(row.value for row in rows if row.editable_path == "N1.values")
        assert isinstance(values, list)
        values.append(9)
        assert document_value(document, "N1.values") == [1, 2]
        assert calls == before_calls


@pytest.mark.parametrize("path", ["", ".N1", "N1.", "N1..freq"])
def test_document_read_and_meta_reject_unaddressable_paths(
    tmp_path: Path, path: str
) -> None:
    entry, _, _ = create_entry(tmp_path)
    document = snapshot(entry.setup)
    for read in (document_value, document_meta):
        with pytest.raises(ValueError, match="path"):
            read(document, path)


@pytest.mark.parametrize(
    "path", ["missing.freq", "general.unknown", "general.created_at.year"]
)
def test_document_value_missing_is_distinct_from_metadata_absence(
    tmp_path: Path, path: str
) -> None:
    entry, _, _ = create_entry(tmp_path)
    document = snapshot(entry.setup)
    with pytest.raises(AttributeError, match=path):
        document_value(document, path)
    assert document_meta(document, path) is None


@pytest.mark.parametrize("extra_policy", ["forbid", "ignore", "allow"])
def test_public_document_read_preserves_forward_minor_visibility(
    tmp_path: Path, extra_policy: str
) -> None:
    class Measured(ComponentSchema):
        freq: float

    if extra_policy == "forbid":
        Measured.model_config = ConfigDict(extra="forbid")
    elif extra_policy == "ignore":
        Measured.model_config = ConfigDict(extra="ignore")
    else:
        Measured.model_config = ConfigDict(extra="allow")
    Measured.model_rebuild(force=True)
    with registered_model("notebook/read-forward", Measured) as kind:
        entry, _, _ = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, freq=10.0)
        path = source_path(entry.setup)
        yaml = YAML(typ="rt")
        stored = yaml.load(path)
        stored["format_version"] = "1.2"
        stored["components"]["N1"]["future"] = {"count": 2}
        yaml.dump(stored, path)
        entry.setup.refresh()
        before = path.read_bytes()
        document = snapshot(entry.setup)
        if extra_policy == "allow":
            assert document_value(document, "N1.future.count") == 2
        else:
            with pytest.raises(AttributeError, match="future"):
                document_value(document, "N1.future.count")
        assert document_value(document, "N1.freq") == 10.0
        assert path.read_bytes() == before


@pytest.fixture
def memory_point(registry_state_guard: None) -> Generator[PointDocument]:
    class Measured(ComponentSchema):
        model_config = ConfigDict(extra="forbid")
        freq: float = Field(gt=0)
        wiring: dict[str, float]
        nullable: float | None = None

    with registered_model("notebook/memory-point", Measured) as kind:
        yield PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-06T00:00:00Z"},
                "components": {
                    "N1": {"kind": kind, "freq": 10.0, "wiring": {"delay": 0.2}}
                },
                "provenance": {
                    "N1.wiring.delay": {
                        "source": "manual",
                        "kind": None,
                        "run_id": None,
                        "at": "2026-10-05T00:00:00Z",
                        "stderr": None,
                        "cloned_from": {
                            "entry_id": "9d25d7f2-f8c2-4893-bd36-1c27f8f48b61",
                            "point": "old",
                        },
                    }
                },
            }
        )


@pytest.mark.parametrize("stderr", [None, 0.0, 0.125])
def test_manual_update_prepares_isolated_working_value_and_source(
    memory_point: PointDocument, stderr: float | None
) -> None:
    before = memory_point.model_copy(deep=True)
    started = datetime.now(timezone.utc)
    updated = manual_update(memory_point, "N1.freq", 12.0, stderr=stderr)
    finished = datetime.now(timezone.utc)
    assert document_value(updated, "N1.freq") == 12.0
    source = document_meta(updated, "N1.freq")
    assert source is not None
    assert source.source == "manual"
    assert source.kind is None and source.run_id is None
    assert source.stderr == stderr and source.cloned_from is None
    assert started <= datetime.fromisoformat(source.at) <= finished
    assert updated.general.created_at == before.general.created_at
    assert updated.provenance["N1.wiring.delay"] == before.provenance["N1.wiring.delay"]
    assert memory_point == before
    updated.components.clear()
    updated.provenance.clear()
    assert memory_point == before


def test_manual_update_accepts_container_leaves_and_preserves_input(
    memory_point: PointDocument,
) -> None:
    value: YamlMap = {"delay": 0.4, "offset": 0.1}
    updated = manual_update(memory_point, "N1.wiring", value, stderr=0.02)
    value["delay"] = 99.0
    assert document_value(updated, "N1.wiring") == {"delay": 0.4, "offset": 0.1}
    for path in ("N1.wiring", "N1.wiring.delay", "N1.wiring.offset"):
        source = document_meta(updated, path)
        assert source is not None
        assert source.source == "manual" and source.stderr == 0.02
        assert source.cloned_from is None
    assert memory_point.provenance["N1.wiring.delay"]["cloned_from"] is not None
    general = manual_update(updated, "general.ext.temperature", 0.03)
    assert document_value(general, "general.ext.temperature") == 0.03
    assert document_meta(general, "general.ext.temperature") is not None
    assert document_value(updated, "general.ext") == {}
    assert general.general.created_at == memory_point.general.created_at


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        ("", 1.0, ValueError),
        ("N1", 1.0, ValueError),
        ("N1..freq", 1.0, ValueError),
        ("missing.freq", 1.0, AttributeError),
        ("N1.freq.child", 1.0, AttributeError),
        ("N1.freq", -1.0, ValidationError),
        ("N1.unknown", 1.0, ValidationError),
        ("general.created_at", "not-a-time", ValidationError),
    ],
)
def test_manual_update_rejects_invalid_write_without_mutating_input(
    memory_point: PointDocument, path: str, value: float | str, error: type[Exception]
) -> None:
    before = memory_point.model_copy(deep=True)
    with pytest.raises(error):
        manual_update(memory_point, path, value)
    assert memory_point == before


@pytest.mark.parametrize("stderr", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_manual_update_rejects_invalid_stderr_without_mutating_input(
    memory_point: PointDocument, stderr: float
) -> None:
    before = memory_point.model_copy(deep=True)
    with pytest.raises(ValueError, match="stderr"):
        manual_update(memory_point, "N1.freq", 12.0, stderr=stderr)
    assert memory_point == before


@pytest.mark.parametrize("owner", ["setup", "point"])
def test_accept_source_attaches_staged_value_without_rerunning_model_hooks(
    tmp_path: Path, registry_state_guard: None, owner: str
) -> None:
    hooks: list[str] = []

    class Measured(ComponentSchema):
        freq: float
        wiring: dict[str, float]

        def model_post_init(self, context: object) -> None:
            hooks.append("validated")

    with registered_model("notebook/source-stage", Measured) as kind:
        entry, _, _ = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, freq=10.0, wiring={"delay": 0.2})
        view = entry.setup if owner == "setup" else entry.new_point("working")
        before_disk = source_path(view).read_bytes()
        with view.edit() as draft:
            draft.set("N1.wiring", {"delay": 0.4, "offset": 0.1})
            before_source = snapshot(draft)
            event = LedgerEvent(
                id="92f1a2ca-71bf-4fa5-98eb-ff77d29f091a",
                kind="accepted",
                at="2026-10-06T00:00:00Z",
                entry_id=entry.entry_id,
                origin="notebook",
                payload=AcceptedPayload(
                    writes=(AcceptedWrite(path="N1.wiring.delay", value=0.4),),
                    analysis_summary={},
                ),
            )
            entry.ledger.append(event)
            provenance = Provenance(event.id, "fit", None, event.at, 0.03)
            before_hooks = hooks.copy()
            accept_source(draft, "N1.wiring", provenance=provenance)
            captured = snapshot(draft)
            assert captured.components == before_source.components
            assert captured.general == before_source.general
            assert hooks == before_hooks
            for path in ("N1.wiring", "N1.wiring.delay", "N1.wiring.offset"):
                assert document_meta(captured, path) == provenance
            assert source_path(view).read_bytes() == before_disk
        assert view.meta("N1.wiring.delay") == provenance
        assert view.meta("N1.wiring.offset") == provenance
        view.refresh()
        assert view.meta("N1.wiring.delay") == provenance


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize(
    ("path", "source", "error"),
    [
        ("", "manual", ValueError),
        ("N1..rate", "manual", ValueError),
        ("N1.missing", "manual", AttributeError),
        ("N1.rate", "absent-event", ValueError),
    ],
)
def test_accept_source_rejects_invalid_path_or_source_without_changing_draft(
    tmp_path: Path, owner: str, path: str, source: str, error: type[Exception]
) -> None:
    entry, _, _ = create_entry(tmp_path)
    entry.setup.add_component("N1", kind="fake/drive/a", rate=10.0)
    view = entry.setup if owner == "setup" else entry.new_point("working")
    with view.edit() as draft:
        before = snapshot(draft)
        with pytest.raises(error) as caught:
            accept_source(
                draft,
                path,
                provenance=Provenance(source, None, None, "2026-10-06T00:00:00Z", None),
            )
        if source == "absent-event":
            assert isinstance(caught.value.__cause__, KeyError)
        assert snapshot(draft) == before


@pytest.mark.parametrize(
    "path", ["N1.ext.notes", "N1.ext.notes.level", "general.ext.nullable"]
)
def test_accept_source_clears_clone_origin_and_accepts_explicit_null(
    tmp_path: Path, path: str
) -> None:
    entry, _, _ = create_entry(tmp_path)
    entry.setup.add_component("N1", kind="fake/drive/a", rate=10.0)
    original = entry.new_point("original")
    with original.edit() as draft:
        draft.set("N1.ext.notes", {"level": 1.0})
        draft.set("general.ext.nullable", None)
    original_before = snapshot(original)
    clone = entry.new_point("clone", clone_from="original")
    before = snapshot(clone)
    assert before.provenance[path]["cloned_from"] is not None
    source = Provenance("manual", None, None, "2026-10-06T00:00:00Z", 0.0)
    with clone.edit() as draft:
        accept_source(draft, path, provenance=source)
        after = snapshot(draft)
        assert after.components == before.components
        assert after.general == before.general
        assert "cloned_from" not in after.provenance[path]
        assert after.provenance[path]["stderr"] == 0.0
        if path == "N1.ext.notes":
            assert "cloned_from" not in after.provenance["N1.ext.notes.level"]
        else:
            assert after.provenance["N1.ext.notes"] == before.provenance["N1.ext.notes"]
    clone.refresh()
    assert snapshot(clone).provenance[path] == after.provenance[path]
    assert document_value(snapshot(clone), "general.ext.nullable") is None
    assert snapshot(original) == original_before
