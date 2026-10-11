"""Detached document rows, static units, logical values and isolated snapshots."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_serializer
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    ComponentSchema,
    ResultEntry,
    document_fields,
    document_meta,
    document_unit,
    document_value,
    snapshot,
    source_path,
)
from zcu_tools.resources.entry.schema import PointDocument, SetupDocument

from .view_support import create_entry, registered_model


@pytest.mark.parametrize("owner", ["setup", "point"])
def test_static_units_follow_declared_fields_without_reading_values(
    tmp_path: Path, registry_state_guard: None, owner: str
) -> None:
    def reject_schema_callback(_schema: object) -> None:
        raise AssertionError("Static unit reads must not execute schema callbacks")

    class Nested(BaseModel):
        arbitrary: float = Field(default=2.0, json_schema_extra={"unit": "quux"})

    class Measured(ComponentSchema):
        frequency: float = Field(default=12.0, json_schema_extra={"unit": "MHz"})
        unset: float | None = Field(default=None, json_schema_extra={"unit": "us"})
        nested: Nested = Field(default_factory=Nested)
        absent: Nested | None = None
        plain: float = 3.0
        values: list[float] = Field(default_factory=lambda: [1.0])
        labeled_values: list[float] = Field(
            default_factory=lambda: [1.0], json_schema_extra={"unit": "list-label"}
        )
        labeled_mapping: dict[str, float] = Field(
            default_factory=lambda: {"sample": 2.0},
            json_schema_extra={"unit": "mapping-label"},
        )
        dynamic: float = Field(default=4.0, json_schema_extra=reject_schema_callback)
        flux: float = 0.0
        ext: YamlMap = Field(default_factory=dict)

    with registered_model("notebook/static-units", Measured) as kind:
        entry, _, _ = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, ext={"frequency": 5.0})
        view = entry.setup if owner == "setup" else entry.new_point("working")
        document = snapshot(view)
        assert document_unit(document, "N1.frequency") == "MHz"
        assert document_unit(document, "N1.unset") == "us"
        assert document_unit(document, "N1.nested.arbitrary") == "quux"
        assert document_unit(document, "N1.labeled_values") == "list-label"
        assert document_unit(document, "N1.labeled_mapping") == "mapping-label"
        for path in (
            "N1.absent.arbitrary",
            "N1.plain",
            "N1.dynamic",
            "N1.flux",
            "N1.ext.frequency",
            "N1.values",
            "N1.values.0",
            "N1.labeled_values.0",
            "N1.labeled_mapping.sample",
            "N1.missing",
            "missing.frequency",
            "general.ext.frequency",
            "general.ext.flux_value",
        ):
            assert document_unit(document, path) is None
        assert document_value(document, "N1.frequency") == 12.0
        with pytest.raises(AttributeError):
            document_value(document, "N1.unset")
        assert snapshot(view) == document


@pytest.mark.parametrize("path", ["", ".N1.frequency", "N1..frequency", "N1."])
def test_static_units_reject_unaddressable_dotted_paths(
    tmp_path: Path, registry_state_guard: None, path: str
) -> None:
    entry, _, _ = create_entry(tmp_path)
    with pytest.raises(ValueError, match="dotted path"):
        document_unit(snapshot(entry.setup), path)


@pytest.mark.parametrize("unit", ["", 3, None])
def test_static_units_reject_malformed_metadata(
    tmp_path: Path, registry_state_guard: None, unit: str | int | None
) -> None:
    class Measured(ComponentSchema):
        value: float = Field(default=1.0, json_schema_extra={"unit": unit})

    with registered_model("notebook/invalid-unit", Measured) as kind:
        entry, _, _ = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind)
        with pytest.raises(ValueError, match="unit"):
            document_unit(snapshot(entry.setup), "N1.value")


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
        cached = snapshot(entry.setup)
        yaml.dump(stored, path)
        before = path.read_bytes()
        if extra_policy == "forbid":
            with pytest.raises(ValidationError, match="future"):
                entry.setup.refresh()
            assert snapshot(entry.setup) == cached
        else:
            entry.setup.refresh()
        document = snapshot(entry.setup)
        if extra_policy == "allow":
            assert document_value(document, "N1.future.count") == 2
        else:
            with pytest.raises(AttributeError, match="future"):
                document_value(document, "N1.future.count")
        assert document_value(document, "N1.freq") == 10.0
        assert path.read_bytes() == before
