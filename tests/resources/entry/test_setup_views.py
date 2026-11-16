"""Setup view validation, forward-minor fields and shared drafts."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    ResultEntry,
    UnknownKindError,
    document_revision,
    snapshot,
)
from zcu_tools.resources.entry.views import FieldView


@pytest.mark.parametrize("point", [None, "one"])
def test_parameters_opener_reads_only_its_selected_file(
    entry: ResultEntry, point: str | None
) -> None:
    if point is not None:
        entry.new_point(point)
    directory = entry.result_path if point is None else entry.result_path / "points" / point
    (directory / "library.yaml").write_text("invalid: [", encoding="utf-8")
    (directory / "aliases.yaml").write_text("invalid: [", encoding="utf-8")

    assert entry.parameters(point=point).description is None


@pytest.mark.parametrize("point", [None, "one"])
def test_parameter_refresh_reports_installed_content_changes(
    entry: ResultEntry, point: str | None
) -> None:
    if point is not None:
        entry.new_point(point)
    view = entry.parameters(point)
    assert view.refresh() is False
    writer = ResultEntry.open(
        entry.result_path.name,
        result_root=entry.result_path.parent,
        database_root=entry.database_path.parent,
    ).parameters(point)
    writer.description = "external edit"
    assert view.description != "external edit"
    assert view.refresh(strict=True) is True
    assert view.description == "external edit"
    assert view.refresh(strict=True) is False


def test_strict_refresh_rejects_invalid_global_candidate_and_retains_publication(
    entry: ResultEntry,
) -> None:
    entry.parameters().add_component("R1", kind="fake/sensor", rate=6500.0)
    source = entry.result_path / "parameters.yaml"
    installed = document_revision(entry.parameters())
    write_setup_component(source, "R1", {"kind": "fake/sensor", "rate": "bad"})
    candidate = source.read_bytes()

    with pytest.raises(ValidationError, match="rate"):
        entry.parameters().refresh(strict=True)

    assert entry.parameters().R1.rate == 6500.0
    assert document_revision(entry.parameters()) == installed
    assert source.read_bytes() == candidate
    write_setup_component(source, "R1", {"kind": "fake/sensor", "rate": 6600.0})
    entry.parameters().refresh(strict=True)
    assert entry.parameters().R1.rate == 6600.0
    assert document_revision(entry.parameters()) != installed


def field_view(value: object) -> FieldView:
    """Narrow a container read without bypassing the public view contract."""
    assert isinstance(value, FieldView)
    return value


def test_newer_minor_preserves_unknown_fields_while_known_values_use_working_units(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    entry.parameters().add_component("R1", kind="fake/sensor", rate=6500.0)
    write_forward_setup(setup_path)
    document = YAML(typ="safe").load(setup_path)
    before = setup_path.read_bytes()

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert setup_path.read_bytes() == before
    assert reopened.parameters().R1.rate == pytest.approx(6500.0)
    assert field_view(reopened.parameters().R1.wiring).delay == pytest.approx(1.2)
    reopened.parameters().R1.rate = 6600.0
    with reopened.parameters().edit() as draft:
        draft.set("R1.wiring.delay", 1.5)
        draft.general.description = "edited with the older reader"

    document["components"]["R1"]["rate"] = 6600.0
    document["components"]["R1"]["wiring"]["delay"] = 1.5
    document["general"]["description"] = "edited with the older reader"
    stored = YAML(typ="safe").load(setup_path)
    assert stored == document
    again = ResultEntry.open("entry", result_root=results, database_root=database)
    assert again.parameters().R1.rate == pytest.approx(6600.0)
    assert field_view(again.parameters().R1.wiring).delay == pytest.approx(1.5)
    assert again.parameters().description == "edited with the older reader"


@pytest.mark.parametrize(
    ("operation", "path"),
    [
        ("read", "R1.future_physical"),
        ("write", "R1.future_physical"),
        ("set", "R1.future_physical"),
        ("wiring", "R1.wiring.future_wiring"),
        ("general", "general.future_general"),
        ("add", "R2.future_physical"),
    ],
)
def test_newer_minor_keeps_unknown_fields_outside_the_public_typed_interface(
    entry_roots: tuple[Path, Path], entry: ResultEntry, operation: str, path: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    entry.parameters().add_component("R1", kind="fake/sensor", rate=6500.0)
    write_forward_setup(setup_path)
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "read":
            _ = reopened.parameters().R1.future_physical
        elif operation == "write":
            reopened.parameters().R1.future_physical = 1.0
        elif operation == "set":
            with reopened.parameters().edit() as draft:
                draft.set(path, 1.0)
        elif operation == "wiring":
            field_view(reopened.parameters().R1.wiring).future_wiring = 1.0
        elif operation == "general":
            reopened.parameters().general.future_general = 1.0
        else:
            reopened.parameters().add_component(
                "R2", kind="fake/sensor", future_physical=1.0
            )

    error = AttributeError if operation == "read" else ValidationError
    with pytest.raises(error):
        perform_operation()
    assert setup_path.read_bytes() == before
    assert reopened.parameters().R1.rate == pytest.approx(6500.0)


@pytest.mark.parametrize("operation", ["open", "refresh"])
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("rate", "not a frequency", ValueError),
        ("wiring", {"ch": None}, ValidationError),
        ("kind", "future-kind", UnknownKindError),
    ],
)
def test_newer_minor_still_validates_known_values_and_kinds(
    entry_roots: tuple[Path, Path],
    entry: ResultEntry,
    operation: str,
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    entry.parameters().add_component("R1", kind="fake/sensor", rate=6500.0)
    write_forward_setup(setup_path)
    document = YAML(typ="safe").load(setup_path)
    document["components"]["R1"][field] = value
    with setup_path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(document, stream)
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "refresh":
            entry.parameters().refresh()
            view = entry.parameters()
        else:
            view = ResultEntry.open(
                "entry", result_root=results, database_root=database
            ).parameters()
        assert view.description is None
        snapshot(view)

    expected_error = UnknownKindError if error is UnknownKindError else ValidationError
    with pytest.raises(expected_error):
        perform_operation()
    assert setup_path.read_bytes() == before
    assert entry.parameters().description is None


@pytest.mark.parametrize("method", ["attribute", "path"])
def test_description_alias_and_general_path_validate_before_changing_the_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry, method: str
) -> None:
    results, database = entry_roots
    with entry.parameters().edit() as draft:

        def perform_operation() -> None:
            if method == "attribute":
                # Dynamic attribute input exercises runtime rejection without a typed assignment.
                attribute = "description"
                setattr(draft, attribute, 123)
            else:
                draft.set("general.description", 123)

        with pytest.raises(ValidationError):
            perform_operation()
        assert draft.description is None
        draft.description = "valid after the caught error"
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.parameters().description == "valid after the caught error"


@pytest.mark.parametrize("path", ["R1.raet", "R1.wiring.dealy", "general.descriptin"])
def test_dotted_path_typos_preserve_the_draft_and_sources(
    entry_roots: tuple[Path, Path], entry: ResultEntry, path: str
) -> None:
    results, _database = entry_roots
    entry.parameters().add_component("R1", kind="fake/sensor", rate=6500.0)
    setup_path = results / "entry" / "parameters.yaml"
    before = setup_path.read_bytes()
    previous = entry.parameters().meta("R1.rate")
    with entry.parameters().edit() as draft:
        with pytest.raises(ValidationError):
            draft.set(path, 1.0)
        assert draft.R1.rate == 6500.0
        assert entry.parameters().meta("R1.rate") == previous
    assert setup_path.read_bytes() == before
    assert entry.parameters().meta("R1.rate") == previous


@pytest.mark.parametrize("method", ["attribute", "path"])
@pytest.mark.parametrize("field", ["rate", "wiring.ch"])
def test_draft_attribute_and_path_validation_reject_bad_values_without_tainting_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry, method: str, field: str
) -> None:
    results, database = entry_roots
    entry.parameters().add_component(
        "R1", kind="fake/sensor", rate=6500.0, wiring={"ch": 1}
    )
    with entry.parameters().edit() as draft:
        draft.description = "the remaining valid draft may commit"

        def perform_operation() -> None:
            if method == "path":
                draft.set(f"R1.{field}", "invalid")
            elif field == "rate":
                draft.R1.rate = "invalid"
            else:
                field_view(draft.R1.wiring).ch = "invalid"

        with pytest.raises(ValidationError):
            perform_operation()
        assert draft.R1.rate == pytest.approx(6500.0)
        assert field_view(draft.R1.wiring).ch == 1

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.parameters().description == "the remaining valid draft may commit"
    assert reopened.parameters().R1.rate == pytest.approx(6500.0)
    assert field_view(reopened.parameters().R1.wiring).ch == 1


def test_path_set_and_attribute_edits_use_the_same_shared_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    entry.parameters().add_component(
        "R1", kind="fake/sensor", rate=6500.0, wiring={"ch": 1}
    )
    before = setup_path.read_bytes()
    with entry.parameters().edit() as draft:
        draft.set("R1.rate", 6550.0)
        draft.set("R1.wiring.ch", 2)
        draft.set("R1.ext.note", "same draft")
        draft.set("general.ext.temperature", 0.03)
        draft.set("general.description", "prepared")
        assert draft.R1.rate == pytest.approx(6550.0)
        assert draft.general.ext.temperature == 0.03
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.parameters().R1.rate == pytest.approx(6550.0)
    assert field_view(reopened.parameters().R1.wiring).ch == 2
    assert field_view(reopened.parameters().R1.ext).note == "same draft"
    assert reopened.parameters().general.ext.temperature == 0.03
    assert reopened.parameters().description == "prepared"
    assert YAML(typ="safe").load(setup_path)["components"]["R1"]["rate"] == 6550.0


def test_general_extensions_support_direct_and_shared_draft_writes_without_scaling(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    assert entry.parameters().general.description is None
    entry.parameters().general.description = "prepared metadata"
    assert entry.parameters().description == "prepared metadata"
    entry.parameters().general.ext.temperature = 0.02
    entry.parameters().general.ext["_arbitrary-key"] = {
        "rate": 12.3,
        "flags": [True, None],
    }
    before = setup_path.read_bytes()
    with entry.parameters().edit() as draft:
        draft.general.ext.temperature = 0.03
        draft.general.ext.note = "cooldown"
        draft.description = "physical environment is not encoded in the name"
        assert draft.general.description == draft.description
        assert draft.general.ext.temperature == 0.03
        assert entry.parameters().general.ext.temperature == 0.02
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.parameters().general.ext.temperature == 0.03
    assert reopened.parameters().general.ext.note == "cooldown"
    restored = reopened.parameters().general.ext["_arbitrary-key"]
    assert isinstance(restored, FieldView)
    assert restored.rate == 12.3
    assert restored.flags == [True, None]

    assert (
        reopened.parameters().description
        == "physical environment is not encoded in the name"
    )
    document = YAML(typ="safe").load(setup_path)
    assert document["general"]["ext"]["temperature"] == 0.03


def test_component_draft_edits_publish_together_only_after_context_exit(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    entry.parameters().add_component(
        "R1", kind="fake/sensor", rate=6500.0, wiring={"ch": 1}
    )
    entry.parameters().add_component("R2", kind="fake/sensor", rate=6600.0)
    before = setup_path.read_bytes()
    with entry.parameters().edit() as draft:
        draft.R1.rate = 6550.0
        field_view(draft.R1.wiring).ch = 2
        field_view(draft.R1.ext).note = "shared draft"
        draft.R2.rate = 6650.0
        assert draft.R1.rate == pytest.approx(6550.0)
        assert entry.parameters().R1.rate == pytest.approx(6500.0)
        assert entry.parameters().R2.rate == pytest.approx(6600.0)
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.parameters().R1.rate == pytest.approx(6550.0)
    assert field_view(reopened.parameters().R1.wiring).ch == 2
    assert field_view(reopened.parameters().R1.ext).note == "shared draft"
    assert reopened.parameters().R2.rate == pytest.approx(6650.0)


@pytest.mark.parametrize("kind", ["fake/drive/a", "fake/drive/b"])
def test_reference_chains_survive_reopening_and_can_be_rewired(
    entry_roots: tuple[Path, Path], entry: ResultEntry, kind: str
) -> None:
    results, database = entry_roots
    entry.parameters().add_component("A1", kind="fake/booster", level=0.2)
    entry.parameters().add_component("I1", kind="fake/supply", level=0.3)
    entry.parameters().add_component("I2", kind="fake/supply", level=-0.4)
    entry.parameters().add_component(
        "R1", kind="fake/sensor", rate=6500.0, booster="A1"
    )
    entry.parameters().add_component("R2", kind="fake/sensor", rate=6600.0)
    entry.parameters().add_component("Q1", kind=kind, sense="R1", source="I1")
    entry.parameters().Q1.sense = "R2"
    entry.parameters().Q1.source = "I2"

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.parameters().Q1.sense == "R2"
    assert reopened.parameters().Q1.source == "I2"
    assert reopened.parameters().R1.booster == "A1"
    assert reopened.parameters().R1.rate == pytest.approx(6500.0)
    assert reopened.parameters().I2.level == pytest.approx(-0.4)
    document = YAML(typ="safe").load(results / "entry" / "parameters.yaml")
    assert document["components"]["Q1"]["sense"] == "R2"
    assert document["components"]["Q1"]["source"] == "I2"
    assert document["components"]["R1"]["booster"] == "A1"


def test_device_current_round_trips_without_using_the_entry_name(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    entry.parameters().add_component("I1", kind="fake/supply", level=0.25)
    assert entry.parameters().I1.level == pytest.approx(0.25)
    entry.parameters().I1.level = -0.4

    document = YAML(typ="safe").load(results / "entry" / "parameters.yaml")
    assert document["components"]["I1"]["level"] == pytest.approx(-0.4)
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.parameters().I1.kind == "fake/supply"
    assert reopened.parameters().I1.level == pytest.approx(-0.4)


def test_component_extensions_preserve_arbitrary_yaml_values_without_unit_conversion(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    entry.parameters().add_component(
        "R1", kind="fake/sensor", rate=6500.0, ext={"rate": 12.3}
    )
    extension = field_view(entry.parameters().R1.ext)
    assert extension.rate == 12.3
    extension.note = "unscaled annotation"
    payload: YamlMap = {"rate": 321.0, "optional": None, "flags": [True, "cold"]}
    extension["_arbitrary-key"] = payload

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert field_view(reopened.parameters().R1.ext).note == "unscaled annotation"
    restored = field_view(reopened.parameters().R1.ext)["_arbitrary-key"]
    assert isinstance(restored, FieldView)
    assert restored.rate == 321.0
    assert restored.optional is None
    assert restored.flags == [True, "cold"]
    document = YAML(typ="safe").load(setup_path)["components"]["R1"]
    assert document["rate"] == 6500.0
    assert document["ext"] == {
        "rate": 12.3,
        "note": "unscaled annotation",
        "_arbitrary-key": payload,
    }


@pytest.mark.parametrize("field", ["ch", "ro_ch", "bias_ch"])
@pytest.mark.parametrize("operation", ["add", "write", "open"])
def test_wiring_channels_reject_explicit_null_without_losing_the_valid_snapshot(
    entry_roots: tuple[Path, Path], entry: ResultEntry, field: str, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    entry.parameters().add_component("R1", kind="fake/sensor", wiring={field: 2})
    if operation == "open":
        write_setup_component(
            setup_path, "R1", {"kind": "fake/sensor", "wiring": {field: None}}
        )
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.parameters().add_component(
                "R2", kind="fake/sensor", wiring={field: None}
            )
        elif operation == "write":
            setattr(entry.parameters().R1.wiring, field, None)
        else:
            opened = ResultEntry.open(
                "entry", result_root=results, database_root=database
            )
            assert opened.parameters().description is None
            _ = getattr(opened.parameters().R1.wiring, field)

    error = ValueError if operation == "open" else ValidationError
    with pytest.raises(error, match=field) as failure:
        perform_operation()
    assert field in str(failure.value)
    assert setup_path.read_bytes() == before
    assert getattr(entry.parameters().R1.wiring, field) == 2


def test_wiring_fields_stay_separate_and_use_their_declared_working_units(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    entry.parameters().add_component(
        "R1",
        kind="fake/sensor",
        rate=6500.0,
        wiring={"ch": 2, "delay": 0.4},
    )
    sensor = entry.parameters().R1
    assert field_view(sensor.wiring).ch == 2
    assert field_view(sensor.wiring).delay == pytest.approx(0.4)
    field_view(sensor.wiring).bias_ch = 3
    field_view(sensor.wiring).delay = 0.5

    document = YAML(typ="safe").load(setup_path)["components"]["R1"]
    assert document["wiring"]["delay"] == pytest.approx(0.5)
    assert document["wiring"]["bias_ch"] == 3
    assert "ch" not in document and "bias_ch" not in document
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.parameters().R1.rate == pytest.approx(6500.0)
    assert field_view(reopened.parameters().R1.wiring).delay == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["edit", "description", "general"])
@pytest.mark.parametrize("operation", ["add", "open"])
def test_component_names_cannot_shadow_public_view_members(
    entry_roots: tuple[Path, Path], entry: ResultEntry, name: str, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    if operation == "open":
        write_setup_component(setup_path, name, {"kind": "fake/sensor"})
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.parameters().add_component(name, kind="fake/sensor")
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

    with pytest.raises(ValueError, match="component name"):
        perform_operation()
    assert setup_path.read_bytes() == before


def test_absent_optional_physical_fields_use_the_declared_default_without_writes(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, _database = entry_roots
    entry.parameters().add_component("R1", kind="fake/sensor")
    document = YAML(typ="safe").load(results / "entry" / "parameters.yaml")
    assert "rate" not in document["components"]["R1"]
    assert entry.parameters().R1.rate is None


@pytest.mark.parametrize("operation", ["add", "read", "write", "open"])
def test_forbidden_component_fields_preserve_values_sources_and_disk(
    entry_roots: tuple[Path, Path], entry: ResultEntry, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    entry.parameters().add_component("R1", kind="fake/sensor", rate=6500.0)
    if operation == "open":
        write_setup_component(
            setup_path, "R1", {"kind": "fake/sensor", "rate": 6.5e9, "raet": 6.6e9}
        )
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.parameters().add_component("R2", kind="fake/sensor", raet=6600.0)
        elif operation == "read":
            _ = entry.parameters().R1.raet
        elif operation == "write":
            entry.parameters().R1.raet = 6600.0
        else:
            opened = ResultEntry.open(
                "entry", result_root=results, database_root=database
            )
            assert opened.parameters().R1.rate == 6.5e9
            snapshot(opened.parameters())

    previous = entry.parameters().meta("R1.rate")
    error = AttributeError if operation == "read" else ValidationError

    with pytest.raises(error):
        perform_operation()
    assert setup_path.read_bytes() == before
    assert entry.parameters().R1.rate == 6500.0
    assert entry.parameters().meta("R1.rate") == previous


@pytest.mark.parametrize("operation", ["add", "open"])
def test_unknown_kinds_report_the_source_component_and_a_close_name(
    entry_roots: tuple[Path, Path], entry: ResultEntry, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    if operation == "open":
        write_setup_component(setup_path, "R1", {"kind": "fake/sensr"})
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.parameters().add_component("R1", kind="fake/sensr")
        else:
            opened = ResultEntry.open(
                "entry", result_root=results, database_root=database
            )
            _ = opened.parameters().R1.kind

    with pytest.raises(UnknownKindError) as failure:
        perform_operation()

    assert failure.value.source == setup_path
    assert failure.value.component == "R1"
    assert failure.value.kind == "fake/sensr"
    assert "fake/sensor" in failure.value.suggestions
    assert setup_path.read_bytes() == before


def test_component_frequency_round_trips_in_working_units(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "parameters.yaml"
    entry.parameters().add_component("R1", kind="fake/sensor", rate=6500.0)
    sensor = entry.parameters().R1
    assert sensor.rate == 6500.0
    assert YAML(typ="safe").load(setup_path)["components"]["R1"]["rate"] == 6500.0

    sensor.rate = 6550.0
    assert sensor.rate == 6550.0
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.parameters().R1.rate == 6550.0
    assert YAML(typ="safe").load(setup_path)["components"]["R1"]["rate"] == 6550.0


def test_added_component_survives_reopening_with_its_declared_kind(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    entry.parameters().add_component(
        "R1", kind="fake/sensor", ext={"note": "sense line"}
    )

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.parameters().R1.kind == "fake/sensor"
    document = YAML(typ="safe").load(results / "entry" / "parameters.yaml")
    assert document["components"]["R1"]["ext"]["note"] == "sense line"


def write_forward_setup(path: Path) -> None:
    document = YAML(typ="safe").load(path)
    document["format_version"] = "2.7"
    document["future_top"] = {"next": [1, "two"]}
    document["general"]["future_general"] = "next metadata"
    document["components"]["R1"]["wiring"] = {"delay": 1.2}
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(document, stream)


def write_setup_component(path: Path, name: str, fields: YamlMap) -> None:
    document = YAML(typ="safe").load(path)
    document["components"][name] = fields
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(document, stream)
