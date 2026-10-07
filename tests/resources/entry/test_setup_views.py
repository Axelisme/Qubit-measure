"""Setup view validation, forward-minor fields and shared drafts."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    ResultEntry,
    UnknownKindError,
)
from zcu_tools.resources.entry.views import FieldView


def field_view(value: object) -> FieldView:
    """Narrow a container read without bypassing the public view contract."""
    assert isinstance(value, FieldView)
    return value


def test_newer_minor_preserves_unknown_fields_while_known_values_use_working_units(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    write_forward_setup(setup_path)
    document = YAML(typ="safe").load(setup_path)
    before = setup_path.read_bytes()

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert setup_path.read_bytes() == before
    assert reopened.setup.R1.rate == pytest.approx(6500.0)
    assert field_view(reopened.setup.R1.wiring).delay == pytest.approx(1.2)
    reopened.setup.R1.rate = 6600.0
    with reopened.setup.edit() as draft:
        draft.set("R1.wiring.delay", 1.5)
        draft.general.description = "edited with the older reader"

    document["components"]["R1"]["rate"] = 6600.0
    document["components"]["R1"]["wiring"]["delay"] = 1.5
    document["general"]["description"] = "edited with the older reader"
    stored = YAML(typ="safe").load(setup_path)
    # Source timestamps change on acceptance; source behavior has its own seam tests.
    stored.pop("provenance")
    document.pop("provenance")
    assert stored == document
    again = ResultEntry.open("entry", result_root=results, database_root=database)
    assert again.setup.R1.rate == pytest.approx(6600.0)
    assert field_view(again.setup.R1.wiring).delay == pytest.approx(1.5)
    assert again.setup.description == "edited with the older reader"


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
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    write_forward_setup(setup_path)
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "read":
            _ = reopened.setup.R1.future_physical
        elif operation == "write":
            reopened.setup.R1.future_physical = 1.0
        elif operation == "set":
            with reopened.setup.edit() as draft:
                draft.set(path, 1.0)
        elif operation == "wiring":
            field_view(reopened.setup.R1.wiring).future_wiring = 1.0
        elif operation == "general":
            reopened.setup.general.future_general = 1.0
        else:
            reopened.setup.add_component("R2", kind="fake/sensor", future_physical=1.0)

    error = AttributeError if operation == "read" else ValidationError
    with pytest.raises(error):
        perform_operation()
    assert setup_path.read_bytes() == before
    assert reopened.setup.R1.rate == pytest.approx(6500.0)


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
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    write_forward_setup(setup_path)
    document = YAML(typ="safe").load(setup_path)
    document["components"]["R1"][field] = value
    with setup_path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(document, stream)
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "refresh":
            entry.setup.refresh()
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

    with pytest.raises(error):
        perform_operation()
    assert setup_path.read_bytes() == before
    assert entry.setup.R1.rate == pytest.approx(6500.0)


@pytest.mark.parametrize("method", ["attribute", "path"])
def test_description_alias_and_general_path_validate_before_changing_the_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry, method: str
) -> None:
    results, database = entry_roots
    with entry.setup.edit() as draft:

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
    assert reopened.setup.description == "valid after the caught error"


@pytest.mark.parametrize("path", ["R1.raet", "R1.wiring.dealy", "general.descriptin"])
def test_dotted_path_typos_preserve_the_draft_and_sources(
    entry_roots: tuple[Path, Path], entry: ResultEntry, path: str
) -> None:
    results, _database = entry_roots
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    setup_path = results / "entry" / "setup.yaml"
    before = setup_path.read_bytes()
    previous = entry.setup.meta("R1.rate")
    with entry.setup.edit() as draft:
        with pytest.raises(ValidationError):
            draft.set(path, 1.0)
        assert draft.R1.rate == 6500.0
        assert entry.setup.meta("R1.rate") == previous
    assert setup_path.read_bytes() == before
    assert entry.setup.meta("R1.rate") == previous


@pytest.mark.parametrize("method", ["attribute", "path"])
@pytest.mark.parametrize("field", ["rate", "wiring.ch"])
def test_draft_attribute_and_path_validation_reject_bad_values_without_tainting_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry, method: str, field: str
) -> None:
    results, database = entry_roots
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0, wiring={"ch": 1})
    with entry.setup.edit() as draft:
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
    assert reopened.setup.description == "the remaining valid draft may commit"
    assert reopened.setup.R1.rate == pytest.approx(6500.0)
    assert field_view(reopened.setup.R1.wiring).ch == 1


def test_path_set_and_attribute_edits_use_the_same_shared_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0, wiring={"ch": 1})
    before = setup_path.read_bytes()
    with entry.setup.edit() as draft:
        draft.set("R1.rate", 6550.0)
        draft.set("R1.wiring.ch", 2)
        draft.set("R1.ext.note", "same draft")
        draft.set("general.ext.temperature", 0.03)
        draft.set("general.description", "prepared")
        assert draft.R1.rate == pytest.approx(6550.0)
        assert draft.general.ext.temperature == 0.03
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.rate == pytest.approx(6550.0)
    assert field_view(reopened.setup.R1.wiring).ch == 2
    assert field_view(reopened.setup.R1.ext).note == "same draft"
    assert reopened.setup.general.ext.temperature == 0.03
    assert reopened.setup.description == "prepared"
    assert YAML(typ="safe").load(setup_path)["components"]["R1"]["rate"] == 6550.0


@pytest.mark.parametrize("operation", ["direct", "draft"])
def test_entry_identity_cannot_be_replaced_through_general_or_a_whole_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry, operation: str
) -> None:
    results, _database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    original_id = entry.entry_id
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        replacement = "00000000-0000-0000-0000-000000000000"
        if operation == "direct":
            entry.setup.general.entry_id = replacement
        else:
            with entry.setup.edit() as draft:
                draft.R1.rate = 6550.0
                draft.description = "must be rolled back with the invalid identity"
                draft.general.entry_id = replacement

    with pytest.raises(ValueError, match="entry_id is immutable"):
        perform_operation()
    assert setup_path.read_bytes() == before
    assert entry.entry_id == original_id
    assert entry.setup.general.entry_id == original_id
    assert entry.setup.description is None
    assert entry.setup.R1.rate == pytest.approx(6500.0)


def test_general_extensions_support_direct_and_shared_draft_writes_without_scaling(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    assert entry.setup.general.description is None
    entry.setup.general.description = "prepared metadata"
    assert entry.setup.description == "prepared metadata"
    entry.setup.general.ext.temperature = 0.02
    entry.setup.general.ext["_arbitrary-key"] = {"rate": 12.3, "flags": [True, None]}
    before = setup_path.read_bytes()
    with entry.setup.edit() as draft:
        draft.general.ext.temperature = 0.03
        draft.general.ext.note = "cooldown"
        draft.description = "physical environment is not encoded in the name"
        assert draft.general.description == draft.description
        assert draft.general.ext.temperature == 0.03
        assert entry.setup.general.ext.temperature == 0.02
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.general.ext.temperature == 0.03
    assert reopened.setup.general.ext.note == "cooldown"
    restored = reopened.setup.general.ext["_arbitrary-key"]
    assert isinstance(restored, FieldView)
    assert restored.rate == 12.3
    assert restored.flags == [True, None]
    assert reopened.setup.general.entry_id == entry.entry_id
    assert (
        reopened.setup.description == "physical environment is not encoded in the name"
    )
    document = YAML(typ="safe").load(setup_path)
    assert document["general"]["ext"]["temperature"] == 0.03


def test_component_draft_edits_publish_together_only_after_context_exit(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0, wiring={"ch": 1})
    entry.setup.add_component("R2", kind="fake/sensor", rate=6600.0)
    before = setup_path.read_bytes()
    with entry.setup.edit() as draft:
        draft.R1.rate = 6550.0
        field_view(draft.R1.wiring).ch = 2
        field_view(draft.R1.ext).note = "shared draft"
        draft.R2.rate = 6650.0
        assert draft.R1.rate == pytest.approx(6550.0)
        assert entry.setup.R1.rate == pytest.approx(6500.0)
        assert entry.setup.R2.rate == pytest.approx(6600.0)
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.rate == pytest.approx(6550.0)
    assert field_view(reopened.setup.R1.wiring).ch == 2
    assert field_view(reopened.setup.R1.ext).note == "shared draft"
    assert reopened.setup.R2.rate == pytest.approx(6650.0)


@pytest.mark.parametrize("kind", ["fake/drive/a", "fake/drive/b"])
def test_reference_chains_survive_reopening_and_can_be_rewired(
    entry_roots: tuple[Path, Path], entry: ResultEntry, kind: str
) -> None:
    results, database = entry_roots
    entry.setup.add_component("A1", kind="fake/booster", level=0.2)
    entry.setup.add_component("I1", kind="fake/supply", level=0.3)
    entry.setup.add_component("I2", kind="fake/supply", level=-0.4)
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0, booster="A1")
    entry.setup.add_component("R2", kind="fake/sensor", rate=6600.0)
    entry.setup.add_component("Q1", kind=kind, sense="R1", source="I1")
    entry.setup.Q1.sense = "R2"
    entry.setup.Q1.source = "I2"

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.Q1.sense == "R2"
    assert reopened.setup.Q1.source == "I2"
    assert reopened.setup.R1.booster == "A1"
    assert reopened.setup.R1.rate == pytest.approx(6500.0)
    assert reopened.setup.I2.level == pytest.approx(-0.4)
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert document["components"]["Q1"]["sense"] == "R2"
    assert document["components"]["Q1"]["source"] == "I2"
    assert document["components"]["R1"]["booster"] == "A1"


def test_device_current_round_trips_without_using_the_entry_name(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    entry.setup.add_component("I1", kind="fake/supply", level=0.25)
    assert entry.setup.I1.level == pytest.approx(0.25)
    entry.setup.I1.level = -0.4

    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert document["components"]["I1"]["level"] == pytest.approx(-0.4)
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.I1.kind == "fake/supply"
    assert reopened.setup.I1.level == pytest.approx(-0.4)


def test_component_extensions_preserve_arbitrary_yaml_values_without_unit_conversion(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0, ext={"rate": 12.3})
    extension = field_view(entry.setup.R1.ext)
    assert extension.rate == 12.3
    extension.note = "unscaled annotation"
    payload: YamlMap = {"rate": 321.0, "optional": None, "flags": [True, "cold"]}
    extension["_arbitrary-key"] = payload

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert field_view(reopened.setup.R1.ext).note == "unscaled annotation"
    restored = field_view(reopened.setup.R1.ext)["_arbitrary-key"]
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
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="fake/sensor", wiring={field: 2})
    if operation == "open":
        write_setup_component(
            setup_path, "R1", {"kind": "fake/sensor", "wiring": {field: None}}
        )
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.setup.add_component("R2", kind="fake/sensor", wiring={field: None})
        elif operation == "write":
            setattr(entry.setup.R1.wiring, field, None)
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

    with pytest.raises(ValidationError) as failure:
        perform_operation()
    assert field in str(failure.value)
    assert setup_path.read_bytes() == before
    assert getattr(entry.setup.R1.wiring, field) == 2


def test_wiring_fields_stay_separate_and_use_their_declared_working_units(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component(
        "R1",
        kind="fake/sensor",
        rate=6500.0,
        wiring={"ch": 2, "delay": 0.4},
    )
    sensor = entry.setup.R1
    assert field_view(sensor.wiring).ch == 2
    assert field_view(sensor.wiring).delay == pytest.approx(0.4)
    field_view(sensor.wiring).bias_ch = 3
    field_view(sensor.wiring).delay = 0.5

    document = YAML(typ="safe").load(setup_path)["components"]["R1"]
    assert document["wiring"]["delay"] == pytest.approx(0.5)
    assert document["wiring"]["bias_ch"] == 3
    assert "ch" not in document and "bias_ch" not in document
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.rate == pytest.approx(6500.0)
    assert field_view(reopened.setup.R1.wiring).delay == pytest.approx(0.5)


@pytest.mark.parametrize("name", ["edit", "description", "general"])
@pytest.mark.parametrize("operation", ["add", "open"])
def test_component_names_cannot_shadow_public_view_members(
    entry_roots: tuple[Path, Path], entry: ResultEntry, name: str, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    if operation == "open":
        write_setup_component(setup_path, name, {"kind": "fake/sensor"})
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.setup.add_component(name, kind="fake/sensor")
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

    with pytest.raises(ValueError, match="component name"):
        perform_operation()
    assert setup_path.read_bytes() == before


def test_absent_optional_physical_fields_are_not_fabricated_or_readable(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, _database = entry_roots
    entry.setup.add_component("R1", kind="fake/sensor")
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert "rate" not in document["components"]["R1"]
    with pytest.raises(AttributeError, match=r"R1\.rate.*not set"):
        _ = entry.setup.R1.rate


@pytest.mark.parametrize("operation", ["add", "read", "write", "open"])
def test_forbidden_component_fields_preserve_values_sources_and_disk(
    entry_roots: tuple[Path, Path], entry: ResultEntry, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    if operation == "open":
        write_setup_component(
            setup_path, "R1", {"kind": "fake/sensor", "rate": 6.5e9, "raet": 6.6e9}
        )
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.setup.add_component("R2", kind="fake/sensor", raet=6600.0)
        elif operation == "read":
            _ = entry.setup.R1.raet
        elif operation == "write":
            entry.setup.R1.raet = 6600.0
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

    previous = entry.setup.meta("R1.rate")
    error = AttributeError if operation == "read" else ValidationError
    with pytest.raises(error):
        perform_operation()
    assert setup_path.read_bytes() == before
    assert entry.setup.R1.rate == 6500.0
    assert entry.setup.meta("R1.rate") == previous


@pytest.mark.parametrize("operation", ["add", "open"])
def test_unknown_kinds_report_the_source_component_and_a_close_name(
    entry_roots: tuple[Path, Path], entry: ResultEntry, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    if operation == "open":
        write_setup_component(setup_path, "R1", {"kind": "fake/sensr"})
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.setup.add_component("R1", kind="fake/sensr")
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

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
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    sensor = entry.setup.R1
    assert sensor.rate == 6500.0
    assert YAML(typ="safe").load(setup_path)["components"]["R1"]["rate"] == 6500.0

    sensor.rate = 6550.0
    assert sensor.rate == 6550.0
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.rate == 6550.0
    assert YAML(typ="safe").load(setup_path)["components"]["R1"]["rate"] == 6550.0


def test_added_component_survives_reopening_with_its_declared_kind(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    entry.setup.add_component("R1", kind="fake/sensor", ext={"note": "sense line"})

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.kind == "fake/sensor"
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert document["components"]["R1"]["ext"]["note"] == "sense line"


def write_forward_setup(path: Path) -> None:
    document = YAML(typ="safe").load(path)
    document["format_version"] = "1.7"
    document["future_top"] = {"next": [1, "two"]}
    document["general"]["future_general"] = "next metadata"
    document["components"]["R1"]["future_physical"] = 7.5e9
    document["components"]["R1"]["wiring"] = {
        "delay": 1.2,
        "future_wiring": {"channel": 3},
    }
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(document, stream)


def write_setup_component(path: Path, name: str, fields: YamlMap) -> None:
    document = YAML(typ="safe").load(path)
    document["components"][name] = fields
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(document, stream)
