"""ResultEntry creation, identity and filesystem transactions through public APIs."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    MissingReferenceError,
    RenameRecoveryError,
    ResultEntry,
    UnknownFieldError,
    UnknownKindError,
    rename_entry,
)


@pytest.fixture
def entry_roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "results", tmp_path / "Database"


@pytest.fixture
def entry(entry_roots: tuple[Path, Path]) -> ResultEntry:
    results, database = entry_roots
    return ResultEntry.create("entry", result_root=results, database_root=database)


def test_newer_minor_preserves_unknown_fields_while_known_values_use_working_units(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="resonator", freq=6500.0)
    write_forward_setup(setup_path)
    document = YAML(typ="safe").load(setup_path)
    before = setup_path.read_bytes()

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert setup_path.read_bytes() == before
    assert reopened.setup.R1.freq == pytest.approx(6500.0)
    assert reopened.setup.R1.wiring.time_of_flight == pytest.approx(1.2)
    reopened.setup.R1.freq = 6600.0
    with reopened.setup.edit() as draft:
        draft.set("R1.wiring.time_of_flight", 1.5)
        draft.general.description = "edited with the older reader"

    document["components"]["R1"]["freq"] = 6.6e9
    document["components"]["R1"]["wiring"]["time_of_flight"] = 1.5e-6
    document["general"]["description"] = "edited with the older reader"
    stored = YAML(typ="safe").load(setup_path)
    # Source timestamps change on acceptance; source behavior has its own seam tests.
    stored.pop("provenance")
    document.pop("provenance")
    assert stored == document
    again = ResultEntry.open("entry", result_root=results, database_root=database)
    assert again.setup.R1.freq == pytest.approx(6600.0)
    assert again.setup.R1.wiring.time_of_flight == pytest.approx(1.5)
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
    entry.setup.add_component("R1", kind="resonator", freq=6500.0)
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
            reopened.setup.R1.wiring.future_wiring = 1.0
        elif operation == "general":
            reopened.setup.general.future_general = 1.0
        else:
            reopened.setup.add_component("R2", kind="resonator", future_physical=1.0)

    with pytest.raises(UnknownFieldError) as failure:
        perform_operation()
    assert failure.value.path == path
    assert setup_path.read_bytes() == before
    assert reopened.setup.R1.freq == pytest.approx(6500.0)


@pytest.mark.parametrize("operation", ["open", "refresh"])
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("freq", "not a frequency", ValueError),
        ("wiring", {"ch": None}, ValidationError),
        ("amplifier", "absent", MissingReferenceError),
        ("kind", "future-kind", UnknownKindError),
    ],
)
def test_newer_minor_still_validates_known_values_references_and_kinds(
    entry_roots: tuple[Path, Path],
    entry: ResultEntry,
    operation: str,
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="resonator", freq=6500.0)
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
    assert entry.setup.R1.freq == pytest.approx(6500.0)


@pytest.mark.parametrize("method", ["attribute", "path"])
def test_description_alias_and_general_path_validate_before_changing_the_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry, method: str
) -> None:
    results, database = entry_roots
    with entry.setup.edit() as draft:

        def perform_operation() -> None:
            if method == "attribute":
                setattr(draft, "description", 123)  # noqa: B010 -- Invalid runtime input to a typed property.
            else:
                draft.set("general.description", 123)

        with pytest.raises(ValidationError):
            perform_operation()
        assert draft.description is None
        draft.description = "valid after the caught error"
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.description == "valid after the caught error"


@pytest.mark.parametrize(
    ("path", "field", "suggestion"),
    [
        ("R1.frq", "frq", "freq"),
        ("R1.wiring.time_of_flit", "time_of_flit", "time_of_flight"),
        ("general.descriptin", "descriptin", "description"),
    ],
)
def test_dotted_path_typos_report_the_same_field_location_and_suggestions(
    entry_roots: tuple[Path, Path],
    entry: ResultEntry,
    path: str,
    field: str,
    suggestion: str,
) -> None:
    results, _database = entry_roots
    entry.setup.add_component("R1", kind="resonator", freq=6500.0)
    setup_path = results / "entry" / "setup.yaml"
    before = setup_path.read_bytes()
    with entry.setup.edit() as draft:
        with pytest.raises(UnknownFieldError) as failure:
            draft.set(path, 1.0)
        assert failure.value.path == path
        assert failure.value.field == field
        assert suggestion in failure.value.suggestions
    assert setup_path.read_bytes() == before
    assert entry.setup.R1.freq == pytest.approx(6500.0)


@pytest.mark.parametrize("method", ["attribute", "path"])
def test_missing_reference_discards_all_other_shared_draft_changes(
    entry_roots: tuple[Path, Path], entry: ResultEntry, method: str
) -> None:
    results, _database = entry_roots
    entry.setup.add_component("A1", kind="amplifier/jpa")
    entry.setup.add_component("R1", kind="resonator", freq=6500.0, amplifier="A1")
    setup_path = results / "entry" / "setup.yaml"
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        with entry.setup.edit() as draft:
            draft.description = "discarded"
            draft.R1.freq = 6550.0
            if method == "attribute":
                draft.R1.amplifier = "absent"
            else:
                draft.set("R1.amplifier", "absent")

    with pytest.raises(MissingReferenceError):
        perform_operation()
    assert setup_path.read_bytes() == before
    assert entry.setup.description is None
    assert entry.setup.R1.freq == pytest.approx(6500.0)
    assert entry.setup.R1.amplifier == "A1"


@pytest.mark.parametrize("method", ["attribute", "path"])
@pytest.mark.parametrize("field", ["freq", "wiring.ch"])
def test_draft_attribute_and_path_validation_reject_bad_values_without_tainting_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry, method: str, field: str
) -> None:
    results, database = entry_roots
    entry.setup.add_component("R1", kind="resonator", freq=6500.0, wiring={"ch": 1})
    with entry.setup.edit() as draft:
        draft.description = "the remaining valid draft may commit"

        def perform_operation() -> None:
            if method == "path":
                draft.set(f"R1.{field}", "invalid")
            elif field == "freq":
                draft.R1.freq = "invalid"
            else:
                draft.R1.wiring.ch = "invalid"

        with pytest.raises(ValidationError):
            perform_operation()
        assert draft.R1.freq == pytest.approx(6500.0)
        assert draft.R1.wiring.ch == 1

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.description == "the remaining valid draft may commit"
    assert reopened.setup.R1.freq == pytest.approx(6500.0)
    assert reopened.setup.R1.wiring.ch == 1


def test_path_set_and_attribute_edits_use_the_same_shared_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="resonator", freq=6500.0, wiring={"ch": 1})
    before = setup_path.read_bytes()
    with entry.setup.edit() as draft:
        draft.set("R1.freq", 6550.0)
        draft.set("R1.wiring.ch", 2)
        draft.set("R1.ext.note", "same draft")
        draft.set("general.ext.temperature", 0.03)
        draft.set("general.description", "prepared")
        assert draft.R1.freq == pytest.approx(6550.0)
        assert draft.general.ext.temperature == 0.03
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.freq == pytest.approx(6550.0)
    assert reopened.setup.R1.wiring.ch == 2
    assert reopened.setup.R1.ext.note == "same draft"
    assert reopened.setup.general.ext.temperature == 0.03
    assert reopened.setup.description == "prepared"
    assert YAML(typ="safe").load(setup_path)["components"]["R1"]["freq"] == 6.55e9


@pytest.mark.parametrize("operation", ["direct", "draft"])
def test_entry_identity_cannot_be_replaced_through_general_or_a_whole_draft(
    entry_roots: tuple[Path, Path], entry: ResultEntry, operation: str
) -> None:
    results, _database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="resonator", freq=6500.0)
    original_id = entry.entry_id
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        replacement = "00000000-0000-0000-0000-000000000000"
        if operation == "direct":
            entry.setup.general.entry_id = replacement
        else:
            with entry.setup.edit() as draft:
                draft.R1.freq = 6550.0
                draft.description = "must be rolled back with the invalid identity"
                draft.general.entry_id = replacement

    with pytest.raises(ValueError, match="entry_id is immutable"):
        perform_operation()
    assert setup_path.read_bytes() == before
    assert entry.entry_id == original_id
    assert entry.setup.general.entry_id == original_id
    assert entry.setup.description is None
    assert entry.setup.R1.freq == pytest.approx(6500.0)


def test_general_extensions_support_direct_and_shared_draft_writes_without_scaling(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    assert entry.setup.general.description is None
    entry.setup.general.description = "prepared metadata"
    assert entry.setup.description == "prepared metadata"
    entry.setup.general.ext.temperature = 0.02
    entry.setup.general.ext["_arbitrary.key"] = {"freq": 12.3, "flags": [True, None]}
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
    assert reopened.setup.general.ext["_arbitrary.key"] == {
        "freq": 12.3,
        "flags": [True, None],
    }
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
    entry.setup.add_component("R1", kind="resonator", freq=6500.0, wiring={"ch": 1})
    entry.setup.add_component("R2", kind="resonator", freq=6600.0)
    before = setup_path.read_bytes()
    with entry.setup.edit() as draft:
        draft.R1.freq = 6550.0
        draft.R1.wiring.ch = 2
        draft.R1.ext.note = "shared draft"
        draft.R2.freq = 6650.0
        assert draft.R1.freq == pytest.approx(6550.0)
        assert entry.setup.R1.freq == pytest.approx(6500.0)
        assert entry.setup.R2.freq == pytest.approx(6600.0)
        assert setup_path.read_bytes() == before

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.freq == pytest.approx(6550.0)
    assert reopened.setup.R1.wiring.ch == 2
    assert reopened.setup.R1.ext.note == "shared draft"
    assert reopened.setup.R2.freq == pytest.approx(6650.0)


@pytest.mark.parametrize("kind", ["qubit/transmon", "qubit/fluxonium"])
def test_reference_chains_survive_reopening_and_can_be_rewired(
    entry_roots: tuple[Path, Path], entry: ResultEntry, kind: str
) -> None:
    results, database = entry_roots
    entry.setup.add_component("A1", kind="amplifier/jpa", current=0.2)
    entry.setup.add_component("I1", kind="device/current_source", current=0.3)
    entry.setup.add_component("I2", kind="device/current_source", current=-0.4)
    entry.setup.add_component("R1", kind="resonator", freq=6500.0, amplifier="A1")
    entry.setup.add_component("R2", kind="resonator", freq=6600.0)
    entry.setup.add_component("Q1", kind=kind, readout="R1", flux_source="I1")
    entry.setup.Q1.readout = "R2"
    entry.setup.Q1.flux_source = "I2"

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.Q1.readout == "R2"
    assert reopened.setup.Q1.flux_source == "I2"
    assert reopened.setup.R1.amplifier == "A1"
    assert reopened.setup.R1.freq == pytest.approx(6500.0)
    assert reopened.setup.I2.current == pytest.approx(-0.4)
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert document["components"]["Q1"]["readout"] == "R2"
    assert document["components"]["Q1"]["flux_source"] == "I2"
    assert document["components"]["R1"]["amplifier"] == "A1"


@pytest.mark.parametrize(
    ("kind", "field", "target_kind"),
    [
        ("resonator", "amplifier", "amplifier/jpa"),
        ("qubit/fluxonium", "readout", "resonator"),
        ("qubit/fluxonium", "flux_source", "device/current_source"),
        ("qubit/transmon", "readout", "resonator"),
        ("qubit/transmon", "flux_source", "device/current_source"),
    ],
)
@pytest.mark.parametrize("operation", ["add", "write", "open", "refresh"])
def test_missing_component_references_report_location_without_publishing_invalid_values(
    entry_roots: tuple[Path, Path],
    entry: ResultEntry,
    kind: str,
    field: str,
    target_kind: str,
    operation: str,
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("T1", kind=target_kind)
    entry.setup.add_component("C1", kind=kind, **{field: "T1"})
    if operation in ("open", "refresh"):
        write_setup_component(setup_path, "C1", {"kind": kind, field: "absent"})
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.setup.add_component("C2", kind=kind, **{field: "absent"})
        elif operation == "write":
            setattr(entry.setup.C1, field, "absent")
        elif operation == "refresh":
            entry.setup.refresh()
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

    with pytest.raises(MissingReferenceError) as failure:
        perform_operation()
    component = "C2" if operation == "add" else "C1"
    assert failure.value.source == setup_path
    assert failure.value.component == component
    assert failure.value.field == field
    assert failure.value.target == "absent"
    assert f"{component}.{field}" in str(failure.value)
    assert setup_path.read_bytes() == before
    assert getattr(entry.setup.C1, field) == "T1"
    if operation == "add":
        with pytest.raises(AttributeError, match="C2"):
            _ = entry.setup.C2


@pytest.mark.parametrize(
    ("kind", "field", "working_value", "stored_value"),
    [
        ("resonator", "kappa", 2.5, 2.5e6),
        ("amplifier/jpa", "freq", 6500.0, 6.5e9),
        ("amplifier/jpa", "gain", 20.0, 20.0),
        ("amplifier/jpa", "current", 0.2, 0.0002),
        ("qubit/fluxonium", "freq", 500.0, 5e8),
        ("qubit/fluxonium", "EJ", 8.0, 8e9),
        ("qubit/fluxonium", "EC", 1.2, 1.2e9),
        ("qubit/fluxonium", "EL", 0.5, 5e8),
        ("qubit/fluxonium", "flux_half", -0.4, -0.0004),
        ("qubit/fluxonium", "flux_period", 0.8, 0.0008),
        ("qubit/fluxonium", "pi_len", 0.04, 4e-8),
        ("qubit/fluxonium", "t1", 50.0, 5e-5),
        ("qubit/fluxonium", "t2", 25.0, 2.5e-5),
        ("qubit/fluxonium", "pi_gain", 0.3, 0.3),
        ("qubit/transmon", "freq", 5000.0, 5e9),
        ("qubit/transmon", "EJ", 20.0, 2e10),
        ("qubit/transmon", "EC", 0.3, 3e8),
        ("qubit/transmon", "pi_len", 0.02, 2e-8),
        ("qubit/transmon", "t1", 40.0, 4e-5),
        ("qubit/transmon", "t2", 20.0, 2e-5),
        ("qubit/transmon", "pi_gain", 0.2, 0.2),
    ],
)
def test_builtin_physical_fields_round_trip_through_add_write_and_reopen(
    entry_roots: tuple[Path, Path],
    entry: ResultEntry,
    kind: str,
    field: str,
    working_value: float,
    stored_value: float,
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("C1", kind=kind, **{field: working_value})
    component = entry.setup.C1
    assert getattr(component, field) == pytest.approx(working_value)
    assert YAML(typ="safe").load(setup_path)["components"]["C1"][
        field
    ] == pytest.approx(stored_value)

    setattr(component, field, working_value * 2)
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert getattr(reopened.setup.C1, field) == pytest.approx(working_value * 2)
    assert YAML(typ="safe").load(setup_path)["components"]["C1"][
        field
    ] == pytest.approx(stored_value * 2)


def test_device_current_round_trips_without_using_the_entry_name(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    entry.setup.add_component("I1", kind="device/current_source", current=0.25)
    assert entry.setup.I1.current == pytest.approx(0.25)
    entry.setup.I1.current = -0.4

    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert document["components"]["I1"]["current"] == pytest.approx(-0.0004)
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.I1.kind == "device/current_source"
    assert reopened.setup.I1.current == pytest.approx(-0.4)


def test_component_extensions_preserve_arbitrary_yaml_values_without_unit_conversion(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="resonator", freq=6500.0, ext={"freq": 12.3})
    extension = entry.setup.R1.ext
    assert extension.freq == 12.3
    extension.note = "unscaled annotation"
    payload: YamlMap = {"freq": 321.0, "optional": None, "flags": [True, "cold"]}
    extension["_arbitrary.key"] = payload

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.ext.note == "unscaled annotation"
    assert reopened.setup.R1.ext["_arbitrary.key"] == payload
    document = YAML(typ="safe").load(setup_path)["components"]["R1"]
    assert document["freq"] == 6.5e9
    assert document["ext"] == {
        "freq": 12.3,
        "note": "unscaled annotation",
        "_arbitrary.key": payload,
    }


@pytest.mark.parametrize("field", ["ch", "ro_ch", "flux_ch"])
@pytest.mark.parametrize("operation", ["add", "write", "open"])
def test_wiring_channels_reject_explicit_null_without_losing_the_valid_snapshot(
    entry_roots: tuple[Path, Path], entry: ResultEntry, field: str, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="resonator", wiring={field: 2})
    if operation == "open":
        write_setup_component(
            setup_path, "R1", {"kind": "resonator", "wiring": {field: None}}
        )
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.setup.add_component("R2", kind="resonator", wiring={field: None})
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
        kind="resonator",
        freq=6500.0,
        wiring={"ch": 2, "time_of_flight": 0.4},
    )
    resonator = entry.setup.R1
    assert resonator.wiring.ch == 2
    assert resonator.wiring.time_of_flight == pytest.approx(0.4)
    resonator.wiring.flux_ch = 3
    resonator.wiring.time_of_flight = 0.5

    document = YAML(typ="safe").load(setup_path)["components"]["R1"]
    assert document["wiring"]["time_of_flight"] == pytest.approx(0.5e-6)
    assert document["wiring"]["flux_ch"] == 3
    assert "ch" not in document and "flux_ch" not in document
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.freq == pytest.approx(6500.0)
    assert reopened.setup.R1.wiring.time_of_flight == pytest.approx(0.5)


@pytest.mark.parametrize(
    "name", ["", "Q.1", "1Q", "_private", "for", "edit", "description", "general"]
)
@pytest.mark.parametrize("operation", ["add", "open"])
def test_component_names_must_support_unambiguous_public_attribute_access(
    entry_roots: tuple[Path, Path], entry: ResultEntry, name: str, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    if operation == "open":
        write_setup_component(setup_path, name, {"kind": "resonator"})
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.setup.add_component(name, kind="resonator")
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

    with pytest.raises(ValueError, match="component name"):
        perform_operation()
    assert setup_path.read_bytes() == before


def test_absent_optional_physical_fields_are_not_fabricated_or_readable(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, _database = entry_roots
    entry.setup.add_component("R1", kind="resonator")
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert "freq" not in document["components"]["R1"]
    with pytest.raises(AttributeError, match=r"R1\.freq.*not set"):
        _ = entry.setup.R1.freq


@pytest.mark.parametrize("operation", ["add", "read", "write", "open"])
def test_unknown_component_fields_report_the_path_and_a_close_name(
    entry_roots: tuple[Path, Path], entry: ResultEntry, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="resonator", freq=6500.0)
    if operation == "open":
        write_setup_component(
            setup_path, "R1", {"kind": "resonator", "freq": 6.5e9, "frq": 6.6e9}
        )
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.setup.add_component("R2", kind="resonator", frq=6600.0)
        elif operation == "read":
            _ = entry.setup.R1.frq
        elif operation == "write":
            entry.setup.R1.frq = 6600.0
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

    with pytest.raises(UnknownFieldError) as failure:
        perform_operation()

    component = "R2" if operation == "add" else "R1"
    assert failure.value.path == f"{component}.frq"
    assert failure.value.field == "frq"
    assert "freq" in failure.value.suggestions
    assert setup_path.read_bytes() == before


@pytest.mark.parametrize("operation", ["add", "open"])
def test_unknown_kinds_report_the_source_component_and_a_close_name(
    entry_roots: tuple[Path, Path], entry: ResultEntry, operation: str
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    if operation == "open":
        write_setup_component(setup_path, "R1", {"kind": "resonatr"})
    before = setup_path.read_bytes()

    def perform_operation() -> None:
        if operation == "add":
            entry.setup.add_component("R1", kind="resonatr")
        else:
            ResultEntry.open("entry", result_root=results, database_root=database)

    with pytest.raises(UnknownKindError) as failure:
        perform_operation()

    assert failure.value.source == setup_path
    assert failure.value.component == "R1"
    assert failure.value.kind == "resonatr"
    assert "resonator" in failure.value.suggestions
    assert setup_path.read_bytes() == before


def test_component_frequency_round_trips_between_si_and_working_units(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    entry.setup.add_component("R1", kind="resonator", freq=6500.0)
    resonator = entry.setup.R1
    assert resonator.freq == pytest.approx(6500.0)
    assert YAML(typ="safe").load(setup_path)["components"]["R1"]["freq"] == 6.5e9

    resonator.freq = 6550.0
    assert resonator.freq == pytest.approx(6550.0)
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.freq == pytest.approx(6550.0)
    assert YAML(typ="safe").load(setup_path)["components"]["R1"]["freq"] == 6.55e9


def test_added_component_survives_reopening_with_its_declared_kind(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    entry.setup.add_component("R1", kind="resonator", ext={"note": "readout line"})

    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.R1.kind == "resonator"
    document = YAML(typ="safe").load(results / "entry" / "setup.yaml")
    assert document["components"]["R1"]["ext"]["note"] == "readout line"


def write_forward_setup(path: Path) -> None:
    document = YAML(typ="safe").load(path)
    document["format_version"] = "1.7"
    document["future_top"] = {"next": [1, "two"]}
    document["general"]["future_general"] = "next metadata"
    document["components"]["R1"]["future_physical"] = 7.5e9
    document["components"]["R1"]["wiring"] = {
        "time_of_flight": 1.2e-6,
        "future_wiring": {"channel": 3},
    }
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(document, stream)


def write_setup_component(path: Path, name: str, fields: YamlMap) -> None:
    document = YAML(typ="safe").load(path)
    document["components"][name] = fields
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(document, stream)


def read_entry_files(path: Path) -> dict[Path, bytes]:
    return {
        item.relative_to(path): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


@pytest.mark.parametrize("root_index", [0, 1])
def test_open_rejects_entry_links_that_escape_the_configured_root(
    entry_roots: tuple[Path, Path], root_index: int
) -> None:
    results, database = entry_roots
    linked_path = entry_roots[root_index] / "entry"
    linked_path.parent.mkdir()
    linked_path.symlink_to(results.parent / "external", target_is_directory=True)

    with pytest.raises(ValueError, match="entry escapes its root"):
        ResultEntry.open("entry", result_root=results, database_root=database)


def test_setup_reads_use_memory_even_when_the_file_is_unavailable(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, _database = entry_roots
    entry.setup.description = "cached annotation"
    original_id = entry.entry_id
    setup_path = results / "entry" / "setup.yaml"
    setup_path.unlink()

    assert entry.setup.description == "cached annotation"
    assert entry.entry_id == original_id
    with pytest.raises(FileNotFoundError) as failure:
        entry.setup.refresh()
    assert failure.value.filename == str(setup_path)
    assert entry.setup.description == "cached annotation"
    assert entry.entry_id == original_id


def test_setup_edit_discards_body_failure_and_allows_the_next_transaction(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, _database = entry_roots
    entry.setup.description = "committed annotation"
    setup_path = results / "entry" / "setup.yaml"
    before = setup_path.read_bytes()

    def abort_transaction() -> None:
        with entry.setup.edit() as draft:
            draft.description = "discarded annotation"
            assert entry.setup.description == "committed annotation"
            raise RuntimeError("abort sentinel")

    with pytest.raises(RuntimeError, match="abort sentinel"):
        abort_transaction()
    assert entry.setup.description == "committed annotation"
    assert setup_path.read_bytes() == before

    entry.setup.description = "next annotation"
    assert entry.setup.description == "next annotation"


def test_entry_identity_is_read_only_and_refresh_cannot_publish_a_replacement(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, _database = entry_roots
    original_id = entry.entry_id
    entry.setup.description = "original annotation"
    replacement_id = "00000000-0000-0000-0000-000000000001"
    with pytest.raises(AttributeError, match="entry_id"):
        setattr(entry, "entry_id", replacement_id)  # noqa: B010 -- Exercise the readonly descriptor at runtime; static assignment is invalid.

    setup_path = results / "entry" / "setup.yaml"
    yaml = YAML(typ="safe")
    with setup_path.open(encoding="utf-8") as stream:
        document = yaml.load(stream)
    document["general"]["entry_id"] = replacement_id
    document["general"]["description"] = "external annotation"
    with setup_path.open("w", encoding="utf-8") as stream:
        yaml.dump(document, stream)
    before = setup_path.read_bytes()

    with pytest.raises(ValueError, match="entry_id is immutable"):
        entry.setup.refresh()

    assert entry.entry_id == original_id
    assert entry.setup.description == "original annotation"
    assert setup_path.read_bytes() == before


def test_setup_description_uses_one_transaction_for_direct_and_edit_writes(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    assert entry.setup.description is None

    entry.setup.description = "chip annotation"
    with entry.setup.edit() as draft:
        draft.description = "updated annotation"

    assert entry.setup.description == "updated annotation"
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.setup.description == "updated annotation"
    with (results / "entry" / "setup.yaml").open(encoding="utf-8") as stream:
        document = YAML(typ="safe").load(stream)
    assert document["general"]["description"] == "updated annotation"


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("entry_id", "not-a-uuid"),
        ("created_at", "yesterday"),
        ("created_at", "2026-10-04T06:00:00"),
        ("created_at", "2026-10-04T06:00:00+08:00"),
    ],
)
def test_open_validates_uuid_and_utc_created_at_before_publishing_entry(
    entry_roots: tuple[Path, Path],
    entry: ResultEntry,
    field: str,
    invalid: str,
) -> None:
    results, database = entry_roots
    setup_path = results / "entry" / "setup.yaml"
    yaml = YAML(typ="safe")
    with setup_path.open(encoding="utf-8") as stream:
        document = yaml.load(stream)
    document["general"][field] = invalid
    with setup_path.open("w", encoding="utf-8") as stream:
        yaml.dump(document, stream)
    before = setup_path.read_bytes()

    with pytest.raises(ValidationError) as failure:
        ResultEntry.open("entry", result_root=results, database_root=database)

    assert ("general", field) in [error["loc"] for error in failure.value.errors()]
    assert setup_path.read_bytes() == before


@pytest.mark.parametrize("recovery_reason", ["io-failure", "destination-reappeared"])
def test_failed_rename_recovery_reports_current_paths_and_both_causes(
    entry_roots: tuple[Path, Path],
    entry: ResultEntry,
    monkeypatch: pytest.MonkeyPatch,
    recovery_reason: str,
) -> None:
    results, database = entry_roots
    before = read_entry_files(results / "entry")
    original_rename = Path.rename
    cause = OSError("second rename failure")
    recovery_cause = OSError("rename recovery failure")

    def fail_second_and_recovery(source: Path, target: str | Path) -> Path:
        if source == database / "entry":
            if recovery_reason == "destination-reappeared":
                (results / "entry").mkdir()
                (results / "entry" / "external.bin").write_bytes(b"unrelated data")
            raise cause
        if source == results / "renamed" and recovery_reason == "io-failure":
            raise recovery_cause
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_second_and_recovery)
    with pytest.raises(RenameRecoveryError, match="Rename recovery failed") as failure:
        rename_entry("entry", "renamed", result_root=results, database_root=database)

    error = failure.value
    assert error.moved_result == results / "renamed"
    assert error.pending_database == database / "renamed"
    assert error.recovery_destination == results / "entry"
    assert error.cause is cause
    assert error.__cause__ is cause
    if recovery_reason == "io-failure":
        assert error.recovery_cause is recovery_cause
    else:
        assert error.recovery_cause.filename == str(results / "entry")
        assert (results / "entry" / "external.bin").read_bytes() == b"unrelated data"
    assert read_entry_files(results / "renamed") == before
    assert (database / "entry").is_dir()
    assert not (database / "renamed").exists()


def test_rename_recovers_first_root_when_second_rename_fails(
    entry_roots: tuple[Path, Path], entry: ResultEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, database = entry_roots
    before = read_entry_files(results / "entry")
    original_rename = Path.rename
    cause = OSError("second rename failure")

    def fail_second(source: Path, target: str | Path) -> Path:
        if source == database / "entry":
            raise cause
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="second rename failure") as failure:
        rename_entry("entry", "renamed", result_root=results, database_root=database)

    assert failure.value is cause
    assert read_entry_files(results / "entry") == before
    assert not (results / "renamed").exists()
    assert not (database / "renamed").exists()
    assert (
        ResultEntry.open("entry", result_root=results, database_root=database).entry_id
        == entry.entry_id
    )


@pytest.mark.parametrize("root_index", [0, 1])
@pytest.mark.parametrize(
    "shape", ["empty-directory", "file", "broken-symlink", "external-symlink"]
)
def test_rename_refuses_existing_destination_before_moving_either_root(
    entry_roots: tuple[Path, Path], entry: ResultEntry, root_index: int, shape: str
) -> None:
    results, database = entry_roots
    destination = entry_roots[root_index] / "renamed"
    if shape == "empty-directory":
        destination.mkdir()
    elif shape == "file":
        destination.write_bytes(b"unrelated data")
    elif shape == "external-symlink":
        target = results.parent / "external"
        target.mkdir()
        (target / "external.bin").write_bytes(b"unrelated data")
        destination.symlink_to(target, target_is_directory=True)
    else:
        destination.symlink_to("missing-target")
    before_result = read_entry_files(results / "entry")
    before_database = read_entry_files(database / "entry")

    with pytest.raises(FileExistsError) as failure:
        rename_entry("entry", "renamed", result_root=results, database_root=database)

    assert failure.value.filename == str(destination)
    assert read_entry_files(results / "entry") == before_result
    assert read_entry_files(database / "entry") == before_database
    assert (
        ResultEntry.open("entry", result_root=results, database_root=database).entry_id
        == entry.entry_id
    )
    if shape == "file":
        assert destination.read_bytes() == b"unrelated data"
    elif shape == "broken-symlink":
        assert destination.readlink() == Path("missing-target")
    elif shape == "external-symlink":
        assert (
            results.parent / "external" / "external.bin"
        ).read_bytes() == b"unrelated data"
        assert destination.is_symlink()
    else:
        assert destination.is_dir()


def test_rename_moves_both_roots_preserving_identity_and_all_file_bytes(
    entry_roots: tuple[Path, Path], entry: ResultEntry
) -> None:
    results, database = entry_roots
    result_path = results / "entry"
    database_path = database / "entry"
    (result_path / "records" / "synthetic.json").write_bytes(b'{"unchanged": true}')
    (database_path / "data.bin").write_bytes(b"synthetic data")
    result_before = read_entry_files(result_path)
    database_before = read_entry_files(database_path)

    rename_entry(
        "entry", "renamed (no meaning)", result_root=results, database_root=database
    )

    assert not result_path.exists()
    assert not database_path.exists()
    assert read_entry_files(results / "renamed (no meaning)") == result_before
    assert read_entry_files(database / "renamed (no meaning)") == database_before
    assert (
        ResultEntry.open(
            "renamed (no meaning)", result_root=results, database_root=database
        ).entry_id
        == entry.entry_id
    )


@pytest.mark.parametrize("operation", ["create", "open"])
@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../escape",
        "nested/name",
        "nested\\\\name",
        "C:escape",
        "nul\u0000name",
        "absolute",
    ],
)
def test_entry_names_are_safe_single_path_components(
    entry_roots: tuple[Path, Path], name: str, operation: str
) -> None:
    results, database = entry_roots
    if name == "absolute":
        name = str(results.parent / "absolute-outside-root")
    action = ResultEntry.create if operation == "create" else ResultEntry.open

    with pytest.raises(ValueError, match="single path component"):
        action(name, result_root=results, database_root=database)


def test_create_cleans_only_new_entry_when_second_root_cannot_be_created(
    entry_roots: tuple[Path, Path],
) -> None:
    results, database = entry_roots
    database.write_bytes(b"unrelated preexisting file")

    with pytest.raises(NotADirectoryError):
        ResultEntry.create("entry", result_root=results, database_root=database)

    assert not (results / "entry").exists()
    assert database.read_bytes() == b"unrelated preexisting file"


@pytest.mark.parametrize("root_index", [0, 1])
@pytest.mark.parametrize(
    "shape", ["legacy-directory", "file", "broken-symlink", "external-symlink"]
)
def test_create_rejects_existing_destination_without_touching_either_entry(
    entry_roots: tuple[Path, Path], root_index: int, shape: str
) -> None:
    results, database = entry_roots
    existing = entry_roots[root_index] / "entry"
    existing.parent.mkdir()
    if shape == "legacy-directory":
        existing.mkdir()
        marker = existing / "legacy-data.hdf5"
        marker.write_bytes(b"original measurement")
    elif shape == "file":
        existing.write_bytes(b"original measurement")
        marker = existing
    elif shape == "external-symlink":
        target = existing.parent.parent / "external"
        target.mkdir()
        marker = target / "legacy-data.hdf5"
        marker.write_bytes(b"original measurement")
        existing.symlink_to(target, target_is_directory=True)
    else:
        existing.symlink_to("missing-target")
        marker = None
    other = entry_roots[1 - root_index] / "entry"

    with pytest.raises(FileExistsError) as failure:
        ResultEntry.create("entry", result_root=results, database_root=database)

    assert failure.value.filename == str(existing)
    assert not other.exists()
    if marker is not None:
        assert marker.read_bytes() == b"original measurement"
    else:
        assert existing.is_symlink()
        assert existing.readlink() == Path("missing-target")


@pytest.mark.parametrize("missing", ["database", "setup.yaml", "points", "records"])
def test_open_rejects_incomplete_entry_with_actual_missing_path(
    entry_roots: tuple[Path, Path], missing: str
) -> None:
    results, database = entry_roots
    ResultEntry.create("entry", result_root=results, database_root=database)
    missing_path = (
        database / "entry" if missing == "database" else results / "entry" / missing
    )
    if missing_path.is_dir():
        missing_path.rmdir()
    else:
        missing_path.unlink()

    with pytest.raises(FileNotFoundError) as failure:
        ResultEntry.open("entry", result_root=results, database_root=database)

    assert failure.value.filename == str(missing_path)


def test_open_reads_existing_identity_without_rewriting_setup(tmp_path: Path) -> None:
    results = tmp_path / "results"
    database = tmp_path / "Database"
    entry = ResultEntry.create(
        "plain (label)", result_root=results, database_root=database
    )
    setup_path = results / "plain (label)" / "setup.yaml"
    before = setup_path.read_bytes()

    reopened = ResultEntry.open(
        "plain (label)", result_root=results, database_root=database
    )

    assert reopened.entry_id == entry.entry_id
    assert setup_path.read_bytes() == before


def test_create_builds_new_format_entry_with_uuid_and_utc_identity(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    database = tmp_path / "Database"
    entry = ResultEntry.create("Q12_2D[3]", result_root=results, database_root=database)

    assert UUID(entry.entry_id).version == 4
    result_path = results / "Q12_2D[3]"
    assert (result_path / "records").is_dir()
    assert (result_path / "points").is_dir()
    assert (database / "Q12_2D[3]").is_dir()
    with (result_path / "setup.yaml").open(encoding="utf-8") as stream:
        document = YAML(typ="safe").load(stream)
    assert document["format"] == "zcu.parameter-container"
    assert document["format_version"] == "1.0"
    assert document["general"]["entry_id"] == entry.entry_id
    assert datetime.fromisoformat(
        document["general"]["created_at"]
    ).utcoffset() == timezone.utc.utcoffset(None)
