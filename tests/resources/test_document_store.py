from collections.abc import Mapping
from contextlib import ExitStack
from math import isnan
from pathlib import Path

import pytest
from filelock import Timeout
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from zcu_tools.format_version import FormatError, VersionError, YamlValue
from zcu_tools.resources.document_store import (
    ConflictError,
    DocumentChange,
    DocumentStore,
    LockTimeoutError,
    UnitSpec,
)


class SyntheticDocument(BaseModel):
    format: str
    format_version: str
    values: dict[str, float]


class ExtensionDocument(SyntheticDocument):
    ext: dict[str, float]


class SequenceDocument(BaseModel):
    format: str
    format_version: str
    values: list[dict[str, bool | int]]


class TypedScalarDocument(BaseModel):
    format: str
    format_version: str
    value: bool | int


class DynamicUnitDocument(BaseModel):
    format: str
    format_version: str
    dimension: str
    value: float


class NestedNumericDocument(BaseModel):
    format: str
    format_version: str
    values: dict[str, dict[str, float]]


class NullableDocument(BaseModel):
    format: str
    format_version: str
    values: dict[str, float | None]


class KnownValues(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: float = Field(ge=0)
    right: float


class StrictDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: str
    format_version: str
    values: KnownValues


@pytest.fixture
def document_path(tmp_path: Path) -> Path:
    path = tmp_path / "document.yaml"
    path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalues:\n  left: 1.0\n  right: 2.0\n",
        encoding="utf-8",
    )
    return path


def make_store(path: Path) -> DocumentStore[SyntheticDocument]:
    return DocumentStore(path, SyntheticDocument, format="synthetic")


def make_scalar_store(path: Path) -> DocumentStore[TypedScalarDocument]:
    return DocumentStore(path, TypedScalarDocument, format="synthetic")


def make_nullable_store(path: Path) -> DocumentStore[NullableDocument]:
    return DocumentStore(path, NullableDocument, format="synthetic")


def test_snapshot_is_an_independent_memory_only_typed_copy(document_path: Path) -> None:
    store = make_store(document_path)
    document_path.unlink()

    snapshot = store.snapshot()
    assert snapshot.values == {"left": 1.0, "right": 2.0}
    snapshot.values["left"] = 99.0
    assert store.snapshot().values["left"] == 1.0


def test_interleaved_edits_merge_different_fields(document_path: Path) -> None:
    first = make_store(document_path)
    second = make_store(document_path)
    with first.edit() as first_draft:
        first_draft.values["left"] = 10.0
        with second.edit() as second_draft:
            second_draft.values["right"] = 20.0

    assert first.snapshot().values == {"left": 10.0, "right": 20.0}
    assert make_store(document_path).snapshot().values == {"left": 10.0, "right": 20.0}
    assert second.snapshot().values == {"left": 1.0, "right": 20.0}


def test_same_field_conflict_rejects_the_whole_transaction(document_path: Path) -> None:
    first = make_store(document_path)
    second = make_store(document_path)
    with ExitStack() as stack:
        first_draft = stack.enter_context(first.edit())
        first_draft.values["left"] = 10.0
        first_draft.values["right"] = 99.0
        with second.edit() as second_draft:
            second_draft.values["left"] = 20.0
        with pytest.raises(ConflictError) as caught:
            stack.close()

    error = caught.value
    assert error.source == document_path
    assert error.path == ("values", "left")
    assert error.original == 1.0
    assert error.current == 20.0
    assert str(document_path) in str(error)
    assert make_store(document_path).snapshot().values == {"left": 20.0, "right": 2.0}
    assert first.snapshot().values == {"left": 1.0, "right": 2.0}


def test_replace_failure_preserves_original_and_cleans_temporary_file(
    document_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(document_path)
    with store.locked():
        pass
    before = document_path.read_bytes()
    siblings = set(document_path.parent.iterdir())
    failure = OSError("replace failed")

    def fail_replace(_source: Path, _target: Path | str) -> Path:
        raise failure

    with ExitStack() as stack:
        draft = stack.enter_context(store.edit())
        draft.values["left"] = 10.0
        monkeypatch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="replace failed") as caught:
            stack.close()

    assert caught.value is failure
    assert document_path.read_bytes() == before
    assert store.snapshot().values == {"left": 1.0, "right": 2.0}
    assert set(document_path.parent.iterdir()) == siblings


def test_roundtrip_preserves_comments_order_and_unchanged_numbers(
    document_path: Path,
) -> None:
    document_path.write_text(
        "# Document note\nformat: synthetic\nformat_version: '1.0'\n"
        "values:\n  left: 1.000 # edited\n  right: 2.000 # unchanged\n",
        encoding="utf-8",
    )
    store = make_store(document_path)
    with store.edit() as draft:
        draft.values["left"] = 10.0

    text = document_path.read_text(encoding="utf-8")
    assert "# Document note" in text
    assert "# edited" in text
    assert "right: 2.000 # unchanged" in text
    assert text.index("format:") < text.index("format_version:") < text.index("values:")
    assert text.index("left:") < text.index("right:")
    assert make_store(document_path).snapshot().values == {"left": 10.0, "right": 2.0}


def test_commit_notifies_after_publication_and_unlock_and_can_unsubscribe(
    document_path: Path,
) -> None:
    store = make_store(document_path)
    other = DocumentStore(
        document_path, SyntheticDocument, format="synthetic", lock_timeout=0.01
    )
    events: list[DocumentChange] = []
    snapshots: list[dict[str, float]] = []
    unlocked: list[bool] = []

    def observe(change: DocumentChange) -> None:
        events.append(change)
        snapshots.append(store.snapshot().values)
        with other.locked():
            unlocked.append(True)

    unsubscribe = store.subscribe(observe)
    with store.edit() as draft:
        draft.values["left"] = 10.0

    assert events == [DocumentChange(document_path, (("values", "left"),), "commit")]
    assert snapshots == [{"left": 10.0, "right": 2.0}]
    assert unlocked == [True]
    unsubscribe()
    with store.edit() as draft:
        draft.values["left"] = 30.0
    assert len(events) == 1


def test_refresh_publishes_external_changes_once(document_path: Path) -> None:
    store = make_store(document_path)
    other = make_store(document_path)
    events: list[DocumentChange] = []
    store.subscribe(events.append)
    with other.edit() as draft:
        draft.values["left"] = 10.0

    assert store.snapshot().values["left"] == 1.0
    assert store.refresh() is True
    assert store.snapshot().values["left"] == 10.0
    assert events == [DocumentChange(document_path, (("values", "left"),), "refresh")]
    assert store.refresh() is False
    assert len(events) == 1


def test_observer_failure_is_reported_separately_from_committed_state(
    document_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = make_store(document_path)
    events: list[DocumentChange] = []
    error = ValueError("observer failed")

    def fail_observer(_change: DocumentChange) -> None:
        raise error

    store.subscribe(fail_observer)
    store.subscribe(events.append)
    with store.edit() as draft:
        draft.values["left"] = 10.0

    assert store.snapshot().values["left"] == 10.0
    assert make_store(document_path).snapshot().values["left"] == 10.0
    assert events == [DocumentChange(document_path, (("values", "left"),), "commit")]
    assert any(
        record.levelname == "ERROR" and record.exc_info and record.exc_info[1] is error
        for record in caplog.records
    )


@pytest.mark.parametrize(
    ("si_unit", "working_unit", "si_value"),
    [
        ("Hz", "MHz", 1_000_000.0),
        ("Hz", "GHz", 1_000_000_000.0),
        ("s", "us", 0.000001),
        ("s", "µs", 0.000001),
        ("A", "mA", 0.001),
        ("1", "1", 1.0),
    ],
)
def test_units_roundtrip_values_and_stderr_without_touching_other_nodes(
    document_path: Path, si_unit: str, working_unit: str, si_value: float
) -> None:
    document_path.write_text(
        f"format: synthetic\nformat_version: '1.0'\nvalues:\n"
        f"  left: {si_value}\n  stderr: {si_value / 10}\n"
        "  unchanged: 4.000 # unchanged\next:\n  left: 9e6 # extension\n",
        encoding="utf-8",
    )
    unit = UnitSpec(si_unit, working_unit)
    store = DocumentStore(
        document_path,
        SyntheticDocument,
        format="synthetic",
        units={("values", "left"): unit, ("values", "stderr"): unit},
    )
    assert store.snapshot().values["left"] == pytest.approx(1.0)
    assert store.snapshot().values["stderr"] == pytest.approx(0.1)
    with store.edit() as draft:
        draft.values["left"] = 2.0

    assert store.snapshot().values["left"] == pytest.approx(2.0)
    on_disk = make_store(document_path).snapshot()
    assert on_disk.values["left"] == pytest.approx(2 * si_value)
    assert on_disk.values["stderr"] == pytest.approx(si_value / 10)
    text = document_path.read_text(encoding="utf-8")
    assert "  unchanged: 4.000 # unchanged" in text
    assert "  left: 9e6 # extension" in text


def test_added_subtree_converts_declared_values_and_stderr_to_si(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalues: {}\n",
        encoding="utf-8",
    )
    units = {
        ("values", "Q1", "freq"): UnitSpec("Hz", "MHz"),
        ("values", "Q1", "stderr"): UnitSpec("Hz", "MHz"),
    }
    store = DocumentStore(
        document_path, NestedNumericDocument, format="synthetic", units=units
    )
    with store.edit() as draft:
        draft.values["Q1"] = {"freq": 5.0, "stderr": 0.1}

    assert store.snapshot().values["Q1"] == pytest.approx({"freq": 5.0, "stderr": 0.1})
    reopened = DocumentStore(
        document_path, NestedNumericDocument, format="synthetic", units=units
    )
    assert reopened.snapshot().values == store.snapshot().values
    on_disk = DocumentStore(document_path, NestedNumericDocument, format="synthetic")
    assert on_disk.snapshot().values["Q1"] == pytest.approx(
        {"freq": 5_000_000.0, "stderr": 100_000.0}
    )


def test_newer_minor_keeps_unknown_nested_fields_outside_the_typed_snapshot(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.2'\nvalues:\n"
        "  left: 1.0\n  right: 2.0\n  future: 7.000 # future nested\n"
        "future_top: [next, version] # future top\n",
        encoding="utf-8",
    )
    store = DocumentStore(document_path, StrictDocument, format="synthetic")
    assert store.snapshot().model_dump() == {
        "format": "synthetic",
        "format_version": "1.2",
        "values": {"left": 1.0, "right": 2.0},
    }
    with store.edit() as draft:
        draft.values.left = 10.0

    assert store.snapshot().values.left == 10.0
    assert store.snapshot().format_version == "1.2"
    text = document_path.read_text(encoding="utf-8")
    assert "  future: 7.000 # future nested" in text
    assert "future_top: [next, version] # future top" in text
    assert (
        DocumentStore(document_path, StrictDocument, format="synthetic")
        .snapshot()
        .values.left
        == 10.0
    )


def test_custom_validation_rejects_the_merged_document_before_publication(
    document_path: Path,
) -> None:
    error = ValueError("combined budget exceeded")

    def bounded_total(document: SyntheticDocument) -> None:
        if sum(document.values.values()) > 10:
            raise error

    first = DocumentStore(
        document_path, SyntheticDocument, format="synthetic", validate=bounded_total
    )
    second = DocumentStore(
        document_path, SyntheticDocument, format="synthetic", validate=bounded_total
    )
    events: list[DocumentChange] = []
    first.subscribe(events.append)
    stack = ExitStack()
    draft = stack.enter_context(first.edit())
    draft.values["left"] = 8.0
    with second.edit() as other:
        other.values["right"] = 8.0
    committed = document_path.read_bytes()
    with pytest.raises(ValueError, match="combined budget exceeded") as raised:
        stack.close()

    assert raised.value is error
    assert document_path.read_bytes() == committed
    assert first.snapshot().values == {"left": 1.0, "right": 2.0}
    assert second.snapshot().values == {"left": 1.0, "right": 8.0}
    assert events == []


def test_real_entry_lock_blocks_an_independent_commit_and_is_reusable(
    document_path: Path,
) -> None:
    lock_path = document_path.parent / ".entry.lock"
    first = DocumentStore(
        document_path, SyntheticDocument, format="synthetic", lock_path=lock_path
    )
    second = DocumentStore(
        document_path,
        SyntheticDocument,
        format="synthetic",
        lock_path=lock_path,
        lock_timeout=0.01,
    )
    first_stack = ExitStack()
    second_stack = ExitStack()
    first_draft = first_stack.enter_context(first.edit())
    second_draft = second_stack.enter_context(second.edit())
    first_draft.values["left"] = 10.0
    second_draft.values["right"] = 20.0
    original = document_path.read_bytes()
    with first.locked(), pytest.raises(LockTimeoutError) as raised:
        second_stack.close()

    assert raised.value.lock_path == lock_path
    assert raised.value.timeout == 0.01
    assert isinstance(raised.value.__cause__, Timeout)
    assert document_path.read_bytes() == original
    assert second.snapshot().values == {"left": 1.0, "right": 2.0}
    first_stack.close()
    with second.edit() as draft:
        draft.values["right"] = 30.0
    assert make_store(document_path).snapshot().values == {"left": 10.0, "right": 30.0}


@pytest.mark.parametrize("failure", ["body", "nested"])
def test_aborted_edit_discards_the_draft_and_allows_the_next_transaction(
    document_path: Path, failure: str
) -> None:
    store = make_store(document_path)
    original = document_path.read_bytes()
    events: list[DocumentChange] = []
    store.subscribe(events.append)

    def abort_edit() -> None:
        with store.edit() as draft:
            draft.values["left"] = 10.0
            if failure == "body":
                raise RuntimeError("body aborted")
            with store.edit() as nested:
                nested.values["right"] = 20.0

    with pytest.raises(RuntimeError, match=failure):
        abort_edit()
    assert document_path.read_bytes() == original
    assert store.snapshot().values == {"left": 1.0, "right": 2.0}
    assert events == []
    with store.edit() as draft:
        draft.values["right"] = 30.0
    assert make_store(document_path).snapshot().values == {"left": 1.0, "right": 30.0}


@pytest.mark.parametrize(
    ("failure", "error_type", "message"),
    [
        ("format", FormatError, "format"),
        ("version", VersionError, "format_version"),
        ("schema", ValidationError, "values.left"),
    ],
)
def test_invalid_draft_headers_and_schema_do_not_reach_disk_or_snapshot(
    document_path: Path, failure: str, error_type: type[ValueError], message: str
) -> None:
    store = DocumentStore(document_path, StrictDocument, format="synthetic")
    original = document_path.read_bytes()
    events: list[DocumentChange] = []
    store.subscribe(events.append)

    def invalid_edit() -> None:
        with store.edit() as draft:
            if failure == "format":
                draft.format = "wrong"
            elif failure == "version":
                draft.format_version = "2.0"
            else:
                draft.values.left = -1.0

    with pytest.raises(error_type, match=message):
        invalid_edit()
    assert document_path.read_bytes() == original
    assert store.snapshot().values.left == 1.0
    assert events == []
    with store.edit() as draft:
        draft.values.left = 10.0
    assert store.snapshot().values.left == 10.0


@pytest.mark.parametrize("original_null", [False, True])
def test_conflicts_distinguish_missing_from_null_and_keep_dot_keys_intact(
    document_path: Path, original_null: bool
) -> None:
    if original_null:
        document_path.write_text(
            document_path.read_text(encoding="utf-8") + "  Q1.t1: null\n",
            encoding="utf-8",
        )
    first = make_nullable_store(document_path)
    second = make_nullable_store(document_path)
    original_snapshot = first.snapshot()
    with ExitStack() as stack:
        draft = stack.enter_context(first.edit())
        if original_null:
            del draft.values["Q1.t1"]
        else:
            draft.values["Q1.t1"] = None
        with second.edit() as other:
            other.values["Q1.t1"] = 7.0
        committed = document_path.read_bytes()
        with pytest.raises(ConflictError) as caught:
            stack.close()

    assert caught.value.path == ("values", "Q1.t1")
    assert (caught.value.original is None) is original_null
    assert caught.value.current == 7.0
    assert document_path.read_bytes() == committed
    assert first.snapshot() == original_snapshot


def test_added_null_and_deleted_fields_roundtrip_as_distinct_changes(
    document_path: Path,
) -> None:
    store = make_nullable_store(document_path)
    events: list[DocumentChange] = []
    store.subscribe(events.append)
    with store.edit() as draft:
        draft.values["Q1.t1"] = None
        del draft.values["left"]

    assert make_nullable_store(document_path).snapshot().values == {
        "right": 2.0,
        "Q1.t1": None,
    }
    assert events == [
        DocumentChange(
            document_path, (("values", "left"), ("values", "Q1.t1")), "commit"
        )
    ]
    with store.edit() as draft:
        del draft.values["Q1.t1"]
    assert make_nullable_store(document_path).snapshot().values == {"right": 2.0}


def test_unit_resolver_uses_the_current_document_on_refresh_and_commit(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\ndimension: frequency\nvalue: 1000000\n",
        encoding="utf-8",
    )

    def resolve_units(
        document: Mapping[str, YamlValue],
    ) -> dict[tuple[str, ...], UnitSpec]:
        unit = (
            UnitSpec("Hz", "MHz")
            if document["dimension"] == "frequency"
            else UnitSpec("A", "mA")
        )
        return {("value",): unit}

    store = DocumentStore(
        document_path, DynamicUnitDocument, format="synthetic", units=resolve_units
    )
    assert store.snapshot().value == 1.0
    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\ndimension: current\nvalue: 0.001\n",
        encoding="utf-8",
    )
    assert store.refresh() is True
    assert store.snapshot().dimension == "current"
    assert store.snapshot().value == 1.0
    with store.edit() as draft:
        draft.value = 3.0
    assert store.snapshot().value == 3.0
    assert DocumentStore(
        document_path, DynamicUnitDocument, format="synthetic"
    ).snapshot().value == pytest.approx(0.003)


def test_boolean_to_equal_number_is_a_structural_change(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalue: true\n", encoding="utf-8"
    )
    store = make_scalar_store(document_path)
    events: list[DocumentChange] = []
    store.subscribe(events.append)
    with store.edit() as draft:
        draft.value = 1

    assert type(store.snapshot().value) is int
    on_disk = make_scalar_store(document_path)
    assert type(on_disk.snapshot().value) is int
    assert on_disk.snapshot().value == 1
    assert events == [DocumentChange(document_path, (("value",),), "commit")]


def test_boolean_to_equal_number_causes_a_concurrent_edit_conflict(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalue: true\n", encoding="utf-8"
    )
    first = make_scalar_store(document_path)
    second = make_scalar_store(document_path)
    with ExitStack() as stack:
        draft = stack.enter_context(first.edit())
        draft.value = 2
        with second.edit() as other:
            other.value = 1
        committed = document_path.read_bytes()
        with pytest.raises(ConflictError) as raised:
            stack.close()

    assert raised.value.path == ("value",)
    assert raised.value.original is True
    assert type(raised.value.current) is int
    assert raised.value.current == 1
    assert document_path.read_bytes() == committed
    assert first.snapshot().value is True


def test_boolean_numeric_changes_inside_a_sequence_are_not_discarded(
    document_path: Path,
) -> None:
    document_path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalues: [{enabled: true}]\n",
        encoding="utf-8",
    )
    store = DocumentStore(document_path, SequenceDocument, format="synthetic")
    with store.edit() as draft:
        draft.values[0]["enabled"] = 1

    assert type(store.snapshot().values[0]["enabled"]) is int
    on_disk = DocumentStore(document_path, SequenceDocument, format="synthetic")
    assert type(on_disk.snapshot().values[0]["enabled"]) is int
    assert on_disk.snapshot().values == [{"enabled": 1}]


def test_unchanged_nan_extension_roundtrips_without_a_false_change(
    document_path: Path,
) -> None:
    document_path.write_text(
        document_path.read_text(encoding="utf-8")
        + "ext:\n  calibration: .nan # untyped extension\n",
        encoding="utf-8",
    )
    store = DocumentStore(document_path, ExtensionDocument, format="synthetic")
    with store.edit() as draft:
        draft.values["left"] = 10.0

    assert store.snapshot().values["left"] == 10.0
    assert isnan(store.snapshot().ext["calibration"])
    assert store.refresh() is False
    assert "  calibration: .nan # untyped extension" in document_path.read_text(
        encoding="utf-8"
    )
