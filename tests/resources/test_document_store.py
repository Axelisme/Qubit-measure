from contextlib import ExitStack
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from zcu_tools.resources.document_store import (
    ConflictError,
    DocumentChange,
    DocumentStore,
    UnitSpec,
)


class SyntheticDocument(BaseModel):
    format: str
    format_version: str
    values: dict[str, float]


class KnownValues(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: float
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
