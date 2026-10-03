from contextlib import ExitStack
from pathlib import Path

import pytest
from pydantic import BaseModel
from zcu_tools.resources.document_store import (
    ConflictError,
    DocumentChange,
    DocumentStore,
)


class SyntheticDocument(BaseModel):
    format: str
    format_version: str
    values: dict[str, float]


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
