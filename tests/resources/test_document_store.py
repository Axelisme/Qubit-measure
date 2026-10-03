from contextlib import ExitStack
from pathlib import Path

import pytest
from pydantic import BaseModel
from zcu_tools.resources.document_store import ConflictError, DocumentStore


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
