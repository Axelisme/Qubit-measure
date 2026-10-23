"""Installed raw-byte versions through the public document transaction seam."""

from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError
from zcu_tools.resources.document_store import DocumentStore

from tests.resources._document_store_fakes import SyntheticDocument


def make_store(path: Path) -> DocumentStore[SyntheticDocument]:
    return DocumentStore(path, SyntheticDocument, format="synthetic")


def test_revision_tracks_only_successfully_installed_bytes(document_path: Path) -> None:
    store = make_store(document_path)
    assert hasattr(store, "document_revision"), "Stores must expose installed revisions"
    initial = store.document_revision
    assert initial.path == document_path.resolve()
    assert initial.revision == sha256(document_path.read_bytes()).hexdigest()
    assert store.source_path == initial.path

    # Even a comment-only external edit is a new byte revision, not a new value.
    document_path.write_bytes(document_path.read_bytes() + b"\n# external\n")
    assert store.document_revision == initial
    store.refresh()
    refreshed = store.document_revision
    assert refreshed.revision != initial.revision
    assert refreshed.revision == sha256(document_path.read_bytes()).hexdigest()

    with store.edit() as draft:
        draft.values["left"] = 99.0
    committed = store.document_revision
    assert committed.revision != refreshed.revision
    assert make_store(document_path).snapshot().values["left"] == 99.0
    assert committed.revision == sha256(document_path.read_bytes()).hexdigest()

    # Reading a valid external version at edit entry must not publish on abort.
    valid = document_path.read_bytes() + b"# another revision\n"
    document_path.write_bytes(valid)
    with pytest.raises(RuntimeError, match="abort"), store.edit():
        raise RuntimeError("abort")
    assert store.document_revision == committed
    with store.edit():
        pass
    assert store.document_revision.revision == sha256(valid).hexdigest()
    installed = store.document_revision
    document_path.write_text("not a document", encoding="utf-8")
    with pytest.raises(ValidationError):
        store.refresh()
    assert store.document_revision == installed
    assert store.snapshot().values["left"] == 99.0
