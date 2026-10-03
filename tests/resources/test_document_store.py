from pathlib import Path

from pydantic import BaseModel
from zcu_tools.resources.document_store import DocumentStore


class SyntheticDocument(BaseModel):
    format: str
    format_version: str
    values: dict[str, float]


def test_snapshot_is_an_independent_memory_only_typed_copy(tmp_path: Path) -> None:
    path = tmp_path / "document.yaml"
    path.write_text(
        "format: synthetic\nformat_version: '1.0'\nvalues:\n  left: 1.0\n  right: 2.0\n",
        encoding="utf-8",
    )
    store = DocumentStore(path, SyntheticDocument, format="synthetic")
    path.unlink()

    snapshot = store.snapshot()
    assert snapshot.values == {"left": 1.0, "right": 2.0}
    snapshot.values["left"] = 99.0
    assert store.snapshot().values["left"] == 1.0
