"""Creation and identity of a result entry across two explicit roots."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ruamel.yaml import YAML

from zcu_tools.resources.document_store import DocumentStore

from .schema import SetupDocument


class ResultEntry:
    def __init__(self, result_path: Path, database_path: Path) -> None:
        self._result_path = result_path
        self._database_path = database_path
        self._setup_store = DocumentStore(
            result_path / "setup.yaml",
            SetupDocument,
            format="zcu.parameter-container",
            lock_path=result_path / ".entry.lock",
        )

    @property
    def entry_id(self) -> str:
        return self._setup_store.snapshot().general.entry_id

    @classmethod
    def create(
        cls, name: str, *, result_root: str | Path, database_root: str | Path
    ) -> "ResultEntry":
        result_path = Path(result_root) / name
        database_path = Path(database_root) / name
        result_path.mkdir(parents=True)
        database_path.mkdir(parents=True)
        (result_path / "records").mkdir()
        (result_path / "points").mkdir()
        document = {
            "format": "zcu.parameter-container",
            "format_version": "1.0",
            "general": {
                "entry_id": str(uuid4()),
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            },
            "components": {},
            "provenance": {},
        }
        with (result_path / "setup.yaml").open("x", encoding="utf-8") as stream:
            YAML(typ="rt").dump(document, stream)
        return cls(result_path, database_path)

    @classmethod
    def open(  # noqa: ARG003 -- Frozen declaration; next contract cycle implements open.
        cls, name: str, *, result_root: str | Path, database_root: str | Path
    ) -> "ResultEntry":
        raise NotImplementedError("Opening an existing entry is not implemented")
