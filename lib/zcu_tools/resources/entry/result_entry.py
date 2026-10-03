"""Creation and identity of a result entry across two explicit roots."""

import errno
import os
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from ruamel.yaml import YAML

from zcu_tools.format_version import YamlValue
from zcu_tools.resources.document_store import DocumentStore, FieldPath, UnitSpec

from .errors import PartialCommitError
from .registry import component_registry
from .schema import SetupDocument
from .views import SetupView


def _entry_path(root: str | Path, name: str, *, new_destination: bool = False) -> Path:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or Path(name).is_absolute()
        or PureWindowsPath(name).anchor
    ):
        raise ValueError(f"{name!r}: expected a single path component")
    root_path = Path(root)
    path = root_path / name
    if new_destination and (path.exists() or path.is_symlink()):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))
    if path.resolve().parent != root_path.resolve():
        raise ValueError(f"{path}: entry escapes its root")
    return path


def rename_entry(
    old: str, new: str, *, result_root: str | Path, database_root: str | Path
) -> None:
    result_old = _entry_path(result_root, old)
    database_old = _entry_path(database_root, old)
    result_new = _entry_path(result_root, new, new_destination=True)
    database_new = _entry_path(database_root, new, new_destination=True)
    ResultEntry.open(old, result_root=result_root, database_root=database_root)
    result_old.rename(result_new)
    try:
        database_old.rename(database_new)
    except OSError as cause:
        try:
            if result_old.exists() or result_old.is_symlink():
                raise FileExistsError(
                    errno.EEXIST, os.strerror(errno.EEXIST), str(result_old)
                )
            result_new.rename(result_old)
        except OSError as recovery_cause:
            raise PartialCommitError(
                completed=(result_new,),
                pending=(database_new,),
                recovery_failed=(result_old,),
                cause=cause,
                recovery_cause=recovery_cause,
            ) from cause
        raise


class ResultEntry:
    def __init__(self, result_path: Path, database_path: Path) -> None:
        self._result_path = result_path
        self._database_path = database_path
        self._entry_id: str | None = None
        self._setup_store = DocumentStore(
            result_path / "setup.yaml",
            SetupDocument,
            format="zcu.parameter-container",
            units=self._setup_units,
            validate=self._validate_setup,
            lock_path=result_path / ".entry.lock",
        )

    def _setup_units(
        self, document: Mapping[str, YamlValue]
    ) -> Mapping[FieldPath, UnitSpec]:
        result: dict[FieldPath, UnitSpec] = {}
        components = document.get("components")
        if isinstance(components, dict):
            for name, fields in components.items():
                if isinstance(fields, dict) and isinstance(
                    kind := fields.get("kind"), str
                ):
                    for path, spec in component_registry.units(
                        kind, source=self._result_path / "setup.yaml", component=name
                    ).items():
                        result[("components", name, *path)] = spec
        return result

    def _validate_setup(self, document: SetupDocument) -> None:
        if self._entry_id is None:
            self._entry_id = document.general.entry_id
        elif document.general.entry_id != self._entry_id:
            raise ValueError(
                f"{self._result_path / 'setup.yaml'}: entry_id is immutable; "
                f"expected {self._entry_id!r}, got {document.general.entry_id!r}"
            )

    @property
    def setup(self) -> SetupView:
        return SetupView(self._setup_store, self._result_path / "setup.yaml")

    @property
    def entry_id(self) -> str:
        return self._setup_store.snapshot().general.entry_id

    @classmethod
    def create(
        cls, name: str, *, result_root: str | Path, database_root: str | Path
    ) -> "ResultEntry":
        result_path = _entry_path(result_root, name, new_destination=True)
        database_path = _entry_path(database_root, name, new_destination=True)
        created: list[Path] = []
        try:
            for destination in (result_path, database_path):
                destination.mkdir(parents=True)
                created.append(destination)
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
        except BaseException:
            # These entry directories belong to this call, not to the caller's roots.
            for directory in reversed(created):
                shutil.rmtree(directory)
            raise

    @classmethod
    def open(
        cls, name: str, *, result_root: str | Path, database_root: str | Path
    ) -> "ResultEntry":
        result_path = _entry_path(result_root, name)
        database_path = _entry_path(database_root, name)
        for directory in (
            result_path,
            database_path,
            result_path / "points",
            result_path / "records",
        ):
            if not directory.exists():
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), str(directory)
                )
            if not directory.is_dir():
                raise NotADirectoryError(
                    errno.ENOTDIR, os.strerror(errno.ENOTDIR), str(directory)
                )
        return cls(result_path, database_path)
