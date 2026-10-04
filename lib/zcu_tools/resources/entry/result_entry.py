"""Creation and identity of a result entry across two explicit roots."""

import errno
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from ruamel.yaml import YAML

from zcu_tools.resources.document_store import DocumentStore

from . import _point_origin
from .errors import RenameRecoveryError
from .points import PointView
from .registry import component_registry
from .schema import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    PointDocument,
    SetupDocument,
)
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
            raise RenameRecoveryError(
                moved_result=result_new,
                pending_database=database_new,
                recovery_destination=result_old,
                cause=cause,
                recovery_cause=recovery_cause,
            ) from cause
        raise


class ResultEntry:
    def __init__(self, result_path: Path, database_path: Path) -> None:
        self._result_path = result_path
        self._database_path = database_path
        self._entry_id: str | None = None
        source = result_path / "setup.yaml"

        # Bind diagnostics to this handle without shared mutable model context.
        class EntrySetupDocument(SetupDocument):
            _source = source

        self._setup_store = DocumentStore[SetupDocument](
            source,
            EntrySetupDocument,
            format=PARAMETER_FORMAT,
            supported_version=PARAMETER_VERSION,
            validate=self._validate_setup,
        )

    def _validate_setup(self, document: SetupDocument) -> None:
        component_registry.validate_references(
            document.components, source=self._result_path / "setup.yaml"
        )
        if self._entry_id is None:
            self._entry_id = document.general.entry_id
        elif document.general.entry_id != self._entry_id:
            raise ValueError(
                f"{self._result_path / 'setup.yaml'}: entry_id is immutable; "
                f"expected {self._entry_id!r}, got {document.general.entry_id!r}"
            )

    @property
    def setup(self) -> SetupView:
        return SetupView(
            self._setup_store,
            self._result_path / "setup.yaml",
            ledger=self._result_path / "records/ledger.jsonl",
            entry_id=self.entry_id,
        )

    @property
    def entry_id(self) -> str:
        return self._setup_store.snapshot().general.entry_id

    def list_points(self) -> list[str]:
        """Return sorted labels of directories containing both point files.

        Inspect points/ on disk for point.yaml and module_cfg.yaml; incomplete
        directories are omitted. File contents are not loaded or validated here.
        Missing or unreadable points/ raises the underlying filesystem error.
        """
        return sorted(
            path.name
            for path in (self._result_path / "points").iterdir()
            if path.is_dir()
            and (path / "point.yaml").is_file()
            and (path / "module_cfg.yaml").is_file()
        )

    def new_point(
        self, label: str, *, clone_from: str | PointView | None = None
    ) -> PointView:
        """Create an independent complete point and return its validated view.

        label is a safe single path segment; existing destinations are rejected.
        Without clone_from, copy the latest setup components and provenance.
        Otherwise copy only the same-entry source point and module_cfg, renew its
        creation time and mark sources with that direct clone origin. References
        resolve only inside the new point. Neither path changes its source.
        Creation failure removes only the new destination. No cross-point
        transaction, active point selection or data/image copy is provided.
        """
        destination = _entry_path(
            self._result_path / "points", label, new_destination=True
        )
        destination.mkdir()
        try:
            if clone_from is not None:
                self._clone_point(destination, clone_from)
            else:
                self._seed_point(destination)
            return self.use_point(label)
        except BaseException:
            shutil.rmtree(destination)
            raise

    def _seed_point(self, destination: Path) -> None:
        # Copy the validated YAML snapshot while the template lock is held.
        with self._setup_store.locked():
            self._setup_store.refresh()
            components = self._setup_store.snapshot().components
            yaml = YAML(typ="rt")
            with (self._result_path / "setup.yaml").open(encoding="utf-8") as stream:
                template = yaml.load(stream)
            document = {
                "format": template["format"],
                "format_version": template["format_version"],
                "general": {
                    "created_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                },
                "components": deepcopy(template.get("components", {})),
                "provenance": deepcopy(template.get("provenance", {})),
            }
            with (destination / "point.yaml").open("x", encoding="utf-8") as stream:
                yaml.dump(document, stream)
            # Missing YAML defaults may have been generated in the template's
            # native model. Copy those values too, in the same working units.
            with self._point_store(destination / "point.yaml").edit() as draft:
                draft.components = components
        with (destination / "module_cfg.yaml").open("x", encoding="utf-8") as stream:
            YAML(typ="rt").dump(
                {"format": "zcu.module-library", "format_version": "1.0"}, stream
            )

    def _clone_point(self, destination: Path, clone_from: str | PointView) -> None:
        source = (
            _point_origin.clone_source(clone_from, self._result_path)
            if isinstance(clone_from, PointView)
            else _entry_path(self._result_path / "points", clone_from)
        )
        store = self._point_store(source / "point.yaml")
        with store.locked():
            store.refresh()
            components = store.snapshot().components
            for filename in ("point.yaml", "module_cfg.yaml"):
                shutil.copyfile(source / filename, destination / filename)
        cloned_store = self._point_store(destination / "point.yaml")
        with cloned_store.edit() as draft:
            draft.components = components
            draft.general.created_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            for metadata in draft.provenance.values():
                metadata["cloned_from"] = {
                    "entry_id": self.entry_id,
                    "point": source.name,
                }

    def use_point(self, label: str) -> PointView:
        """Load one complete point and return an independently bound view.

        label follows new_point's single-segment path rules. Both point.yaml and
        module_cfg.yaml must exist; the latter is not parsed here. Invalid labels
        raise ValueError and missing files raise FileNotFoundError. Original
        models validate required fields, field validators, canonical values and
        same-document references. The view binds one Store and never reads setup.
        No files are committed, other snapshots changed or active point selected.
        """
        directory = _entry_path(self._result_path / "points", label)
        source = directory / "point.yaml"
        module = directory / "module_cfg.yaml"
        if not module.is_file():
            raise FileNotFoundError(module)
        return PointView(
            self._point_store(source),
            source,
            ledger=self._result_path / "records/ledger.jsonl",
            entry_id=self.entry_id,
        )

    def _point_store(self, source: Path) -> DocumentStore[PointDocument]:
        class EntryPointDocument(PointDocument):
            _source = source

        return DocumentStore[PointDocument](
            source,
            EntryPointDocument,
            format=PARAMETER_FORMAT,
            supported_version=PARAMETER_VERSION,
            validate=lambda document: component_registry.validate_references(
                document.components, source=source
            ),
        )

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
