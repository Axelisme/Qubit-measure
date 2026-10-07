"""Creation and identity of a result entry across two explicit roots."""

from __future__ import annotations

import errno
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ruamel.yaml import YAML

from zcu_tools.cfg_model import WaveformAssetInfo
from zcu_tools.format_version import YamlValue
from zcu_tools.resources.document_store import DocumentStore

from . import _point_origin
from ._path_segments import validate_path_segment
from .errors import RenameRecoveryError
from .ledger import RecordsLedger
from .ledger_models import ImportPayload, LedgerEvent, Origin, validate_origin
from .points import PointView
from .provenance import Provenance, ProvenanceData, validate_source
from .schema import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    PointDocument,
    SetupDocument,
    is_forward_minor,
)
from .views import SetupView, prepare_point_update

if TYPE_CHECKING:
    from zcu_tools.resources.context.library import ModuleLibrary, ModuleLibraryDocument


def _entry_path(root: str | Path, name: str, *, new_destination: bool = False) -> Path:
    validate_path_segment(name, field="name")
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

    @property
    def result_path(self) -> Path:
        """Return this entry's exact Result directory, not the parent root."""
        return self._result_path

    @property
    def database_path(self) -> Path:
        """Return this entry's exact Database directory, not the parent root."""
        return self._database_path

    @property
    def ledger(self) -> RecordsLedger:
        """Bind this entry's records directory and UUID without caching events.

        Reads and appends use RecordsLedger's lock, schema and failure contracts.
        Missing records directories raise; accessing this property writes no files.
        """
        return RecordsLedger(self._result_path / "records", entry_id=self.entry_id)

    def updated_point(
        self,
        document: PointDocument,
        path: str,
        value: YamlValue,
        *,
        source: Provenance | None = None,
    ) -> PointDocument:
        """Return an isolated working-unit point candidate for a memory command.

        path is a concrete component/general dotted field, without role expansion.
        Reuse the view's field/schema validation. Omitted source is manual;
        explicit Provenance is validated against this entry's actual ledger and
        clears clone origin on accepted leaves. Preserve unmodified fields and
        created_at. Input/schema/source failures leave inputs unchanged.
        No point/store/file creation, notification or event append. Manual does
        not read a ledger; explicit event sources may read local ledger evidence.
        """
        return prepare_point_update(
            document,
            path,
            value,
            source=source,
            ledger=self._result_path / "records/ledger.jsonl",
            entry_id=self.entry_id,
        )

    def validate_library(self, document: ModuleLibraryDocument) -> None:
        """Validate a memory library's keys/Specs/sources against this entry.

        Delegate the existing library document validation without mutation,
        materialization, asset access, YAML I/O or event production. Manual
        sources do not read ledger evidence. Invalid keys/schema/sources raise;
        explicit local event sources retain ledger/attachment/I/O failures.
        """
        document.validate_content(
            ledger=self._result_path / "records/ledger.jsonl", entry_id=self.entry_id
        )

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
        self,
        label: str,
        *,
        clone_from: str | PointView | None = None,
        description: str | None = None,
        origin: Origin = "notebook",
        call_id: str | None = None,
    ) -> PointView:
        """Create an independent complete point and return its validated view.

        label is a safe single path segment; existing destinations are rejected.
        Without clone_from, copy the latest setup components and provenance.
        clone_from is a same-entry label, <entry>/<point>, or a ResultEntry-bound
        PointView using the same resolved result/database roots. Reload published
        point and module_cfg, renew creation time and mark the direct clone origin.
        Cross-entry nonmanual sources become local import events with copied JSON
        attachments; same-entry source IDs remain unchanged. origin/call_id name
        the import operation: MCP needs a nonempty ID, other origins require None.
        description=None preserves seed/clone metadata; a string explicitly sets it,
        including empty text, within this creation/cleanup scope.
        No source files, data or images change. Failure removes only the new point;
        completed imports may remain orphaned. No cross-file transaction is provided.
        """
        validate_origin(origin, call_id)
        destination = _entry_path(
            self._result_path / "points", label, new_destination=True
        )
        destination.mkdir()
        try:
            if clone_from is not None:
                self._clone_point(
                    destination, clone_from, origin=origin, call_id=call_id
                )
            else:
                self._seed_point(destination)
            point = self.use_point(label)
            if description is not None:
                with point.edit() as draft:
                    draft.set("general.description", description)
            return point
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

    def _clone_source(self, clone_from: str | PointView) -> tuple[ResultEntry, Path]:
        result_root = self._result_path.parent
        database_root = self._database_path.parent
        expected_id = None
        if isinstance(clone_from, PointView):
            identity = _point_origin.clone_source(
                clone_from,
                result_root=result_root,
                database_root=database_root,
            )
            source = identity.source.parent
            name = source.parent.parent.name
            expected_id = identity.entry_id
        else:
            segments = clone_from.split("/")
            if len(segments) == 1:
                return self, _entry_path(self._result_path / "points", clone_from)
            if len(segments) != 2:
                raise ValueError("clone_from must be label or <entry>/<point>")
            name, label = segments
            source = _entry_path(_entry_path(result_root, name) / "points", label)
        if source.parent.parent.resolve() == self._result_path.resolve():
            entry = self
        else:
            entry = ResultEntry.open(
                name, result_root=result_root, database_root=database_root
            )
        if expected_id is not None and entry.entry_id != expected_id:
            raise ValueError("PointView source entry_id has changed")
        return entry, source

    def _clone_point(
        self,
        destination: Path,
        clone_from: str | PointView,
        *,
        origin: Origin,
        call_id: str | None,
    ) -> None:
        source_entry, source = self._clone_source(clone_from)
        store = source_entry._point_store(source / "point.yaml")
        with store.locked():
            store.refresh()
            point_document = store.snapshot()
            shutil.copyfile(source / "point.yaml", destination / "point.yaml")
        library_store = self._library_store(source / "module_cfg.yaml")
        with library_store.locked():
            library_store.refresh()
            library_document = library_store.snapshot()
            library_document.validate_content(
                ledger=source_entry._result_path / "records/ledger.jsonl",
                entry_id=source_entry.entry_id,
            )
            shutil.copyfile(source / "module_cfg.yaml", destination / "module_cfg.yaml")
        forward_minor = is_forward_minor(
            {
                "format": point_document.format,
                "format_version": point_document.format_version,
            },
            source=source / "point.yaml",
        )
        # Validate known source fields without narrowing the point's future-field policy.
        point_sources = {
            path: ProvenanceData.model_validate(
                metadata, extra="ignore" if forward_minor else None
            )
            for path, metadata in point_document.provenance.items()
        }
        for metadata in point_sources.values():
            validate_source(
                metadata.to_provenance(),
                source_entry._result_path / "records/ledger.jsonl",
                source_entry.entry_id,
            )
        event_ids = tuple(
            dict.fromkeys(
                metadata.source
                for metadata in (
                    *point_sources.values(),
                    *library_document.provenance.values(),
                )
                if metadata.source != "manual"
            )
        )
        imports: dict[str, str] = {}
        if source_entry is not self:
            imports = self._import_sources(
                event_ids, source_entry, origin=origin, call_id=call_id
            )
        cloned_store = self._point_store(destination / "point.yaml")
        with cloned_store.edit() as draft:
            draft.components = point_document.components
            draft.general.created_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            for metadata in draft.provenance.values():
                event_id = metadata.get("source")
                if isinstance(event_id, str) and event_id in imports:
                    metadata["source"] = imports[event_id]
                metadata["cloned_from"] = {
                    "entry_id": source_entry.entry_id,
                    "point": source.name,
                }
        # New destination is private until both independent file commits succeed.
        cloned_library = self._library_store(destination / "module_cfg.yaml")
        with cloned_library.edit() as draft:
            for metadata in draft.provenance.values():
                if metadata.source in imports:
                    metadata.source = imports[metadata.source]
                metadata.cloned_from = {
                    "entry_id": source_entry.entry_id,
                    "point": source.name,
                }
            draft.validate_content(
                ledger=self._result_path / "records/ledger.jsonl",
                entry_id=self.entry_id,
            )
        cloned_library.snapshot().validate_content(
            ledger=self._result_path / "records/ledger.jsonl", entry_id=self.entry_id
        )

    def _library_store(self, source: Path) -> DocumentStore[ModuleLibraryDocument]:
        # Library imports entry provenance; defer composition until both packages load.
        from zcu_tools.resources.context.library import (
            MODULE_LIBRARY_FORMAT,
            MODULE_LIBRARY_VERSION,
            ModuleLibraryDocument,
        )

        return DocumentStore(
            source,
            ModuleLibraryDocument,
            format=MODULE_LIBRARY_FORMAT,
            supported_version=MODULE_LIBRARY_VERSION,
        )

    def _import_sources(
        self,
        event_ids: tuple[str, ...],
        source_entry: ResultEntry,
        *,
        origin: Origin,
        call_id: str | None,
    ) -> dict[str, str]:
        imports: dict[str, str] = {}
        source_ledger = source_entry.ledger
        destination_ledger = self.ledger
        for event_id in event_ids:
            try:
                evidence = source_ledger.get(event_id)
            except KeyError as exc:
                raise ValueError(
                    f"source ledger event {event_id!r} does not exist"
                ) from exc
            imported = LedgerEvent(
                id=str(uuid4()),
                kind="import",
                at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                entry_id=self.entry_id,
                origin=origin,
                call_id=call_id,
                payload=ImportPayload(
                    source_entry_id=source_entry.entry_id,
                    source_event_id=event_id,
                    source_event_json=evidence.event_json,
                    source_record=evidence.event.record,
                ),
            )
            destination_ledger.append(imported, record=evidence.record)
            imports[event_id] = imported.id
        return imports

    def use_point(self, label: str) -> PointView:
        """Load one complete point and return an independently bound view.

        label follows new_point's single-segment path rules. Both point.yaml and
        module_cfg.yaml must exist; the latter is not parsed here. Invalid labels
        raise ValueError and missing files raise FileNotFoundError. Original
        models apply their required fields, defaults and validators. References
        are resolved only on request. The view binds one Store and never reads setup.
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
            result_root=self._result_path.parent,
            database_root=self._database_path.parent,
        )

    def module_library(
        self, label: str, *, assets: WaveformAssetInfo | None = None
    ) -> ModuleLibrary:
        """Open a point-local library with this entry's ledger and optional assets.

        label follows use_point's safe single-segment rules. Only module_cfg.yaml
        is loaded; the point and assets are not read and no active point is selected.
        Omit assets for Spec-only reads/edits; materializing Arb requires a provider.
        The existing library must have a valid header, schema and local provenance.
        Invalid labels raise ValueError; missing files and validation/I/O errors
        propagate. Opening creates no missing document.
        """
        # Avoid the library -> entry.provenance -> ResultEntry import cycle.
        from zcu_tools.resources.context.library import ModuleLibrary

        directory = _entry_path(self._result_path / "points", label)
        return ModuleLibrary(
            directory / "module_cfg.yaml",
            assets=assets,
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
        )

    @classmethod
    def list_names(cls, *, result_root: str | Path) -> tuple[str, ...]:
        """List sorted direct entry directories containing setup.yaml, without parsing.

        Missing/non-directory/inaccessible roots retain their filesystem errors.
        No legacy lookup, content validation or directory creation occurs.
        """
        return tuple(
            sorted(
                path.name
                for path in Path(result_root).iterdir()
                if path.is_dir() and (path / "setup.yaml").is_file()
            )
        )

    @classmethod
    def create(
        cls, name: str, *, result_root: str | Path, database_root: str | Path
    ) -> ResultEntry:
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
    ) -> ResultEntry:
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
