"""Result-owned entry identity with a separate Database output location."""

from __future__ import annotations

import errno
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import TypeAdapter, ValidationError
from ruamel.yaml import YAML

from zcu_tools.cfg_model import WaveformAssetInfo
from zcu_tools.resources.document_store import DocumentStore
from zcu_tools.resources.sample_table import JsonlSampleTable

from . import _point_origin
from ._path_segments import validate_path_segment
from .aliases import EntryAliases, load_entry_aliases, save_entry_aliases
from .errors import DestinationExistsError
from .ledger import RecordsLedger
from .ledger_models import (
    AnalyzedPayload,
    ImportPayload,
    JsonObject,
    LedgerEvent,
    Origin,
    validate_origin,
)
from .points import PointView
from .schema import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    PointDocument,
    SetupDocument,
)
from .storage import bind_sample_table
from .views import (
    SetupView,
    validate_view_components,
)

if TYPE_CHECKING:
    from zcu_tools.resources.context.library import ModuleLibrary, ModuleLibraryDocument


def _create_empty_library(path: Path) -> None:
    from zcu_tools.resources.context.library import (
        MODULE_LIBRARY_FORMAT,
        MODULE_LIBRARY_VERSION,
    )

    with path.open("x", encoding="utf-8") as stream:
        YAML(typ="rt").dump(
            {
                "format": MODULE_LIBRARY_FORMAT,
                "format_version": f"{MODULE_LIBRARY_VERSION.major}.{MODULE_LIBRARY_VERSION.minor}",
            },
            stream,
        )


def _entry_path(root: str | Path, name: str, *, new_destination: bool = False) -> Path:
    validate_path_segment(name, field="name")
    root_path = Path(root)
    path = root_path / name
    if new_destination and (path.exists() or path.is_symlink()):
        raise DestinationExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))
    if path.resolve().parent != root_path.resolve():
        raise ValueError(f"{path}: entry escapes its root")
    return path


def rename_entry(old: str, new: str, *, result_root: str | Path) -> None:
    """Rename only result_root/old to result_root/new without rewriting contents.

    Names must be safe single path segments within result_root. An existing
    destination, including a symlink, raises DestinationExistsError before moving.
    Filesystem errors propagate; Database outputs are not inspected or moved.
    Existing handles remain bound to their original paths; reopen using new.
    """
    result_old = _entry_path(result_root, old)
    result_new = _entry_path(result_root, new, new_destination=True)
    result_old.rename(result_new)


@dataclass(frozen=True, kw_only=True)
class EntryIdentity:
    """Fixed values for creating a new entry, not ownership of existing paths.

    entry_id is the UUID to write to setup. created_at is a timezone-aware UTC
    datetime; non-UTC or naive times raise ValueError, invalid IDs ValidationError.
    These values do not authorize opening, replacing or repairing any entry.
    """

    entry_id: UUID
    created_at: datetime

    def __post_init__(self) -> None:
        TypeAdapter(UUID).validate_python(self.entry_id, strict=True)
        if self.created_at.utcoffset() != timedelta(0):
            raise ValueError("created_at must be a timezone-aware UTC datetime")


class ResultEntry:
    def __init__(self, result_path: Path, database_path: Path) -> None:
        self._result_path = result_path
        self._database_path = database_path
        self._entry_id: str | None = None
        source = result_path / "setup.yaml"

        # Bind diagnostics to this handle without shared mutable model context.
        class EntrySetupDocument(SetupDocument):
            _source = source

        try:
            self._setup_store = DocumentStore[SetupDocument](
                source,
                EntrySetupDocument,
                format=PARAMETER_FORMAT,
                supported_version=PARAMETER_VERSION,
                validate=self._validate_setup,
            )
        except ValidationError as exc:
            raise ValueError(f"{source.resolve()}: {exc}") from exc

    def _validate_setup(self, document: SetupDocument) -> None:
        validate_view_components(document, source=self._result_path / "setup.yaml")
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

    @property
    def result_path(self) -> Path:
        """Return this entry's exact Result directory, not the parent root."""
        return self._result_path

    @property
    def predictors_path(self) -> Path:
        """Return this entry\'s Result predictors directory without creating it.

        The read-only binding does not list, load or validate predictor files.
        """
        return self._result_path / "predictors"

    @property
    def database_path(self) -> Path:
        """Return this entry's exact Database directory, not the parent root."""
        return self._database_path

    @property
    def dispersive_path(self) -> Path:
        """Return the entry's dispersive directory without creating it.

        File names and content belong to resources.dispersive_file. A symlink
        escaping this entry raises ValueError; no component or ledger is read.
        """
        path = self._result_path / "dispersive"
        if not path.resolve().is_relative_to(self._result_path.resolve()):
            raise ValueError(f"{path}: dispersive directory escapes its entry")
        return path

    @property
    def t1_curve_path(self) -> Path:
        """Return the entry's T1 curve directory without creating it.

        File names and content belong to resources.t1_curve_file. A symlink
        escaping this entry raises ValueError; no component or ledger is read.
        """
        path = self._result_path / "t1_curve"
        if not path.resolve().is_relative_to(self._result_path.resolve()):
            raise ValueError(f"{path}: t1_curve directory escapes its entry")
        return path

    def load_aliases(self, *, point: str | None = None) -> EntryAliases:
        """Read setup aliases, or one existing point's aliases when point is set.

        point=None selects <entry>/aliases.yaml. Otherwise point is a safe
        single-segment label selecting <entry>/points/<point>/aliases.yaml;
        invalid labels raise ValueError, and missing point.yaml or module_cfg.yaml
        raises FileNotFoundError. Neither parameter document is parsed or changed.
        A missing alias file returns an empty EntryAliases, without copying or
        falling back to setup. Preserve order, empty bindings and raw $ links.
        Accept zcu.aliases 1.x; header/schema/YAML and I/O errors propagate.
        No component lookup or writes occur. Syntax validation belongs to the
        caller's alias table owner.
        """
        aliases = load_entry_aliases(self._aliases_path(point))
        return EntryAliases(aliases={}) if aliases is None else aliases

    def save_aliases(self, aliases: EntryAliases, *, point: str | None = None) -> None:
        """Atomically replace setup or point aliases with the named raw table.

        aliases is an EntryAliases containing ordered string bindings, not
        resolved names. point=None selects setup; a safe single-segment point
        label selects only that point's aliases.yaml. Invalid labels raise
        ValueError; missing point.yaml or module_cfg.yaml raises FileNotFoundError.
        Neither parameter document is parsed or changed. Write zcu.aliases 1.0
        after strict string validation, using sibling-temp replacement. Schema,
        YAML and I/O errors propagate; failed replacement retains the old file.
        Other alias tables, parameters, library, ledger and preferences do not
        change. No multiwriter merge or crash guarantee is offered.
        """
        save_entry_aliases(self._aliases_path(point), aliases)

    def _aliases_path(self, point: str | None) -> Path:
        if point is None:
            return self._result_path / "aliases.yaml"
        directory = _entry_path(self._result_path / "points", point)
        for name in ("point.yaml", "module_cfg.yaml"):
            source = directory / name
            if not source.is_file():
                raise FileNotFoundError(source)
        return directory / "aliases.yaml"

    @property
    def ledger(self) -> RecordsLedger:
        """Bind this entry's records directory and UUID without caching events.

        Reads and appends use RecordsLedger's lock, schema and failure contracts.
        Missing records directories raise; accessing this property writes no files.
        """
        return RecordsLedger(self._result_path / "records", entry_id=self.entry_id)

    def record_analysis(
        self,
        analysis_kind: str,
        *,
        run_ids: Sequence[str],
        summary: JsonObject,
        record: JsonObject | None = None,
    ) -> str:
        """Append cross-run analysis evidence and return its new UUID string.

        analysis_kind is the caller's method name. run_ids contains input run
        identifiers in caller order, may be empty, and is not resolved against
        run data. summary and optional record are strict JSON objects with
        finite numbers; their contents have no framework-defined schema.

        The analyzed event uses this entry's UUID, current UTC time and notebook
        origin. A non-None record (including {}) is attached as records/<id>.json.
        The returned ID can be read with ledger.get(id). No values are written;
        callers use ordinary point or setup edits and choose any metadata.

        Payload validation and RecordsLedger I/O/validation errors propagate
        unchanged. The ledger owns locking and publication; failure may leave
        an orphan attachment or partial tail, with no rollback or recovery.
        """
        event = LedgerEvent(
            id=str(uuid4()),
            kind="analyzed",
            at=datetime.now(timezone.utc).isoformat(),
            entry_id=self.entry_id,
            origin="notebook",
            payload=AnalyzedPayload(
                analysis_kind=analysis_kind,
                run_ids=tuple(run_ids),
                summary=summary,
            ),
        )
        self.ledger.append(event, record=record)
        return event.id

    def samples(self, component: str) -> JsonlSampleTable:
        """Bind this component's append-only sample history without touching disk.

        component is a safe single path segment, not a role alias. Rows live at
        result_path/samples/<component>.jsonl; no component schema or ledger is
        read. Unsafe names and symlinks escaping this entry raise ValueError.
        The returned table owns append, latest-per-run read and CSV export.
        """
        return bind_sample_table(self._result_path, component)

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
    ) -> PointView:
        """Create an independent complete point and return its validated view.

        label is a safe single path segment; existing destinations raise
        DestinationExistsError before writes.
        Copy the complete published source YAML: setup.yaml without clone_from,
        otherwise the source point.yaml. Renew created_at in the independent copy.
        clone_from is a same-entry label, <entry>/<point>, or a ResultEntry-bound
        PointView using the same resolved result/database roots. Reload published
        point and module_cfg and renew creation time. Copy aliases.yaml from
        setup when seeding, or from the source point when cloning, only if present.
        Missing source aliases stay absent, without setup fallback during clone.
        All alias tables remain independent after creation. Point values and
        opaque metadata copy verbatim. Library sources keep the library contract:
        cross-entry nonmanual library sources become local import events with
        copied JSON attachments, and library sources mark the direct clone
        origin. origin identifies the importing frontend as gui, notebook or mcp.
        description=None preserves the source description; a string explicitly sets it,
        including empty text, within this creation/cleanup scope.
        Reading the source library may first back up its original bytes and
        clean invalid entries and their sources from its module_cfg. Clone copies
        only retained entries and imports only retained sources. Data and images
        are not copied or changed; other source files are unchanged.
        Failure removes only the new point. Completed source-library cleanup and
        its backup remain, and completed imports may remain orphaned. No cross-file
        transaction is provided.
        """
        validate_origin(origin)
        destination = _entry_path(
            self._result_path / "points", label, new_destination=True
        )
        destination.mkdir()
        try:
            if clone_from is not None:
                self._clone_point(destination, clone_from, origin=origin)
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
            shutil.copyfile(
                self._result_path / "setup.yaml", destination / "point.yaml"
            )
            # Missing YAML defaults may have been generated in the template's
            # native model. Copy those values too, in the same working units.
            with self._point_store(destination / "point.yaml").edit() as draft:
                draft.components = components
                draft.general.created_at = (
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                )
        _create_empty_library(destination / "module_cfg.yaml")
        aliases = self._result_path / "aliases.yaml"
        if aliases.exists():
            shutil.copyfile(aliases, destination / "aliases.yaml")

    def _clone_source(self, clone_from: str | PointView) -> tuple[ResultEntry, Path]:
        result_root = self._result_path.parent
        database_root = self._database_path.parent
        expected_id = None
        if isinstance(clone_from, PointView):
            identity = _point_origin.clone_source(
                clone_from,
                result_root=result_root,
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
    ) -> None:
        source_entry, source = self._clone_source(clone_from)
        store = source_entry._point_store(source / "point.yaml")
        with store.locked():
            store.refresh()
            point_document = store.snapshot()
            shutil.copyfile(source / "point.yaml", destination / "point.yaml")
        aliases = source / "aliases.yaml"
        if aliases.exists():
            shutil.copyfile(aliases, destination / "aliases.yaml")
        library_store = source_entry._library_store(source / "module_cfg.yaml")
        with library_store.locked():
            library_store.refresh()
            library_document = library_store.snapshot()
            shutil.copyfile(source / "module_cfg.yaml", destination / "module_cfg.yaml")
        event_ids = tuple(
            dict.fromkeys(
                metadata.source
                for metadata in library_document.provenance.values()
                if metadata.source != "manual"
            )
        )
        imports: dict[str, str] = {}
        if source_entry is not self:
            imports = self._import_sources(event_ids, source_entry, origin=origin)
        cloned_store = self._point_store(destination / "point.yaml")
        with cloned_store.edit() as draft:
            draft.components = point_document.components
            draft.general.created_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
        # This private copy has already passed source-owner cleanup. Its foreign
        # IDs must be rewritten before destination-owner source validation.
        from zcu_tools.resources.context.library import (
            MODULE_LIBRARY_FORMAT,
            MODULE_LIBRARY_VERSION,
            ModuleLibraryDocument,
        )

        cloned_library = DocumentStore(
            destination / "module_cfg.yaml",
            ModuleLibraryDocument,
            format=MODULE_LIBRARY_FORMAT,
            supported_version=MODULE_LIBRARY_VERSION,
        )
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
        from zcu_tools.resources.context.library_store import open_library_store

        return open_library_store(
            source,
            ledger=self._result_path / "records/ledger.jsonl",
            entry_id=self.entry_id,
        )

    def _import_sources(
        self,
        event_ids: tuple[str, ...],
        source_entry: ResultEntry,
        *,
        origin: Origin,
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
        models apply their required fields, defaults and validators. Schema
        ValidationError becomes ValueError with the full document path and original
        diagnostic. The original error remains its cause. References are resolved
        only on request. The view binds one Store and never reads setup.
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
            entry_id=self.entry_id,
            result_root=self._result_path.parent,
            database_root=self._database_path.parent,
        )

    def module_library(
        self, label: str | None = None, *, assets: WaveformAssetInfo | None = None
    ) -> ModuleLibrary:
        """Open the setup or point library with this entry's ledger and optional assets.

        None selects entry/module_cfg.yaml, initialized by create. A string
        selects a point library and follows use_point's safe single-segment rules.
        Only module_cfg.yaml
        is loaded; the point and assets are not read and no active point is selected.
        Omit assets for Spec-only reads/edits; materializing Arb requires a provider.
        The existing library must have a valid header and container. Invalid
        existing Specs or entry-local provenance are backed up and removed under
        the document lock; new edits remain strict. Invalid labels raise ValueError;
        missing files, container errors and ledger I/O/schema failures propagate.
        Opening creates no missing document or parent directory.
        """
        # Avoid the library -> entry.provenance -> ResultEntry import cycle.
        from zcu_tools.resources.context.library import ModuleLibrary

        directory = (
            self._result_path
            if label is None
            else _entry_path(self._result_path / "points", label)
        )
        return ModuleLibrary(
            directory / "module_cfg.yaml",
            assets=assets,
            ledger=self._result_path / "records/ledger.jsonl",
            entry_id=self.entry_id,
        )

    def _point_store(self, source: Path) -> DocumentStore[PointDocument]:
        class EntryPointDocument(PointDocument):
            _source = source

        try:
            return DocumentStore[PointDocument](
                source,
                EntryPointDocument,
                format=PARAMETER_FORMAT,
                supported_version=PARAMETER_VERSION,
                validate=lambda document: validate_view_components(
                    document, source=source
                ),
            )
        except ValidationError as exc:
            raise ValueError(f"{source.resolve()}: {exc}") from exc

    @classmethod
    def read_identity(cls, name: str, *, result_root: str | Path) -> EntryIdentity:
        """Read setup header and identity without component registration or locks.

        name is a safe single entry name under result_root. Validate the shared
        parameter-container header and general UUID/UTC creation time only;
        do not interpret component kinds or require point/records directories.
        Invalid names, identity/header/YAML and escaping symlinks raise ValueError.
        Missing setup and other filesystem errors propagate. No files are written.
        """
        entry = cls(_entry_path(result_root, name), Path("."))
        general = entry._setup_store.snapshot().general
        return EntryIdentity(
            entry_id=UUID(general.entry_id),
            created_at=datetime.fromisoformat(general.created_at),
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
        cls,
        name: str,
        *,
        result_root: str | Path,
        database_root: str | Path,
        identity: EntryIdentity | None = None,
    ) -> ResultEntry:
        """Create a new Result entry and bind a Database output location.

        name is a safe single path segment. identity fixes the setup UUID and UTC
        creation time, or None generates both now. An existing Result destination
        raises DestinationExistsError before writes. Database is neither inspected
        nor created; its configured path is database_root/name, for later Save.
        Schema/I/O failures propagate and remove only directories created by this
        call, retaining caller roots. No cross-file/power-loss guarantee is made.
        """
        if identity is None:
            identity = EntryIdentity(
                entry_id=uuid4(), created_at=datetime.now(timezone.utc)
            )
        result_path = _entry_path(result_root, name, new_destination=True)
        database_path = Path(database_root) / name
        result_path.mkdir(parents=True)
        try:
            (result_path / "records").mkdir()
            (result_path / "points").mkdir()
            document = {
                "format": PARAMETER_FORMAT,
                "format_version": f"{PARAMETER_VERSION.major}.{PARAMETER_VERSION.minor}",
                "general": {
                    "entry_id": str(identity.entry_id),
                    "created_at": identity.created_at.isoformat().replace(
                        "+00:00", "Z"
                    ),
                },
                "components": {},
                "metadata": {},
            }
            with (result_path / "setup.yaml").open("x", encoding="utf-8") as stream:
                YAML(typ="rt").dump(document, stream)
            _create_empty_library(result_path / "module_cfg.yaml")
            return cls(result_path, database_path)
        except BaseException:
            # mkdir succeeded, so this call owns the new Result directory.
            shutil.rmtree(result_path)
            raise

    @classmethod
    def open(
        cls, name: str, *, result_root: str | Path, database_root: str | Path
    ) -> ResultEntry:
        """Open a complete Result entry and bind database_root/name for Save.

        name must be a safe single path segment. The Result entry, points and
        records directories must exist; setup must validate its format and UUID.
        Missing paths propagate filesystem errors. Schema ValidationError becomes
        ValueError with the full document path and original diagnostic. The original
        error remains its cause. Other schema errors propagate. Database is not inspected or created.
        """
        result_path = _entry_path(result_root, name)
        database_path = Path(database_root) / name
        for directory in (
            result_path,
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
