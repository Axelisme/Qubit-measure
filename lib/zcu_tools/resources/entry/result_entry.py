"""Result-owned entry identity with a separate Database output location."""

from __future__ import annotations

import errno
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from zcu_tools.cfg_model import WaveformAssetInfo
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.document_names import (
    ALIASES_FILENAME,
    LEGACY_SETUP_FILENAME,
    LIBRARY_FILENAME,
    PARAMETERS_FILENAME,
)
from zcu_tools.resources.document_store import DocumentRevision, DocumentStore
from zcu_tools.resources.parameter_metadata import validate_parameter_header
from zcu_tools.resources.record_table import JsonlRecordTable

from . import _parameter_origin
from ._path_segments import validate_path_segment
from .aliases import (
    AliasReadResult,
    EntryAliases,
    load_entry_aliases_with_revision,
    read_entry_aliases,
    save_entry_aliases,
)
from .errors import DestinationExistsError, MissingPointError
from .ledger import RecordsLedger
from .ledger_models import (
    AnalyzedPayload,
    JsonObject,
    LedgerEvent,
    LedgerEventContent,
)
from .parameter_view import ParameterView
from .registry import component_registry
from .schema import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    ParameterContent,
    ParameterDocument,
    ParameterGeneral,
)
from .storage import bind_record_table
from .views import (
    cached_parameter_content,
    parameter_origin,
    validate_view_components,
)

if TYPE_CHECKING:
    from zcu_tools.resources.context.library import ModuleLibrary


def validate_point_creation(
    mode: Literal["inherit", "empty", "clone"],
    clone_from: str | ParameterView | None,
) -> None:
    """Validate creation mode/source presence without reading or writing resources.

    inherit and empty reject clone_from; clone requires a source label or view.
    Unknown modes and invalid source combinations raise ValueError. Source
    ownership and label validation remain with ResultEntry.new_point.
    """
    if mode not in ("inherit", "empty", "clone"):
        raise ValueError(f"Unknown point creation mode: {mode!r}")
    if (mode == "clone") != (clone_from is not None):
        raise ValueError("clone mode requires clone_from; other modes forbid it")


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


def entry_component_paths(entry_root: Path) -> tuple[Path, ...]:
    """List global then sorted point YAML paths without parsing component schemas.

    entry_root is the exact Result entry directory. A parameters.yaml file identifies
    an entry, as in ResultEntry.list_names; otherwise raise ValueError. Include
    every existing points/*/parameters.yaml file, even without library.yaml.
    Missing points/ means no points. Do not read YAML or create files or locks.
    """
    global_path = entry_root / PARAMETERS_FILENAME
    if not entry_root.is_dir() or not global_path.is_file():
        raise ValueError(
            f"{entry_root}: not an entry directory; expected {PARAMETERS_FILENAME}"
        )
    return (
        global_path,
        *sorted(
            path
            for path in entry_root.glob(f"points/*/{PARAMETERS_FILENAME}")
            if path.is_file()
        ),
    )


def entry_predictor_paths(entry_root: Path) -> tuple[Path, ...]:
    """List sorted predictors/*.json in an existing Result entry without parsing.

    parameters.yaml identifies an entry, as for entry_component_paths. A missing
    predictors directory yields no paths. Reject a symlink escaping the entry
    with ValueError; do not read documents or create directories or locks.
    """
    entry_component_paths(entry_root)
    root = entry_root.resolve()
    paths = tuple(
        sorted(
            path
            for path in (entry_root / "predictors").glob("*.json")
            if path.is_file()
        )
    )
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"{path}: predictor escapes entry")
    return paths


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

    created_at is a timezone-aware UTC datetime written to global.
    Non-UTC or naive times raise ValueError.
    These values do not authorize opening, replacing or repairing any entry.
    """

    created_at: datetime

    def __post_init__(self) -> None:
        if self.created_at.utcoffset() != timedelta(0):
            raise ValueError("created_at must be a timezone-aware UTC datetime")


class ResultEntry:
    def __init__(self, result_path: Path, database_path: Path) -> None:
        self._result_path = result_path
        self._database_path = database_path
        source = result_path / PARAMETERS_FILENAME

        # Bind diagnostics to this handle without shared mutable model context.
        class EntryGlobalDocument(ParameterDocument):
            _source = source
            _upgrade_target = result_path

        try:
            self._global_store = DocumentStore[ParameterDocument](
                source,
                EntryGlobalDocument,
                format=PARAMETER_FORMAT,
                supported_version=PARAMETER_VERSION,
                validate=self._validate_global,
                read_validate=lambda document: self._validate_parameter_read(
                    document, source
                ),
            )
        except ValidationError as exc:
            raise ValueError(f"{source.resolve()}: {exc}") from exc
        self._global = ParameterView(
            self._global_store,
            source,
            result_root=result_path.parent,
            database_root=database_path.parent,
        )

    def _validate_global(self, document: ParameterDocument) -> None:
        validate_view_components(
            document, source=self._result_path / PARAMETERS_FILENAME
        )

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

    def load_aliases(self, *, point: str | None = None) -> EntryAliases:
        """Read global aliases, or one existing point's aliases when point is set.

        point=None selects <entry>/aliases.yaml. Otherwise point is a safe
        single-segment label selecting <entry>/points/<point>/aliases.yaml;
        invalid labels raise ValueError. Only aliases.yaml is read; neither
        parameters.yaml nor library.yaml needs to exist or is parsed or changed.
        A missing alias file returns an empty EntryAliases, without copying or
        falling back to global. Preserve order, empty bindings and raw $ links.
        Accept zcu.aliases 1.x; header/schema/YAML and I/O errors propagate.
        No component lookup or writes occur. Syntax validation belongs to the
        caller's alias table owner.
        """
        aliases, _ = self.load_aliases_with_revision(point=point)
        return aliases

    def load_aliases_with_revision(
        self, *, point: str | None = None
    ) -> tuple[EntryAliases, DocumentRevision]:
        """Read aliases and their exact validated byte revision together.

        Selection and failures follow load_aliases. A missing aliases file
        returns an empty table with revision=None, never a global fallback.
        """
        aliases, revision = load_entry_aliases_with_revision(self._aliases_path(point))
        return (EntryAliases(aliases={}) if aliases is None else aliases), revision

    def read_aliases(self, *, point: str | None = None) -> AliasReadResult:
        """Observe aliases once with their raw revision and deferred content error.

        Selection follows load_aliases. The result retains invalid bytes' hash
        and original error, not a valid empty fallback. Missing files are empty
        with revision=None; other filesystem errors propagate. No writes occur.
        """
        return read_entry_aliases(self._aliases_path(point))

    def save_aliases(
        self, aliases: EntryAliases, *, point: str | None = None
    ) -> DocumentRevision:
        """Atomically replace global or point aliases with the named raw table.

        aliases is an EntryAliases containing ordered scalar/list bindings, not
        resolved names. point=None selects global; a safe single-segment point
        label selects only that point's aliases.yaml. Invalid labels raise
        ValueError; missing parameters.yaml raises FileNotFoundError. library.yaml
        need not exist. Neither parameter document is parsed or changed.
        Write zcu.aliases 1.0 after strict string validation, using sibling-temp
        replacement. Schema,
        YAML and I/O errors propagate; failed replacement retains the old file.
        Other alias tables, parameters, library, ledger and preferences do not
        change. Return the exact written bytes' revision. No multiwriter merge
        or crash guarantee is offered.
        """
        destination = self._aliases_path(point)
        if point is not None:
            source = destination.parent / PARAMETERS_FILENAME
            if not source.is_file():
                raise FileNotFoundError(source)
        return save_entry_aliases(destination, aliases)

    def _aliases_path(self, point: str | None) -> Path:
        if point is None:
            return self._result_path / ALIASES_FILENAME
        directory = _entry_path(self._result_path / "points", point)
        return directory / ALIASES_FILENAME

    @property
    def ledger(self) -> RecordsLedger:
        """Bind this entry's records directory without caching events.

        Reads and appends use RecordsLedger's lock, schema and failure contracts.
        Missing records read as empty; the first append creates the directory.
        Accessing this property writes no files.
        """
        return RecordsLedger(self._result_path / "records")

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

        The analyzed event uses current UTC time and notebook
        origin. A non-None record (including {}) is attached as records/<id>.json.
        The returned ID can be read with ledger.get(id). No values are written;
        callers use ordinary point or global edits and choose any metadata.

        Payload validation and RecordsLedger I/O/validation errors propagate
        unchanged. The ledger owns locking and publication; failure may leave
        an orphan attachment or partial tail, with no rollback or recovery.
        """
        event = LedgerEvent(
            id=str(uuid4()),
            kind="analyzed",
            at=datetime.now(timezone.utc).isoformat(),
            origin="notebook",
            payload=AnalyzedPayload(
                analysis_kind=analysis_kind,
                run_ids=tuple(run_ids),
                summary=summary,
            ),
        )
        self.ledger.append(event, record=record)
        return event.id

    def records(self, component: str) -> JsonlRecordTable:
        """Bind this component's append-only record history without touching disk.

        component is a safe single path segment, not a role alias. The table
        stamps it into every appended row's component field. Rows live at
        result_path/records/components/<component>.jsonl; no component schema or ledger is
        read. Unsafe names and symlinks escaping this entry raise ValueError.
        The returned table owns append, latest-per-run read and CSV export.
        """
        return bind_record_table(self._result_path, component)

    def list_points(self) -> list[str]:
        """Return sorted labels of directories containing parameters.yaml.

        Library and aliases files are not required or inspected. Parameter
        contents are not loaded or validated here. Missing points/ returns [];
        other filesystem errors propagate.
        """
        if not (self._result_path / "points").exists():
            return []
        return sorted(
            path.name
            for path in (self._result_path / "points").iterdir()
            if path.is_dir() and (path / PARAMETERS_FILENAME).is_file()
        )

    def point_description(self, label: str) -> str:
        """Read only general.description for a safe single-segment point label.

        Missing/null descriptions return empty text. Other fields, component
        kinds and versions are not interpreted. Invalid YAML, non-mapping
        general, non-string descriptions and I/O failures propagate with the
        document path. No library or aliases are read. No locks are acquired
        and no files are changed or created.
        """
        source = _entry_path(self._result_path / "points", label) / PARAMETERS_FILENAME
        try:
            with source.open(encoding="utf-8") as stream:
                document = TypeAdapter(dict[object, object]).validate_python(
                    YAML(typ="safe").load(stream)
                )
            general = TypeAdapter(dict[object, object]).validate_python(
                document.get("general", {})
            )
            return (
                TypeAdapter(str | None).validate_python(general.get("description"))
                or ""
            )
        except (ValueError, YAMLError) as error:
            raise ValueError(f"{source}: {error}") from error

    def validate_point_parameters(self, label: str) -> None:
        """Probe one point parameter document without opening library or aliases.

        label is a safe single path segment. Require a readable YAML mapping
        with this owner's header and schema version. Component/leaf values stay
        opaque until used or edited. Parse/header/I/O failures propagate for
        callers to isolate per point. No availability guard is added to reads.
        No view is installed, and no documents are changed or created.
        """
        source = _entry_path(self._result_path / "points", label) / PARAMETERS_FILENAME
        self._point_store(source)

    def new_point(
        self,
        label: str,
        *,
        mode: Literal["inherit", "empty", "clone"] = "inherit",
        clone_from: str | ParameterView | None = None,
        description: str | None = None,
    ) -> ParameterView:
        """Create an independent complete point and return its validated view.

        label is a safe single path segment; existing destinations raise
        DestinationExistsError before writes.
        mode='inherit' (default) copies global parameters, library and aliases.
        mode='empty' creates only minimal parameters with fresh general defaults,
        no components, metadata or optional files. Neither mode accepts clone_from.
        mode='clone' requires clone_from and copies that point's three documents.
        Invalid modes or mode/source combinations raise ValueError before writes.
        Renew created_at in every independent copy.
        clone_from is a same-entry label, <entry>/<point>, or a ResultEntry-bound
        ParameterView using the same resolved result root. A global view raises
        ValueError before writes with guidance to use mode="inherit". Reload published
        point parameters and library and renew creation time. Copy aliases.yaml from
        global when seeding, or from the source point when cloning, only if present.
        Missing source aliases or library stay absent, without global fallback
        during clone or inherit. Library copies go through the library owner.
        All alias tables remain independent after creation. Point values and
        opaque metadata copy verbatim. Library Specs copy without source records,
        ledger events or attachments. Invalid disk Specs are skipped in the copy;
        source bytes stay unchanged. description=None preserves the source
        description; a string explicitly sets it, including empty text.
        Data and images are not copied or changed. Failure removes only the new
        point. No cross-file transaction is provided.
        """
        validate_point_creation(mode, clone_from)
        if isinstance(clone_from, ParameterView):
            origin = parameter_origin(clone_from)
            if origin.source.resolve().parent.parent == origin.result_root:
                raise ValueError(
                    'Cannot clone a global ParameterView; use mode="inherit" instead'
                )
        destination = _entry_path(
            self._result_path / "points", label, new_destination=True
        )
        destination.mkdir()
        try:
            if clone_from is not None:
                self._clone_point(destination, clone_from)
            elif mode == "empty":
                self._empty_point(destination, description=description)
            else:
                self._seed_point(destination)
            point = self.parameters(label)
            if description is not None:
                with point.edit() as draft:
                    draft.set("general.description", description)
            return point
        except BaseException:
            shutil.rmtree(destination)
            raise

    def _empty_point(self, destination: Path, *, description: str | None) -> None:
        policy = component_registry.require_schema_policy()
        document = {
            "format": PARAMETER_FORMAT,
            "format_version": f"{PARAMETER_VERSION.major}.{PARAMETER_VERSION.minor}",
            "schema_version": policy.current_version,
            "general": {
                "created_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "description": description,
            },
        }
        with (destination / PARAMETERS_FILENAME).open("x", encoding="utf-8") as stream:
            YAML(typ="rt").dump(document, stream)

    def initialize_point_parameters(self, label: str) -> ParameterView:
        """Replace an existing point's parameters with the current global template.

        Use after new_point(mode="empty") for caller-owned, unpublished points.
        Copy parameters only; never read or copy library or aliases. Under the
        template lock, refresh and validate global components, preserve raw YAML
        and forward-minor unknown fields, metadata, description and general.ext,
        and persist generated component defaults with a new UTC created_at.
        The source stays unchanged. Return a newly bound view; prior captures
        are not refreshed. Invalid labels, missing points, invalid global content
        or I/O errors propagate. Creation/publication rollback belongs to the
        caller: a failure after copying may leave replaced point parameters.
        """
        self.parameters(label)
        destination = _entry_path(self._result_path / "points", label)
        self._initialize_point_parameters(destination)
        return self.parameters(label)

    def _initialize_point_parameters(self, destination: Path) -> None:
        # Copy the validated YAML snapshot while the template lock is held.
        with self._global_store.locked():
            self._global_store.refresh()
            content = cached_parameter_content(self.parameters())
            components = {
                name: content.component(name) for name in content.component_names()
            }
            shutil.copyfile(
                self._result_path / PARAMETERS_FILENAME,
                destination / PARAMETERS_FILENAME,
            )
            # Missing YAML defaults may have been generated in the template's
            # native model. Copy those values too, in the same working units.
            with self._point_store(destination / PARAMETERS_FILENAME).edit() as draft:
                draft.components = components
                draft.general.created_at = (
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                )

    def _seed_point(self, destination: Path) -> None:
        self._initialize_point_parameters(destination)
        from zcu_tools.resources.context.library_store import copy_library

        copy_library(
            self._result_path / LIBRARY_FILENAME, destination / LIBRARY_FILENAME
        )
        aliases = self._result_path / ALIASES_FILENAME
        if aliases.exists():
            shutil.copyfile(aliases, destination / ALIASES_FILENAME)

    def _clone_source(
        self, clone_from: str | ParameterView
    ) -> tuple[ResultEntry, Path]:
        result_root = self._result_path.parent
        database_root = self._database_path.parent
        if isinstance(clone_from, ParameterView):
            identity = _parameter_origin.clone_source(
                clone_from,
                result_root=result_root,
            )
            source = identity.source.parent
            name = source.parent.parent.name
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
        return entry, source

    def _clone_point(
        self,
        destination: Path,
        clone_from: str | ParameterView,
    ) -> None:
        source_entry, source = self._clone_source(clone_from)
        store = source_entry._point_store(source / PARAMETERS_FILENAME)
        with store.locked():
            store.refresh()
            point_document = store.snapshot()
            shutil.copyfile(
                source / PARAMETERS_FILENAME, destination / PARAMETERS_FILENAME
            )
        aliases = source / ALIASES_FILENAME
        if aliases.exists():
            shutil.copyfile(aliases, destination / ALIASES_FILENAME)
        from zcu_tools.resources.context.library_store import copy_library

        copy_library(source / LIBRARY_FILENAME, destination / LIBRARY_FILENAME)
        cloned_store = self._point_store(destination / PARAMETERS_FILENAME)
        with cloned_store.edit() as draft:
            draft.components = point_document.components
            draft.general.created_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    def parameters(self, point: str | None = None) -> ParameterView:
        """Return the global view, or load an independently bound point view.

        None selects this entry's cached global parameters. A point label
        follows new_point's single-segment path rules. Only parameters.yaml
        must exist; library.yaml is neither required nor parsed here. Invalid
        labels raise ValueError. Missing parameters.yaml raises MissingPointError,
        a FileNotFoundError subclass; other I/O and reference errors propagate. Original
        models apply their required fields, defaults and validators. Schema
        ValidationError becomes ValueError with the full document path and original
        diagnostic. The original error remains its cause. References are resolved
        only on request. A point Store never reads global parameters; the
        global view uses its installed snapshot. No files are committed,
        other snapshots changed or active point selected.
        """
        if point is None:
            return self._global
        directory = _entry_path(self._result_path / "points", point)
        source = directory / PARAMETERS_FILENAME
        try:
            source.stat()
        except FileNotFoundError as error:
            raise MissingPointError(source) from error
        return ParameterView(
            self._point_store(source),
            source,
            result_root=self._result_path.parent,
            database_root=self._database_path.parent,
        )

    def module_library(
        self, label: str | None = None, *, assets: WaveformAssetInfo | None = None
    ) -> ModuleLibrary:
        """Open a global or point library without consulting the ledger.

        None selects entry/library.yaml; a label selects its point library
        under parameters' safe path rules. assets is optional for Spec-only use.
        A missing library at either scope is empty with revision None; the
        first nonempty edit creates it. External creation conflicts with edits
        until refresh installs that candidate.
        Invalid disk Specs are skipped and reported by snapshot.invalid_entries.
        Header/container/I/O errors propagate. Opening never writes or creates
        files, reads point/assets, or selects an active point. Drafts stay strict.
        """
        from zcu_tools.resources.context.library import ModuleLibrary

        directory = (
            self._result_path
            if label is None
            else _entry_path(self._result_path / "points", label)
        )
        return ModuleLibrary(
            directory / LIBRARY_FILENAME, assets=assets, allow_absent=True
        )

    def _validate_parameter_read(self, document: YamlMap, source: Path) -> None:
        validate_parameter_header(document, source=source)
        validate_view_components(ParameterContent(document), source=source)
        component_registry.require_schema_policy().require_current(
            document, source=source, upgrade_target=self._result_path
        )

    def _point_store(self, source: Path) -> DocumentStore[ParameterDocument]:
        class EntryPointDocument(ParameterDocument):
            _source = source
            _upgrade_target = self._result_path

        try:
            return DocumentStore[ParameterDocument](
                source,
                EntryPointDocument,
                format=PARAMETER_FORMAT,
                supported_version=PARAMETER_VERSION,
                validate=lambda document: validate_view_components(
                    document, source=source
                ),
                read_validate=lambda document: self._validate_parameter_read(
                    document, source
                ),
            )
        except ValidationError as exc:
            raise ValueError(f"{source.resolve()}: {exc}") from exc

    @classmethod
    def read_events_snapshot(
        cls, name: str, *, result_root: str | Path
    ) -> tuple[LedgerEvent, ...]:
        """Read validated historical events for a safe single-segment entry name.

        result_root is the exact Result root, not an entry or Database directory.
        Missing entry, records directory or ledger returns (). An existing entry
        must have a valid global header and UTC creation time; components are not
        interpreted. Ledger validation and I/O errors propagate. Symlinks must
        preserve root/entry/records containment and keep the ledger in records,
        otherwise raise ValueError. No entry, file, directory or lock is created,
        and attachments are not read.
        """
        return tuple(
            content.project()
            for content in cls.read_contents_snapshot(name, result_root=result_root)
        )

    @classmethod
    def read_contents_snapshot(
        cls, name: str, *, result_root: str | Path
    ) -> tuple[LedgerEventContent, ...]:
        """Read partial historical events for a safe single-segment entry name.

        result_root is the exact Result root, not an entry or Database directory.
        Missing entry/records/ledger returns (). Validate global identity, ledger
        JSON, versions, event/run IDs, duplicates and complete tails. Other fields
        validate on use or explicit project(). I/O and validation errors propagate.
        Symlink containment matches read_events_snapshot. No files, directories
        or locks are created; attachments are not read.
        """
        root = Path(result_root).absolute()
        result = _entry_path(root, name)
        for path in (root, result):
            if not path.resolve().is_relative_to(path):
                raise ValueError(f"{path}: path escapes its entry location")
        if not result.exists():
            return ()
        cls.read_identity(name, result_root=root)
        records = result / "records"
        if not records.resolve().is_relative_to(records):
            raise ValueError(f"{records}: records path escapes its entry location")
        if not records.exists():
            return ()
        ledger = records / "ledger.jsonl"
        if not ledger.resolve().is_relative_to(records):
            raise ValueError(f"{ledger}: ledger path escapes records")
        return RecordsLedger(records).read_contents_snapshot()

    @classmethod
    def read_identity(cls, name: str, *, result_root: str | Path) -> EntryIdentity:
        """Read global header and identity without component registration or locks.

        name is a safe single entry name under result_root. Validate the shared
        parameter-container header and general UTC creation time only;
        do not interpret component kinds or require point/records directories.
        Invalid names, identity/header/YAML and escaping symlinks raise ValueError.
        Missing global and other filesystem errors propagate. No files are written.
        """
        source = _entry_path(result_root, name) / PARAMETERS_FILENAME
        if not source.resolve().is_relative_to(source.parent.resolve()):
            raise ValueError(f"{source}: global escapes its entry")
        try:
            with source.open(encoding="utf-8") as stream:
                document = TypeAdapter(YamlMap).validate_python(
                    YAML(typ="safe").load(stream)
                )
            validate_parameter_header(document, source=source)
            general = ParameterGeneral.model_validate(document.get("general"))
        except (ValueError, YAMLError) as error:
            raise ValueError(f"{source}: {error}") from error
        return EntryIdentity(
            created_at=datetime.fromisoformat(general.created_at),
        )

    @classmethod
    def list_names(cls, *, result_root: str | Path) -> tuple[str, ...]:
        """List sorted direct entry directories containing parameters.yaml, without parsing.

        Missing/non-directory/inaccessible roots retain their filesystem errors.
        No legacy lookup, content validation or directory creation occurs.
        """
        return tuple(
            sorted(
                path.name
                for path in Path(result_root).iterdir()
                if path.is_dir() and (path / PARAMETERS_FILENAME).is_file()
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

        name is a safe single path segment. identity fixes the UTC creation
        time, or None uses the current time. An existing Result destination
        raises DestinationExistsError before writes. Database is neither inspected
        nor created; its configured path is database_root/name, for later Save.
        Schema/I/O failures propagate and remove only directories created by this
        call, retaining caller roots. No cross-file/power-loss guarantee is made.
        """
        from .registry import component_registry

        policy = component_registry.require_schema_policy()
        if identity is None:
            identity = EntryIdentity(created_at=datetime.now(timezone.utc))
        result_path = _entry_path(result_root, name, new_destination=True)
        database_path = Path(database_root) / name
        result_path.mkdir(parents=True)
        try:
            (result_path / "records").mkdir()
            (result_path / "points").mkdir()
            document = {
                "format": PARAMETER_FORMAT,
                "schema_version": policy.current_version,
                "format_version": f"{PARAMETER_VERSION.major}.{PARAMETER_VERSION.minor}",
                "general": {
                    "created_at": identity.created_at.isoformat().replace(
                        "+00:00", "Z"
                    ),
                },
                "components": {},
                "metadata": {},
            }
            with (result_path / PARAMETERS_FILENAME).open(
                "x", encoding="utf-8"
            ) as stream:
                YAML(typ="rt").dump(document, stream)
            _create_empty_library(result_path / LIBRARY_FILENAME)
            return cls(result_path, database_path)
        except BaseException:
            # mkdir succeeded, so this call owns the new Result directory.
            shutil.rmtree(result_path)
            raise

    @classmethod
    def open(
        cls, name: str, *, result_root: str | Path, database_root: str | Path
    ) -> ResultEntry:
        """Open a Result entry from global and bind database_root/name for Save.

        name must be a safe single path segment. The Result entry directory and
        global must exist and validate its header; field values validate on use.
        Missing points/ and records/ are empty and are not created.
        A legacy-only global filename raises ValueError naming the offline filename
        upgrade command; no legacy content is read. Other missing paths propagate
        filesystem errors. Schema ValidationError becomes
        ValueError with the full document path and original diagnostic. The original
        error remains its cause. Other schema errors propagate. Database is not inspected or created.
        """
        result_path = _entry_path(result_root, name)
        database_path = Path(database_root) / name
        for directory in (result_path,):
            if not directory.exists():
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), str(directory)
                )
            if not directory.is_dir():
                raise NotADirectoryError(
                    errno.ENOTDIR, os.strerror(errno.ENOTDIR), str(directory)
                )
        if (
            not (result_path / PARAMETERS_FILENAME).exists()
            and (result_path / LEGACY_SETUP_FILENAME).is_file()
        ):
            raise ValueError(
                f"{result_path / LEGACY_SETUP_FILENAME}: legacy filename; run "
                f"scripts/upgrade_parameters.py --scope filenames {str(result_path)!r}"
            )
        return cls(result_path, database_path)
