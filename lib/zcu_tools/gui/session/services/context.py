# tolerated(file-size): Entry/point installation, cached captures and inactive reloads
# share the same global/active document sets and publication transaction.
"""Owner-thread entry/point bindings and fixed session parameter reads."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from ruamel.yaml.error import YAMLError

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.owner import OwnerThreadGuard
from zcu_tools.gui.session.aliases import AliasRow, AliasSnapshot, AliasTable
from zcu_tools.gui.session.context_types import (
    ContextSource,
    ContextSourceBinding,
    DocumentSetRef,
    EntryInfo,
    FileRevision,
    FileSyncStatus,
    LibraryChange,
    PointBindingError,
    WorkingPointInfo,
)
from zcu_tools.gui.session.events import (
    AliasesChangedPayload,
    ContextSwitchedPayload,
    EntryDocumentsChangedPayload,
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.gui.session.parameters import DocumentSet
from zcu_tools.gui.session.services.arb_waveform import ARB_WAVEFORMS_VERSION_KEY
from zcu_tools.gui.session.services.context_writeback import ContextWritebackTarget
from zcu_tools.gui.session.services.file_sync import FileSync
from zcu_tools.gui.session.types import (
    ContextReadiness,
    DocumentSetCapture,
    RoleBinding,
)
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ScalarValue,
    ValueInfo,
    ValueLookup,
    ValueRef,
    resolve_value_ref,
)
from zcu_tools.resources.context.library import ModuleLibrary
from zcu_tools.resources.entry import (
    DestinationExistsError,
    ResultEntry,
    document_revision,
    parameter_content,
    snapshot,
    source_path,
    validate_path_segment,
)
from zcu_tools.resources.entry.aliases import AliasFile, EntryAliases
from zcu_tools.resources.entry.ledger_models import JsonObject
from zcu_tools.resources.entry.schema import (
    ParameterDocument,
)
from zcu_tools.resources.entry.storage import StorageBinding, capture_storage
from zcu_tools.resources.entry.writeback import (
    ArbAssetWriteback,
    LibraryUpdate,
    ModuleWriteback,
    ParameterUpdate,
    WaveformWriteback,
    WritebackEvidence,
    WritebackItem,
    WritebackResult,
    apply_writeback,
    commit_library_updates,
    commit_parameter_updates,
    publish_arb_asset_updates,
)

if TYPE_CHECKING:
    from zcu_tools.cfg_model import ModuleSpec, WaveformAssetInfo, WaveformSpec
    from zcu_tools.format_version import YamlMap, YamlValue
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_types import (
        PointCreateRequest,
    )
    from zcu_tools.gui.session.state import SessionState
    from zcu_tools.gui.session.types import SessionEnv
    from zcu_tools.resources.entry import (
        ParameterView,
    )
    from zcu_tools.resources.entry.ledger import RecordsLedger


class ContextService:
    """Own entry/point installation, fixed parameter captures, and global aliases.

    Commands and captures run on the creating owner thread. Resource preparation
    precedes publication, so failure never replaces a valid binding. The service
    keeps writable resource handles private; consumers get isolated snapshots or
    the explicit entry_ledger capability. Construction does not open an entry,
    read hardware, or bind assets for EMPTY.
    """

    def __init__(
        self,
        state: SessionState,
        bus: BaseEventBus,
        values: ValueLookup | None = None,
        *,
        bind_assets: Callable[[Path], WaveformAssetInfo],
    ) -> None:
        """Use owner State/bus, optional value facade, and explicit asset binding.

        bind_assets receives entry.database_path / 'arb_waveforms' on entry/point
        installation. Its capability stays with the library capture, not a
        global Database selection. Binding errors abort publication. values
        defaults to State's facade and does not change parameter content/version.
        Entry installation loads global saved rows; clearing resets the table.
        Selecting or creating a point replaces rows with that point's saved
        table. Missing aliases means empty, without cross-scope or preset
        fallback. No component publication fills aliases automatically.
        """
        self._owner = OwnerThreadGuard()
        self._state = state
        self._bus = bus
        self._bind_assets = bind_assets
        self._values = values if values is not None else state.session_env.values
        self._entry: ResultEntry | None = None
        self._entry_info: EntryInfo | None = None
        self._global_set: DocumentSet | None = None
        self._active_set: DocumentSet | None = None
        self._points: tuple[WorkingPointInfo, ...] = ()
        self._source_binding: ContextSourceBinding | None = None
        self._binding_id = 0
        self._file_sync = FileSync(
            bus, installed=self.get_installed_file_revisions, install=self._reload_file
        )
        if state.session_env.values is not self._values:
            state.set_context(replace(state.session_env, values=self._values))

    def install_entry(self, entry: ResultEntry) -> EntryInfo:
        """Install an opened entry's global document set without selecting a point.

        Prepare cached global content, global library/aliases, entry storage,
        point summaries and assets before publication. Each bad point becomes
        an unavailable row. Alias syntax/content faults remain lazy in captures;
        other preparation failures retain the old binding and any created entry.
        Success publishes GLOBAL with active_label=None, bumps context once and
        emits one switch fact after every resource and source binding is installed.
        SoC, device and predictor capabilities remain with their own owners.
        """
        self._owner.assert_owner()
        global_parameters = entry.parameters()
        points = self._load_points(entry)
        assets = self._bind_assets(entry.database_path / "arb_waveforms")
        library = entry.library(assets=assets)
        documents = DocumentSet(global_parameters, library, entry.aliases())
        parameters = DocumentSetCapture.from_set(
            documents,
            target=DocumentSetRef(entry_name=entry.result_path.name),
            storage=capture_storage(global_parameters),
        )
        info = EntryInfo(
            entry.result_path.name,
            entry.result_path,
            entry.database_path,
        )
        sources = (
            ContextSource("global", source_path(global_parameters), None),
            ContextSource("module_cfg", library.source_path, None),
        )
        self._entry = entry
        self._entry_info = info
        self._global_set = documents
        self._active_set = documents
        self._points = points
        self._publish(parameters, sources, ContextReadiness.GLOBAL, active_label=None)
        return info

    def get_source_binding(self) -> ContextSourceBinding | None:
        """Return detached cached persistent sources, or None without an opened entry.

        GLOBAL binds global parameters and library. POINT retains global parameters and binds
        the selected point and its library. All sources change in one publication.

        This internal source ID is for facts/refresh, never part of read targets.
        Query performs no I/O and does not select a point.
        """
        return deepcopy(self._source_binding)

    def get_file_revisions(self) -> tuple[FileRevision, ...]:
        """Return the active three file versions, or () without an opened entry.

        Paths are absolute. Aliases use the observed bytes hash even when their
        content failed validation; None means absent only. No disk reads or
        implicit refresh occur. Failed writes and point preparation retain the
        previous installed versions.
        """
        self._owner.assert_owner()
        return () if self._active_set is None else self._active_set.revisions()

    def get_installed_file_revisions(self) -> tuple[FileRevision, ...]:
        """Return cached active versions, then entry versions when a point is active.

        Each set is ordered parameters/library/aliases. Paths are absolute;
        missing library/aliases revisions are None. No entry returns (). This
        query does no I/O or refresh, including while external edits are pending.
        Watcher and Inspect use these versions; entry.info uses active-only
        get_file_revisions instead.
        """
        self._owner.assert_owner()
        active = self.get_file_revisions()
        if self._active_set is self._global_set:
            return active
        if self._global_set is None:
            raise RuntimeError("Installed global documents disappeared")
        return active + self._global_set.revisions()

    def get_file_sync_status(self) -> tuple[FileSyncStatus, ...]:
        """Return installed-file observations without I/O; switches clear them."""
        return self._file_sync.statuses()

    def check_file_revision(self, expected: FileRevision) -> None:
        """Compare a draft with disk; see FileSync.check_revision for failures."""
        self._file_sync.check_revision(expected)

    def reload_file(self, path: Path, expected_revision: str | None) -> FileRevision:
        """Strictly install one owner file; see FileSync.reload for failures.

        Accept any absolute installed path, including inactive entry files.
        Active content facts follow installation; inactive entry reloads publish
        only file status and preserve the active content version and selection.
        Old captures remain fixed. Caller owns holder admission. No repair, backup or document writes occur.
        """
        return self._file_sync.reload(path, expected_revision)

    def synchronize_files(self, *, admit_reload: bool) -> None:
        """Scan installed files with host execution-time admission; no repair or writes."""
        self._file_sync.synchronize(admit_reload=admit_reload)

    def _reload_file(self, file: FileRevision) -> FileRevision:
        if (
            self._active_set is not self._global_set
            and file not in self.get_file_revisions()
        ):
            return self._reload_entry_file(file)
        parameters = self._published_parameters()
        target = parameters.target
        documents = self._active_set
        if documents is None or target is None:
            raise RuntimeError("Active document resources disappeared")
        if file.document == "aliases":
            alias_file = self._require_open_entry().aliases(target.point)
            saved = alias_file.read().require_valid()
            candidate = AliasTable(
                tuple(AliasRow(name, value) for name, value in saved.aliases.items())
            )
            self._install_alias_rows(candidate, alias_file)
        else:
            if file.document == "parameters":
                documents.parameters.refresh(strict=True)
            else:
                documents.library.refresh(strict=True)
            self._publish_write(
                parameters.storage, parameter=file.document == "parameters"
            )
        return next(
            item for item in self.get_file_revisions() if item.path == file.path
        )

    def _reload_entry_file(self, file: FileRevision) -> FileRevision:
        """Strictly install an inactive entry candidate without active content facts."""
        documents = self._global_set
        if documents is None:
            raise RuntimeError("Installed entry resources disappeared")
        if file.document == "parameters":
            documents.parameters.refresh(strict=True)
        elif file.document == "library":
            documents.library.refresh(strict=True)
        else:
            alias_file = self._require_open_entry().aliases()
            saved = alias_file.read().require_valid()
            table = AliasTable(
                tuple(AliasRow(name, value) for name, value in saved.aliases.items())
            )
            documents.alias_file = alias_file
            documents.aliases = AliasSnapshot(table, document_revision(alias_file))
        return next(
            item
            for item in self.get_installed_file_revisions()
            if item.path == file.path
        )

    def get_predictors_path(self) -> Path | None:
        """Return the opened entry predictor directory, or None without an entry.

        Read the ResultEntry-owned path without I/O, directory creation or point selection.
        """
        return None if self._entry is None else self._entry.predictors_path

    def get_entry_info(self) -> EntryInfo | None:
        """Return cached entry metadata, or None when no entry is opened.

        An opened entry is readable in both GLOBAL and POINT.
        This query performs no I/O and never refreshes or selects a point.
        """
        return deepcopy(self._entry_info)

    def current_point_label(self) -> str | None:
        """Return the selected saved point label, or None without selection; no I/O."""
        parameters = self._state.session_env.parameters
        target = None if parameters is None else parameters.target
        return None if target is None else target.point

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        """Return cached descriptions/parameter availability, empty without an entry.

        refresh=True prepares a fresh list before replacing cache. Point document
        failures become unavailable rows with errors; discovery failures propagate,
        preserving the previous cache. Library and aliases are not inspected.
        """
        if refresh:
            self._owner.assert_owner()
            if self._entry is not None:
                self._points = self._load_points(self._entry)
        return deepcopy(self._points)

    def use_point(self, point: str | None = None) -> DocumentSetRef:
        """Select global documents for None/omitted, or the named saved point.

        Prepare parameters/library/assets/aliases before publishing the new
        focus. Any saved label may be selected, not just the current focus.
        Global selection opens a fresh entry handle and refreshes the point list.
        Missing aliases give an empty table; no global/preset fallback is used.
        Alias content/syntax errors are captured with the observed bytes hash
        for lazy consumers, leaving concrete parameter/library reads available.

        No opened entry raises FailedPreconditionError(no_entry). Missing point,
        schema, source and I/O failures retain the previous binding, versions,
        revisions and captures. The returned target has no selection generation.
        """
        self._owner.assert_owner()
        if point is None:
            fresh = self._fresh_entry()
            self.install_entry(fresh)
            return DocumentSetRef(entry_name=fresh.result_path.name)
        entry = self._require_open_entry()
        view = entry.parameters(point)
        parameters, documents, sources, info = self._prepare_point(entry, point, view)
        self._install_point(documents, parameters, sources, info)
        return DocumentSetRef(entry_name=entry.result_path.name, point=point)

    def new_point(self, request: PointCreateRequest) -> DocumentSetRef:
        """Create with the explicit inherit/empty/clone mode, then install POINT.

        No opened entry raises FailedPreconditionError(no_entry). Pass description
        unchanged to ResultEntry.new_point. An unsafe
        label raises InvalidInputError(point_label_invalid) before source reads or
        writes, preserving binding, files, ledger and versions. Label safety follows
        the entry owner's single path component rule. An existing destination
        raises InvalidInputError with the entry owner's FileExistsError message,
        preserving the same state. Other resource errors propagate. Post-creation
        binding failure raises PointBindingError from its cause, retaining the
        point and old binding without retry/deletion.
        """
        self._owner.assert_owner()
        entry = self._require_open_entry()
        try:
            validate_path_segment(request.label, field="label")
        except ValueError as exc:
            raise InvalidInputError(
                str(exc), reason_code="point_label_invalid"
            ) from exc
        clone = (
            entry.parameters(request.clone_from)
            if request.clone_from is not None
            else None
        )
        try:
            point = entry.new_point(
                request.label,
                mode=request.mode,
                clone_from=clone,
                description=request.description,
            )
        except DestinationExistsError as exc:
            raise InvalidInputError(str(exc)) from exc
        try:
            parameters, documents, sources, info = self._prepare_point(
                entry, request.label, point
            )
        except Exception as exc:
            # Resource creation succeeded; report the retained point as recovery.
            raise PointBindingError(
                entry_name=entry.result_path.name, point=request.label
            ) from exc
        self._install_point(documents, parameters, sources, info)
        return DocumentSetRef(entry_name=entry.result_path.name, point=request.label)

    def clear_context(self) -> None:
        """Publish EMPTY, clear entry/point, retaining non-parameter handles.

        Publish a switch fact after installation; do not delete resources.
        """
        self._owner.assert_owner()
        self._entry = None
        self._entry_info = None
        self._global_set = None
        self._active_set = None
        self._points = ()
        self._publish(None, (), ContextReadiness.EMPTY, active_label=None)

    def inspect_document(self, *, point: str | None = None) -> ParameterDocument:
        """Return isolated installed global or focused point content without I/O.

        None selects global. A label must equal the focused point, otherwise
        raise InvalidInputError(point_not_focused). No entry raises
        FailedPreconditionError. Changing returned data does not write.
        """
        self._owner.assert_owner()
        return self.read_context(point=point).whole_document()

    def add_component(
        self, name: str, kind: str, fields: YamlMap, *, point: str | None = None
    ) -> None:
        """Add one component with working-unit fields using its registered kind.

        None edits global. A label must equal the focused point, otherwise
        raise InvalidInputError(point_not_focused) before validation or I/O.
        Invalid names/kinds/required fields/duplicates leave content unchanged.
        Success on the current point installs its source before publishing one
        parameter change. A nonactive entry edit installs cache/revision before
        an EntryDocumentsChangedPayload fact, without changing active content or
        existing captures.
        No entry/context, schema, conflict and I/O errors propagate.
        """
        self._owner.assert_owner()
        if self._entry is None:
            raise FailedPreconditionError("No entry is opened", reason_code="no_entry")
        documents = self._documents_at(point)
        if documents is None:
            raise FailedPreconditionError("No entry is opened", reason_code="no_entry")
        if documents is not self._active_set:
            documents.parameters.add_component(name, kind=kind, **fields)
            self._bus.emit(EntryDocumentsChangedPayload())
            return
        parameters = self._published_parameters()
        documents.parameters.add_component(name, kind=kind, **fields)
        self._publish_write(
            parameters.storage,
            parameter=True,
        )

    def read_context(
        self, *, aliases: Mapping[str, str] | None = None, point: str | None = None
    ) -> DocumentSetCapture:
        """Capture installed documents/storage and all alias rows without I/O.

        point=None captures global documents. A label must equal the focused
        point, otherwise InvalidInputError(point_not_focused) is raised before
        reading sources. Captures use installed handles, never disk candidates,
        and do not change selection or bypass pending.

        aliases maps temporary component/device names to authored rows for this
        capture only, retaining dependent links. Names need not exist in catalogs.
        Overrides do not change global rows, versions or selection; invalid names
        raise InvalidInputError(alias_invalid). $alias returns a name string and
        $alias.path resolves lazily without kind checks or concrete-name fallback.
        Missing/unbound aliases raise RoleResolutionError by name. Bare paths are
        independent. Reads do not perform I/O or follow later source/alias edits.
        Alias-file faults remain fixed and affect only alias reads/projections.
        EMPTY raises FailedPreconditionError(no_context). Necessary source faults
        raise ContextSourceUnavailableError, never returning old values.
        """
        self._owner.assert_owner()
        parameters = self._state.session_env.parameters
        documents = self._documents_at(point)
        saved_aliases = (
            AliasSnapshot(AliasTable()) if documents is None else documents.aliases
        )
        if point is None:
            if self._entry is None or documents is None:
                raise FailedPreconditionError(
                    "No entry is opened", reason_code="no_context"
                )
            parameters = DocumentSetCapture.from_set(
                documents,
                target=DocumentSetRef(entry_name=self._entry.result_path.name),
                storage=capture_storage(documents.parameters),
            )
        if parameters is None:
            raise FailedPreconditionError(
                "Select or create a working point first", reason_code="no_context"
            )
        table = saved_aliases.table
        for alias, component in (aliases or {}).items():
            table = table.with_override(alias, component)
        return parameters.bind_aliases(table.rows, {}, alias_error=saved_aliases.error)

    def get_alias_rows(self, *, point: str | None = None) -> tuple[AliasRow, ...]:
        """Read installed authored rows without expansion or I/O.

        None selects global. A label must equal the focused point, otherwise
        raise InvalidInputError(point_not_focused). Pending candidates are not read.
        No entry returns empty rows. Invalid aliases raise their captured error.
        Missing files supply empty rows without fallback.
        """
        self._owner.assert_owner()
        documents = self._documents_at(point)
        return () if documents is None else documents.aliases.rows

    def _install_alias_rows(self, candidate: AliasTable, alias_file: AliasFile) -> None:
        documents = self._active_set
        if documents is None:
            raise RuntimeError("Active document set disappeared")
        previous = documents.aliases.table.rows
        had_error = documents.aliases.error is not None
        documents.alias_file = alias_file
        documents.aliases = AliasSnapshot(candidate, document_revision(alias_file))
        if candidate.rows != previous or had_error:
            self._state.version.bump("context")
            self._bus.emit(AliasesChangedPayload(rows=candidate.rows))

    def apply_alias_rows(
        self, rows: tuple[AliasRow, ...], *, point: str | None = None
    ) -> None:
        """Validate and atomically save rows to the chosen aliases.yaml.

        Runs on the owner thread. Without an opened entry raise
        FailedPreconditionError(no_context) before validation or I/O. None writes
        global aliases. A label must equal the focused point,
        otherwise raise InvalidInputError(point_not_focused) before validation
        or I/O. Global writes do not change an active point or existing captures.
        Validate syntax/order, save, then install the exact revision and publish
        changed rows. InvalidInputError(alias_invalid) and storage errors retain
        rows, versions and the old file. Identical active rows save without an
        active content event. Nonactive entry commits publish
        EntryDocumentsChangedPayload after cache installation. An installed
        missing file created externally raises revision_mismatch rather than
        overwriting it. No ledger or other document is written.
        """
        self._owner.assert_owner()
        if self._entry is None:
            raise FailedPreconditionError(
                "Open an entry first", reason_code="no_context"
            )
        documents = self._documents_at(point)
        candidate = AliasTable(rows)
        if documents is None:
            raise RuntimeError("Installed documents disappeared")
        installed = documents.aliases
        if installed.revision is not None and installed.revision.revision is None:
            self.check_file_revision(
                FileRevision("aliases", installed.revision.path, None)
            )
        alias_file = documents.alias_file
        revision = alias_file.save(
            EntryAliases(aliases={row.alias: row.value for row in candidate.rows})
        )
        if documents is not self._active_set:
            documents.aliases = AliasSnapshot(candidate, revision)
            self._bus.emit(EntryDocumentsChangedPayload())
        else:
            self._install_alias_rows(candidate, alias_file)

    def capture_storage(self, parameters: DocumentSetCapture) -> StorageBinding:
        """Return historical storage for a fixed capture on the owner sequence.

        Current entry and point are not read. Source faults propagate;
        unbound captures raise FailedPreconditionError(storage_unavailable).
        No refresh, file write, output reservation or ledger append occurs.
        """
        self._owner.assert_owner()
        return parameters.capture_storage()

    def read_target(
        self, target: DocumentSetRef, *, roles: RoleBinding | None = None
    ) -> DocumentSetCapture:
        """Read current values at a fixed target without changing selection.

        target.point=None reads global parameters/library; a label reads that
        point's independent views in the currently opened entry.
        Supplied role names stay fixed. Missing required parameter documents,
        cross-entry, schema, source and I/O failures propagate. This read publishes
        no versions or events. It opens only parameters/library, never aliases,
        and does not follow live alias rows.
        """
        self._owner.assert_owner()
        entry = self._require_write_target(target)
        assets = self._bind_assets(entry.database_path / "arb_waveforms")
        library = entry.library(target.point, assets=assets)
        parameters = (
            self._fresh_entry().parameters()
            if target.point is None
            else entry.parameters(target.point)
        )
        return DocumentSetCapture(
            point=parameter_content(parameters),
            library=library.snapshot(),
            target=target,
            roles=roles,
            storage=capture_storage(parameters),
        )

    def read_record(
        self, target: DocumentSetRef, component: str, run_id: str
    ) -> JsonObject | None:
        """Read one run's latest entry JSONL record without changing selection.

        target validates the opened entry identity, rejecting cross-entry targets;
        no point selection or point-file read is required. Record history itself
        belongs to the entry, independent of point. The
        record owner returns a detached complete JSON row or None if absent.
        Invalid component/run_id, corrupt history and I/O errors propagate. No
        dataframe projection, context publication or refresh is performed here.
        """
        self._owner.assert_owner()
        return self._require_write_target(target).records(component).latest(run_id)

    def apply_writeback(
        self,
        target: DocumentSetRef,
        items: tuple[WritebackItem, ...],
        *,
        evidence: WritebackEvidence,
    ) -> WritebackResult:
        """Accept nonempty ordered concrete writes at a fixed entry or point target.

        Validate opened entry membership and the captured parameter resource
        before append. Resource load errors propagate. The executor appends
        accepted, then commits assets, parameters, library and individual records,
        preserving each group's order. Assets are shared within the target entry.
        point=None writes global files; a label writes that point's files even
        when another scope is selected. Parameter/library setters require an opened context,
        not a point selection. Record validation belongs to its resource owner.
        Each committed document publishes once; detached writes never select it.
        Return the accepted event ID and actual committed values. Item failures
        raise WritebackApplyError with committed indices and original cause.
        An asset I/O error may leave a confirmed prefix; a preflight conflict
        or failed parameter/library group writes nothing in its group.
        Earlier groups and accepted event remain. Ledger append errors propagate
        unchanged. Cross-entry targets fail before append.
        No point creation, retry or cross-file rollback occurs.
        """
        self._owner.assert_owner()
        if not items:
            raise ValueError("writeback items must be nonempty")
        entry = self._require_write_target(target)
        if target.point is None:
            self._fresh_entry()
        else:
            entry.parameters(target.point)
        return apply_writeback(
            items,
            target=ContextWritebackTarget(self, target, entry),
            evidence=evidence,
        )

    def publish_arb_assets(
        self, target: DocumentSetRef, items: tuple[ArbAssetWriteback, ...]
    ) -> Iterator[YamlMap]:
        """Commit distinct asset keys to the target's opened entry on the owner thread.

        Iterate on the owner thread to receive each status and summary in
        order. Bump the shared asset version once when there is a confirmed
        prefix, even if later asset I/O fails. A preflight failure yields no
        receipts and does not bump. Target, validation and I/O errors propagate.
        """
        self._owner.assert_owner()
        entry = self._require_write_target(target)
        confirmed = False
        try:
            for receipt in publish_arb_asset_updates(entry.database_path, items):
                confirmed = True
                yield receipt
        finally:
            if confirmed:
                self._state.version.bump(ARB_WAVEFORMS_VERSION_KEY)

    def set_parameter(
        self,
        target: DocumentSetRef,
        path: str,
        value: YamlValue,
        *,
        metadata: YamlMap | None = None,
    ) -> YamlValue:
        """Write one concrete path and return its detached actual working-unit value.

        Use the captured target, not the current selection. point=None writes
        global parameters; a label writes one point file in the opened entry.
        A nonselected target stays unselected.
        metadata=None keeps the path's opaque metadata; a mapping replaces it
        in the same commit ({} clears). No source, time or event is added.
        Active-target success publishes its content fact after resource commit.
        A nonactive global write updates installed cache/revision and emits
        EntryDocumentsChangedPayload without changing active content or captures.
        Cross-entry, missing entry, path/schema/metadata
        representation, conflict and I/O failures propagate without publishing
        a candidate. No retry or batch receipt.
        """
        return self.set_parameters(target, (ParameterUpdate(path, value, metadata),))[0]

    def set_parameters(
        self, target: DocumentSetRef, updates: tuple[ParameterUpdate, ...]
    ) -> tuple[YamlValue, ...]:
        """Commit nonempty ordered point updates in one edit and publish once.

        ParameterUpdate defines path/value/metadata semantics. Target, selection
        and failure rules match set_parameter. Return detached values captured
        after each validated draft update, only when the entire group commits.
        An invalid update leaves all group values and metadata unwritten and
        publishes nothing. Later overwrites do not change earlier receipts.
        """
        self._owner.assert_owner()
        if not updates:
            raise ValueError("parameter updates must be nonempty")
        entry = self._require_write_target(target)
        parameters = self._published_parameters()
        if target.point is None:
            if self._global_set is None:
                raise RuntimeError("Installed entry parameters disappeared")
            point = self._global_set.parameters
        elif target == parameters.target and self._active_set is not None:
            point = self._active_set.parameters
        else:
            point = entry.parameters(target.point)
        actual = commit_parameter_updates(point, updates)
        document = snapshot(point)
        self._points = tuple(
            replace(info, description=document.general.description or "")
            if info.label == target.point
            else info
            for info in self._points
        )
        if target != parameters.target:
            if target.point is None:
                self._bus.emit(EntryDocumentsChangedPayload())
            return actual
        self._publish_write(
            replace(parameters.storage, description=document.general.description)
            if parameters.storage is not None
            else None,
            parameter=True,
        )
        return actual

    def set_module(
        self,
        target: DocumentSetRef,
        path: str,
        value: ModuleSpec,
    ) -> YamlMap:
        """Write a module and return its detached actual cached Spec mapping.

        Target/publication/error rules match set_parameter.
        Use one library transaction without source records.
        No lowering, materialization, accepted event production or batch receipt.
        """
        return self.set_modules(target, (LibraryUpdate(ModuleWriteback(path, value)),))[
            0
        ]

    def set_waveform(
        self,
        target: DocumentSetRef,
        path: str,
        value: WaveformSpec,
    ) -> YamlMap:
        """Write a waveform and return its detached actual cached Spec mapping.

        Target/publication/error rules match set_module. Owner rejects
        invalid roots/families before installing any candidate.
        """
        return self.set_modules(
            target, (LibraryUpdate(WaveformWriteback(path, value)),)
        )[0]

    def set_modules(
        self, target: DocumentSetRef, updates: tuple[LibraryUpdate, ...]
    ) -> tuple[YamlMap, ...]:
        """Commit a nonempty mixed module/waveform group and publish once.

        LibraryUpdate defines each key, family and Spec. Target and
        failure rules match set_module. Each update validates in one draft;
        any failure leaves the entire group unwritten and publishes nothing.
        Return each detached validated value only after the group commits,
        preserving update order and values before subsequent overwrites.
        """
        self._owner.assert_owner()
        if not updates:
            raise ValueError("library updates must be nonempty")
        entry = self._require_write_target(target)
        parameters = self._published_parameters()
        handle = self._write_library(entry, target, parameters)
        actual = commit_library_updates(handle, updates)
        if target != parameters.target:
            if target.point is None:
                self._bus.emit(EntryDocumentsChangedPayload())
            return actual
        self._publish_write(
            parameters.storage,
            parameter=False,
        )
        return actual

    def apply_library_change(
        self, parameters: DocumentSetCapture, change: LibraryChange
    ) -> None:
        """Apply one structure command to the captured target in one transaction.

        parameters supplies the destination target. change uses concrete
        library roots; it never resolves global aliases. Put/replace write the
        change's persistent Spec as given, with manual sources; child library
        keys stay keys. Rename moves Spec and sources; delete does not cascade
        references. Replace checks source/destination and commits relocation
        plus value in one library transaction.
        Same-entry nonselected point writes do not select or publish.
        Global writes use the installed global library handle. A nonactive global
        commit updates its cache/revision before EntryDocumentsChangedPayload;
        the active point, version and existing captures remain unchanged.
        Cross-entry/missing entry/Spec family/source/conflict/I/O failures
        propagate, leaving this command uncommitted and publishing no events.
        """
        self._owner.assert_owner()
        target = parameters.target
        if target is None:
            raise FailedPreconditionError(
                "Select a working point first", reason_code="no_context"
            )
        entry = self._require_write_target(target)
        selected = self._published_parameters()
        handle = self._write_library(entry, target, selected)
        with handle.edit() as draft:
            candidate = change.apply(draft.snapshot())
            draft.replace_contents(candidate)
        if parameters.target != selected.target:
            if target.point is None:
                self._bus.emit(EntryDocumentsChangedPayload())
            return
        self._publish_write(
            selected.storage,
            parameter=False,
        )

    def entry_ledger(self, entry_name: str) -> RecordsLedger:
        """Return only the currently opened entry's ledger; never append an event.

        No entry raises no_entry; a different name in the current result root raises entry_not_open.
        Use capture_storage(parameters).ledger for a result's historical origin.
        Ledger capabilities are not serialized into wire data.
        """
        self._owner.assert_owner()
        return self._require_entry(entry_name).ledger

    def _documents_at(self, point: str | None) -> DocumentSet | None:
        if point is not None and point != self.current_point_label():
            raise InvalidInputError(
                f"Point {point!r} is not the focused point",
                reason_code="point_not_focused",
            )
        return self._global_set if point is None else self._active_set

    def _require_open_entry(self) -> ResultEntry:
        if self._entry is None:
            raise FailedPreconditionError("No entry is opened", reason_code="no_entry")
        return self._entry

    def _fresh_entry(self) -> ResultEntry:
        """Validate global files without mutating installed resource handles."""
        entry = self._require_open_entry()
        return ResultEntry.open(
            entry.result_path.name,
            result_root=entry.result_path.parent,
            database_root=entry.database_path.parent,
        )

    def _require_entry(self, entry_name: str) -> ResultEntry:
        entry = self._require_open_entry()
        if entry_name != entry.result_path.name:
            raise FailedPreconditionError(
                f"Entry {entry_name!r} is not opened", reason_code="entry_not_open"
            )
        return entry

    def _require_write_target(self, target: DocumentSetRef) -> ResultEntry:
        """Reject cross-entry targets before opening a resource."""
        return self._require_entry(target.entry_name)

    def _published_parameters(self) -> DocumentSetCapture:
        """Read the carrier without requiring unrelated selected-source data."""
        parameters = self._state.session_env.parameters
        if parameters is None:
            raise FailedPreconditionError(
                "Select or create a working point first", reason_code="no_context"
            )
        return parameters

    def _write_library(
        self, entry: ResultEntry, target: DocumentSetRef, parameters: DocumentSetCapture
    ) -> ModuleLibrary:
        """Use installed entry/active handles, otherwise open a detached point handle."""
        if target.point is None and self._global_set is not None:
            return self._global_set.library
        if target == parameters.target and self._active_set is not None:
            return self._active_set.library
        return entry.library(target.point)

    def _publish_write(
        self, storage: StorageBinding | None, *, parameter: bool
    ) -> None:
        """Install a committed projection/version before its single content fact."""
        documents = self._active_set
        target = self._published_parameters().target
        if documents is None or target is None:
            raise RuntimeError("Active document resources disappeared")
        parameters = DocumentSetCapture.from_set(
            documents, target=target, roles=None, storage=storage
        )
        self._state.set_context(replace(self._state.session_env, parameters=parameters))
        self._state.version.bump("context")
        payload = MdChangedPayload if parameter else MlChangedPayload
        self._bus.emit(
            payload(binding_id=self._binding_id, content_changed=True, fault=None)
        )

    def _load_points(self, entry: ResultEntry) -> tuple[WorkingPointInfo, ...]:
        points: list[WorkingPointInfo] = []
        for label in entry.list_points():
            description = ""
            try:
                description = entry.point_description(label)
                entry.validate_point_parameters(label)
            except (OSError, ValueError, YAMLError) as error:
                points.append(WorkingPointInfo(label, description, False, str(error)))
            else:
                points.append(WorkingPointInfo(label, description))
        return tuple(points)

    def _prepare_point(
        self, entry: ResultEntry, label: str, point: ParameterView
    ) -> tuple[
        DocumentSetCapture,
        DocumentSet,
        tuple[ContextSource, ...],
        WorkingPointInfo,
    ]:
        if self._global_set is None:
            raise RuntimeError("Opened entry has no global binding")
        assets = self._bind_assets(entry.database_path / "arb_waveforms")
        library = entry.library(label, assets=assets)
        documents = DocumentSet(point, library, entry.aliases(label))
        parameters = DocumentSetCapture.from_set(
            documents,
            target=DocumentSetRef(entry_name=entry.result_path.name, point=label),
            storage=capture_storage(point),
        )
        sources = (
            ContextSource("global", source_path(self._global_set.parameters), None),
            ContextSource("point", source_path(point), None),
            ContextSource("module_cfg", library.source_path, None),
        )
        info = WorkingPointInfo(label, point.description or "")
        return parameters, documents, sources, info

    def _install_point(
        self,
        documents: DocumentSet,
        parameters: DocumentSetCapture,
        sources: tuple[ContextSource, ...],
        info: WorkingPointInfo,
    ) -> None:
        points = {cached.label: cached for cached in self._points}
        points[info.label] = info
        self._points = tuple(points[label] for label in sorted(points))
        self._active_set = documents
        self._publish(
            parameters, sources, ContextReadiness.POINT, active_label=info.label
        )

    def _publish(
        self,
        parameters: DocumentSetCapture | None,
        sources: tuple[ContextSource, ...],
        readiness: ContextReadiness,
        *,
        active_label: str | None,
    ) -> None:
        self._file_sync.clear()
        if not sources:
            self._source_binding = None
        else:
            self._binding_id += 1
            self._source_binding = ContextSourceBinding(self._binding_id, sources)
        info = self._entry_info
        self._state.set_context(
            replace(
                self._state.session_env,
                parameters=parameters,
                result_dir=str(info.result_path) if info is not None else "",
                database_path=str(info.database_path) if info is not None else "",
                active_label=active_label,
                readiness=readiness,
            )
        )
        self._state.version.bump("context")
        self._bus.emit(
            ContextSwitchedPayload(
                binding_id=self._source_binding.binding_id
                if self._source_binding
                else None,
                sources=self.get_source_binding(),
            )
        )

    def get_session_env(self) -> SessionEnv:
        """Return the currently published environment; not a frozen Run snapshot."""
        return self._state.session_env

    def has_context(self) -> bool:
        """Return whether global or point parameters are published."""
        return self._state.session_env.has_context()

    def is_active_context(self) -> bool:
        """Return whether a file-backed global or point environment is published."""
        return self._state.session_env.is_active()

    def list_value_sources(self) -> tuple[ValueInfo, ...]:
        """Describe cached session values without reading hardware or files."""
        return self._values.describe()

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        """Resolve one registered value, optionally as int/float/str/bool.

        Return its description and scalar. Unknown keys raise MissingValue;
        conversion/provider errors propagate through the ValueLookup contract.
        """
        ref = ValueRef(key=key, type_name=type_name)
        for info in self._values.describe():
            if info.key == key:
                return info, resolve_value_ref(ref, self._values)
        raise MissingValue(key, f"Value source {key!r} is not registered")
