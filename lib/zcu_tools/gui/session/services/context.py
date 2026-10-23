"""Owner-thread entry/point bindings and fixed session parameter reads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.owner import OwnerThreadGuard
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.gui.session.context_types import (
    ContextSource,
    ContextSourceBinding,
    EntryInfo,
    LibraryChange,
    PointBindingError,
    SavedPoint,
    WorkingPointInfo,
)
from zcu_tools.gui.session.events import (
    AliasesChangedPayload,
    ContextSwitchedPayload,
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.gui.session.types import ContextReadiness, ParameterContext, RoleBinding
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ScalarValue,
    ValueInfo,
    ValueLookup,
    ValueRef,
    resolve_value_ref,
)
from zcu_tools.resources.context.library import (
    LibrarySnapshot,
    ModuleLibrary,
)
from zcu_tools.resources.entry import (
    DestinationExistsError,
    document_value,
    snapshot,
    source_path,
    validate_path_segment,
)
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.entry.ledger_models import JsonObject
from zcu_tools.resources.entry.schema import (
    PointDocument,
    SetupDocument,
)
from zcu_tools.resources.entry.storage import PointStorageBinding, capture_point_storage
from zcu_tools.resources.entry.writeback import (
    LibraryUpdate,
    ModuleWriteback,
    ParameterUpdate,
    WaveformWriteback,
    WritebackEvidence,
    WritebackItem,
    WritebackResult,
    apply_writeback,
)

if TYPE_CHECKING:
    from zcu_tools.cfg_model import ModuleSpec, WaveformAssetInfo, WaveformSpec
    from zcu_tools.format_version import YamlMap, YamlValue
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_types import (
        ContextTarget,
        PointCreateRequest,
        PointUseRequest,
    )
    from zcu_tools.gui.session.state import SessionState
    from zcu_tools.gui.session.types import SessionEnv
    from zcu_tools.resources.entry import (
        PointView,
        ResultEntry,
        SetupView,
    )
    from zcu_tools.resources.entry.ledger import RecordsLedger
    from zcu_tools.resources.entry.provenance import Provenance


class _ContextWritebackTarget:
    """Bind the context's target setters to the shared ordered execution loop."""

    def __init__(
        self, context: ContextService, target: SavedPoint, entry: ResultEntry
    ) -> None:
        self._context = context
        self._target = target
        self._entry = entry
        self.entry_id = entry.entry_id
        self.ledger = entry.ledger
        self.point = target.point

    def set_parameters(
        self, updates: tuple[ParameterUpdate, ...]
    ) -> tuple[YamlValue, ...]:
        return self._context.set_parameters(self._target, updates)

    def set_modules(self, updates: tuple[LibraryUpdate, ...]) -> tuple[YamlMap, ...]:
        return self._context.set_modules(self._target, updates)

    def append_sample(
        self,
        component: str,
        fields: JsonObject,
        *,
        run_id: str,
        at: datetime,
        source: str,
    ) -> JsonObject:
        return self._entry.samples(component).append(
            run_id=run_id, point=self.point, at=at, source=source, **fields
        )


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
        alias_rows: tuple[AliasRow, ...] = (),
    ) -> None:
        """Use owner State/bus, optional value facade, and explicit asset binding.

        bind_assets receives entry.database_path / 'arb_waveforms' on entry/point
        installation. Its capability stays with the library capture, not a
        global Database selection. Binding errors abort publication. values
        defaults to State's facade and does not change parameter content/version.
        alias_rows supplies initial memory-only rows before an entry is opened.
        Entry installation and clearing reset the table to empty. Selecting or
        creating a point replaces it with that point's saved rows; a missing
        file is empty, with no setup or caller-preset fallback. No component
        publication fills aliases automatically.
        """
        self._owner = OwnerThreadGuard()
        self._state = state
        self._bus = bus
        self._bind_assets = bind_assets
        self._values = values if values is not None else state.session_env.values
        self._entry: ResultEntry | None = None
        self._entry_info: EntryInfo | None = None
        self._setup: SetupView | None = None
        self._point: PointView | None = None
        self._library: ModuleLibrary | None = None
        self._points: tuple[WorkingPointInfo, ...] = ()
        self._source_binding: ContextSourceBinding | None = None
        self._binding_id = 0
        self._aliases = AliasTable(alias_rows)
        if state.session_env.values is not self._values:
            state.set_context(replace(state.session_env, values=self._values))

    def install_entry(self, entry: ResultEntry) -> EntryInfo:
        """Prepare an opened entry and atomically publish metadata without selecting a point.

        Prepare setup, the complete point-description cache, and entry-scoped
        assets before replacing any binding. Alias rows reset to empty because
        no point is selected; setup aliases are not read. Failure retains the old binding,
        not deleting an entry already created by resources. Success bumps the
        context once and publishes a switch fact after installation. It clears
        point selection and publishes EMPTY with parameters=None, while retaining
        entry metadata, the point list and setup source binding. SoC, device and
        independent predictor capabilities remain with their own owners.
        """
        self._owner.assert_owner()
        setup = entry.setup
        points = self._load_points(entry)
        self._bind_assets(entry.database_path / "arb_waveforms")
        info = EntryInfo(
            entry.entry_id,
            entry.result_path.name,
            entry.result_path,
            entry.database_path,
        )
        sources = (ContextSource("setup", source_path(setup), None),)
        self._entry = entry
        self._entry_info = info
        self._setup = setup
        self._point = None
        self._library = None
        self._points = points
        self._aliases = AliasTable()
        self._publish(None, sources, ContextReadiness.EMPTY, active_label="")
        return info

    def get_source_binding(self) -> ContextSourceBinding | None:
        """Return detached cached persistent sources, or None without an opened entry.

        An opened entry without a selected point is EMPTY with a setup source
        binding; selecting a point adds its point/library sources.

        This internal source ID is for facts/refresh, never part of read targets.
        Query performs no I/O and does not select a point.
        """
        return deepcopy(self._source_binding)

    def get_predictors_path(self) -> Path | None:
        """Return the opened entry predictor directory, or None without an entry.

        Read the ResultEntry-owned path without I/O, directory creation or point selection.
        """
        return None if self._entry is None else self._entry.predictors_path

    def get_entry_info(self) -> EntryInfo | None:
        """Return cached entry metadata, or None when no entry is opened.

        An opened entry remains readable in EMPTY before point selection.
        This query performs no I/O and never refreshes or selects a point.
        """
        return deepcopy(self._entry_info)

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        """Return cached label/descriptions, empty without an opened entry.

        refresh=True completely prepares a fresh list before replacing cache.
        Read/validation failures propagate, preserving the previous cache.
        """
        if refresh:
            self._owner.assert_owner()
            if self._entry is not None:
                self._points = self._load_points(self._entry)
        return deepcopy(self._points)

    def use_point(self, request: PointUseRequest) -> SavedPoint:
        """Prepare point/library/assets/aliases, then install ACTIVE and publish facts.

        Load point-local aliases, including an empty table for a missing file.
        No setup or preset fallback is used. Alias syntax and storage errors
        propagate before replacing the binding or table.

        No opened entry raises FailedPreconditionError(no_entry). Preparation
        failure retains the old binding. The returned target has no session
        selection generation.
        """
        self._owner.assert_owner()
        entry = self._require_open_entry()
        point = entry.use_point(request.label)
        parameters, library, sources, info, aliases = self._prepare_point(
            entry, request.label, point
        )
        self._aliases = aliases
        self._install_point(point, library, parameters, sources, info)
        return SavedPoint(entry_id=entry.entry_id, point=request.label)

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        """Create from setup or a same-entry clone, then install ACTIVE.

        No opened entry raises FailedPreconditionError(no_entry). Pass description
        and origin unchanged to ResultEntry.new_point. An unsafe
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
            entry.use_point(request.clone_from)
            if request.clone_from is not None
            else None
        )
        try:
            point = entry.new_point(
                request.label,
                clone_from=clone,
                description=request.description,
                origin=request.origin.origin,
            )
        except DestinationExistsError as exc:
            raise InvalidInputError(str(exc)) from exc
        try:
            parameters, library, sources, info, aliases = self._prepare_point(
                entry, request.label, point
            )
        except Exception as exc:
            # Resource creation succeeded; report the retained point as recovery.
            raise PointBindingError(
                entry_id=entry.entry_id, label=request.label
            ) from exc
        self._aliases = aliases
        self._install_point(point, library, parameters, sources, info)
        return SavedPoint(entry_id=entry.entry_id, point=request.label)

    def clear_context(self) -> None:
        """Publish EMPTY, clear entry/point, retaining non-parameter handles.

        Publish a switch fact after installation; do not delete resources.
        """
        self._owner.assert_owner()
        self._entry = None
        self._entry_info = None
        self._setup = None
        self._point = None
        self._library = None
        self._points = ()
        self._aliases = AliasTable()
        self._publish(None, (), ContextReadiness.EMPTY, active_label="")

    def inspect_document(self, *, setup: bool = False) -> SetupDocument | PointDocument:
        """Return the cached setup template or current point document without I/O.

        setup=True reads the opened entry template; False reads the published
        selected saved point. No entry/context raises FailedPreconditionError.
        Returned data is isolated; changing it does not write.
        """
        self._owner.assert_owner()
        if setup:
            if self._setup is None:
                raise FailedPreconditionError(
                    "No entry is opened", reason_code="no_entry"
                )
            return snapshot(self._setup)
        return self.read_context().document()

    def add_component(
        self, name: str, kind: str, fields: YamlMap, *, setup: bool = False
    ) -> None:
        """Add one component with working-unit fields using its registered kind.

        setup=True edits only the opened entry template. Otherwise edit the
        selected saved point in one resource transaction.
        Invalid names/kinds/required fields/duplicates leave content unchanged.
        Success on the current point installs its source before publishing one
        parameter change. Setup edits never change existing points or captures.
        No entry/context, schema, conflict and I/O errors propagate.
        """
        self._owner.assert_owner()
        if self._entry is None:
            raise FailedPreconditionError("No entry is opened", reason_code="no_entry")
        if setup:
            if self._setup is None:
                raise FailedPreconditionError(
                    "No setup is opened", reason_code="no_entry"
                )
            self._setup.add_component(name, kind=kind, **fields)
            return
        parameters = self._published_parameters()
        if self._point is None:
            raise FailedPreconditionError(
                "Select or create a working point first", reason_code="no_context"
            )
        self._point.add_component(name, kind=kind, **fields)
        document = snapshot(self._point)
        self._publish_write(
            ParameterContext(
                point=document,
                library=parameters.library,
                target=parameters.target,
                roles=None,
                storage=parameters.storage,
            ),
            parameter=True,
        )

    def set_setup_parameter(self, path: str, value: YamlValue) -> None:
        """Edit an existing concrete setup field in working units.

        Commit through the setup resource, never an existing point. No entry,
        invalid path/schema, conflict and I/O failures propagate; no retry.
        """
        self._owner.assert_owner()
        if self._setup is None:
            raise FailedPreconditionError("No entry is opened", reason_code="no_entry")
        with self._setup.edit() as draft:
            draft.set(path, value)

    def read_context(
        self, *, aliases: Mapping[str, str] | None = None
    ) -> ParameterContext:
        """Capture published point/library/storage and all current alias rows.

        aliases optionally maps aliases to temporary component or device names.
        Names need not belong to the point or device catalog. Overrides retain
        dependent authored links and change only this
        capture, never global rows, versions, publication or selection. Invalid
        identifiers raise InvalidInputError(alias_invalid). $alias returns its
        name string; $alias.path reads resolve lazily without consumer kind
        checks. Missing or
        unbound aliases raise RoleResolutionError naming the alias, with no
        concrete-name fallback. Bare concrete paths remain independent.
        The capture never follows later source or alias edits and performs no
        I/O. EMPTY raises FailedPreconditionError(no_context). Necessary source
        faults raise ContextSourceUnavailableError, never returning old values.
        """
        self._owner.assert_owner()
        parameters = self._state.session_env.parameters
        if parameters is None:
            raise FailedPreconditionError(
                "Select or create a working point first", reason_code="no_context"
            )
        table = self._aliases
        for alias, component in (aliases or {}).items():
            table = table.with_override(alias, component)
        return parameters.bind_aliases(table.rows, {})

    def get_alias_rows(self) -> tuple[AliasRow, ...]:
        """Read authored rows without expansion or I/O, even without a point.

        Entry installation/clearing leaves an empty table; explicit memory-only
        set_alias_rows may populate it before selecting a point.
        """
        self._owner.assert_owner()
        return self._aliases.rows

    def set_alias_rows(self, rows: tuple[AliasRow, ...]) -> None:
        """Install a complete ordered memory-only table on the owner sequence.

        Invalid syntax, duplicate names and non-backward $ links raise
        InvalidInputError(alias_invalid) without changing rows or versions.
        Bindings are checked only when read, not against the current point here.
        A changed table bumps context once and publishes AliasesChangedPayload
        after installation. An identical explicit submission publishes nothing
        and writes no file or ledger. No point is required. Selecting a point
        replaces these memory-only rows with its saved table.
        """
        self._owner.assert_owner()
        previous = self._aliases.rows
        self._aliases.replace(rows)
        if self._aliases.rows != previous:
            self._state.version.bump("context")
            self._bus.emit(AliasesChangedPayload(rows=self._aliases.rows))

    def apply_alias_rows(self, rows: tuple[AliasRow, ...]) -> None:
        """Validate and atomically save authored rows to the selected point.

        Runs on the owner thread. Without an opened entry and selected point,
        raise FailedPreconditionError(no_context) before validation or I/O.
        Otherwise validate syntax/order, save that point's aliases.yaml, then
        install memory and publish changes. Never write setup aliases.
        InvalidInputError(alias_invalid) and storage errors preserve rows,
        versions and the old file; causes propagate unchanged. Identical rows
        still save but publish no event. No ledger or other point-container
        write occurs; set_alias_rows remains memory-only.
        """
        self._owner.assert_owner()
        label = self._state.session_env.active_label
        if self._entry is None or self._point is None or not label:
            raise FailedPreconditionError(
                "Select or create a working point first", reason_code="no_context"
            )
        candidate = AliasTable(rows)
        self._entry.save_aliases(
            EntryAliases(aliases={row.alias: row.value for row in candidate.rows}),
            point=label,
        )
        self.set_alias_rows(candidate.rows)

    def capture_storage(self, parameters: ParameterContext) -> PointStorageBinding:
        """Return historical storage for a fixed capture on the owner sequence.

        Current entry and point are not read. Source faults propagate;
        unbound captures raise FailedPreconditionError(storage_unavailable).
        No refresh, file write, output reservation or ledger append occurs.
        """
        self._owner.assert_owner()
        return parameters.capture_storage()

    def read_target(
        self, target: ContextTarget, *, roles: RoleBinding | None = None
    ) -> ParameterContext:
        """Read current values at a fixed target without changing selection.

        SavedPoint opens independent point/library views in the currently opened
        entry and may perform file I/O. Supplied role names
        stay fixed. Cross-entry, missing point, schema, source and
        I/O failures propagate. This read publishes no versions or events.
        """
        self._owner.assert_owner()
        entry = self._require_write_target(target)
        point = entry.use_point(target.point)
        assets = self._bind_assets(entry.database_path / "arb_waveforms")
        library = entry.module_library(target.point, assets=assets)
        return ParameterContext(
            point=snapshot(point),
            library=library.snapshot(),
            target=target,
            roles=roles,
            storage=capture_point_storage(point),
        )

    def read_sample(
        self, target: ContextTarget, component: str, run_id: str
    ) -> JsonObject | None:
        """Read one run's latest entry JSONL sample without changing selection.

        target validates the opened entry identity, rejecting cross-entry targets;
        no point selection or point-file read is required. Sample history itself
        belongs to the entry, independent of point. The
        sample owner returns a detached complete JSON row or None if absent.
        Invalid component/run_id, corrupt history and I/O errors propagate. No
        dataframe projection, context publication or refresh is performed here.
        """
        self._owner.assert_owner()
        return self._require_write_target(target).samples(component).latest(run_id)

    def apply_writeback(
        self,
        target: ContextTarget,
        items: tuple[WritebackItem, ...],
        *,
        evidence: WritebackEvidence,
    ) -> WritebackResult:
        """Accept nonempty ordered concrete writes at a fixed SavedPoint.

        Validate the opened entry and captured SavedPoint before append.
        ResultEntry.use_point validates its label, existence and point document;
        ValueError/FileNotFoundError and other load errors propagate before
        append. The executor appends accepted, then commits one point group,
        one library group and individual samples, preserving each group's order.
        Parameter/library setters require a currently selected point and
        commit the captured point's files, not the selected point's files when
        different. Samples need no selection and append to entry JSONL with the
        captured point label. Thus a direct call without selection may append
        accepted and then fail at a parameter/library item; sample-only can
        succeed. Application GuardService admission rejects missing selection
        before calling this entry point.
        Each successful point/library group installs and publishes once.
        A failing item rolls back its whole group, not previously committed
        groups. Detached point writes never select it.
        Return the accepted event and each owner's actual committed values.
        WritebackApplyError retains this result's successful prefix, event_id,
        applied and original cause for item failures; append errors propagate
        unchanged. A same-entry selection change never redirects the target;
        cross-entry targets fail before append. No point creation, retry or
        rollback occurs.
        """
        self._owner.assert_owner()
        if not items:
            raise ValueError("writeback items must be nonempty")
        entry = self._require_write_target(target)
        entry.use_point(target.point)
        return apply_writeback(
            items,
            target=_ContextWritebackTarget(self, target, entry),
            evidence=evidence,
        )

    def set_parameter(
        self,
        target: ContextTarget,
        path: str,
        value: YamlValue,
        *,
        metadata: YamlMap | None = None,
    ) -> YamlValue:
        """Write one concrete path and return its detached actual working-unit value.

        Use the captured target, not the current selection. SavedPoint writes
        one point file in the opened entry; a nonselected point stays unselected.
        metadata=None keeps the path's opaque metadata; a mapping replaces it
        in the same commit ({} clears). No source, time or event is added.
        Success publishes only when writing the selected target, after resource
        commit/validation. Cross-entry, missing selection, path/schema/metadata
        representation, conflict and I/O failures propagate without publishing
        a candidate. No retry or batch receipt.
        """
        return self.set_parameters(target, (ParameterUpdate(path, value, metadata),))[0]

    def set_parameters(
        self, target: ContextTarget, updates: tuple[ParameterUpdate, ...]
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
        point = (
            self._point
            if target == parameters.target and self._point is not None
            else entry.use_point(target.point)
        )
        actual: list[YamlValue] = []
        with point.edit() as draft:
            for update in updates:
                draft.set(update.path, update.value)
                if update.metadata is not None:
                    draft.set_meta(update.path, update.metadata)
                actual.append(document_value(snapshot(draft), update.path))
        document = snapshot(point)
        self._points = tuple(
            replace(info, description=document.general.description or "")
            if info.label == target.point
            else info
            for info in self._points
        )
        if target != parameters.target:
            return tuple(actual)
        self._publish_write(
            ParameterContext(
                point=document,
                library=parameters.library,
                target=target,
                roles=None,
                storage=(
                    replace(
                        parameters.storage, description=document.general.description
                    )
                    if parameters.storage is not None
                    else None
                ),
            ),
            parameter=True,
        )
        return tuple(actual)

    def set_module(
        self,
        target: ContextTarget,
        path: str,
        value: ModuleSpec,
        *,
        source: Provenance | None = None,
    ) -> YamlMap:
        """Write a module and return its detached actual cached Spec mapping.

        Target/publication/error rules match set_parameter. Omitted source is
        manual; explicit Provenance is validated by the library owner against
        this entry's ledger. SavedPoint uses one library transaction.
        No lowering, materialization, accepted event production or batch receipt.
        """
        return self.set_modules(
            target, (LibraryUpdate(ModuleWriteback(path, value), source),)
        )[0]

    def set_waveform(
        self,
        target: ContextTarget,
        path: str,
        value: WaveformSpec,
        *,
        source: Provenance | None = None,
    ) -> YamlMap:
        """Write a waveform and return its detached actual cached Spec mapping.

        Target/publication/source/error rules match set_module. Owner rejects
        invalid roots/families/sources before installing any candidate.
        """
        return self.set_modules(
            target, (LibraryUpdate(WaveformWriteback(path, value), source),)
        )[0]

    def set_modules(
        self, target: ContextTarget, updates: tuple[LibraryUpdate, ...]
    ) -> tuple[YamlMap, ...]:
        """Commit a nonempty mixed module/waveform group and publish once.

        LibraryUpdate defines each key, family, Spec and source. Target and
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
        actual: list[YamlMap] = []
        with handle.edit() as draft:
            for update in updates:
                item = update.item
                if isinstance(item, ModuleWriteback):
                    draft.set_module(item.path, item.value, provenance=update.source)
                    kind = "module"
                else:
                    draft.set_waveform(item.path, item.value, provenance=update.source)
                    kind = "waveform"
                actual.append(draft.snapshot().entry_value(item.path, kind=kind))
        library = handle.snapshot()
        if target != parameters.target:
            return tuple(actual)
        self._publish_write(
            ParameterContext(
                point=parameters.document(),
                library=library,
                target=target,
                roles=None,
                storage=parameters.storage,
            ),
            parameter=False,
        )
        return tuple(actual)

    def apply_library_change(
        self, parameters: ParameterContext, change: LibraryChange
    ) -> None:
        """Apply one structure command to the captured target in one transaction.

        parameters supplies the destination target. change uses concrete
        library roots; it never resolves global aliases. Put/replace write the
        change's persistent Spec as given, with manual sources; child library
        keys stay keys. Rename moves Spec and sources; delete does not cascade
        references. Replace checks source/destination and commits relocation
        plus value in one library transaction.
        Same-entry nonselected SavedPoint writes do not select or publish.
        Cross-entry/missing selection/Spec family/source/conflict/I/O failures
        propagate, leaving this command uncommitted and publishing no events.
        """
        self._owner.assert_owner()
        entry = self._require_write_target(parameters.target)
        selected = self._published_parameters()
        handle = self._write_library(entry, parameters.target, selected)
        with handle.edit() as draft:
            candidate = self._library_change_candidate(draft.snapshot(), change)
            draft.replace_contents(candidate)
        if parameters.target != selected.target:
            return
        library = handle.snapshot()
        self._publish_write(
            ParameterContext(
                point=selected.document(),
                library=library,
                target=parameters.target,
                roles=None,
                storage=selected.storage,
            ),
            parameter=False,
        )

    def _library_change_candidate(
        self, library: LibrarySnapshot, change: LibraryChange
    ) -> LibrarySnapshot:
        """Prepare one isolated structure change using the resource owner's operations."""
        destination = change.new_path if change.new_path is not None else change.path
        if change.action == "delete":
            return library.deleted(change.path, kind=change.kind)
        if change.action == "rename":
            return library.renamed(change.path, destination, kind=change.kind)
        if change.action == "replace":
            library.entry_value(change.path, kind=change.kind)
            if destination != change.path:
                library = library.renamed(change.path, destination, kind=change.kind)
        if change.module is not None:
            return library.replaced_module(destination, change.module)
        if change.waveform is not None:
            return library.replaced_waveform(destination, change.waveform)
        raise InvalidInputError("Library put/replace requires a Spec")

    def entry_ledger(self, entry_id: str) -> RecordsLedger:
        """Return only the currently opened entry's ledger; never append an event.

        No entry raises no_entry; a different UUID raises entry_not_open.
        Use capture_storage(parameters).ledger for a result's historical origin.
        Ledger capabilities are not serialized into wire data.
        """
        self._owner.assert_owner()
        return self._require_entry(entry_id).ledger

    def _require_open_entry(self) -> ResultEntry:
        if self._entry is None:
            raise FailedPreconditionError("No entry is opened", reason_code="no_entry")
        return self._entry

    def _require_entry(self, entry_id: str) -> ResultEntry:
        entry = self._require_open_entry()
        if entry_id != entry.entry_id:
            raise FailedPreconditionError(
                f"Entry {entry_id!r} is not opened", reason_code="entry_not_open"
            )
        return entry

    def _require_write_target(self, target: ContextTarget) -> ResultEntry:
        """Reject cross-entry targets before opening a resource."""
        return self._require_entry(target.entry_id)

    def _published_parameters(self) -> ParameterContext:
        """Read the carrier without requiring unrelated selected-source data."""
        parameters = self._state.session_env.parameters
        if parameters is None:
            raise FailedPreconditionError(
                "Select or create a working point first", reason_code="no_context"
            )
        return parameters

    def _write_library(
        self, entry: ResultEntry, target: SavedPoint, parameters: ParameterContext
    ) -> ModuleLibrary:
        """Use the selected handle for conflicts, or open a detached Spec-only handle."""
        if target == parameters.target and self._library is not None:
            return self._library
        return entry.module_library(target.point)

    def _publish_write(self, parameters: ParameterContext, *, parameter: bool) -> None:
        """Install a committed projection/version before its single content fact."""
        self._state.set_context(replace(self._state.session_env, parameters=parameters))
        self._state.version.bump("context")
        payload = MdChangedPayload if parameter else MlChangedPayload
        self._bus.emit(
            payload(binding_id=self._binding_id, content_changed=True, fault=None)
        )

    def _load_points(self, entry: ResultEntry) -> tuple[WorkingPointInfo, ...]:
        return tuple(
            WorkingPointInfo(
                label, snapshot(entry.use_point(label)).general.description or ""
            )
            for label in entry.list_points()
        )

    def _prepare_point(
        self, entry: ResultEntry, label: str, point: PointView
    ) -> tuple[
        ParameterContext,
        ModuleLibrary,
        tuple[ContextSource, ...],
        WorkingPointInfo,
        AliasTable,
    ]:
        if self._setup is None:
            raise RuntimeError("Opened entry has no setup binding")
        assets = self._bind_assets(entry.database_path / "arb_waveforms")
        library = entry.module_library(label, assets=assets)
        document = snapshot(point)
        parameters = ParameterContext(
            point=document,
            library=library.snapshot(),
            target=SavedPoint(entry_id=entry.entry_id, point=label),
            roles=None,
            storage=capture_point_storage(point),
        )
        sources = (
            ContextSource("setup", source_path(self._setup), None),
            ContextSource("point", source_path(point), None),
            ContextSource("module_cfg", library.source_path, None),
        )
        info = WorkingPointInfo(label, document.general.description or "")
        saved_aliases = entry.load_aliases(point=label)
        aliases = AliasTable(
            tuple(
                AliasRow(name, value) for name, value in saved_aliases.aliases.items()
            )
        )
        return parameters, library, sources, info, aliases

    def _install_point(
        self,
        point: PointView,
        library: ModuleLibrary,
        parameters: ParameterContext,
        sources: tuple[ContextSource, ...],
        info: WorkingPointInfo,
    ) -> None:
        points = {cached.label: cached for cached in self._points}
        points[info.label] = info
        self._points = tuple(points[label] for label in sorted(points))
        self._point = point
        self._library = library
        self._publish(
            parameters, sources, ContextReadiness.ACTIVE, active_label=info.label
        )

    def _publish(
        self,
        parameters: ParameterContext | None,
        sources: tuple[ContextSource, ...],
        readiness: ContextReadiness,
        *,
        active_label: str,
    ) -> None:
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
        """Return whether a saved working point environment is published."""
        return self._state.session_env.has_context()

    def is_active_context(self) -> bool:
        """Return whether the published environment targets a persisted point."""
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
