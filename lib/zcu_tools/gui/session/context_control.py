"""Narrow fixed-read, entry/point and alias control for shared consumers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from zcu_tools.cfg_model import ModuleSpec, WaveformSpec
    from zcu_tools.format_version import YamlMap, YamlValue
    from zcu_tools.gui.session.aliases import AliasRow
    from zcu_tools.gui.session.context_types import (
        ContextSourceBinding,
        ContextTarget,
        EntryInfo,
        LibraryChange,
        PointCreateRequest,
        PointUseRequest,
        SavedPoint,
        WorkingPointInfo,
    )
    from zcu_tools.gui.session.services.context import ContextService
    from zcu_tools.gui.session.types import ParameterContext, RoleBinding
    from zcu_tools.gui.session.value_lookup import ScalarValue, ValueInfo
    from zcu_tools.resources.entry.ledger_models import JsonObject
    from zcu_tools.resources.entry.provenance import Provenance
    from zcu_tools.resources.entry.schema import PointDocument, SetupDocument
    from zcu_tools.resources.entry.storage import PointStorageBinding
    from zcu_tools.resources.entry.writeback import WritebackEvidence, WritebackItem


class ContextControlPort(Protocol):
    """Cached reads and owner-thread commands using ContextService's contract.

    Reads never refresh implicitly. read_context rejects EMPTY/unavailable
    sources, fixing current alias rows with optional temporary named overrides. Point commands validate the
    opened entry UUID before resource preparation. Preparation failures preserve
    the binding; a created-but-unbound point raises PointBindingError.
    """

    def inspect_document(self, *, setup: bool = False) -> SetupDocument | PointDocument:
        """Return the cached setup template or current point document without I/O.

        setup=True reads the opened entry template; False reads the published
        selected saved point. No entry/context raises FailedPreconditionError.
        Returned data is isolated; changing it does not write.
        """
        ...

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
        ...

    def set_setup_parameter(self, path: str, value: YamlValue) -> None:
        """Edit an existing concrete setup field in working units.

        Commit through the setup resource, never an existing point. No entry,
        invalid path/schema, conflict and I/O failures propagate; no retry.
        """
        ...

    def read_context(
        self, *, aliases: Mapping[str, str] | None = None
    ) -> ParameterContext:
        """Fix point/library/storage and all alias rows without I/O or kind checks.

        aliases maps alias names to temporary concrete components, preserving
        dependent links without changing global state. Later edits do not change
        the capture. Missing/unbound $ aliases fail by name when read, never
        falling back to bare concrete paths. EMPTY/source/invalid override errors
        propagate; see ContextService.read_context for the full contract.
        """
        ...

    def capture_storage(self, parameters: ParameterContext) -> PointStorageBinding:
        """Return historical origin/roles on the owner thread without I/O.

        Source faults propagate. Unbound captures raise
        FailedPreconditionError(storage_unavailable); current selection is ignored.
        """
        ...

    def read_target(
        self, target: ContextTarget, *, roles: RoleBinding | None = None
    ) -> ParameterContext:
        """Read current values at a fixed target without selecting it.

        SavedPoint may read point/library files in the opened entry. Supplied roles stay fixed;
        cross-entry, missing point, schema and I/O errors propagate.
        No versions or events are published.
        """
        ...

    def read_sample(
        self, target: ContextTarget, component: str, run_id: str
    ) -> JsonObject | None:
        """Read the latest entry sample for a run through the sample owner.

        No selection/publication changes occur. Target admission, unsafe names,
        invalid run_id, corrupt history and I/O failures propagate. A missing row
        returns None. SavedPoint reads entry JSONL history.
        """
        ...

    def apply_writeback(
        self,
        target: ContextTarget,
        items: tuple[WritebackItem, ...],
        *,
        evidence: WritebackEvidence,
    ) -> str:
        """Accept at the fixed target through the context owner and shared loop.

        Nonempty writes require a captured SavedPoint in the opened entry.
        SavedPoint parameter/library values persist and samples append to entry
        JSONL. Successful prefix writes publish before later failures. Item failure raises
        WritebackApplyError with accepted event_id, applied count and original
        cause. Target admission and ledger append errors propagate unchanged.
        No implicit point creation, saving, rollback or retry is provided.
        """
        ...

    def set_parameter(
        self,
        target: ContextTarget,
        path: str,
        value: YamlValue,
        *,
        metadata: YamlMap | None = None,
    ) -> None:
        """Write a concrete point/general path using the captured target.

        Runs on the owner thread. metadata=None keeps the path's opaque
        metadata; a mapping replaces it in the same point commit ({} clears).
        Cross-entry/missing-selection/schema/metadata representation/conflict/
        I/O failures propagate. Nonselected SavedPoint writes do not select it.
        Success follows resource validation/commit.
        """
        ...

    def set_module(
        self,
        target: ContextTarget,
        path: str,
        value: ModuleSpec,
        *,
        source: Provenance | None = None,
    ) -> None:
        """Write a Spec at a concrete module root; same target/source failures."""
        ...

    def set_waveform(
        self,
        target: ContextTarget,
        path: str,
        value: WaveformSpec,
        *,
        source: Provenance | None = None,
    ) -> None:
        """Write a Spec at a waveform root; same target/source failures."""
        ...

    def apply_library_change(
        self, parameters: ParameterContext, change: LibraryChange
    ) -> None:
        """Commit one fixed-target structure command via ContextService.

        Put/replace write the change's persistent Spec as given; rename/delete
        move or remove an existing root. The command runs in one library
        transaction. Writes require a selected working point. Detached
        SavedPoint writes do not select or publish. Invalid target, Spec
        family, source, conflict and I/O failures write and publish nothing.
        """
        ...

    def get_entry_info(self) -> EntryInfo | None:
        """Read cached entry identity/paths, or None without an entry."""
        ...

    def current_point_label(self) -> str | None:
        """Read the selected saved point label, or None when none is selected; no I/O."""
        ...

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        """Read cached point descriptions; True refreshes via the owner."""
        ...

    def use_point(self, request: PointUseRequest) -> SavedPoint:
        """Prepare/select a saved point in the opened entry."""
        ...

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        """Create from setup/clone and select, preserving attribution/errors.

        Existing destination raises InvalidInputError with the entry owner's
        exists message before writes, preserving binding, files and versions.
        Unsafe label raises InvalidInputError(point_label_invalid) before source
        reads or writes, preserving selection, files, ledger and versions.
        The entry owner defines safe single path segments. Other resource errors
        propagate; a created point with binding failure is retained and reported
        as PointBindingError. No entry raises FailedPreconditionError(no_entry).
        """
        ...

    def get_source_binding(self) -> ContextSourceBinding | None:
        """Read detached source facts without I/O."""
        ...

    def get_alias_rows(self) -> tuple[AliasRow, ...]:
        """Read immutable authored global rows on the owner thread without I/O.

        Links remain authored strings, not resolved component names. No entry
        is required; an uninitialized table may be empty.
        """
        ...

    def set_alias_rows(self, rows: tuple[AliasRow, ...]) -> None:
        """Atomically submit the complete ordered session-only alias table.

        Run on the owner thread; no entry or currently resolvable component is
        required. Invalid syntax, duplicate names and links to self/later rows
        raise InvalidInputError(alias_invalid) and preserve rows and versions.
        Changed rows publish AliasesChangedPayload after installation; identical
        submissions publish nothing. Either submission disables later default
        initialization. No files, preferences or ledger records are written.
        """
        ...

    def apply_alias_rows(self, rows: tuple[AliasRow, ...]) -> None:
        """Confirm the ordered table on the owner thread.

        Validate syntax/order, then atomically save to the opened entry before
        memory/event installation. EMPTY is memory-only. InvalidInputError or
        storage errors propagate, retaining old rows, versions and file. Saved
        empty rows are authoritative on reopen. Identical rows still save but
        emit no change; successful confirmation disables default initialization.
        No point, ledger or preferences write occurs.
        """
        ...

    def list_value_sources(self) -> tuple[ValueInfo, ...]:
        """Describe the cached registered session values."""
        ...

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        """Resolve a registered value, optionally converting int/float/str/bool."""
        ...


class ContextControlFacet(ContextControlPort):
    """Forward the narrow contract to the single context owner, without policy."""

    def __init__(self, *, context: ContextService) -> None:
        """Use the session's owning ContextService; construction performs no I/O."""
        self._context = context

    def inspect_document(self, *, setup: bool = False) -> SetupDocument | PointDocument:
        """Read the owner's isolated template/current point without I/O."""
        return self._context.inspect_document(setup=setup)

    def add_component(
        self, name: str, kind: str, fields: YamlMap, *, setup: bool = False
    ) -> None:
        """Add through the owning service; see ContextControlPort."""
        self._context.add_component(name, kind, fields, setup=setup)

    def set_setup_parameter(self, path: str, value: YamlValue) -> None:
        """Edit only the owning setup template."""
        self._context.set_setup_parameter(path, value)

    def read_context(
        self, *, aliases: Mapping[str, str] | None = None
    ) -> ParameterContext:
        """Return the fixed alias-aware capture described by ContextControlPort."""
        return self._context.read_context(aliases=aliases)

    def capture_storage(self, parameters: ParameterContext) -> PointStorageBinding:
        """Return the fixed capture's storage using the service contract."""
        return self._context.capture_storage(parameters)

    def read_target(
        self, target: ContextTarget, *, roles: RoleBinding | None = None
    ) -> ParameterContext:
        """Read a captured target using ContextControlPort's current-value policy."""
        return self._context.read_target(target, roles=roles)

    def read_sample(
        self, target: ContextTarget, component: str, run_id: str
    ) -> JsonObject | None:
        """Use the context owner's entry JSONL read contract."""
        return self._context.read_sample(target, component, run_id)

    def apply_writeback(
        self,
        target: ContextTarget,
        items: tuple[WritebackItem, ...],
        *,
        evidence: WritebackEvidence,
    ) -> str:
        """Use ContextControlPort's ordered acceptance and prefix contract."""
        return self._context.apply_writeback(target, items, evidence=evidence)

    def set_parameter(
        self,
        target: ContextTarget,
        path: str,
        value: YamlValue,
        *,
        metadata: YamlMap | None = None,
    ) -> None:
        self._context.set_parameter(target, path, value, metadata=metadata)

    def set_module(
        self,
        target: ContextTarget,
        path: str,
        value: ModuleSpec,
        *,
        source: Provenance | None = None,
    ) -> None:
        self._context.set_module(target, path, value, source=source)

    def set_waveform(
        self,
        target: ContextTarget,
        path: str,
        value: WaveformSpec,
        *,
        source: Provenance | None = None,
    ) -> None:
        self._context.set_waveform(target, path, value, source=source)

    def apply_library_change(
        self, parameters: ParameterContext, change: LibraryChange
    ) -> None:
        """Apply the structure command using ContextControlPort's target policy."""
        self._context.apply_library_change(parameters, change)

    def get_entry_info(self) -> EntryInfo | None:
        return self._context.get_entry_info()

    def current_point_label(self) -> str | None:
        # Published parameters exist only while a saved point is selected.
        parameters = self._context.get_session_env().parameters
        return None if parameters is None else parameters.target.point

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        return self._context.list_points(refresh=refresh)

    def use_point(self, request: PointUseRequest) -> SavedPoint:
        return self._context.use_point(request)

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        return self._context.new_point(request)

    def get_source_binding(self) -> ContextSourceBinding | None:
        return self._context.get_source_binding()

    def get_alias_rows(self) -> tuple[AliasRow, ...]:
        """Read authored rows using ContextControlPort's session-memory contract."""
        return self._context.get_alias_rows()

    def set_alias_rows(self, rows: tuple[AliasRow, ...]) -> None:
        """Commit the dialog's ordered rows through the owning service."""
        self._context.set_alias_rows(rows)

    def apply_alias_rows(self, rows: tuple[AliasRow, ...]) -> None:
        """Confirm rows using the port\'s validate/save/install contract."""
        self._context.apply_alias_rows(rows)

    def list_value_sources(self) -> tuple[ValueInfo, ...]:
        return self._context.list_value_sources()

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        return self._context.read_value_source(key, type_name)
