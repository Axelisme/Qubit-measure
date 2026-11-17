"""Narrow fixed-read, entry/point and alias control for shared consumers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from zcu_tools.cfg_model import ModuleSpec, WaveformSpec
    from zcu_tools.format_version import YamlMap, YamlValue
    from zcu_tools.gui.session.aliases import AliasRow
    from zcu_tools.gui.session.context_types import (
        ContextSourceBinding,
        DocumentSetRef,
        EntryInfo,
        FileRevision,
        FileSyncStatus,
        LibraryChange,
        PointCreateRequest,
        WorkingPointInfo,
    )
    from zcu_tools.gui.session.types import DocumentSetCapture, RoleBinding
    from zcu_tools.gui.session.value_lookup import ScalarValue, ValueInfo
    from zcu_tools.resources.entry.ledger_models import JsonObject
    from zcu_tools.resources.entry.schema import ParameterDocument
    from zcu_tools.resources.entry.storage import StorageBinding
    from zcu_tools.resources.entry.writeback import (
        WritebackEvidence,
        WritebackItem,
        WritebackResult,
    )


class ContextControlPort(Protocol):
    """Cached reads and owner-thread commands using ContextService's contract.

    Reads never refresh implicitly. read_context rejects EMPTY/unavailable
    sources, fixing current alias rows with optional temporary named overrides. Point commands validate the
    opened entry name before resource preparation. Preparation failures preserve
    the binding; a created-but-unbound point raises PointBindingError.
    """

    def inspect_document(self, *, point: str | None = None) -> ParameterDocument:
        """Return isolated installed global or focused point content without I/O.

        None selects global. A label must equal the focused point, otherwise
        raise InvalidInputError(point_not_focused). No entry raises
        FailedPreconditionError. Changing returned data does not write.
        """
        ...

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
        ...

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
        ...

    def capture_storage(self, parameters: DocumentSetCapture) -> StorageBinding:
        """Return historical origin/roles on the owner thread without I/O.

        Source faults propagate. Unbound captures raise
        FailedPreconditionError(storage_unavailable); current selection is ignored.
        """
        ...

    def read_target(
        self, target: DocumentSetRef, *, roles: RoleBinding | None = None
    ) -> DocumentSetCapture:
        """Read current values at a fixed target without selecting it.

        target.point=None reads global parameters/library; a label opens that
        point's independent views. Only parameters/library are opened, never
        aliases. Supplied role names stay fixed, without following live aliases.
        Cross-entry, missing point (MissingPointError), schema, source and I/O
        errors propagate. No selection, versions or events change.
        """
        ...

    def read_record(
        self, target: DocumentSetRef, component: str, run_id: str
    ) -> JsonObject | None:
        """Read the latest entry record for a run through the record owner.

        No selection/publication changes occur. Target admission, unsafe names,
        invalid run_id, corrupt history and I/O failures propagate. A missing row
        returns None. DocumentSetRef reads entry JSONL history.
        """
        ...

    def apply_writeback(
        self,
        target: DocumentSetRef,
        items: tuple[WritebackItem, ...],
        *,
        evidence: WritebackEvidence,
    ) -> WritebackResult:
        """Accept at the fixed target through the context owner and shared loop.

        Nonempty writes require a captured DocumentSetRef in the opened entry.
        Parameter/library values persist in that target's files; records append to entry
        JSONL. Successful prefix writes publish before later failures. Item failure raises
        WritebackApplyError with the event ID, committed values, applied count and original
        cause. Target admission and ledger append errors propagate unchanged.
        Return the accepted event ID and detached per-item actual values.
        No implicit point creation, saving, rollback or retry is provided.
        """
        ...

    def set_parameter(
        self,
        target: DocumentSetRef,
        path: str,
        value: YamlValue,
        *,
        metadata: YamlMap | None = None,
    ) -> YamlValue:
        """Write a concrete point/general path using the captured target.

        Runs on the owner thread. metadata=None keeps the path's opaque
        metadata; a mapping replaces it in the same point commit ({} clears).
        Cross-entry/missing-selection/schema/metadata representation/conflict/
        I/O failures propagate. Nonselected DocumentSetRef writes do not select it.
        Return the detached actual value after resource validation/commit, without I/O.
        """
        ...

    def set_module(
        self,
        target: DocumentSetRef,
        path: str,
        value: ModuleSpec,
    ) -> YamlMap:
        """Write a module; return its actual cached Spec mapping after commit."""
        ...

    def set_waveform(
        self,
        target: DocumentSetRef,
        path: str,
        value: WaveformSpec,
    ) -> YamlMap:
        """Write a waveform; return its actual cached Spec mapping after commit."""
        ...

    def apply_library_change(
        self, parameters: DocumentSetCapture, change: LibraryChange
    ) -> None:
        """Commit one fixed-target structure command via ContextService.

        Put/replace write the change's persistent Spec as given; rename/delete
        move or remove an existing root. The command runs in one library
        transaction. Writes require an opened entry. DocumentSetRef uses installed
        entry handles and notifies nonactive entry consumers without changing the
        active point or its version. Detached
        DocumentSetRef writes do not select or publish. Invalid target, Spec
        family, conflict and I/O failures write and publish nothing.
        """
        ...

    def get_predictors_path(self) -> Path | None:
        """Return the opened entry predictor directory, or None; no I/O or creation."""
        ...

    def get_file_revisions(self) -> tuple[FileRevision, ...]:
        """Read the active global/point file versions, or (); no I/O or refresh."""
        ...

    def get_installed_file_revisions(self) -> tuple[FileRevision, ...]:
        """Read active versions plus entry versions in POINT, without I/O.

        Each set is ordered parameters/library/aliases. Paths are absolute;
        missing library/aliases revisions are None. No entry returns ().
        """
        ...

    def get_file_sync_status(self) -> tuple[FileSyncStatus, ...]:
        """Return recent external observations without I/O; cleared on point switch."""
        ...

    def check_file_revision(self, expected: FileRevision) -> None:
        """Compare draft path/version with disk; mismatch rejects without mutation.

        None expects an absent library or aliases file. Invalid paths/base raise
        InvalidInputError; mismatch raises FailedPreconditionError(revision_mismatch).
        """
        ...

    def reload_file(self, path: Path, expected_revision: str | None) -> FileRevision:
        """Strictly install one absolute current owner file, then publish facts.

        Compare expected_revision with the installed base, None only for absent
        library or aliases. Invalid paths/candidates raise InvalidInputError with path/reason;
        stale base raises FailedPreconditionError(revision_mismatch). Failure
        retains memory/disk; caller owns holder admission. Captures stay fixed.
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

    def use_point(self, point: str | None = None) -> DocumentSetRef:
        """Select global for None/omitted, or any saved point label in the entry.

        No opened entry raises no_entry. Preparation failure preserves binding,
        versions, revisions and captures. Alias format/syntax errors are deferred
        to consumers. See ContextService.use_point for resource failure semantics.
        """
        ...

    def new_point(self, request: PointCreateRequest) -> DocumentSetRef:
        """Create from global/clone and select, preserving attribution/errors.

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

    def get_alias_rows(self, *, point: str | None = None) -> tuple[AliasRow, ...]:
        """Read installed authored rows without expansion or I/O.

        None selects global. A label must equal the focused point, otherwise
        raise InvalidInputError(point_not_focused). Pending candidates are not read.
        No entry returns empty rows. Invalid aliases raise their captured error.
        Missing files supply empty rows without fallback.
        """
        ...

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
        ...

    def list_value_sources(self) -> tuple[ValueInfo, ...]:
        """Describe the cached registered session values."""
        ...

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        """Resolve a registered value, optionally converting int/float/str/bool."""
        ...
