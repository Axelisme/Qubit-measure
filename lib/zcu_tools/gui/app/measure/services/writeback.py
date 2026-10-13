from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import JsonValue

from zcu_tools.cfg_model import ModuleSpec
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.adapter import (
    ModuleWriteback,
    ParameterWriteback,
    SampleWriteback,
    WritebackItem,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.form import FormWritePath
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import CommandOrigin
from zcu_tools.gui.session.library_entry import ModuleEntryForm
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.entry import RoleResolutionError, document_unit
from zcu_tools.resources.entry import writeback as concrete
from zcu_tools.resources.entry.ledger_models import AcceptedWrite, JsonObject

_GUI_ORIGIN = CommandOrigin("gui")


# Kind prefixes are deliberately draft-local. Two independent drafts can both
# expose ``param-1`` without sharing any Form or mutable item state.
_KIND_PREFIX = {
    ParameterWriteback: "param",
    ModuleWriteback: "ml",
    SampleWriteback: "sample",
}

# Sentinel for "argument not supplied" in set_item_field (None is a real value).
_UNSET: Any = object()


def _sample_fields(
    fields: JsonObject,
    parameters: ParameterContext | None,
    device_snapshot: Mapping[str, JsonObject] | None,
) -> JsonObject:
    """Bind automatic GUI coordinates once, or explain why no sample is offered."""
    if parameters is None:
        raise FailedPreconditionError(
            "Sample writeback requires a Run $flux_dev binding"
        )
    try:
        device_name = parameters.value("$flux_dev")
    except RoleResolutionError as exc:
        raise FailedPreconditionError(
            f"Sample writeback requires $flux_dev: {exc}"
        ) from exc
    if not isinstance(device_name, str) or not device_name:
        raise FailedPreconditionError("Sample writeback requires a Run $flux_dev name")
    if device_snapshot is None or device_name not in device_snapshot:
        raise FailedPreconditionError(f"Run snapshot has no device {device_name!r}")
    device = device_snapshot[device_name]
    value, unit = device.get("value"), device.get("value_unit")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FailedPreconditionError(
            f"Run device {device_name!r} has no numeric value"
        )
    if unit not in ("A", "V", "native"):
        raise FailedPreconditionError(
            f"Run device {device_name!r} has no usable value_unit"
        )
    return copy.deepcopy(fields) | {"dev_value": value, "dev_unit": unit}


@dataclass(frozen=True)
class WritebackEdit:
    """One ordered agent edit; omitted values differ from explicit None.

    session_id is the draft-local item ID. target_name, when given, retargets
    the item. proposed_value, when supplied, replaces a parameter proposal.
    edits, when given, is one Form write batch for a module item: model-field
    leaf paths mapped to JSON input values, which may be {"expr": text}.
    """

    session_id: str
    target_name: str | None = None
    proposed_value: Any = _UNSET
    edits: Mapping[FormWritePath, JsonValue] | None = None


@dataclass(frozen=True)
class WritebackWritten:
    """Actual destination values around a successful context write.

    id is the draft-local item ID. kind identifies parameter/module/sample.
    target is a concrete point path, library key or sample component. before is
    the latest stored value before application, or None when absent. after is the
    value read after successful application. Sample values are complete JSONL
    rows, including accepted metadata, not the producer-only proposal fields.
    """

    id: str
    kind: Literal["parameter", "module", "sample"]
    target: str
    before: object
    after: object


def _destination_path(item: WritebackItem, parameters: ParameterContext) -> str:
    if isinstance(item, SampleWriteback):
        return item.target_name
    if isinstance(item, ParameterWriteback):
        return parameters.resolve_path(item.target_name)
    return parameters.library_key(item.target_name)


def _destination_value(
    item: WritebackItem, parameters: ParameterContext, path: str
) -> object:
    if isinstance(item, ParameterWriteback):
        result = parameters.lookup(path)
        return result.value if result.present else None
    library = parameters.library
    if path not in library.module_keys():
        return None
    return library.entry_value(path, kind="module")


@dataclass(frozen=True)
class WritebackValues:
    """Complete preview facts for one item at its captured destination.

    actual_path is the captured parameter path, library key or sample component,
    after any draft-local retarget edit. current is the latest stored value at
    that destination, or None if missing. proposed is the complete validated
    value; for a module it is the Spec from the item's Form, evaluated against
    the creation capture and dumped in the same shape as a stored library
    entry. unit is the resources-provided static schema label for display only,
    or None for destinations without one (including ext and module items).
    Values never convert.
    """

    actual_path: str
    current: object
    proposed: object
    unit: str | None


@dataclass
class _DraftEntry:
    """Writeback's private item/Form association.

    ``WritebackItem`` is an adapter proposal and intentionally does not carry
    an editing Form or presentation summaries. A module item's Form (None for
    other items and without a captured context) and S2 display-only baseline
    summaries stay here, behind ``WritebackDraft`` and ``WritebackService``
    (app/service-owned read model, not adapter contract).
    """

    item: WritebackItem
    form: ModuleEntryForm | None = None
    # S2 display-only baseline — not on WritebackItem, not persisted/wire.
    current_summary: str | None = None
    proposed_summary: str | None = None
    applied: bool = False


@dataclass(frozen=True)
class _RunSource:
    """Captured Run facts, independent of the editable destination/proposals."""

    experiment_name: str
    run_id: str | None
    device_snapshot: Mapping[str, JsonObject] | None


class WritebackDraft:
    """Opaque, service-owned writeback draft.

    A draft is the only handle a workflow needs after handing proposal items to
    :class:`WritebackService`. Its item list is a read-model projection; module
    item Forms remain private to the service and reach viewers through
    ``WritebackService.get_item_form``.
    Mutations should use the service (or the small forwarding methods below), so
    the same draft is shared by the UI and remote editing surfaces.
    """

    __slots__ = (
        "_service",
        "_entries",
        "_closed",
        "_parameters",
        "_run_source",
        "_unavailable_reasons",
    )

    def __init__(
        self,
        service: WritebackService,
        entries: list[_DraftEntry],
        parameters: ParameterContext | None,
        *,
        run_source: _RunSource,
        unavailable_reasons: tuple[str, ...],
    ) -> None:
        self._run_source = copy.deepcopy(run_source)
        self._unavailable_reasons = unavailable_reasons
        self._service = service
        self._entries = entries
        self._parameters = parameters
        self._closed = False

    @property
    def parameters(self) -> ParameterContext | None:
        """Fixed creation capture, or None for a context-free empty draft.

        This read-only source owns role bindings, library values and the write
        target. Later selection changes never replace it.
        """
        return self._parameters

    @property
    def device_snapshot(self) -> Mapping[str, JsonObject] | None:
        """Detached Run-time device settings, or None when not captured."""
        return copy.deepcopy(self._run_source.device_snapshot)

    @property
    def unavailable_reasons(self) -> tuple[str, ...]:
        """Reasons for omitted sample proposals, fixed when the draft was created."""
        return self._unavailable_reasons

    @property
    def experiment_name(self) -> str:
        """Experiment name fixed at creation, used as accepted evidence kind."""
        return self._run_source.experiment_name

    @property
    def run_id(self) -> str | None:
        """Historical RunMetadata identity fixed at creation, or None without it.

        Save does not change this value. It identifies a Run, not whether that
        Run has a saved file; native Load retains the file's original identity.
        """
        return self._run_source.run_id

    @property
    def items(self) -> tuple[WritebackItem, ...]:
        """Current item projection, without any editor identity."""
        return tuple(entry.item for entry in self._entries)

    def require_owner(
        self, service: WritebackService, *, allow_closed: bool = False
    ) -> None:
        """Require the creating service and an active draft before owner operations.

        A different service raises InvalidInputError. A closed draft raises
        FailedPreconditionError unless allow_closed permits idempotent teardown.
        This check has no side effects.
        """
        if self._service is not service:
            raise InvalidInputError("unknown writeback draft")
        if self._closed and not allow_closed:
            raise FailedPreconditionError("writeback draft has been torn down")

    def entries_for(self, service: WritebackService) -> tuple[_DraftEntry, ...]:
        """Lend item/Form associations to the creating service while active.

        The tuple fixes order; the service may edit the contained associations.
        Frontends use items or WritebackService.get_item_form instead.
        Foreign/closed access raises as for require_owner.
        """
        self.require_owner(service)
        return tuple(self._entries)

    def close_for(self, service: WritebackService) -> None:
        """Mark this draft closed for its creating service, idempotently.

        Foreign access raises InvalidInputError. Closing releases no resources;
        all later active owner operations raise FailedPreconditionError.
        """
        self.require_owner(service, allow_closed=True)
        self._closed = True

    @property
    def is_active(self) -> bool:
        """Whether this draft has not been torn down."""
        return not self._closed

    def preview(self) -> list[WritebackItem]:
        return self._service.preview_draft(self)

    def edit(
        self,
        session_id: str,
        *,
        selected: bool | None = None,
        target_name: str | None = None,
        proposed_value: Any = _UNSET,
        edits: Mapping[FormWritePath, JsonValue] | None = None,
    ) -> None:
        self._service.edit_draft(
            self,
            session_id,
            selected=selected,
            target_name=target_name,
            proposed_value=proposed_value,
            edits=edits,
        )

    def apply(
        self,
        *,
        item_ids: tuple[str, ...] | None = None,
        origin: CommandOrigin = _GUI_ORIGIN,
    ) -> dict[str, Any]:
        return self._service.apply_draft(self, item_ids=item_ids, origin=origin)

    def teardown(self) -> None:
        self._service.teardown_draft(self)


def _entries_to_apply(
    entries: Sequence[_DraftEntry], item_ids: tuple[str, ...] | None
) -> list[_DraftEntry]:
    """Resolve explicit IDs independently of the GUI checkbox selection."""
    if item_ids is None:
        return [entry for entry in entries if entry.item.selected]
    requested = set(item_ids)
    if len(requested) != len(item_ids):
        raise InvalidInputError("duplicate writeback item IDs")
    unknown = requested - {entry.item.session_id for entry in entries}
    if unknown:
        raise InvalidInputError(f"unknown writeback item IDs: {sorted(unknown)}")
    return [entry for entry in entries if entry.item.session_id in requested]


class WritebackService:
    """Own opaque transactional drafts and their module item Forms.

    The primary interface is ``create_draft`` → ``preview_draft`` / ``edit_draft``
    → ``apply_draft`` / ``teardown_draft``. It accepts already-computed proposal
    items and has no knowledge of analysis stages, subtabs, figures, or adapter
    hooks.
    """

    def __init__(
        self,
        write_port: ContextControlPort,
        *,
        arb_names: Callable[[], Sequence[str]],
    ) -> None:
        """Write through write_port; read arb data options from arb_names.

        arb_names returns the arb waveform asset keys available to module
        Forms. It is called once per draft that has a module item, when the
        draft is created; every arb data field in that draft accepts only
        those names for its lifetime.
        """
        self._write = write_port
        self._arb_names = arb_names

    # ------------------------------------------------------------------
    # Opaque draft lifecycle
    # ------------------------------------------------------------------

    def create_draft(
        self,
        items: Iterable[WritebackItem],
        *,
        parameters: ParameterContext | None,
        experiment_name: str,
        run_id: str | None = None,
        device_snapshot: Mapping[str, JsonObject] | None = None,
    ) -> WritebackDraft:
        """Create a draft with one Form per module item at the creation capture.

        The input is a proposal collection, not an adapter or stage. Items are
        shallow-copied so draft-local ``session_id``, selection, target and value
        edits cannot mutate the adapter's proposal objects. Each module proposal
        is projected into a library entry Form (ModuleEntryForm): child library
        keys stay keys, and the optional hardware fields the shared Form
        generator leaves out return to Spec defaults. A proposal that is not a
        concrete persistent ModuleSpec raises TypeError and no draft is returned.

        parameters fixes expression/library resolution, arb options and the
        destination target for the draft lifetime. None is valid for an
        empty/context-free proposal: module items are listed but have no Form,
        so editing, preview and nonempty application fail. Baseline display
        uses this capture; no live selection provider is retained.
        experiment_name is the captured experiment name, never a frontend/event kind.
        run_id is the source RunMetadata.run_id, including unsaved Runs and
        historical native Loads; None means no native acquisition evidence.
        It stays fixed through Save, selection changes, edits and application.
        Application copies it into accepted evidence and library provenance.
        device_snapshot maps Run-fixed device names to their JSON settings; None
        means no device evidence. Samples require run_id and a captured flux
        binding with value/value_unit. Missing facts omit only those proposals
        and appear in draft.unavailable_reasons. Sample fields are detached and
        automatic device coordinates replace any producer-supplied coordinates.
        """
        if not experiment_name.strip():
            raise InvalidInputError("experiment_name must be nonempty")
        proposals = list(items)
        arb_names: tuple[str, ...] = ()
        if parameters is not None and any(
            isinstance(item, ModuleWriteback) for item in proposals
        ):
            arb_names = tuple(self._arb_names())
        entries: list[_DraftEntry] = []
        unavailable_reasons: list[str] = []
        for raw_item in proposals:
            item = self._copy_item(raw_item)
            if isinstance(item, SampleWriteback):
                try:
                    if run_id is None or not run_id.strip():
                        raise FailedPreconditionError(
                            "Sample writeback requires a captured run_id"
                        )
                    item.proposed_value = _sample_fields(
                        item.proposed_value, parameters, device_snapshot
                    )
                except FailedPreconditionError as exc:
                    unavailable_reasons.append(f"{item.target_name}: {exc}")
                    continue
            prefix = self._kind_prefix(item)
            item.session_id = self._next_session_id(entries, prefix)
            item.selected = True
            entry = _DraftEntry(item)
            # Baseline capture — display-only, app/service-owned.
            self._capture_baseline(entry, parameters)
            if isinstance(item, ModuleWriteback) and parameters is not None:
                entry.form = self._open_form(
                    entry, item.proposal, parameters, arb_names
                )
            entries.append(entry)
        return WritebackDraft(
            self,
            entries,
            parameters,
            run_source=_RunSource(experiment_name, run_id, device_snapshot),
            unavailable_reasons=tuple(unavailable_reasons),
        )

    def preview_draft(self, draft: WritebackDraft) -> list[WritebackItem]:
        self._require_draft(draft)
        return list(draft.items)

    def preview_values(
        self, draft: WritebackDraft, *, session_id: str | None = None
    ) -> dict[str, WritebackValues]:
        """Read actual current values at the captured target, without selecting it.

        Proposed module values evaluate against the creation capture, never
        current focus, point, or slot references. Missing destinations have
        current=None. An invalid module Form raises LibraryEntryInvalid.
        session_id=None reads every item, including unselected items. A supplied
        session_id reads only that item's facts, without lowering siblings; an
        unknown ID raises KeyError before reading the destination. Empty bulk
        reads return {}; empty item reads raise KeyError. Nonempty reads require
        a still-readable captured target. Inactive/foreign drafts are rejected.
        Target/source/I/O and requested proposal validation errors propagate.
        """
        self._require_draft(draft)
        items = draft.items
        if session_id is not None:
            items = tuple(item for item in items if item.session_id == session_id)
            if not items:
                raise KeyError(session_id)
        if not items:
            return {}
        parameters = self._require_parameters(draft)
        current = self._write.read_target(parameters.target, roles=parameters.roles)
        values: dict[str, WritebackValues] = {}
        for item in items:
            path = _destination_path(item, parameters)
            if isinstance(item, SampleWriteback):
                proposed = item.proposed_value
            elif isinstance(item, ParameterWriteback):
                proposed = AcceptedWrite.model_validate(
                    {"path": path, "value": item.proposed_value}
                ).value
            else:
                entry = self._find_draft_entry(draft, item.session_id)
                proposed = self._module_spec(entry).model_dump(mode="python")
            values[item.session_id] = WritebackValues(
                actual_path=path,
                current=self._read_destination_value(draft, item, current, path),
                proposed=proposed,
                unit=(
                    document_unit(current.document(), path)
                    if isinstance(item, ParameterWriteback)
                    else None
                ),
            )
        return values

    def _read_destination_value(
        self,
        draft: WritebackDraft,
        item: WritebackItem,
        current: ParameterContext,
        path: str,
    ) -> object:
        """Read one captured destination consistently for preview and receipts."""
        if isinstance(item, SampleWriteback) and draft.run_id is not None:
            parameters = self._require_parameters(draft)
            return self._write.read_sample(parameters.target, path, draft.run_id)
        return _destination_value(item, current, path)

    @staticmethod
    def _require_parameters(draft: WritebackDraft) -> ParameterContext:
        parameters = draft.parameters
        if parameters is None:
            raise FailedPreconditionError("Writeback requires a captured context.")
        return parameters

    def edit_draft(
        self,
        draft: WritebackDraft,
        session_id: str,
        *,
        selected: bool | None = None,
        target_name: str | None = None,
        proposed_value: Any = _UNSET,
        edits: Mapping[FormWritePath, JsonValue] | None = None,
    ) -> None:
        """Apply one draft-local edit, in the order selection, target, value, edits.

        selected only changes the GUI default apply selection. target_name
        retargets any item. proposed_value replaces a parameter proposal; other
        items raise InvalidInputError. edits writes one batch into a module
        item's Form with FormState.write semantics: the batch installs whole,
        ordinary invalid values stay in the Form and are reported by preview
        and apply, and structurally unknown paths raise InvalidInputError
        naming the item without changing the Form. Edits on a non-module item
        raise InvalidInputError; on a module item without a captured context,
        FailedPreconditionError. Earlier changes in this call are kept when a
        later one fails. Content changes mark the item unapplied.
        """
        self._require_draft(draft)
        entry = self._find_draft_entry(draft, session_id)
        item = entry.item
        if selected is not None:
            item.selected = selected
        if target_name is not None and target_name != item.target_name:
            item.target_name = target_name
            entry.applied = False
            # Keep the bounded proposed summary in sync for a module retarget.
            if isinstance(item, ModuleWriteback):
                # Preserve original current_summary; recompute proposed_summary
                # from new target (still bounded, not full cfg).
                is_update = entry.current_summary == "present"
                action = "update" if is_update else "create"
                if item.role_id:
                    entry.proposed_summary = f"{action} {item.role_id}"
                else:
                    entry.proposed_summary = f"{action} → {item.target_name}"
        if proposed_value is not _UNSET:
            if not isinstance(item, ParameterWriteback):
                raise InvalidInputError(
                    f"{session_id!r} is not a parameter item; proposed_value invalid"
                )
            if not self._values_equal(proposed_value, item.proposed_value):
                item.proposed_value = proposed_value
                entry.proposed_summary = self._format_value_summary(proposed_value)
                entry.applied = False
        if edits is not None:
            self._write_form(entry, edits)

    def _write_form(
        self, entry: _DraftEntry, edits: Mapping[FormWritePath, JsonValue]
    ) -> None:
        """Write one batch into a module item's Form; see edit_draft for errors."""
        session_id = entry.item.session_id
        if not isinstance(entry.item, ModuleWriteback):
            raise InvalidInputError(
                f"{session_id!r} is not a module item; edits invalid"
            )
        form = self._require_form(entry)
        try:
            form.state.write(edits, form.env)
        except ValueError as exc:
            # FormState rejects unknown/overlapping paths and missing sources
            # before installing anything; report them as the agent's input error.
            raise InvalidInputError(f"{session_id!r}: {exc}") from exc

    def get_item_form(self, draft: WritebackDraft, session_id: str) -> ModuleEntryForm:
        """Return the module item's Form that edits, preview and apply all use.

        Viewers write it with form.state.write(batch, form.env), execute
        form.state.run_command(path, name, form.env), or use a FormEditor
        on (form.state, form.env). Both operations publish through the same
        state subscription and mark the item unapplied. App callers obtain
        admission through get_writeback_item_form_for_pane immediately before
        each operation, not just when opening the view. A non-module
        item raises InvalidInputError; a module item without a captured
        context raises FailedPreconditionError. Inactive/foreign drafts and
        unknown IDs raise as for edit_draft.
        """
        self._require_draft(draft)
        entry = self._find_draft_entry(draft, session_id)
        if not isinstance(entry.item, ModuleWriteback):
            raise InvalidInputError(f"{session_id!r} is not a module item; no Form")
        return self._require_form(entry)

    def apply_draft(
        self,
        draft: WritebackDraft,
        *,
        item_ids: tuple[str, ...] | None = None,
        origin: CommandOrigin = _GUI_ORIGIN,
    ) -> dict[str, Any]:
        """Evaluate against the draft capture, append accepted, then set in order.

        Role paths resolve through the creation capture, never current focus.
        All selected values, including module Specs from their Forms, are built
        before the ledger append; an invalid module Form raises
        LibraryEntryInvalid with its field paths and writes nothing. Append
        failure writes nothing. On item failure, mark the successful prefix applied and re-raise
        the original cause unchanged, retaining accepted evidence and writes.
        Empty selection is a no-op without context. Nonempty writes require a
        captured SavedPoint in its still-open entry. GUI defaults
        to the GUI origin; MCP callers explicitly identify the MCP frontend.
        Unknown/duplicate IDs or concrete destinations fail before any write.
        Accepted payload and every library provenance use draft.run_id;
        application never re-reads Run or Save state to determine its identity.
        Each parameter's metadata is replaced by {"stderr": x} when the item has
        stderr, else by {}; no source, kind, run_id or time is written there.
        """
        self._require_draft(draft)
        selected_entries = _entries_to_apply(draft.entries_for(self), item_ids)
        written: dict[str, list[str]] = {
            "parameter": [],
            "ml_modules": [],
            "samples": [],
        }
        if not selected_entries:
            return {"applied_ids": [], "written": written}
        parameters = self._require_parameters(draft)
        destinations: set[str] = set()
        items: list[concrete.WritebackItem] = []
        for entry in selected_entries:
            item = entry.item
            if isinstance(item, SampleWriteback):
                path = item.target_name
                items.append(concrete.SampleWriteback(path, item.proposed_value))
                written["samples"].append(path)
            elif isinstance(item, ParameterWriteback):
                path = parameters.resolve_path(item.target_name)
                write = AcceptedWrite.model_validate(
                    {"path": path, "value": item.proposed_value}
                )
                metadata: YamlMap = (
                    {"stderr": item.stderr} if item.stderr is not None else {}
                )
                items.append(concrete.ParameterWriteback(path, write.value, metadata))
                written["parameter"].append(path)
            else:
                path = parameters.library_key(item.target_name)
                items.append(
                    concrete.ModuleWriteback(
                        path, self._module_spec(entry), item.stderr
                    )
                )
                written["ml_modules"].append(path)
            if path in destinations:
                raise InvalidInputError(f"duplicate writeback destination: {path!r}")
            destinations.add(path)
        try:
            self._write.apply_writeback(
                parameters.target,
                tuple(items),
                evidence=concrete.WritebackEvidence(
                    kind=draft.experiment_name,
                    origin=origin.origin,
                    run_id=draft.run_id,
                ),
            )
        except concrete.WritebackApplyError as exc:
            for entry in selected_entries[: exc.applied]:
                entry.applied = True
            # Keep GUI/wire classification and the producer's own cause chain.
            cause = exc.__cause__
            if cause is None:
                raise RuntimeError(
                    "WritebackApplyError requires its original cause"
                ) from exc
            raise cause from cause.__cause__
        for entry in selected_entries:
            entry.applied = True
        return {
            "applied_ids": [entry.item.session_id for entry in selected_entries],
            "written": written,
        }

    def write_draft(
        self,
        draft: WritebackDraft,
        changes: tuple[WritebackEdit, ...],
        *,
        origin: CommandOrigin = _GUI_ORIGIN,
    ) -> list[WritebackWritten]:
        """Edit sequentially and return actual fixed-target before/after values.

        The owner reads the captured destination before and after application;
        no current selection or stale capture substitutes for the receipt.
        Targets in receipts are concrete paths resolved from the creation capture.
        Empty changes are a no-op. Edit failures, including a listed module
        whose Form is invalid (LibraryEntryInvalid), retain the successful
        draft prefix without writes. Read/append/setter failures propagate; accepted
        evidence and any successful writes remain. MCP supplies
        CommandOrigin("mcp"); GUI uses the default.
        """
        self._require_draft(draft)
        if not changes:
            return []
        parameters = self._require_parameters(draft)
        ids = tuple(change.session_id for change in changes)
        if len(set(ids)) != len(ids):
            raise InvalidInputError("Repeated writeback IDs")
        items = []
        for change in changes:
            self.edit_draft(
                draft,
                change.session_id,
                target_name=change.target_name,
                proposed_value=change.proposed_value,
                edits=change.edits,
            )
            entry = self._find_draft_entry(draft, change.session_id)
            if isinstance(entry.item, ModuleWriteback):
                # Built only to reject invalid input before any context read or write.
                self._module_spec(entry)
            items.append(entry.item)
        paths = [_destination_path(item, parameters) for item in items]
        current = self._write.read_target(parameters.target, roles=parameters.roles)
        before = [
            self._read_destination_value(draft, item, current, path)
            for item, path in zip(items, paths, strict=True)
        ]
        self.apply_draft(draft, item_ids=ids, origin=origin)
        after = self._write.read_target(parameters.target, roles=parameters.roles)
        return [
            WritebackWritten(
                id=item.session_id,
                kind=(
                    "sample"
                    if isinstance(item, SampleWriteback)
                    else "parameter"
                    if isinstance(item, ParameterWriteback)
                    else "module"
                ),
                target=path,
                before=previous,
                after=self._read_destination_value(draft, item, after, path),
            )
            for item, path, previous in zip(items, paths, before, strict=True)
        ]

    def teardown_draft(self, draft: WritebackDraft) -> None:
        """Close a draft; later calls are no-ops and later uses raise.

        Item Forms hold no session or resource, so closing releases nothing.
        """
        self._require_draft(draft, allow_closed=True)
        draft.close_for(self)

    # Short behavior-oriented aliases for callers that already hold the opaque
    # draft. They keep the public surface independent of stage/subtab vocabulary.
    def preview(self, draft: WritebackDraft) -> list[WritebackItem]:
        return self.preview_draft(draft)

    def edit(self, draft: WritebackDraft, session_id: str, **changes: Any) -> None:
        self.edit_draft(draft, session_id, **changes)

    def apply(self, draft: WritebackDraft) -> dict[str, Any]:
        return self.apply_draft(draft)

    def teardown(self, draft: WritebackDraft) -> None:
        self.teardown_draft(draft)

    def _require_draft(
        self, draft: WritebackDraft, *, allow_closed: bool = False
    ) -> None:
        draft.require_owner(self, allow_closed=allow_closed)

    @staticmethod
    def _copy_item(item: WritebackItem) -> WritebackItem:
        copied = copy.copy(item)
        copied.session_id = ""
        copied.selected = True
        return copied

    @staticmethod
    def _kind_prefix(item: WritebackItem) -> str:
        for item_type, prefix in _KIND_PREFIX.items():
            if isinstance(item, item_type):
                return prefix
        raise InvalidInputError(
            f"unsupported writeback item type: {type(item).__name__}"
        )

    @staticmethod
    def _next_session_id(entries: list[_DraftEntry], prefix: str) -> str:
        count = sum(entry.item.session_id.startswith(f"{prefix}-") for entry in entries)
        return f"{prefix}-{count + 1}"

    def _find_draft_entry(self, draft: WritebackDraft, session_id: str) -> _DraftEntry:
        for entry in draft.entries_for(self):
            if entry.item.session_id == session_id:
                return entry
        raise InvalidInputError(f"unknown writeback session_id: {session_id!r}")

    def _open_form(
        self,
        entry: _DraftEntry,
        proposal: ModuleSpec,
        parameters: ParameterContext,
        arb_names: Sequence[str],
    ) -> ModuleEntryForm:
        form = ModuleEntryForm(proposal, parameters, arb_names=arb_names)
        # The Form lives exactly as long as its entry, so the subscription is
        # never removed and its unsubscribe handle is not kept.
        form.state.subscribe(lambda: self._mark_unapplied(entry))
        return form

    @staticmethod
    def _require_form(entry: _DraftEntry) -> ModuleEntryForm:
        if entry.form is None:
            raise FailedPreconditionError(
                f"{entry.item.session_id!r} has no Form without a captured context"
            )
        return entry.form

    def _module_spec(self, entry: _DraftEntry) -> ModuleSpec:
        """Return the module item's Spec; an invalid Form raises LibraryEntryInvalid."""
        return self._require_form(entry).spec()

    # ------------------------------------------------------------------
    # S2 — display-only baseline capture (read-only, single owner)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_value_summary(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, complex):
            return f"{value.real:.4g}{value.imag:+.4g}j"
        if isinstance(value, (list, tuple)):
            return WritebackService._format_sequence_summary(value)
        if isinstance(value, dict):
            return f"map[{len(value)}]"
        text = repr(value)
        return text if len(text) <= 48 else f"{text[:45]}..."

    @staticmethod
    def _format_sequence_summary(value: list[object] | tuple[object, ...]) -> str:
        """Summarize rectangular nonempty scalar rows as a matrix, else as a list."""
        rows: list[list[object] | tuple[object, ...]] = []
        for row in value:
            if not isinstance(row, (list, tuple)):
                return f"list[{len(value)}]"
            rows.append(row)
        if not rows:
            return f"list[{len(value)}]"
        row_lengths = {len(row) for row in rows}
        scalar_cells = all(
            not isinstance(cell, (list, tuple, dict)) for row in rows for cell in row
        )
        if len(row_lengths) == 1 and next(iter(row_lengths)) > 0 and scalar_cells:
            return f"{len(value)} × {next(iter(row_lengths))} matrix"
        return f"list[{len(value)}]"

    def _capture_baseline(
        self, entry: _DraftEntry, parameters: ParameterContext | None
    ) -> None:
        """Capture display-only summaries from the creation context."""
        item = entry.item
        if isinstance(item, SampleWriteback):
            entry.current_summary = "entry sample"
            entry.proposed_summary = self._format_value_summary(item.proposed_value)
        elif isinstance(item, ParameterWriteback):
            current = (
                None if parameters is None else parameters.lookup(item.target_name)
            )
            entry.current_summary = self._format_value_summary(
                current.value if current is not None and current.present else None
            )
            entry.proposed_summary = self._format_value_summary(item.proposed_value)
        elif isinstance(item, ModuleWriteback):
            exists = False
            if parameters is not None:
                path = parameters.library_key(item.target_name)
                exists = path in parameters.library.module_keys()
            entry.current_summary = "present" if exists else "— not present"
            action = "update" if exists else "create"
            role = item.role_id
            entry.proposed_summary = (
                f"{action} {role}" if role else f"{action} → {item.target_name}"
            )

    # App-local Qt presentation projection (S2) — not wire/persistence.
    def get_summaries(
        self, draft: WritebackDraft, session_id: str
    ) -> tuple[str | None, str | None]:
        """Return (current_summary, proposed_summary) for one item."""
        self._require_draft(draft)
        entry = self._find_draft_entry(draft, session_id)
        return entry.current_summary, entry.proposed_summary

    def get_all_summaries(
        self, draft: WritebackDraft
    ) -> dict[str, tuple[str | None, str | None]]:
        """Return mapping session_id -> (current, proposed) for the draft."""
        self._require_draft(draft)
        return {
            e.item.session_id: (e.current_summary, e.proposed_summary)
            for e in draft.entries_for(self)
        }

    def get_all_applied(self, draft: WritebackDraft) -> dict[str, bool]:
        """Return the draft-owned applied state for presentation."""
        self._require_draft(draft)
        return {
            entry.item.session_id: entry.applied for entry in draft.entries_for(self)
        }

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if left is right:
            return True
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _mark_unapplied(entry: _DraftEntry) -> None:
        entry.applied = False
