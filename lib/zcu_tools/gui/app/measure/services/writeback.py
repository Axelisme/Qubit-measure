from __future__ import annotations

import copy
import logging
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from zcu_tools.cfg_model import (
    module_spec_from_raw,
    waveform_spec_from_raw,
)
from zcu_tools.gui.app.measure.adapter import (
    ModuleWriteback,
    ParameterWriteback,
    SampleWriteback,
    WaveformWriteback,
    WritebackItem,
)
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import CommandOrigin
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.entry import RoleResolutionError, document_unit
from zcu_tools.resources.entry import writeback as concrete
from zcu_tools.resources.entry.ledger_models import AcceptedWrite, JsonObject

from .ports import CfgEdit, CfgEditorPort, CfgEditResult

logger = logging.getLogger(__name__)
_GUI_ORIGIN = CommandOrigin("gui")

if TYPE_CHECKING:
    from zcu_tools.gui.cfg.binding import CfgDraft

# Kind prefixes are deliberately draft-local. Two independent drafts can both
# expose ``param-1`` without sharing any editor session or mutable item state.
_KIND_PREFIX = {
    ParameterWriteback: "param",
    ModuleWriteback: "ml",
    WaveformWriteback: "wf",
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
    """One ordered agent edit; omitted values differ from explicit None."""

    session_id: str
    target_name: str | None = None
    proposed_value: Any = _UNSET
    edits: list[dict[str, object]] | None = None


@dataclass(frozen=True)
class WritebackWritten:
    """Actual destination values around a successful context write.

    id is the draft-local item ID. kind identifies parameter/module/waveform/sample.
    target is a concrete point path, library key or sample component. before is
    the latest stored value before application, or None when absent. after is the
    value read after successful application. Sample values are complete JSONL
    rows, including accepted metadata, not the producer-only proposal fields.
    """

    id: str
    kind: Literal["parameter", "module", "waveform", "sample"]
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
    kind = "module" if isinstance(item, ModuleWriteback) else "waveform"
    keys = (
        parameters.library.module_keys()
        if kind == "module"
        else parameters.library.waveform_keys()
    )
    return parameters.library.entry_value(path, kind=kind) if path in keys else None


@dataclass(frozen=True)
class WritebackValues:
    """Complete preview facts for one item at its captured destination.

    actual_path is the captured parameter path, library key or sample component,
    after any draft-local retarget edit. current is the latest stored value at
    that destination, or None if missing. proposed is the complete validated
    value, lowered against the creation capture for cfg items. unit is the
    resources-provided static schema label for display only, or None for
    destinations without one (including ext and cfg items). Values never convert.
    """

    actual_path: str
    current: object
    proposed: object
    unit: str | None


@dataclass
class _DraftEntry:
    """Writeback's private item/session association.

    ``WritebackItem`` is an adapter proposal and intentionally does not carry a
    cfg-editor handle or presentation summaries. Handles and S2 display-only
    baseline summaries stay here, behind ``WritebackDraft`` and
    ``WritebackService`` (app/service-owned read model, not adapter contract).
    """

    item: WritebackItem
    editor_id: str | None = None
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
    :class:`WritebackService`. Its item list is a read-model projection; cfg
    editor identities and teardown bookkeeping remain private to the service.
    Mutations should use the service (or the small forwarding methods below), so
    the same draft is shared by the UI and remote editing surfaces.
    """

    __slots__ = (
        "_service",
        "_identity",
        "_entries",
        "_closed",
        "_parameters",
        "_run_source",
        "_unavailable_reasons",
    )

    def __init__(
        self,
        service: WritebackService,
        identity: str,
        entries: list[_DraftEntry],
        parameters: ParameterContext | None,
        *,
        run_source: _RunSource,
        unavailable_reasons: tuple[str, ...],
    ) -> None:
        self._run_source = copy.deepcopy(run_source)
        self._unavailable_reasons = unavailable_reasons
        self._service = service
        self._identity = identity
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
        """Experiment name fixed at creation, used as accepted provenance kind."""
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

    @property
    def is_active(self) -> bool:
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
        edits: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self._service.edit_draft(
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
    entries: list[_DraftEntry], item_ids: tuple[str, ...] | None
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
    """Own opaque transactional drafts and their cfg-editor sessions.

    The primary interface is ``create_draft`` → ``preview_draft`` / ``edit_draft``
    → ``apply_draft`` / ``teardown_draft``. It accepts already-computed proposal
    items and has no knowledge of analysis stages, subtabs, figures, or adapter
    hooks.
    """

    def __init__(
        self,
        cfg_editor: CfgEditorPort,
        write_port: ContextControlPort,
    ) -> None:
        self._cfg_editor = cfg_editor
        self._write = write_port

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
        """Create a draft and all item-local cfg sessions transactionally.

        The input is a proposal collection, not an adapter or stage. Items are
        shallow-copied so draft-local ``session_id``, selection, target and value
        edits cannot mutate the adapter's proposal objects. If any editor session
        fails to open, every session opened for this draft is torn down before the
        original exception is re-raised; no partially-created draft escapes.

        parameters fixes both expression/library resolution and destination
        target for the draft lifetime. None is valid for an empty/context-free
        proposal, but nonempty application fails without a destination. Baseline
        display uses this capture; no live selection provider is retained.
        experiment_name is the captured experiment name, never a frontend/event kind.
        run_id is the source RunMetadata.run_id, including unsaved Runs and
        historical native Loads; None means no native acquisition evidence.
        It stays fixed through Save, selection changes, edits and application.
        Application copies it into accepted evidence and destination provenance.
        device_snapshot maps Run-fixed device names to their JSON settings; None
        means no device evidence. Samples require run_id and a captured flux
        binding with value/value_unit. Missing facts omit only those proposals
        and appear in draft.unavailable_reasons. Sample fields are detached and
        automatic device coordinates replace any producer-supplied coordinates.
        """
        if not experiment_name.strip():
            raise InvalidInputError("experiment_name must be nonempty")
        identity = uuid.uuid4().hex
        entries: list[_DraftEntry] = []
        unavailable_reasons: list[str] = []
        try:
            for raw_item in items:
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
                entries.append(entry)
                if isinstance(item, (ModuleWriteback, WaveformWriteback)):
                    if item.edit_schema is None:
                        continue
                    editor_id, _ = self._cfg_editor.open_seeded(
                        item.edit_schema,
                        parameters=parameters,
                        gc=False,
                        owner_key=f"writeback:{identity}:{item.session_id}",
                    )
                    entry.editor_id = editor_id
                    editor_draft = self._cfg_editor.get_draft(editor_id)
                    editor_draft.on_change.connect(
                        lambda *_args, draft_entry=entry: self._mark_unapplied(
                            draft_entry
                        )
                    )
        except BaseException:
            # Cleanup is best-effort per session so one teardown failure cannot
            # strand the remaining sessions. The creation failure remains the
            # observable error and no draft is returned.
            self._teardown_entries(entries)
            raise
        return WritebackDraft(
            self,
            identity,
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

        Proposed cfg values lower against the creation capture, never current
        focus, point, or slot references. Missing destinations have current=None.
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
                proposed = schema_to_raw_dict(
                    self._entry_schema(self._find_draft_entry(draft, item.session_id)),
                    parameters,
                )
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
        edits: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        """Apply one draft-local edit; agent cfg changes use aggregate grammar."""
        self._require_draft(draft)
        entry = self._find_draft_entry(draft, session_id)
        item = entry.item
        if selected is not None:
            item.selected = selected
        if target_name is not None and target_name != item.target_name:
            item.target_name = target_name
            entry.applied = False
            # Keep proposed summary in sync for module/waveform retarget (bounded)
            if isinstance(item, (ModuleWriteback, WaveformWriteback)):
                # Preserve original current_summary; recompute proposed_summary
                # from new target (still bounded, not full cfg).
                base_current = entry.current_summary
                is_update = base_current == "present"
                action = "update" if is_update else "create"
                role = getattr(item, "role_id", None)
                if role:
                    entry.proposed_summary = f"{action} {role}"
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

        result = None
        if edits is not None:
            result = self._edit_cfg_entry(entry, edits)
        if result is None:
            return {"valid": True, "removed": [], "added": []}
        return result.to_wire()

    def _edit_cfg_entry(
        self, entry: _DraftEntry, edits: list[dict[str, object]]
    ) -> CfgEditResult:
        """Validate ordered cfg edits and return the editor's aggregate result.

        Non-cfg items and missing sessions fail before parsing. Missing edit fields
        fail before calling the editor; otherwise its result or failure propagates.
        """
        session_id = entry.item.session_id
        if not isinstance(entry.item, (ModuleWriteback, WaveformWriteback)):
            raise InvalidInputError(
                f"{session_id!r} is not a module/waveform item; edits invalid"
            )
        if entry.editor_id is None:
            raise FailedPreconditionError(
                f"{session_id!r} has no editable cfg model to apply edits to"
            )
        typed_edits: list[CfgEdit] = []
        for i, edit in enumerate(edits):
            if "path" not in edit or "value" not in edit:
                raise InvalidInputError(
                    f"edits[{i}] must be an object with 'path' and 'value'"
                )
            typed_edits.append(CfgEdit(str(edit["path"]), edit["value"]))
        return self._cfg_editor.set_fields(
            entry.editor_id, typed_edits, agent_edit=True
        )

    def get_item_draft(self, draft: WritebackDraft, session_id: str) -> CfgDraft:
        """Return an item model for a viewer without exposing its editor id."""
        self._require_draft(draft)
        entry = self._find_draft_entry(draft, session_id)
        if entry.editor_id is None:
            raise FailedPreconditionError(
                f"{session_id!r} has no editable cfg model to attach"
            )
        return self._cfg_editor.get_draft(entry.editor_id)

    def apply_draft(
        self,
        draft: WritebackDraft,
        *,
        item_ids: tuple[str, ...] | None = None,
        origin: CommandOrigin = _GUI_ORIGIN,
    ) -> dict[str, Any]:
        """Lower against the draft capture, append accepted, then set in order.

        Role paths resolve through the creation capture, never current focus.
        All selected values lower before the ledger append. Append failure writes
        nothing. On item failure, mark the successful prefix applied and re-raise
        the original cause unchanged, retaining accepted evidence and writes.
        Empty selection is a no-op without context. Nonempty writes require a
        captured SavedPoint in its still-open entry. GUI defaults
        to the GUI origin; MCP callers explicitly identify the MCP frontend.
        Unknown/duplicate IDs or concrete destinations fail before any write.
        Accepted payload and every destination provenance use draft.run_id;
        application never re-reads Run or Save state to determine its identity.
        """
        self._require_draft(draft)
        selected_entries = _entries_to_apply(draft._entries, item_ids)
        written: dict[str, list[str]] = {
            "parameter": [],
            "ml_modules": [],
            "ml_waveforms": [],
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
                items.append(
                    concrete.ParameterWriteback(path, write.value, item.stderr)
                )
                written["parameter"].append(path)
            else:
                path = parameters.library_key(item.target_name)
                raw = schema_to_raw_dict(self._entry_schema(entry), parameters)
                if isinstance(item, ModuleWriteback):
                    items.append(
                        concrete.ModuleWriteback(
                            path, module_spec_from_raw(raw), item.stderr
                        )
                    )
                    written["ml_modules"].append(path)
                else:
                    items.append(
                        concrete.WaveformWriteback(
                            path, waveform_spec_from_raw(raw), item.stderr
                        )
                    )
                    written["ml_waveforms"].append(path)
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
        Empty changes are a no-op. Edit failures retain the successful draft
        prefix without writes. Read/append/setter failures propagate; accepted
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
            result = self.edit_draft(
                draft,
                change.session_id,
                target_name=change.target_name,
                proposed_value=change.proposed_value,
                edits=change.edits,
            )
            if not result["valid"]:
                raise FailedPreconditionError(
                    f"Invalid writeback draft: {change.session_id!r}"
                )
            items.append(self._find_draft_entry(draft, change.session_id).item)
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
                    if isinstance(item, ModuleWriteback)
                    else "waveform"
                ),
                target=path,
                before=previous,
                after=self._read_destination_value(draft, item, after, path),
            )
            for item, path, previous in zip(items, paths, before, strict=True)
        ]

    def teardown_draft(self, draft: WritebackDraft) -> None:
        """Tear down a draft at most once; cleanup errors never cause a retry."""
        self._require_draft(draft, allow_closed=True)
        if draft._closed:
            return
        # Mark closed before calling driven cleanup. This makes teardown
        # idempotent even if a cfg editor's close implementation raises.
        draft._closed = True
        self._teardown_entries(draft._entries)

    # Short behavior-oriented aliases for callers that already hold the opaque
    # draft. They keep the public surface independent of stage/subtab vocabulary.
    def preview(self, draft: WritebackDraft) -> list[WritebackItem]:
        return self.preview_draft(draft)

    def edit(
        self, draft: WritebackDraft, session_id: str, **changes: Any
    ) -> dict[str, object]:
        return self.edit_draft(draft, session_id, **changes)

    def apply(self, draft: WritebackDraft) -> dict[str, Any]:
        return self.apply_draft(draft)

    def teardown(self, draft: WritebackDraft) -> None:
        self.teardown_draft(draft)

    def _require_draft(
        self, draft: WritebackDraft, *, allow_closed: bool = False
    ) -> None:
        if not isinstance(draft, WritebackDraft) or draft._service is not self:
            raise InvalidInputError("unknown writeback draft")
        if draft._closed and not allow_closed:
            raise FailedPreconditionError("writeback draft has been torn down")

    @staticmethod
    def _copy_item(item: WritebackItem) -> WritebackItem:
        copied = copy.copy(item)
        # Proposal items never carry cfg-editor identity. Treat a dynamic
        # attribute as an invalid caller contract instead of silently lowering it.
        if "editor_id" in vars(item):
            raise InvalidInputError("writeback proposal must not expose editor_id")
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
        for entry in draft._entries:
            if entry.item.session_id == session_id:
                return entry
        raise InvalidInputError(f"unknown writeback session_id: {session_id!r}")

    def _entry_schema(self, entry: _DraftEntry) -> CfgSchema:
        if entry.editor_id is not None:
            return self._cfg_editor.get_draft(entry.editor_id).snapshot()
        item = entry.item
        schema = (
            item.edit_schema
            if isinstance(item, (ModuleWriteback, WaveformWriteback))
            else None
        )
        if schema is None:
            raise FailedPreconditionError(
                f"writeback '{entry.item.session_id}' has no editable schema"
            )
        return schema

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
        elif isinstance(item, (ModuleWriteback, WaveformWriteback)):
            exists = False
            if parameters is not None:
                path = parameters.library_key(item.target_name)
                keys = (
                    parameters.library.module_keys()
                    if isinstance(item, ModuleWriteback)
                    else parameters.library.waveform_keys()
                )
                exists = path in keys
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
            for e in draft._entries
        }

    def get_all_applied(self, draft: WritebackDraft) -> dict[str, bool]:
        """Return the draft-owned applied state for presentation."""
        self._require_draft(draft)
        return {entry.item.session_id: entry.applied for entry in draft._entries}

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

    def _teardown_entries(self, entries: Iterable[_DraftEntry]) -> None:
        for entry in reversed(list(entries)):
            editor_id = entry.editor_id
            entry.editor_id = None
            if editor_id is None:
                continue
            try:
                self._cfg_editor.teardown(editor_id)
            except Exception:
                logger.exception(
                    "writeback draft editor teardown failed: %s", editor_id
                )
