"""Owner-thread entry/point bindings and fixed session parameter reads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.owner import OwnerThreadGuard
from zcu_tools.gui.session.context_types import (
    ComponentFocus,
    ComponentOption,
    ContextSource,
    ContextSourceBinding,
    EntryDraft,
    EntryInfo,
    PointBindingError,
    SavedPoint,
    WorkingPointInfo,
)
from zcu_tools.gui.session.events import (
    ComponentFocusChangedPayload,
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
    MODULE_LIBRARY_FORMAT,
    MODULE_LIBRARY_VERSION,
    LibrarySnapshot,
    ModuleLibrary,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry import snapshot, source_path
from zcu_tools.resources.entry.roles import resolve_names
from zcu_tools.resources.entry.schema import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    PointDocument,
    PointGeneral,
    SetupDocument,
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
    from zcu_tools.resources.entry import PointView, ResultEntry, RoleSpec, SetupView
    from zcu_tools.resources.entry.ledger import RecordsLedger
    from zcu_tools.resources.entry.provenance import Provenance


class ContextService:
    """Own entry/point installation, fixed parameter captures, and session focus.

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
        self._focus = ComponentFocus(None, None, ())
        if state.session_env.values is not self._values:
            state.set_context(replace(state.session_env, values=self._values))

    def install_entry(self, entry: ResultEntry) -> EntryInfo:
        """Prepare an opened entry and atomically publish an empty DRAFT.

        Prepare setup, the complete point-description cache, and entry-scoped
        assets before replacing any binding. Failure retains the old binding,
        not deleting an entry already created by resources. Success bumps the
        context once and publishes switch/focus facts after installation.
        """
        self._owner.assert_owner()
        setup = entry.setup
        points = self._load_points(entry)
        assets = self._bind_assets(entry.database_path / "arb_waveforms")
        parameters = ParameterContext(
            point=PointDocument(
                format=PARAMETER_FORMAT,
                format_version=f"{PARAMETER_VERSION.major}.{PARAMETER_VERSION.minor}",
                general=PointGeneral(created_at=datetime.now(timezone.utc).isoformat()),
            ),
            library=LibrarySnapshot(
                ModuleLibraryDocument(
                    format=MODULE_LIBRARY_FORMAT,
                    format_version=f"{MODULE_LIBRARY_VERSION.major}.{MODULE_LIBRARY_VERSION.minor}",
                ),
                assets=assets,
            ),
            target=EntryDraft(entry_id=entry.entry_id),
            roles=None,
        )
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
        self._publish(parameters, sources, ContextReadiness.DRAFT, active_label="")
        return info

    def get_source_binding(self) -> ContextSourceBinding | None:
        """Return detached cached persistent sources, or None for EMPTY.

        This internal source ID is for facts/refresh, never part of read targets.
        Query performs no I/O and does not select a point.
        """
        return deepcopy(self._source_binding)

    def get_entry_info(self) -> EntryInfo | None:
        """Return the cached opened entry, or None for EMPTY; never refresh."""
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
        """Prepare point/library/assets, then install ACTIVE and publish facts.

        No opened entry raises FailedPreconditionError(no_entry); a different
        entry_id raises entry_not_open. Preparation failure retains the old
        binding. The returned target has no session selection generation.
        """
        self._owner.assert_owner()
        entry = self._require_entry(request.entry_id)
        point = entry.use_point(request.label)
        parameters, library, sources, info = self._prepare_point(
            entry, request.label, point
        )
        self._install_point(point, library, parameters, sources, info)
        return SavedPoint(entry_id=entry.entry_id, point=request.label)

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        """Create from setup or a same-entry clone, then install ACTIVE.

        Validate opened entry UUID before creation. Pass description and origin
        unchanged to ResultEntry.new_point; never copy DRAFT. Resource errors
        propagate. Post-creation binding failure raises PointBindingError from
        its cause, retaining the point and old binding without retry/deletion.
        """
        self._owner.assert_owner()
        entry = self._require_entry(request.entry_id)
        clone = (
            entry.use_point(request.clone_from)
            if request.clone_from is not None
            else None
        )
        point = entry.new_point(
            request.label,
            clone_from=clone,
            description=request.description,
            origin=request.origin.origin,
            call_id=request.origin.call_id,
        )
        try:
            parameters, library, sources, info = self._prepare_point(
                entry, request.label, point
            )
        except Exception as exc:
            # Resource creation succeeded; report the retained point as recovery.
            raise PointBindingError(
                entry_id=entry.entry_id, label=request.label
            ) from exc
        self._install_point(point, library, parameters, sources, info)
        return SavedPoint(entry_id=entry.entry_id, point=request.label)

    def clear_context(self) -> None:
        """Publish EMPTY, clear entry/point/focus, retaining non-parameter handles.

        Publish switch/focus facts after installation; do not delete resources.
        """
        self._owner.assert_owner()
        self._entry = None
        self._entry_info = None
        self._setup = None
        self._point = None
        self._library = None
        self._points = ()
        self._publish(None, (), ContextReadiness.EMPTY, active_label="")

    def inspect_document(self, *, setup: bool = False) -> SetupDocument | PointDocument:
        """Return the cached setup template or current point document without I/O.

        setup=True reads the opened entry template; False reads the published
        DRAFT/ACTIVE point. No entry/context raises FailedPreconditionError.
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
        current DRAFT in memory or ACTIVE point in one resource transaction.
        Invalid names/kinds/required fields/duplicates leave content unchanged.
        Success on the current point publishes one parameter change and updates
        focus options. Setup edits never change existing points or their captures.
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
        parameters = self.read_context()
        if isinstance(parameters.target, EntryDraft):
            document = self._entry.updated_component(
                parameters.document(), name, kind, fields
            )
        else:
            if self._point is None:
                raise FailedPreconditionError(
                    "No point is selected", reason_code="no_context"
                )
            self._point.add_component(name, kind=kind, **fields)
            document = snapshot(self._point)
        self._focus = replace(
            self._focus,
            components=tuple(
                ComponentOption(component_name, component.kind)
                for component_name, component in document.components.items()
            ),
        )
        self._publish_write(
            ParameterContext(
                point=document,
                library=parameters.library,
                target=parameters.target,
                roles=None,
            ),
            parameter=True,
        )
        self._bus.emit(ComponentFocusChangedPayload(focus=self.get_component_focus()))

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

    def read_context(self, *, roles: RoleBinding | None = None) -> ParameterContext:
        """Capture published point/library on the owner sequence without I/O.

        roles=None allows direct component/general paths only. Supplied names
        stay fixed; missing/wrong-kind destinations raise RoleResolutionError.
        EMPTY raises FailedPreconditionError(no_context). Necessary source
        faults raise ContextSourceUnavailableError, never returning old values.
        """
        self._owner.assert_owner()
        parameters = self._state.session_env.parameters
        if parameters is None:
            raise FailedPreconditionError(
                "No parameter context is installed", reason_code="no_context"
            )
        return ParameterContext(
            point=parameters.document(),
            library=parameters.library,
            target=parameters.target,
            roles=roles,
            source_faults=parameters.source_faults,
        )

    def read_target(
        self, target: ContextTarget, *, roles: RoleBinding | None = None
    ) -> ParameterContext:
        """Read current values at a fixed target without changing selection.

        SavedPoint opens independent point/library views in the currently opened
        entry and may perform file I/O. EntryDraft reads the published memory
        draft and requires that draft to remain selected. Supplied role names
        stay fixed. Cross-entry, stale draft, missing point, schema, source and
        I/O failures propagate. This read publishes no versions or events.
        """
        self._owner.assert_owner()
        entry = self._require_write_target(target)
        if isinstance(target, EntryDraft):
            return self.read_context(roles=roles)
        point = entry.use_point(target.point)
        assets = self._bind_assets(entry.database_path / "arb_waveforms")
        library = entry.module_library(target.point, assets=assets)
        return ParameterContext(
            point=snapshot(point),
            library=library.snapshot(),
            target=target,
            roles=roles,
        )

    def capture_roles(
        self,
        declarations: Mapping[str, RoleSpec],
        *,
        focus: str | None,
        explicit: Mapping[str, str],
    ) -> RoleBinding:
        """Resolve declared roles once against the published point snapshot.

        Explicit choices precede focus/via inference. Empty declarations remain
        empty. Entry RoleResolutionError propagates. EMPTY/source faults follow
        read_context policy; no YAML/hardware read occurs.
        """
        parameters = self.read_context()
        names = resolve_names(
            parameters.document().components, declarations, focus, explicit
        )
        return RoleBinding(focus, declarations, names)

    def set_parameter(
        self,
        target: ContextTarget,
        path: str,
        value: YamlValue,
        *,
        source: Provenance | None = None,
    ) -> None:
        """Write one concrete component/general path in working units.

        Use the captured target, not the current selection. SavedPoint writes
        one point file in the opened entry; a nonselected point stays unselected.
        EntryDraft requires that entry's current memory-only DRAFT and writes
        no point file. Omitted source records manual provenance; explicit source
        is validated against this entry's ledger. Success publishes only when
        writing the selected target, after resource commit/validation.
        Cross-entry, stale DRAFT, path/schema/source, conflict and I/O failures
        propagate without publishing a candidate. No retry or batch receipt.
        """
        self._owner.assert_owner()
        entry = self._require_write_target(target)
        parameters = self._published_parameters()
        if isinstance(target, EntryDraft):
            document = entry.updated_point(
                parameters.document(), path, value, source=source
            )
        else:
            point = (
                self._point
                if target == parameters.target and self._point is not None
                else entry.use_point(target.point)
            )
            with point.edit() as draft:
                draft.set(path, value, provenance=source)
            document = snapshot(point)
            self._points = tuple(
                replace(info, description=document.general.description or "")
                if info.label == target.point
                else info
                for info in self._points
            )
            if target != parameters.target:
                return
        self._publish_write(
            ParameterContext(
                point=document,
                library=parameters.library,
                target=target,
                roles=None,
            ),
            parameter=True,
        )

    def set_module(
        self,
        target: ContextTarget,
        path: str,
        value: ModuleSpec,
        *,
        source: Provenance | None = None,
    ) -> None:
        """Write a validated ModuleSpec at a concrete library root.

        Target/publication/source/error rules match set_parameter. SavedPoint
        uses one library transaction; DRAFT replaces an isolated memory library.
        No lowering, materialization, accepted event production or batch receipt.
        """
        self._owner.assert_owner()
        entry = self._require_write_target(target)
        parameters = self._published_parameters()
        if isinstance(target, EntryDraft):
            library = parameters.library.replaced_module(path, value, provenance=source)
            entry.validate_library(library.document())
        else:
            handle = self._write_library(entry, target, parameters)
            with handle.edit() as draft:
                draft.set_module(path, value, provenance=source)
            if target != parameters.target:
                return
            library = handle.snapshot()
        self._publish_write(
            ParameterContext(
                point=parameters.document(), library=library, target=target, roles=None
            ),
            parameter=False,
        )

    def set_waveform(
        self,
        target: ContextTarget,
        path: str,
        value: WaveformSpec,
        *,
        source: Provenance | None = None,
    ) -> None:
        """Write a WaveformSpec at a concrete waveforms root.

        Target/publication/source/error rules match set_module. Owner rejects
        invalid roots/families/sources before installing any candidate.
        """
        self._owner.assert_owner()
        entry = self._require_write_target(target)
        parameters = self._published_parameters()
        if isinstance(target, EntryDraft):
            library = parameters.library.replaced_waveform(
                path, value, provenance=source
            )
            entry.validate_library(library.document())
        else:
            handle = self._write_library(entry, target, parameters)
            with handle.edit() as draft:
                draft.set_waveform(path, value, provenance=source)
            if target != parameters.target:
                return
            library = handle.snapshot()
        self._publish_write(
            ParameterContext(
                point=parameters.document(), library=library, target=target, roles=None
            ),
            parameter=False,
        )

    def entry_ledger(self, entry_id: str) -> RecordsLedger:
        """Return only the currently opened entry's ledger; never append an event.

        No entry raises no_entry; a different UUID raises entry_not_open.
        The live capability is not part of ParameterContext or wire data.
        """
        self._owner.assert_owner()
        return self._require_entry(entry_id).ledger

    def get_component_focus(self) -> ComponentFocus:
        """Return cached target/name/component options without I/O or refresh."""
        return deepcopy(self._focus)

    def set_component_focus(self, name: str | None) -> ComponentFocus:
        """Select an existing component or clear focus; reject unknown names.

        Never change consumer roles or context/cfg revision. Publish a focus
        fact only for a changed selection. None does not trigger auto-selection.
        """
        self._owner.assert_owner()
        if name is not None and not any(
            component.name == name for component in self._focus.components
        ):
            raise InvalidInputError(f"Unknown component {name!r}")
        if name != self._focus.name:
            self._focus = replace(self._focus, name=name)
            self._bus.emit(
                ComponentFocusChangedPayload(focus=self.get_component_focus())
            )
        return self.get_component_focus()

    def _require_entry(self, entry_id: str) -> ResultEntry:
        if self._entry is None:
            raise FailedPreconditionError("No entry is opened", reason_code="no_entry")
        if entry_id != self._entry.entry_id:
            raise FailedPreconditionError(
                f"Entry {entry_id!r} is not opened", reason_code="entry_not_open"
            )
        return self._entry

    def _require_write_target(self, target: ContextTarget) -> ResultEntry:
        """Reject cross-entry and stale memory targets before opening a resource."""
        entry = self._require_entry(target.entry_id)
        if isinstance(target, EntryDraft) and not self.has_draft_context():
            raise FailedPreconditionError(
                "The entry's memory DRAFT is no longer selected",
                reason_code="draft_not_selected",
            )
        return entry

    def _published_parameters(self) -> ParameterContext:
        """Read the carrier without requiring unrelated selected-source data."""
        parameters = self._state.session_env.parameters
        if parameters is None:
            raise RuntimeError("Opened entry has no parameter projection")
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
        ParameterContext, ModuleLibrary, tuple[ContextSource, ...], WorkingPointInfo
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
        )
        sources = (
            ContextSource("setup", source_path(self._setup), None),
            ContextSource("point", source_path(point), None),
            ContextSource("module_cfg", library.source_path, None),
        )
        info = WorkingPointInfo(label, document.general.description or "")
        return parameters, library, sources, info

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
        self._focus = ComponentFocus(
            target=parameters.target if parameters is not None else None,
            name=None,
            components=tuple(
                ComponentOption(name, component.kind)
                for name, component in parameters.document().components.items()
            )
            if parameters is not None
            else (),
        )
        if parameters is None:
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
        self._bus.emit(ComponentFocusChangedPayload(focus=self.get_component_focus()))

    def get_session_env(self) -> SessionEnv:
        """Return the currently published environment; not a frozen Run snapshot."""
        return self._state.session_env

    def has_context(self) -> bool:
        """Return whether any DRAFT/ACTIVE parameter environment is published."""
        return self._state.session_env.has_context()

    def has_draft_context(self) -> bool:
        """Return whether the published environment is memory-only DRAFT."""
        return self._state.session_env.is_draft()

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
