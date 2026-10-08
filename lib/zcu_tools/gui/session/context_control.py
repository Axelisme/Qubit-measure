"""Narrow fixed-read, entry/point and focus surface for shared consumers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from zcu_tools.cfg_model import ModuleSpec, WaveformSpec
    from zcu_tools.format_version import YamlMap, YamlValue
    from zcu_tools.gui.cfg import CfgSchema
    from zcu_tools.gui.session.context_types import (
        ComponentFocus,
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
    from zcu_tools.resources.entry.provenance import Provenance
    from zcu_tools.resources.entry.schema import PointDocument, SetupDocument


class ContextControlPort(Protocol):
    """Cached reads and owner-thread commands using ContextService's contract.

    Reads never refresh implicitly. read_context rejects EMPTY/unavailable
    sources, preserving fixed supplied role names. Point commands validate the
    opened entry UUID before resource preparation. Preparation failures preserve
    the binding; a created-but-unbound point raises PointBindingError. Focus
    commands do not edit consumer roles or bump parameter revisions.
    """

    def inspect_document(self, *, setup: bool = False) -> SetupDocument | PointDocument:
        """Return the cached setup template or current point document without I/O.

        setup=True reads the opened entry template; False reads the published
        DRAFT/ACTIVE point. No entry/context raises FailedPreconditionError.
        Returned data is isolated; changing it does not write.
        """
        ...

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
        ...

    def set_setup_parameter(self, path: str, value: YamlValue) -> None:
        """Edit an existing concrete setup field in working units.

        Commit through the setup resource, never an existing point. No entry,
        invalid path/schema, conflict and I/O failures propagate; no retry.
        """
        ...

    def read_context(self, *, roles: RoleBinding | None = None) -> ParameterContext:
        """Capture published values; EMPTY/source/role errors propagate."""
        ...

    def read_target(
        self, target: ContextTarget, *, roles: RoleBinding | None = None
    ) -> ParameterContext:
        """Read current values at a fixed target without selecting it.

        SavedPoint may read point/library files in the opened entry. EntryDraft
        requires that memory draft to remain selected. Supplied roles stay fixed;
        cross-entry, stale draft, missing point, schema and I/O errors propagate.
        No versions or events are published.
        """
        ...

    def set_parameter(
        self,
        target: ContextTarget,
        path: str,
        value: YamlValue,
        *,
        source: Provenance | None = None,
    ) -> None:
        """Write a concrete point/general path using the captured target.

        Runs on the owner thread. Omitted source is manual; explicit Provenance
        uses this entry's ledger. Cross-entry/stale-DRAFT/schema/source/conflict/
        I/O failures propagate. Nonselected SavedPoint writes do not select it;
        DRAFT is memory-only. Success follows resource validation/commit.
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
        self,
        parameters: ParameterContext,
        change: LibraryChange,
        *,
        lower_module: Callable[[CfgSchema, ParameterContext], ModuleSpec],
        lower_waveform: Callable[[CfgSchema, ParameterContext], WaveformSpec],
    ) -> None:
        """Lower/commit one fixed-target structure command via ContextService.

        The supplied callbacks convert put/replace schemas using parameters;
        rename/delete do not call them. Draft writes require selected DRAFT.
        Detached SavedPoint writes do not select or publish. Invalid target,
        lowering/schema/source/conflict/I/O failures publish nothing.
        """
        ...

    def get_entry_info(self) -> EntryInfo | None:
        """Read cached entry identity/paths, or None without an entry."""
        ...

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        """Read cached point descriptions; True refreshes via the owner."""
        ...

    def use_point(self, request: PointUseRequest) -> SavedPoint:
        """Prepare/select a saved point in the opened entry."""
        ...

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        """Create from setup/clone and select, preserving attribution/errors."""
        ...

    def get_source_binding(self) -> ContextSourceBinding | None:
        """Read detached source facts without I/O."""
        ...

    def get_component_focus(self) -> ComponentFocus:
        """Read cached target/selection/options without I/O."""
        ...

    def set_component_focus(self, name: str | None) -> ComponentFocus:
        """Select a current component or None; unknown names raise."""
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

    def read_context(self, *, roles: RoleBinding | None = None) -> ParameterContext:
        return self._context.read_context(roles=roles)

    def read_target(
        self, target: ContextTarget, *, roles: RoleBinding | None = None
    ) -> ParameterContext:
        """Read a captured target using ContextControlPort's current-value policy."""
        return self._context.read_target(target, roles=roles)

    def set_parameter(
        self,
        target: ContextTarget,
        path: str,
        value: YamlValue,
        *,
        source: Provenance | None = None,
    ) -> None:
        self._context.set_parameter(target, path, value, source=source)

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
        self,
        parameters: ParameterContext,
        change: LibraryChange,
        *,
        lower_module: Callable[[CfgSchema, ParameterContext], ModuleSpec],
        lower_waveform: Callable[[CfgSchema, ParameterContext], WaveformSpec],
    ) -> None:
        """Apply the structure command using ContextControlPort's target policy."""
        self._context.apply_library_change(
            parameters,
            change,
            lower_module=lower_module,
            lower_waveform=lower_waveform,
        )

    def get_entry_info(self) -> EntryInfo | None:
        return self._context.get_entry_info()

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        return self._context.list_points(refresh=refresh)

    def use_point(self, request: PointUseRequest) -> SavedPoint:
        return self._context.use_point(request)

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        return self._context.new_point(request)

    def get_source_binding(self) -> ContextSourceBinding | None:
        return self._context.get_source_binding()

    def get_component_focus(self) -> ComponentFocus:
        return self._context.get_component_focus()

    def set_component_focus(self, name: str | None) -> ComponentFocus:
        return self._context.set_component_focus(name)

    def list_value_sources(self) -> tuple[ValueInfo, ...]:
        return self._context.list_value_sources()

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        return self._context.read_value_source(key, type_name)
