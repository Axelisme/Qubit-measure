"""Narrow fixed-read, entry/point and focus surface for shared consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from zcu_tools.cfg_model import ModuleSpec, WaveformSpec
    from zcu_tools.format_version import YamlValue
    from zcu_tools.gui.session.context_types import (
        ComponentFocus,
        ContextSourceBinding,
        ContextTarget,
        EntryInfo,
        PointCreateRequest,
        PointUseRequest,
        SavedPoint,
        WorkingPointInfo,
    )
    from zcu_tools.gui.session.services.context import ContextService
    from zcu_tools.gui.session.types import ParameterContext, RoleBinding
    from zcu_tools.gui.session.value_lookup import ScalarValue, ValueInfo
    from zcu_tools.resources.entry.provenance import Provenance


class ContextControlPort(Protocol):
    """Cached reads and owner-thread commands using ContextService's contract.

    Reads never refresh implicitly. read_context rejects EMPTY/unavailable
    sources, preserving fixed supplied role names. Point commands validate the
    opened entry UUID before resource preparation. Preparation failures preserve
    the binding; a created-but-unbound point raises PointBindingError. Focus
    commands do not edit consumer roles or bump parameter revisions.
    """

    def read_context(self, *, roles: RoleBinding | None = None) -> ParameterContext:
        """Capture published values; EMPTY/source/role errors propagate."""
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

    def read_context(self, *, roles: RoleBinding | None = None) -> ParameterContext:
        return self._context.read_context(roles=roles)

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
