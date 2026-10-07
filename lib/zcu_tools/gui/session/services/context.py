"""Owner-thread entry/point bindings and fixed session parameter reads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.gui.owner import OwnerThreadGuard
from zcu_tools.gui.session.context_types import ComponentFocus
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ScalarValue,
    ValueInfo,
    ValueLookup,
    ValueRef,
    resolve_value_ref,
)

if TYPE_CHECKING:
    from zcu_tools.cfg_model import WaveformAssetInfo
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_types import (
        ContextSourceBinding,
        EntryInfo,
        PointCreateRequest,
        PointUseRequest,
        SavedPoint,
        WorkingPointInfo,
    )
    from zcu_tools.gui.session.state import SessionState
    from zcu_tools.gui.session.types import ParameterContext, RoleBinding, SessionEnv
    from zcu_tools.resources.entry import ResultEntry, RoleSpec
    from zcu_tools.resources.entry.ledger import RecordsLedger


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
        if state.session_env.values is not self._values:
            state.set_context(replace(state.session_env, values=self._values))

    def install_entry(self, entry: ResultEntry) -> EntryInfo:
        """Prepare an opened entry and atomically publish an empty DRAFT.

        Prepare setup, the complete point-description cache, and entry-scoped
        assets before replacing any binding. Failure retains the old binding,
        not deleting an entry already created by resources. Success bumps the
        context once and publishes switch/focus facts after installation.
        """
        raise NotImplementedError("Entry installation is not implemented")

    def get_source_binding(self) -> ContextSourceBinding | None:
        """Return detached cached persistent sources, or None for EMPTY.

        This internal source ID is for facts/refresh, never part of read targets.
        Query performs no I/O and does not select a point.
        """
        raise NotImplementedError("Source binding query is not implemented")

    def get_entry_info(self) -> EntryInfo | None:
        """Return the cached opened entry, or None for EMPTY; never refresh."""
        raise NotImplementedError("Entry query is not implemented")

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        """Return cached label/descriptions, empty without an opened entry.

        refresh=True completely prepares a fresh list before replacing cache.
        Read/validation failures propagate, preserving the previous cache.
        """
        raise NotImplementedError("Point listing is not implemented")

    def use_point(self, request: PointUseRequest) -> SavedPoint:
        """Prepare point/library/assets, then install ACTIVE and publish facts.

        No opened entry raises FailedPreconditionError(no_entry); a different
        entry_id raises entry_not_open. Preparation failure retains the old
        binding. The returned target has no session selection generation.
        """
        raise NotImplementedError("Point selection is not implemented")

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        """Create from setup or a same-entry clone, then install ACTIVE.

        Validate opened entry UUID before creation. Pass description and origin
        unchanged to ResultEntry.new_point; never copy DRAFT. Resource errors
        propagate. Post-creation binding failure raises PointBindingError from
        its cause, retaining the point and old binding without retry/deletion.
        """
        raise NotImplementedError("Point creation is not implemented")

    def clear_context(self) -> None:
        """Publish EMPTY, clear entry/point/focus, retaining non-parameter handles.

        Publish switch/focus facts after installation; do not delete resources.
        """
        raise NotImplementedError("Context clearing is not implemented")

    def read_context(self, *, roles: RoleBinding | None = None) -> ParameterContext:
        """Capture published point/library on the owner sequence without I/O.

        roles=None allows direct component/general paths only. Supplied names
        stay fixed; missing/wrong-kind destinations raise RoleResolutionError.
        EMPTY raises FailedPreconditionError(no_context). Necessary source
        faults raise ContextSourceUnavailableError, never returning old values.
        """
        raise NotImplementedError("Context capture is not implemented")

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
        raise NotImplementedError("Role capture is not implemented")

    def entry_ledger(self, entry_id: str) -> RecordsLedger:
        """Return only the currently opened entry's ledger; never append an event.

        No entry raises no_entry; a different UUID raises entry_not_open.
        The live capability is not part of ParameterContext or wire data.
        """
        raise NotImplementedError("Entry ledger query is not implemented")

    def get_component_focus(self) -> ComponentFocus:
        """Return cached target/name/component options without I/O or refresh."""
        raise NotImplementedError("Component focus query is not implemented")

    def set_component_focus(self, name: str | None) -> ComponentFocus:
        """Select an existing component or clear focus; reject unknown names.

        Never change consumer roles or context/cfg revision. Publish a focus
        fact only for a changed selection. None does not trigger auto-selection.
        """
        raise NotImplementedError("Component focus selection is not implemented")

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
