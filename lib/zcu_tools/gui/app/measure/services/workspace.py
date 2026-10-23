from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import JsonValue, TypeAdapter

from zcu_tools.gui.app.measure.events.tab import (
    TabActivatedPayload,
    TabAddedPayload,
    TabClosedPayload,
)
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus

from .ports import RestoreDroppedInputs, RestoreIssue, RestoreReport
from .tab_input import TabInput
from .workspace_types import WorkspaceSnapshot, WorkspaceTab

if TYPE_CHECKING:
    from .ports import TabLifecyclePort


class WorkspaceService:
    """Own tab lifecycle and RAM-only tab capture/apply for live catalog reload.

    Capture preserves the shared FormInput JSON including explicit selections.
    Tab resources own recursive restoration against a new source capture.
    This service does not save files or restore tabs at process startup.
    """

    def __init__(
        self,
        state: State,
        tabs: TabLifecyclePort,
        bus: EventBus,
    ) -> None:
        self._state = state
        self._tabs = tabs
        self._bus = bus

    def new_tab(
        self,
        adapter_name: str,
        *,
        input: TabInput | None = None,
        allow_factory_failure: bool = False,
    ) -> str:
        """Create, activate and announce a tab using current global sources.

        Owner-only; input is optional opaque authored Form JSON. The tab/form
        owner restores compatible paths, logs discards and rejects malformed JSON
        before publication. Caller retains input, GUI owns the new tab. Source or
        default-cfg failure publishes no tab fact. Only open_tab_from_file sets
        allow_factory_failure to retain a history tab without a Form; template
        binding and defaults still fail before publication.
        """
        if allow_factory_failure:
            tab_id = self._tabs.new_tab(adapter_name, allow_factory_failure=True)
        else:
            tab_id = (
                self._tabs.new_tab(adapter_name)
                if input is None
                else self._tabs.new_tab(adapter_name, input=input)
            )
        self._state.set_active_tab(tab_id)
        self._bus.emit(
            TabAddedPayload(tab_id=tab_id, adapter_name=adapter_name),
        )
        return tab_id

    def close_tab(self, tab_id: str) -> None:
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError("Cannot close a busy tab")
        self._tabs.close_tab(tab_id)
        self._bus.emit(TabClosedPayload(tab_id=tab_id))

    def set_active_tab(self, tab_id: str) -> None:
        previous = self._state.active_tab_id
        self._state.set_active_tab(tab_id)
        if previous != tab_id:
            self._bus.emit(TabActivatedPayload(tab_id=tab_id))

    def reorder_tabs(self, tab_ids: Sequence[str]) -> None:
        self._state.reorder_tabs(tab_ids)

    def capture_session(self) -> WorkspaceSnapshot:
        """Snapshot authored Form input, including unfinished expressions, without I/O.

        Call on the owner thread. The returned RAM copy belongs to the caller.
        Sources, alias overrides, result models and figures are not included.
        """
        tabs = list(self._state.tabs.items())
        tab_ids = [tab_id for tab_id, _ in tabs]
        active_tab_index = (
            tab_ids.index(self._state.active_tab_id)
            if self._state.active_tab_id in tab_ids
            else None
        )
        payload_tabs = tuple(
            WorkspaceTab(
                adapter_name=tab.adapter_name,
                form_input=TypeAdapter(dict[str, JsonValue]).validate_json(
                    tab.cfg.input.to_json()
                ),
            )
            for _, tab in tabs
        )
        return WorkspaceSnapshot(tabs=payload_tabs, active_tab_index=active_tab_index)

    def apply_session(self, session: WorkspaceSnapshot) -> RestoreReport:
        """Create fresh resources and recursively restore shared FormInput JSON.

        Per-tab preparation failures are rejected_tabs; other tabs still restore.
        Successfully restored tabs report discarded raw/selection paths through dropped_inputs.
        Call on the owner thread for live reload, not process startup. Ordinary
        unfinished/invalid input remains editable. The caller owns the detached
        snapshot; this command does not read or write files.
        """
        restored_by_index: dict[int, str] = {}
        rejected: list[RestoreIssue] = []
        dropped: list[RestoreDroppedInputs] = []
        for index, persisted_tab in enumerate(session.tabs):
            try:
                tab_id = self._tabs.new_tab(
                    persisted_tab.adapter_name, from_dict=persisted_tab
                )
            except (KeyError, ValueError, TypeError) as exc:
                rejected.append(
                    RestoreIssue(
                        persisted_tab.adapter_name, f"Cannot restore form ({exc})"
                    )
                )
                continue
            paths = self._tabs.dropped_input_paths(tab_id)
            if paths:
                dropped.append(RestoreDroppedInputs(persisted_tab.adapter_name, paths))
            restored_by_index[index] = tab_id
            self._bus.emit(
                TabAddedPayload(
                    tab_id=tab_id,
                    adapter_name=persisted_tab.adapter_name,
                ),
            )

        if session.active_tab_index is not None:
            active_tab_id = restored_by_index.get(session.active_tab_index)
            if active_tab_id is not None:
                self._state.set_active_tab(active_tab_id)
        return RestoreReport(
            restored_tabs=len(restored_by_index),
            rejected_tabs=tuple(rejected),
            dropped_inputs=tuple(dropped),
        )
