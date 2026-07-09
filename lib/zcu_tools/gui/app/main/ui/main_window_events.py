"""EventBus coordinator for the measure-gui main window."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.app.main.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabAddedPayload,
    TabClosedPayload,
    TabContentChangedPayload,
    TabInteractionChangedPayload,
)
from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
    MlChangedPayload,
    PredictorChangedPayload,
    SocChangedPayload,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.services import TabSnapshot
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus


class MainWindowEventHost(Protocol):
    """Narrow view operations used by the main-window event coordinator."""

    def add_tab_widget(self, tab_id: str, adapter_name: str) -> None: ...
    def remove_tab_widget(self, tab_id: str) -> None: ...
    def view_tab_ids(self) -> list[str]: ...
    def focus_run_result_panel(self, tab_id: str) -> None: ...

    def refresh_tab_analyze_form(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None: ...
    def refresh_tab_post_analyze_form(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None: ...
    def refresh_tab_writeback(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None: ...
    def refresh_tab_save_paths(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None: ...
    def refresh_tab_figure(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None: ...
    def refresh_tab_post_figure(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None: ...
    def refresh_tab_interaction(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None: ...
    def refresh_run_lock(self, running_tab_id: str | None) -> None: ...
    def refresh_context_panel(self) -> None: ...
    def refresh_predictor_panel(self) -> None: ...
    def refresh_feedback_widget(self) -> None: ...


class MainWindowEventCoordinator:
    """Owns main-window EventBus subscriptions and payload routing."""

    def __init__(self, ctrl: Controller, host: MainWindowEventHost) -> None:
        self._ctrl = ctrl
        self._host = host
        self._subs = EventSubscriptions()

    def bind(self, bus: EventBus) -> None:
        """Subscribe all main-window bus handlers to ``bus``."""
        self._subs.subscribe(
            bus, TabInteractionChangedPayload, self._on_tab_interaction_changed
        )
        self._subs.subscribe(bus, RunStartedPayload, self._on_run_started)
        self._subs.subscribe(bus, RunFinishedPayload, self._on_run_finished)
        self._subs.subscribe(bus, ContextSwitchedPayload, self._on_context_switched)
        self._subs.subscribe(bus, MlChangedPayload, self._on_ml_changed)
        self._subs.subscribe(bus, TabAddedPayload, self._on_tab_added)
        self._subs.subscribe(bus, TabClosedPayload, self._on_tab_closed)
        self._subs.subscribe(
            bus, TabContentChangedPayload, self._on_tab_content_changed
        )
        self._subs.subscribe(bus, PredictorChangedPayload, self._on_predictor_changed)
        self._subs.subscribe(bus, SocChangedPayload, self._on_soc_changed)
        self._subs.subscribe(
            bus, DeviceSetupStartedPayload, self._on_device_setup_started
        )
        self._subs.subscribe(
            bus, DeviceSetupFinishedPayload, self._on_device_setup_finished
        )
        self._subs.subscribe(bus, DeviceChangedPayload, self._on_device_changed)

    def close(self) -> None:
        """Unsubscribe every main-window bus handler."""
        self._subs.unsubscribe_all()

    def _on_tab_interaction_changed(
        self, payload: TabInteractionChangedPayload
    ) -> None:
        tab_id = payload.tab_id
        snapshot = self._ctrl.get_tab_snapshot(tab_id)
        self._host.refresh_tab_writeback(tab_id, snapshot)
        self._host.refresh_tab_interaction(tab_id, snapshot)
        interaction = snapshot.interaction
        if interaction is not None and not (
            interaction.is_running
            or interaction.is_analyzing
            or interaction.is_saving_data
        ):
            if interaction.has_analyze_result:
                self._host.refresh_tab_figure(tab_id, snapshot)
            if interaction.has_post_analyze_result:
                self._host.refresh_tab_post_figure(tab_id, snapshot)
        self._host.refresh_feedback_widget()

    def _on_run_started(self, payload: RunStartedPayload) -> None:
        self._host.refresh_run_lock(payload.tab_id)
        self._host.refresh_feedback_widget()

    def _on_run_finished(self, payload: RunFinishedPayload) -> None:
        self._host.refresh_run_lock(None)
        self._host.refresh_feedback_widget()
        if payload.outcome == "finished":
            self._host.focus_run_result_panel(payload.tab_id)

    def _on_context_switched(self, payload: ContextSwitchedPayload) -> None:
        del payload
        self._host.refresh_context_panel()
        for tab_id in self._host.view_tab_ids():
            snapshot = self._ctrl.get_tab_snapshot(tab_id)
            self._host.refresh_tab_writeback(tab_id, snapshot)
            self._host.refresh_tab_save_paths(tab_id, snapshot)
            self._host.refresh_tab_interaction(tab_id, snapshot)

    def _on_ml_changed(self, payload: MlChangedPayload) -> None:
        del payload
        for tab_id in self._host.view_tab_ids():
            snapshot = self._ctrl.get_tab_snapshot(tab_id)
            self._host.refresh_tab_writeback(tab_id, snapshot)
            self._host.refresh_tab_interaction(tab_id, snapshot)

    def _on_tab_added(self, payload: TabAddedPayload) -> None:
        self._host.add_tab_widget(payload.tab_id, payload.adapter_name)

    def _on_tab_closed(self, payload: TabClosedPayload) -> None:
        self._host.remove_tab_widget(payload.tab_id)
        self._host.refresh_run_lock(self._ctrl.get_running_tab_id())

    def _on_tab_content_changed(self, payload: TabContentChangedPayload) -> None:
        tab_id = payload.tab_id
        snapshot = self._ctrl.get_tab_snapshot(tab_id)
        self._host.refresh_tab_analyze_form(tab_id, snapshot)
        self._host.refresh_tab_post_analyze_form(tab_id, snapshot)
        self._host.refresh_tab_writeback(tab_id, snapshot)
        self._host.refresh_tab_save_paths(tab_id, snapshot)
        self._host.refresh_tab_figure(tab_id, snapshot)
        self._host.refresh_tab_post_figure(tab_id, snapshot)
        self._host.refresh_tab_interaction(tab_id, snapshot)
        self._host.refresh_feedback_widget()

    def _on_predictor_changed(self, payload: PredictorChangedPayload) -> None:
        del payload
        self._host.refresh_predictor_panel()

    def _on_soc_changed(self, payload: SocChangedPayload) -> None:
        del payload
        self._host.refresh_run_lock(self._ctrl.get_running_tab_id())
        self._host.refresh_feedback_widget()

    def _on_device_setup_started(self, payload: DeviceSetupStartedPayload) -> None:
        del payload
        self._host.refresh_feedback_widget()

    def _on_device_setup_finished(self, payload: DeviceSetupFinishedPayload) -> None:
        del payload
        self._host.refresh_feedback_widget()

    def _on_device_changed(self, payload: DeviceChangedPayload) -> None:
        del payload
        self._host.refresh_feedback_widget()


__all__ = ["MainWindowEventCoordinator", "MainWindowEventHost"]
