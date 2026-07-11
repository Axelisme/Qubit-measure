"""EventBus coordinator for the measure-gui main window."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, TypeVar

from zcu_tools.gui.app.main.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabAddedPayload,
    TabClosedPayload,
    TabContentChangedPayload,
    TabContentFact,
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
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
    def has_tab_widget(self, tab_id: str) -> bool: ...
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
    def clear_tab_plot(self, tab_id: str) -> None: ...
    def refresh_tab_interaction(
        self, tab_id: str, snapshot: TabSnapshot | None = None
    ) -> None: ...
    def refresh_run_lock(self, running_tab_id: str | None) -> None: ...
    def refresh_context_panel(self) -> None: ...
    def refresh_predictor_panel(self) -> None: ...
    def refresh_feedback_widget(self) -> None: ...


class _TabReaction(Enum):
    ANALYZE_FORM = auto()
    POST_ANALYZE_FORM = auto()
    WRITEBACK = auto()
    FIGURE = auto()
    POST_FIGURE = auto()
    INTERACTION = auto()
    FEEDBACK = auto()
    CLEAR_PLOT = auto()


_INTERACTION_REACTIONS: dict[TabInteractionFact, tuple[_TabReaction, ...]] = {
    TabInteractionFact.RUN_START_REJECTED: (
        _TabReaction.ANALYZE_FORM,
        _TabReaction.POST_ANALYZE_FORM,
        _TabReaction.WRITEBACK,
        _TabReaction.INTERACTION,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.PRIMARY_ANALYZE_STARTED: (
        _TabReaction.INTERACTION,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED: (
        _TabReaction.INTERACTION,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.PRIMARY_ANALYZE_FAILED: (
        _TabReaction.INTERACTION,
        _TabReaction.FIGURE,
        _TabReaction.POST_FIGURE,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.PRIMARY_ANALYZE_CANCELLED: (
        _TabReaction.INTERACTION,
        _TabReaction.FIGURE,
        _TabReaction.POST_FIGURE,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.PRIMARY_ANALYZE_START_REJECTED: (
        _TabReaction.INTERACTION,
        _TabReaction.FIGURE,
        _TabReaction.POST_FIGURE,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.POST_ANALYZE_STARTED: (
        _TabReaction.INTERACTION,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.POST_ANALYZE_SUCCEEDED: (
        _TabReaction.INTERACTION,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.POST_ANALYZE_FAILED: (
        _TabReaction.INTERACTION,
        _TabReaction.FIGURE,
        _TabReaction.POST_FIGURE,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.POST_ANALYZE_START_REJECTED: (
        _TabReaction.INTERACTION,
        _TabReaction.FIGURE,
        _TabReaction.POST_FIGURE,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.SAVE_STARTED: (
        _TabReaction.INTERACTION,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.SAVE_SUCCEEDED: (
        _TabReaction.INTERACTION,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.SAVE_FAILED: (
        _TabReaction.INTERACTION,
        _TabReaction.FEEDBACK,
    ),
    TabInteractionFact.ANALYZE_PARAMS_CHANGED: (),
    TabInteractionFact.POST_ANALYZE_PARAMS_CHANGED: (),
    TabInteractionFact.SAVE_PATHS_CHANGED: (),
}

_CONTENT_REACTIONS: dict[TabContentFact, tuple[_TabReaction, ...]] = {
    TabContentFact.RUN_RESULT_COMMITTED: (
        _TabReaction.ANALYZE_FORM,
        _TabReaction.POST_ANALYZE_FORM,
        _TabReaction.WRITEBACK,
        _TabReaction.INTERACTION,
    ),
    TabContentFact.LOADED_RESULT_COMMITTED: (
        _TabReaction.CLEAR_PLOT,
        _TabReaction.ANALYZE_FORM,
        _TabReaction.POST_ANALYZE_FORM,
        _TabReaction.WRITEBACK,
        _TabReaction.INTERACTION,
    ),
    TabContentFact.PRIMARY_ANALYSIS_COMMITTED: (
        _TabReaction.POST_ANALYZE_FORM,
        _TabReaction.WRITEBACK,
        _TabReaction.FIGURE,
        _TabReaction.INTERACTION,
    ),
    TabContentFact.POST_ANALYSIS_COMMITTED: (
        _TabReaction.FIGURE,
        _TabReaction.POST_FIGURE,
        _TabReaction.INTERACTION,
    ),
}


_FactT = TypeVar("_FactT", bound=Enum)


def _validate_reaction_matrix(
    name: str,
    fact_type: type[_FactT],
    matrix: Mapping[_FactT, object],
) -> None:
    """Fast-fail when a closed fact enum and its reaction matrix drift."""
    expected = set(fact_type)
    actual = set(matrix)
    missing = expected - actual
    extra = actual - expected
    if not missing and not extra:
        return

    def labels(values: Iterable[object]) -> list[str]:
        return sorted(
            str(value.value) if isinstance(value, Enum) else repr(value)
            for value in values
        )

    raise RuntimeError(
        f"{name} reaction matrix key mismatch: "
        f"missing={labels(missing)}, extra={labels(extra)}"
    )


_validate_reaction_matrix(
    "interaction",
    TabInteractionFact,
    _INTERACTION_REACTIONS,
)
_validate_reaction_matrix(
    "content",
    TabContentFact,
    _CONTENT_REACTIONS,
)


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
        self._react_to_tab(payload.tab_id, _INTERACTION_REACTIONS[payload.fact])

    def _on_run_started(self, payload: RunStartedPayload) -> None:
        self._react_to_tab(
            payload.tab_id,
            (
                _TabReaction.ANALYZE_FORM,
                _TabReaction.POST_ANALYZE_FORM,
                _TabReaction.WRITEBACK,
                _TabReaction.INTERACTION,
            ),
        )
        self._host.refresh_run_lock(payload.tab_id)
        self._host.refresh_feedback_widget()

    def _on_run_finished(self, payload: RunFinishedPayload) -> None:
        self._react_to_tab(payload.tab_id, (_TabReaction.INTERACTION,))
        self._host.refresh_run_lock(None)
        self._host.refresh_feedback_widget()
        if payload.outcome == "finished" and self._host.has_tab_widget(payload.tab_id):
            self._host.focus_run_result_panel(payload.tab_id)

    def _on_context_switched(self, payload: ContextSwitchedPayload) -> None:
        del payload
        self._host.refresh_context_panel()
        for tab_id in self._host.view_tab_ids():
            snapshot = self._ctrl.get_tab_snapshot(tab_id)
            self._host.refresh_tab_save_paths(tab_id, snapshot)
            self._host.refresh_tab_interaction(tab_id, snapshot)

    def _on_tab_added(self, payload: TabAddedPayload) -> None:
        self._host.add_tab_widget(payload.tab_id, payload.adapter_name)

    def _on_tab_closed(self, payload: TabClosedPayload) -> None:
        self._host.remove_tab_widget(payload.tab_id)
        self._host.refresh_run_lock(self._ctrl.get_running_tab_id())

    def _on_tab_content_changed(self, payload: TabContentChangedPayload) -> None:
        self._react_to_tab(payload.tab_id, _CONTENT_REACTIONS[payload.fact])

    def _react_to_tab(self, tab_id: str, reactions: tuple[_TabReaction, ...]) -> None:
        """Apply a fact's ordered View reactions with at most one snapshot read."""
        if not reactions or not self._host.has_tab_widget(tab_id):
            return
        snapshot = self._ctrl.get_tab_snapshot(tab_id)
        for reaction in reactions:
            if reaction is _TabReaction.ANALYZE_FORM:
                self._host.refresh_tab_analyze_form(tab_id, snapshot)
            elif reaction is _TabReaction.POST_ANALYZE_FORM:
                self._host.refresh_tab_post_analyze_form(tab_id, snapshot)
            elif reaction is _TabReaction.WRITEBACK:
                self._host.refresh_tab_writeback(tab_id, snapshot)
            elif reaction is _TabReaction.FIGURE:
                self._host.refresh_tab_figure(tab_id, snapshot)
            elif reaction is _TabReaction.POST_FIGURE:
                self._host.refresh_tab_post_figure(tab_id, snapshot)
            elif reaction is _TabReaction.INTERACTION:
                self._host.refresh_tab_interaction(tab_id, snapshot)
            elif reaction is _TabReaction.FEEDBACK:
                self._host.refresh_feedback_widget()
            elif reaction is _TabReaction.CLEAR_PLOT:
                self._host.clear_tab_plot(tab_id)

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
