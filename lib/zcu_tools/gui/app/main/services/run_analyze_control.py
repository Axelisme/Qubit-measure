"""App-facing run/analyze control facet for driving adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.app.main.adapter import AnalysisMode, AnalyzeRequest
from zcu_tools.gui.app.main.events.tab import TabContentChangedPayload

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import InteractiveHost, InteractiveSession
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.plotting import FigureContainer

    from .analyze import AnalyzeService
    from .guard import AnalyzePermit, GuardService
    from .load import LoadService, LoadTabResultOutcome
    from .ports import TabSnapshot
    from .post_analyze import PostAnalyzeService
    from .run import RunService
    from .tab import TabService


class RunAnalyzeRenderHost(Protocol):
    """Render surface needed by run/analyze operations."""

    def make_live_container(self, tab_id: str) -> FigureContainer | None: ...

    def mount_interactive_analysis(
        self,
        tab_id: str,
        session_factory: Callable[[InteractiveHost], InteractiveSession],
        on_finish: Callable[[InteractiveSession], None],
    ) -> None: ...

    def unmount_interactive_analysis(self, tab_id: str) -> None: ...


class RunAnalyzeControlPort(Protocol):
    """App-facing run/load/analyze operation surface for driving adapters."""

    def has_tab(self, tab_id: str) -> bool: ...
    def get_running_tab_id(self) -> str | None: ...
    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot: ...

    def start_run(self, tab_id: str) -> int: ...
    def load_tab_result(self, tab_id: str, data_path: str) -> LoadTabResultOutcome: ...
    def cancel_run(self) -> bool: ...

    def cancel_analyze(self, tab_id: str) -> bool: ...
    def get_tab_analyze_result(self, tab_id: str) -> object | None: ...
    def analyze(self, tab_id: str, analyze_params_instance: object) -> int: ...

    def start_post_analyze(
        self, tab_id: str, post_analyze_params_instance: object
    ) -> int: ...
    def get_post_analyze_result(self, tab_id: str) -> object | None: ...


class RunAnalyzeControlFacet:
    """Composite adapter over run/load/analyze services."""

    def __init__(
        self,
        *,
        state: State,
        bus: EventBus,
        guard: GuardService,
        tab: TabService,
        load: LoadService,
        run: RunService,
        analyze: AnalyzeService,
        post_analyze: PostAnalyzeService,
        render_host: Callable[[], RunAnalyzeRenderHost | None],
    ) -> None:
        self._state = state
        self._bus = bus
        self._guard = guard
        self._tab = tab
        self._load = load
        self._run = run
        self._analyze = analyze
        self._post_analyze = post_analyze
        self._render_host = render_host

    def has_tab(self, tab_id: str) -> bool:
        return self._state.has_tab(tab_id)

    def get_running_tab_id(self) -> str | None:
        return self._state.running_tab_id

    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot:
        return self._tab.get_snapshot(tab_id)

    def start_run(self, tab_id: str) -> int:
        permit = self._guard.acquire_run_permit(tab_id)
        host = self._render_host()
        live_container = host.make_live_container(tab_id) if host is not None else None
        return self._run.start_run(permit, live_container)

    def load_tab_result(self, tab_id: str, data_path: str) -> LoadTabResultOutcome:
        permit = self._guard.acquire_load_permit(tab_id)
        outcome = self._load.load_result(permit, data_path)
        tab = self._state.get_tab(tab_id)
        has_analyze_params = False
        if tab.adapter.capabilities.analysis is not AnalysisMode.NONE:
            self._tab.initialize_tab_analyze_params(tab_id)
            has_analyze_params = True
        self._bus.emit(TabContentChangedPayload(tab_id=tab_id))
        return replace(outcome, has_analyze_params=has_analyze_params)

    def cancel_run(self) -> bool:
        return self._run.cancel_run()

    def cancel_analyze(self, tab_id: str) -> bool:
        host = self._render_host()
        if host is not None:
            host.unmount_interactive_analysis(tab_id)
        return self._analyze.cancel_interactive(tab_id)

    def get_tab_analyze_result(self, tab_id: str) -> object | None:
        return self._tab.get_tab_analyze_result(tab_id)

    def analyze(self, tab_id: str, analyze_params_instance: object) -> int:
        permit = self._guard.acquire_analyze_permit(tab_id)
        tab = self._state.get_tab(tab_id)
        if tab.adapter.capabilities.analysis is AnalysisMode.INTERACTIVE:
            return self._start_interactive_analyze(
                tab_id, permit, analyze_params_instance
            )
        host = self._render_host()
        figure_container = (
            host.make_live_container(tab_id) if host is not None else None
        )
        return self._analyze.start_analyze(
            permit, analyze_params_instance, figure_container
        )

    def _start_interactive_analyze(
        self, tab_id: str, permit: AnalyzePermit, analyze_params_instance: object
    ) -> int:
        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        req = AnalyzeRequest(
            run_result=tab.run_result,
            analyze_params=analyze_params_instance,
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
        token = self._analyze.start_interactive(permit)
        host = self._render_host()
        if host is not None:
            host.mount_interactive_analysis(
                tab_id,
                lambda ihost: tab.adapter.setup_interactive_analysis(req, ihost),
                lambda session: self._analyze.finish_interactive(tab_id, session),
            )
        return token

    def start_post_analyze(
        self, tab_id: str, post_analyze_params_instance: object
    ) -> int:
        host = self._render_host()
        figure_container = (
            host.make_live_container(tab_id) if host is not None else None
        )
        return self._post_analyze.start_post_analyze(
            tab_id, post_analyze_params_instance, figure_container
        )

    def get_post_analyze_result(self, tab_id: str) -> object | None:
        return self._tab.get_tab_post_analyze_result(tab_id)
