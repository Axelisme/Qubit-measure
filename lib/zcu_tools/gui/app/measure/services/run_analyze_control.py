"""App-facing run/analyze control facet for driving adapters."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol

from zcu_tools.gui.app.measure.adapter import AnalysisMode, AnalyzeRequest
from zcu_tools.gui.app.measure.catalog import ExperimentAccess
from zcu_tools.gui.app.measure.events.tab import (
    TabContentChangedPayload,
    TabContentFact,
)
from zcu_tools.gui.cfg.resource import CfgRef, CfgStaleError
from zcu_tools.gui.expected_error import FailedPreconditionError

from .plot_lifecycle import discard_unpublished_plots

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.app.measure.ui.interactive_frontend import (
        InteractiveFrontend,
        InteractiveFrontendEnv,
    )
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.plotting import FigureContainer
    from zcu_tools.gui.session.ports import OwnerScheduler
    from zcu_tools.plotting.plots import Plots

    from .analyze import ActiveInteractive, AnalyzeService
    from .guard import AnalyzePermit, GuardService
    from .load import LoadService, LoadTabResultOutcome
    from .ports import TabSnapshot
    from .post_analyze import PostAnalyzeService
    from .run import RunService
    from .tab import TabService


class RunAnalyzeRenderHost(Protocol):
    """Render surface needed by run/analyze operations (per-pane, S2)."""

    def make_run_container(self, tab_id: str) -> FigureContainer | None: ...

    def make_analysis_container(self, tab_id: str) -> FigureContainer | None: ...

    def make_post_analysis_container(self, tab_id: str) -> FigureContainer | None: ...

    def mount_interactive_analysis(
        self,
        tab_id: str,
        frontend_factory: Callable[[InteractiveFrontendEnv], InteractiveFrontend],
    ) -> None: ...

    def unmount_interactive_analysis(
        self, tab_id: str, *, restore_result: bool = False
    ) -> None: ...

    def discard_interactive_preview(self, tab_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ActiveTabOperation:
    op: int
    tab: str
    kind: Literal["run", "analyze"]


class RunAnalyzeControlPort(Protocol):
    """App-facing run/load/analyze operation surface for driving adapters."""

    def has_tab(self, tab_id: str) -> bool: ...
    def get_running_tab_id(self) -> str | None: ...
    def active_tab_operations(self) -> tuple[ActiveTabOperation, ...]:
        """Domain-admitted handles; busy may also include a start reservation."""
        ...

    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot: ...

    def start_run(self, tab_id: str, expected: CfgRef) -> int: ...
    def load_tab_result(self, tab_id: str, data_path: str) -> LoadTabResultOutcome: ...
    def cancel_run(self) -> bool: ...

    def cancel_analyze(self, tab_id: str) -> bool: ...
    def get_tab_analyze_result(self, tab_id: str) -> object | None: ...
    def analyze(self, tab_id: str, analyze_params_instance: object) -> int: ...
    def get_interactive(self, tab_id: str) -> ActiveInteractive | None: ...
    def finish_interactive(self, tab_id: str) -> bool: ...

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
        owner_scheduler: OwnerScheduler,
        run_background: Callable[
            [
                Callable[[], object],
                Callable[[object], None],
                Callable[[Exception], None],
            ],
            None,
        ]
        | None = None,
        access: ExperimentAccess | None = None,
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
        self._owner_scheduler = owner_scheduler
        self._run_background = run_background
        self._access = access if access is not None else ExperimentAccess()

    def has_tab(self, tab_id: str) -> bool:
        return self._state.has_tab(tab_id)

    def get_running_tab_id(self) -> str | None:
        return self._state.running_tab_id

    def active_tab_operations(self) -> tuple[ActiveTabOperation, ...]:
        """Read admitted handles, including during synchronous startup notifications."""
        operations: list[ActiveTabOperation] = []
        running = self._state.running_tab_id
        # State also reserves busy during start. Run publishes its domain handle
        # only after begin succeeds; registration and failed-submit cleanup can
        # synchronously notify before that transfer. Reads omit the reservation.
        token = self._run.active_token
        if running is not None and token is not None:
            operations.append(ActiveTabOperation(token, running, "run"))
        operations.extend(
            ActiveTabOperation(token, tab, "analyze")
            for service in (self._analyze, self._post_analyze)
            for tab, token in service.active_operations()
        )
        return tuple(sorted(operations, key=lambda op: op.op))

    def get_tab_snapshot(self, tab_id: str) -> TabSnapshot:
        return self._tab.get_snapshot(tab_id)

    def start_run(self, tab_id: str, expected: CfgRef) -> int:
        """Accept exactly the caller's publication, without refresh or substitution."""
        self._access.require_available()
        actual = self._state.get_tab(tab_id).cfg.observe().ref
        if expected != actual:
            raise CfgStaleError(expected, actual)
        permit = self._guard.acquire_run_permit(
            tab_id, expected_revision=expected.revision
        )
        self._ensure_tab_idle(tab_id)
        host = self._render_host()
        live_container = host.make_run_container(tab_id) if host is not None else None
        return self._run.start_run(permit, plots=self._new_plots(live_container))

    def load_tab_result(self, tab_id: str, data_path: str) -> LoadTabResultOutcome:
        self._access.require_available()
        permit = self._guard.acquire_load_permit(tab_id)
        outcome = self._load.load_result(permit, data_path)
        self._run.release_view_plots(tab_id)
        tab = self._state.get_tab(tab_id)
        has_analyze_params = False
        if tab.adapter.capabilities.analysis is not AnalysisMode.NONE:
            self._tab.initialize_tab_analyze_params(tab_id)
            has_analyze_params = True
        preparation = self._tab.prepare_result_analysis(tab_id)
        self._bus.emit(
            TabContentChangedPayload(
                tab_id=tab_id,
                fact=TabContentFact.LOADED_RESULT_COMMITTED,
            )
        )
        return replace(
            outcome,
            has_analyze_params=preparation.has_params,
            analysis_error=preparation.error,
        )

    def cancel_run(self) -> bool:
        return self._run.cancel_run()

    def cancel_analyze(self, tab_id: str) -> bool:
        host = self._render_host()
        if host is not None:
            host.unmount_interactive_analysis(tab_id, restore_result=True)
        return self._analyze.cancel_interactive(tab_id)

    def get_tab_analyze_result(self, tab_id: str) -> object | None:
        return self._tab.get_tab_analyze_result(tab_id)

    def get_interactive(self, tab_id: str) -> ActiveInteractive | None:
        return self._analyze.get_interactive(tab_id)

    def finish_interactive(self, tab_id: str) -> bool:
        active = self._analyze.get_interactive(tab_id)
        if active is None:
            raise FailedPreconditionError(
                f"tab {tab_id!r} has no active interactive analysis"
            )
        host = self._render_host()
        if host is not None:
            host.discard_interactive_preview(tab_id)
            # Only unmount after validation; failure leaves the frontend editable.
            # The plugin creates committed figures separately from this preview.
            active.plugin.can_finish(active.session.snapshot())
            host.unmount_interactive_analysis(tab_id)
        terminal = self._analyze.finish_plugin(tab_id)
        if terminal and host is not None:
            host.unmount_interactive_analysis(tab_id, restore_result=True)
        return terminal

    def analyze(self, tab_id: str, analyze_params_instance: object) -> int:
        self._access.require_available()
        permit = self._guard.acquire_analyze_permit(tab_id)
        self._ensure_tab_idle(tab_id)
        tab = self._state.get_tab(tab_id)
        if tab.adapter.capabilities.analysis is AnalysisMode.INTERACTIVE:
            return self._start_interactive_analyze(
                tab_id, permit, analyze_params_instance
            )
        host = self._render_host()
        figure_container = (
            host.make_analysis_container(tab_id) if host is not None else None
        )
        return self._analyze.start_analyze(
            permit, analyze_params_instance, plots=self._new_plots(figure_container)
        )

    def _start_interactive_analyze(
        self, tab_id: str, permit: AnalyzePermit, analyze_params_instance: object
    ) -> int:
        tab = self._state.get_tab(tab_id)
        ctx = self._state.session_env
        req = AnalyzeRequest(
            run_result=tab.run.result,
            analyze_params=analyze_params_instance,
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
        host = self._render_host()
        if host is None:
            raise FailedPreconditionError(
                "interactive analysis requires an attached render host"
            )
        plots = self._new_plots(None)
        try:
            plugin = tab.adapter.make_interactive_plugin(req, plots=plots)
            if self._run_background is not None:
                plugin.bind_background(self._run_background)
            token = self._analyze.start_plugin(
                permit,
                plugin,
                self._owner_scheduler,
                analyze_params_instance=analyze_params_instance,
                plots=plots,
            )
        except Exception:
            try:
                discard_unpublished_plots(plots)
            except Exception:
                logger.exception(
                    "interactive setup plot cleanup failed: tab_id=%r", tab_id
                )
            raise
        try:
            active = self._analyze.get_interactive(tab_id)
            if active is None:
                raise RuntimeError("interactive operation has no service-owned session")
            host.mount_interactive_analysis(
                tab_id,
                lambda env: tab.adapter.make_interactive_frontend(
                    plugin,
                    active.session,
                    env,
                    lambda: self.finish_interactive(tab_id),
                    lambda: self.cancel_analyze(tab_id),
                    plots=active.plots,
                ),
            )
        except Exception:
            try:
                host.unmount_interactive_analysis(tab_id, restore_result=True)
            except Exception:
                logger.exception(
                    "failed to unmount interactive analysis after setup failure: tab_id=%r",
                    tab_id,
                )
            if not self._analyze.cancel_interactive(tab_id):
                logger.error(
                    "interactive analysis setup failed without an active operation: tab_id=%r",
                    tab_id,
                )
            raise
        return token

    def start_post_analyze(
        self, tab_id: str, post_analyze_params_instance: object
    ) -> int:
        self._ensure_tab_idle(tab_id)
        host = self._render_host()
        figure_container = (
            host.make_post_analysis_container(tab_id) if host is not None else None
        )
        return self._post_analyze.start_post_analyze(
            tab_id,
            post_analyze_params_instance,
            plots=self._new_plots(figure_container),
        )

    def _new_plots(self, container: FigureContainer | None) -> Plots:
        from zcu_tools.gui.plotting.explicit import QtPlotHost
        from zcu_tools.plotting.plots import NonPresentingHost, Plots

        host = (
            NonPresentingHost()
            if container is None
            else QtPlotHost(container, self._owner_scheduler)
        )
        return Plots(host)

    def _ensure_tab_idle(self, tab_id: str) -> None:
        self._access.require_available()
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

    def get_post_analyze_result(self, tab_id: str) -> object | None:
        return self._tab.get_tab_post_analyze_result(tab_id)
