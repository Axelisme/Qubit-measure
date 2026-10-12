"""Declared native scenarios and observations for Window ownership contracts."""

from collections.abc import Callable

import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows import figures as figures_module
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import FormModel

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.coherence import CoherencePlan, CoherenceTunables

from ._reload_source import InjectedCatalog
from ._window_support import RpcRequest


def install_admission_observer(
    window: WorkflowWindow[Calibrated], catalog_source: InjectedCatalog
) -> list[str]:
    """Record init/step so cancellation before native acceptance proves neither ran."""
    calls: list[str] = []

    def init(_env: InitEnv[Calibrated], _plan: CoherencePlan) -> None:
        calls.append("init")

    @workflow(
        "admission-observer", plan=CoherencePlan, tunables=CoherenceTunables, init=init
    )
    def observe(
        env: WorkflowEnv[Calibrated],
        plan: CoherencePlan,
        tun: CoherenceTunables,
        state: None,
    ) -> Step[None, int]:
        del env, plan, tun, state
        calls.append("step")
        yield from ()
        return Done()

    catalog_source.install(observe, window)
    return calls


def admission_canceller(
    window: WorkflowWindow[Calibrated],
    action: str,
    old: TunablesHost[FormModel, Calibrated] | None,
    fixed: WorkflowController[FormModel, Calibrated],
) -> tuple[
    Callable[[], None],
    list[bool],
    list[tuple[str | None, int | None, bool, bool]],
    list[str],
]:
    """Observe the pending target and rejected Apply, then cancel through Stop/close."""
    close_results: list[bool] = []
    observed: list[tuple[str | None, int | None, bool, bool]] = []
    errors: list[str] = []

    def cancel_on_notification() -> None:
        binding = window.controller.binding
        observed.append(
            (
                window.controller.run_id,
                window.controller.operation_id,
                window.controller is fixed,
                binding is not None and binding is not old and binding.engine is None,
            )
        )
        try:
            window.apply_tunables(
                "notification", {("t1", "guess_us"): 55.0}, actor=Actor("agent", "rpc")
            )
        except RuntimeError as error:
            errors.append(str(error))
        if action == "close":
            close_results.append(window.close())
        else:
            window.stop("notification")

    return cancel_on_notification, close_results, observed, errors


def install_pause_plot_workflow(
    window: WorkflowWindow[Calibrated], catalog_source: InjectedCatalog
) -> tuple[list[float], list[int]]:
    """Pause after drawing; record fixed inputs and figure identity across Resume."""
    attempts: list[float] = []
    figures: list[int] = []

    def init(_env: InitEnv[Calibrated], _plan: CoherencePlan) -> None:
        return None

    @workflow("pause-window", plan=CoherencePlan, tunables=CoherenceTunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated],
        plan: CoherencePlan,
        tun: CoherenceTunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append(env.context.pi_length_us)

        def draw(figure: Figure) -> None:
            figures.append(id(figure))
            figure.clear()
            figure.subplots().plot([0, 1], [len(attempts), len(attempts)])

        if len(attempts) == 1:
            env.plots.subplots("live")
        env.plots.draw("live", draw)
        bar = env.pbar("work", 2)
        bar.set_progress(len(attempts))
        bar.refresh()
        if len(attempts) == 1:
            window.pause("paused")
            yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
    return attempts, figures


def install_reader_plot_workflow(
    window: WorkflowWindow[Calibrated], catalog_source: InjectedCatalog
) -> None:
    """Commit red then pause with blue so reads must expose the latest native plots."""
    attempts: list[int] = []

    def init(_env: InitEnv[Calibrated], _plan: CoherencePlan) -> int:
        return 0

    @workflow("read-plots", plan=CoherencePlan, tunables=CoherenceTunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated],
        plan: CoherencePlan,
        tun: CoherenceTunables,
        state: int,
    ) -> Step[int, int]:
        del plan, tun
        attempts.append(state)
        if state == 0:
            env.plots.subplots("live")
            env.plots.subplots("reference")
            env.plots.draw("live", lambda figure: figure.set_facecolor("red"))
            yield from ()
            return Next(1, 1)
        env.plots.draw("live", lambda figure: figure.set_facecolor("blue"))
        if len(attempts) == 2:
            window.pause("reader")
            yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)


def install_rpc_pause_workflow(
    window: WorkflowWindow[Calibrated],
    harness: StartHarness,
    rpc: RpcRequest,
    catalog_source: InjectedCatalog,
) -> tuple[list[float], list[dict[str, object]]]:
    """Request Pause during a step and retain receipts and fixed input observations."""
    attempts: list[float] = []
    receipts: list[dict[str, object]] = []

    def init(_env: InitEnv[Calibrated], _plan: CoherencePlan) -> None:
        return None

    @workflow("rpc-pause", plan=CoherencePlan, tunables=CoherenceTunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated],
        plan: CoherencePlan,
        tun: CoherenceTunables,
        state: None,
    ) -> Step[None, int]:
        del plan, tun, state
        attempts.append(env.context.pi_length_us)
        env.pbar("rpc-work", 2).set_progress(len(attempts))
        if len(attempts) == 1:
            receipts.append(rpc("workflow.pause", {"run_id": "rpc-run"}))
            assert harness.gate.snapshot()
            yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.install(step, window)
    return attempts, receipts


def assert_figure_render_failure(
    window: WorkflowWindow[Calibrated], rpc: RpcRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify RPC projects renderer failure while direct Window reads re-raise it."""

    def failed_render(_figure: Figure) -> bytes:
        raise RuntimeError("Renderer failed")

    monkeypatch.setattr(figures_module, "render_figure_png", failed_render)
    failed = rpc("workflow.figure", {"run_id": "reader", "name": "live"})
    error = failed["error"]
    assert failed["ok"] is False and isinstance(error, dict)
    assert error["code"] == "controller_error" and error["message"] == "Renderer failed"
    with pytest.raises(RuntimeError, match="Renderer failed"):
        window.figure_png("reader", "live")
