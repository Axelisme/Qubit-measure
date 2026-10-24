"""Plan editing and figure availability through registered synthetic workflows."""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import Live, WorkflowEnv, workflow
from zcu_tools.gui.app.measure.adapter import RunRequest
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.workflow_attachment import WorkflowAttachment
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form import Display, FormModel
from zcu_tools.gui.form_adapter import Form, FormContext, LinRange, SweepInput
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._operation_plots import attachment as plot_attachment
from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg, SampleRun
from tests.gui.app.measure._workflow_window import toggle_tab_family


class Plan(FormModel):
    fluxes: list[float] = Field(default_factory=lambda: [0.0])


class SweepPlan(FormModel):
    fluxes: Annotated[SweepInput, Display(label="Fluxes", selector=False)] = Field(
        default_factory=lambda: LinRange(start=0, stop=1, n=3)
    )


class RequiredSweepPlan(FormModel):
    fluxes: Annotated[SweepInput, Display(label="Fluxes", selector=False)] = Field(
        description="Ordered absolute device setpoints"
    )


class FluxAdapter(SampleAdapter):
    def run(self, req: RunRequest, cfg: SampleCfg, *, context: RunContext) -> SampleRun:
        values = np.linspace(cfg.sweep.start, cfg.sweep.stop, cfg.sweep.expts)
        viewer = context.plots.liveplot_1d(
            "flux trace", "Flux", f"fluxes={values.tolist()}"
        )
        viewer.update(values, values)
        return super().run(req, cfg, context=context)


class RequiredFluxAdapter(FluxAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[SampleCfg]:
        del ctx
        return Form(SampleCfg, {})


class Tunables(FormModel):
    gain: float = 1.0


class NativePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    fluxes: list[float]


class NativeTunables(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    gain: float


@workflow("flux-list", "Synthetic float-list plan plotting")
async def _plot_fluxes(
    env: WorkflowEnv[None], plan: NativePlan, tun: Live[NativeTunables]
) -> None:
    if plan.fluxes:
        viewer = env.plots.liveplot_1d("flux trace", "Flux", f"fluxes={plan.fluxes}")
        values = np.asarray(plan.fluxes)
        viewer.update(values, values * tun.current.gain)
    await env.wait_until(env.started_at)


@workflow("flux-sweep", "Synthetic sweep plan plotting")
async def _plot_sweep(
    env: WorkflowEnv[None], plan: NativePlan, tun: Live[NativeTunables]
) -> None:
    values = np.asarray(plan.fluxes)
    viewer = env.plots.liveplot_1d("flux trace", "Flux", f"fluxes={plan.fluxes}")
    viewer.update(values, values * tun.current.gain)
    await env.wait_until(env.started_at)


@pytest.fixture
def plan_window(qapp: QApplication, tmp_path: Path) -> Iterator[MainWindow]:
    yield from _host_plan_window(qapp, tmp_path, required=False)


@pytest.fixture
def required_plan_window(qapp: QApplication, tmp_path: Path) -> Iterator[MainWindow]:
    yield from _host_plan_window(qapp, tmp_path, required=True)


def _host_plan_window(
    qapp: QApplication, tmp_path: Path, *, required: bool
) -> Iterator[MainWindow]:
    registry = Registry()
    registry.register(
        "sweep-experiment", RequiredFluxAdapter if required else FluxAdapter
    )
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
        progress_transport=DirectProgressTransport(),
    )
    ctrl.workflow_control().configure(
        (
            WorkflowAttachment(
                script=_plot_fluxes,
                plan_model=Plan,
                knobs_model=Tunables,
                plan_to_cfg=lambda form, env, snapshot: NativePlan(fluxes=form.fluxes),
                knobs_to_cfg=lambda form: NativeTunables(gain=form.gain),
                seed=lambda plan, form: None,
            ),
            WorkflowAttachment(
                script=_plot_sweep,
                plan_model=RequiredSweepPlan if required else SweepPlan,
                knobs_model=Tunables,
                plan_to_cfg=lambda form, env, snapshot: NativePlan(
                    fluxes=np.linspace(
                        form.fluxes.start, form.fluxes.stop, form.fluxes.n
                    ).tolist()
                ),
                knobs_to_cfg=lambda form: NativeTunables(gain=form.gain),
                seed=lambda plan, form: None,
            ),
            plot_attachment(),
        )
    )
    ctrl.setup_control.new_entry(EntryRequest("plan-ui"))
    ctrl.setup_control.new_point(PointCreateRequest("source", "", CommandOrigin("gui")))
    board = white_noise()
    ctrl.install_soc(board.soc, board.soccfg)
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    window.refresh_context_panel()
    window.show()
    qapp.processEvents()
    try:
        yield window
    finally:
        facet = ctrl.workflow_control()
        for tab in facet.tabs():
            if tab.busy and tab.run_id is not None:
                facet.stop(tab.run_id)
        qt_until(
            lambda: all(not tab.busy for tab in facet.tabs()), label="workflow teardown"
        )
        ctrl.close()
        window.dispose()


def _snapshot(driver: ComponentDriver, root: Path, name: str) -> None:
    driver.screenshot(root / f"{name}.png")
    (root / f"{name}.txt").write_text(repr(driver.visible()), encoding="utf-8")


def _open(driver: ComponentDriver, root: Path, name: str = "flux-list") -> None:
    toggle_tab_family(driver)
    _snapshot(driver, root, "01-workflows")
    driver.select_menu("New Tab ▾", (name,))
    _snapshot(driver, root, "02-plan")


def _done(driver: ComponentDriver) -> bool:
    return driver.observe(object_name="workflow.status").text == "done" and any(
        w.text == "Start" and w.enabled for w in driver.visible().widgets
    )


def test_operation_groups_replace_old_set_and_retain_workflow_trend(
    plan_window, tmp_path
):
    driver = ComponentDriver(plan_window)
    _open(driver, tmp_path, "operation-plots")
    driver.click("Start")
    qt_until(
        lambda: (
            driver.observe(object_name="workflow.ask").visible
            and '"point": 1' in driver.observe(object_name="workflow.ask").text
        ),
        label="first operation question",
    )
    _snapshot(driver, tmp_path, "03-first-operation")
    assert tuple(axes.ylabel for axes in driver.plots().axes) == (
        "Workflow trend",
        "Attempt 1",
        "Diagnostic",
    )
    driver.enter_text(object_name="workflow.answer", value='{"value": 1}')
    driver.click("Answer")
    qt_until(
        lambda: (
            '"point": 2' in driver.observe(object_name="workflow.ask").text
            and driver.observe("Answer").enabled
        ),
        label="replacement operation question",
    )
    _snapshot(driver, tmp_path, "07-replacement-operation")
    assert tuple(axes.ylabel for axes in driver.plots().axes) == (
        "Workflow trend",
        "Attempt 2",
    )
    driver.enter_text(object_name="workflow.answer", value='{"value": 2}')
    driver.click("Answer")
    qt_until(lambda: _done(driver), label="operation workflow terminal delivery")
    _snapshot(driver, tmp_path, "11-terminal-retained")
    assert tuple(axes.ylabel for axes in driver.plots().axes) == (
        "Workflow trend",
        "Attempt 2",
    )


def test_plan_float_list_edit_errors_and_native_lowering(plan_window, tmp_path):
    driver = ComponentDriver(plan_window)
    _open(driver, tmp_path)
    driver.enter_text(object_name="form.input.fluxes", value="[invalid]")
    _snapshot(driver, tmp_path, "03-invalid-json")
    invalid = driver.observe(object_name="form.input.fluxes")
    assert invalid.text == "[invalid]"
    assert "Invalid JSON" in driver.observe(object_name="form.row.fluxes").tooltip
    driver.click("Start")
    _snapshot(driver, tmp_path, "04-start-refused")
    assert "Invalid JSON" in driver.observe(object_name="workflow.message").text
    driver.enter_text(object_name="form.input.fluxes", value="[0.1, 0.2, 0.3]")
    _snapshot(driver, tmp_path, "05-three-values")
    edited = driver.observe(object_name="form.input.fluxes")
    assert edited.text == "[0.1, 0.2, 0.3]"
    assert "Invalid JSON" not in driver.observe(object_name="form.row.fluxes").tooltip
    driver.click("Start")
    qt_until(lambda: _done(driver), label="finite float-list run")
    _snapshot(driver, tmp_path, "06-native-values-plotted")
    axes = driver.plots().axes
    assert len(axes) == 1
    assert axes[0].ylabel == "fluxes=[0.1, 0.2, 0.3]"
    assert axes[0].visible_curves == 1


def test_sweep_fluxes_edit_errors_and_lowering_match_experiment(plan_window, tmp_path):
    driver = ComponentDriver(plan_window)
    _open(driver, tmp_path, "flux-sweep")
    driver.click(object_name="form.fold.fluxes")
    _snapshot(driver, tmp_path, "03-expanded-fluxes")
    assert any(
        widget.object_name == "form.input.fluxes.start"
        for widget in driver.visible().widgets
    ), "Fluxes must expand into editable sweep coordinates"
    driver.enter_text(object_name="form.input.fluxes.start", value="0.1")
    _snapshot(driver, tmp_path, "04-start")
    driver.enter_text(object_name="form.input.fluxes.stop", value="0.3")
    _snapshot(driver, tmp_path, "05-stop")
    driver.enter_text(object_name="form.input.fluxes.n", value="0")
    _snapshot(driver, tmp_path, "06-invalid-count")
    assert (
        "greater than or equal to 1"
        in driver.observe(object_name="form.row.fluxes.n").tooltip
    )
    driver.click(object_name="form.fold.fluxes")
    _snapshot(driver, tmp_path, "06a-collapsed-invalid")
    driver.click("Start")
    _snapshot(driver, tmp_path, "07-invalid-start-refused")
    assert (
        "greater than or equal to 1"
        in driver.observe(object_name="workflow.message").text
    )
    driver.click(object_name="form.fold.fluxes")
    _snapshot(driver, tmp_path, "07a-reopened-invalid")
    driver.enter_text(object_name="form.input.fluxes.n", value="3")
    _snapshot(driver, tmp_path, "08-valid-count")
    driver.click(object_name="form.fold.fluxes")
    _snapshot(driver, tmp_path, "08a-collapsed-valid")
    driver.click("Start")
    qt_until(lambda: _done(driver), label="finite sweep workflow")
    _snapshot(driver, tmp_path, "09-lowered-sweep")
    workflow_axes = driver.plots().axes
    assert len(workflow_axes) == 1
    assert workflow_axes[0].ylabel == "fluxes=[0.1, 0.2, 0.3]"
    toggle_tab_family(driver)
    _snapshot(driver, tmp_path, "10-experiments")
    driver.select_menu("New Tab ▾", ("sweep-experiment",))
    _snapshot(driver, tmp_path, "11-experiment-form")
    driver.click(object_name="form.fold.sweep")
    _snapshot(driver, tmp_path, "12-expanded-experiment-sweep")
    for field, value in (("start", "0.1"), ("stop", "0.3"), ("n", "3")):
        driver.enter_text(object_name=f"form.input.sweep.{field}", value=value)
        _snapshot(driver, tmp_path, f"13-experiment-{field}")
    driver.click("Run")
    qt_until(
        lambda: (
            any(w.text == "Run" and w.enabled for w in driver.visible().widgets)
            and bool(driver.plots().axes)
        ),
        label="finite experiment run terminal delivery",
    )
    _snapshot(driver, tmp_path, "14-experiment-lowered-sweep")
    assert driver.plots().axes == workflow_axes


def test_required_sweep_without_default_is_editable_like_experiment(
    required_plan_window, tmp_path
):
    driver = ComponentDriver(required_plan_window)
    driver.select_menu("New Tab ▾", ("sweep-experiment",))
    _snapshot(driver, tmp_path, "required-01-experiment")
    if not driver.observe(object_name="form.input.sweep.start").visible:
        driver.click(object_name="form.fold.sweep")
        _snapshot(driver, tmp_path, "required-01a-expanded-experiment")
    fresh_experiment_values = tuple(
        driver.observe(object_name=f"form.input.sweep.{field}").text
        for field in ("start", "stop", "n")
    )
    assert fresh_experiment_values == ("", "", "")
    for field in ("start", "stop", "n"):
        assert (
            "Field required"
            in driver.observe(object_name=f"form.row.sweep.{field}").tooltip
        )
    assert not driver.observe("Run").enabled
    for field, value in (("start", "0.1"), ("stop", "0.3"), ("n", "3")):
        driver.enter_text(object_name=f"form.input.sweep.{field}", value=value)
        _snapshot(driver, tmp_path, f"required-02-experiment-{field}")
    assert driver.observe("Run").enabled
    driver.click("Run")
    qt_until(
        lambda: (
            any(w.text == "Run" and w.enabled for w in driver.visible().widgets)
            and bool(driver.plots().axes)
        ),
        label="required experiment run terminal delivery",
    )
    _snapshot(driver, tmp_path, "required-03-experiment-result")
    experiment_axes = driver.plots().axes
    assert len(experiment_axes) == 1
    assert experiment_axes[0].ylabel == "fluxes=[0.1, 0.2, 0.3]"
    _open(driver, tmp_path, "flux-sweep")
    _snapshot(driver, tmp_path, "required-04-workflow")
    assert any(
        widget.object_name == "form.fold.fluxes" for widget in driver.visible().widgets
    ), "A required Fluxes sweep must offer editable coordinates without a preset"
    if not any(
        widget.object_name == "form.input.fluxes.start"
        for widget in driver.visible().widgets
    ):
        driver.click(object_name="form.fold.fluxes")
        _snapshot(driver, tmp_path, "required-04a-expanded-workflow")
    assert (
        tuple(
            driver.observe(object_name=f"form.input.fluxes.{field}").text
            for field in ("start", "stop", "n")
        )
        == fresh_experiment_values
    )
    for field in ("start", "stop", "n"):
        assert (
            "Field required"
            in driver.observe(object_name=f"form.row.fluxes.{field}").tooltip
        )
    driver.click(object_name="form.fold.fluxes")
    _snapshot(driver, tmp_path, "required-04b-collapsed-missing-workflow")
    driver.click("Start")
    _snapshot(driver, tmp_path, "required-04c-missing-workflow-refused")
    assert "Field required" in driver.observe(object_name="workflow.message").text
    driver.click(object_name="form.fold.fluxes")
    _snapshot(driver, tmp_path, "required-04d-reopened-missing-workflow")
    for field, value in (("start", "0.1"), ("stop", "0.3"), ("n", "3")):
        driver.enter_text(object_name=f"form.input.fluxes.{field}", value=value)
        _snapshot(driver, tmp_path, f"required-05-workflow-{field}")
    driver.click(object_name="form.fold.fluxes")
    _snapshot(driver, tmp_path, "required-06-workflow-collapsed")
    driver.click("Start")
    qt_until(lambda: _done(driver), label="required sweep workflow terminal delivery")
    _snapshot(driver, tmp_path, "required-07-workflow-result")
    assert driver.plots().axes == experiment_axes


def test_workflow_figures_appear_and_release_on_replacement(plan_window, tmp_path):
    driver = ComponentDriver(plan_window)
    _open(driver, tmp_path)
    assert driver.plots().axes == ()
    driver.click("Start")
    qt_until(lambda: _done(driver), label="finite figure-producing run")
    _snapshot(driver, tmp_path, "03-figure-arrived")
    assert len(driver.plots().axes) == 1
    driver.enter_text(object_name="form.input.fluxes", value="[]")
    assert driver.observe(object_name="form.input.fluxes").text == "[]"
    _snapshot(driver, tmp_path, "04-empty-plan")
    driver.click("Start")
    qt_until(lambda: _done(driver), label="finite plot-free replacement run")
    _snapshot(driver, tmp_path, "05-figures-released")
    assert driver.plots().axes == ()
