"""Window RPC call types, Start batches, and synthetic Window view sessions.

Nothing here imports user workflow packages. The layout forms only need enough
fields and collapsible sections to make the old single-column layout tall;
the figures workflow only needs named figures that a test script draws.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Annotated

from matplotlib.figure import Figure
from pydantic import Field
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QWidget
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog
from zcu_tools.gui.app.workflows.remote import FormWrite
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import Display, FormEnv, FormModel
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.resources.entry import ResultEntry

from tests.experiment.workflows._engine_fakes import Plan, State, Tunables
from tests.gui._user_input import mouse_double_click

from ._reload_source import SourceNamespace
from ._start_fakes import StartHarness

RpcRequest = Callable[[str, dict[str, object]], dict[str, object]]

QUBIT_KIND = "fake/layout-qubit"
FIGURE = "synthetic/trend"


def start_plan_batch() -> list[FormWrite]:
    """Supply a branch and expression, leaving points to fresh Start defaults."""
    return [
        {"path": ["samples", "kind"], "value": "list"},
        {"path": ["samples", "values"], "value": [1.0, 1.1]},
        {"path": ["scale"], "value": {"expr": "0.06 * 2"}},
    ]


class SectionForm(FormModel):
    """Eight ordinary leaves standing in for one collapsible settings group."""

    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 3.0
    delta: float = 4.0
    epsilon: float = 5.0
    zeta: float = 6.0
    eta: float = 7.0
    theta: float = 8.0


class WidePlanForm(FormModel):
    """Start plan whose expanded sections are far taller than one screen."""

    count: int = Field(default=3, ge=1)
    first: Annotated[SectionForm, Display(label="First", collapsed=True)] = Field(
        default_factory=SectionForm
    )
    second: Annotated[SectionForm, Display(label="Second", collapsed=True)] = Field(
        default_factory=SectionForm
    )
    third: Annotated[SectionForm, Display(label="Third", collapsed=True)] = Field(
        default_factory=SectionForm
    )


class WideKnobsForm(FormModel):
    """Tunables with the native positive value and two collapsible groups."""

    value: int = Field(default=1, ge=1)
    timing: Annotated[SectionForm, Display(label="Timing", collapsed=True)] = Field(
        default_factory=SectionForm
    )
    limits: Annotated[SectionForm, Display(label="Limits", collapsed=True)] = Field(
        default_factory=SectionForm
    )


class SmallPlanForm(FormModel):
    """Second workflow's plan, distinct from the wide one."""

    count: int = Field(default=1, ge=1)


class SmallKnobsForm(FormModel):
    """Second workflow's tunables."""

    value: int = Field(default=2, ge=1)


def _init(env: InitEnv[None], plan: Plan) -> State:
    del env, plan
    return State()


def wide_registration(
    on_point: Callable[[int], None],
) -> WorkflowAttachment[Plan, Tunables, State, int, WidePlanForm, WideKnobsForm, None]:
    """Declare "synthetic_wide": a live figure and bar, then on_point per point.

    on_point receives the point index on the worker; a test may request Pause
    there, and the following wait discards that uncommitted point.
    """

    def step(
        env: WorkflowEnv[None], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        if FIGURE not in env.plots:
            env.plots.subplots(FIGURE)

        def draw(figure: Figure) -> None:
            figure.clear()
            figure.subplots().plot(range(len(state.values)), state.values)

        env.plots.draw(FIGURE, draw)
        env.pbar("points", plan.count).refresh()
        if state.index >= plan.count:
            return Done()
        on_point(state.index)
        yield from env.wait_until(env.started_at)
        return Next(tun.value, State(state.index + 1, [*state.values, tun.value]))

    def plan_to_cfg(form: WidePlanForm, env: FormEnv, snapshot: RunSnapshot) -> Plan:
        del env, snapshot
        return Plan(count=form.count)

    declared = workflow("synthetic_wide", plan=Plan, tunables=Tunables, init=_init)(
        step
    )
    return WorkflowAttachment(
        step=declared,
        plan_model=WidePlanForm,
        knobs_model=WideKnobsForm,
        plan_to_cfg=plan_to_cfg,
        knobs_to_cfg=lambda form: Tunables(value=form.value),
        seed=lambda plan, form: None,
    )


def small_registration() -> WorkflowAttachment[
    Plan, Tunables, State, int, SmallPlanForm, SmallKnobsForm, None
]:
    """Declare "synthetic_small", which finishes without drawing."""

    def step(
        env: WorkflowEnv[None], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del env, plan, tun, state
        yield from ()
        return Done()

    def plan_to_cfg(form: SmallPlanForm, env: FormEnv, snapshot: RunSnapshot) -> Plan:
        del env, snapshot
        return Plan(count=form.count)

    declared = workflow("synthetic_small", plan=Plan, tunables=Tunables, init=_init)(
        step
    )
    return WorkflowAttachment(
        step=declared,
        plan_model=SmallPlanForm,
        knobs_model=SmallKnobsForm,
        plan_to_cfg=plan_to_cfg,
        knobs_to_cfg=lambda form: Tunables(value=form.value),
        seed=lambda plan, form: None,
    )


@dataclass(frozen=True)
class FigurePoint:
    """What one "synthetic_figures" point does with its named figures.

    create names are registered first and left undrawn; draw names are then
    created if missing and drawn in this order. Each drawn figure shows one
    line ending at point index + 1, under an axes title equal to its name.
    """

    draw: tuple[str, ...]
    create: tuple[str, ...] = ()


def _draw_point(name: str, index: int, figure: Figure) -> None:
    figure.clear()
    axes = figure.subplots()
    axes.plot([0, 1], [0, index + 1])
    axes.set_title(name)


def figures_registration(
    points: Callable[[], Sequence[FigurePoint]],
    on_point: Callable[[int], None],
) -> WorkflowAttachment[
    Plan, Tunables, State, int, SmallPlanForm, SmallKnobsForm, None
]:
    """Declare "synthetic_figures": one point per FigurePoint, then Done.

    points is read when each point runs, so a test sets the script before
    Start. on_point receives the index after that point's figures are drawn,
    on the worker; the following wait discards the point if it requests Pause.
    """

    def step(
        env: WorkflowEnv[None], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del plan
        script = points()
        if state.index >= len(script):
            return Done()
        point = script[state.index]
        for name in (*point.create, *point.draw):
            if name not in env.plots:
                env.plots.subplots(name)
        for name in point.draw:
            env.plots.draw(name, partial(_draw_point, name, state.index))
        on_point(state.index)
        yield from env.wait_until(env.started_at)
        return Next(tun.value, State(state.index + 1, [*state.values, tun.value]))

    def plan_to_cfg(form: SmallPlanForm, env: FormEnv, snapshot: RunSnapshot) -> Plan:
        del env, snapshot
        return Plan(count=form.count)

    declared = workflow("synthetic_figures", plan=Plan, tunables=Tunables, init=_init)(
        step
    )
    return WorkflowAttachment(
        step=declared,
        plan_model=SmallPlanForm,
        knobs_model=SmallKnobsForm,
        plan_to_cfg=plan_to_cfg,
        knobs_to_cfg=lambda form: Tunables(value=form.value),
        seed=lambda plan, form: None,
    )


def bind_point(harness: StartHarness, root: Path) -> ResultEntry:
    """Open entry "sample" at point "source" with $qubit bound to Q1.

    The services publish their own context/alias events; nothing is cached.
    """
    entry = ResultEntry.create(
        "sample", result_root=root / "result", database_root=root / "Database"
    )
    entry.setup.add_component("Q1", kind=QUBIT_KIND)
    entry.new_point("source", description="layout contract")
    context = harness.services.context
    context.install_entry(entry)
    context.use_point(PointUseRequest("source"))
    context.set_alias_rows((AliasRow("qubit", "Q1"),))
    return entry


def install_soc(harness: StartHarness) -> None:
    """Publish a mock SoC directly in state, as a caller-owned connection would."""
    soc, soccfg = make_mock_soc()
    harness.state.set_context(
        replace(harness.state.session_env, soc=soc, soccfg=soccfg)
    )


def form_editable(panel: QWidget) -> bool:
    """Whether a user can type into panel's form.

    True when some form.input text field under panel is enabled and not
    read-only. A run leaves the Start forms enabled and readable, with every
    text field read-only.
    """
    return any(
        field.objectName().startswith("form.input.")
        and field.isEnabled()
        and not field.isReadOnly()
        for field in panel.findChildren(QLineEdit)
    )


def expand_all(widget: QWidget, qapp: QApplication) -> None:
    """Double-click every enabled folded form.fold.* title, including titles
    revealed by an outer one.

    A folded title's text starts with "\u25b8". Disabled titles (a detached
    form) get no mouse input from a user and are left as they are.
    """
    while True:
        folds = [
            title
            for title in widget.findChildren(QLabel)
            if title.objectName().startswith("form.fold.")
            and title.text().startswith("\u25b8")
            and title.isEnabled()
            and title.isVisibleTo(widget)
        ]
        if not folds:
            return
        for title in folds:
            mouse_double_click(title, title.rect().center())
        qapp.processEvents()


class WindowSession:
    """One real Window over StartHarness owners and the synthetic catalog.

    The catalog lists synthetic_wide (default), synthetic_small, then
    synthetic_figures; source.catalog is what the next Reload publishes.
    during_point, when set, receives each wide point's index on its worker
    before that point's pause check, so a test can observe the Window while
    the run is live or end the run there. pause_at is the wide workflow's
    point index at which its own worker requests Pause; None lets it run to
    Done. figure_points is the figures
    workflow's script; each index in figure_pauses requests Pause once, so
    Resume reruns that point and continues. close drains pending work before
    closing Qt.
    """

    def __init__(self, root: Path) -> None:
        self.harness = StartHarness()
        self.root = root
        self.during_point: Callable[[int], None] | None = None
        self.pause_at: int | None = None
        self.figure_points: Sequence[FigurePoint] = ()
        self.figure_pauses: set[int] = set()
        self.source = SourceNamespace(
            root, "synthetic_layout_source", self.fresh_catalog()
        )
        self.window = WorkflowWindow(
            self.harness.state,
            self.harness.services,
            self.harness.owner,
            registrations=self.source.catalog.attachments,
            loader=self.source.loader,
        )

    def fresh_catalog(self) -> WorkflowCatalog[None]:
        """Declare both synthetic workflows again, as a source reload would."""
        return WorkflowCatalog(
            (
                wide_registration(self._on_point),
                small_registration(),
                figures_registration(lambda: self.figure_points, self._on_figure),
            )
        )

    def _on_point(self, index: int) -> None:
        if self.during_point is not None:
            self.during_point(index)
        if index != self.pause_at:
            return
        self._pause_run()

    def _on_figure(self, index: int) -> None:
        if index not in self.figure_pauses:
            return
        self.figure_pauses.discard(index)
        self._pause_run()

    def _pause_run(self) -> None:
        run_id = self.window.controller.run_id
        if run_id is None:
            raise RuntimeError("A running point requires a run identity")
        self.window.pause(run_id)

    def close(self, qapp: QApplication) -> None:
        controller = self.window.controller
        if self.harness.background.work is not None:
            if controller.operation_id is None:
                raise RuntimeError("Pending work requires an operation")
            self.harness.handles.cancel(controller.operation_id)
            self.harness.complete_segment()
        self.window.close()
        self.window.deleteLater()
        qapp.processEvents()
        self.source.close()
