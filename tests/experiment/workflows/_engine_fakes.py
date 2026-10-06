"""Deterministic host ports for public Engine seam tests."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    DeviceSetup,
    DeviceSnapshot,
    Engine,
    EnginePorts,
    InitEnv,
    RunIdentity,
    RunPaths,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar

START = datetime(2026, 10, 7, tzinfo=UTC)


class ManualClock:
    """Advance only when the Engine requests a future wait."""

    def __init__(self) -> None:
        self.value = START
        self.targets: list[datetime] = []
        self.on_wait: Callable[[StopSignal], object] | None = None

    def now(self) -> datetime:
        return self.value

    def wait_until(self, target: datetime, cancel_signal: StopSignal) -> None:
        self.targets.append(target)
        if self.on_wait is not None:
            self.on_wait(cancel_signal)
        if not cancel_signal.is_set():
            self.value = target


class RecordingDevices:
    """Record native setpoints and setup calls without touching hardware."""

    def __init__(self) -> None:
        self.values: list[tuple[str, float]] = []
        self.settings: list[tuple[DeviceSetup, ...]] = []
        self.on_set: Callable[[StopSignal], object] | None = None
        self.on_setup: Callable[[StopSignal], object] | None = None

    def set_value(self, name: str, value: float, cancel_signal: StopSignal) -> None:
        self.values.append((name, value))
        if self.on_set is not None:
            self.on_set(cancel_signal)

    def setup(
        self, settings: tuple[DeviceSetup, ...], cancel_signal: StopSignal
    ) -> None:
        self.settings.append(settings)
        if self.on_setup is not None:
            self.on_setup(cancel_signal)


class RecordingPlots:
    """Use native detached Figures and record line snapshots."""

    def __init__(self) -> None:
        self.figures: dict[str, Figure] = {}
        self.snapshots: list[tuple[NDArray[np.float64], ...]] = []
        self.error: Exception | None = None

    def axes(self, name: str) -> Axes:
        figure = Figure()
        self.figures[name] = figure
        return figure.subplots()

    def refresh(self, figure: Figure) -> None:
        if self.error is not None:
            raise self.error
        self.snapshots.append(
            tuple(
                np.asarray(line.get_ydata(), dtype=np.float64).copy()
                for axes in figure.axes
                for line in axes.lines
            )
        )


class RecordingBar(TQDMProgressBar):
    """Silent runtime progress bar with observable close lifecycle."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs, file=StringIO())
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


class Plan(BaseModel):
    """Number of points and nested input data to exercise copy isolation."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    count: int = Field(default=2, ge=1)
    values: list[int] = Field(default_factory=lambda: [10])


class Tunables(BaseModel):
    """Positive test value captured once per step."""

    model_config = ConfigDict(extra="forbid")
    value: int = Field(default=1, ge=1)


@dataclass
class State:
    """Next point index and mutable values carried only by committed Nexts."""

    index: int = 0
    values: list[int] = field(default_factory=list)


@dataclass
class Cfg:
    """Mutable experiment config and optional native device setup."""

    value: int = 1
    dev: tuple[DeviceSetup, ...] | None = None


@dataclass
class Rig:
    """One isolated host/Engine and its dynamic on-disk outputs."""

    root: Path
    clock: ManualClock = field(default_factory=ManualClock)
    devices: RecordingDevices = field(default_factory=RecordingDevices)
    plots: RecordingPlots = field(default_factory=RecordingPlots)
    initialized: list[datetime] = field(default_factory=list)
    bars: list[RecordingBar] = field(default_factory=list)
    engine: Engine[int] = field(init=False)
    paths: RunPaths = field(init=False)

    def __post_init__(self) -> None:
        self.paths = RunPaths(self.root / "metadata", self.root / "data")
        self.engine = Engine(
            EnginePorts(
                self.plots, self.progress, self.clock, context=7, devices=self.devices
            )
        )

    def progress(self, **kwargs: object) -> RecordingBar:
        bar = RecordingBar(**kwargs)
        self.bars.append(bar)
        return bar

    def init(self, env: InitEnv[int], _plan: Plan) -> State:
        self.initialized.append(env.started_at)
        return State()

    def start(
        self,
        step: Callable[[WorkflowEnv[int], Plan, Tunables, State], Step[State, int]],
        *,
        plan: Plan | None = None,
    ) -> None:
        # A fresh declared wrapper avoids mutating shared module-level callables.
        def invoke(
            env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
        ) -> Step[State, int]:
            return step(env, plan, tun, state)

        declared = workflow(
            "example",
            plan=Plan,
            tunables=Tunables,
            state=State,
            record=int,
            init=self.init,
        )(invoke)
        self.engine.start(
            declared,
            plan=Plan() if plan is None else plan,
            tunables=Tunables(),
            paths=self.paths,
            identity=RunIdentity("run", DeviceSnapshot(()), "offline"),
        )

    def events(self) -> tuple[dict[str, JsonValue], ...]:
        adapter = TypeAdapter(dict[str, JsonValue])
        return tuple(
            adapter.validate_json(line)
            for line in (self.paths.metadata_root / "journal.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )

    def kinds(self) -> tuple[JsonValue, ...]:
        return tuple(event["kind"] for event in self.events())
