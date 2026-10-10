"""Deterministic host ports for public Engine seam tests."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Completed,
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
from zcu_tools.plotting.plots import NonPresentingHost, Plots
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
    """Record requested native setpoints without touching hardware."""

    def __init__(self) -> None:
        self.values: list[tuple[str, float]] = []
        self.on_set: Callable[[StopSignal], object] | None = None

    def set_value(self, name: str, value: float, cancel_signal: StopSignal) -> None:
        self.values.append((name, value))
        if self.on_set is not None:
            self.on_set(cancel_signal)



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
    """Mutable experiment value and optional role-to-device-name mapping."""

    value: int = 1
    dev: dict[str, str] | None = None


def save_integer(source: Completed[Cfg, int], path: Path) -> None:
    """Write the paired cfg value and integer result to the engine temporary."""
    path.write_text(f"{source.cfg.value}:{source.result}", encoding="utf-8")


@dataclass
class Rig:
    """One isolated host/Engine and its dynamic on-disk outputs."""

    root: Path
    clock: ManualClock = field(default_factory=ManualClock)
    devices: RecordingDevices = field(default_factory=RecordingDevices)
    plots: Plots = field(default_factory=lambda: Plots(NonPresentingHost()))
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
        devices: DeviceSnapshot | None = None,
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
            init=self.init,
        )(invoke)
        self.engine.start(
            declared,
            plan=Plan() if plan is None else plan,
            tunables=Tunables(),
            paths=self.paths,
            identity=RunIdentity(
                "run",
                DeviceSnapshot({}) if devices is None else devices,
                SoftwareProvenance(
                    software_versions={},
                    git_commit=None,
                    git_dirty=None,
                    qick_version=None,
                    soc_fingerprint=None,
                    hostname="offline",
                ),
            ),
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
