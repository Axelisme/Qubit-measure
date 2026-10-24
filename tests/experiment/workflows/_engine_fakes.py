"""Deterministic ports and worker lifetime for public async Engine tests."""

from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from time import monotonic

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import FakeDevice
from zcu_tools.experiment.metadata import SoftwareProvenance
from zcu_tools.experiment.script_host import PendingQuestion
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import (
    Completed,
    Engine,
    EnginePorts,
    Failed,
    Live,
    RunIdentity,
    RunPaths,
    RunStatus,
    WorkflowEnv,
    WorkflowPlots,
    WorkflowScript,
    workflow,
)
from zcu_tools.plotting.plots import NonPresentingHost, PlotHost

from ._engine_ports import START as START
from ._engine_ports import ManualClock, RecordingBar
from ._engine_ports import RecordingDevices as DeviceRecorder


class RecordingDevices(DeviceRecorder):
    """Expose a detached connected mapping without owning driver lifetimes."""

    def get_connected_devices(self) -> dict[str, FakeDevice]:
        return dict(self.drivers)


class Plan(BaseModel):
    """Fixed point count and nested input for copy-isolation scenarios."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    count: int = Field(default=2, ge=1)
    values: list[int] = Field(default_factory=lambda: [10])


class Tunables(BaseModel):
    """Positive value read through Live at author-selected boundaries."""

    model_config = ConfigDict(extra="forbid")
    value: int = Field(default=1, ge=1)


class Point(BaseModel):
    value: int


class Cfg(ConfigBase):
    """Minimal core-compatible input, with optional device roles."""

    value: int = 1
    dev: dict[str, str] = Field(default_factory=dict)
    nested: list[int] = Field(default_factory=lambda: [2])


def completed[T](outcome: Completed[T] | Failed) -> Completed[T]:
    """Require a successful delivery before checking its payload."""
    assert isinstance(outcome, Completed), outcome
    return outcome


def identity(run_id: str = "run") -> RunIdentity:
    return RunIdentity(
        run_id,
        SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname="offline",
        ),
    )


def until(predicate: Callable[[], bool], label: str) -> None:
    """Wait for a public condition; elapsed time only guards against deadlock."""
    deadline = monotonic() + 10
    tick = Event()
    while not predicate():
        assert monotonic() < deadline, f"Timed out awaiting {label}"
        tick.wait(0.001)


@dataclass
class Rig:
    """One isolated Engine and its caller-owned ports and output roots."""

    root: Path
    native: bool = False
    clock: ManualClock = field(default_factory=ManualClock)
    devices: RecordingDevices = field(default_factory=RecordingDevices)
    plot_host: PlotHost = field(default_factory=NonPresentingHost)
    plots: WorkflowPlots = field(init=False)
    bars: list[RecordingBar] = field(default_factory=list)
    engine: Engine[int] = field(init=False)
    paths: RunPaths = field(init=False)

    def __post_init__(self) -> None:
        self.plots = WorkflowPlots(self.plot_host)
        self.paths = RunPaths(self.root / "metadata", self.root / "data")
        board = white_noise() if self.native else None
        self.engine = Engine(
            EnginePorts(
                self.plots,
                self.progress,
                self.clock,
                context=7,
                devices=self.devices,
                soc=None if board is None else board.soc,
                soccfg=None if board is None else board.soccfg,
            )
        )

    def progress(
        self,
        iterable: Iterable[object] | None = None,
        desc: str | None = None,
        **kwargs: object,
    ) -> RecordingBar:
        bar = RecordingBar(iterable=iterable, desc=desc, **kwargs)
        self.bars.append(bar)
        return bar

    def start(
        self,
        script: WorkflowScript[Plan, Tunables, None, int],
        *,
        plan: Plan | None = None,
    ) -> None:
        async def invoke(
            env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]
        ) -> None:
            await script(env, plan, tun)

        self.engine.start(
            workflow("example", "Synthetic async workflow")(invoke),
            plan=Plan() if plan is None else plan,
            tunables=Tunables(),
            paths=self.paths,
            identity=identity(),
        )

    def events(self) -> tuple[dict[str, JsonValue], ...]:
        adapter = TypeAdapter(dict[str, JsonValue])
        return tuple(
            adapter.validate_json(line)
            for line in (self.paths.metadata_root / "journal.jsonl")
            .read_text()
            .splitlines()
        )

    def kinds(self) -> tuple[JsonValue, ...]:
        return tuple(event["kind"] for event in self.events())

    def close(self) -> None:
        self.plots.finish(present=False)
        self.plots.release()


@contextmanager
def running(rig: Rig) -> Generator[Future[RunStatus]]:
    """Keep execute on one worker until Stop/return, including paused/Ask time."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rig.engine.execute, "run")
        try:
            yield future
        finally:
            rig.engine.stop("run")
            future.result(timeout=10)


def pending(rig: Rig) -> str:
    """Wait until the public Ask seam exposes an answerable request id."""
    until(lambda: rig.engine.pending_question("run") is not None, "pending question")
    return question(rig).request_id


def question(rig: Rig) -> PendingQuestion:
    """Read an expected pending request through the public Engine seam."""
    value = rig.engine.pending_question("run")
    assert value is not None
    return value


def paused(rig: Rig) -> None:
    until(
        lambda: (
            (status := rig.engine.status()) is not None and status.lifecycle == "paused"
        ),
        "paused lifecycle",
    )
