"""Environment capability and display contracts, without executing requests."""

from collections.abc import Generator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from qick import QickConfig
from zcu_tools.experiment.workflows import (
    EnginePorts,
    InitEnv,
    MissingCapability,
    WorkflowEnv,
)
from zcu_tools.experiment.workflows.env import Displays
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar

from ._engine_fakes import ManualClock, RecordingDevices

START = datetime(2026, 10, 6, tzinfo=UTC)


class RecordingBar(TQDMProgressBar):
    """Concrete runtime bar with a visible close flag for lifecycle observations."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


@pytest.fixture
def ports() -> Generator[EnginePorts[str], None, None]:
    plots = Plots(NonPresentingHost())
    try:
        yield EnginePorts(plots, RecordingBar, ManualClock(), context="snapshot")
    finally:
        plots.finish(present=False)
        plots.release()


@pytest.fixture
def displays(ports: EnginePorts[str]) -> Generator[Displays, None, None]:
    owner = Displays(ports.progress)
    try:
        yield owner
    finally:
        owner.close()


@pytest.fixture
def env(
    tmp_path: Path, displays: Displays, ports: EnginePorts[str]
) -> WorkflowEnv[str]:
    return WorkflowEnv(START, ports, tmp_path, displays)


def test_init_exposes_original_start_and_context() -> None:
    initial = InitEnv(START, "snapshot")
    assert initial.started_at == START
    assert initial.context == "snapshot"


@pytest.mark.parametrize(
    "start",
    [
        datetime(2026, 10, 6),
        datetime(2026, 10, 6, tzinfo=timezone(timedelta(hours=8))),
    ],
)
def test_naive_and_non_utc_start_are_rejected(start: datetime) -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        InitEnv(start, "snapshot")


def test_absent_context_is_not_silently_returned(
    displays: Displays,
    tmp_path: Path,
    ports: EnginePorts[str],
) -> None:
    initial: InitEnv[str] = InitEnv(START, None)
    step: WorkflowEnv[str] = WorkflowEnv(
        START, replace(ports, context=None), tmp_path, displays
    )
    with pytest.raises(MissingCapability, match="context"):
        _ = initial.context
    with pytest.raises(MissingCapability, match="context"):
        _ = step.context


@pytest.mark.parametrize("port", ["soc", "devices"])
def test_absent_native_port_raises_on_accessor(
    env: WorkflowEnv[str], port: str
) -> None:
    with pytest.raises(MissingCapability, match=port):
        _ = env.soc if port == "soc" else env.devices


def test_native_accessors_return_injected_handles_without_operating(
    ports: EnginePorts[str], displays: Displays, tmp_path: Path
) -> None:
    soc, devices = object(), RecordingDevices()
    native = replace(ports, soc=soc, soccfg=QickConfig({}), devices=devices)
    env = WorkflowEnv(START, native, tmp_path, displays)
    assert env.soc is soc
    assert env.devices is devices
    assert devices.values == []


@pytest.mark.parametrize(
    ("name", "value"),
    [("", 1.0), (" ", 1.0), ("flux", np.nan), ("flux", np.inf)],
)
def test_invalid_setpoint_is_rejected_before_suspension(
    env: WorkflowEnv[str],
    name: str,
    value: float,
) -> None:
    request = env.set_device(name, value, unit="native")
    with pytest.raises(ValueError, match="finite absolute value"):
        next(request)


def test_wait_rejects_naive_target_before_suspension(env: WorkflowEnv[str]) -> None:
    request = env.wait_until(datetime(2026, 10, 6))
    with pytest.raises(ValueError, match="timezone-aware"):
        next(request)


@dataclass
class Cfg:
    """Pure fake config; value is one integer experiment input."""

    value: int = 1


def test_anonymous_experiment_is_rejected_before_suspension(
    env: WorkflowEnv[str],
) -> None:
    request = env.run(lambda _run: 1, Cfg(), save=lambda _completed, _path: None)
    with pytest.raises(ValueError, match="ASCII function name"):
        next(request)


def test_plots_are_shared_across_step_environments(
    env: WorkflowEnv[str], displays: Displays, tmp_path: Path, ports: EnginePorts[str]
) -> None:
    viewer = env.plots.liveplot_1d("main", "x", "y")
    next_step = WorkflowEnv(START, ports, tmp_path / "next", displays)
    assert env.plots is next_step.plots is ports.plots
    viewer.update(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
    np.testing.assert_array_equal(
        next_step.plots["main"].axes[0].lines[0].get_ydata(), [2.0, 3.0]
    )


def test_bar_progress_and_identity_are_retained_until_owner_closes(
    env: WorkflowEnv[str],
    displays: Displays,
    tmp_path: Path,
    ports: EnginePorts[str],
) -> None:
    bar = env.pbar("points", 10)
    bar.set_progress(3)
    next_step = WorkflowEnv(START, ports, tmp_path / "next", displays)
    assert next_step.pbar("points", 10) is bar
    assert next_step.pbar("points", 10).n == 3
    assert isinstance(bar, RecordingBar)
    assert not bar.closed
    displays.close()
    assert bar.closed


def test_same_bar_name_cannot_change_total(env: WorkflowEnv[str]) -> None:
    bar = env.pbar("points", 10)
    with pytest.raises(ValueError, match="total changed"):
        env.pbar("points", 11)
    assert bar.total == 10


@pytest.mark.parametrize(("name", "total"), [("", 1), ("points", -1), ("points", True)])
def test_invalid_bar_declaration_is_rejected(
    env: WorkflowEnv[str],
    name: str,
    total: int,
) -> None:
    with pytest.raises(ValueError, match="nonnegative integer total"):
        env.pbar(name, total)
