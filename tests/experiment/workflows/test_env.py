"""Environment capability and display contracts, without executing requests."""

from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.experiment.workflows import InitEnv, MissingCapability, WorkflowEnv
from zcu_tools.experiment.workflows.env import Displays
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar

START = datetime(2026, 10, 6, tzinfo=UTC)


class RecordingPlots:
    """Plot seam recording detached line data for each refresh."""

    def __init__(self) -> None:
        self.snapshots: list[tuple[NDArray[np.float64], ...]] = []

    def axes(self, name: str) -> Axes:
        return Figure().subplots()

    def refresh(self, figure: Figure) -> None:
        self.snapshots.append(
            tuple(
                np.asarray(line.get_ydata(), dtype=np.float64).copy()
                for axes in figure.axes
                for line in axes.lines
            )
        )


class RecordingBar(TQDMProgressBar):
    """Concrete runtime bar with a visible close flag for lifecycle observations."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


@pytest.fixture
def displays() -> Generator[Displays, None, None]:
    owner = Displays(RecordingPlots(), RecordingBar)
    try:
        yield owner
    finally:
        owner.close()


@pytest.fixture
def env(tmp_path: Path, displays: Displays) -> WorkflowEnv[str]:
    return WorkflowEnv(START, "snapshot", tmp_path, displays)


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
) -> None:
    initial: InitEnv[str] = InitEnv(START, None)
    step: WorkflowEnv[str] = WorkflowEnv(START, None, tmp_path, displays)
    with pytest.raises(MissingCapability, match="context"):
        _ = initial.context
    with pytest.raises(MissingCapability, match="context"):
        _ = step.context


@pytest.mark.parametrize(
    ("name", "value"),
    [("", 1.0), (" ", 1.0), ("flux", np.nan), ("flux", np.inf)],
)
def test_invalid_setpoint_is_rejected_before_suspension(
    env: WorkflowEnv[str],
    name: str,
    value: float,
) -> None:
    request = env.set_device(name, value)
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


def test_axes_are_stable_across_step_environments(
    env: WorkflowEnv[str],
    displays: Displays,
    tmp_path: Path,
) -> None:
    axes = env.axes("main")
    (line,) = axes.plot([0.0, 1.0], [2.0, 3.0])
    next_step = WorkflowEnv(START, "snapshot", tmp_path / "next", displays)
    assert next_step.axes("main") is axes
    np.testing.assert_array_equal(
        next_step.axes("main").lines[0].get_ydata(), [2.0, 3.0]
    )
    assert env.axes("other") is not axes
    assert line.axes is axes


def test_display_refresh_publishes_current_figure_data(
    tmp_path: Path,
) -> None:
    plots = RecordingPlots()
    owner = Displays(plots, RecordingBar)
    env = WorkflowEnv(START, "snapshot", tmp_path, owner)
    axes = env.axes("main")
    (line,) = axes.plot([0.0, 1.0], [2.0, 3.0])
    owner.refresh()
    line.set_ydata([10.0, 20.0])
    owner.refresh()
    np.testing.assert_array_equal(plots.snapshots[0][0], [2.0, 3.0])
    np.testing.assert_array_equal(plots.snapshots[1][0], [10.0, 20.0])


def test_bar_progress_and_identity_are_retained_until_owner_closes(
    env: WorkflowEnv[str],
    displays: Displays,
    tmp_path: Path,
) -> None:
    bar = env.pbar("points", 10)
    bar.set_progress(3)
    next_step = WorkflowEnv(START, "snapshot", tmp_path / "next", displays)
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


@pytest.mark.parametrize("name", ["", " "])
def test_empty_axes_name_is_rejected(env: WorkflowEnv[str], name: str) -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        env.axes(name)
