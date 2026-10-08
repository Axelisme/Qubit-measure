"""Workflow data callbacks use explicit Plots without artist ownership in Run."""

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Completed,
    DeviceSnapshot,
    Done,
    Engine,
    EnginePorts,
    InitEnv,
    Next,
    Run,
    RunIdentity,
    RunPaths,
    Step,
    WorkflowEnv,
    WorkflowStep,
    workflow,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from ._engine_fakes import (
    Cfg,
    ManualClock,
    Plan,
    RecordingBar,
    State,
    Tunables,
    save_integer,
)


@pytest.fixture
def plots() -> Generator[Plots, None, None]:
    owner = Plots(NonPresentingHost())
    try:
        yield owner
    finally:
        owner.finish(present=False)
        owner.release()


def test_run_live_supplies_captured_axes_and_latest_complex_samples(
    plots: Plots,
) -> None:
    run = Run(Cfg(), StopSignal())
    viewer = plots.liveplot_1d("trace", "Time", "Signal")
    seen: list[NDArray[np.complex128]] = []

    def live(
        axes: tuple[NDArray[np.float64], ...], samples: NDArray[np.complex128]
    ) -> None:
        seen.append(samples.copy())
        viewer.update(axes[0], samples.real)

    run.bind_live(live)
    xs = np.array([0.0, 1.0])
    buffer = run.buffer((2,), axes=(xs,))
    xs[:] = 100.0
    buffer.set(np.array([1.0 + 2.0j, 3.0 + 4.0j]))
    buffer.set(np.array([5.0 + 6.0j, 7.0 + 8.0j]))
    np.testing.assert_array_equal(seen[-1], [5.0 + 6.0j, 7.0 + 8.0j])
    line = plots["trace"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [0.0, 1.0])
    np.testing.assert_array_equal(line.get_ydata(), [5.0, 7.0])
    assert run.live_error is None


def test_two_dimensional_live_callback_preserves_axis_order_and_partial_nan(
    plots: Plots,
) -> None:
    run = Run(Cfg(), StopSignal())
    viewer = plots.liveplot_2d("grid", "Frequency", "Gain")
    run.bind_live(lambda axes, samples: viewer.update(axes[0], axes[1], samples.real))
    xs, ys = np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])
    buffer = run.buffer((2, 3), axes=(xs, ys))
    data = np.array([[1.0 + 2.0j, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.complex128)
    buffer.set(data)
    image = plots["grid"].axes[0].images[0]
    np.testing.assert_array_equal(image.get_array(), data.real.T)
    assert run.live_error is None


def test_live_ambiguity_retains_original_failure() -> None:
    run = Run(Cfg(), StopSignal())
    run.bind_live(lambda _axes, _samples: None)
    run.buffer((2,), axes=(np.array([0.0, 1.0]),))
    with pytest.raises(ValueError, match="unambiguous buffer") as caught:
        run.buffer((2,), axes=(np.array([0.0, 1.0]),))
    assert run.live_error is caught.value


def test_live_must_bind_once_before_buffer_creation() -> None:
    run = Run(Cfg(), StopSignal())
    run.bind_live(lambda _axes, _samples: None)
    with pytest.raises(ValueError, match="once before buffer"):
        run.bind_live(lambda _axes, _samples: None)
    late = Run(Cfg(), StopSignal())
    late.buffer((1,), axes=(np.array([0.0]),))
    with pytest.raises(ValueError, match="once before buffer"):
        late.bind_live(lambda _axes, _samples: None)


def _start(
    engine: Engine[int],
    step: WorkflowStep[Plan, Tunables, State, int, int],
    tmp_path: Path,
) -> None:
    engine.start(
        step,
        plan=Plan(),
        tunables=Tunables(),
        paths=RunPaths(tmp_path / "metadata", tmp_path / "data"),
        identity=RunIdentity(
            "run",
            DeviceSnapshot({}),
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


@pytest.mark.parametrize("rethrow", [False, True])
def test_engine_live_failure_is_not_recoverable_acquisition_failure(
    plots: Plots, tmp_path: Path, rethrow: bool
) -> None:
    cause = ValueError("plot callback failed")
    observed: list[Completed[Cfg, int]] = []

    def acquire(run: Run[Cfg]) -> int:
        buffer = run.buffer((1,), axes=(np.array([0.0]),))
        with run.schedule(buffer) as schedule:
            schedule.batch({"write": lambda _step: buffer.set(np.array([2.0 + 0j]))})
        if rethrow:
            run.cancel_signal.raise_if_error()
        return 1

    def live(
        _axes: tuple[NDArray[np.float64], ...], _samples: NDArray[np.complex128]
    ) -> None:
        raise cause

    def init(_env: InitEnv[int], _plan: Plan) -> State:
        return State()

    @workflow("live_failure", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del plan, tun
        result = yield from env.run(acquire, Cfg(), save=save_integer, live=live)
        if isinstance(result, Completed):
            observed.append(result)
        return Next(1, state)

    engine = Engine(EnginePorts(plots, RecordingBar, ManualClock(), context=7))
    _start(engine, step, tmp_path)
    status = engine.execute("run")
    assert status.lifecycle == "failed"
    assert "plot callback failed" in str(status.reason)
    assert status.committed_seq == 0
    assert observed == []
    assert not tuple((tmp_path / "data").rglob("*.h5"))


def test_engine_reuses_injected_plots_across_steps_and_saves_live_figure(
    plots: Plots, tmp_path: Path
) -> None:
    viewer = plots.liveplot_1d("trace", "Time", "Signal")
    seen: list[Plots] = []

    def acquire(run: Run[Cfg]) -> int:
        buffer = run.buffer((2,), axes=(np.array([0.0, 1.0]),))
        buffer.set(np.array([1.0 + 2.0j, 3.0 + 4.0j]))
        return 1

    def init(_env: InitEnv[int], _plan: Plan) -> State:
        return State()

    @workflow("live_trace", plan=Plan, tunables=Tunables, init=init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del plan, tun
        seen.append(env.plots)
        if state.index:
            env.plots.save("trace", env.iter_dir / "trace.png")
            return Done()
        result = yield from env.run(
            acquire,
            Cfg(),
            save=save_integer,
            live=lambda axes, samples: viewer.update(axes[0], samples.real),
        )
        assert isinstance(result, Completed)
        state.index += 1
        return Next(result.result, state)

    engine = Engine(EnginePorts(plots, RecordingBar, ManualClock(), context=7))
    _start(engine, step, tmp_path)
    status = engine.execute("run")
    assert status.lifecycle == "done"
    assert status.committed_seq == 1
    assert len(seen) == 2 and all(owner is plots for owner in seen)
    np.testing.assert_array_equal(
        plots["trace"].axes[0].lines[0].get_ydata(), [1.0, 3.0]
    )
    saved = tuple((tmp_path / "data").rglob("trace.png"))
    assert len(saved) == 1 and saved[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
