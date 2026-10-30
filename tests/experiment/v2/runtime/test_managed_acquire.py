"""Schedule-owned acquisition, independently of program construction."""

from collections.abc import Callable, Iterator

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.acquisition_facts import StopDecision
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.simulation import SimulationEnvironment, white_noise
from zcu_tools.experiment.v2.runtime import ManageOptions, Schedule, SignalBuffer
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.progress_bar import use_pbar_factory

from tests.experiment.v2.runtime._flow_fakes import (
    RecordingProgressBar,
    recording_pbar_factory,
)
from tests.program._acquisition_soc import make_chunk_acquisition


@pytest.fixture
def simulated_program() -> Iterator[tuple[SimulationEnvironment, ModularProgramV2]]:
    """Borrow a seeded, zero-noise real program and close its environment after use."""
    env = white_noise(std_adc=0, seed=11)
    readout = PulseReadoutCfg(
        gen_ch=0,
        nqz=1,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_ch=0,
        ro_length=0.1,
        freq=100.0,
    ).build("readout")
    program = ModularProgramV2(env.soccfg, modules=[readout], reps=2)
    try:
        yield env, program
    finally:
        env.devices.close_all_devices()


def test_managed_integrated_writes_scan_slot_and_notifies_live_buffer(
    simulated_program: tuple[SimulationEnvironment, ModularProgramV2],
) -> None:
    env, program = simulated_program
    updates = []
    buffer = SignalBuffer(
        (2, 1), on_update=lambda data: updates.append(data.copy()), update_interval=1
    )
    with Schedule({}, buffer, scope=Scope("run")) as schedule:
        for _, step in schedule.scan("point", [10, 20]):
            result = step.managed_acquire(program, env.soc, rounds=2)
            np.testing.assert_array_equal(result, [0j])
            assert np.shares_memory(result, buffer.array)
    np.testing.assert_array_equal(buffer.array, [[0j], [0j]])
    assert len(updates) == 6  # Two completed rounds plus final per point.
    assert np.isnan(updates[0][1]).all()


def test_managed_summed_writes_raw_cumulative_sums_by_default(
    simulated_program: tuple[SimulationEnvironment, ModularProgramV2],
) -> None:
    env, program = simulated_program
    updates = []
    buffer = SignalBuffer(
        (1,),
        dtype=np.float64,
        on_update=lambda data: updates.append(data.copy()),
        update_interval=1,
    )

    def count_shots(iq):
        return np.ones((*iq[0].shape[:2], 1), dtype=np.float64).sum(axis=1)

    with Schedule({}, buffer, scope=Scope("run")) as schedule:
        result = schedule.managed_acquire_summed(
            program, env.soc, processor=count_shots, rounds=2
        )
        np.testing.assert_array_equal(result, [4.0])
        assert np.shares_memory(result, buffer.array)
    np.testing.assert_array_equal(updates, [[2.0], [4.0], [4.0]])


def test_managed_decimated_writes_default_complex_trace_and_live_updates(
    simulated_program: tuple[SimulationEnvironment, ModularProgramV2],
) -> None:
    env, program = simulated_program
    samples = program.ro_chs[0]["length"]
    updates = []
    buffer = SignalBuffer(
        (2, samples),
        on_update=lambda data: updates.append(data.copy()),
        update_interval=1,
    )
    with Schedule({}, buffer, scope=Scope("run")) as schedule:
        result = schedule.managed_acquire_decimated(program, env.soc, rounds=2)
        np.testing.assert_array_equal(result, np.zeros((2, samples), dtype=complex))
        assert np.shares_memory(result, buffer.array)
    assert len(updates) == 3
    np.testing.assert_array_equal(buffer.array, np.zeros((2, samples), dtype=complex))


@pytest.mark.parametrize("mode", ["integrated", "decimated", "summed"])
def test_managed_converter_is_selected_by_none_not_callable_truthiness(
    simulated_program: tuple[SimulationEnvironment, ModularProgramV2],
    mode: str,
) -> None:
    env, program = simulated_program
    buffer = SignalBuffer((1,))

    class Converter:
        """A valid converter whose truthiness does not indicate availability."""

        def __bool__(self) -> bool:
            return False

        def __call__(self, raw: object) -> np.ndarray:
            return np.array([-7.0])

    with Schedule({}, buffer, scope=Scope("run")) as schedule:
        if mode == "integrated":
            result = schedule.managed_acquire(
                program, env.soc, options=ManageOptions(converter=Converter())
            )
        elif mode == "decimated":
            result = schedule.managed_acquire_decimated(
                program, env.soc, options=ManageOptions(converter=Converter())
            )
        else:
            result = schedule.managed_acquire_summed(
                program,
                env.soc,
                processor=lambda iq: np.ones((1, 1)),
                options=ManageOptions(converter=Converter()),
            )
        np.testing.assert_array_equal(result, [-7.0])


def _policy_poll(
    program: ModularProgramV2,
    root: Scope,
    ending: str,
) -> Callable[[], list[tuple[int, tuple[list[NDArray[np.int64]], dict[str, object]]]]]:
    """Return deterministic transport rounds with the requested completion boundary."""
    calls = 0

    def poll():
        nonlocal calls
        calls += 1
        if ending == "before_data":
            root.stop.set()
            return []
        if ending == "retry" and calls == 1:
            raise RuntimeError("transport unavailable")
        if ending in {"failure", "interrupt"} and calls == 2:
            if ending == "interrupt":
                raise KeyboardInterrupt
            raise RuntimeError("transport failed after one round")
        first = calls == (2 if ending == "retry" else 1)
        iq = [3, 4] if first else [9, 12]
        raw = np.tile(iq, (2, 1)) * program.ro_chs[0]["length"]
        return [(2, ([raw.astype(np.int64)], {}))]

    return poll


@pytest.mark.parametrize(
    "ending",
    [
        "complete",
        "retry",
        "data_stop",
        "external_stop",
        "before_data",
        "interrupt",
        "failure",
    ],
)
def test_managed_integrated_policy_keeps_completed_rounds_and_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    ending: str,
) -> None:
    soc, program = make_chunk_acquisition(reps=2, reads=1)
    root = Scope("run")
    converted = []
    observed = []
    bars: list[RecordingProgressBar] = []
    caplog.set_level("INFO", logger="zcu_tools.experiment.v2.runtime.program_builder")
    buffer = SignalBuffer((1,))

    def convert(raw: list[NDArray[np.float64]]) -> NDArray[np.complex128]:
        converted.append(raw[0].copy())
        if ending == "external_stop":
            root.stop.set()
        return raw[0][0].dot([1, 1j]).reshape(1)

    def stop() -> StopDecision:
        observed.append(buffer.array.copy())
        return StopDecision(met=ending == "data_stop", value=12.5)

    monkeypatch.setattr(soc, "poll_data", _policy_poll(program, root, ending))
    result = buffer.array
    with (
        use_pbar_factory(recording_pbar_factory(bars)),
        Schedule({}, buffer, scope=root) as schedule,
    ):
        result = schedule.managed_acquire(
            program,
            soc,
            rounds=2,
            options=ManageOptions(
                retry=int(ending == "retry"), converter=convert, stop_condition=stop
            ),
        )
        expected_status = {
            "external_stop": "stopped",
            "before_data": "stopped",
            "interrupt": "interrupted",
            "failure": "failed",
        }.get(ending, "completed")
        assert schedule.outcome.status == expected_status
        if ending == "data_stop":
            assert not schedule.stop.is_set()
            assert not root.stop.is_set()
            np.testing.assert_array_equal(observed, [[3 + 4j]])
    if ending == "before_data":
        assert converted == []
        assert np.isnan(result).all()
    elif ending in {"complete", "retry"}:
        np.testing.assert_array_equal(result, [6 + 8j])
        assert len(converted) == 3  # Both cumulative rounds and the final result.
    else:
        np.testing.assert_array_equal(result, [3 + 4j])
        assert len(converted) == (2 if ending == "data_stop" else 1)
    rounds_bar = next(bar for bar in bars if bar.desc == "rounds")
    assert rounds_bar.total == 2
    assert rounds_bar.reset_count == (2 if ending == "retry" else 1)
    assert rounds_bar.leave
    assert rounds_bar.closed
    messages = [record.message for record in caplog.records]
    assert any("attempt 2 after RuntimeError" in message for message in messages) == (
        ending == "retry"
    )
    assert any(
        "stop value 12.5, completed rounds 1" in message for message in messages
    ) == (ending == "data_stop")
