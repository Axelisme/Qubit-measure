from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.runtime.schedule import (
    AcquireOptions,
    Schedule,
    SignalBuffer,
)
from zcu_tools.progress_bar import use_pbar_factory

from tests.experiment.v2.runtime._flow_cfg import make_flow_cfg as _cfg
from tests.experiment.v2.runtime._flow_fakes import (
    FakeCachedProgram,
    FakeDecimatedProgram,
    FakeModule,
    FakeProgram,
    RecordingProgressBar,
    identity_array,
    recording_pbar_factory,
)


def test_build_and_acquire_builds_program_and_updates_buffer():
    result = None
    FakeProgram.instances.clear()
    buffer_updates: list[np.ndarray] = []
    module = FakeModule("readout")
    signals_buffer = SignalBuffer(
        (1,),
        dtype=np.float64,
        on_update=lambda data: buffer_updates.append(data.copy()),
        update_interval=None,
    )
    with Schedule(_cfg(reps=4, rounds=2), signals_buffer, stop=StopSignal()) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeProgram, marker="kw")
            .add(module)
            .declare_sweep("loop", 3)
            .build_and_acquire(
                raw2signal_fn=identity_array,
                tag="x",
                options=AcquireOptions(stop_condition=lambda: False),
            )
        )
        assert sched.cfg.reps == 4
        assert sched.cfg.rounds == 2

    assert result is not None
    assert np.allclose(result, [2.0])
    assert np.allclose(signals_buffer.array, [2.0])
    assert np.allclose(buffer_updates[0], [1.0])
    assert np.allclose(buffer_updates[-1], [2.0])

    assert len(FakeProgram.instances) == 1
    instance = FakeProgram.instances[0]
    assert instance.soccfg == "soccfg"
    assert instance.modules[0] is not module
    assert instance.modules[0].name == "readout"
    assert instance.sweep == [("loop", 3)]
    assert instance.marker == "kw"
    assert instance.reps == 4
    assert instance.rounds == 2
    assert instance.acquire_kwargs == {"tag": "x"}
    assert instance.cancel_flag_seen is True


def test_program_builder_progress_leave_defaults_root_and_can_be_overridden():
    progress_bars: list[RecordingProgressBar] = []

    def run_root_acquire(progress_leave: bool | None = None) -> None:
        with Schedule(
            _cfg(rounds=2), SignalBuffer((1,), dtype=np.float64), stop=StopSignal()
        ) as sched:
            builder = sched.prog_builder(
                "soc",
                "soccfg",
                program_cls=FakeProgram,
                marker="kw",
            ).add(FakeModule("readout"))
            if progress_leave is None:
                builder.build_and_acquire(raw2signal_fn=identity_array)
            else:
                builder.build_and_acquire(
                    raw2signal_fn=identity_array,
                    options=AcquireOptions(progress_leave=progress_leave),
                )

    with use_pbar_factory(recording_pbar_factory(progress_bars)):
        run_root_acquire()
        run_root_acquire(progress_leave=False)

    assert [bar.leave for bar in progress_bars] == [True, False]
    assert all(bar.closed for bar in progress_bars)


@pytest.mark.parametrize("progress_leave", [None, True, False])
def test_step_program_progress_closes_with_default_or_explicit_leave(
    progress_leave: bool | None,
) -> None:
    progress_bars: list[RecordingProgressBar] = []
    signals_buffer = SignalBuffer((1, 1), dtype=np.float64)

    with (
        use_pbar_factory(recording_pbar_factory(progress_bars)),
        Schedule(_cfg(rounds=2), signals_buffer, stop=StopSignal()) as sched,
    ):
        for _, step in sched.scan("host", [0]):
            step.prog_builder(
                "soc", "soccfg", program_cls=FakeProgram, marker="kw"
            ).add(FakeModule("readout")).build_and_acquire(
                raw2signal_fn=identity_array,
                options=AcquireOptions(progress_leave=progress_leave),
            )
            rounds_bar = next(bar for bar in progress_bars if bar.desc == "rounds")
            assert rounds_bar.leave is (
                False if progress_leave is None else progress_leave
            )
            assert rounds_bar.closed
            assert not progress_bars[0].closed

    assert all(bar.closed for bar in progress_bars)
    np.testing.assert_allclose(signals_buffer.array, [[2.0]])


def test_run_program_reuses_caller_owned_program_cache():
    FakeCachedProgram.instances.clear()
    signals_buffer = SignalBuffer((3,), dtype=np.float64)
    programs: dict[float, FakeCachedProgram] = {}

    with Schedule(_cfg(rounds=1), signals_buffer, stop=StopSignal()) as sched:
        for value, step in sched.scan("value", [1.0, 1.0, 2.0]):
            # tolerated: FlowCfg.value is a dynamic Pydantic extra, not a typed field.
            setattr(step.cfg, "value", value)  # noqa: B010
            builder = step.prog_builder(
                "soc",
                "soccfg",
                program_cls=FakeCachedProgram,
            ).add(FakeModule("readout"))
            if value not in programs:
                program = builder.build()
                assert isinstance(program, FakeCachedProgram)
                programs[value] = program
            builder.run_program(
                programs[value],
                raw2signal_fn=identity_array,
            )

    assert np.allclose(signals_buffer.array, [1.0, 1.0, 2.0])
    assert len(FakeCachedProgram.instances) == 2
    assert FakeCachedProgram.instances[0].acquire_count == 2
    assert FakeCachedProgram.instances[1].acquire_count == 1
    assert set(programs) == {1.0, 2.0}


def test_build_and_acquire_decimated_uses_decimated_method_and_default_conversion():
    result = None
    FakeDecimatedProgram.instances.clear()
    buffer_updates: list[np.ndarray] = []
    signals_buffer = SignalBuffer(
        (2,),
        on_update=lambda data: buffer_updates.append(data.copy()),
        update_interval=None,
    )

    with Schedule(_cfg(reps=1, rounds=2), signals_buffer, stop=StopSignal()) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeDecimatedProgram)
            .add(FakeModule("readout"))
            .build_and_acquire_decimated(
                tag="trace",
                options=AcquireOptions(stop_condition=lambda: False),
            )
        )

    assert result is not None
    assert np.allclose(result, [2.0 + 0.0j, 0.0 + 2.0j])
    assert np.allclose(signals_buffer.array, [2.0 + 0.0j, 0.0 + 2.0j])
    assert np.allclose(buffer_updates[0], [1.0 + 0.0j, 0.0 + 1.0j])
    assert np.allclose(buffer_updates[-1], [2.0 + 0.0j, 0.0 + 2.0j])

    assert len(FakeDecimatedProgram.instances) == 1
    instance = FakeDecimatedProgram.instances[0]
    assert instance.acquire_decimated_kwargs == {"tag": "trace"}
    assert instance.cancel_flag_seen is True


def test_schedule_registers_program_derived_buffer_before_decimated_run():
    result = None
    signals_buffer = None
    FakeDecimatedProgram.instances.clear()
    buffer_updates: list[np.ndarray] = []

    with Schedule(_cfg(reps=1, rounds=2), stop=StopSignal()) as sched:
        builder = sched.prog_builder(
            "soc",
            "soccfg",
            program_cls=FakeDecimatedProgram,
        ).add(FakeModule("readout"))
        program = builder.build()
        assert isinstance(program, FakeDecimatedProgram)
        times = program.get_time_axis(ro_index=0)
        signals_buffer = SignalBuffer(
            (len(times),),
            on_update=lambda data: buffer_updates.append(data.copy()),
            update_interval=None,
        )
        sched.register_buffer(signals_buffer)
        result = builder.run_program_decimated(program)

    assert result is not None
    assert np.allclose(result, [2.0 + 0.0j, 0.0 + 2.0j])
    assert signals_buffer is not None
    assert np.allclose(signals_buffer.array, [2.0 + 0.0j, 0.0 + 2.0j])
    assert np.allclose(buffer_updates[-1], [2.0 + 0.0j, 0.0 + 2.0j])
    assert len(FakeDecimatedProgram.instances) == 1
