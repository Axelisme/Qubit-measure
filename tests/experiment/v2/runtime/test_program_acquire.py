from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.script_host import CallContext, ScriptHost
from zcu_tools.experiment.v2.runtime.program_builder import AcquireOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.program.acquisition import CancelFlagProtocol, StoppedPartialAcquireError
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


@pytest.mark.parametrize("decimated", [False, True])
@pytest.mark.parametrize("before_data", [False, True])
def test_cancel_one_acquire_keeps_partial_and_next_acquire_runs_normally(
    decimated: bool,
    before_data: bool,
) -> None:
    host = ScriptHost(Scope("workflow", name="test"))
    paths: list[str] = []
    produced: list[tuple[int, int]] = []
    results: list[NDArray[np.float64]] = []

    def operation(ctx: CallContext) -> None:
        with Schedule(
            _cfg(rounds=3), SignalBuffer(1, dtype=np.float64), scope=ctx.scope
        ) as sched:
            assert sched.stop is not ctx.scope.stop, "Schedule must own its stop"

            class Program:
                calls = 0

                def acquire(
                    self,
                    soc: object,
                    *,
                    rounds: int,
                    progress: bool,
                    round_hook: Callable[
                        [int, NDArray[np.float64], CancelFlagProtocol], object
                    ],
                    cancel_flag: CancelFlagProtocol,
                    **_kwargs: object,
                ) -> NDArray[np.float64]:
                    self.calls += 1
                    assert sched.scope.active_path.endswith(
                        "acquire#" + str(self.calls)
                    )
                    paths.append(host.active_path)
                    if self.calls == 1 and before_data:
                        assert host.cancel("acquire").cancelled
                        raise StoppedPartialAcquireError("no completed round")
                    raw = np.array([np.nan])
                    for ir in range(1, rounds + 1):
                        if cancel_flag.is_set():
                            break
                        raw = np.array([float(ir)])
                        produced.append((self.calls, ir))
                        round_hook(ir, raw, cancel_flag)
                        if self.calls == 1:
                            assert host.cancel("acquire").cancelled
                    return raw

                acquire_decimated = acquire

            builder = sched.prog_builder("soc", "soccfg")
            runner = builder.run_program_decimated if decimated else builder.run_program
            program = Program()
            for _ in range(2):
                results.append(
                    runner(
                        program,
                        rounds=3,
                        raw2signal_fn=identity_array,
                        options=AcquireOptions(retry=3),
                    ).copy()
                )
                assert sched.scope.active_path == sched.scope.path
                assert not sched.is_stop()
                assert sched.outcome.status == "completed"
                assert sched.stop.error is None
            assert program.calls == 2
        assert not ctx.scope.stop.is_set()

    async def script() -> None:
        result = await host.call(operation)
        assert result.partial_result
        later = await host.call(lambda ctx: ctx.scope.stop.partial_result)
        assert later.result is False
        assert later.partial_result is False

    host.run(script())
    np.testing.assert_allclose(results[0], [np.nan if before_data else 1.0])
    np.testing.assert_allclose(results[1], [3.0])
    assert produced == ([] if before_data else [(1, 1)]) + [(2, 1), (2, 2), (2, 3)]
    assert paths == [
        "workflow:test/operation#1/schedule#1/acquire#1",
        "workflow:test/operation#1/schedule#1/acquire#2",
    ]


def test_stop_condition_completes_without_partial_or_scope_cancellation() -> None:
    host = ScriptHost(Scope("workflow", name="test"))

    def operation(ctx: CallContext) -> None:
        with Schedule(
            _cfg(rounds=2), SignalBuffer(1, dtype=np.float64), scope=ctx.scope
        ) as sched:
            assert sched.stop is not ctx.scope.stop
            builder = sched.prog_builder(
                "soc", "soccfg", program_cls=FakeProgram, marker="kw"
            ).add(FakeModule("readout"))
            builder.build_and_acquire(
                rounds=2,
                raw2signal_fn=identity_array,
                options=AcquireOptions(stop_condition=lambda: True),
            )
            assert not sched.stop.partial_result
            assert not sched.is_stop()
            assert sched.outcome.status == "completed"
        assert not ctx.scope.stop.partial_result

    async def script() -> None:
        assert not (await host.call(operation)).partial_result

    host.run(script())


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
    with Schedule(_cfg(reps=4, rounds=2), signals_buffer, scope=Scope("run")) as sched:
        result = (
            sched.prog_builder(
                "soc", "soccfg", reps=4, program_cls=FakeProgram, marker="kw"
            )
            .add(module)
            .declare_sweep("loop", 3)
            .build_and_acquire(
                rounds=2,
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
    assert instance.acquire_kwargs == {"tag": "x", "rounds": 2}
    assert instance.cancel_flag_seen is True


def test_program_builder_progress_leave_defaults_root_and_can_be_overridden():
    progress_bars: list[RecordingProgressBar] = []

    def run_root_acquire(progress_leave: bool | None = None) -> None:
        with Schedule(
            _cfg(rounds=2), SignalBuffer((1,), dtype=np.float64), scope=Scope("run")
        ) as sched:
            builder = sched.prog_builder(
                "soc",
                "soccfg",
                program_cls=FakeProgram,
                marker="kw",
            ).add(FakeModule("readout"))
            if progress_leave is None:
                builder.build_and_acquire(rounds=2, raw2signal_fn=identity_array)
            else:
                builder.build_and_acquire(
                    rounds=2,
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
        Schedule(_cfg(rounds=2), signals_buffer, scope=Scope("run")) as sched,
    ):
        for _, step in sched.scan("host", [0]):
            step.prog_builder(
                "soc", "soccfg", program_cls=FakeProgram, marker="kw"
            ).add(FakeModule("readout")).build_and_acquire(
                rounds=2,
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

    with Schedule(_cfg(rounds=1), signals_buffer, scope=Scope("run")) as sched:
        for value, step in sched.scan("value", [1.0, 1.0, 2.0]):
            # tolerated: FlowCfg.value is a dynamic Pydantic extra, not a typed field.
            setattr(step.cfg, "value", value)  # noqa: B010
            builder = step.prog_builder(
                "soc",
                "soccfg",
                program_cls=FakeCachedProgram,
                value=step.cfg.value,
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

    with Schedule(_cfg(reps=1, rounds=2), signals_buffer, scope=Scope("run")) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeDecimatedProgram)
            .add(FakeModule("readout"))
            .build_and_acquire_decimated(
                rounds=2,
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
    assert instance.acquire_decimated_kwargs == {"tag": "trace", "rounds": 2}
    assert instance.cancel_flag_seen is True


def test_schedule_registers_program_derived_buffer_before_decimated_run():
    result = None
    signals_buffer = None
    FakeDecimatedProgram.instances.clear()
    buffer_updates: list[np.ndarray] = []

    with Schedule(_cfg(reps=1, rounds=2), scope=Scope("run")) as sched:
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
        result = builder.run_program_decimated(program, rounds=2)

    assert result is not None
    assert np.allclose(result, [2.0 + 0.0j, 0.0 + 2.0j])
    assert signals_buffer is not None
    assert np.allclose(signals_buffer.array, [2.0 + 0.0j, 0.0 + 2.0j])
    assert np.allclose(buffer_updates[-1], [2.0 + 0.0j, 0.0 + 2.0j])
    assert len(FakeDecimatedProgram.instances) == 1
