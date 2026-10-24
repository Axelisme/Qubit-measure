from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.progress_bar import use_pbar_factory

from tests.experiment.v2.runtime._flow_cfg import make_flow_cfg as _cfg
from tests.experiment.v2.runtime._flow_fakes import (
    FakeModule,
    FakeProgram,
    RecordingProgressBar,
    identity_array,
    recording_pbar_factory,
)


@pytest.mark.parametrize("step_owner", [False, True])
def test_builder_passes_controls_and_isolates_modules_sweeps_and_schedule_cfg(
    step_owner,
):
    from zcu_tools.program.v2 import SweepCfg

    cfg = _cfg(reps=99)
    module = FakeModule("readout")
    sweep = SweepCfg(start=1.0, stop=2.0, step=0.5, expts=3)
    with Schedule(cfg, SignalBuffer(1), scope=Scope("run")) as sched:
        repeat = sched.repeat("host", 1)
        owner = list(repeat)[0][1] if step_owner else sched
        builder = (
            owner.prog_builder(
                "soc",
                "soccfg",
                reps=5,
                initial_delay=0.25,
                relax_delay=12.5,
                program_cls=FakeProgram,
                marker="kw",
            )
            .add(module)
            .declare_sweep("freq", sweep)
        )
        first = builder.build()
        first.modules[0].name = "changed"
        first.sweep.clear()
        sched.cfg.reps = 8
        second = builder.build()
        assert second.modules[0].name == "readout"
        assert second.sweep[0][1].start == 1.0
        assert second.reps == 5
        assert second.initial_delay == 0.25
        assert second.relax_delay == 12.5
    assert cfg.reps == 99
    assert module.name == "readout"
    assert sweep.start == 1.0


def test_program_builder_skips_absent_modules_and_preserves_order() -> None:
    with Schedule(_cfg(), SignalBuffer(1), scope=Scope("run")) as sched:
        program = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeProgram, marker="kw")
            .add(None, FakeModule("drive"), [None, FakeModule("readout"), None])
            .add_reset("reset", None)
            .build()
        )
        assert [module.name for module in program.modules] == ["drive", "readout"]


@pytest.mark.parametrize("decimated", [False, True])
def test_run_program_passes_explicit_rounds_to_producer_and_retry_progress(decimated):
    calls = []
    bars: list[RecordingProgressBar] = []

    class Program:
        def acquire(self, soc, *, rounds=1, round_hook, cancel_flag, **kwargs):
            calls.append(rounds)
            if len(calls) == 1:
                raise RuntimeError("retry")
            for count in range(1, rounds + 1):
                round_hook(count, np.array([float(count)]), cancel_flag)
            return np.array([float(rounds)])

        acquire_decimated = acquire

    with (
        Schedule({}, SignalBuffer(1, dtype=np.float64), scope=Scope("run")) as sched,
        use_pbar_factory(recording_pbar_factory(bars)),
    ):
        builder = sched.prog_builder("soc", "soccfg")
        runner = builder.run_program_decimated if decimated else builder.run_program
        from zcu_tools.experiment.v2.runtime import AcquireOptions

        result = runner(
            Program(),
            rounds=3,
            options=AcquireOptions(retry=1),
            raw2signal_fn=identity_array,
        )
        np.testing.assert_array_equal(result, [3.0])
        assert sched.outcome.status == "completed"
    assert calls == [3, 3]
    assert bars[0].total == 3
    assert bars[0].n == 3
    assert bars[0].closed


def test_schedule_requires_explicit_modules():
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, scope=Scope("run")) as sched:
        builder = sched.prog_builder("soc", "soccfg", marker="x")
        try:
            builder.build()
        except ValueError as exc:
            assert "explicit modules" in str(exc)
        else:
            raise AssertionError("ProgramBuilder.build should require modules")
