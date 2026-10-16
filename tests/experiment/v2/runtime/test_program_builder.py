from __future__ import annotations

import numpy as np
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.program.v2 import ProgramV2Cfg

from tests.experiment.v2.runtime._flow_cfg import FlowCfg
from tests.experiment.v2.runtime._flow_cfg import make_flow_cfg as _cfg
from tests.experiment.v2.runtime._flow_fakes import FakeModule, FakeProgram


def test_program_builder_build_returns_program_with_isolated_cfg():
    program = None
    fresh_program = None
    FakeProgram.instances.clear()
    module = FakeModule("readout")
    init_cfg = _cfg(reps=5, rounds=7)

    with Schedule(
        init_cfg, SignalBuffer((1,), dtype=np.float64), stop=StopSignal()
    ) as sched:
        builder = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeProgram, marker="kw")
            .add(module)
            .declare_sweep("loop", 3)
        )
        program = builder.build()
        assert isinstance(program, FakeProgram)
        assert type(program.cfg_model) is FlowCfg
        program.cfg_model.reps = 99
        assert getattr(sched.cfg, "reps") == 5

    assert getattr(init_cfg, "reps") == 5
    assert program is not None
    assert program.soccfg == "soccfg"
    assert program.modules[0] is not module
    assert program.modules[0].name == "readout"
    assert program.sweep == [("loop", 3)]
    assert program.marker == "kw"
    assert program.reps == 5
    assert program.rounds == 7

    with Schedule(
        init_cfg, SignalBuffer((1,), dtype=np.float64), stop=StopSignal()
    ) as sched:
        fresh_program = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeProgram, marker="kw")
            .add(module)
            .build()
        )

    assert isinstance(fresh_program, FakeProgram)
    assert fresh_program.reps == 5


def test_program_builder_projects_mapping_cfg_to_program_cfg() -> None:
    program = None
    FakeProgram.instances.clear()
    init_cfg = {"reps": 3, "rounds": 4, "experiment_only": "ignored"}

    with Schedule(
        init_cfg, SignalBuffer((1,), dtype=np.float64), stop=StopSignal()
    ) as sched:
        program = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeProgram, marker="kw")
            .add(FakeModule("readout"))
            .build()
        )

    assert program is not None
    assert isinstance(program.cfg_model, ProgramV2Cfg)
    assert type(program.cfg_model) is ProgramV2Cfg
    assert program.reps == 3
    assert program.rounds == 4
    assert not hasattr(program.cfg_model, "experiment_only")


def test_program_builder_rejects_cfg_without_program_fields() -> None:
    with Schedule(
        {"experiment_only": "ignored"},
        SignalBuffer((1,), dtype=np.float64),
        stop=StopSignal(),
    ) as sched:
        builder = sched.prog_builder(
            "soc", "soccfg", program_cls=FakeProgram, marker="kw"
        ).add(FakeModule("readout"))
        try:
            builder.build()
        except TypeError as exc:
            assert "ProgramV2Cfg" in str(exc)
        else:
            raise AssertionError(
                "ProgramBuilder should reject cfg without runtime fields"
            )


def test_program_builder_accepts_cfg_override() -> None:
    program = None
    FakeProgram.instances.clear()

    with Schedule(
        {"reps": 1, "rounds": 1},
        SignalBuffer((1,), dtype=np.float64),
        stop=StopSignal(),
    ) as sched:
        program = (
            sched.prog_builder(
                "soc",
                "soccfg",
                cfg=_cfg(reps=8, rounds=9),
                program_cls=FakeProgram,
                marker="kw",
            )
            .add(FakeModule("readout"))
            .build()
        )

    assert program is not None
    assert program.reps == 8
    assert program.rounds == 9


def test_schedule_requires_explicit_modules():
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, stop=StopSignal()) as sched:
        builder = sched.prog_builder("soc", "soccfg", marker="x")
        try:
            builder.build()
        except ValueError as exc:
            assert "explicit modules" in str(exc)
        else:
            raise AssertionError("ProgramBuilder.build should require modules")
