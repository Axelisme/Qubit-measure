from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer

from tests.experiment.v2.runtime._flow_cfg import make_flow_cfg as _cfg
from tests.experiment.v2.runtime._flow_fakes import (
    FakeModule,
    FakeScalarProgram,
    identity_array,
)


@dataclass
class FlowEnv:
    label: str
    scale: float


def test_schedule_scan_targets_buffer_step_without_mutating_env():
    signals_buffer = SignalBuffer((3,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, scope=Scope("run")) as sched:
        for value, step in sched.scan("value", [10.0, 20.0, 30.0]):
            assert step.value == value
            assert isinstance(step.index, int)
            # tolerated: FlowCfg stores value as an undeclared Pydantic extra.
            setattr(step.cfg, "value", value)  # noqa: B010
            step.prog_builder(
                "soc",
                "soccfg",
                program_cls=FakeScalarProgram,
                value=step.cfg.value,
            ).add(FakeModule("readout")).build_and_acquire(
                raw2signal_fn=identity_array,
            )

        assert sched.env == {}

    assert np.allclose(signals_buffer.array, [10.0, 20.0, 30.0])


def test_schedule_repeat_targets_step_buffer_without_mutating_env():
    signals_buffer = SignalBuffer((3,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, scope=Scope("run")) as sched:
        for index, step in sched.repeat("round", 3, interval=0.0):
            assert step.index == index
            assert step.value == index
            assert step.path == (index,)
            # tolerated: FlowCfg stores value as an undeclared Pydantic extra.
            setattr(step.cfg, "value", index)  # noqa: B010
            step.prog_builder(
                "soc",
                "soccfg",
                program_cls=FakeScalarProgram,
                value=step.cfg.value,
            ).add(FakeModule("readout")).build_and_acquire(
                raw2signal_fn=identity_array,
            )

    assert np.allclose(signals_buffer.array, [0.0, 1.0, 2.0])


def test_schedule_env_accepts_dataclass_context():
    env = FlowEnv(label="typed", scale=2.0)

    with Schedule(_cfg(), env=env, scope=Scope("run")) as sched:
        assert sched.env.label == "typed"
        for value, step in sched.scan("value", [3.0]):
            assert step.env is env
            assert step.env.scale * value == 6.0


def test_schedule_nested_repeat_scan_targets_inner_step_buffer():
    signals_buffer = SignalBuffer((2, 3), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, scope=Scope("run")) as sched:
        for repeat_index, repeat_step in sched.repeat("round", 2):
            for length_value, step in repeat_step.scan("length", [10.0, 20.0, 30.0]):
                assert step.path == (repeat_index, step.index)
                # tolerated: FlowCfg stores value as an undeclared Pydantic extra.
                setattr(step.cfg, "value", repeat_index * 100.0 + length_value)  # noqa: B010
                step.prog_builder(
                    "soc",
                    "soccfg",
                    program_cls=FakeScalarProgram,
                    value=step.cfg.value,
                ).add(FakeModule("readout")).build_and_acquire(
                    raw2signal_fn=identity_array,
                )

    assert np.allclose(
        signals_buffer.array,
        [[10.0, 20.0, 30.0], [110.0, 120.0, 130.0]],
    )


def test_schedule_repeat_validates_times_and_interval():
    with Schedule(_cfg(), scope=Scope("run")) as sched:
        try:
            list(sched.repeat("round", -1))
        except ValueError as exc:
            assert "times" in str(exc)
        else:
            raise AssertionError("repeat should reject negative times")

        try:
            list(sched.repeat("round", 1, interval=-0.1))
        except ValueError as exc:
            assert "interval" in str(exc)
        else:
            raise AssertionError("repeat should reject negative interval")

