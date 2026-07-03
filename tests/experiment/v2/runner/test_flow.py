from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import ConfigDict
from zcu_tools.experiment import ExpCfgModel
from zcu_tools.experiment.v2.runner import (
    Schedule,
    ScheduleStep,
    SignalBuffer,
    StopSignal,
    schedule_stop_scope,
)
from zcu_tools.program.v2 import Module, ProgramV2Cfg


class FlowCfg(ProgramV2Cfg, ExpCfgModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


@dataclass
class FlowEnv:
    label: str
    scale: float


def _identity_array(raw: np.ndarray) -> np.ndarray:
    return raw


def _cfg(**values: Any) -> FlowCfg:
    cfg = FlowCfg()
    for name, value in values.items():
        setattr(cfg, name, value)
    return cfg


class FakeModule(Module):
    def __init__(self, name: str) -> None:
        self.name = name

    def init(self, prog: Any) -> None:
        pass

    def run(self, prog: Any, t: Any = 0.0) -> Any:
        return t


class RecordingBuffer:
    def __init__(
        self,
        data: Any,
        *,
        on_update: Callable[[ScheduleStep[Any, Any, Any]], None] | None = None,
    ) -> None:
        self.data = data
        self._on_update = on_update

    def trigger_update(
        self,
        step: ScheduleStep[Any, Any, Any] | None = None,
        *,
        flush: bool = False,
    ) -> None:
        if step is not None and self._on_update is not None:
            self._on_update(step)


class RootUpdateRecordingBuffer:
    def __init__(self, data: Any) -> None:
        self.data = data
        self.calls: list[tuple[ScheduleStep[Any, Any, Any] | None, bool]] = []

    def trigger_update(
        self,
        step: ScheduleStep[Any, Any, Any] | None = None,
        *,
        flush: bool = False,
    ) -> None:
        self.calls.append((step, flush))


class FakeProgram:
    instances: list[FakeProgram] = []

    def __init__(
        self,
        soccfg: Any,
        cfg: FlowCfg,
        *,
        modules: list[Module],
        sweep: list[tuple[str, Any]] | None,
        marker: str,
    ) -> None:
        self.soccfg = soccfg
        self.cfg_model = cfg
        self.modules = modules
        self.sweep = sweep
        self.marker = marker
        self.reps = getattr(cfg, "reps", None)
        self.rounds = getattr(cfg, "rounds", None)
        self.acquire_kwargs: dict[str, Any] = {}
        self.stop_checker_count = 0
        FakeProgram.instances.append(self)

    def acquire(
        self,
        soc: Any,
        *,
        progress: bool,
        round_hook,
        stop_checkers,
        **kwargs: Any,
    ) -> np.ndarray:
        self.acquire_kwargs = dict(kwargs)
        self.stop_checker_count = len(stop_checkers)
        assert soc == "soc"
        assert not progress
        assert all(not checker() for checker in stop_checkers)
        round_hook(1, np.array([1.0]))
        return np.array([2.0])

    def acquire_decimated(
        self,
        soc: Any,
        *,
        progress: bool,
        round_hook,
        stop_checkers,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        raise NotImplementedError


class FakeScalarProgram:
    def __init__(
        self,
        soccfg: Any,
        cfg: FlowCfg,
        *,
        modules: list[Module],
        sweep: list[tuple[str, Any]] | None,
    ) -> None:
        self.modules = modules
        self.cfg_model = cfg
        self.sweep = sweep

    def acquire(self, *_args: Any, round_hook, **_kwargs: Any) -> np.ndarray:
        raw = np.array(float(getattr(self.cfg_model, "value")))
        round_hook(1, raw)
        return raw

    def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
        raise NotImplementedError


class FakeCachedProgram:
    instances: list[FakeCachedProgram] = []

    def __init__(
        self,
        soccfg: Any,
        cfg: FlowCfg,
        *,
        modules: list[Module],
        sweep: list[tuple[str, Any]] | None,
    ) -> None:
        self.soccfg = soccfg
        self.cfg_model = cfg
        self.modules = modules
        self.sweep = sweep
        self.acquire_count = 0
        FakeCachedProgram.instances.append(self)

    def acquire(self, *_args: Any, round_hook, **_kwargs: Any) -> np.ndarray:
        self.acquire_count += 1
        raw = np.array(float(getattr(self.cfg_model, "value")))
        round_hook(1, raw)
        return raw

    def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
        raise NotImplementedError


class FakeDecimatedProgram:
    instances: list[FakeDecimatedProgram] = []

    def __init__(
        self,
        soccfg: Any,
        cfg: FlowCfg,
        *,
        modules: list[Module],
        sweep: list[tuple[str, Any]] | None,
    ) -> None:
        self.soccfg = soccfg
        self.cfg_model = cfg
        self.modules = modules
        self.sweep = sweep
        self.acquire_decimated_kwargs: dict[str, Any] = {}
        self.stop_checker_count = 0
        FakeDecimatedProgram.instances.append(self)

    def acquire(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def acquire_decimated(
        self,
        soc: Any,
        *,
        progress: bool,
        round_hook,
        stop_checkers,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        self.acquire_decimated_kwargs = dict(kwargs)
        self.stop_checker_count = len(stop_checkers)
        assert soc == "soc"
        assert not progress
        assert all(not checker() for checker in stop_checkers)
        round_hook(1, [np.array([[1.0, 0.0], [0.0, 1.0]])])
        return [np.array([[2.0, 0.0], [0.0, 2.0]])]

    def get_time_axis(self, ro_index: int = 0) -> np.ndarray:
        return np.arange(2.0)


class FlakyBuildProgram:
    instances: list[FlakyBuildProgram] = []

    def __init__(
        self,
        soccfg: Any,
        cfg: FlowCfg,
        *,
        modules: list[Module],
        sweep: list[tuple[str, Any]] | None,
    ) -> None:
        self.cfg_model = cfg
        self.modules = modules
        self.sweep = sweep
        self.acquire_count = 0
        FlakyBuildProgram.instances.append(self)

    def acquire(self, *_args: Any, round_hook, **_kwargs: Any) -> np.ndarray:
        self.acquire_count += 1
        if len(FlakyBuildProgram.instances) == 1:
            raise RuntimeError("temporary failure")
        raw = np.array([4.0])
        round_hook(1, np.array([3.0]))
        return raw

    def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
        raise NotImplementedError


class AlwaysFailingProgram:
    instances: list[AlwaysFailingProgram] = []

    def __init__(
        self,
        soccfg: Any,
        cfg: FlowCfg,
        *,
        modules: list[Module],
        sweep: list[tuple[str, Any]] | None,
    ) -> None:
        self.cfg_model = cfg
        self.modules = modules
        self.sweep = sweep
        self.acquire_count = 0
        AlwaysFailingProgram.instances.append(self)

    def acquire(self, *_args: Any, round_hook, **_kwargs: Any) -> np.ndarray:
        self.acquire_count += 1
        raw = np.array([float(len(AlwaysFailingProgram.instances))])
        round_hook(1, raw)
        raise RuntimeError("permanent failure")

    def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
        raise NotImplementedError


class ConstructorFailingProgram:
    def __init__(
        self,
        soccfg: Any,
        cfg: FlowCfg,
        *,
        modules: list[Module],
        sweep: list[tuple[str, Any]] | None,
    ) -> None:
        if getattr(cfg, "value") == 1.0:
            raise RuntimeError("build failure")
        self.cfg_model = cfg
        self.modules = modules
        self.sweep = sweep

    def acquire(self, *_args: Any, round_hook, **_kwargs: Any) -> np.ndarray:
        raw = np.array([float(getattr(self.cfg_model, "value"))])
        round_hook(1, raw)
        return raw

    def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
        raise NotImplementedError


class InterruptingProgram:
    instances: list[InterruptingProgram] = []

    def __init__(
        self,
        soccfg: Any,
        cfg: FlowCfg,
        *,
        modules: list[Module],
        sweep: list[tuple[str, Any]] | None,
    ) -> None:
        self.soccfg = soccfg
        self.cfg_model = cfg
        self.modules = modules
        self.sweep = sweep
        self.acquire_count = 0
        InterruptingProgram.instances.append(self)

    def acquire(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        self.acquire_count += 1
        raise KeyboardInterrupt

    def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
        raise NotImplementedError


def test_schedule_stop_scope_supplies_default_stop_signal() -> None:
    stop = StopSignal()
    stop.set_stop()

    with schedule_stop_scope(stop):
        with Schedule(_cfg()) as sched:
            assert sched.is_stop()

    with Schedule(_cfg()) as sched:
        assert not sched.is_stop()


def test_schedule_child_buffer_syncs_result_buffer_without_throttling() -> None:
    root = {"task": {"signals": np.full((1,), np.nan)}}
    updates: list[tuple[tuple[Any, ...], np.ndarray]] = []
    result_buffer = RecordingBuffer(
        root,
        on_update=lambda step: updates.append(
            (step.path, root["task"]["signals"].copy())
        ),
    )

    def child(step):
        signals_step = step.child("signals")
        buffer = signals_step.buffer((1,), dtype=np.float64)
        buffer.set(np.array([1.0]))
        buffer.set(np.array([2.0]))

    with Schedule(_cfg(), result_buffer) as sched:
        sched.batch({"task": child})

    np.testing.assert_allclose(root["task"]["signals"], np.array([2.0]))
    assert [path for path, _ in updates] == [("task", "signals"), ("task", "signals")]
    np.testing.assert_allclose(updates[0][1], np.array([1.0]))
    np.testing.assert_allclose(updates[1][1], np.array([2.0]))


def test_program_builder_build_returns_program_with_isolated_cfg():
    FakeProgram.instances.clear()
    module = FakeModule("readout")
    init_cfg = _cfg(reps=5, rounds=7)

    with Schedule(init_cfg, SignalBuffer((1,), dtype=np.float64)) as sched:
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
    assert program.soccfg == "soccfg"
    assert program.modules[0] is not module
    assert program.modules[0].name == "readout"
    assert program.sweep == [("loop", 3)]
    assert program.marker == "kw"
    assert program.reps == 5
    assert program.rounds == 7

    with Schedule(init_cfg, SignalBuffer((1,), dtype=np.float64)) as sched:
        fresh_program = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeProgram, marker="kw")
            .add(module)
            .build()
        )

    assert isinstance(fresh_program, FakeProgram)
    assert fresh_program.reps == 5


def test_program_builder_projects_mapping_cfg_to_program_cfg() -> None:
    FakeProgram.instances.clear()
    init_cfg = {"reps": 3, "rounds": 4, "experiment_only": "ignored"}

    with Schedule(init_cfg, SignalBuffer((1,), dtype=np.float64)) as sched:
        program = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeProgram, marker="kw")
            .add(FakeModule("readout"))
            .build()
        )

    assert isinstance(program.cfg_model, ProgramV2Cfg)
    assert type(program.cfg_model) is ProgramV2Cfg
    assert program.reps == 3
    assert program.rounds == 4
    assert not hasattr(program.cfg_model, "experiment_only")


def test_program_builder_rejects_cfg_without_program_fields() -> None:
    with Schedule(
        {"experiment_only": "ignored"},
        SignalBuffer((1,), dtype=np.float64),
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
    FakeProgram.instances.clear()

    with Schedule(
        {"reps": 1, "rounds": 1}, SignalBuffer((1,), dtype=np.float64)
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

    assert program.reps == 8
    assert program.rounds == 9


def test_build_and_acquire_builds_program_and_updates_buffer():
    FakeProgram.instances.clear()
    buffer_updates: list[np.ndarray] = []
    module = FakeModule("readout")
    signals_buffer = SignalBuffer(
        (1,),
        dtype=np.float64,
        on_update=lambda data: buffer_updates.append(data.copy()),
        update_interval=None,
    )
    with Schedule(_cfg(reps=4, rounds=2), signals_buffer) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeProgram, marker="kw")
            .add(module)
            .declare_sweep("loop", 3)
            .build_and_acquire(
                raw2signal_fn=_identity_array,
                tag="x",
                stop_checkers=[lambda: False],
            )
        )
        assert getattr(sched.cfg, "reps") == 4
        assert getattr(sched.cfg, "rounds") == 2

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
    assert instance.stop_checker_count == 2


def test_build_and_acquire_rebuilds_program_on_retry():
    FlakyBuildProgram.instances.clear()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FlakyBuildProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=_identity_array, retry=1)
        )
        assert sched.outcome.status == "completed"

    assert np.allclose(result, [4.0])
    assert np.allclose(signals_buffer.array, [4.0])
    assert len(FlakyBuildProgram.instances) == 2
    assert FlakyBuildProgram.instances[0].acquire_count == 1
    assert FlakyBuildProgram.instances[1].acquire_count == 1


def test_build_and_acquire_returns_partial_on_keyboard_interrupt_without_retrying():
    InterruptingProgram.instances.clear()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=InterruptingProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=_identity_array, retry=3)
        )
        assert sched.is_stop() is True
        assert sched.outcome.status == "interrupted"
        assert isinstance(sched.outcome.exception, KeyboardInterrupt)
        np.testing.assert_allclose(result, np.array([np.nan]), equal_nan=True)

    assert len(InterruptingProgram.instances) == 1
    assert InterruptingProgram.instances[0].acquire_count == 1
    np.testing.assert_allclose(signals_buffer.array, np.array([np.nan]), equal_nan=True)


def test_build_and_acquire_returns_last_partial_after_retry_exhaustion():
    AlwaysFailingProgram.instances.clear()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=AlwaysFailingProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=_identity_array, retry=1)
        )
        assert sched.is_stop() is True
        assert sched.outcome.status == "failed"
        assert isinstance(sched.outcome.exception, RuntimeError)
        assert sched.outcome.reason == "RuntimeError: permanent failure"

    np.testing.assert_allclose(result, np.array([2.0]))
    np.testing.assert_allclose(signals_buffer.array, np.array([2.0]))
    assert len(AlwaysFailingProgram.instances) == 2
    assert AlwaysFailingProgram.instances[0].acquire_count == 1
    assert AlwaysFailingProgram.instances[1].acquire_count == 1


def test_scan_returns_partial_when_program_build_fails():
    signals_buffer = SignalBuffer((3,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer) as sched:
        for value, step in sched.scan("value", [0.0, 1.0, 2.0]):
            setattr(step.cfg, "value", value)
            step.prog_builder(
                "soc",
                "soccfg",
                program_cls=ConstructorFailingProgram,
            ).add(FakeModule("readout")).build_and_acquire(
                raw2signal_fn=_identity_array,
            )
        assert sched.outcome.status == "failed"
        assert isinstance(sched.outcome.exception, RuntimeError)
        assert sched.outcome.reason == "RuntimeError: build failure"

    np.testing.assert_allclose(
        signals_buffer.array,
        np.array([0.0, np.nan, np.nan]),
        equal_nan=True,
    )


def test_run_program_reuses_caller_owned_program_cache():
    FakeCachedProgram.instances.clear()
    signals_buffer = SignalBuffer((3,), dtype=np.float64)
    programs: dict[float, FakeCachedProgram] = {}

    with Schedule(_cfg(rounds=1), signals_buffer) as sched:
        for value, step in sched.scan("value", [1.0, 1.0, 2.0]):
            setattr(step.cfg, "value", value)
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
                raw2signal_fn=_identity_array,
            )

    assert np.allclose(signals_buffer.array, [1.0, 1.0, 2.0])
    assert len(FakeCachedProgram.instances) == 2
    assert FakeCachedProgram.instances[0].acquire_count == 2
    assert FakeCachedProgram.instances[1].acquire_count == 1
    assert set(programs) == {1.0, 2.0}


def test_build_and_acquire_decimated_uses_decimated_method_and_default_conversion():
    FakeDecimatedProgram.instances.clear()
    buffer_updates: list[np.ndarray] = []
    signals_buffer = SignalBuffer(
        (2,),
        on_update=lambda data: buffer_updates.append(data.copy()),
        update_interval=None,
    )

    with Schedule(_cfg(reps=1, rounds=2), signals_buffer) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeDecimatedProgram)
            .add(FakeModule("readout"))
            .build_and_acquire_decimated(
                tag="trace",
                stop_checkers=[lambda: False],
            )
        )

    assert np.allclose(result, [2.0 + 0.0j, 0.0 + 2.0j])
    assert np.allclose(signals_buffer.array, [2.0 + 0.0j, 0.0 + 2.0j])
    assert np.allclose(buffer_updates[0], [1.0 + 0.0j, 0.0 + 1.0j])
    assert np.allclose(buffer_updates[-1], [2.0 + 0.0j, 0.0 + 2.0j])

    assert len(FakeDecimatedProgram.instances) == 1
    instance = FakeDecimatedProgram.instances[0]
    assert instance.acquire_decimated_kwargs == {"tag": "trace"}
    assert instance.stop_checker_count == 2


def test_schedule_registers_program_derived_buffer_before_decimated_run():
    FakeDecimatedProgram.instances.clear()
    buffer_updates: list[np.ndarray] = []

    with Schedule(_cfg(reps=1, rounds=2)) as sched:
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

    assert np.allclose(result, [2.0 + 0.0j, 0.0 + 2.0j])
    assert np.allclose(signals_buffer.array, [2.0 + 0.0j, 0.0 + 2.0j])
    assert np.allclose(buffer_updates[-1], [2.0 + 0.0j, 0.0 + 2.0j])
    assert len(FakeDecimatedProgram.instances) == 1


def test_schedule_rejects_multiple_root_buffers_at_construction() -> None:
    with np.testing.assert_raises_regex(ValueError, "at most one root result buffer"):
        Schedule(
            _cfg(),
            RecordingBuffer(np.zeros((1,))),
            RecordingBuffer(np.zeros((1,))),
        )


def test_schedule_rejects_registering_multiple_root_buffers() -> None:
    first = RecordingBuffer(np.zeros((1,)))
    second = RecordingBuffer(np.zeros((1,)))

    with Schedule(_cfg(), first) as sched:
        with np.testing.assert_raises_regex(
            ValueError, "at most one root result buffer"
        ):
            sched.register_buffer(second)


def test_schedule_allows_registering_one_root_buffer_later() -> None:
    buffer = RecordingBuffer(np.zeros((1,)))

    with Schedule(_cfg()) as sched:
        sched.register_buffer(buffer)
        assert sched.data is buffer.data


def test_schedule_root_trigger_update_passes_none_step() -> None:
    buffer = RootUpdateRecordingBuffer(np.zeros((1,)))

    with Schedule(_cfg(), buffer) as sched:
        sched.trigger_update(flush=True)

    assert buffer.calls == [(None, True)]


def test_signal_buffer_write_triggers_update_and_manual_trigger():
    updates: list[np.ndarray] = []
    buffer = SignalBuffer(
        (2,),
        dtype=np.float64,
        on_update=lambda data: updates.append(data.copy()),
        update_interval=None,
    )

    buffer.set(np.array([1.0, 2.0]))
    slot = buffer[1]
    slot.set(np.array(3.0))
    buffer.trigger_update()

    assert np.asarray(slot.value).item() == 3.0
    assert np.allclose(updates[0], [1.0, 2.0])
    assert np.allclose(updates[1], [1.0, 3.0])
    assert np.allclose(updates[2], [1.0, 3.0])


def test_schedule_uses_own_stop_signal():
    stop = StopSignal()
    stop.set_stop()

    with Schedule(_cfg(), stop=stop) as sched:
        assert sched.is_stop() is True

    stop.clear_stop()
    with Schedule(_cfg(), stop=stop) as sched:
        assert sched.is_stop() is False
        sched.set_stop()
        assert stop.is_stop() is True


def test_schedule_operations_require_context():
    sched = Schedule(_cfg(rounds=1), SignalBuffer((1,), dtype=np.float64))

    try:
        sched.prog_builder("soc", "soccfg", marker="x")
    except RuntimeError as exc:
        assert "with Schedule" in str(exc)
    else:
        raise AssertionError("Schedule operations should require context manager")


def test_schedule_step_data_operations_require_context() -> None:
    root = {"child": {"signals": np.full((1,), np.nan)}}

    with Schedule(_cfg(), RecordingBuffer(root)) as sched:
        step = next(iter(sched.batch({"child": lambda child: child}).values()))
        assert step.path == ("child",)

    for operation in (
        lambda: step.data,
        lambda: step.set_data({"signals": np.array([1.0])}),
        lambda: step.child("signals"),
        step.trigger_update,
    ):
        try:
            operation()
        except RuntimeError as exc:
            assert "with Schedule" in str(exc)
        else:
            raise AssertionError(
                "ScheduleStep operations should require active context"
            )


def test_schedule_requires_explicit_modules():
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer) as sched:
        builder = sched.prog_builder("soc", "soccfg", marker="x")
        try:
            builder.build()
        except ValueError as exc:
            assert "explicit modules" in str(exc)
        else:
            raise AssertionError("ProgramBuilder.build should require modules")


def test_schedule_scan_targets_buffer_step_without_mutating_env():
    signals_buffer = SignalBuffer((3,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer) as sched:
        for value, step in sched.scan("value", [10.0, 20.0, 30.0]):
            assert step.value == value
            assert isinstance(step.index, int)
            setattr(step.cfg, "value", value)
            step.prog_builder(
                "soc",
                "soccfg",
                program_cls=FakeScalarProgram,
            ).add(FakeModule("readout")).build_and_acquire(
                raw2signal_fn=_identity_array,
            )

        assert sched.env == {}

    assert np.allclose(signals_buffer.array, [10.0, 20.0, 30.0])


def test_schedule_repeat_targets_step_buffer_without_mutating_env():
    signals_buffer = SignalBuffer((3,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer) as sched:
        for index, step in sched.repeat("round", 3, interval=0.0):
            assert step.index == index
            assert step.value == index
            assert step.path == (index,)
            setattr(step.cfg, "value", index)
            step.prog_builder(
                "soc",
                "soccfg",
                program_cls=FakeScalarProgram,
            ).add(FakeModule("readout")).build_and_acquire(
                raw2signal_fn=_identity_array,
            )

    assert np.allclose(signals_buffer.array, [0.0, 1.0, 2.0])


def test_schedule_env_accepts_dataclass_context():
    env = FlowEnv(label="typed", scale=2.0)

    with Schedule(_cfg(), env=env) as sched:
        assert sched.env.label == "typed"
        for value, step in sched.scan("value", [3.0]):
            assert step.env is env
            assert step.env.scale * value == 6.0


def test_schedule_nested_repeat_scan_targets_inner_step_buffer():
    signals_buffer = SignalBuffer((2, 3), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer) as sched:
        for repeat_index, repeat_step in sched.repeat("round", 2):
            for length_value, step in repeat_step.scan("length", [10.0, 20.0, 30.0]):
                assert step.path == (repeat_index, step.index)
                setattr(step.cfg, "value", repeat_index * 100.0 + length_value)
                step.prog_builder(
                    "soc",
                    "soccfg",
                    program_cls=FakeScalarProgram,
                ).add(FakeModule("readout")).build_and_acquire(
                    raw2signal_fn=_identity_array,
                )

    assert np.allclose(
        signals_buffer.array,
        [[10.0, 20.0, 30.0], [110.0, 120.0, 130.0]],
    )


def test_schedule_repeat_validates_times_and_interval():
    with Schedule(_cfg()) as sched:
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


def test_schedule_repeat_obeys_stop_signal():
    seen: list[int] = []

    with Schedule(_cfg()) as sched:
        for index, _step in sched.repeat("round", 3):
            seen.append(index)
            sched.set_stop()
        assert sched.outcome.status == "stopped"

    assert seen == [0]


def test_schedule_batch_runs_children_with_isolated_cfgs():
    order: list[str] = []

    def child_a(step):
        order.append("a")
        assert step.path == ("a",)
        setattr(step.cfg, "marker", "mutated")
        return getattr(step.cfg, "marker")

    def child_b(step):
        order.append("b")
        assert step.path == ("b",)
        return getattr(step.cfg, "marker")

    init_cfg = _cfg(marker="base")
    with Schedule(init_cfg) as sched:
        results = sched.batch({"a": child_a, "b": child_b})
        assert getattr(sched.cfg, "marker") == "base"

    assert order == ["a", "b"]
    assert results == {"a": "mutated", "b": "base"}
    assert getattr(init_cfg, "marker") == "base"


def test_schedule_batch_returns_partial_on_child_exception_without_retry():
    root = {"child": {"signals": np.full((1,), np.nan)}}
    attempts = 0
    later_called = False

    def failing(step):
        nonlocal attempts
        attempts += 1
        signals_step = step.child("signals")
        buffer = signals_step.buffer((1,), dtype=np.float64)
        buffer.set(np.array([1.0]))
        raise RuntimeError("temporary failure")

    def later(_step):
        nonlocal later_called
        later_called = True
        return "later"

    with Schedule(_cfg(), RecordingBuffer(root)) as sched:
        results = sched.batch({"child": failing, "later": later})
        assert results == {}
        assert sched.outcome.status == "failed"
        assert isinstance(sched.outcome.exception, RuntimeError)
        assert sched._local_buffers == {}

    assert attempts == 1
    assert later_called is False
    np.testing.assert_allclose(root["child"]["signals"], np.array([1.0]))


def test_schedule_batch_string_key_default_program_target_marks_failed_outcome():
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    def child(step):
        setattr(step.cfg, "value", 1.0)
        step.prog_builder(
            "soc",
            "soccfg",
            program_cls=FakeScalarProgram,
        ).add(FakeModule("readout")).build_and_acquire(
            raw2signal_fn=_identity_array,
        )

    with Schedule(_cfg(rounds=1), signals_buffer) as sched:
        results = sched.batch({"child": child})
        assert results == {}
        assert sched.outcome.status == "failed"
        assert sched.outcome.reason is not None
        assert "integer-indexed path" in sched.outcome.reason
        assert "SignalBuffer slot" in sched.outcome.reason


def test_schedule_batch_string_key_uses_child_local_default_buffer():
    root = {"child": {"signals": np.full((1,), np.nan)}}

    def child(step):
        signals_step = step.child("signals")
        signals_step.buffer((1,), dtype=np.float64)
        setattr(signals_step.cfg, "value", 6.0)
        signals_step.prog_builder(
            "soc",
            "soccfg",
            program_cls=FakeScalarProgram,
        ).add(FakeModule("readout")).build_and_acquire(
            raw2signal_fn=_identity_array,
        )

    with Schedule(_cfg(rounds=1), RecordingBuffer(root)) as sched:
        sched.batch({"child": child})

    assert np.allclose(root["child"]["signals"], [6.0])


def test_schedule_child_buffer_validates_target_before_acquire() -> None:
    root = {
        "child": {
            "signals": np.full((2,), np.nan),
            "metadata": {"value": 0},
        }
    }

    def child(step):
        signals_step = step.child("signals")
        try:
            signals_step.buffer((1,), dtype=np.float64)
        except ValueError as exc:
            assert "shape" in str(exc)
        else:
            raise AssertionError("buffer should reject shape mismatch")

        metadata_step = step.child("metadata")
        try:
            metadata_step.buffer((1,), dtype=np.float64)
        except ValueError as exc:
            assert "NDArray" in str(exc)
        else:
            raise AssertionError("buffer should reject non-array targets")

    with Schedule(_cfg(), RecordingBuffer(root)) as sched:
        sched.batch({"child": child})


def test_schedule_child_local_buffers_are_cleared_after_batch() -> None:
    root = {"child": {"signals": np.full((1,), np.nan)}}

    with Schedule(_cfg(), RecordingBuffer(root)) as sched:

        def child(step):
            signals_step = step.child("signals")
            buffer = signals_step.buffer((1,), dtype=np.float64)
            assert ("child", "signals") in sched._local_buffers
            buffer.set(np.array([3.0]))

        sched.batch({"child": child})
        assert sched._local_buffers == {}

    np.testing.assert_allclose(root["child"]["signals"], np.array([3.0]))


def test_schedule_batch_does_not_retry_keyboard_interrupt():
    attempts = 0
    later_called = False

    def interrupted(_step):
        nonlocal attempts
        attempts += 1
        raise KeyboardInterrupt

    def later(_step):
        nonlocal later_called
        later_called = True
        return "later"

    with Schedule(_cfg()) as sched:
        results = sched.batch({"a": interrupted, "b": later})
        assert sched.is_stop() is True
        assert sched.outcome.status == "interrupted"

    assert results == {}
    assert attempts == 1
    assert later_called is False
