"""Synthetic program, buffer and progress adapters for runtime contract tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.runtime.schedule import ScheduleStep
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import Module
from zcu_tools.progress_bar import BaseProgressBar
from zcu_tools.progress_bar.base import ProgressTotal, ProgressValue

from tests.experiment.v2.runtime._flow_cfg import FlowCfg


def identity_array(raw: np.ndarray) -> np.ndarray:
    """Return the acquired array unchanged, without copying or conversion."""
    return raw


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


class RecordingProgressBar(BaseProgressBar):
    def __init__(
        self,
        *,
        total: ProgressTotal = None,
        desc: str = "",
        leave: bool = True,
        disabled: bool = False,
    ) -> None:
        self._total = total
        self._desc = desc
        self.leave = leave
        self.disabled = disabled
        self.closed = False
        self.reset_count = 0
        self._n: ProgressValue = 0

    def set_description(self, description: str) -> None:
        self._desc = description

    def update(self, value: ProgressValue = 1) -> None:
        self._n += value

    def set_progress(self, value: ProgressValue) -> None:
        self._n = value

    def reset(self) -> None:
        self.reset_count += 1
        self._n = 0

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    @property
    def total(self) -> ProgressTotal:
        return self._total

    @total.setter
    def total(self, value: ProgressTotal) -> None:
        self._total = value

    @property
    def n(self) -> ProgressValue:
        return self._n

    @property
    def desc(self) -> str:
        return self._desc


def recording_pbar_factory(
    records: list[RecordingProgressBar],
) -> Callable[..., RecordingProgressBar]:
    """Build a progress factory that appends each created bar to ``records``."""

    def factory(*args: Any, **kwargs: Any) -> RecordingProgressBar:
        desc = kwargs.get("desc", args[1] if len(args) > 1 else "")
        total = kwargs.get("total", args[2] if len(args) > 2 else None)
        bar = RecordingProgressBar(
            total=total,
            desc=str(desc) if desc else "",
            leave=bool(kwargs.get("leave", True)),
            disabled=bool(kwargs.get("disable", False)),
        )
        records.append(bar)
        return bar

    return factory


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
        self.cancel_flag_seen = False
        FakeProgram.instances.append(self)

    def acquire(
        self,
        soc: Any,
        *,
        progress: bool,
        round_hook,
        cancel_flag,
        **kwargs: Any,
    ) -> np.ndarray:
        self.acquire_kwargs = dict(kwargs)
        self.cancel_flag_seen = True
        assert soc == "soc"
        assert not progress
        assert not cancel_flag.is_set()
        round_hook(1, np.array([1.0]), cancel_flag)
        return np.array([2.0])

    def acquire_decimated(
        self,
        soc: Any,
        *,
        progress: bool,
        round_hook,
        cancel_flag,
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

    def acquire(
        self, *_args: Any, round_hook, cancel_flag, **_kwargs: Any
    ) -> np.ndarray:
        # tolerated: FlowCfg.value is a dynamic Pydantic extra, not a typed field.
        raw = np.array(float(getattr(self.cfg_model, "value")))  # noqa: B009
        round_hook(1, raw, cancel_flag)
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

    def acquire(
        self, *_args: Any, round_hook, cancel_flag, **_kwargs: Any
    ) -> np.ndarray:
        self.acquire_count += 1
        # tolerated: FlowCfg.value is a dynamic Pydantic extra, not a typed field.
        raw = np.array(float(getattr(self.cfg_model, "value")))  # noqa: B009
        round_hook(1, raw, cancel_flag)
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
        self.cancel_flag_seen = False
        FakeDecimatedProgram.instances.append(self)

    def acquire(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def acquire_decimated(
        self,
        soc: Any,
        *,
        progress: bool,
        round_hook,
        cancel_flag,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        self.acquire_decimated_kwargs = dict(kwargs)
        self.cancel_flag_seen = True
        assert soc == "soc"
        assert not progress
        assert not cancel_flag.is_set()
        round_hook(1, [np.array([[1.0, 0.0], [0.0, 1.0]])], cancel_flag)
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

    def acquire(
        self, *_args: Any, round_hook, cancel_flag, **_kwargs: Any
    ) -> np.ndarray:
        self.acquire_count += 1
        if self.acquire_count == 1:
            raise RuntimeError("temporary failure")
        raw = np.array([4.0])
        round_hook(1, np.array([3.0]), cancel_flag)
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

    def acquire(
        self, *_args: Any, round_hook, cancel_flag, **_kwargs: Any
    ) -> np.ndarray:
        self.acquire_count += 1
        raw = np.array([float(self.acquire_count)])
        round_hook(1, raw, cancel_flag)
        raise RuntimeError("permanent failure")

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


class StopBeforeDataProgram:
    external_stop: StopSignal | None = None

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

    def acquire(
        self, *_args: Any, round_hook, cancel_flag, **_kwargs: Any
    ) -> np.ndarray:
        assert self.external_stop is not None
        self.external_stop.set()
        raise StoppedPartialAcquireError(
            "acquire stopped before the first round completed"
        )

    def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
        raise NotImplementedError
