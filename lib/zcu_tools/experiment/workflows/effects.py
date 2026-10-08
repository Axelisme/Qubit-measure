"""Core-owned suspension requests and their typed execution seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from pathlib import Path
from re import fullmatch
from typing import Protocol

from .display import Live1D, Live2D, Live2DRow
from .models import Completed, Effect, Failed
from .run import Run


class EffectExecutor(Protocol):
    """Execute effects without exposing heterogeneous response types to callers.

    The engine implements this seam. Methods return only after the operation
    settles; cancellation discards the step instead of delivering a response.
    Exceptions propagate to the engine's source-aware failure boundary.
    """

    def run[Cfg, Result](
        self,
        request: RunEffect[Cfg, Result],
    ) -> Completed[Cfg, Result] | Failed:
        """Execute request and save before returning its typed outcome.

        request contains validated name, experiment, cfg, saver, optional live,
        and setup_devices. Failures and cancellation never produce Completed.
        """
        ...

    def set_device(self, name: str, value: float) -> None:
        """Set a connected device's absolute native value, or raise."""
        ...

    def wait_until(self, target: datetime) -> None:
        """Wait for an aware deadline, or discard the step on cancellation."""
        ...


@dataclass
class RunEffect[Cfg, Result](Effect):
    """Internal request; experiment/cfg/save/live/setup_devices match env.run.

    name is the validated ASCII function-name stem used for run files.
    outcome is absent until successful dispatch, then Completed or Failed.
    The generator retains this exact request to retrieve its typed response.
    """

    experiment: Callable[[Run[Cfg]], Result]
    cfg: Cfg
    save: Callable[[Completed[Cfg, Result], Path], None]
    live: Live1D | Live2D | Live2DRow | None
    setup_devices: bool
    name: str = field(init=False)
    outcome: Completed[Cfg, Result] | Failed | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        name = getattr(self.experiment, "__name__", None)
        if not isinstance(name, str) or fullmatch(r"[A-Za-z0-9_]+", name) is None:
            raise ValueError(
                "Experiment must have an ASCII function name; no anonymous callables"
            )
        self.name = name

    def execute(self, executor: EffectExecutor) -> None:
        """Execute this request once; propagate cancellation and operation errors."""
        if self.outcome is not None:
            raise RuntimeError("Run effect has already completed")
        self.outcome = executor.run(self)


@dataclass
class DeviceEffect(Effect):
    """Internal absolute setpoint request; done is true after adapter return."""

    name: str
    value: float
    done: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.name.strip() or not isfinite(self.value):
            raise ValueError(
                "Device setpoint requires a name and finite absolute value"
            )

    def execute(self, executor: EffectExecutor) -> None:
        """Apply name/value once; propagate cancellation and device errors."""
        if self.done:
            raise RuntimeError("Device effect has already completed")
        executor.set_device(self.name, self.value)
        self.done = True


@dataclass
class WaitEffect(Effect):
    """Internal aware-deadline request; done is true after clock return."""

    target: datetime
    done: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.target.tzinfo is None or self.target.utcoffset() is None:
            raise ValueError("Wait target must be timezone-aware")

    def execute(self, executor: EffectExecutor) -> None:
        """Wait once; propagate cancellation and clock errors."""
        if self.done:
            raise RuntimeError("Wait effect has already completed")
        executor.wait_until(self.target)
        self.done = True


def dispatch_effect(effect: Effect, executor: EffectExecutor) -> None:
    """Execute a core request, rejecting direct yields and unknown subclasses.

    A workflow must use yield from env.<effect>(...), not yield a generator
    or construct a custom Effect. A request receives no response on failure.
    """
    if not isinstance(effect, (RunEffect, DeviceEffect, WaitEffect)):
        raise TypeError("Use yield from env.<effect>(...); unknown effect request")
    effect.execute(executor)
