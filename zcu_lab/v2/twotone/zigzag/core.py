from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.program.v2 import LoadValue, Pulse, Readout, Reset, TimelineBuilder
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class ZigZagResult:
    times: NDArray[np.int64]
    signals: NDArray[np.complex128]


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class ZigZagCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    X90_pulse: Annotated[Pulse.Cfg, FieldMeta(label="X90 Pulse")]
    X180_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="X180 Pulse")] = None
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    n_times: Annotated[
        int,
        FieldMeta(
            label="Max repetitions",
            description="Repetition counts run from zero to this value.",
        ),
    ] = Field(ge=0)
    repeat_on: Annotated[
        Literal["X90_pulse", "X180_pulse"],
        FieldMeta(
            label="Repeat pulse",
            description="X90 repeats in pairs; each count adds one pi rotation.",
        ),
    ] = "X180_pulse"

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


class ZigZagExp(Experiment[ZigZagResult, ZigZagCfg]):
    # times is an int counter (a.u.) -> Axis dtype int64, scale IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[ZigZagResult, ZigZagCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("times", label="Times", unit="a.u.", dtype=np.int64),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=ZigZagResult,
        cfg_type=ZigZagCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/zigzag",
    )

    def run(self, cfg: ZigZagCfg, *, context: RunContext) -> ZigZagResult:
        """Measure accumulated rotation errors versus the repeated gate count.

        Reset, apply X90, repeat the selected gate, then read out mean complex IQ.
        Return integer counts from zero through n_times and the corresponding IQ;
        each count adds two X90 pulses or one X180 pulse.
        The selected repeat pulse must exist.
        Cancelled acquisition preserves completed-round averages or NaN if empty.
        """
        # 1. Prepare the count axis and live view without changing the caller's cfg.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        times = np.arange(0, cfg.n_times + 1)

        viewer = live_line(context.plots, "measurement", "Times", "Signal")
        signals_buffer = SignalBuffer(
            (len(times),),
            on_update=lambda data: viewer.update(
                times.astype(np.float64),
                zigzag_signal2real(data),
            ),
        )
        with Schedule(None, signals_buffer, scope=context.scope) as sched:
            repeat_pulse = getattr(cfg, cfg.repeat_on)
            if repeat_pulse is None:
                raise ValueError(f"Repeat on pulse {cfg.repeat_on} not found")
            counts = [
                n * (2 if cfg.repeat_on == "X90_pulse" else 1)
                for n in range(cfg.n_times + 1)
            ]
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("times", len(times))
            builder.emit(
                LoadValue(
                    "load_repeat_count",
                    values=counts,
                    idx_reg="times",
                    val_reg="repeat_count",
                )
            )
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("X90_pulse", cfg.X90_pulse))
            # Legacy repeats flush the reference at entry and after every gate.
            builder.delay_by_timeline()
            with builder.repeat(
                "zigzag_loop", "repeat_count", range_hint=(0, max(counts))
            ):
                builder.emit(Pulse(cfg.repeat_on, repeat_pulse))
                builder.delay_by_timeline()
            builder.emit(Readout("readout", cfg.readout))
            sched.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return ZigZagResult(times=times, signals=signals_buffer.array)
