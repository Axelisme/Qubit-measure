from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.program.v2 import (
    LoadValue,
    Module,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class ZigZagResult:
    times: NDArray[np.int64]
    signals: NDArray[np.complex128]


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class ZigZagModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    X90_pulse: Annotated[PulseCfg, FieldMeta(label="X90 Pulse")]
    X180_pulse: Annotated[PulseCfg | None, FieldMeta(label="X180 Pulse")] = None
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class ZigZagCfg(ProgramV2Cfg, ExpCfgModel):
    modules: ZigZagModuleCfg
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


def build_zigzag_modules(cfg: ZigZagCfg) -> list[Module]:
    """Build reset → X90 → repeated gates → readout for offline compilation.

    This does not access hardware or mutate cfg.
    The caller must declare the ``times`` loop with ``cfg.n_times + 1`` points.
    X90 repetition counts correspond to pairs of pulses.
    Raise ValueError if the selected repeat pulse is absent.
    The cfg must satisfy ZigZagCfg validation.
    """
    repeat_pulse = getattr(cfg.modules, cfg.repeat_on)
    if repeat_pulse is None:
        raise ValueError(f"Repeat on pulse {cfg.repeat_on} not found")
    counts = [
        n * (2 if cfg.repeat_on == "X90_pulse" else 1) for n in range(cfg.n_times + 1)
    ]
    return [
        LoadValue(
            "load_repeat_count",
            values=counts,
            idx_reg="times",
            val_reg="repeat_count",
        ),
        Reset("reset", cfg.modules.reset),
        Pulse("X90_pulse", cfg.modules.X90_pulse),
        Repeat(
            "zigzag_loop", n="repeat_count", range_hint=(0, max(counts))
        ).add_content(Pulse(f"loop_{cfg.repeat_on}", repeat_pulse)),
        Readout("readout", cfg.modules.readout),
    ]


class ZigZagExp(PersistableExperiment[ZigZagResult, ZigZagCfg]):
    # times is an int counter (a.u.) -> Axis dtype int64, scale IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(Axis("times", "Times", "a.u.", IDENTITY, np.int64),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=ZigZagResult,
        cfg_type=ZigZagCfg,
        cfg_schema_version="2.0",
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

        viewer = context.plots.liveplot_1d("measurement", "Times", "Signal")
        signals_buffer = SignalBuffer(
            (len(times),),
            on_update=lambda data: viewer.update(
                times.astype(np.float64),
                zigzag_signal2real(data),
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            # 2. Count X90 gates in pairs so each step adds a nominal pi rotation.
            modules = sched.cfg.modules
            repeat_pulse = getattr(modules, sched.cfg.repeat_on)
            if repeat_pulse is None:
                raise ValueError(f"Repeat on pulse {sched.cfg.repeat_on} not found")
            counts = [
                n * (2 if sched.cfg.repeat_on == "X90_pulse" else 1)
                for n in range(sched.cfg.n_times + 1)
            ]

            # 3. Load the count, prepare with X90, repeat gates, then read out.
            builder = (
                sched.prog_builder(soc, soccfg)
                .add(
                    LoadValue(
                        "load_repeat_count",
                        values=counts,
                        idx_reg="times",
                        val_reg="repeat_count",
                    ),
                    Reset("reset", modules.reset),
                    Pulse("X90_pulse", modules.X90_pulse),
                    Repeat(
                        "zigzag_loop", n="repeat_count", range_hint=(0, max(counts))
                    ).add_content(Pulse(f"loop_{sched.cfg.repeat_on}", repeat_pulse)),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("times", len(times))
            )

            # 4. Average acquisition rounds and keep partial data on cancellation.
            _ = builder.build_and_acquire()

        return ZigZagResult(times=times, signals=signals_buffer.array)
