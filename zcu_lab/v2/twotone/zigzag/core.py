from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from qick.asm_v2 import QickParam, QickSweep1D
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
    BathResetCfg,
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
    TwoPulseResetCfg,
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


def _reset_pi(reset: ResetCfg | None) -> PulseCfg:
    """Return the terminal qubit pulse of a two-pulse/bath reset; reject others."""
    if isinstance(reset, TwoPulseResetCfg):
        return reset.pulse2_cfg
    if isinstance(reset, BathResetCfg):
        return reset.pi2_cfg
    raise ValueError(
        "reset phase cycling requires a two-pulse or bath reset with a final qubit pulse"
    )


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
    reset_phase_cycle: Annotated[
        bool, FieldMeta(label="Cycle reset pi phase (0/180)")
    ] = Field(
        default=False,
        description="Alternate the final reset qubit pulse by 180 degrees between averaging sweeps. Requires even reps and a scalar reset phase; adds no RF pulses or programmed wait.",
    )

    @model_validator(mode="after")
    def _validate_phase_cycle(self) -> Self:
        if self.reset_phase_cycle:
            if self.reps < 2 or self.reps % 2:
                raise ValueError(
                    "reset phase cycling requires an even reps count of at least 2"
                )
            if isinstance(_reset_pi(self.modules.reset).phase, QickParam):
                raise ValueError("reset phase cycling requires a scalar reset phase")
        return self


def build_zigzag_modules(cfg: ZigZagCfg) -> list[Module]:
    """Build reset → X90 → repeated gates → readout for offline compilation.

    This does not access hardware or mutate cfg.
    The caller must declare the ``times`` loop with ``cfg.n_times + 1`` points.
    X90 repetition counts correspond to pairs of pulses.
    Reset phase cycling changes only the final reset pulse's phase by 180 degrees
    on the outer ``reps`` loop; every repetition count receives balanced phases.
    It averages coherent preparation errors and does not correct population or
    repeated-gate errors. No RF pulse or programmed delay is added.
    Raise ValueError if the selected repeat pulse is absent or phase cycling has
    an unsupported reset/phase. The cfg must satisfy ZigZagCfg validation.
    """
    repeat_pulse = getattr(cfg.modules, cfg.repeat_on)
    if repeat_pulse is None:
        raise ValueError(f"Repeat on pulse {cfg.repeat_on} not found")
    counts = [
        n * (2 if cfg.repeat_on == "X90_pulse" else 1) for n in range(cfg.n_times + 1)
    ]
    reset_cfg = cfg.modules.reset
    if cfg.reset_phase_cycle:
        reset_cfg = deepcopy(reset_cfg)
        pulse = _reset_pi(reset_cfg)
        if isinstance(pulse.phase, QickParam):
            raise ValueError("reset phase cycling requires a scalar reset phase")
        pulse.phase = QickSweep1D(
            "reps", pulse.phase, pulse.phase + 180.0 * (cfg.reps - 1)
        )
    return [
        LoadValue(
            "load_repeat_count",
            values=counts,
            idx_reg="times",
            val_reg="repeat_count",
        ),
        Reset("reset", reset_cfg),
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
        each count adds two X90 pulses or one X180 pulse. Optional reset phase
        cycling averages coherent preparation errors, not repeated-gate errors.
        The selected repeat pulse must exist. Phase cycling requires a two-pulse
        or bath reset with scalar final qubit phase and even reps of at least two.
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

            # Balance reset phases on outer reps; add no pulses or waiting time.
            reset_cfg = modules.reset
            if sched.cfg.reset_phase_cycle:
                reset_cfg = deepcopy(reset_cfg)
                pulse = _reset_pi(reset_cfg)
                if isinstance(pulse.phase, QickParam):
                    raise ValueError(
                        "reset phase cycling requires a scalar reset phase"
                    )
                pulse.phase = QickSweep1D(
                    "reps", pulse.phase, pulse.phase + 180.0 * (sched.cfg.reps - 1)
                )

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
                    Reset("reset", reset_cfg),
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
