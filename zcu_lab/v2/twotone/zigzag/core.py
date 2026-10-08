from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from qick.asm_v2 import QickParam, QickSweep1D
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils import setup_devices
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
    reset: ResetCfg | None = None
    X90_pulse: PulseCfg
    X180_pulse: PulseCfg | None = None
    readout: ReadoutCfg


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
    n_times: int = Field(ge=0)
    repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse"
    reset_phase_cycle: bool = Field(
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
    """Build the acquisition sequence without hardware access or mutating cfg.

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
        tag="twotone/ge/zigzag",
    )

    def run(self, cfg: ZigZagCfg, *, context: RunContext) -> ZigZagResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )

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
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(build_zigzag_modules(sched.cfg))
                .declare_sweep("times", len(times))
                .build_and_acquire()
            )

        return ZigZagResult(times=times, signals=signals_buffer.array)
