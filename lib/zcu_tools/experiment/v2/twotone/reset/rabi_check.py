from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class RabiCheckResult:
    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]
    reset_states: NDArray[np.int64] = field(
        default_factory=lambda: np.array([0, 1, 2], dtype=np.int64)
    )
    cfg_snapshot: RabiCheckCfg | None = None


def reset_rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class RabiCheckModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    rabi_pulse: PulseCfg
    tested_reset: ResetCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class RabiCheckSweepCfg(ConfigBase):
    gain: SweepCfg


class RabiCheckCfg(ProgramV2Cfg, ExpCfgModel):
    modules: RabiCheckModuleCfg
    sweep: RabiCheckSweepCfg


class RabiCheckExp(PersistableExperiment[RabiCheckResult, RabiCheckCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("gains", "Amplitude", "a.u.", scale=IDENTITY, dtype=np.float64),
            Axis("reset_states", "Reset", "None", scale=IDENTITY, dtype=np.int64),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=RabiCheckResult,
        cfg_type=RabiCheckCfg,
        tag="twotone/reset/rabi_check",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: RabiCheckCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> RabiCheckResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.rabi_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, RabiCheckCfg],
            update_hook: Callable | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            # Attach gain sweep to initialization pulse
            gain_param = sweep2param("gain", cfg.sweep.gain)
            modules.rabi_pulse.set_param("gain", gain_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                sweep=[
                    ("reset_sel", 3),
                    ("gain", cfg.sweep.gain),
                ],
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("rabi_pulse", modules.rabi_pulse),
                    Branch(
                        "reset_sel",
                        [],
                        Reset("tested_reset_1", modules.tested_reset),
                        [
                            Reset("tested_reset_2", modules.tested_reset),
                            Pulse("pi_pulse", modules.pi_pulse),
                        ],
                    ),
                    Readout("readout", modules.readout),
                ],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot1D(
            "Pulse gain", "Amplitude", segment_kwargs=dict(num_lines=3)
        ) as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (3, len(gains)),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        gains, reset_rabi_signal2real(data)
                    ),
                )
                signals_buffer.measure(measure_fn, pbar_n=run.cfg.rounds)
                signals = signals_buffer.array

        return RabiCheckResult(gains, signals, cfg_snapshot=cfg)

    @retrieve_result
    def analyze(self, result: RabiCheckResult | None = None) -> Figure:
        """Analyze reset rabi check results. (No specific analysis implemented)"""
        assert result is not None, "no result found"

        gains, signals = result.gains, result.signals
        real_signals = reset_rabi_signal2real(signals)

        wo_signals, w_signals, wp_signals = real_signals

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(gains, wo_signals, label="Without Reset", marker=".")
        ax.plot(gains, w_signals, label="With Reset", marker=".")
        ax.plot(gains, wp_signals, label="  + Pi Pulse", marker=".")
        ax.legend()
        ax.grid(True)

        return fig
