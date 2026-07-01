from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
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
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    TwoPulseReset,
    sweep2param,
)
from zcu_tools.program.v2.modules import TwoPulseResetCfg
from zcu_tools.utils.process import SmoothMethod, smooth_signal_nd


@dataclass(frozen=True)
class PowerResult:
    gains1: NDArray[np.float64]
    gains2: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: PowerCfg | None = None


class PowerModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    tested_reset: TwoPulseResetCfg
    readout: ReadoutCfg


class PowerSweepCfg(ConfigBase):
    gain1: SweepCfg
    gain2: SweepCfg


class PowerCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerModuleCfg
    sweep: PowerSweepCfg


class PowerExp(PersistableExperiment[PowerResult, PowerCfg]):
    # Both axes are gains in a.u. -> scale=IDENTITY (default)
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("gains2", "Power2", "a.u."),  # inner
            Axis("gains1", "Power1", "a.u."),  # outer
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        tag="twotone/reset/dual_tone/power",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: PowerCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> PowerResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)

        # Check that reset pulse is dual pulse type
        modules = cfg.modules

        reset_cfg = modules.tested_reset
        gains1 = sweep2array(
            cfg.sweep.gain1,
            "gain",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse1_cfg.ch},
        )
        gains2 = sweep2array(
            cfg.sweep.gain2,
            "gain",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse2_cfg.ch},
        )

        def dual_reset_gain_signal2real(signals: NDArray) -> np.ndarray:
            # Choose reference point based on sweep direction (use minimum power point)
            ref_i = 0 if gains1[0] < gains1[-1] else -1
            ref_j = 0 if gains2[0] < gains2[-1] else -1
            return np.abs(signals - signals[ref_i, ref_j])

        with LivePlot2D("Gain1 (a.u.)", "Gain2 (a.u.)") as viewer:
            signals_buffer = SignalBuffer(
                (len(gains1), len(gains2)),
                on_update=lambda data: viewer.update(
                    gains1, gains2, dual_reset_gain_signal2real(data)
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
                modules = sched.cfg.modules
                modules.tested_reset.set_param(
                    "gain1", sweep2param("gain1", sched.cfg.sweep.gain1)
                )
                modules.tested_reset.set_param(
                    "gain2", sweep2param("gain2", sched.cfg.sweep.gain2)
                )
                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        TwoPulseReset("tested_reset", modules.tested_reset),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("gain1", sched.cfg.sweep.gain1)
                    .declare_sweep("gain2", sched.cfg.sweep.gain2)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array

        return PowerResult(gains1, gains2, signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(
        self,
        result: PowerResult | None = None,
        *,
        smooth: float = 1.0,
        smooth_method: SmoothMethod = "wavelet",
        wavelet: str = "sym4",
        wavelet_level: int = 0,
        xname: str | None = None,
        yname: str | None = None,
    ) -> tuple[float, float, Figure]:
        assert result is not None, "no result found"

        gains1, gains2, signals = result.gains1, result.gains2, result.signals

        # Apply smoothing for peak finding
        signals_smooth = smooth_signal_nd(
            signals,
            method=smooth_method,
            sigma=smooth,
            axes=(0, 1),
            wavelet=wavelet,
            wavelet_level=wavelet_level,
        )

        ref_i = 0 if gains1[0] < gains1[-1] else -1
        ref_j = 0 if gains2[0] < gains2[-1] else -1
        amp2D = np.abs(signals_smooth - signals_smooth[ref_i, ref_j])

        # Determine if we should look for max or min
        if amp2D[0, 0] < np.mean(amp2D):
            gain1_opt = gains1[np.argmax(np.max(amp2D, axis=1))]
            gain2_opt = gains2[np.argmax(np.max(amp2D, axis=0))]
        else:
            gain1_opt = gains1[np.argmin(np.min(amp2D, axis=1))]
            gain2_opt = gains2[np.argmin(np.min(amp2D, axis=0))]
            amp2D = np.mean(amp2D) - amp2D

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.imshow(
            amp2D.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(gains1[0], gains1[-1], gains2[0], gains2[-1]),
        )
        peak_label = f"({gain1_opt:.1f}, {gain2_opt:.1f}) a.u."
        ax.scatter(gain1_opt, gain2_opt, color="r", s=40, marker="*", label=peak_label)
        if xname is not None:
            ax.set_xlabel(f"{xname} gain (a.u.)", fontsize=14)
        if yname is not None:
            ax.set_ylabel(f"{yname} gain (a.u.)", fontsize=14)
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return gain1_opt, gain2_opt, fig
