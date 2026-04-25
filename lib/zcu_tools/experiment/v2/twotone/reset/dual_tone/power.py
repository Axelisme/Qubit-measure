from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    TwoPulseReset,
    sweep2param,
)
from zcu_tools.program.v2.modules import TwoPulseResetCfg
from zcu_tools.utils.datasaver import load_data, save_data

# (gains1, gains2, signals_2d)
PowerResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class PowerModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
    tested_reset: TwoPulseResetCfg
    readout: ReadoutCfg


class PowerSweepCfg(ConfigBase):
    gain1: SweepCfg
    gain2: SweepCfg


class PowerCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerModuleCfg
    sweep: PowerSweepCfg


class PowerExp(AbsExperiment[PowerResult, PowerCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: PowerCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> PowerResult:
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

        gain1_param = sweep2param("gain1", cfg.sweep.gain1)
        gain2_param = sweep2param("gain2", cfg.sweep.gain2)
        modules.tested_reset.set_param("gain1", gain1_param)
        modules.tested_reset.set_param("gain2", gain2_param)

        def dual_reset_gain_signal2real(signals: NDArray) -> np.ndarray:
            # Choose reference point based on sweep direction (use minimum power point)
            ref_i = 0 if gains1[0] < gains1[-1] else -1
            ref_j = 0 if gains2[0] < gains2[-1] else -1
            return np.abs(signals - signals[ref_i, ref_j])

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, PowerCfg],
            update_hook: Optional[Callable],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules
            return ModularProgramV2(
                soccfg,
                cfg,
                sweep=[("gain1", cfg.sweep.gain1), ("gain2", cfg.sweep.gain2)],
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    TwoPulseReset("tested_reset", modules.tested_reset),
                    Readout("readout", modules.readout),
                ],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                **(acquire_kwargs or {}),
            )

        with LivePlot2D("Gain1 (a.u.)", "Gain2 (a.u.)") as viewer:
            signals = run_task(
                task=Task(
                    pbar_n=cfg.rounds,
                    measure_fn=measure_fn,
                    result_shape=(len(gains1), len(gains2)),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    gains1, gains2, dual_reset_gain_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = deepcopy(cfg)
        self.last_result = (gains1, gains2, signals)

        return gains1, gains2, signals

    def analyze(
        self,
        result: Optional[PowerResult] = None,
        *,
        smooth: float = 1.0,
        xname: Optional[str] = None,
        yname: Optional[str] = None,
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains1, gains2, signals = result

        # Apply smoothing for peak finding
        signals_smooth = gaussian_filter(signals, smooth)

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

    def save(
        self,
        filepath: str,
        result: Optional[PowerResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/dual_tone/power",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains1, gains2, signals = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Power1", "unit": "a.u.", "values": gains1},
            y_info={"name": "Power2", "unit": "a.u.", "values": gains2},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PowerResult:
        signals, gains1, gains2, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert gains1 is not None and gains2 is not None
        assert len(gains1.shape) == 1 and len(gains2.shape) == 1
        assert signals.shape == (len(gains2), len(gains1))

        signals = signals.T  # transpose back

        gains1 = gains1.astype(np.float64)
        gains2 = gains2.astype(np.float64)
        signals = signals.astype(np.complex128)

        if comment is not None:

            cfg, _, _ = parse_comment(comment)

            if cfg is not None:

                self.last_cfg = PowerCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (gains1, gains2, signals)

        return gains1, gains2, signals
