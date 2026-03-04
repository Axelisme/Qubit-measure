from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typeguard import check_type
from typing_extensions import NotRequired, TypedDict

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskCfg, run_task
from zcu_tools.liveplot import LivePlotter2D, LivePlotter2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.program.v2.modules import TwoPulseResetCfg
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import minus_background, rotate2real


def dual_reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals))


# (fpts1, fpts2, signals_2d)
FreqResult = Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]


class FreqModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    tested_reset: TwoPulseResetCfg
    readout: ReadoutCfg


class FreqCfg(ModularProgramCfg, TaskCfg):
    modules: FreqModuleCfg
    sweep: Dict[str, SweepCfg]


class FreqExp(AbsExperiment[FreqResult, FreqCfg]):
    def run_soft(self, soc, soccfg, cfg: Dict[str, Any]) -> FreqResult:
        _cfg = check_type(deepcopy(cfg), FreqCfg)

        # Check that reset pulse is dual pulse type
        modules = _cfg["modules"]

        fpt1_sweep = _cfg["sweep"]["freq1"]
        fpt2_sweep = _cfg["sweep"]["freq2"]
        _cfg["sweep"] = {"freq1": fpt1_sweep}

        fpts1 = sweep2array(fpt1_sweep)  # predicted frequency points
        fpts2 = sweep2array(fpt2_sweep)  # predicted frequency points

        Reset.set_param(
            modules["tested_reset"], "freq1", sweep2param("freq1", fpt1_sweep)
        )

        with LivePlotter2DwithLine(
            "Frequency1 (MHz)",
            "Frequency2 (MHz)",
            line_axis=0,
        ) as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="freq2",
                    sweep_values=fpts2.tolist(),
                    update_cfg_fn=lambda _, ctx, fpt2: Reset.set_param(
                        ctx.cfg["modules"]["tested_reset"], "freq2", fpt2
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            (modules := ctx.cfg["modules"])
                            and (
                                ModularProgramV2(
                                    soccfg,
                                    ctx.cfg,
                                    modules=[
                                        Reset("reset", modules.get("reset")),
                                        Pulse("init_pulse", modules.get("init_pulse")),
                                        Reset("tested_reset", modules["tested_reset"]),
                                        Readout("readout", modules["readout"]),
                                    ],
                                ).acquire(soc, progress=False, callback=update_hook)
                            )
                        ),
                        result_shape=(len(fpts1),),
                    ),
                ),
                init_cfg=_cfg,
                update_hook=lambda ctx: viewer.update(
                    fpts1, fpts2, dual_reset_signal2real(np.asarray(ctx.data).T)
                ),
            )
            signals = np.asarray(signals).T

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (fpts1, fpts2, signals)

        return fpts1, fpts2, signals

    def run_hard(self, soc, soccfg, cfg: Dict[str, Any]) -> FreqResult:
        _cfg = check_type(deepcopy(cfg), FreqCfg)

        # Check that reset pulse is dual pulse type
        modules = _cfg["modules"]

        # Ensure freq1 is the outer loop for better visualization
        _cfg["sweep"] = {
            "freq1": _cfg["sweep"]["freq1"],
            "freq2": _cfg["sweep"]["freq2"],
        }

        fpts1 = sweep2array(_cfg["sweep"]["freq1"])
        fpts2 = sweep2array(_cfg["sweep"]["freq2"])

        Reset.set_param(
            modules["tested_reset"],
            "freq1",
            sweep2param("freq1", _cfg["sweep"]["freq1"]),
        )
        Reset.set_param(
            modules["tested_reset"],
            "freq2",
            sweep2param("freq2", _cfg["sweep"]["freq2"]),
        )

        with LivePlotter2D("Frequency1 (MHz)", "Frequency2 (MHz)") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        (modules := ctx.cfg["modules"])
                        and (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset("reset", modules.get("reset")),
                                    Pulse("init_pulse", modules.get("init_pulse")),
                                    Reset("tested_reset", modules["tested_reset"]),
                                    Readout("readout", modules["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        )
                    ),
                    result_shape=(len(fpts1), len(fpts2)),
                ),
                init_cfg=_cfg,
                update_hook=lambda ctx: viewer.update(
                    fpts1, fpts2, dual_reset_signal2real(ctx.data)
                ),
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (fpts1, fpts2, signals)

        return fpts1, fpts2, signals

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        method: Literal["soft", "hard"] = "soft",
    ) -> FreqResult:
        if method == "soft":
            return self.run_soft(soc, soccfg, cfg)
        else:
            return self.run_hard(soc, soccfg, cfg)

    def analyze(
        self,
        result: Optional[FreqResult] = None,
        *,
        smooth: float = 1.0,
        xname: Optional[str] = None,
        yname: Optional[str] = None,
        corner_as_background: bool = False,
    ) -> Tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts1, fpts2, signals = result

        # Apply smoothing for peak finding
        signals_smooth = gaussian_filter(signals, smooth)

        # Find peak in amplitude
        if corner_as_background:
            amps = np.abs(signals_smooth - signals_smooth[0, 0])
        else:
            amps = np.abs(minus_background(signals_smooth))

        freq1_opt = fpts1[np.argmax(np.max(amps, axis=1))]
        freq2_opt = fpts2[np.argmax(np.max(amps, axis=0))]

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.imshow(
            rotate2real(signals.T).real,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(fpts1[0], fpts1[-1], fpts2[0], fpts2[-1]),
        )
        peak_label = f"({freq1_opt:.1f}, {freq2_opt:.1f}) MHz"
        ax.scatter(freq1_opt, freq2_opt, color="r", s=40, marker="*", label=peak_label)
        if xname is not None:
            ax.set_xlabel(f"{xname} Frequency (MHz)", fontsize=14)
        if yname is not None:
            ax.set_ylabel(f"{yname} Frequency (MHz)", fontsize=14)
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return freq1_opt, freq2_opt, fig

    def save(
        self,
        filepath: str,
        result: Optional[FreqResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/dual_tone/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts1, fpts2, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency1", "unit": "Hz", "values": fpts1 * 1e6},
            y_info={"name": "Frequency2", "unit": "Hz", "values": fpts2 * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        signals, fpts1, fpts2 = load_data(filepath, **kwargs)
        assert fpts1 is not None and fpts2 is not None
        assert len(fpts1.shape) == 1 and len(fpts2.shape) == 1
        assert signals.shape == (len(fpts2), len(fpts1))

        fpts1 = fpts1 * 1e-6  # Hz -> MHz
        fpts2 = fpts2 * 1e-6  # Hz -> MHz
        signals = signals.T  # transpose back

        fpts1 = fpts1.astype(np.float64)
        fpts2 = fpts2.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (fpts1, fpts2, signals)

        return fpts1, fpts2, signals
