from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D, LivePlotter2DwithLine
from zcu_tools.program.v2 import ModularProgramV2, Pulse, Readout, Reset, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import minus_background, rotate2real

from ....runner import HardTask, Runner, SoftTask


def dual_reset_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(minus_background(signals))


# (fpts1, fpts2, signals_2d)
DualToneResetFreqResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


class FreqExperiment(AbsExperiment[DualToneResetFreqResultType]):
    def run_soft(self, soc, soccfg, cfg: Dict[str, Any]) -> DualToneResetFreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        if cfg["tested_reset"]["type"] != "two_pulse":
            raise ValueError("This experiment only supports dual-tone reset")

        fpt1_sweep = cfg["sweep"]["freq1"]
        fpt2_sweep = cfg["sweep"]["freq2"]
        cfg["sweep"] = {"freq1": fpt1_sweep}

        fpts1 = sweep2array(fpt1_sweep)  # predicted frequency points
        fpts2 = sweep2array(fpt2_sweep)  # predicted frequency points

        Reset.set_param(cfg["tested_reset"], "freq1", sweep2param("freq1", fpt1_sweep))

        with LivePlotter2DwithLine(
            "Frequency2 (MHz)",
            "Frequency1 (MHz)",
            line_axis=1,
            segment2d_kwargs={"flip": True},
        ) as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="freq2",
                    sweep_values=fpts2,
                    update_cfg_fn=lambda _, ctx, fpt2: Reset.set_param(
                        ctx.cfg["tested_reset"], "freq2", fpt2
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset(
                                        "reset", ctx.cfg.get("reset", {"type": "none"})
                                    ),
                                    Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                                    Reset("tested_reset", ctx.cfg["tested_reset"]),
                                    Readout("readout", ctx.cfg["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        ),
                        result_shape=(len(fpts1),),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    fpts2, fpts1, dual_reset_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals).T

        # Cache results
        self.last_cfg = cfg
        self.last_result = (fpts1, fpts2, signals)

        return fpts1, fpts2, signals

    def run_hard(self, soc, soccfg, cfg: Dict[str, Any]) -> DualToneResetFreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Check that reset pulse is dual pulse type
        if cfg["tested_reset"]["type"] != "two_pulse":
            raise ValueError("This experiment only supports dual-tone reset")

        # Ensure freq1 is the outer loop for better visualization
        cfg["sweep"] = {
            "freq1": cfg["sweep"]["freq1"],
            "freq2": cfg["sweep"]["freq2"],
        }

        fpts1 = sweep2array(cfg["sweep"]["freq1"])
        fpts2 = sweep2array(cfg["sweep"]["freq2"])

        Reset.set_param(
            cfg["tested_reset"], "freq1", sweep2param("freq1", cfg["sweep"]["freq1"])
        )
        Reset.set_param(
            cfg["tested_reset"], "freq2", sweep2param("freq2", cfg["sweep"]["freq2"])
        )

        with LivePlotter2D(
            "Frequency2 (MHz)", "Frequency1 (MHz)", segment_kwargs={"flip": True}
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                                Reset("tested_reset", ctx.cfg["tested_reset"]),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(fpts1), len(fpts2)),
                ),
                update_hook=lambda ctx: viewer.update(
                    fpts1, fpts2, dual_reset_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (fpts1, fpts2, signals)

        return fpts1, fpts2, signals

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        method: Literal["soft", "hard"] = "soft",
    ) -> DualToneResetFreqResultType:
        if method == "soft":
            return self.run_soft(soc, soccfg, cfg)
        else:
            return self.run_hard(soc, soccfg, cfg)

    def analyze(
        self,
        result: Optional[DualToneResetFreqResultType] = None,
        *,
        smooth: float = 1.0,
        xname: Optional[str] = None,
        yname: Optional[str] = None,
        corner_as_background: bool = False,
    ) -> Tuple[float, float, plt.Figure]:
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
        assert isinstance(fig, plt.Figure)

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
        result: Optional[DualToneResetFreqResultType] = None,
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
