from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typeguard import check_type
from typing_extensions import Any, Literal, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.utils import sweep2array
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
    TwoPulseReset,
    sweep2param,
)
from zcu_tools.program.v2.modules import TwoPulseResetCfg
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import minus_background, rotate2real


def dual_reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals))


# (freqs1, freqs2, signals_2d)
FreqResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class FreqModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    tested_reset: TwoPulseResetCfg
    readout: ReadoutCfg


class FreqCfg(ModularProgramCfg, TaskCfg):
    modules: FreqModuleCfg
    sweep: dict[str, SweepCfg]


class FreqExp(AbsExperiment[FreqResult, FreqCfg]):
    def run_soft(self, soc, soccfg, cfg: dict[str, Any]) -> FreqResult:
        _cfg = check_type(deepcopy(cfg), FreqCfg)
        modules = _cfg["modules"]

        freq1_sweep = _cfg["sweep"]["freq1"]
        freq2_sweep = _cfg["sweep"]["freq2"]
        _cfg["sweep"] = {"freq1": freq1_sweep}  # remove freq2 from sweep

        reset_cfg = modules["tested_reset"]
        freqs1 = sweep2array(
            freq1_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg["pulse1_cfg"]["ch"]},
        )
        freqs2 = sweep2array(
            freq2_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg["pulse2_cfg"]["ch"]},
            allow_array=True,
        )

        freq1_param = sweep2param("freq1", freq1_sweep)
        TwoPulseReset.set_param(modules["tested_reset"], "freq1", freq1_param)

        with LivePlotter2DwithLine(
            "Frequency1 (MHz)", "Frequency2 (MHz)", line_axis=0
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: (
                        (modules := ctx.cfg["modules"])
                        and (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset("reset", modules.get("reset")),
                                    Pulse("init_pulse", modules.get("init_pulse")),
                                    TwoPulseReset(
                                        "tested_reset", modules["tested_reset"]
                                    ),
                                    Readout("readout", modules["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        )
                    ),
                    result_shape=(len(freqs1),),
                ).scan(
                    "freq2",
                    freqs2.tolist(),
                    before_each=lambda _, ctx, freq2: Reset.set_param(
                        ctx.cfg["modules"]["tested_reset"], "freq2", freq2
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    freqs1, freqs2, dual_reset_signal2real(np.asarray(ctx.root_data).T)
                ),
            )
            signals = np.asarray(signals).T

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (freqs1, freqs2, signals)

        return freqs1, freqs2, signals

    def run_hard(self, soc, soccfg, cfg: dict[str, Any]) -> FreqResult:
        _cfg = check_type(deepcopy(cfg), FreqCfg)
        modules = _cfg["modules"]


        # Ensure freq1 is the outer loop for better visualization
        _cfg["sweep"] = {
            "freq1": _cfg["sweep"]["freq1"],
            "freq2": _cfg["sweep"]["freq2"],
        }

        reset_cfg = modules["tested_reset"]
        freqs1 = sweep2array(
            _cfg["sweep"]["freq1"],
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg["pulse1_cfg"]["ch"]},
        )
        freqs2 = sweep2array(
            _cfg["sweep"]["freq2"],
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg["pulse2_cfg"]["ch"]},
        )

        freq1_param = sweep2param("freq1", _cfg["sweep"]["freq1"])
        freq2_param = sweep2param("freq2", _cfg["sweep"]["freq2"])
        TwoPulseReset.set_param(modules["tested_reset"], "freq1", freq1_param)
        TwoPulseReset.set_param(modules["tested_reset"], "freq2", freq2_param)

        with LivePlotter2D("Frequency1 (MHz)", "Frequency2 (MHz)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: (
                        (modules := ctx.cfg["modules"])
                        and (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset("reset", modules.get("reset")),
                                    Pulse("init_pulse", modules.get("init_pulse")),
                                    TwoPulseReset(
                                        "tested_reset", modules["tested_reset"]
                                    ),
                                    Readout("readout", modules["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        )
                    ),
                    result_shape=(len(freqs1), len(freqs2)),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    freqs1, freqs2, dual_reset_signal2real(ctx.root_data)
                ),
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (freqs1, freqs2, signals)

        return freqs1, freqs2, signals

    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
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
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs1, freqs2, signals = result

        # Apply smoothing for peak finding
        signals_smooth = gaussian_filter(signals, smooth)

        # Find peak in amplitude
        if corner_as_background:
            amps = np.abs(signals_smooth - signals_smooth[0, 0])
        else:
            amps = np.abs(minus_background(signals_smooth))

        freq1_opt = freqs1[np.argmax(np.max(amps, axis=1))]
        freq2_opt = freqs2[np.argmax(np.max(amps, axis=0))]

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.imshow(
            rotate2real(signals.T).real,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(freqs1[0], freqs1[-1], freqs2[0], freqs2[-1]),
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

        freqs1, freqs2, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency1", "unit": "Hz", "values": freqs1 * 1e6},
            y_info={"name": "Frequency2", "unit": "Hz", "values": freqs2 * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        signals, freqs1, freqs2 = load_data(filepath, **kwargs)
        assert freqs1 is not None and freqs2 is not None
        assert len(freqs1.shape) == 1 and len(freqs2.shape) == 1
        assert signals.shape == (len(freqs2), len(freqs1))

        freqs1 = freqs1 * 1e-6  # Hz -> MHz
        freqs2 = freqs2 * 1e-6  # Hz -> MHz
        signals = signals.T  # transpose back

        freqs1 = freqs1.astype(np.float64)
        freqs2 = freqs2.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (freqs1, freqs2, signals)

        return freqs1, freqs2, signals
