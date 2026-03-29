from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typeguard import check_type
from typing_extensions import Any, Literal, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
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
from zcu_tools.program.v2.modules import BathResetCfg
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real

FreqGainResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class FreqGainModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    tested_reset: BathResetCfg
    readout: ReadoutCfg


class FreqGainCfg(ModularProgramCfg, TaskCfg):
    modules: FreqGainModuleCfg
    sweep: dict[str, SweepCfg]


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class FreqGainExp(AbsExperiment[FreqGainResult, FreqGainCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> FreqGainResult:
        _cfg = check_type(deepcopy(cfg), FreqGainCfg)
        modules = _cfg["modules"]

        reset_cfg = modules["tested_reset"]
        gains = sweep2array(
            _cfg["sweep"]["gain"],
            "gain",
            {"soccfg": soccfg, "gen_ch": reset_cfg["cavity_tone_cfg"]["ch"]},
        )
        freqs = sweep2array(
            _cfg["sweep"]["freq"],
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg["cavity_tone_cfg"]["ch"]},
        )

        gain_param = sweep2param("gain", _cfg["sweep"]["gain"])
        freq_param = sweep2param("freq", _cfg["sweep"]["freq"])
        Reset.set_param(modules["tested_reset"], "res_gain", gain_param)
        Reset.set_param(modules["tested_reset"], "res_freq", freq_param)

        with LivePlotter2D(
            "Cavity Frequency (MHz)", "Cavity drive Gain (a.u.)"
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: (
                        (modules := ctx.cfg["modules"])
                        and (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                sweep=[
                                    ("gain", ctx.cfg["sweep"]["gain"]),
                                    ("freq", ctx.cfg["sweep"]["freq"]),
                                ],
                                modules=[
                                    Reset("reset", modules.get("reset")),
                                    Pulse("init_pulse", modules.get("init_pulse")),
                                    Reset("tested_reset", modules["tested_reset"]),
                                    Readout("readout", modules["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        )
                    ),
                    result_shape=(len(gains), len(freqs)),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, gains, bathreset_signal2real(ctx.root_data).T
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (gains, freqs, signals)

        return gains, freqs, signals

    def analyze(
        self,
        result: Optional[FreqGainResult] = None,
        smooth: float = 1.0,
        find: Literal["min", "max", "med"] = "min",
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, freqs, signals = result

        # Apply smoothing for peak finding
        signals_smooth: NDArray[np.complex128] = gaussian_filter(signals, smooth)  # type: ignore

        # Find peak in amplitude
        real_signals = bathreset_signal2real(signals_smooth)

        if find == "max":
            gain_opt = gains[np.argmax(np.max(real_signals, axis=1))]
            freq_opt = freqs[np.argmax(np.max(real_signals, axis=0))]
        elif find == "min":
            gain_opt = gains[np.argmin(np.min(real_signals, axis=1))]
            freq_opt = freqs[np.argmin(np.min(real_signals, axis=0))]
        else:
            # med_value = np.median(real_signals)
            med_dists = np.abs(real_signals - np.median(real_signals))
            gain_opt = gains[np.argmin(np.min(med_dists, axis=1))]
            freq_opt = freqs[np.argmin(np.min(med_dists, axis=0))]

        fig, ax = plt.subplots()
        assert isinstance(fig, Figure)

        ax.imshow(
            real_signals,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=(freqs[0], freqs[-1], gains[0], gains[-1]),
        )
        peak_label = f"({gain_opt:.2f} a.u., {freq_opt:.1f} MHz)"
        ax.scatter(freq_opt, gain_opt, color="r", s=40, marker="*", label=peak_label)
        ax.set_xlabel("Cavity Frequency (MHz)", fontsize="x-large")
        ax.set_ylabel("Cavity drive Gain (a.u.)", fontsize="x-large")
        ax.legend(fontsize="x-large")
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return gain_opt, freq_opt, fig

    def save(
        self,
        filepath: str,
        result: Optional[FreqGainResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/freq_gain",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, freqs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Cavity drive Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Cavity Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqGainResult:
        signals, gains, freqs = load_data(filepath, **kwargs)
        assert gains is not None and freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert signals.shape == (len(freqs), len(gains))

        freqs = freqs * 1e-6  # Hz -> MHz
        signals = signals.T  # transpose back

        gains = gains.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (gains, freqs, signals)

        return gains, freqs, signals
