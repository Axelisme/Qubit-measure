from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v1.twotone import RFreqTwoToneProgram
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_resonence_freq
from zcu_tools.utils.process import rotate2real

from ..template import sweep_hard_template


def qubfreq_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


FreqResultType = Tuple[np.ndarray, np.ndarray]


class FreqExperiment(AbsExperiment[FreqResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> FreqResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        sweep_cfg = cfg["sweep"]["freq"]

        fpts = sweep2array(sweep_cfg)

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = RFreqTwoToneProgram(soccfg, cfg)
            (avgi, avgq), _ = prog.acquire(soc, progress=progress, callback=cb)
            return avgi[0] + 1j * avgq[0]

        signals = sweep_hard_template(
            cfg,
            measure_fn,
            LivePlotter1D("Frequency (MHz)", "Amplitude", disable=not progress),
            ticks=(fpts,),
            signal2real=qubfreq_signal2real,
        )

        prog = RFreqTwoToneProgram(soccfg, cfg)
        fpts_real = prog.get_sweep_pts()

        self.last_cfg = cfg
        self.last_result = (fpts_real, signals)

        return fpts_real, signals

    def analyze(
        self,
        result: Optional[FreqResultType] = None,
        *,
        type: Literal["lor", "sinc"] = "lor",
        asym: bool = False,
        plot_fit: bool = True,
        max_contrast: bool = True,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        val_mask = ~np.isnan(signals)
        fpts = fpts[val_mask]
        signals = signals[val_mask]

        y = rotate2real(signals).real if max_contrast else np.abs(signals)

        freq, freq_err, kappa, _, y_fit, _ = fit_resonence_freq(fpts, y, type, asym)

        plt.figure(figsize=config.figsize)
        plt.tight_layout()
        plt.plot(fpts, y, label="signal", marker="o", markersize=3)
        if plot_fit:
            plt.plot(fpts, y_fit, label=f"fit, kappa={kappa:.1g} MHz")
            label = f"f_q = {freq:.5g} ± {freq_err:.1g} MHz"
            plt.axvline(freq, color="r", ls="--", label=label)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
        plt.legend()
        plt.show()

        return freq, kappa

    def save(
        self,
        filepath: str,
        result: Optional[FreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "MHz", "values": fpts},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
