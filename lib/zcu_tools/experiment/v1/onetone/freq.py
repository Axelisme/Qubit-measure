from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, map2adcfreq, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v1 import OneToneProgram
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_resonence_freq

from ..template import sweep1D_soft_template

FreqResultType = Tuple[np.ndarray, np.ndarray]


class FreqExperiment(AbsExperiment[FreqResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FreqResultType:
        cfg = deepcopy(cfg)

        res_pulse = cfg["dac"]["res_pulse"]

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        sweep_cfg = cfg["sweep"]["freq"]

        fpts = sweep2array(sweep_cfg, allow_array=True)
        fpts = map2adcfreq(soccfg, fpts, res_pulse["ch"], cfg["adc"]["chs"][0])

        del cfg["sweep"]

        def updateCfg(cfg, _, f):
            cfg["dac"]["res_pulse"]["freq"] = f

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = OneToneProgram(soccfg, cfg)
            sum_d, _ = prog.acquire(soc, progress=False, callback=cb)
            avg_d = [d / cfg["reps"] for d in sum_d]
            return avg_d[0].dot([1, 1j])

        signals = sweep1D_soft_template(
            cfg,
            measure_fn,
            LivePlotter1D("Frequency (MHz)", "Amplitude", disable=not progress),
            xs=fpts,
            updateCfg=updateCfg,
            progress=progress,
        )

        self.last_cfg = cfg
        self.last_result = (fpts, signals)

        return fpts, signals

    def analyze(
        self,
        result: Optional[FreqResultType] = None,
        *,
        type: Literal["lor", "sinc"] = "lor",
        asym: bool = False,
        plot_fit: bool = True,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        val_mask = ~np.isnan(signals)
        fpts = fpts[val_mask]
        signals = signals[val_mask]

        amps = np.abs(signals)

        freq, freq_err, kappa, _, y_fit, _ = fit_resonence_freq(fpts, amps, type, asym)

        plt.figure(figsize=config.figsize)
        plt.tight_layout()
        plt.plot(fpts, amps, label="signal", marker="o", markersize=3)
        if plot_fit:
            plt.plot(fpts, y_fit, label=f"fit, $kappa$={kappa:.1g} MHz")
            label = f"$f_res$ = {freq:.5g} +/- {freq_err:.1g} MHz"
            plt.axvline(freq, color="r", ls="--", label=label)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Magnitude (a.u.)")
        plt.legend()
        plt.show()

        return freq, kappa

    def save(
        self,
        filepath: str,
        result: Optional[FreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "onetone/freq",
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
