from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_resonence_freq

from ..template import sweep_hard_template

FreqResultType = Tuple[np.ndarray, np.ndarray]


class FreqExperiment(AbsExperiment[FreqResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FreqResultType:
        cfg = deepcopy(cfg)

        res_pulse = cfg["readout"]["pulse_cfg"]

        # Ensure the sweep section is in canonical single-axis form.
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        sweep_cfg = cfg["sweep"]["freq"]

        # Predicted frequency points (before mapping to ADC domain)
        fpts = sweep2array(sweep_cfg)  # MHz

        # Attach sweep parameter to the QICK program – using a *QickParam* so
        # that it is executed inside the FPGA loop («hard sweep»).
        res_pulse["freq"] = sweep2param("freq", sweep_cfg)

        # Build program once; *OneToneProgram* will evaluate the sweep
        prog = OneToneProgram(soccfg, cfg)

        # Run acquisition via common *template* helper
        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D("Frequency (MHz)", "Amplitude", disable=not progress),
            ticks=(fpts,),
        )

        # Retrieve the actual frequencies *used* by the program; because of
        # FPGA coercion they can differ slightly from the requested points.
        fpts_real = prog.get_pulse_param("readout_pulse", "freq", as_array=True)
        assert isinstance(fpts_real, np.ndarray), "fpts should be an ndarray"

        # record last cfg and result
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
