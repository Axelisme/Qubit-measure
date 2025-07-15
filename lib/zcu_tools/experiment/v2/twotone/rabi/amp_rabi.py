from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real

from ...template import sweep_hard_template


def rabi_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


AmpRabiResultType = Tuple[np.ndarray, np.ndarray]  # (amps, signals)


class AmpRabiExperiment(AbsExperiment[AmpRabiResultType]):
    """Rabi oscillation by varying pulse *amplitude* (gain)."""

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> AmpRabiResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        gain_sweep = cfg["sweep"]["gain"]

        cfg["qub_pulse"]["gain"] = sweep2param("gain", gain_sweep)

        amps = sweep2array(gain_sweep)  # predicted

        prog = TwoToneProgram(soccfg, cfg)

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D("Pulse gain", "Amplitude", disable=not progress),
            ticks=(amps,),
            signal2real=rabi_signal2real,
        )

        amps_real = prog.get_pulse_param("qubit_pulse", "gain", as_array=True)
        assert isinstance(amps_real, np.ndarray)

        self.last_cfg = cfg
        self.last_result = (amps_real, signals)

        return amps_real, signals

    def analyze(
        self,
        result: Optional[AmpRabiResultType] = None,
        *,
        decay: bool = False,
        plot: bool = True,
        max_contrast: bool = True,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result
        if max_contrast:
            real_signals = rotate2real(signals).real
        else:
            real_signals = np.abs(signals)

        pi_amp, pi2_amp, y_fit, _ = fit_rabi(pdrs, real_signals, decay=decay)

        if plot:
            xs_fit = np.linspace(pdrs[0], pdrs[-1], 1001)

            plt.figure(figsize=config.figsize)
            plt.tight_layout()
            plt.plot(pdrs, real_signals, label="meas", ls="-", marker="o", markersize=3)
            plt.plot(xs_fit, y_fit, label="fit")
            plt.axvline(pi_amp, ls="--", c="red", label=f"pi = {pi_amp:.3g}")
            plt.axvline(pi2_amp, ls="--", c="red", label=f"pi/2 = {pi2_amp:.3g}")
            plt.xlabel("Pulse gain (a.u.)")
            plt.ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
            plt.legend(loc=4)
            plt.show()

        return pi_amp, pi2_amp

    def save(
        self,
        filepath: str,
        result: Optional[AmpRabiResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/rabi_gain",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Gain", "unit": "", "values": pdrs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
