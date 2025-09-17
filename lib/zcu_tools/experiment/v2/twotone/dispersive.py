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
from zcu_tools.utils.fitting import fit_resonence_freq

from ..template import sweep_hard_template

DispersiveResultType = Tuple[np.ndarray, np.ndarray]


def dispersive_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(signals)


class DispersiveExperiment(AbsExperiment[DispersiveResultType]):
    """Dispersive shift measurement experiment.

    Measures the resonator frequency response with and without a qubit π pulse
    to characterize the dispersive coupling between qubit and resonator.

    The experiment sweeps the readout frequency with two different qubit states:
    1. Ground state (no qubit pulse)
    2. Excited state (with π pulse)

    This allows measurement of the dispersive shift χ/2π and resonator linewidth κ.
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> DispersiveResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        res_pulse = cfg["readout"]["pulse_cfg"]
        qub_pulse = cfg["qub_pulse"]

        # Canonicalise sweep section to single-axis form
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        # Prepend ge sweep to inner loop for measuring both ground and excited states
        cfg["sweep"] = {
            "ge": {"start": 0, "stop": qub_pulse["gain"], "expts": 2},
            "freq": cfg["sweep"]["freq"],
        }

        # Set with/without π gain for qubit pulse
        qub_pulse["gain"] = sweep2param("ge", cfg["sweep"]["ge"])
        res_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

        prog = TwoToneProgram(soccfg, cfg)

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D(
                "Frequency (MHz)", "Amplitude", num_lines=2, disable=not progress
            ),
            ticks=(fpts,),
            signal2real=dispersive_signal2real,
        )

        # Get the actual pulse gains and frequency points
        fpts_real = prog.get_pulse_param("readout_pulse", "freq", as_array=True)
        assert isinstance(fpts_real, np.ndarray), "fpts should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (fpts_real, signals)

        return fpts_real, signals

    def fitt_wo_abcd(
        self,
        fpts: np.ndarray,
        g_signals: np.ndarray,
        e_signals: np.ndarray,
        asym: bool = True,
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        g_amps, e_amps = np.abs(g_signals), np.abs(e_signals)
        g_freq, _, g_kappa, _, g_fit, _ = fit_resonence_freq(fpts, g_amps, asym=asym)
        e_freq, _, e_kappa, _, e_fit, _ = fit_resonence_freq(fpts, e_amps, asym=asym)
        return g_freq, g_kappa, e_freq, e_kappa, g_fit, e_fit

    def fitt_by_abcd(
        self, fpts: np.ndarray, g_signals: np.ndarray, e_signals: np.ndarray
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        try:
            from abcd_rf_fit import analyze
        except ImportError:
            print(
                "cannot import abcd_rf_fit, do you have it installed? please check: <https://github.com/UlysseREGLADE/abcd_rf_fit.git>"
            )
            raise

        g_fit = analyze(1e6 * fpts, g_signals, "hm", fit_edelay=True)
        g_param = g_fit.tolist()
        g_freq, g_kappa = g_param[0] * 1e-6, g_param[1] * 1e-6  # MHz

        e_fit = analyze(1e6 * fpts, e_signals, "hm", fit_edelay=True)
        e_param = e_fit.tolist()
        e_freq, e_kappa = e_param[0] * 1e-6, e_param[1] * 1e-6  # MHz

        return (
            g_freq,
            g_kappa,
            e_freq,
            e_kappa,
            np.abs(g_fit(1e6 * fpts)),
            np.abs(e_fit(1e6 * fpts)),
        )

    def analyze(
        self,
        result: Optional[DispersiveResultType] = None,
        *,
        plot: bool = True,
        use_abcd: bool = False,
        **kwargs,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result
        amps = np.abs(signals)
        g_amps, e_amps = amps[0, :], amps[1, :]

        if use_abcd:
            g_freq, g_kappa, e_freq, e_kappa, g_fit, e_fit = self.fitt_by_abcd(
                fpts, signals[0, :], signals[1, :], **kwargs
            )
        else:
            g_freq, g_kappa, e_freq, e_kappa, g_fit, e_fit = self.fitt_wo_abcd(
                fpts, signals[0, :], signals[1, :], **kwargs
            )

        # Calculate dispersive shift and average linewidth
        chi = abs(g_freq - e_freq) / 2  # dispersive shift χ/2π
        avg_kappa = (g_kappa + e_kappa) / 2  # average linewidth κ/2π

        if plot:
            plt.figure(figsize=config.figsize)
            plt.tight_layout()

            # Plot data and fits
            plt.plot(fpts, g_amps, marker=".", c="b", label="Ground state")
            plt.plot(fpts, e_amps, marker=".", c="r", label="Excited state")
            plt.plot(fpts, g_fit, "b-", alpha=0.7)
            plt.plot(fpts, e_fit, "r-", alpha=0.7)

            # Mark resonance frequencies
            label_g = f"Ground: {g_freq:.1f} MHz, κ = {g_kappa:.1f} MHz"
            label_e = f"Excited: {e_freq:.1f} MHz, κ = {e_kappa:.1f} MHz"
            plt.axvline(g_freq, color="b", ls="--", alpha=0.7, label=label_g)
            plt.axvline(e_freq, color="r", ls="--", alpha=0.7, label=label_e)

            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Amplitude (a.u.)")
            plt.title(
                f"Dispersive shift χ/2π = {chi:.3f} MHz, κ/2π = {avg_kappa:.1f} MHz"
            )
            plt.legend()
            plt.grid(True)
            plt.show()

        return chi, avg_kappa

    def save(
        self,
        filepath: str,
        result: Optional[DispersiveResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/dispersive",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Amplitude", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
