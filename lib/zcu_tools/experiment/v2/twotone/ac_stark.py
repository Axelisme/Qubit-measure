from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import NonUniformImage

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    check_no_post_delay,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fitlor
from zcu_tools.utils.process import minus_background, rotate2real

from ..template import sweep_hard_template

# (amps, freqs, signals2D)
AcStarkResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def acstark_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(minus_background(signals, axis=1)).real  # type: ignore


def get_resonance_freq(
    pdrs: np.ndarray, fpts: np.ndarray, amps: np.ndarray, cutoff=None
) -> np.ndarray:
    s_pdrs = []
    s_fpts = []
    prev_freq = fitlor(fpts, amps[0])[0][3]

    fitparams = [None, None, None, prev_freq, None]
    for pdr, amp in zip(pdrs, amps):
        curr_freq = fitlor(fpts, amp, fitparams=fitparams)[0][3]
        if abs(curr_freq - prev_freq) < 0.1 * (fpts[-1] - fpts[0]):
            s_pdrs.append(pdr)
            s_fpts.append(curr_freq)

            prev_freq = curr_freq
            fitparams[3] = curr_freq

    return np.array(s_pdrs), np.array(s_fpts)


class AcStarkExperiment(AbsExperiment[AcStarkResultType]):
    """AC Stark shift experiment.

    Measures the frequency shift of a qubit transition as a function
    of the amplitude and frequency of a drive field (Stark pulse).
    This experiment uses two pulses: one with variable amplitude (stark_pulse1)
    and another with variable frequency (stark_pulse2).
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> AcStarkResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Ensure stark_pulse1 has no post_delay to avoid timing conflicts
        check_no_post_delay(cfg["stark_pulse1"], "stark_pulse1")

        # Force the order of sweep (gain outer, freq inner for better visualization)
        gain_sweep = cfg["sweep"]["gain"]
        freq_sweep = cfg["sweep"]["freq"]
        cfg["sweep"] = {"gain": gain_sweep, "freq": freq_sweep}

        # Create modular program with Stark pulses
        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(
                    name="stark_pulse1",
                    cfg={
                        **cfg["stark_pulse1"],
                        "gain": sweep2param("gain", gain_sweep),
                    },
                ),
                Pulse(
                    name="stark_pulse2",
                    cfg={
                        **cfg["stark_pulse2"],
                        "freq": sweep2param("freq", freq_sweep),
                    },
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        amps = sweep2array(gain_sweep)  # predicted amplitudes
        freqs = sweep2array(freq_sweep)  # predicted frequencies

        # Run 2D hard sweep
        signals2D = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter2D(
                "Stark Pulse Gain",
                "Frequency (MHz)",
                title="AC Stark Shift",
                disable=not progress,
            ),
            ticks=(amps, freqs),
            signal2real=acstark_signal2real,
        )

        # Get actual parameters used by the FPGA
        amps_real = prog.get_pulse_param("stark_pulse1", "gain", as_array=True)
        freqs_real = prog.get_pulse_param("stark_pulse2", "freq", as_array=True)
        assert isinstance(amps_real, np.ndarray), "amps should be an array"
        assert isinstance(freqs_real, np.ndarray), "freqs should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (amps_real, freqs_real, signals2D)

        return amps_real, freqs_real, signals2D

    def analyze(
        self,
        result: Optional[AcStarkResultType] = None,
        *,
        plot: bool = True,
        chi: float,
        kappa: float,
        deg: int = 2,
        cutoff: Optional[float] = None,
    ) -> float:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        pdrs, fpts, signals = result

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(pdrs < cutoff)[0]
            pdrs = pdrs[valid_indices]
            signals = signals[valid_indices, :]

        amps = rotate2real(minus_background(signals, axis=1)).real
        amps /= np.std(amps, axis=1, keepdims=True)
        s_pdrs, s_fpts = get_resonance_freq(pdrs, fpts, amps)

        pdrs2 = pdrs**2
        s_pdrs2 = s_pdrs**2

        # fitting max_freqs with ax2 + bx + c
        x2_fit = np.linspace(min(pdrs2), max(pdrs2), 100)
        if deg == 1:
            b, c = np.polyfit(s_pdrs2, s_fpts, 1)
            y_fit = b * x2_fit + c
        elif deg == 2:
            a, b, c = np.polyfit(s_pdrs2, s_fpts, 2)
            y_fit = a * x2_fit**2 + b * x2_fit + c
        else:
            raise ValueError(f"Degree {deg} is not supported.")

        # Calculate the Stark shift
        eta = kappa**2 / (kappa**2 + chi**2)
        ac_coeff = abs(b) / (2 * eta * chi)

        # plot the data and the fitted polynomial
        if plot:
            avg_n = ac_coeff * pdrs2

            fig, ax1 = plt.subplots(figsize=config.figsize)

            # Use NonUniformImage for better visualization with pdr^2 as x-axis
            im = NonUniformImage(ax1, cmap="viridis", interpolation="nearest")
            im.set_data(avg_n, fpts, amps.T)
            im.set_extent([avg_n[0], avg_n[-1], fpts[0], fpts[-1]])
            ax1.add_image(im)

            # Set proper limits for the plot
            ax1.set_xlim(avg_n[0], avg_n[-1])
            ax1.set_ylim(fpts[0], fpts[-1])

            # Plot the resonance frequencies and fitted curve
            ax1.plot(ac_coeff * s_pdrs2, s_fpts, ".", c="k")

            # Fit curve in terms of pdr^2
            label = r"$\bar n$" + f" = {ac_coeff:.2g} " + r"$gain^2$"
            n_fit = ac_coeff * x2_fit
            ax1.plot(n_fit, y_fit, "-", label=label)

            # Create secondary x-axis for pdr^2 (Readout Gain²)
            ax2 = ax1.twiny()

            # main x-axis: avg_n, secondary x-axis: pdr^2
            # avg_n = ac_coeff * pdrs^2
            ax1.set_xticks(ax1.get_xticks())
            # ax1.set_xticklabels([f"{avg_n:.1f}" for avg_n in ax1.get_xticks()])
            ax1.set_xlabel(r"Average Photon Number ($\bar n$)", fontsize=14)

            # 上方次 x 軸顯示 pdr
            avgn_ticks = ax1.get_xticks()
            pdr_ticks = np.sqrt(avgn_ticks / ac_coeff)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(avgn_ticks)
            ax2.set_xticklabels([f"{pdr:.2f}" for pdr in pdr_ticks])
            ax2.set_xlabel("Readout Gain (a.u.)", fontsize=14)

            ax1.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
            ax1.legend(fontsize="x-large")
            ax1.tick_params(axis="both", which="major", labelsize=12)

            fig.tight_layout()
            plt.show()

        return ac_coeff

    def save(
        self,
        filepath: str,
        result: Optional[AcStarkResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ac_stark",
        **kwargs,
    ) -> None:
        """Save AC Stark experiment data."""
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        amps, freqs, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Stark Pulse Gain", "unit": "a.u.", "values": amps},
            y_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )
