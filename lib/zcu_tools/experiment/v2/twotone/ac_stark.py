from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import NonUniformImage

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fitlor
from zcu_tools.utils.process import minus_background, rotate2real

from ..template import sweep2D_soft_hard_template

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
    of the amplitude of a drive field (Stark pulse).
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

        # Force the order of sweep (gain outer, freq inner for better visualization)
        gain_sweep = cfg["sweep"]["gain"]
        freq_sweep = cfg["sweep"]["freq"]

        # use soft sweep for gain
        cfg["sweep"] = dict(freq=freq_sweep)

        # uniform in square space
        pdrs = np.sqrt(
            np.linspace(
                gain_sweep["start"] ** 2,
                gain_sweep["stop"] ** 2,
                gain_sweep["expts"],
            )
        )
        freqs = sweep2array(freq_sweep)  # predicted frequencies

        cfg["stark_pulse1"]["gain"] = pdrs[0]
        cfg["stark_pulse2"]["freq"] = sweep2param("freq", freq_sweep)

        def updateCfg(cfg, _, pdr) -> None:
            cfg["stark_pulse1"]["gain"] = pdr

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            # Create modular program with Stark pulses
            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", reset_cfg=cfg.get("reset")),
                    Pulse(name="stark_pulse1", cfg=cfg["stark_pulse1"]),
                    Pulse(name="stark_pulse2", cfg=cfg["stark_pulse2"]),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )

            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        # Run 2D soft-hard sweep
        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Stark Pulse Gain (a.u.)",
                "Frequency (MHz)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=pdrs,
            ys=freqs,
            updateCfg=updateCfg,
            signal2real=acstark_signal2real,
            progress=progress,
        )

        # Get actual parameters used by the FPGA
        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(name="stark_pulse1", cfg=cfg["stark_pulse1"]),
                Pulse(name="stark_pulse2", cfg=cfg["stark_pulse2"]),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )
        fpts = prog.get_pulse_param("stark_pulse2", "freq", as_array=True)
        assert isinstance(fpts, np.ndarray), "freqs should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (pdrs, fpts, signals2D)

        return pdrs, fpts, signals2D

    def analyze(
        self,
        result: Optional[AcStarkResultType] = None,
        *,
        plot: bool = True,
        chi: float,
        kappa: float,
        deg: int = 1,
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

        pdrs, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Stark Pulse Gain", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )


AcStarkByRamseyResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


class AcStarkByRamseyExperiment(AbsExperiment[AcStarkResultType]):
    """AC Stark shift experiment.

    Measures the frequency shift of a qubit transition as a function
    of the amplitude and frequency of a drive field (Stark pulse).
    This experiment uses one pulse: one with variable amplitude (stark_pulse1)
    and measure the detuning of the qubit transition by Ramsey experiment.
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        detune: float = 0.0,
        progress: bool = True,
    ) -> AcStarkResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        if cfg["stark_pulse"]["style"] not in ["const", "flat_top"]:
            raise ValueError(
                "This experiment only supports const and flat_top stark pulse."
            )

        # Force the order of sweep (gain outer, freq inner for better visualization)
        gain_sweep = cfg["sweep"]["gain"]
        len_sweep = cfg["sweep"]["length"]

        # use soft sweep for gain
        cfg["sweep"] = dict(length=len_sweep)

        # uniform in square space
        pdrs = np.sqrt(
            np.linspace(
                gain_sweep["start"] ** 2,
                gain_sweep["stop"] ** 2,
                gain_sweep["expts"],
            )
        )
        t2r_spans = sweep2array(len_sweep)  # predicted lengths

        cfg["stark_pulse"]["gain"] = pdrs[0]
        cfg["stark_pulse"]["length"] = sweep2param("length", len_sweep)

        def updateCfg(cfg, _, pdr) -> None:
            cfg["stark_pulse"]["gain"] = pdr

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            # Create modular program with Stark pulses
            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    make_reset("reset", reset_cfg=cfg.get("reset")),
                    Pulse(name="pi2_pulse1", cfg=cfg["pi2_pulse"]),
                    Pulse(name="stark_pulse", cfg=cfg["stark_pulse"]),
                    Pulse(
                        name="pi2_pulse2",
                        cfg={  # activate detune
                            **cfg["pi2_pulse"],
                            "phase": cfg["pi2_pulse"].get("phase", 0.0)
                            + 360 * detune * t2r_spans,
                        },
                    ),
                    make_readout("readout", readout_cfg=cfg["readout"]),
                ],
            )

            return prog.acquire(soc, progress=False, callback=cb)[0][0].dot([1, 1j])

        # Run 2D soft-hard sweep
        signals2D = sweep2D_soft_hard_template(
            cfg,
            measure_fn,
            LivePlotter2DwithLine(
                "Stark Pulse Gain (a.u.)",
                "Time (us)",
                line_axis=1,
                num_lines=2,
                disable=not progress,
            ),
            xs=pdrs,
            ys=t2r_spans,
            updateCfg=updateCfg,
            signal2real=acstark_signal2real,
            progress=progress,
        )

        # Get actual parameters used by the FPGA
        prog = ModularProgramV2(
            soccfg, cfg, modules=[Pulse(name="stark_pulse", cfg=cfg["stark_pulse"])]
        )
        ts = prog.get_pulse_param("stark_pulse", "length", as_array=True)
        assert isinstance(ts, np.ndarray), "ts should be an array"

        # Cache results
        self.last_cfg = cfg
        self.last_result = (pdrs, ts, signals2D)

        return pdrs, ts, signals2D

    def analyze(
        self,
        result: Optional[AcStarkResultType] = None,
        *,
        plot: bool = True,
        chi: float,
        kappa: float,
        deg: int = 1,
        cutoff: Optional[float] = None,
        max_contrast: bool = True,
    ) -> float:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        pdrs, ts, signals = result

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(pdrs < cutoff)[0]
            pdrs = pdrs[valid_indices]
            signals = signals[valid_indices, :]

        if max_contrast:
            real_signals = rotate2real(signals).real
        else:
            real_signals = np.abs(signals)

        s_fpts = ts
        pdrs2 = pdrs**2

        # fitting max_freqs with ax2 + bx + c
        x2_fit = np.linspace(min(pdrs2), max(pdrs2), 100)
        if deg == 1:
            b, c = np.polyfit(pdrs2, s_fpts, 1)
            y_fit = b * x2_fit + c
        elif deg == 2:
            a, b, c = np.polyfit(pdrs2, s_fpts, 2)
            y_fit = a * x2_fit**2 + b * x2_fit + c
        else:
            raise ValueError(f"Degree {deg} is not supported.")

        # Calculate the Stark shift
        eta = kappa**2 / (kappa**2 + chi**2)
        ac_coeff = abs(b) / (2 * eta * chi)

        # plot the data and the fitted polynomial
        if plot:
            avg_n = ac_coeff * pdrs2

            fig, (ax1, ax2) = plt.subplots(figsize=config.figsize, ncols=2, sharex=True)

            # Use NonUniformImage for better visualization with pdr^2 as x-axis
            im = NonUniformImage(ax1, cmap="viridis", interpolation="nearest")
            im.set_data(avg_n, ts, real_signals.T)
            im.set_extent([avg_n[0], avg_n[-1], ts[0], ts[-1]])
            ax1.add_image(im)

            # Set proper limits for the plot
            ax1.set_xlim(avg_n[0], avg_n[-1])
            ax1.set_ylim(ts[0], ts[-1])

            # Plot the resonance frequencies and fitted curve
            ax2.plot(ac_coeff * pdrs2, s_fpts, ".", c="k")

            # Fit curve in terms of pdr^2
            label = r"$\bar n$" + f" = {ac_coeff:.2g} " + r"$gain^2$"
            n_fit = ac_coeff * x2_fit
            ax2.plot(n_fit, y_fit, "-", label=label)

            # Create secondary x-axis for pdr^2 (Readout Gain²)
            ax3 = ax2.twiny()

            # main x-axis: avg_n, secondary x-axis: pdr^2
            # avg_n = ac_coeff * pdrs^2
            ax3.set_xlabel(r"Average Photon Number ($\bar n$)", fontsize=14)

            # 上方次 x 軸顯示 pdr
            avgn_ticks = ax2.get_xticks()
            pdr_ticks = np.sqrt(avgn_ticks / ac_coeff)
            ax3.set_xlim(ax2.get_xlim())
            ax3.set_xticks(avgn_ticks)
            ax3.set_xticklabels([f"{pdr:.2f}" for pdr in pdr_ticks])
            ax3.set_xlabel("Readout Gain (a.u.)", fontsize=14)

            ax2.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
            ax2.legend(fontsize="x-large")
            ax2.tick_params(axis="both", which="major", labelsize=12)

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

        pdrs, Ts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Stark Pulse Gain", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
