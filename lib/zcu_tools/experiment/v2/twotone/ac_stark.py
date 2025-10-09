from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import NonUniformImage

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D, LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    check_block_mode,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fitlor
from zcu_tools.utils.process import minus_background, rotate2real

from ..runner import HardTask, Runner, SoftTask

# (amps, freqs, signals2D)
AcStarkResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def acstark_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(minus_background(signals, axis=1)).real  # type: ignore


def get_resonance_freq(
    xs: np.ndarray, fpts: np.ndarray, amps: np.ndarray
) -> np.ndarray:
    s_xs = []
    s_fpts = []
    prev_freq = fitlor(fpts, amps[0])[0][3]

    fitparams = [None, None, None, prev_freq, None]
    for x, amp in zip(xs, amps):
        curr_freq = fitlor(fpts, amp, fitparams=fitparams)[0][3]
        if abs(curr_freq - prev_freq) < 0.1 * (fpts[-1] - fpts[0]):
            s_xs.append(x)
            s_fpts.append(curr_freq)

            prev_freq = curr_freq
            fitparams[3] = curr_freq

    return np.array(s_xs), np.array(s_fpts)


class AcStarkExperiment(AbsExperiment[AcStarkResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> AcStarkResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        gain_sweep = cfg["sweep"].pop("gain")

        # uniform in square space
        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequencies
        pdrs = np.sqrt(
            np.linspace(
                gain_sweep["start"] ** 2, gain_sweep["stop"] ** 2, gain_sweep["expts"]
            )
        )

        cfg["stark_pulse2"]["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

        with LivePlotter2DwithLine(
            "Stark Pulse Gain (a.u.)",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=2,
            uniform=False,
            disable=not progress,
        ) as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="resonator gain",
                    sweep_values=pdrs,
                    update_cfg_fn=lambda _, ctx, pdr: Pulse.set_param(
                        ctx.cfg["stark_pulse1"], "gain", pdr
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    make_reset("reset", ctx.cfg.get("reset")),
                                    Pulse("stark_pulse1", ctx.cfg["stark_pulse1"]),
                                    Pulse("stark_pulse2", ctx.cfg["stark_pulse2"]),
                                    make_readout("readout", ctx.cfg["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        ),
                        result_shape=(len(fpts),),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    pdrs, fpts, acstark_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (pdrs, fpts, signals)

        return pdrs, fpts, signals

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


PhotonLookbackResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def photonlookback_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class PhotonLookbackExperiment(AbsExperiment[PhotonLookbackResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> PhotonLookbackResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        qub_pulse = cfg["qub_pulse"]
        check_block_mode("qub_pulse", qub_pulse, want_block=False)

        # let length be outer loop
        cfg["sweep"] = {
            "offset": cfg["sweep"]["offset"],
            "freq": cfg["sweep"]["freq"],
        }

        # predict point
        offsets = sweep2array(cfg["sweep"]["offset"])
        fpts = sweep2array(cfg["sweep"]["freq"])

        # swept by FPGA (hard sweep)
        qub_pulse["pre_delay"] = sweep2param("offset", cfg["sweep"]["offset"])
        qub_pulse["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

        signals = Runner(
            task=HardTask(
                measure_fn=lambda ctx, update_hook: (
                    ModularProgramV2(
                        soccfg,
                        ctx.cfg,
                        modules=[
                            make_reset("reset", reset_cfg=ctx.cfg.get("reset")),
                            Pulse(name="qub_pulse", cfg=ctx.cfg["qub_pulse"]),
                            Pulse(name="probe_pulse", cfg=ctx.cfg["probe_pulse"]),
                            make_readout("readout", readout_cfg=ctx.cfg["readout"]),
                        ],
                    ).acquire(soc, progress=progress, callback=update_hook)
                ),
                result_shape=(len(offsets), len(fpts)),
            ),
            liveplotter=LivePlotter2D(
                "Time (us)", "Frequency (MHz)", disable=not progress
            ),
            update_hook=lambda viewer, ctx: viewer.update(
                offsets, fpts, photonlookback_signal2real(np.asarray(ctx.get_data()))
            ),
        ).run(cfg)
        signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (offsets, fpts, signals)

        return offsets, fpts, signals

    def analyze(self, result: Optional[PhotonLookbackResultType] = None) -> float:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        offsets, fpts, signals = result

        amps = rotate2real(minus_background(signals, axis=1)).real
        amps /= np.std(amps, axis=1, keepdims=True)
        s_offsets, s_fpts = get_resonance_freq(offsets, fpts, amps)

        # plot the data and the fitted polynomial
        fig, ax = plt.subplots(figsize=config.figsize)

        ax.imshow(
            amps.T,
            origin="lower",
            interpolation="none",
            aspect="auto",
            extent=[offsets[0], offsets[-1], fpts[0], fpts[-1]],
        )
        ax.plot(s_offsets, s_fpts, ".", c="k")

        ax.set_xlabel("Time (us)", fontsize=14)
        ax.set_ylabel("Qubit Frequency (MHz)", fontsize=14)

        fig.tight_layout()
        plt.show()

    def save(
        self,
        filepath: str,
        result: Optional[PhotonLookbackResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/photon_lookback",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        offsets, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": offsets * 1e-6},
            y_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
