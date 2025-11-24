from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fitlor
from zcu_tools.utils.process import minus_background, rotate2real

from ..runner import HardTask, SoftTask, TaskConfig, run_task

# (amps, freqs, signals2D)
AcStarkResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def acstark_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    real_signals = rotate2real(minus_background(signals, axis=1)).real

    valid_mask = np.any(~np.isnan(real_signals), axis=1)

    if not np.any(valid_mask):
        return real_signals

    valid_signals = real_signals[valid_mask, :]

    min_vals = np.nanmin(valid_signals, axis=1, keepdims=True)
    max_vals = np.nanmax(valid_signals, axis=1, keepdims=True)
    valid_signals = (valid_signals - min_vals) / (max_vals - min_vals)

    real_signals[valid_mask, :] = valid_signals

    return real_signals


def get_resonance_freq(
    xs: NDArray[np.float64], fpts: NDArray[np.float64], amps: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    s_xs = []
    s_fpts = []

    prev_freq = np.nan
    for x, amp in zip(xs, amps):
        if np.any(np.isnan(amp)):
            continue

        curr_freq = fitlor(fpts, amp)[0][3]

        if abs(curr_freq - prev_freq) > 0.1 * (fpts[-1] - fpts[0]):
            continue

        prev_freq = curr_freq

        s_xs.append(x)
        s_fpts.append(curr_freq)

    return np.array(s_xs), np.array(s_fpts)


class AcStarkTaskConfig(TaskConfig, ModularProgramCfg):
    stark_pulse1: PulseCfg
    stark_pulse2: PulseCfg
    readout: ReadoutCfg


class AcStarkExperiment(AbsExperiment):
    def run(
        self,
        soc,
        soccfg,
        cfg: AcStarkTaskConfig,
        *,
        earlystop_snr: Optional[float] = None,
    ) -> AcStarkResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        assert isinstance(cfg["sweep"], dict)
        gain_sweep = cfg["sweep"].pop("gain")

        # uniform in square space
        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequencies
        pdrs = np.sqrt(
            np.linspace(
                gain_sweep["start"] ** 2, gain_sweep["stop"] ** 2, gain_sweep["expts"]
            )
        )

        Pulse.set_param(
            cfg["stark_pulse2"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        with LivePlotter2DwithLine(
            "Stark Pulse Gain (a.u.)",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=2,
            uniform=False,
        ) as viewer:
            ax1d = viewer.get_ax("1d")

            signals = run_task(
                task=SoftTask(
                    sweep_name="resonator gain",
                    sweep_values=pdrs.tolist(),
                    update_cfg_fn=lambda _, ctx, pdr: Pulse.set_param(
                        ctx.cfg["stark_pulse1"], "gain", pdr
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            (
                                prog := ModularProgramV2(
                                    soccfg,
                                    ctx.cfg,
                                    modules=[
                                        Reset(
                                            "reset",
                                            ctx.cfg.get("reset", {"type": "none"}),
                                        ),
                                        Pulse("stark_pulse1", ctx.cfg["stark_pulse1"]),
                                        Pulse("stark_pulse2", ctx.cfg["stark_pulse2"]),
                                        Readout("readout", ctx.cfg["readout"]),
                                    ],
                                )
                            ).acquire(
                                soc,
                                progress=False,
                                callback=wrap_earlystop_check(
                                    prog,
                                    update_hook,
                                    earlystop_snr,
                                    signal2real_fn=np.abs,
                                    snr_hook=lambda snr: ax1d.set_title(
                                        f"snr = {snr:.1f}"
                                    ),
                                ),
                            )
                        ),
                        result_shape=(len(fpts),),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    pdrs, fpts, acstark_signal2real(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (pdrs, fpts, signals)

        return pdrs, fpts, signals

    def analyze(
        self,
        result: Optional[AcStarkResultType] = None,
        *,
        chi: float,
        kappa: float,
        deg: int = 1,
        cutoff: Optional[float] = None,
    ) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        pdrs, fpts, signals = result

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(pdrs < cutoff)[0]
            pdrs = pdrs[valid_indices]
            signals = signals[valid_indices, :]

        amps = acstark_signal2real(signals)
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
        avg_n = ac_coeff * pdrs2

        fig, ax1 = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        # Use NonUniformImage for better visualization with pdr^2 as x-axis
        im = NonUniformImage(ax1, cmap="viridis", interpolation="nearest")
        im.set_data(avg_n, fpts, amps.T)
        im.set_extent((avg_n[0], avg_n[-1], fpts[0], fpts[-1]))
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
        plt.show(fig)

        return ac_coeff, fig

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
