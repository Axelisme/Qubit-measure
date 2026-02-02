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
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramCfg,
    ModularProgramV2,
    NonBlocking,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fitlor
from zcu_tools.utils.process import rotate2real


# (amps, freqs, signals2D)
AcStarkResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def acstark_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    real_signals = rotate2real(signals).real

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

        param, _ = fitlor(fpts, amp)
        curr_freq = param[3]

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


class AcStarkExp(AbsExperiment[AcStarkResultType, AcStarkTaskConfig]):
    def run(
        self,
        soc,
        soccfg,
        cfg: AcStarkTaskConfig,
        *,
        earlystop_snr: Optional[float] = None,
    ) -> AcStarkResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        if cfg["stark_pulse1"].get("block_mode", True):
            raise ValueError("Stark pulse 1 must not in block mode")

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
        ax2.set_xticklabels([f"{pdr:.2g}" for pdr in pdr_ticks])
        ax2.set_xlabel("Readout Gain (a.u.)", fontsize=14)

        ax1.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
        ax1.legend(fontsize="x-large")
        ax1.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

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

    def load(self, filepath: str, **kwargs) -> AcStarkResultType:
        signals2D, pdrs, fpts = load_data(filepath, **kwargs)
        assert fpts is not None
        assert len(pdrs.shape) == 1 and len(fpts.shape) == 1
        assert signals2D.shape == (len(pdrs), len(fpts))

        fpts = fpts * 1e-6  # Hz -> MHz

        pdrs = pdrs.astype(np.float64)
        fpts = fpts.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (pdrs, fpts, signals2D)

        return pdrs, fpts, signals2D


def acstark_ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AcStarkRamseyTaskConfig(TaskConfig, ModularProgramCfg):
    stark_pulse: PulseCfg
    pi_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg

    wait_delay: float


class AcStarkRamseyExp(AbsExperiment):
    def run(
        self, soc, soccfg, cfg: AcStarkRamseyTaskConfig, *, detune: float = 0.0
    ) -> AcStarkResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        assert isinstance(cfg["sweep"], dict)
        gain_sweep = cfg["sweep"].pop("gain")

        # uniform in square space
        lens = sweep2array(cfg["sweep"]["length"])
        pdrs = np.sqrt(
            np.linspace(
                gain_sweep["start"] ** 2, gain_sweep["stop"] ** 2, gain_sweep["expts"]
            )
        )

        t2r_spans = sweep2param("length", cfg["sweep"]["length"])

        with LivePlotter2DwithLine(
            "Stark Pulse Gain (a.u.)",
            "Time (us)",
            line_axis=1,
            num_lines=2,
            uniform=False,
        ) as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="resonator gain",
                    sweep_values=pdrs.tolist(),
                    update_cfg_fn=lambda _, ctx, pdr: Pulse.set_param(
                        ctx.cfg["stark_pulse"], "gain", pdr
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset(
                                        "reset",
                                        ctx.cfg.get("reset", {"type": "none"}),
                                    ),
                                    NonBlocking(
                                        [
                                            Delay(
                                                "wait_delay",
                                                delay=ctx.cfg["wait_delay"],
                                                hard_delay=False,
                                            ),
                                            Pulse("pi2_pulse1", ctx.cfg["pi2_pulse"]),
                                            Delay(
                                                "t2_delay",
                                                delay=t2r_spans,
                                                hard_delay=False,
                                            ),
                                            Pulse(
                                                name="pi2_pulse2",
                                                cfg={
                                                    **ctx.cfg["pi2_pulse"],
                                                    "phase": ctx.cfg["pi2_pulse"][
                                                        "phase"
                                                    ]
                                                    + 360 * detune * t2r_spans,
                                                },
                                            ),
                                        ]
                                    ),
                                    Pulse("stark_pulse", ctx.cfg["stark_pulse"]),
                                    Readout("readout", ctx.cfg["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        ),
                        result_shape=(len(lens),),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    pdrs, lens, acstark_ramsey_signal2real(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (pdrs, lens, signals)

        return pdrs, lens, signals

    def analyze(
        self,
        result: Optional[AcStarkResultType] = None,
        *,
        detune: float = 0.0,
        cutoff: Optional[float] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        pdrs, lens, signals = result

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(pdrs < cutoff)[0]
            pdrs = pdrs[valid_indices]
            signals = signals[valid_indices, :]

        min_interval = 0.2  # MHz
        n_length = signals.shape[1]
        n_length = max(n_length, int(n_length / (np.ptp(lens) * min_interval)))

        real_signals = acstark_ramsey_signal2real(signals)
        fft_signals = np.abs(
            np.fft.fft(
                signals - np.mean(signals, axis=1, keepdims=True), n=n_length, axis=1
            )
        )
        fft_freqs = np.fft.fftfreq(n_length, d=(np.ptp(lens) / (signals.shape[1] - 1)))
        freq_mask = np.logical_and(fft_freqs >= detune - 5.0, fft_freqs <= detune + 5.0)
        fft_freqs = fft_freqs[freq_mask] - detune
        fft_signals = fft_signals[:, freq_mask]

        fft_signals /= np.max(fft_signals, axis=1, keepdims=True)

        pdrs2 = pdrs**2

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        assert isinstance(fig, Figure)

        im1 = NonUniformImage(ax1, cmap="viridis", interpolation="nearest")
        im1.set_data(pdrs2, lens, real_signals.T)
        im1.set_extent((pdrs2[0], pdrs2[-1], lens[0], lens[-1]))
        ax1.add_image(im1)
        ax1.set_ylim(lens[0], lens[-1])
        ax1.set_ylabel("Time (us)", fontsize=14)

        im2 = NonUniformImage(ax2, cmap="viridis", interpolation="nearest")
        im2.set_data(pdrs2, fft_freqs, fft_signals.T)
        im2.set_extent((pdrs2[0], pdrs2[-1], fft_freqs[0], fft_freqs[-1]))
        ax2.add_image(im2)
        ax2.set_xlim(pdrs2[0], pdrs2[-1])
        ax2.set_ylim(-5.0, 5.0)
        ax2.set_xlabel("Stark Pulse Gain² (a.u.²)", fontsize=14)
        ax2.set_ylabel("FFT Detune (MHz)", fontsize=14)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[AcStarkResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ac_stark_ramsey",
        **kwargs,
    ) -> None:
        """Save AC Stark experiment data."""
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        pdrs, lens, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Stark Pulse Gain", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Time", "unit": "s", "values": lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> AcStarkResultType:
        signals2D, pdrs, lens = load_data(filepath, **kwargs)
        assert pdrs is not None and lens is not None
        assert len(pdrs.shape) == 1 and len(lens.shape) == 1
        assert signals2D.shape == (len(lens), len(pdrs))

        lens = lens * 1e6  # s -> us
        signals2D = signals2D.T  # transpose back

        pdrs = pdrs.astype(np.float64)
        lens = lens.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (pdrs, lens, signals2D)

        return pdrs, lens, signals2D
