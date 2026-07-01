from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import NonUniformImage
from numpy import float64
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.singleshot.util import (
    calc_populations,
    correct_populations,
)
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D, LivePlot2D, MultiLivePlot, make_plot_frame
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import fitlor
from zcu_tools.utils.process import minus_background


def _default_population_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class AcStarkResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    populations: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=_default_population_states
    )
    cfg_snapshot: AcStarkCfg | None = None


def get_resonance_freq(
    xs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    populations: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s_xs = []
    s_freqs = []

    prev_freq = np.nan
    for x, pop in zip(xs, populations):
        if np.any(np.isnan(pop)):
            continue

        param, _ = fitlor(freqs, pop)
        curr_freq = param[3]

        if abs(curr_freq - prev_freq) > 0.1 * (freqs[-1] - freqs[0]):
            continue

        prev_freq = curr_freq

        s_xs.append(x)
        s_freqs.append(curr_freq)

    return np.array(s_xs), np.array(s_freqs)


class AcStarkModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    stark_pulse1: PulseCfg
    stark_pulse2: PulseCfg
    readout: ReadoutCfg


class AcStarkSweepCfg(ConfigBase):
    gain: SweepCfg
    freq: SweepCfg


class AcStarkCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AcStarkModuleCfg
    sweep: AcStarkSweepCfg


class AcStarkExp(PersistableExperiment[AcStarkResult, AcStarkCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "population_states",
                "GE Population",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ, dtype=np.float64),
            Axis(
                "gains",
                "Stark Pulse Gain",
                "a.u.",
                scale=IDENTITY,
                dtype=np.float64,
            ),
        ),
        z=ZSpec("populations", "Population", "a.u.", dtype=np.float64),
        result_type=AcStarkResult,
        cfg_type=AcStarkCfg,
        tag="singleshot/ac_stark",
    )

    def run(
        self,
        soc,
        soccfg,
        cfg: AcStarkCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> AcStarkResult:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gain_sweep = cfg.sweep.gain

        # uniform in square space
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.stark_pulse2.ch},
        )
        gains = np.sqrt(
            np.linspace(gain_sweep.start**2, gain_sweep.stop**2, gain_sweep.expts)
        )
        gains = sweep2array(
            gains, "gain", {"soccfg": soccfg, "gen_ch": modules.stark_pulse1.ch}
        )

        freq_param = sweep2param("freq", cfg.sweep.freq)
        modules.stark_pulse2.set_param("freq", freq_param)

        fig, axs = make_plot_frame(2, 2, plot_instant=True, figsize=(8, 6))

        def make_plotter2d(ax: Axes) -> LivePlot2D:
            return LivePlot2D(
                "Stark Pulse Gain (a.u.)",
                "Probe Frequency (MHz)",
                uniform=False,
                segment_kwargs=dict(vmin=0.0, vmax=1.0),
                existed_axes=[[ax]],
            )

        def make_plotter1d(ax: Axes) -> LivePlot1D:
            ax.set_ylim(0.0, 1.0)
            return LivePlot1D(
                "Probe Frequency (MHz)",
                "Population",
                existed_axes=[[ax]],
                segment_kwargs=dict(
                    num_lines=3,
                    line_kwargs=[
                        dict(label="Ground"),
                        dict(label="Excited"),
                        dict(label="Other"),
                    ],
                ),
            )

        with MultiLivePlot(
            fig,
            dict(
                g_2d=make_plotter2d(axs[0][0]),
                e_2d=make_plotter2d(axs[0][1]),
                o_2d=make_plotter2d(axs[1][0]),
                cur_1d=make_plotter1d(axs[1][1]),
            ),
        ) as viewer:

            def measure_fn(
                ctx: TaskState[NDArray[np.float64], Any, AcStarkCfg], update_hook
            ) -> list[NDArray[float64]]:
                modules = ctx.cfg.modules
                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        Pulse("stark_pulse1", modules.stark_pulse1, block_mode=False),
                        Pulse("stark_pulse2", modules.stark_pulse2),
                        Readout("readout", modules.readout),
                    ],
                    sweep=[("freq", ctx.cfg.sweep.freq)],
                ).acquire(
                    soc,
                    progress=False,
                    round_hook=update_hook,
                    stop_checkers=[ctx.is_stop],
                    g_center=g_center,
                    e_center=e_center,
                    ge_radius=radius,
                )

            with MeasureSession(cfg) as run:

                def plot_fn(data: NDArray[np.float64]) -> None:
                    i = int(run.env.get("idx", 0))

                    populations = calc_populations(data)

                    viewer.get_plotter("g_2d").update(
                        gains, freqs, populations[..., 0], refresh=False
                    )
                    viewer.get_plotter("e_2d").update(
                        gains, freqs, populations[..., 1], refresh=False
                    )
                    viewer.get_plotter("o_2d").update(
                        gains, freqs, populations[..., 2], refresh=False
                    )
                    viewer.get_plotter("cur_1d").update(
                        freqs, populations[i, :].T, refresh=False
                    )

                    viewer.refresh()

                buffer = run.buffer(
                    (len(gains), len(freqs), 2),
                    dtype=np.float64,
                    on_update=plot_fn,
                )
                for step in run.scan("resonator gain", gains.tolist()):
                    step.cfg.modules.stark_pulse1.set_param("gain", step.value)
                    run.env["idx"] = step.index
                    buffer[step].measure(
                        measure_fn,
                        raw2signal_fn=lambda raw: raw[0][0],
                        pbar_n=1,
                    )
                signals = buffer.array
        plt.close(fig)

        # Cache results
        self.last_result = AcStarkResult(
            gains=gains, freqs=freqs, populations=signals, cfg_snapshot=cfg
        )

        return self.last_result

    def analyze(
        self,
        chi: float,
        result: AcStarkResult | None = None,
        *,
        kappa: float,
        confusion_matrix: NDArray[np.float64] | None = None,
        cutoff: float | None = None,
    ) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, freqs, populations = result.gains, result.freqs, result.populations

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            populations = populations[valid_indices]

        populations = calc_populations(populations)  # (xs, 2, Ts, 3)

        populations = correct_populations(populations, confusion_matrix)

        # merge two populations into one
        populations = (
            np.abs(minus_background(populations[..., 0]))
            + np.abs(minus_background(populations[..., 1]))
        ) / 2

        s_gains, s_freqs = get_resonance_freq(gains, freqs, populations)

        gains2 = gains**2
        s_gains2 = s_gains**2

        # fitting max_freqs with ax2 + bx + c
        x2_fit = np.linspace(min(gains2), max(gains2), 100)
        b, c = np.polyfit(s_gains2, s_freqs, 1)
        y_fit = b * x2_fit + c

        # Calculate the Stark shift. eta accounts for the finite resonator
        # linewidth kappa relative to the dispersive shift chi (matches the
        # twotone AcStarkExp.analyze formula).
        eta = kappa**2 / (kappa**2 + chi**2)
        ac_coeff = abs(b) / (2 * eta * chi)

        # plot the data and the fitted polynomial
        avg_n = ac_coeff * gains2

        fig, ax1 = plt.subplots()
        assert isinstance(fig, Figure)

        # Use NonUniformImage for better visualization with gain^2 as x-axis
        im = NonUniformImage(ax1, cmap="RdBu_r", interpolation="nearest")
        im.set_data(avg_n, freqs, populations.T)
        im.set_extent((avg_n[0], avg_n[-1], freqs[0], freqs[-1]))
        ax1.add_image(im)

        # Set proper limits for the plot
        ax1.set_xlim(avg_n[0], avg_n[-1])
        ax1.set_ylim(freqs[0], freqs[-1])

        # Plot the resonance frequencies and fitted curve
        ax1.plot(ac_coeff * s_gains2, s_freqs, ".", c="k")

        # Fit curve in terms of gain^2
        label = r"$\bar n$" + f" = {ac_coeff:.2g} " + r"$gain^2$"
        gain_fit = ac_coeff * x2_fit
        ax1.plot(gain_fit, y_fit, "-", label=label, color="y")

        # Create secondary x-axis for gain^2 (Readout Gain²)
        ax2 = ax1.twiny()

        # main x-axis: avg_n, secondary x-axis: gain^2
        # avg_n = ac_coeff * gains^2
        ax1.set_xticks(ax1.get_xticks())
        # ax1.set_xticklabels([f"{avg_n:.1f}" for avg_n in ax1.get_xticks()])
        ax1.set_xlabel(r"Average Photon Number ($\bar n$)", fontsize=14)

        # 上方次 x 軸顯示 gain
        avgn_ticks = ax1.get_xticks()
        gain_ticks = np.sqrt(avgn_ticks / ac_coeff)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(avgn_ticks)
        ax2.set_xticklabels([f"{gain:.2g}" for gain in gain_ticks])
        ax2.set_xlabel("Readout Gain (a.u.)", fontsize=14)

        ax1.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
        ax1.legend(fontsize="x-large")
        ax1.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return ac_coeff, fig

    def plot(
        self,
        result: AcStarkResult | None = None,
        *,
        ac_coeff: float,
        confusion_matrix: NDArray[np.float64] | None = None,
        cutoff: float | None = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, freqs, populations = result.gains, result.freqs, result.populations

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            populations = populations[valid_indices]

        populations = calc_populations(populations)  # (xs, 2, Ts, 3)

        populations = correct_populations(populations, confusion_matrix)

        gains2 = gains**2

        # plot the data and the fitted polynomial
        photons = ac_coeff * gains2

        # fig, ax1 = plt.subplots(figsize=config.figsize)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

        max_p = np.max(populations).item()

        # Use NonUniformImage for better visualization with gain^2 as x-axis
        im = NonUniformImage(ax1, cmap="RdBu_r", interpolation="nearest")
        im.set_data(photons, freqs, populations[..., 0].T)
        im.set_extent((photons[0], photons[-1], freqs[0], freqs[-1]))
        im.set_clim(0.0, max_p)
        ax1.add_image(im)
        ax1.set_xlim(photons[0], photons[-1])
        ax1.set_ylim(freqs[0], freqs[-1])
        ax1.set_aspect("auto")
        ax1.set_title("Ground")
        ax1.set_xlabel(r"$\bar n$", fontsize=14)
        ax1.set_ylabel("Frequency (MHz)", fontsize=14)

        im = NonUniformImage(ax2, cmap="RdBu_r", interpolation="nearest")
        im.set_data(photons, freqs, populations[..., 1].T)
        im.set_extent((photons[0], photons[-1], freqs[0], freqs[-1]))
        im.set_clim(0.0, max_p)
        ax2.add_image(im)
        ax2.set_xlim(photons[0], photons[-1])
        ax2.set_ylim(freqs[0], freqs[-1])
        ax2.set_aspect("auto")
        ax2.set_title("Excited")
        ax2.set_xlabel(r"$\bar n$", fontsize=14)

        im = NonUniformImage(ax3, cmap="RdBu_r", interpolation="nearest")
        im.set_data(photons, freqs, populations[..., 2].T)
        im.set_extent((photons[0], photons[-1], freqs[0], freqs[-1]))
        im.set_clim(0.0, max_p)
        ax3.add_image(im)
        ax3.set_xlim(photons[0], photons[-1])
        ax3.set_ylim(freqs[0], freqs[-1])
        ax3.set_aspect("auto")
        ax3.set_title("Other")
        ax3.set_xlabel(r"$\bar n$", fontsize=14)

        fig.tight_layout()

        return fig
