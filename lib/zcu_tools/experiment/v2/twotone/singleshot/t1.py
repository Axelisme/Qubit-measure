from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired


from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task, SoftTask
from zcu_tools.liveplot import (
    LivePlotter1D,
    LivePlotter2D,
    MultiLivePlotter,
    make_plot_frame,
)
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_ge_decay

# (times, signals)
T1ResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def calc_transition_rate(g_p: float, e_p: float, t1: float) -> Tuple[float, float]:
    """Calculate transition rates from T1 times and steady populations."""
    if t1 == 0.0:
        return 0.0, 0.0

    # Using detailed balance: p_g * gamma_ge = p_e * gamma_eg
    # And total rate: gamma_total = gamma_ge + gamma_eg = 1 / t1

    gamma_ge = (e_p / (g_p + e_p)) / t1
    gamma_eg = (g_p / (g_p + e_p)) / t1

    return gamma_ge, gamma_eg


def calc_populations(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    g_pop, e_pop = signals[..., 0], signals[..., 1]
    return np.stack([g_pop, e_pop, 1 - g_pop - e_pop], axis=-1).real


class T1TaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1Experiment(AbsExperiment):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: T1TaskConfig,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        ts = sweep2array(cfg["sweep"]["length"])

        with LivePlotter1D(
            "Time (us)",
            "Amplitude",
            segment_kwargs=dict(
                num_lines=3,
                line_kwargs=[
                    dict(label="Ground"),
                    dict(label="Excited"),
                    dict(label="Other"),
                ],
            ),
        ) as viewer:
            viewer.get_ax().set_ylim(0.0, 1.0)

            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                                Delay(
                                    name="t1_delay",
                                    delay=sweep2param(
                                        "length", ctx.cfg["sweep"]["length"]
                                    ),
                                ),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(
                            soc,
                            progress=False,
                            callback=update_hook,
                            g_center=g_center,
                            e_center=e_center,
                            population_radius=radius,
                        )
                    ),
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(ts), 2),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(ts, calc_populations(ctx.data).T),
            )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, signals)

        return ts, signals

    def analyze(self, result: Optional[T1ResultType] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        populations = calc_populations(signals)

        (
            (t1, _, g_fit_signals, g_params),
            (_, _, e_fit_signals, e_params),
        ) = fit_ge_decay(lens, populations[:, 0], populations[:, 1], share_t1=True)

        gamma_ge, gamma_eg = calc_transition_rate(g_params[0], e_params[0], t1)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.set_title(f"Γ_ge={gamma_ge:.3f} μs⁻¹, Γ_eg={gamma_eg:.3f} μs⁻¹")
        ax.plot(lens, g_fit_signals, color="blue", ls="--", label="Ground Fit")
        ax.plot(lens, e_fit_signals, color="red", ls="--", label="Excited Fit")
        plot_kwargs = dict(ls="-", marker=".", markersize=3)
        ax.plot(lens, populations[:, 0], color="blue", label="Ground", **plot_kwargs)
        ax.plot(lens, populations[:, 1], color="red", label="Excited", **plot_kwargs)
        ax.plot(lens, populations[:, 2], color="green", label="Other", **plot_kwargs)
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Population")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[T1ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            y_info={"name": "GE population", "unit": "a.u.", "values": [0, 1]},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )


class T1WithToneTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneExperiment(AbsExperiment):
    def run(
        self,
        soc,
        soccfg,
        cfg: T1WithToneTaskConfig,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        Pulse.set_param(
            cfg["test_pulse"], "length", sweep2param("length", cfg["sweep"]["length"])
        )

        ts = sweep2array(cfg["sweep"]["length"])

        with LivePlotter1D(
            "Time (us)",
            "Amplitude",
            segment_kwargs=dict(
                num_lines=3,
                line_kwargs=[
                    dict(label="Ground"),
                    dict(label="Excited"),
                    dict(label="Other"),
                ],
            ),
        ) as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                                Pulse("test_pulse", ctx.cfg["test_pulse"]),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(
                            soc,
                            progress=False,
                            callback=update_hook,
                            g_center=g_center,
                            e_center=e_center,
                            population_radius=radius,
                        )
                    ),
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(ts), 2),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(ts, calc_populations(ctx.data).T),
            )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, signals)

        return ts, signals

    def analyze(self, result: Optional[T1ResultType] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        populations = calc_populations(signals)

        (
            (t1, _, g_fit_signals, g_params),
            (_, _, e_fit_signals, e_params),
        ) = fit_ge_decay(lens, populations[:, 0], populations[:, 1], share_t1=True)

        gamma_ge, gamma_eg = calc_transition_rate(g_params[0], e_params[0], t1)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.set_title(f"Γ_ge={gamma_ge:.3f} μs⁻¹, Γ_eg={gamma_eg:.3f} μs⁻¹")
        ax.plot(lens, g_fit_signals, color="blue", ls="--", label="Ground Fit")
        ax.plot(lens, e_fit_signals, color="red", ls="--", label="Excited Fit")
        plot_kwargs = dict(ls="-", marker=".", markersize=3)
        ax.plot(lens, populations[:, 0], color="blue", label="Ground", **plot_kwargs)
        ax.plot(lens, populations[:, 1], color="red", label="Excited", **plot_kwargs)
        ax.plot(lens, populations[:, 2], color="green", label="Other", **plot_kwargs)
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Population")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[T1ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1_with_tone",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            y_info={"name": "GE population", "unit": "a.u.", "values": [0, 1]},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )


# (values, times, signals)
T1SweepResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class T1WithToneSweepTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweepExperiment(AbsExperiment):
    def run(
        self,
        soc,
        soccfg,
        cfg: T1WithToneSweepTaskConfig,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1SweepResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        gain_sweep = cfg["sweep"]["gain"]
        cfg["sweep"] = {"length": cfg["sweep"]["length"]}

        Pulse.set_param(
            cfg["test_pulse"], "length", sweep2param("length", cfg["sweep"]["length"])
        )

        gains = sweep2array(gain_sweep, allow_array=True)  # predicted
        ts = sweep2array(cfg["sweep"]["length"])  # predicted times

        fig, axs = make_plot_frame(2, 2, figsize=(8, 8))

        with MultiLivePlotter(
            fig,
            dict(
                plot_2d_g=LivePlotter2D(
                    "Readout Gain", "Time (us)", existed_axes=[[axs[0][0]]]
                ),
                plot_2d_e=LivePlotter2D(
                    "Readout Gain", "Time (us)", existed_axes=[[axs[1][0]]]
                ),
                plot_2d_o=LivePlotter2D(
                    "Readout Gain", "Time (us)", existed_axes=[[axs[1][1]]]
                ),
                plot_1d=LivePlotter1D(
                    "Time (us)",
                    "Population",
                    existed_axes=[[axs[0][1]]],
                    segment_kwargs=dict(
                        num_lines=3,
                        line_kwargs=[
                            dict(label="Ground"),
                            dict(label="Excited"),
                            dict(label="Other"),
                        ],
                    ),
                ),
            ),
        ) as viewer:

            def update_fn(i, ctx, gain) -> None:
                Pulse.set_param(ctx.cfg["test_pulse"], "gain", gain)
                ctx.env_dict["idx"] = i

            def plot_fn(ctx) -> None:
                i = ctx.env_dict["idx"]

                populations = calc_populations(np.asarray(ctx.data))

                viewer.get_plotter("plot_2d_g").update(
                    gains, ts, populations[..., 0], refresh=False
                )
                viewer.get_plotter("plot_2d_e").update(
                    gains, ts, populations[..., 1], refresh=False
                )
                viewer.get_plotter("plot_2d_o").update(
                    gains, ts, populations[..., 2], refresh=False
                )
                viewer.get_plotter("plot_1d").update(
                    ts, populations[i].T, refresh=False
                )

                viewer.refresh()

            signals = run_task(
                task=SoftTask(
                    sweep_name="gain",
                    sweep_values=gains.tolist(),
                    update_cfg_fn=update_fn,
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset(
                                        "reset", ctx.cfg.get("reset", {"type": "none"})
                                    ),
                                    Pulse(name="pi_pulse", cfg=ctx.cfg["pi_pulse"]),
                                    Pulse(name="test_pulse", cfg=ctx.cfg["test_pulse"]),
                                    Readout("readout", cfg=ctx.cfg["readout"]),
                                ],
                            ).acquire(
                                soc,
                                progress=False,
                                callback=update_hook,
                                g_center=g_center,
                                e_center=e_center,
                                population_radius=radius,
                            )
                        ),
                        raw2signal_fn=lambda raw: raw[0][0],
                        result_shape=(len(ts), 2),
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
            signals = np.asarray(signals)
        plt.close(fig)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (gains, ts, signals)

        return gains, ts, signals

    def analyze(self, result: Optional[T1SweepResultType] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, Ts, signals = result

        populations = calc_populations(signals)

        gammas = np.zeros((len(gains), 2))
        for i in range(len(gains)):
            (
                (t1, _, _, g_params),
                (_, _, _, e_params),
            ) = fit_ge_decay(
                Ts, populations[i, :, 0], populations[i, :, 1], share_t1=True
            )
            gamma_ge, gamma_eg = calc_transition_rate(g_params[0], e_params[0], t1)
            gammas[i] = (gamma_ge, gamma_eg)

        fig, axs = make_plot_frame(2, 2, figsize=(8, 8))
        assert isinstance(fig, Figure)

        axs[0][0].set_title("Ground Population")
        im1 = axs[0][0].imshow(
            populations[..., 0].T,
            extent=[gains[0], gains[-1], Ts[-1], Ts[0]],
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        fig.colorbar(im1, ax=axs[0][0], label="Population")
        axs[0][0].set_xlabel("Readout Gain")
        axs[0][0].set_ylabel("Time (s)")

        axs[0][1].set_title("Excited Population")
        im2 = axs[0][1].imshow(
            populations[..., 1].T,
            extent=[gains[0], gains[-1], Ts[-1], Ts[0]],
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        fig.colorbar(im2, ax=axs[0][1], label="Population")
        axs[0][1].set_xlabel("Readout Gain")
        axs[0][1].set_ylabel("Time (s)")

        axs[1][0].set_title("Other Population")
        im3 = axs[1][0].imshow(
            populations[..., 2].T,
            extent=[gains[0], gains[-1], Ts[-1], Ts[0]],
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        fig.colorbar(im3, ax=axs[1][0], label="Population")

        axs[1][1].set_title("Transition Rates")
        axs[1][1].plot(
            gains, gammas[:, 0], label="Γ_ge", color="blue", marker=".", ls="-"
        )
        axs[1][1].plot(
            gains, gammas[:, 1], label="Γ_eg", color="red", marker=".", ls="-"
        )
        axs[1][1].set_xlabel("Readout Gain")
        axs[1][1].set_ylabel("Transition Rate (μs⁻¹)")
        axs[1][1].legend()
        axs[1][1].grid(True)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[T1SweepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1_with_tone_sweep_singleshot",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, Ts, signals = result
        filepath = Path(filepath)

        save_data(
            filepath=filepath.with_name(filepath.name + "_g_population.npz"),
            x_info={"name": "Readout Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={
                "name": "Ground Populations",
                "unit": "a.u.",
                "values": signals[..., 0].T,
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )
        save_data(
            filepath=filepath.with_name(filepath.name + "_e_population.npz"),
            x_info={"name": "Readout Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={
                "name": "Excited Populations",
                "unit": "a.u.",
                "values": signals[..., 1].T,
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )
