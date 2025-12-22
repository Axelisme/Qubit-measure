from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
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
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_transition_rates, fit_ge_decay

# (times, signals)
T1ResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def calc_transition_rate(g_p: float, e_p: float, t1: float) -> Tuple[float, float]:
    """Calculate transition rates from T1 times and steady populations."""
    if np.isclose(t1, 0.0, atol=1e-1) or not np.isfinite(t1):
        return np.nan, np.nan

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


class T1Experiment(AbsExperiment[T1ResultType, T1TaskConfig]):
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

    def analyze(
        self,
        result: Optional[T1ResultType] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        populations = calc_populations(signals)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        (
            (t1, _, g_fit_signals, g_params),
            (_, _, e_fit_signals, e_params),
        ) = fit_ge_decay(lens, populations[:, 0], populations[:, 1], share_t1=True)

        gamma_ge, gamma_eg = calc_transition_rate(g_params[0], e_params[0], t1)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.set_title(
            f"T_1 = {t1:.1f} μs, Γ_ge={gamma_ge:.3f} μs⁻¹, Γ_eg={gamma_eg:.3f} μs⁻¹"
        )
        ax.plot(lens, g_fit_signals, color="blue", ls="--", label="Ground Fit")
        ax.plot(lens, e_fit_signals, color="red", ls="--", label="Excited Fit")
        plot_kwargs = dict(ls="-", marker=".", markersize=3)
        ax.plot(lens, populations[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
        ax.plot(lens, populations[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
        ax.plot(lens, populations[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Population")
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

    def load(self, filepath: str, **kwargs) -> T1ResultType:
        signals, Ts, y_values = load_data(filepath, **kwargs)
        assert Ts is not None and y_values is not None
        assert len(Ts.shape) == 1 and len(y_values.shape) == 1
        assert signals.shape == (len(y_values), len(Ts))

        Ts = Ts * 1e6  # s -> us
        signals = signals.T  # transpose back

        Ts = Ts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (Ts, signals)

        return Ts, signals


class T1WithToneTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithTone(AbsExperiment[T1ResultType, T1WithToneTaskConfig]):
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

    def analyze(
        self,
        result: Optional[T1ResultType] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        populations = calc_populations(signals)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        rates, fit_pops, _ = fit_transition_rates(lens, populations)

        T_g = rates[0] + rates[5]
        T_e = rates[1] + rates[2]
        t1 = 1.0 / (T_g + T_e)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.set_title(f"T_1 = {t1:.1f} μs, Γ_ge={T_g:.3f} μs⁻¹, Γ_eg={T_e:.3f} μs⁻¹")
        ax.plot(lens, fit_pops[:, 0], color="blue", ls="--", label="Ground Fit")
        ax.plot(lens, fit_pops[:, 1], color="red", ls="--", label="Excited Fit")
        ax.plot(lens, fit_pops[:, 2], color="green", ls="--", label="Other Fit")
        plot_kwargs = dict(ls="-", marker=".", markersize=3)
        ax.plot(lens, populations[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
        ax.plot(lens, populations[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
        ax.plot(lens, populations[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Population")
        # ax.set_ylim(0.0, 1.0)
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

    def load(self, filepath: str, **kwargs) -> T1ResultType:
        signals, Ts, y_values = load_data(filepath, **kwargs)
        assert Ts is not None and y_values is not None
        assert len(Ts.shape) == 1 and len(y_values.shape) == 1
        assert signals.shape == (len(y_values), len(Ts))

        Ts = Ts * 1e6  # s -> us
        signals = signals.T  # transpose back

        Ts = Ts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (Ts, signals)

        return Ts, signals


# (values, times, signals)
T1SweepResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class T1WithToneSweepTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: NotRequired[PulseCfg]
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweep(AbsExperiment[T1SweepResultType, T1WithToneSweepTaskConfig]):
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

        gains = np.sqrt(
            np.linspace(
                gain_sweep["start"] ** 2, gain_sweep["stop"] ** 2, gain_sweep["expts"]
            )
        )
        ts = sweep2array(cfg["sweep"]["length"])  # predicted times

        fig, axs = make_plot_frame(2, 2, figsize=(12, 6))

        with MultiLivePlotter(
            fig,
            dict(
                plot_2d_g=LivePlotter2D(
                    "Readout Gain",
                    "Time (us)",
                    uniform=False,
                    existed_axes=[[axs[0][0]]],
                ),
                plot_2d_e=LivePlotter2D(
                    "Readout Gain",
                    "Time (us)",
                    uniform=False,
                    existed_axes=[[axs[1][0]]],
                ),
                plot_2d_o=LivePlotter2D(
                    "Readout Gain",
                    "Time (us)",
                    uniform=False,
                    existed_axes=[[axs[1][1]]],
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
                                    Pulse("pi_pulse", cfg=ctx.cfg.get("pi_pulse")),
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

    def analyze(
        self,
        result: Optional[T1SweepResultType] = None,
        *,
        ac_coeff: Optional[float] = None,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, Ts, signals = result

        valid_mask = np.all(np.isfinite(signals), axis=(1, 2))
        gains = gains[valid_mask]
        signals = signals[valid_mask]

        populations = calc_populations(signals)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        num_gain = populations.shape[0]
        min_rate = 0.1 / np.max(Ts)

        worst_loss = 0.0
        worst_pop = None
        worst_fit = None

        # (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        transition_rates = np.zeros((num_gain, 6), dtype=np.float64)
        for i, pop in enumerate(tqdm(populations, desc="Fitting transition rates")):
            rates, fit_pop, _ = fit_transition_rates(Ts, pop)
            transition_rates[i] = rates

            loss = np.mean(np.abs(fit_pop - pop))
            if loss > worst_loss:
                worst_loss = loss
                worst_pop = pop
                worst_fit = fit_pop
        transition_rates[transition_rates < min_rate] = 0.0

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        assert worst_pop is not None and worst_fit is not None
        ax.scatter(Ts, worst_pop[:, 0], label="G", color="blue", s=1)
        ax.scatter(Ts, worst_pop[:, 1], label="E", color="red", s=1)
        ax.scatter(Ts, worst_pop[:, 2], label="O", color="green", s=1)
        ax.plot(Ts, worst_fit[:, 0], color="blue", ls="--")
        ax.plot(Ts, worst_fit[:, 1], color="red", ls="--")
        ax.plot(Ts, worst_fit[:, 2], color="green", ls="--")
        ax.grid(True)
        ax.set_title(f"Worst fit loss: {worst_loss:.3e}")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Population")
        plt.show(fig)

        if ac_coeff is None:
            xs = gains
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * gains**2
            xlabel = r"$\bar n$"

        fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

        def _plot_population(ax, pop, label):
            ax.scatter([], [], s=0, label=label)
            ax.imshow(pop.T, aspect="auto", extent=(xs[0], xs[-1], Ts[-1], Ts[0]))
            ax.set_ylabel("Time (μs)")
            ax.legend()

        _plot_population(axs[0][0], populations[..., 0], "Ground")
        _plot_population(axs[1][0], populations[..., 1], "Excited")
        _plot_population(axs[2][0], populations[..., 2], "Other")

        # (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        R_go = transition_rates[..., 4]
        R_g = transition_rates[..., 0] + R_go
        R_eo = transition_rates[..., 2]
        R_e = transition_rates[..., 1] + R_eo

        axs[0][1].plot(xs, R_g, label="Γ_ge + Γ_go", color="blue")
        axs[0][1].plot(xs, R_go, label="Γ_go", color="blue", ls="--")

        axs[1][1].plot(xs, R_e, label="Γ_eg + Γ_eo", color="red")
        axs[1][1].plot(xs, R_eo, label="Γ_eo", color="red", ls="--")

        axs[2][1].plot(xs, R_eo, label="Γ_eo", color="red", ls="--")
        axs[2][1].plot(xs, R_go, label="Γ_go", color="blue", ls="--")

        max_rate = np.nanmax([R_g, R_e])
        for ax in (axs[0][1], axs[1][1], axs[2][1]):
            ax.legend()
            ax.set_ylim(0, max_rate * 1.1)
            ax.grid(True)
            ax.set_ylabel("Rate (μs⁻¹)")
        axs[2][1].set_xlabel(xlabel)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[T1SweepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/ge/t1_with_tone_sweep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, Ts, signals = result
        _filepath = Path(filepath)

        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_g_population")),
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
            filepath=str(_filepath.with_name(_filepath.name + "_e_population")),
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

    def load(self, filepath: List[str], **kwargs) -> T1SweepResultType:
        g_filepath, e_filepath = filepath

        # Load ground populations
        g_pop, gains, Ts = load_data(g_filepath, **kwargs)
        assert gains is not None and Ts is not None
        assert len(gains.shape) == 1 and len(Ts.shape) == 1
        assert g_pop.shape == (len(gains), len(Ts))

        # Load excited populations
        e_pop, gains_e, Ts_e = load_data(e_filepath, **kwargs)
        assert gains_e is not None and Ts_e is not None
        assert e_pop.shape == (len(gains_e), len(Ts_e))
        assert np.array_equal(gains, gains_e) and np.array_equal(Ts, Ts_e)

        Ts = Ts * 1e6  # s -> us

        # Reconstruct signals shape: (gains, ts, 2)
        signals = np.stack([g_pop, e_pop], axis=-1)

        gains = gains.astype(np.float64)
        Ts = Ts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (gains, Ts, signals)

        return gains, Ts, signals
