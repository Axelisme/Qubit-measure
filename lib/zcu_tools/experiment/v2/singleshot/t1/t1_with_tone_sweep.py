from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typing_extensions import NotRequired
from scipy.ndimage import gaussian_filter

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.liveplot import (
    LivePlotter1D,
    LivePlotter2D,
    MultiLivePlotter,
    make_plot_frame,
)
from zcu_tools.program.v2 import (
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
from zcu_tools.utils.fitting import fit_transition_rates
from zcu_tools.experiment.v2.utils import round_zcu_time

from ..util import calc_populations

# (values, times, signals)
T1SweepResult = Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


class T1WithToneSweepCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: NotRequired[PulseCfg]
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweepExp(AbsExperiment[T1SweepResult, T1WithToneSweepCfg]):
    def run(self, *args, uniform: bool = False, **kwargs) -> T1SweepResult:
        if uniform:
            return self._run_hard(*args, **kwargs)
        else:
            return self._run_non_unifrom(*args, **kwargs)

    def _run_non_unifrom(
        self,
        soc,
        soccfg,
        cfg: T1WithToneSweepCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1SweepResult:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        gain_sweep = cfg["sweep"]["gain"]
        len_sweep = cfg["sweep"]["length"]
        del cfg["sweep"]

        gains = np.sqrt(
            np.linspace(
                gain_sweep["start"] ** 2, gain_sweep["stop"] ** 2, gain_sweep["expts"]
            )
        )

        if isinstance(len_sweep, dict):
            ts = (
                np.linspace(
                    len_sweep["start"] ** (1 / 2),
                    len_sweep["stop"] ** (1 / 2),
                    len_sweep["expts"],
                )
                ** 2
            )
        else:
            ts = np.asarray(len_sweep)
        ts = round_zcu_time(ts, soccfg, gen_ch=cfg["test_pulse"]["ch"])
        ts = np.unique(ts)

        def measure_fn(ctx, update_hook):
            rounds = ctx.cfg.pop("rounds", 1)
            ctx.cfg["rounds"] = 1

            acc_populations = np.zeros((len(ts), 2), dtype=np.float64)
            for ir in range(rounds):
                for i, t1_delay in enumerate(ts):
                    Pulse.set_param(ctx.cfg["test_pulse"], "length", t1_delay)
                    raw_i = ModularProgramV2(
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
                        g_center=g_center,
                        e_center=e_center,
                        population_radius=radius,
                    )

                    acc_populations[i] += raw_i[0][0]

                update_hook(ir, acc_populations / (ir + 1))

            return acc_populations / rounds

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

            populations = run_task(
                task=SoftTask(
                    sweep_name="gain",
                    sweep_values=gains.tolist(),
                    update_cfg_fn=update_fn,
                    sub_task=HardTask(
                        measure_fn=measure_fn,
                        raw2signal_fn=lambda raw: raw,
                        result_shape=(len(ts), 2),
                        dtype=np.float64,
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
            populations = np.asarray(populations)
        plt.close(fig)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (gains, ts, populations)

        return gains, ts, populations

    def _run_hard(
        self,
        soc,
        soccfg,
        cfg: T1WithToneSweepCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1SweepResult:
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

            populations = run_task(
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
                        dtype=np.float64,
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
            populations = np.asarray(populations)
        plt.close(fig)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (gains, ts, populations)

        return gains, ts, populations

    def analyze(
        self,
        result: Optional[T1SweepResult] = None,
        *,
        ac_coeff: Optional[float] = None,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, Ts, populations = result
        populations = np.real(populations).astype(np.float64)

        valid_mask = np.all(np.isfinite(populations), axis=(1, 2))
        gains = gains[valid_mask]
        populations = populations[valid_mask]

        populations = gaussian_filter(populations, sigma=0.5, axes=(0, 1))

        populations = calc_populations(populations)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        num_gain = populations.shape[0]

        worst_loss = 0.0
        worst_pop = None
        worst_fit = None

        # (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        rates = np.zeros((num_gain, 6), dtype=np.float64)
        rate_Covs = np.zeros((num_gain, 6, 6), dtype=np.float64)
        for i, pop in enumerate(tqdm(populations, desc="Fitting transition rates")):
            rate, _, fit_pop, (_, pCov) = fit_transition_rates(Ts, pop)
            rates[i] = rate
            rate_Covs[i] = pCov[:6, :6]

            loss = np.mean(np.abs(fit_pop - pop))
            if loss > worst_loss:
                worst_loss = loss
                worst_pop = pop
                worst_fit = fit_pop

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

        axs = cast(List[List[Axes]], axs)

        def _plot_population(ax, pop, label):
            ax.scatter([], [], s=0, label=label)
            ax.imshow(pop.T, aspect="auto", extent=(xs[0], xs[-1], Ts[-1], Ts[0]))
            ax.set_ylabel("Time (μs)")
            ax.legend()

        _plot_population(axs[0][0], populations[..., 0], "Ground")
        _plot_population(axs[1][0], populations[..., 1], "Excited")
        _plot_population(axs[2][0], populations[..., 2], "Other")

        # (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        R_go = rates[..., 4]
        R_g = rates[..., 0] + rates[..., 4]
        R_eo = rates[..., 2]
        R_e = rates[..., 1] + rates[..., 2]
        # Rerr_go = np.sqrt(rate_Covs[..., 4, 4])
        # Rerr_g = np.sqrt(
        #     rate_Covs[..., 0, 0] + rate_Covs[..., 4, 4] + 2 * rate_Covs[..., 0, 4]
        # )
        # Rerr_eo = np.sqrt(rate_Covs[..., 2, 2])
        # Rerr_e = np.sqrt(
        #     rate_Covs[..., 1, 1] + rate_Covs[..., 2, 2] + 2 * rate_Covs[..., 1, 2]
        # )

        # g_kwargs = dict(capsize=2, color="blue")
        # axs[0][1].errorbar(xs, R_g, Rerr_g, label="Γ_ge + Γ_go", **g_kwargs)  # type: ignore
        # axs[0][1].errorbar(xs, R_go, Rerr_go, label="Γ_go", ls="--", **g_kwargs)  # type: ignore

        # e_kwargs = dict(capsize=2, color="red")
        # axs[1][1].errorbar(xs, R_e, Rerr_e, label="Γ_eg + Γ_eo", **e_kwargs)  # type: ignore
        # axs[1][1].errorbar(xs, R_eo, Rerr_eo, label="Γ_eo", ls="--", **e_kwargs)  # type: ignore

        # o_kwargs = dict(ls="--", capsize=2)
        # axs[2][1].errorbar(xs, R_eo, Rerr_eo, label="Γ_eo", color="red", **o_kwargs)  # type: ignore
        # axs[2][1].errorbar(xs, R_go, Rerr_go, label="Γ_go", color="blue", **o_kwargs)  # type: ignore

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
        result: Optional[T1SweepResult] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/ge/t1_with_tone_sweep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, Ts, populations = result
        _filepath = Path(filepath)

        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_g_population")),
            x_info={"name": "Readout Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={
                "name": "Ground Populations",
                "unit": "a.u.",
                "values": populations[..., 0].T,
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
                "values": populations[..., 1].T,
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: List[str], **kwargs) -> T1SweepResult:
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
        populations = np.stack([g_pop, e_pop], axis=-1)

        gains = gains.astype(np.float64)
        Ts = Ts.astype(np.float64)
        populations = np.real(populations).astype(np.float64)

        self.last_cfg = None
        self.last_result = (gains, Ts, populations)

        return gains, Ts, populations
