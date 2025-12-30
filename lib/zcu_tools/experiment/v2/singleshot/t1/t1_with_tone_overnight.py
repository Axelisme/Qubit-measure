from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typing_extensions import List, NotRequired, Optional, Tuple

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import (
    HardTask,
    RepeatOverTime,
    ReTryIfFail,
    TaskConfig,
    TaskContextView,
    run_task,
)
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

T1WithToneOvernightResult = Tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]
]


class T1WithToneOvernightCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    test_pulse: PulseCfg
    readout: ReadoutCfg

    interval: float


class T1WithToneOvernightExp(
    AbsExperiment[T1WithToneOvernightResult, T1WithToneOvernightCfg]
):
    def run(self, *args, unifrom: bool = False, **kwargs) -> T1WithToneOvernightResult:
        if unifrom:
            return self._run_uniform(*args, **kwargs)
        else:
            return self._run_non_uniform(*args, **kwargs)

    def _run_uniform(
        self,
        soc,
        soccfg,
        cfg: T1WithToneOvernightCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        num_times: int = 50,
        fail_retry: int = 3,
    ) -> T1WithToneOvernightResult:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        Pulse.set_param(
            cfg["test_pulse"], "length", sweep2param("length", cfg["sweep"]["length"])
        )

        iters = np.arange(num_times)
        ts = sweep2array(cfg["sweep"]["length"])

        fig, axs = make_plot_frame(2, 2, figsize=(12, 6))
        axs[0][1].set_ylim(0, 1)

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

            def plot_fn(ctx: TaskContextView) -> None:
                i = ctx.env_dict["repeat_idx"]

                populations = calc_populations(np.asarray(ctx.data))

                viewer.get_plotter("plot_2d_g").update(
                    iters, ts, populations[..., 0], refresh=False
                )
                viewer.get_plotter("plot_2d_e").update(
                    iters, ts, populations[..., 1], refresh=False
                )
                viewer.get_plotter("plot_2d_o").update(
                    iters, ts, populations[..., 2], refresh=False
                )
                viewer.get_plotter("plot_1d").update(
                    ts, populations[i].T, refresh=False
                )

                viewer.refresh()

            populations = run_task(
                task=RepeatOverTime(
                    name="Iteration",
                    num_times=num_times,
                    interval=cfg["interval"],
                    task=ReTryIfFail(
                        max_retries=fail_retry,
                        task=HardTask(
                            measure_fn=lambda ctx, update_hook: (
                                ModularProgramV2(
                                    soccfg,
                                    ctx.cfg,
                                    modules=[
                                        Reset(
                                            "reset",
                                            ctx.cfg.get("reset", {"type": "none"}),
                                        ),
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
                            ),
                            raw2signal_fn=lambda raw: raw[0][0],
                            result_shape=(len(ts), 2),
                            dtype=np.float64,
                        ),
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
        plt.close(fig)
        populations = np.asarray(populations)  # (iters, shots, 2)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (iters, ts, populations)

        return iters, ts, populations

    def _run_non_uniform(
        self,
        soc,
        soccfg,
        cfg: T1WithToneOvernightCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        num_times: int = 50,
        fail_retry: int = 3,
    ) -> T1WithToneOvernightResult:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"]["length"]
        del cfg["sweep"]

        iters = np.arange(num_times)

        if isinstance(len_sweep, dict):
            ts = (
                np.linspace(
                    len_sweep["start"] ** (1 / 1.3),
                    len_sweep["stop"] ** (1 / 1.3),
                    len_sweep["expts"],
                )
                ** 1.3
            )
        else:
            ts = np.asarray(len_sweep)
        ts = round_zcu_time(ts, soccfg)
        ts = np.unique(ts)

        def measure_fn(ctx, update_hook):
            rounds = ctx.cfg.pop("rounds", 1)
            ctx.cfg["rounds"] = 1

            acc_populations = np.zeros_like(ts, dtype=np.float64)
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
                        callback=update_hook,
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

            def plot_fn(ctx: TaskContextView) -> None:
                i = ctx.env_dict["repeat_idx"]

                populations = calc_populations(np.asarray(ctx.data))

                viewer.get_plotter("plot_2d_g").update(
                    iters, ts, populations[..., 0], refresh=False
                )
                viewer.get_plotter("plot_2d_e").update(
                    iters, ts, populations[..., 1], refresh=False
                )
                viewer.get_plotter("plot_2d_o").update(
                    iters, ts, populations[..., 2], refresh=False
                )
                viewer.get_plotter("plot_1d").update(
                    ts, populations[i].T, refresh=False
                )

                viewer.refresh()

            populations = run_task(
                task=RepeatOverTime(
                    name="Iteration",
                    num_times=num_times,
                    interval=cfg["interval"],
                    task=ReTryIfFail(
                        max_retries=fail_retry,
                        task=HardTask(
                            measure_fn=measure_fn,
                            raw2signal_fn=lambda raw: raw,
                            result_shape=(len(ts), 2),
                            dtype=np.float64,
                        ),
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
        plt.close(fig)
        populations = np.asarray(populations)  # (iters, shots, 2)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (iters, ts, populations)

        return iters, ts, populations

    def analyze(
        self,
        result: Optional[T1WithToneOvernightResult] = None,
        *,
        ac_coeff: Optional[float] = None,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        iters, Ts, populations = result

        valid_mask = np.all(np.isfinite(populations), axis=(1, 2))
        iters = iters[valid_mask]
        populations = populations[valid_mask]

        populations = calc_populations(populations)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        num_iter = iters.shape[0]

        # (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        transition_rates = np.zeros((num_iter, 6), dtype=np.float64)
        for i, pop in enumerate(tqdm(populations, desc="Fitting transition rates")):
            transition_rates[i] = fit_transition_rates(Ts, pop)[0]

        fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

        def _plot_population(ax, pop, label):
            ax.scatter([], [], s=0, label=label)
            ax.imshow(pop.T, aspect="auto", extent=(iters[0], iters[-1], Ts[-1], Ts[0]))
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

        axs[0][1].plot(iters, R_g, label="Γ_ge + Γ_go", color="blue")
        axs[0][1].plot(iters, R_go, label="Γ_go", color="blue", ls="--")

        axs[1][1].plot(iters, R_e, label="Γ_eg + Γ_eo", color="red")
        axs[1][1].plot(iters, R_eo, label="Γ_eo", color="red", ls="--")

        axs[2][1].plot(iters, R_eo, label="Γ_eo", color="red", ls="--")
        axs[2][1].plot(iters, R_go, label="Γ_go", color="blue", ls="--")

        max_rate = np.nanmax([R_g, R_e])
        for ax in (axs[0][1], axs[1][1], axs[2][1]):
            ax.legend()
            ax.set_ylim(0, max_rate * 1.1)
            ax.grid(True)
            ax.set_ylabel("Rate (μs⁻¹)")
        axs[2][1].set_xlabel("Iteration")

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[T1WithToneOvernightResult] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/t1_overnight",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        iters, Ts, populations = result
        _filepath = Path(filepath)

        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_g_population")),
            x_info={"name": "Iteration", "unit": "a.u.", "values": iters},
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
            x_info={"name": "Iteration", "unit": "a.u.", "values": iters},
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

    def load(self, filepath: List[str], **kwargs) -> T1WithToneOvernightResult:
        g_filepath, e_filepath = filepath

        # Load ground populations
        g_pop, iters_g, Ts = load_data(g_filepath, **kwargs)
        assert iters_g is not None and Ts is not None
        assert len(iters_g.shape) == 1 and len(Ts.shape) == 1
        assert g_pop.shape == (len(iters_g), len(Ts))

        # Load excited populations
        e_pop, iters_e, Ts_e = load_data(e_filepath, **kwargs)
        assert iters_e is not None and Ts_e is not None
        assert e_pop.shape == (len(iters_e), len(Ts_e))
        assert np.array_equal(iters_g, iters_e) and np.array_equal(Ts, Ts_e)

        Ts = Ts * 1e6  # s -> us

        # Reconstruct signals shape: (gains, ts, 2)
        populations = np.stack([g_pop, e_pop], axis=-1)

        iters_g = iters_g.astype(np.int64)
        Ts = Ts.astype(np.float64)
        populations = populations.astype(np.float64)

        self.last_cfg = None
        self.last_result = (iters_g, Ts, populations)

        return iters_g, Ts, populations
