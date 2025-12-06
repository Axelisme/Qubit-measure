from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skopt import Optimizer
from skopt.space import Real
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import make_ge_sweep
from zcu_tools.experiment.v2.runner import (
    HardTask,
    SoftTask,
    TaskConfig,
    TaskContext,
    run_task,
)
from zcu_tools.experiment.v2.utils import snr_as_signal
from zcu_tools.liveplot import LivePlotterScatter, MultiLivePlotter, instant_plot
from zcu_tools.program import SweepCfg
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
from zcu_tools.utils.datasaver import save_data

AutoOptResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


class ReadoutOptimizer:
    def __init__(
        self,
        fpt_sweep: SweepCfg,
        pdr_sweep: SweepCfg,
        len_sweep: SweepCfg,
        num_points: int,
    ) -> None:
        self.num_points = num_points

        self.optimizer = Optimizer(
            dimensions=[
                Real(name="freq", low=fpt_sweep.min, high=fpt_sweep.max),
                Real(name="gain", low=pdr_sweep.min, high=pdr_sweep.max),
                Real(name="length", low=len_sweep.min, high=len_sweep.max),
            ],
            n_initial_points=num_points // 3,
            base_estimator="ET",
            acq_func="EI",
            n_jobs=-1,
            acq_optimizer="auto",
        )
        self.last_param = None

    def next_params(
        self, i: int, last_snr: Optional[float]
    ) -> Optional[Tuple[float, float, float]]:
        if i >= self.num_points:
            return None

        if last_snr is not None:
            self.optimizer.tell(self.last_param, last_snr)

        self.last_param = self.optimizer.ask()
        return self.last_param


class AutoOptimizeTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class AutoOptimizeExperiment(AbsExperiment):
    def run(
        self, soc, soccfg, cfg: AutoOptimizeTaskConfig, num_points: int
    ) -> AutoOptResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        fpt_sweep = cfg["sweep"]["freq"]
        pdr_sweep = cfg["sweep"]["gain"]
        len_sweep = cfg["sweep"]["length"]
        cfg["sweep"] = {"ge": make_ge_sweep()}

        Pulse.set_param(
            cfg["qub_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        optimizer = ReadoutOptimizer(fpt_sweep, pdr_sweep, len_sweep, num_points)

        # (num_points, [freq, gain, length])
        params = np.full((num_points, 3), np.nan, dtype=np.float64)

        def update_fn(i: int, ctx: TaskContext, _) -> None:
            ctx.env_dict["index"] = i

            last_snr = None
            if i > 0:
                last_snr = np.abs(ctx.data[i - 1])
            cur_params = optimizer.next_params(i, last_snr)

            if cur_params is None:
                # TODO: Better way to early stop
                raise KeyboardInterrupt("No more parameters to optimize.")

            Readout.set_param(ctx.cfg["readout"], "freq", cur_params[0])
            Readout.set_param(ctx.cfg["readout"], "gain", cur_params[1])
            Readout.set_param(ctx.cfg["readout"], "length", cur_params[2])

        # initialize figure and axes
        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle("Readout Auto Optimization")

        ax_iter = fig.add_subplot(gs[:, 0])
        ax_freq = fig.add_subplot(gs[0, 1])
        ax_gain = fig.add_subplot(gs[1, 1])
        ax_len = fig.add_subplot(gs[2, 1])

        instant_plot(fig)  # show the figure immediately

        with MultiLivePlotter(
            fig,
            plotters=dict(
                iter_scatter=LivePlotterScatter(
                    "Iteration", "SNR (a.u.)", existed_axes=[[ax_iter]]
                ),
                freq_scatter=LivePlotterScatter(
                    "Frequency (MHz)", "SNR (a.u.)", existed_axes=[[ax_freq]]
                ),
                gain_scatter=LivePlotterScatter(
                    "Readout Gain (a.u.)", "SNR (a.u.)", existed_axes=[[ax_gain]]
                ),
                len_scatter=LivePlotterScatter(
                    "Readout Length (us)", "SNR (a.u.)", existed_axes=[[ax_len]]
                ),
            ),
        ) as viewer:

            def plot_fn(ctx: TaskContext) -> None:
                idx: int = ctx.env_dict["index"]
                snrs = np.abs(ctx.data)  # (num_points, )

                cur_freq, cur_gain, cur_len = params[idx, :]

                fig.suptitle(
                    f"Iteration {idx}, Frequency: {1e-3 * cur_freq:.4g} (GHz), Gain: {cur_gain:.2g} (a.u.), Length: {cur_len:.2g} (us)"
                )

                viewer.get_plotter("iter_scatter").update(
                    np.arange(num_points), snrs, refresh=False
                )
                viewer.get_plotter("freq_scatter").update(
                    params[:, 0], snrs, refresh=False
                )
                viewer.get_plotter("freq_scatter").update(
                    params[:, 1], snrs, refresh=False
                )
                viewer.get_plotter("len_scatter").update(
                    params[:, 2], snrs, refresh=False
                )
                viewer.refresh()

            results = run_task(
                task=SoftTask(
                    sweep_name="Iteration",
                    sweep_values=list(range(num_points)),
                    update_cfg_fn=update_fn,
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
                                        Pulse("qub_pulse", ctx.cfg["qub_pulse"]),
                                        Readout("readout", ctx.cfg["readout"]),
                                    ],
                                )
                            )
                            and (
                                prog.acquire(
                                    soc,
                                    progress=False,
                                    callback=update_hook,
                                    record_statistic=True,
                                ),
                                prog.get_covariance(),
                                prog.get_median(),
                            )
                        ),
                        raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    ),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
            signals = np.asarray(results)

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (params, signals)

        return params, signals

    def analyze(
        self, result: Optional[AutoOptResultType] = None
    ) -> Tuple[float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        params, signals = result
        snrs = np.abs(signals)

        max_id = np.nanargmax(snrs)
        max_snr = float(snrs[max_id])
        best_params = params[max_id, :]

        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle("Readout Auto Optimization")

        ax_iter = fig.add_subplot(gs[:, 0])
        ax_freq = fig.add_subplot(gs[0, 1])
        ax_gain = fig.add_subplot(gs[1, 1])
        ax_len = fig.add_subplot(gs[2, 1])

        ax_iter.scatter(np.arange(len(snrs)), snrs, s=1)
        ax_iter.axhline(max_snr, color="r", ls="--", label=f"best = {max_snr:.2g}")
        ax_iter.scatter([max_id], [max_snr], color="r", marker="*")
        ax_iter.set_xlabel("Iteration")
        ax_iter.set_ylabel("SNR")
        ax_iter.legend()
        ax_iter.grid(True)

        def plot_ax(ax, param_idx, label_name) -> None:
            ax.scatter(params[:, param_idx], snrs, s=1)
            best_value = best_params[param_idx]
            ax.axvline(best_value, color="r", ls="--", label=f"best = {best_value:.2g}")
            ax.scatter([best_value], [max_snr], color="r", marker="*")
            ax.set_xlabel(label_name)
            ax.set_ylabel("SNR")
            ax.legend()
            ax.grid(True)

        plot_ax(ax_freq, 0, "Frequency (MHz)")
        plot_ax(ax_gain, 1, "Readout Gain (a.u.)")
        plot_ax(ax_len, 2, "Readout Length (us)")

        return float(best_params[0]), float(best_params[1]), float(best_params[2]), fig

    def save(
        self,
        filepath: str,
        result: Optional[AutoOptResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/auto",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        params, signals = result

        filepath = Path(filepath)

        x_info = {
            "name": "Iteration",
            "unit": "a.u.",
            "values": np.arange(params.shape[0]),
        }

        save_data(
            filepath=str(filepath.with_name(filepath.name + "_params")),
            x_info=x_info,
            y_info={"name": "Parameter Type", "unit": "a.u.", "values": [0, 1, 2]},
            z_info={"name": "Parameters", "unit": "a.u.", "values": params.T},
            comment=comment,
            tag=tag + "/params",
            **kwargs,
        )

        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
            x_info=x_info,
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag + "/signals",
            **kwargs,
        )
