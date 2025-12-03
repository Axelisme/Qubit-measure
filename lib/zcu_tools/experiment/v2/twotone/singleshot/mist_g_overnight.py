from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import (
    HardTask,
    RepeatOverTime,
    TaskConfig,
    run_task,
)
from zcu_tools.liveplot import (
    LivePlotter2D,
    LivePlotter1D,
    make_plot_frame,
    MultiLivePlotter,
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
from zcu_tools.utils.datasaver import save_data

MISTPowerDepOvernightResultType = Tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.complex128]
]


def calc_populations(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    g_pops, e_pops = signals[..., 0], signals[..., 1]
    return np.stack([g_pops, e_pops, 1 - g_pops - e_pops], axis=-1)


class MISTPowerDepOvernightTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg
    interval: float


class MISTPowerDepOvernight(AbsExperiment):
    def run(
        self,
        soc,
        soccfg,
        cfg: MISTPowerDepOvernightTaskConfig,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        num_times=50,
        fail_retry=3,
    ) -> MISTPowerDepOvernightResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

        iters = np.arange(num_times, dtype=np.int64)
        gains = sweep2array(cfg["sweep"]["gain"])  # predicted amplitudes

        Pulse.set_param(
            cfg["probe_pulse"], "gain", sweep2param("gain", cfg["sweep"]["gain"])
        )

        fig, axs = make_plot_frame(2, 2, figsize=(8, 8))

        with MultiLivePlotter(
            fig,
            dict(
                plot_2d_g=LivePlotter2D(
                    "Iteration", "Readout Gain", existed_axes=[[axs[0][0]]]
                ),
                plot_2d_e=LivePlotter2D(
                    "Iteration", "Readout Gain", existed_axes=[[axs[1][0]]]
                ),
                plot_2d_o=LivePlotter2D(
                    "Iteration", "Readout Gain", existed_axes=[[axs[1][1]]]
                ),
                plot_1d=LivePlotter1D(
                    "Readout Gain",
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

            def plot_fn(ctx) -> None:
                i = ctx.env_dict["repeat_idx"]

                populations = calc_populations(np.asarray(ctx.data))

                viewer.get_plotter("plot_2d_g").update(
                    iters, gains, populations[..., 0], refresh=False
                )
                viewer.get_plotter("plot_2d_e").update(
                    iters, gains, populations[..., 1], refresh=False
                )
                viewer.get_plotter("plot_2d_o").update(
                    iters, gains, populations[..., 2], refresh=False
                )
                viewer.get_plotter("plot_1d").update(
                    gains, populations[i].T, refresh=False
                )

                viewer.refresh()

            signals = run_task(
                task=RepeatOverTime(
                    name="repeat_over_time",
                    num_times=num_times,
                    interval=cfg["interval"],
                    task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset(
                                        "reset", ctx.cfg.get("reset", {"type": "none"})
                                    ),
                                    Pulse(
                                        name="init_pulse", cfg=ctx.cfg.get("init_pulse")
                                    ),
                                    Pulse(
                                        name="probe_pulse", cfg=ctx.cfg["probe_pulse"]
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
                        result_shape=(len(gains), 2),
                        dtype=np.float64,
                    ),
                    fail_retry=fail_retry,
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )
            signals = np.asarray(signals)
        plt.close(fig)

        # record the last result
        self.last_cfg = cfg
        self.last_result = (iters, gains, signals)

        return iters, gains, signals

    def analyze(
        self, result: Optional[MISTPowerDepOvernightResultType] = None, *, ac_coeff=None
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        iters, pdrs, signals = result

        populations = calc_populations(signals)

        if ac_coeff is None:
            xs = pdrs
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * pdrs**2
            xlabel = r"$\bar n$"

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        assert isinstance(fig, Figure)

        axs[0].imshow(
            populations[..., 0],
            aspect="auto",
            origin="lower",
            extent=(xs[0], xs[-1], iters[0], iters[-1]),
        )
        axs[0].set_ylabel("Iteration")
        axs[0].set_title("Ground State Population")
        fig.colorbar(axs[0].images[0], ax=axs[0], label="Population")

        axs[1].imshow(
            populations[..., 1],
            aspect="auto",
            origin="lower",
            extent=(xs[0], xs[-1], iters[0], iters[-1]),
        )
        axs[1].set_ylabel("Iteration")
        axs[1].set_title("Excited State Population")
        fig.colorbar(axs[1].images[0], ax=axs[1], label="Population")

        axs[2].imshow(
            populations[..., 2],
            aspect="auto",
            origin="lower",
            extent=(xs[0], xs[-1], iters[0], iters[-1]),
        )
        axs[2].set_xlabel(xlabel)
        axs[2].set_ylabel("Iteration")
        axs[2].set_title("Other State Population")
        fig.colorbar(axs[2].images[0], ax=axs[2], label="Population")

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[MISTPowerDepOvernightResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/mist/pdr_overnight",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        iters, pdrs, overnight_signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Iteration", "unit": "None", "values": iters},
            z_info={
                "name": "Ground Population",
                "unit": "a.u.",
                "values": overnight_signals[..., 0],
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )
        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
            y_info={"name": "Iteration", "unit": "None", "values": iters},
            z_info={
                "name": "Excited Population",
                "unit": "a.u.",
                "values": overnight_signals[..., 1],
            },
            comment=comment,
            tag=tag,
            **kwargs,
        )
