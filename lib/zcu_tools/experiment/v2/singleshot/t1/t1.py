from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array, make_ge_sweep
from zcu_tools.experiment.v2.runner import (
    HardTask,
    TaskConfig,
    run_task,
    TaskContextView,
)
from zcu_tools.liveplot import LivePlotter1D, MultiLivePlotter, make_plot_frame
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
from zcu_tools.utils.fitting.multi_decay import (
    fit_transition_rates,
    calc_lambda_and_amplitude,
)
from zcu_tools.experiment.v2.utils import round_zcu_time

from ..util import calc_populations
from .util import measure_with_sweep

# (times, signals)
T1Result = Tuple[NDArray[np.float64], NDArray[np.float64]]


class T1Cfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1Exp(AbsExperiment[T1Result, T1Cfg]):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: T1Cfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        unifrom: bool = False,
    ) -> T1Result:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"]["length"]

        if unifrom:
            assert isinstance(len_sweep, dict)
            ts = sweep2array(len_sweep)
            ts = round_zcu_time(ts, soccfg)
            cfg["sweep"] = {"length": len_sweep, "ge": make_ge_sweep()}
        else:
            if isinstance(len_sweep, dict):
                ts = np.geomspace(
                    len_sweep["start"], len_sweep["stop"], len_sweep["expts"]
                )
            else:
                ts = np.asarray(len_sweep)
            ts = round_zcu_time(ts, soccfg)
            ts = np.unique(ts)
            cfg["sweep"] = {"ge": make_ge_sweep()}

        fig, axs = make_plot_frame(2, 1, figsize=(12, 6))
        axs[0][0].set_ylim(0, 1)
        axs[0][1].set_ylim(0, 1)

        with MultiLivePlotter(
            fig,
            dict(
                init_g=LivePlotter1D(
                    "Time (us)",
                    "Amplitude",
                    ax=axs[0][0],
                    segment_kwargs=dict(
                        num_lines=3,
                        line_kwargs=[
                            dict(label="Ground"),
                            dict(label="Excited"),
                            dict(label="Other"),
                        ],
                    ),
                ),
                init_e=LivePlotter1D(
                    "Time (us)",
                    "Amplitude",
                    ax=axs[0][1],
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
                populations = calc_populations(np.asarray(ctx.data))  # (N, 2, 3)

                viewer.get_plotter("init_g").update(
                    ts, populations[:, 0].T, refresh=False
                )
                viewer.get_plotter("init_e").update(
                    ts, populations[:, 1].T, refresh=False
                )

                viewer.refresh()

            def measure_fn(ctx: TaskContextView, update_hook):
                ge_param = sweep2param("ge", cfg["sweep"]["ge"])

                def prog_maker(cfg, t1_delay):
                    cfg = deepcopy(cfg)
                    return ModularProgramV2(
                        soccfg,
                        cfg,
                        modules=[
                            Reset("reset", cfg.get("reset", {"type": "none"})),
                            Pulse(
                                "pi_pulse",
                                Pulse.set_param(cfg["pi_pulse"], "on/off", ge_param),
                            ),
                            Delay("t1_delay", delay=t1_delay),
                            Readout("readout", cfg["readout"]),
                        ],
                    )

                acquire_kwargs = dict(
                    soc=soc,
                    progress=False,
                    callback=update_hook,
                    g_center=g_center,
                    e_center=e_center,
                    population_radius=radius,
                )
                if unifrom:
                    len_param = sweep2param("length", cfg["sweep"]["length"])
                    return prog_maker(ctx.cfg, len_param).acquire(**acquire_kwargs)
                else:
                    return measure_with_sweep(
                        ctx, prog_maker, ts.tolist(), sweep_shape=(2,), **acquire_kwargs
                    )

            populations = run_task(
                task=HardTask(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(ts), 2, 2),
                    dtype=np.float64,
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, populations)

        return ts, populations

    def analyze(
        self,
        result: Optional[T1Result] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, populations = result

        populations = calc_populations(populations)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        rates, _, fit_pops, (pOpt, _) = fit_transition_rates(lens, populations)

        lambdas, _ = calc_lambda_and_amplitude(tuple(pOpt))

        t1 = 1.0 / lambdas[2]
        t1_b = 1.0 / lambdas[1]

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.set_title(f"T_1 = {t1:.1f} μs, T_1_b = {t1_b:.1f} μs")
        ax.plot(lens, fit_pops[:, 0], color="blue", ls="--", label="Ground Fit")
        ax.plot(lens, fit_pops[:, 1], color="red", ls="--", label="Excited Fit")
        ax.plot(lens, fit_pops[:, 2], color="green", ls="--", label="Other Fit")
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
        result: Optional[T1Result] = None,
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

    def load(self, filepath: str, **kwargs) -> T1Result:
        populations, Ts, y_values = load_data(filepath, **kwargs)
        assert Ts is not None and y_values is not None
        assert len(Ts.shape) == 1 and len(y_values.shape) == 1
        assert populations.shape == (len(y_values), len(Ts))

        Ts = Ts * 1e6  # s -> us
        populations = populations.T  # transpose back

        Ts = Ts.astype(np.float64)
        populations = populations.astype(np.float64)

        self.last_cfg = None
        self.last_result = (Ts, populations)

        return Ts, populations
