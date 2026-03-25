from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import make_ge_sweep, sweep2array
from zcu_tools.liveplot import LivePlotter1D, MultiLivePlotter, make_plot_frame
from zcu_tools.program import SweepCfg
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
from zcu_tools.utils.fitting.multi_decay import calc_lambdas, fit_dual_transition_rates

from ..util import calc_populations
from .util import measure_with_sweep

# (times, signals)
T1Result: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class T1ModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1Cfg(ModularProgramCfg, TaskCfg):
    modules: T1ModuleCfg
    sweep: dict[str, SweepCfg]


class T1Exp(AbsExperiment[T1Result, T1Cfg]):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        g_center: complex,
        e_center: complex,
        radius: float,
        uniform: bool = False,
    ) -> T1Result:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        _cfg = check_type(deepcopy(cfg), T1Cfg)

        length_sweep = _cfg["sweep"]["length"]

        if uniform:
            assert isinstance(length_sweep, dict)
            lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
            _cfg["sweep"] = {"length": length_sweep, "ge": make_ge_sweep()}
        else:
            if isinstance(length_sweep, dict):
                lengths = np.geomspace(
                    length_sweep["start"], length_sweep["stop"], length_sweep["expts"]
                )
            else:
                lengths = np.asarray(length_sweep)
            lengths = sweep2array(lengths, "time", {"soccfg": soccfg}, allow_array=True)
            lengths = np.unique(lengths)
            _cfg["sweep"] = {"ge": make_ge_sweep()}

        fig, axs = make_plot_frame(1, 2, figsize=(12, 5))
        axs[0][0].set_ylim(0, 1)
        axs[0][1].set_ylim(0, 1)

        with MultiLivePlotter(
            fig,
            dict(
                init_g=LivePlotter1D(
                    "Time (us)",
                    "Amplitude",
                    existed_axes=[[axs[0][0]]],
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

            def plot_fn(ctx: TaskState) -> None:
                populations = calc_populations(np.asarray(ctx.root_data))  # (N, 2, 3)

                viewer.get_plotter("init_g").update(
                    lengths, populations[:, 0].T, refresh=False
                )
                viewer.get_plotter("init_e").update(
                    lengths, populations[:, 1].T, refresh=False
                )

                viewer.refresh()

            def measure_fn(ctx: TaskState, update_hook):
                ge_param = sweep2param("ge", _cfg["sweep"]["ge"])

                def prog_maker(cfg, t1_delay):
                    cfg = deepcopy(cfg)
                    modules = cfg["modules"]
                    return ModularProgramV2(
                        soccfg,
                        cfg,
                        modules=[
                            Reset("reset", modules.get("reset")),
                            Pulse(
                                "pi_pulse",
                                Pulse.set_param(
                                    modules["pi_pulse"], "on/off", ge_param
                                ),
                            ),
                            Delay("t1_delay", delay=t1_delay),
                            Readout("readout", modules["readout"]),
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
                if uniform:
                    len_param = sweep2param("length", _cfg["sweep"]["length"])
                    return prog_maker(ctx.cfg, len_param).acquire(**acquire_kwargs)
                else:
                    return measure_with_sweep(
                        ctx,
                        prog_maker,
                        lengths.tolist(),
                        sweep_shape=(2,),
                        **acquire_kwargs,
                    )

            populations = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(lengths), 2, 2),
                    dtype=np.float64,
                ),
                init_cfg=_cfg,
                on_update=plot_fn,
            )
        plt.close(fig)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (lengths, populations)

        return lengths, populations

    def analyze(
        self,
        result: Optional[T1Result] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
        skip: int = 0,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, populations = result

        lens = lens[skip:]
        populations = populations[skip:]

        populations = calc_populations(populations)  # (N, 2, 3)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        populations1 = populations[:, 0]  # init in g
        populations2 = populations[:, 1]  # init in e

        rate, _, fit_pops1, fit_pops2, *_ = fit_dual_transition_rates(
            lens, populations1, populations2
        )

        lambdas, _ = calc_lambdas(rate)

        t1 = 1.0 / lambdas[2]
        t1_b = 1.0 / lambdas[1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        fig.suptitle(f"T_1 = {t1:.1f} μs, T_1_b = {t1_b:.1f} μs")
        plot_kwargs = dict(ls="-", marker=".", markersize=3)

        ax1.plot(lens, fit_pops1[:, 0], color="blue", ls="--", label="Ground Fit")
        ax1.plot(lens, fit_pops1[:, 1], color="red", ls="--", label="Excited Fit")
        ax1.plot(lens, fit_pops1[:, 2], color="green", ls="--", label="Other Fit")
        ax1.plot(lens, populations1[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
        ax1.plot(lens, populations1[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
        ax1.plot(lens, populations1[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore
        ax1.set_xlabel("Time (μs)")
        ax1.legend(loc=4)
        ax1.grid(True)

        ax2.plot(lens, fit_pops2[:, 0], color="blue", ls="--", label="Ground Fit")
        ax2.plot(lens, fit_pops2[:, 1], color="red", ls="--", label="Excited Fit")
        ax2.plot(lens, fit_pops2[:, 2], color="green", ls="--", label="Other Fit")
        ax2.plot(lens, populations2[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
        ax2.plot(lens, populations2[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
        ax2.plot(lens, populations2[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore
        ax2.set_xlabel("Time (μs)")
        ax2.set_ylabel("Population")
        ax2.legend(loc=4)
        ax2.grid(True)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[T1Result] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/t1",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, populations = result

        populations1 = populations[:, 0]  # init in g
        populations2 = populations[:, 1]  # init in e

        _filepath = Path(filepath)

        # initial in g
        save_data(
            filepath=str(_filepath.with_name(_filepath.stem + "_initg")),
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            y_info={"name": "GE population", "unit": "a.u.", "values": [0, 1]},
            z_info={"name": "Signal", "unit": "a.u.", "values": populations1.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

        # initial in e
        save_data(
            filepath=str(_filepath.with_name(_filepath.stem + "_inite")),
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            y_info={"name": "GE population", "unit": "a.u.", "values": [0, 1]},
            z_info={"name": "Signal", "unit": "a.u.", "values": populations2.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: list[str], **kwargs) -> T1Result:
        g_filepath, e_filepath = filepath

        # Load ground populations
        g_pop, g_Ts, _ = load_data(g_filepath, **kwargs)
        assert g_pop.shape == (len(g_Ts), 2)

        # Load excited populations
        e_pop, e_Ts, _ = load_data(e_filepath, **kwargs)
        assert e_pop.shape == (len(e_Ts), 2)

        assert np.allclose(g_Ts, e_Ts), "Time arrays do not match"

        Ts = g_Ts * 1e6  # s -> us

        # Reconstruct signals shape: (Ts, 2, 2)
        populations = np.stack([g_pop, e_pop], axis=1)

        Ts = Ts.astype(np.float64)
        populations = np.real(populations).astype(np.float64)

        self.last_cfg = None
        self.last_result = (Ts, populations)

        return Ts, populations
