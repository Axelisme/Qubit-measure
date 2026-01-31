from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import List, NotRequired, Optional, Tuple

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import (
    HardTask,
    TaskConfig,
    TaskContextView,
    run_task,
)
from zcu_tools.experiment.v2.utils import round_zcu_time
from zcu_tools.liveplot import LivePlotter1D, MultiLivePlotter, make_plot_frame
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
from zcu_tools.utils.fitting.multi_decay import (
    calc_lambdas,
    fit_dual_transition_rates,
    fit_dual_with_vadality,
)

from ..util import calc_populations
from .util import measure_with_sweep

# (times, signals)
T1WithToneResult = Tuple[NDArray[np.float64], NDArray[np.float64]]


class T1WithToneCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneExp(AbsExperiment[T1WithToneResult, T1WithToneCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: T1WithToneCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        uniform: bool = False,
    ) -> T1WithToneResult:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"]["length"]

        if uniform:
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

                def prog_maker(cfg, t1_param):
                    cfg = deepcopy(cfg)
                    return ModularProgramV2(
                        soccfg,
                        cfg,
                        modules=[
                            Reset("reset", cfg.get("reset", {"type": "none"})),
                            Pulse("init_pulse", cfg.get("init_pulse")),
                            Pulse(
                                "pi_pulse",
                                Pulse.set_param(cfg["pi_pulse"], "on/off", ge_param),
                            ),
                            Pulse(
                                "probe_pulse",
                                Pulse.set_param(cfg["probe_pulse"], "length", t1_param),
                            ),
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
                if uniform:
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
        plt.close(fig)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, populations)

        return ts, populations

    def analyze(
        self,
        result: Optional[T1WithToneResult] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, populations = result

        lens = lens[1:]
        populations = populations[1:]

        populations = calc_populations(populations)  # (N, 2, 3)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        populations1 = populations[:, 0]  # init in g
        populations2 = populations[:, 1]  # init in e

        # fit_dual_with_vadality(lens, populations1, populations2)

        rate, _, fit_pops1, fit_pops2, *_ = fit_dual_transition_rates(
            lens, populations1, populations2
        )

        lambdas, _ = calc_lambdas(rate)

        t1 = 1.0 / lambdas[2]
        t1_b = 1.0 / lambdas[1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

        fig.suptitle(f"T_1 = {t1:.1f} μs, T_1_b = {t1_b:.1f} μs")

        ax1.plot(lens, fit_pops1[:, 0], color="blue", ls="--", label="Ground Fit")
        ax1.plot(lens, fit_pops1[:, 1], color="red", ls="--", label="Excited Fit")
        ax1.plot(lens, fit_pops1[:, 2], color="green", ls="--", label="Other Fit")
        ax1.scatter(lens, populations1[:, 0], color="blue", label="Ground", s=1)
        ax1.scatter(lens, populations1[:, 1], color="red", label="Excited", s=1)
        ax1.scatter(lens, populations1[:, 2], color="green", label="Other", s=1)
        ax1.set_ylabel("Population")
        ax1.legend(loc=4)
        ax1.set_ylim(0, 1)
        ax1.grid(True)

        ax2.plot(lens, fit_pops2[:, 0], color="blue", ls="--", label="Ground Fit")
        ax2.plot(lens, fit_pops2[:, 1], color="red", ls="--", label="Excited Fit")
        ax2.plot(lens, fit_pops2[:, 2], color="green", ls="--", label="Other Fit")
        ax2.scatter(lens, populations2[:, 0], color="blue", label="Ground", s=1)
        ax2.scatter(lens, populations2[:, 1], color="red", label="Excited", s=1)
        ax2.scatter(lens, populations2[:, 2], color="green", label="Other", s=1)
        ax2.set_xlabel("Time (μs)")
        ax2.set_ylabel("Population")
        ax2.legend(loc=4)
        ax2.set_ylim(0, 1)
        ax2.grid(True)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[T1WithToneResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/t1/t1_with_tone",
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

    def load(self, filepath: List[str], **kwargs) -> T1WithToneResult:
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
