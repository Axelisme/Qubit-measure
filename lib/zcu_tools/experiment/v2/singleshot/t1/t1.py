from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, TypeAlias, Union

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D, MultiLivePlot, make_plot_frame
from zcu_tools.program.v2 import (
    Branch,
    Delay,
    DelayAuto,
    LoadValue,
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
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting.multi_decay import calc_lambdas, fit_dual_transition_rates

from ..util import calc_populations

# (times, signals)
T1Result: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class T1ModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1SweepCfg(ConfigBase):
    length: Union[SweepCfg, list[float]]


class T1Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1ModuleCfg
    sweep: T1SweepCfg


class T1Exp(AbsExperiment[T1Result, T1Cfg]):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    def _make_viewer_ctx(self):
        fig, axs = make_plot_frame(1, 2, plot_instant=True, figsize=(12, 5))
        axs[0][0].set_ylim(0, 1)
        axs[0][1].set_ylim(0, 1)

        line_kwargs = [
            dict(label="Ground"),
            dict(label="Excited"),
            dict(label="Other"),
        ]
        viewer = MultiLivePlot(
            fig,
            dict(
                init_g=LivePlot1D(
                    "Time (us)",
                    "Amplitude",
                    existed_axes=[[axs[0][0]]],
                    segment_kwargs=dict(num_lines=3, line_kwargs=line_kwargs),
                ),
                init_e=LivePlot1D(
                    "Time (us)",
                    "Amplitude",
                    existed_axes=[[axs[0][1]]],
                    segment_kwargs=dict(num_lines=3, line_kwargs=line_kwargs),
                ),
            ),
        )
        return fig, viewer

    def _run_uniform(
        self,
        soc,
        soccfg,
        cfg: T1Cfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1Result:
        setup_devices(cfg, progress=True)

        length_sweep = cfg.sweep.length
        assert isinstance(length_sweep, SweepCfg), "uniform mode requires SweepCfg"
        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})

        fig, viewer = self._make_viewer_ctx()

        with viewer:

            def measure_fn(
                ctx: TaskState[NDArray[np.float64], Any, T1Cfg],
                update_hook: Optional[Callable],
            ):
                modules = ctx.cfg.modules
                inner_length_sweep = ctx.cfg.sweep.length
                assert isinstance(inner_length_sweep, SweepCfg), (
                    "uniform mode requires SweepCfg"
                )
                length_param = sweep2param("length", inner_length_sweep)
                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                        Delay("t1_delay", delay=length_param),
                        Readout("readout", modules.readout),
                    ],
                    sweep=[("length", inner_length_sweep), ("ge", 2)],
                ).acquire(
                    soc,
                    progress=False,
                    round_hook=update_hook,
                    g_center=g_center,
                    e_center=e_center,
                    ge_radius=radius,
                )

            def plot_fn(ctx: TaskState) -> None:
                populations = calc_populations(np.asarray(ctx.root_data))  # (N, 2, 3)
                viewer.get_plotter("init_g").update(
                    lengths, populations[:, 0].T, refresh=False
                )
                viewer.get_plotter("init_e").update(
                    lengths, populations[:, 1].T, refresh=False
                )
                viewer.refresh()

            populations = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(lengths), 2, 2),
                    dtype=np.float64,
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=plot_fn,
            )
        plt.close(fig)

        self.last_cfg = deepcopy(cfg)
        self.last_result = (lengths, populations)

        return lengths, populations

    def _run_non_uniform(
        self,
        soc,
        soccfg,
        cfg: T1Cfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1Result:
        setup_devices(cfg, progress=True)

        length_sweep = cfg.sweep.length

        if isinstance(length_sweep, SweepCfg):
            lengths = np.geomspace(
                length_sweep.start, length_sweep.stop, length_sweep.expts
            )
        else:
            lengths = np.asarray(length_sweep)
        length_cycles = np.asarray(
            [int(soccfg.us2cycles(t)) for t in lengths], dtype=np.int64
        )
        length_cycles = np.unique(length_cycles)
        lengths = np.asarray(
            [soccfg.cycles2us(int(cycle)) for cycle in length_cycles], dtype=np.float64
        )

        fig, viewer = self._make_viewer_ctx()

        with viewer:

            def measure_fn(
                ctx: TaskState[NDArray[np.float64], Any, T1Cfg],
                update_hook: Optional[Callable],
            ):
                modules = ctx.cfg.modules
                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                        LoadValue(
                            "load_t1_delay",
                            values=list(length_cycles),
                            idx_reg="length_idx",
                            val_reg="t1_delay_cycle",
                        ),
                        DelayAuto("t1_delay", t="t1_delay_cycle"),
                        Readout("readout", modules.readout),
                    ],
                    sweep=[("length_idx", len(length_cycles)), ("ge", 2)],
                ).acquire(
                    soc,
                    progress=False,
                    round_hook=update_hook,
                    g_center=g_center,
                    e_center=e_center,
                    ge_radius=radius,
                )

            def plot_fn(ctx: TaskState) -> None:
                populations = calc_populations(np.asarray(ctx.root_data))  # (N, 2, 3)
                viewer.get_plotter("init_g").update(
                    lengths, populations[:, 0].T, refresh=False
                )
                viewer.get_plotter("init_e").update(
                    lengths, populations[:, 1].T, refresh=False
                )
                viewer.refresh()

            populations = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(lengths), 2, 2),
                    dtype=np.float64,
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=plot_fn,
            )
        plt.close(fig)

        self.last_cfg = deepcopy(cfg)
        self.last_result = (lengths, populations)

        return lengths, populations

    def run(
        self,
        soc,
        soccfg,
        cfg: T1Cfg,
        g_center: complex,
        e_center: complex,
        radius: float,
        *,
        uniform: bool = False,
    ) -> T1Result:
        if uniform:
            return self._run_uniform(soc, soccfg, cfg, g_center, e_center, radius)
        else:
            return self._run_non_uniform(soc, soccfg, cfg, g_center, e_center, radius)

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
        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

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
        g_pop, g_Ts, _, comment = load_data(g_filepath, return_comment=True, **kwargs)
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

        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                self.last_cfg = T1Cfg.validate_or_warn(cfg, source=g_filepath)
        self.last_result = (Ts, populations)

        return Ts, populations
