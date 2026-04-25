from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D, MultiLivePlot, make_plot_frame
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
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
T1WithToneResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class T1WithToneModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
    pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweepCfg(ConfigBase):
    length: SweepCfg


class T1WithToneCfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1WithToneModuleCfg
    sweep: T1WithToneSweepCfg


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
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        length_sweep = cfg.sweep.length

        if uniform:
            assert isinstance(length_sweep, dict)
            lengths = sweep2array(
                length_sweep,
                "time",
                {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
            )
        else:
            if isinstance(length_sweep, dict):
                lengths = np.geomspace(
                    length_sweep["start"],
                    length_sweep["stop"],
                    length_sweep["expts"],
                    dtype=np.float64,
                )
            else:
                lengths = np.asarray(length_sweep, dtype=np.float64)
            lengths = sweep2array(
                lengths,
                "time",
                {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
                allow_array=True,
            )
            lengths = np.unique(lengths)

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, T1WithToneCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg

            def prog_maker(cfg, length_param) -> ModularProgramV2:
                _cfg = deepcopy(cfg)
                modules = _cfg.modules

                modules.probe_pulse.set_param("length", length_param)

                return ModularProgramV2(
                    soccfg,
                    _cfg,
                    modules=[
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        Branch(
                            "ge",
                            Pulse("probe_pulse_g", modules.probe_pulse),
                            [
                                Pulse("pi_pulse", modules.pi_pulse),
                                Pulse("probe_pulse", modules.probe_pulse),
                            ],
                        ),
                        Readout("readout", modules.readout),
                    ],
                    sweep=(
                        [("length", _cfg.sweep.length), ("ge", 2)]
                        if uniform
                        else [("ge", 2)]
                    ),
                )

            acquire_kwargs = dict(
                soc=soc,
                progress=False,
                round_hook=update_hook,
                g_center=g_center,
                e_center=e_center,
                ge_radius=radius,
            )
            if uniform:
                len_param = sweep2param("length", ctx.cfg.sweep.length)
                return prog_maker(cfg, len_param).acquire(**acquire_kwargs)
            else:
                return measure_with_sweep(
                    ctx,
                    prog_maker,
                    lengths.tolist(),
                    sweep_shape=(2,),
                    **acquire_kwargs,
                )

        fig, axs = make_plot_frame(1, 2, plot_instant=True, figsize=(12, 5))
        axs[0][0].set_ylim(0, 1)
        axs[0][1].set_ylim(0, 1)

        with MultiLivePlot(
            fig,
            dict(
                init_g=LivePlot1D(
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
                init_e=LivePlot1D(
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

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (lengths, populations)

        return lengths, populations

    def analyze(
        self,
        result: Optional[T1WithToneResult] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
        skip: int = 0,
    ) -> tuple[float, float, Figure]:
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

        return t1, t1_b, fig

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

    def load(self, filepath: list[str], **kwargs) -> T1WithToneResult:
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

                self.last_cfg = T1WithToneCfg.validate_or_warn(cfg, source=g_filepath)
        self.last_result = (Ts, populations)

        return Ts, populations
