from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Optional

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import SweepCfg, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.program.v2.twotone import TwoToneModuleCfg
from zcu_tools.utils.datasaver import load_data, save_data

from .util import calc_populations


@dataclass(frozen=True)
class LenRabiResult:
    lengths: NDArray[np.float64]
    signals: NDArray[np.float64]
    cfg_snapshot: Optional[LenRabiCfg] = None


class LenRabiSweepCfg(ConfigBase):
    length: SweepCfg


class LenRabiCfg(TwoToneCfg, ExpCfgModel):
    modules: TwoToneModuleCfg
    sweep: LenRabiSweepCfg


class LenRabiExp(AbsExperiment[LenRabiResult, LenRabiCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: LenRabiCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> LenRabiResult:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        assert modules.qub_pulse.waveform.style in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )

        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        length_param = sweep2param("length", cfg.sweep.length)
        modules.qub_pulse.set_param("length", length_param)

        with LivePlot1D(
            "Length (us)",
            "Signal",
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

            def measure_fn(
                ctx: TaskState[NDArray[np.float64], Any, LenRabiCfg], update_hook
            ):
                return TwoToneProgram(
                    soccfg,
                    ctx.cfg,
                    sweep=[("length", ctx.cfg.sweep.length)],
                ).acquire(
                    soc,
                    progress=False,
                    round_hook=update_hook,
                    stop_checkers=[ctx.is_stop],
                    g_center=g_center,
                    e_center=e_center,
                    ge_radius=radius,
                )

            populations = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(lengths), 2),
                    dtype=np.float64,
                    pbar_n=1,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, calc_populations(ctx.root_data).T
                ),
            )

        # record last cfg and result
        self.last_result = LenRabiResult(
            lengths=lengths, signals=populations, cfg_snapshot=cfg
        )

        return self.last_result

    def analyze(
        self,
        result: Optional[LenRabiResult] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, populations = result.lengths, result.signals

        populations = calc_populations(populations)  # (len, geo)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        plot_kwargs: dict[str, Any] = dict(ls="-", marker="o", markersize=3)
        ax.plot(
            lens, populations[:, 0], color="blue", label="$|0\\rangle$", **plot_kwargs
        )
        ax.plot(
            lens, populations[:, 1], color="red", label="$|1\\rangle$", **plot_kwargs
        )
        ax.plot(
            lens, populations[:, 2], color="green", label="$|L\\rangle$", **plot_kwargs
        )
        ax.set_xlabel("Pulse length (μs)")
        ax.set_ylabel("Population (a.u.)")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[LenRabiResult] = None,
        comment: Optional[str] = None,
        tag: str = "singleshot/ge/rabi_length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, populations = result.lengths, result.signals
        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("result.cfg_snapshot is None")
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            y_info={"name": "GE population", "unit": "a.u.", "values": [0, 1]},
            z_info={"name": "Population", "unit": "a.u.", "values": populations.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> LenRabiResult:
        populations, lens, _, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert lens is not None

        lens = lens.astype(np.float64)
        populations = np.real(populations).astype(np.float64)

        lens = lens * 1e6  # s -> us

        cfg_snapshot = None
        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                cfg_snapshot = LenRabiCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = LenRabiResult(
            lengths=lens, signals=populations, cfg_snapshot=cfg_snapshot
        )

        return self.last_result
