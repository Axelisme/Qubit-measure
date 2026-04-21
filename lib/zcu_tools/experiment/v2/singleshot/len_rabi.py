from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Optional, TypeAlias

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import Pulse, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data

from .util import calc_populations

# (lens, signals)
LenRabiResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class LenRabiCfg(TwoToneCfg, TaskCfg):
    sweep: dict[str, SweepCfg]


class LenRabiExp(AbsExperiment[LenRabiResult, LenRabiCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> LenRabiResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        _cfg = check_type(deepcopy(cfg), LenRabiCfg)  # avoid in-place modification
        setup_devices(_cfg, progress=True)
        modules = _cfg["modules"]

        assert modules["qub_pulse"].waveform.style in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )

        lengths = sweep2array(
            _cfg["sweep"]["length"],
            "time",
            {"soccfg": soccfg, "gen_ch": modules["qub_pulse"].ch},
        )

        length_param = sweep2param("length", _cfg["sweep"]["length"])
        modules["qub_pulse"].set_param("length", length_param)

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

            def measure_fn(ctx, update_hook):
                return TwoToneProgram(
                    soccfg,
                    ctx.cfg,
                    sweep=[("length", ctx.cfg["sweep"]["length"])],
                ).acquire(
                    soc,
                    progress=False,
                    round_hook=update_hook,
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
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, calc_populations(ctx.root_data).T
                ),
            )

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (lengths, populations)

        return lengths, populations

    def analyze(
        self,
        result: Optional[LenRabiResult] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, populations = result

        populations = calc_populations(populations)  # (len, geo)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        plot_kwargs = dict(ls="-", marker="o", markersize=3)
        ax.plot(
            lens, populations[:, 0], color="blue", label="$|0\\rangle$", **plot_kwargs
        )  # type: ignore
        ax.plot(
            lens, populations[:, 1], color="red", label="$|1\\rangle$", **plot_kwargs
        )  # type: ignore
        ax.plot(
            lens, populations[:, 2], color="green", label="$|L\\rangle$", **plot_kwargs
        )  # type: ignore
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

        lens, populations = result
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
        populations, lens, _ = load_data(filepath, **kwargs)
        assert lens is not None

        lens = lens.astype(np.float64)
        populations = np.real(populations).astype(np.float64)

        lens = lens * 1e6  # s -> us

        self.last_cfg = None
        self.last_result = (lens, populations)

        return lens, populations
