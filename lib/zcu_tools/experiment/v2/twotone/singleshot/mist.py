from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
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

MISTPowerDepResultType = Tuple[NDArray[np.float64], NDArray[np.float64]]


def mist_signal2real(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    g_pops, e_pops = signals[:, 0], signals[:, 1]
    return np.stack([g_pops, e_pops, 1 - g_pops - e_pops], axis=-1)


class MISTPowerDepSingleShotTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class MISTPowerDepSingleShot(AbsExperiment):
    def run(
        self,
        soc,
        soccfg,
        cfg: MISTPowerDepSingleShotTaskConfig,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> MISTPowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted amplitudes

        Pulse.set_param(
            cfg["probe_pulse"], "gain", sweep2param("gain", cfg["sweep"]["gain"])
        )

        with LivePlotter1D(
            "Pulse gain",
            "MIST",
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

            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse(name="init_pulse", cfg=ctx.cfg.get("init_pulse")),
                                Pulse(name="probe_pulse", cfg=ctx.cfg["probe_pulse"]),
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
                    result_shape=(len(pdrs), 2),
                    dtype=np.float64,
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    pdrs, mist_signal2real(ctx.data).T
                ),
            )

        # record the last result
        self.last_cfg = cfg
        self.last_result: MISTPowerDepResultType = (pdrs, signals)

        return pdrs, signals

    def analyze(
        self,
        result: Optional[MISTPowerDepResultType] = None,
        *,
        ac_coeff=None,
        log_scale=False,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result
        signals = signals.real

        populations = mist_signal2real(signals)

        if ac_coeff is None:
            xs = pdrs
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * pdrs**2
            xlabel = r"$\bar n$"

        fig, ax = plt.subplots(figsize=(6, 6))

        plot_kwargs = dict(ls="-", marker="o", markersize=1)
        ax.plot(xs, populations[:, 0], color="blue", label="Ground", **plot_kwargs)
        ax.plot(xs, populations[:, 1], color="red", label="Excited", **plot_kwargs)
        ax.plot(xs, populations[:, 2], color="green", label="Other", **plot_kwargs)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Population", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_ylim(0, 1)
        if log_scale:
            ax.set_xscale("log")

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[MISTPowerDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/mist/pdr_singleshot",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
            y_info={"name": "GE population", "unit": "a.u.", "values": [0, 1]},
            z_info={"name": "Population", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> MISTPowerDepResultType:
        signals, pdrs, _ = load_data(filepath, **kwargs)
        assert pdrs is not None

        signals = signals.T  # transpose back

        self.last_cfg = None
        self.last_result = (pdrs, signals)

        return pdrs, signals
