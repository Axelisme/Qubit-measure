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
    LivePlotter1D,
    LivePlotter2DwithLine,
    MultiLivePlotter,
    make_plot_frame,
)
from zcu_tools.notebook.utils import make_sweep
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

MISTPowerDepResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * len(signals)), 1)

    mist_signals = signals - np.mean(signals[:avg_len])

    return np.abs(mist_signals)


class MISTPowerDepTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class MISTPowerDepExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: MISTPowerDepTaskConfig) -> MISTPowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted amplitudes

        Pulse.set_param(
            cfg["probe_pulse"], "gain", sweep2param("gain", cfg["sweep"]["gain"])
        )

        with LivePlotter1D("Pulse gain", "MIST") as viewer:
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
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(pdrs),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(pdrs, mist_signal2real(ctx.data)),
            )

        # record the last result
        self.last_cfg = cfg
        self.last_result = (pdrs, signals)

        return pdrs, signals

    def analyze(
        self,
        result: Optional[MISTPowerDepResultType] = None,
        *,
        g0=None,
        e0=None,
        ac_coeff=None,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        if g0 is None:
            g0 = signals[0]

        amp_diff = np.abs(signals - g0)

        if ac_coeff is None:
            xs = pdrs
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * pdrs**2
            xlabel = r"$\bar n$"

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.plot(xs, amp_diff.T, marker=".")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

    def save(
        self,
        filepath: str,
        result: Optional[MISTPowerDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/mist/pdr",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )


MISTPowerDepOvernightResultType = Tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.complex128]
]


def mist_overnight_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)

    return np.abs(mist_signals)


class MISTPowerDepOvernightTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    init_pulse: PulseCfg
    probe_pulse: PulseCfg
    readout: ReadoutCfg
    interval: float


class MISTPowerDepOvernightExperiment(AbsExperiment):
    def run(
        self,
        soc,
        soccfg,
        cfg: MISTPowerDepOvernightTaskConfig,
        *,
        num_times=50,
        fail_retry=3,
    ) -> MISTPowerDepOvernightResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

        iters = np.arange(num_times, dtype=np.int64)
        pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted amplitudes

        Pulse.set_param(
            cfg["probe_pulse"], "gain", sweep2param("gain", cfg["sweep"]["gain"])
        )

        with LivePlotter2DwithLine(
            "Pulse gain", "Iteration", line_axis=1, title="MIST Overnight"
        ) as viewer:
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
                                    Pulse("init_pulse", cfg=ctx.cfg.get("init_pulse")),
                                    Pulse("probe_pulse", cfg=ctx.cfg["probe_pulse"]),
                                    Readout("readout", ctx.cfg["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        ),
                        result_shape=(len(pdrs),),
                    ),
                    fail_retry=fail_retry,
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    iters, pdrs, mist_overnight_signal2real(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # record the last result
        self.last_cfg = cfg
        self.last_result = (iters, pdrs, signals)

        return iters, pdrs, signals

    def analyze(
        self,
        result: Optional[MISTPowerDepOvernightResultType] = None,
        *,
        g0=None,
        e0=None,
        ac_coeff=None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        _, pdrs, signals = result

        if g0 is None:
            g0 = np.mean(signals[:, 0])

        abs_diff = np.abs(signals - g0)

        if ac_coeff is None:
            xs = pdrs
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * pdrs**2
            xlabel = r"$\bar n$"

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(xs, abs_diff.T)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))

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
            z_info={"name": "Signal", "unit": "a.u.", "values": overnight_signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )


MISTPowerDepGEResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


class MISTPowerDepGETaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    pre_pi_pulse: PulseCfg
    probe_pulse: PulseCfg
    post_pi_pulse: PulseCfg
    readout: ReadoutCfg


def mist_ge_signal2real(signals: np.ndarray) -> np.ndarray:
    avg_len = max(int(0.05 * signals.shape[0]), 1)

    real_signals = signals - np.mean(signals[:avg_len], axis=0, keepdims=True)
    mist_signals = np.abs(0.5 * (real_signals[..., 0] + real_signals[..., 1]))
    decay_signals = np.abs(0.5 * (real_signals[..., 0] - real_signals[..., 1]))

    # return mist_signals  # (pdrs, ge)
    return np.concatenate([mist_signals, decay_signals], axis=1)  # (pdrs, gege)


class MISTPowerDepGE(AbsExperiment):
    def run(
        self, soc, soccfg, cfg: MISTPowerDepGETaskConfig
    ) -> MISTPowerDepGEResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg

        cfg["pre_pi_pulse"] = deepcopy(cfg["pi_pulse"])
        cfg["post_pi_pulse"] = deepcopy(cfg["pi_pulse"])

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        cfg["sweep"] = {
            "gain": cfg["sweep"]["gain"],
            "pre_ge": make_sweep(0.0, 1.0, 2),
            "post_ge": make_sweep(0.0, 1.0, 2),
        }
        pdrs = sweep2array(cfg["sweep"]["gain"])  # predicted amplitudes

        Pulse.set_param(
            cfg["probe_pulse"], "gain", sweep2param("gain", cfg["sweep"]["gain"])
        )
        Pulse.set_param(
            cfg["pre_pi_pulse"],
            "on/off",
            sweep2param("pre_ge", cfg["sweep"]["pre_ge"]),
        )
        Pulse.set_param(
            cfg["post_pi_pulse"],
            "on/off",
            sweep2param("post_ge", cfg["sweep"]["post_ge"]),
        )

        fig, ((ax1, ax2),) = make_plot_frame(1, 2, figsize=(14, 4))

        with MultiLivePlotter(
            fig,
            dict(
                mist=LivePlotter1D(
                    "Pulse gain",
                    "MIST Signal",
                    segment_kwargs=dict(
                        num_lines=2,
                        line_kwargs=[
                            dict(label="Ground", color="blue"),
                            dict(label="Excited", color="red"),
                        ],
                    ),
                    existed_axes=[[ax1]],
                ),
                decay=LivePlotter1D(
                    "Pulse gain",
                    "Decay Signal",
                    segment_kwargs=dict(
                        num_lines=2,
                        line_kwargs=[
                            dict(label="Ground", color="blue"),
                            dict(label="Excited", color="red"),
                        ],
                    ),
                    existed_axes=[[ax2]],
                ),
            ),
        ) as viewer:

            def plot_fn(ctx):
                real_signals = mist_ge_signal2real(ctx.data).T

                viewer.update(
                    dict(
                        mist=(pdrs, real_signals[:2]),
                        decay=(pdrs, real_signals[2:]),
                    )
                )

            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse(name="pre_pi_pulse", cfg=ctx.cfg["pre_pi_pulse"]),
                                Pulse(name="probe_pulse", cfg=ctx.cfg["probe_pulse"]),
                                Pulse(
                                    name="post_pi_pulse", cfg=ctx.cfg["post_pi_pulse"]
                                ),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(pdrs), 2, 2),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )

        # record the last result
        self.last_cfg = cfg
        self.last_result = (pdrs, signals)

        plt.close(fig)

        return pdrs, signals

    def analyze(
        self, result: Optional[MISTPowerDepGEResultType] = None, *, ac_coeff=None
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        real_signals = mist_ge_signal2real(signals)

        if ac_coeff is None:
            xs = pdrs
            xlabel = "probe gain (a.u.)"
        else:
            xs = ac_coeff * pdrs**2
            xlabel = r"$\bar n$"

        # fig, ax1 = plt.subplots(figsize=config.figsize)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.figsize, sharex=True)
        assert isinstance(fig, Figure)

        ax1.plot(xs, real_signals[:, 0], label="Ground", color="blue")
        ax1.plot(xs, real_signals[:, 1], label="Excited", color="red")
        ax1.tick_params(axis="both", which="major", labelsize=12)
        ax1.set_ylabel("MIST Signal", fontsize=14)
        ax1.set_ylim(0, 1.1 * np.max(real_signals))
        ax1.grid(True)
        ax1.legend(fontsize=12)

        ax2.plot(xs, real_signals[:, 2], label="Ground", color="blue")
        ax2.plot(xs, real_signals[:, 3], label="Excited", color="red")
        ax2.tick_params(axis="both", which="major", labelsize=12)
        ax2.set_xlabel(xlabel, fontsize=14)
        ax2.set_ylabel("Decay Signal", fontsize=14)
        ax2.set_ylim(0, 1.1 * np.max(real_signals))
        ax2.grid(True)
        ax2.legend(fontsize=12)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[MISTPowerDepGEResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/mist/pdr_ge",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
            y_info={"name": "GE", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals[:, 0].T},
            comment=comment,
            tag=tag + "/g",
            **kwargs,
        )

        save_data(
            filepath=filepath,
            x_info={"name": "Drive Power (a.u.)", "unit": "a.u.", "values": pdrs},
            y_info={"name": "GE", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals[:, 1].T},
            comment=comment,
            tag=tag + "/e",
            **kwargs,
        )
