from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import Field
from skopt import Optimizer
from skopt.space import Real

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    US_TO_S,
    AbsExperiment,
    GroupedAxesSpec,
    GroupedLoadData,
    RoleAxisSpec,
    RoleSpec,
    RoleZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlotScatter, MultiLivePlot, instant_plot
from zcu_tools.liveplot.backend import close_figure
from zcu_tools.program.v2 import (
    Branch,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)
from zcu_tools.utils.datasaver import LabberPayload


@dataclass(frozen=True)
class AutoOptResult:
    params: NDArray[np.float64]
    signals: NDArray[np.float64]
    cfg_snapshot: Optional["AutoOptCfg"] = None


class ReadoutOptimizer:
    def __init__(
        self,
        freq_sweep: SweepCfg,
        gain_sweep: SweepCfg,
        length_sweep: SweepCfg,
        num_points: int,
    ) -> None:
        self.num_points = num_points

        freqs = sweep2array(freq_sweep, allow_array=True)
        gains = sweep2array(gain_sweep, allow_array=True)
        lengths = sweep2array(length_sweep, allow_array=True)

        self.optimizer = Optimizer(
            dimensions=[
                Real(name="freq", low=freqs.min(), high=freqs.max()),
                Real(name="gain", low=gains.min(), high=gains.max()),
                Real(name="length", low=lengths.min(), high=lengths.max()),
            ],
            n_initial_points=num_points // 2,
            initial_point_generator="lhs",
            base_estimator="ET",
            acq_func="EI",
            # n_jobs=1, not -1: the ExtraTrees model is small, so spreading each
            # ask() across all cores is *slower* (~4x: parallelization overhead
            # dwarfs the work) AND saturates every core, starving the GUI render
            # thread → the window goes laggy during an auto-optimize run. Single
            # core is faster per iter and leaves CPU for the UI.
            n_jobs=1,
            acq_optimizer="auto",
        )
        self.last_param = None

    def next_params(
        self, i: int, last_snr: float | None
    ) -> tuple[float, float, float] | None:
        if i >= self.num_points:
            return None

        if last_snr is not None:
            self.optimizer.tell(self.last_param, -last_snr)

        param = self.optimizer.ask()
        param = cast(tuple[float, float, float] | None, param)

        self.last_param = param
        return param


class AutoOptModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class AutoOptSweepCfg(ConfigBase):
    freq: SweepCfg
    gain: SweepCfg
    length: SweepCfg


class AutoOptCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AutoOptModuleCfg
    sweep: AutoOptSweepCfg
    skew_penalty: float = Field(default=0.0, ge=0.0)


RO_AUTO_READOUT_FREQ_ROLE = "readout_freq"
RO_AUTO_READOUT_GAIN_ROLE = "readout_gain"
RO_AUTO_READOUT_LENGTH_ROLE = "readout_length"
RO_AUTO_SNR_ROLE = "snr"
RO_AUTO_GROUPED_ROLES = (
    RO_AUTO_READOUT_FREQ_ROLE,
    RO_AUTO_READOUT_GAIN_ROLE,
    RO_AUTO_READOUT_LENGTH_ROLE,
    RO_AUTO_SNR_ROLE,
)


def auto_opt_result_to_grouped_payloads(
    result: AutoOptResult,
) -> dict[str, LabberPayload]:
    return RO_AUTO_GROUPED_AXES_SPEC.payloads_from_result(result)


def save_auto_opt_grouped_result(
    filepath: str,
    result: AutoOptResult,
    *,
    comment: str = "",
    tag: str = "twotone/ge/ro_optimize/auto",
) -> str:
    return RO_AUTO_GROUPED_AXES_SPEC.save_grouped_result(
        filepath,
        result,
        comment=comment,
        tag=tag,
    )


def load_auto_opt_grouped_result(filepath: str) -> AutoOptResult:
    return RO_AUTO_GROUPED_AXES_SPEC.load_result(filepath)


def _validate_auto_opt_arrays(
    params: NDArray[np.float64], signals: NDArray[np.float64]
) -> None:
    if params.ndim != 2 or params.shape[1] != 3:
        raise ValueError(
            f"RO auto-optimize params must have shape (N, 3), got {params.shape}"
        )
    if signals.ndim != 1:
        raise ValueError(
            f"RO auto-optimize signals must be 1-D, got shape {signals.shape}"
        )
    if signals.shape[0] != params.shape[0]:
        raise ValueError(
            "RO auto-optimize signals length must match params rows "
            f"(got signals={signals.shape}, params={params.shape})"
        )


def _validate_auto_opt_result(result: AutoOptResult) -> None:
    _validate_auto_opt_arrays(
        np.asarray(result.params, dtype=np.float64),
        np.asarray(result.signals, dtype=np.float64),
    )


def _build_auto_opt_result(data: GroupedLoadData[AutoOptCfg]) -> AutoOptResult:
    params = np.column_stack(
        [
            data.role(RO_AUTO_READOUT_FREQ_ROLE).z,
            data.role(RO_AUTO_READOUT_GAIN_ROLE).z,
            data.role(RO_AUTO_READOUT_LENGTH_ROLE).z,
        ]
    ).astype(np.float64)
    signals = data.role(RO_AUTO_SNR_ROLE).z.astype(np.float64)
    _validate_auto_opt_arrays(params, signals)
    return AutoOptResult(
        params=params,
        signals=signals,
        cfg_snapshot=data.cfg_snapshot,
    )


_RO_AUTO_ITERATION_AXIS = (
    RoleAxisSpec.generated_arange("Iteration", "a.u.", dtype=np.int64),
)
RO_AUTO_GROUPED_AXES_SPEC = GroupedAxesSpec(
    roles=(
        RoleSpec(
            role=RO_AUTO_READOUT_FREQ_ROLE,
            axes=_RO_AUTO_ITERATION_AXIS,
            z=RoleZSpec(
                field_name="params",
                label="Readout Frequency",
                unit="Hz",
                scale=MHZ_TO_HZ,
                dtype=np.float64,
                index=0,
                index_axis=1,
            ),
        ),
        RoleSpec(
            role=RO_AUTO_READOUT_GAIN_ROLE,
            axes=_RO_AUTO_ITERATION_AXIS,
            z=RoleZSpec(
                field_name="params",
                label="Readout Gain",
                unit="a.u.",
                dtype=np.float64,
                index=1,
                index_axis=1,
            ),
        ),
        RoleSpec(
            role=RO_AUTO_READOUT_LENGTH_ROLE,
            axes=_RO_AUTO_ITERATION_AXIS,
            z=RoleZSpec(
                field_name="params",
                label="Readout Length",
                unit="s",
                scale=US_TO_S,
                dtype=np.float64,
                index=2,
                index_axis=1,
            ),
        ),
        RoleSpec(
            role=RO_AUTO_SNR_ROLE,
            axes=_RO_AUTO_ITERATION_AXIS,
            z=RoleZSpec(
                field_name="signals",
                label="SNR",
                unit="a.u.",
                dtype=np.float64,
            ),
        ),
    ),
    result_type=AutoOptResult,
    cfg_type=AutoOptCfg,
    tag="twotone/ge/ro_optimize/auto",
    result_builder=_build_auto_opt_result,
    result_validator=_validate_auto_opt_result,
)


class AutoOptExp(AbsExperiment[AutoOptResult, AutoOptCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: AutoOptCfg,
        *,
        num_points: int,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> AutoOptResult:
        orig_cfg = deepcopy(cfg)
        run_cfg = deepcopy(cfg)
        setup_devices(run_cfg, progress=True)

        freq_sweep = run_cfg.sweep.freq
        gain_sweep = run_cfg.sweep.gain
        len_sweep = run_cfg.sweep.length

        optimizer = ReadoutOptimizer(freq_sweep, gain_sweep, len_sweep, num_points)

        # (num_points, [freq, gain, length])
        params = np.full((num_points, 3), np.nan, dtype=np.float64)

        # initialize figure and axes
        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle("Readout Auto Optimization")

        ax_iter = fig.add_subplot(gs[:, 0])
        ax_freq = fig.add_subplot(gs[0, 1])
        ax_gain = fig.add_subplot(gs[1, 1])
        ax_len = fig.add_subplot(gs[2, 1])

        instant_plot(fig)  # show the figure immediately

        with MultiLivePlot(
            fig,
            plotters=dict(
                iter_scatter=LivePlotScatter(
                    "Iteration", "SNR (a.u.)", existed_axes=[[ax_iter]]
                ),
                freq_scatter=LivePlotScatter(
                    "Frequency (MHz)", "SNR (a.u.)", existed_axes=[[ax_freq]]
                ),
                gain_scatter=LivePlotScatter(
                    "Readout Gain (a.u.)", "SNR (a.u.)", existed_axes=[[ax_gain]]
                ),
                len_scatter=LivePlotScatter(
                    "Readout Length (us)", "SNR (a.u.)", existed_axes=[[ax_len]]
                ),
            ),
        ) as viewer:
            env: dict[str, Any] = {}

            def plot_fn(data: NDArray[np.float64]) -> None:
                idx = int(env["index"])
                snrs = np.abs(data)  # (num_points, )

                cur_freq, cur_gain, cur_len = params[idx, :]

                fig.suptitle(
                    f"Iteration {idx}, Frequency: {1e-3 * cur_freq:.4g} (GHz), Gain: {cur_gain:.2g} (a.u.), Length: {cur_len:.2g} (us)"
                )

                viewer.get_plotter("iter_scatter").update(
                    np.arange(num_points), snrs, refresh=False
                )
                viewer.get_plotter("freq_scatter").update(
                    params[:, 0], snrs, refresh=False
                )
                viewer.get_plotter("gain_scatter").update(
                    params[:, 1], snrs, refresh=False
                )
                viewer.get_plotter("len_scatter").update(
                    params[:, 2], snrs, refresh=False
                )
                viewer.refresh()

            signals_buffer = SignalBuffer(
                (num_points,),
                dtype=np.float64,
                on_update=plot_fn,
            )
            with Schedule(run_cfg, signals_buffer, env_dict=env) as sched:
                for idx, (_, step) in enumerate(
                    sched.scan("Iteration", range(num_points))
                ):
                    sched.env["index"] = idx

                    last_snr = None
                    if idx > 0:
                        last_snr = np.abs(signals_buffer.array[idx - 1])
                    cur_params = optimizer.next_params(idx, last_snr)

                    if cur_params is None:
                        sched.set_stop()
                        break

                    params[idx, :] = cur_params
                    modules = step.cfg.modules
                    modules.readout.set_param("freq", cur_params[0])
                    modules.readout.set_param("gain", cur_params[1])
                    modules.readout.set_param("length", cur_params[2])
                    tracker = MomentTracker()
                    _ = (
                        step.prog_builder(soc, soccfg)
                        .add(
                            Reset("reset", cfg=modules.reset),
                            Branch("ge", [], Pulse("qub_pulse", cfg=modules.qub_pulse)),
                            Readout("readout", cfg=modules.readout),
                        )
                        .declare_sweep("ge", 2)
                        .build_and_acquire(
                            raw2signal_fn=lambda _raw: snr_as_signal(
                                [tracker],
                                ge_axis=1,
                                skew_penalty=sched.cfg.skew_penalty,
                            ),
                            trackers=[tracker],
                            **(acquire_kwargs or {}),
                        )
                    )
                signals = signals_buffer.array
        close_figure(fig)

        # record the last result
        self.last_result = AutoOptResult(params, signals, cfg_snapshot=orig_cfg)

        return self.last_result

    def analyze(
        self, result: AutoOptResult | None = None
    ) -> tuple[float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        params, signals = result.params, result.signals
        snrs = np.abs(signals)

        max_id = np.nanargmax(snrs)
        max_snr = float(snrs[max_id])
        best_params = params[max_id, :]

        figsize = (8, 5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle("Readout Auto Optimization")

        ax_iter = fig.add_subplot(gs[:, 0])
        ax_freq = fig.add_subplot(gs[0, 1])
        ax_gain = fig.add_subplot(gs[1, 1])
        ax_len = fig.add_subplot(gs[2, 1])

        ax_iter.scatter(np.arange(len(snrs)), snrs, s=1)
        ax_iter.axhline(max_snr, color="r", ls="--", label=f"best = {max_snr:.2g}")
        ax_iter.scatter([max_id], [max_snr], color="r", marker="*")
        ax_iter.set_xlabel("Iteration")
        ax_iter.set_ylabel("SNR")
        ax_iter.legend()
        ax_iter.grid(True)

        def plot_ax(ax, param_idx, label_name) -> None:
            ax.scatter(params[:, param_idx], snrs, s=1)
            best_value = best_params[param_idx]
            ax.axvline(best_value, color="r", ls="--", label=f"best = {best_value:.2g}")
            ax.scatter([best_value], [max_snr], color="r", marker="*")
            ax.set_xlabel(label_name)
            ax.set_ylabel("SNR")
            ax.legend()
            ax.grid(True)

        plot_ax(ax_freq, 0, "Frequency (MHz)")
        plot_ax(ax_gain, 1, "Readout Gain (a.u.)")
        plot_ax(ax_len, 2, "Readout Length (us)")

        return float(best_params[0]), float(best_params[1]), float(best_params[2]), fig

    def save(
        self,
        filepath: str,
        result: AutoOptResult | None = None,
        comment: str | None = None,
        tag: str = "twotone/ge/ro_optimize/auto",
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        if result.cfg_snapshot is None:
            raise ValueError("Cannot save result without configuration snapshot")
        RO_AUTO_GROUPED_AXES_SPEC.save_experiment_result(
            filepath,
            result,
            comment=comment,
            tag=tag,
            make_comment_fn=make_comment,
        )

    def load(self, filepath: str) -> AutoOptResult:
        self.last_result = load_auto_opt_grouped_result(filepath)
        return self.last_result
