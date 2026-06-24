from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skopt import Optimizer
from skopt.space import Real

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import (
    Task,
    TaskState,
    run_task,
)
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlotScatter, MultiLivePlot, instant_plot
from zcu_tools.liveplot.backend import close_figure
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
    SweepCfg,
)
from zcu_tools.utils.datasaver import (
    DatasetRole,
    LabberMetadata,
    LabberPayload,
    load_grouped_labber_data,
    save_grouped_labber_data,
)


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
    params = np.asarray(result.params, dtype=np.float64)
    signals = np.asarray(result.signals, dtype=np.float64)
    _validate_auto_opt_arrays(params, signals)

    axes = _iteration_axis(params.shape[0])
    return {
        RO_AUTO_READOUT_FREQ_ROLE: LabberPayload(
            ("Readout Frequency", "Hz", params[:, 0] * 1e6),
            axes=axes,
        ),
        RO_AUTO_READOUT_GAIN_ROLE: LabberPayload(
            ("Readout Gain", "a.u.", params[:, 1]),
            axes=axes,
        ),
        RO_AUTO_READOUT_LENGTH_ROLE: LabberPayload(
            ("Readout Length", "s", params[:, 2] * 1e-6),
            axes=axes,
        ),
        RO_AUTO_SNR_ROLE: LabberPayload(
            ("SNR", "a.u.", signals),
            axes=axes,
        ),
    }


def save_auto_opt_grouped_result(
    filepath: str,
    result: AutoOptResult,
    *,
    comment: str = "",
    tag: str = "twotone/ge/ro_optimize/auto",
) -> str:
    return save_grouped_labber_data(
        filepath,
        auto_opt_result_to_grouped_payloads(result),
        metadata=LabberMetadata(comment=comment, tags=tag),
    )


def load_auto_opt_grouped_result(filepath: str) -> AutoOptResult:
    grouped = load_grouped_labber_data(filepath, required_roles=RO_AUTO_GROUPED_ROLES)
    payloads = {
        str(role): grouped.roles[DatasetRole(role)] for role in RO_AUTO_GROUPED_ROLES
    }
    iterations = _require_shared_iteration_axes(payloads)
    num_points = len(iterations)

    freq_hz = _require_1d_real_role(
        payloads[RO_AUTO_READOUT_FREQ_ROLE],
        RO_AUTO_READOUT_FREQ_ROLE,
        "Readout Frequency",
        "Hz",
        num_points,
    )
    gain = _require_1d_real_role(
        payloads[RO_AUTO_READOUT_GAIN_ROLE],
        RO_AUTO_READOUT_GAIN_ROLE,
        "Readout Gain",
        "a.u.",
        num_points,
    )
    length_s = _require_1d_real_role(
        payloads[RO_AUTO_READOUT_LENGTH_ROLE],
        RO_AUTO_READOUT_LENGTH_ROLE,
        "Readout Length",
        "s",
        num_points,
    )
    snr = _require_1d_real_role(
        payloads[RO_AUTO_SNR_ROLE],
        RO_AUTO_SNR_ROLE,
        "SNR",
        "a.u.",
        num_points,
    )

    params = np.column_stack([freq_hz * 1e-6, gain, length_s * 1e6]).astype(np.float64)
    signals = snr.astype(np.float64)
    _validate_auto_opt_arrays(params, signals)

    cfg_snapshot = None
    comment = grouped.metadata.comment
    if comment:
        cfg, _, _ = parse_comment(comment)
        if cfg is not None:
            cfg_snapshot = AutoOptCfg.validate_or_warn(cfg, source=filepath)

    return AutoOptResult(params=params, signals=signals, cfg_snapshot=cfg_snapshot)


def _iteration_axis(num_points: int) -> list[tuple[str, str, NDArray[np.int64]]]:
    return [("Iteration", "a.u.", np.arange(num_points, dtype=np.int64))]


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


def _require_shared_iteration_axes(
    payloads: Mapping[str, LabberPayload],
) -> NDArray[np.int64]:
    reference: NDArray[np.int64] | None = None
    for role, payload in payloads.items():
        if len(payload.axes) != 1:
            raise ValueError(
                f"RO auto-optimize {role!r} role must have exactly one axis"
            )
        axis = payload.axes[0]
        if axis.name != "Iteration" or axis.unit != "a.u.":
            raise ValueError(
                f"RO auto-optimize {role!r} axis must be 'Iteration' / 'a.u.', "
                f"got {axis.name!r} / {axis.unit!r}"
            )
        values = np.asarray(axis.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(
                f"RO auto-optimize {role!r} Iteration axis must be 1-D, "
                f"got shape {values.shape}"
            )
        rounded = np.round(values)
        if not np.allclose(values, rounded):
            raise ValueError("RO auto-optimize Iteration axis must contain integers")
        iterations = rounded.astype(np.int64)
        if not np.array_equal(iterations, np.arange(len(iterations), dtype=np.int64)):
            raise ValueError("RO auto-optimize Iteration axis must equal arange(N)")
        if reference is None:
            reference = iterations
        elif not np.array_equal(reference, iterations):
            raise ValueError(
                "RO auto-optimize grouped roles disagree on Iteration axis"
            )

    if reference is None:
        raise ValueError("RO auto-optimize grouped result has no roles")
    return reference


def _require_1d_real_role(
    payload: LabberPayload,
    role: str,
    expected_label: str,
    expected_unit: str,
    num_points: int,
) -> NDArray[np.float64]:
    if payload.data.name != expected_label or payload.data.unit != expected_unit:
        raise ValueError(
            f"RO auto-optimize {role!r} z channel is "
            f"{payload.data.name!r} [{payload.data.unit!r}], expected "
            f"{expected_label!r} [{expected_unit!r}]"
        )
    values = np.asarray(payload.z)
    if values.shape != (num_points,):
        raise ValueError(
            f"RO auto-optimize {role!r} z shape {values.shape} != "
            f"expected {(num_points,)}"
        )
    return _real_values(values, f"RO auto-optimize {role!r}")


def _real_values(values: NDArray[Any], context: str) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        if np.any(np.imag(values) != 0.0):
            raise ValueError(f"{context} must not contain imaginary values")
        values = np.real(values)
    return np.asarray(values, dtype=np.float64)


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
        setup_devices(cfg, progress=True)

        freq_sweep = cfg.sweep.freq
        gain_sweep = cfg.sweep.gain
        len_sweep = cfg.sweep.length

        optimizer = ReadoutOptimizer(freq_sweep, gain_sweep, len_sweep, num_points)

        # (num_points, [freq, gain, length])
        params = np.full((num_points, 3), np.nan, dtype=np.float64)

        def update_fn(i: int, ctx: TaskState[Any, Any, AutoOptCfg], _) -> None:
            ctx.env["index"] = i

            last_snr = None
            if i > 0:
                last_snr = np.abs(ctx.root_data[i - 1])
                # last_snr /= np.sqrt(params[i - 1, 2])
            cur_params = optimizer.next_params(i, last_snr)

            if cur_params is None:
                # TODO: Better way to early stop
                raise KeyboardInterrupt("No more parameters to optimize.")

            params[i, :] = cur_params
            modules = ctx.cfg.modules
            modules.readout.set_param("freq", cur_params[0])
            modules.readout.set_param("gain", cur_params[1])
            modules.readout.set_param("length", cur_params[2])

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

            def plot_fn(ctx: TaskState[Any, Any, AutoOptCfg]) -> None:
                idx: int = ctx.env["index"]
                snrs = np.abs(ctx.root_data)  # (num_points, )

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

            def measure_fn(
                ctx: TaskState[NDArray[np.float64], Any, AutoOptCfg], update_hook
            ):
                modules = ctx.cfg.modules
                prog = ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("qub_pulse", modules.qub_pulse)),
                        Readout("readout", modules.readout),
                    ],
                    sweep=[("ge", 2)],
                )
                tracker = MomentTracker()
                prog.acquire(
                    soc,
                    progress=False,
                    round_hook=lambda i, avg_d: update_hook(i, [tracker]),
                    trackers=[tracker],
                    stop_checkers=[ctx.is_stop],
                    **(acquire_kwargs or {}),
                )
                return [tracker]

            results = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=1),
                    dtype=np.float64,
                    pbar_n=cfg.rounds,
                ).scan(
                    "Iteration",
                    list(range(num_points)),
                    before_each=update_fn,
                ),
                init_cfg=cfg,
                on_update=plot_fn,
            )
            signals = np.asarray(results)
        close_figure(fig)

        # record the last result
        self.last_result = AutoOptResult(params, signals, cfg_snapshot=cfg)

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
        cfg = result.cfg_snapshot
        comment = make_comment(cfg, comment)

        save_auto_opt_grouped_result(filepath, result, comment=comment, tag=tag)

    def load(self, filepath: str) -> AutoOptResult:
        self.last_result = load_auto_opt_grouped_result(filepath)
        return self.last_result
