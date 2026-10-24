from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, cast

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import Field
from skopt import Optimizer
from skopt.space import Real
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import LabberPayload
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Branch, Pulse, PulseReadout, Readout, Reset, SweepCfg

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class AutoOptResult:
    """Candidate rows (MHz frequency, normalized gain, RF length in us) and SNR."""

    params: NDArray[np.float64]
    signals: NDArray[np.float64]


@dataclass(frozen=True)
class AutoOptAnalysis:
    best_freq: float
    best_gain: float
    best_length: float


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
        # tolerated(cast): skopt's untyped list must keep its identity, not become a tuple.
        param = cast(tuple[float, float, float] | None, param)

        self.last_param = param
        return param


class AutoOptCfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Qubit pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    freq: Annotated[
        SweepCfg,
        FieldMeta(label="Readout freq", unit="MHz"),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(label="Readout gain", unit="a.u."),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(label="Readout length", unit="us"),
    ]
    skew_penalty: Annotated[float, FieldMeta(label="Skew penalty", decimals=3)] = Field(
        default=0.0, ge=0.0
    )
    num_points: Annotated[int, FieldMeta(label="Optimizer points")] = Field(gt=0)


RO_AUTO_READOUT_FREQ_VARIABLE = "readout_freq"
RO_AUTO_READOUT_GAIN_VARIABLE = "readout_gain"
RO_AUTO_READOUT_LENGTH_VARIABLE = "readout_length"
RO_AUTO_SNR_VARIABLE = "snr"
RO_AUTO_GROUPED_VARIABLES = (
    RO_AUTO_READOUT_FREQ_VARIABLE,
    RO_AUTO_READOUT_GAIN_VARIABLE,
    RO_AUTO_READOUT_LENGTH_VARIABLE,
    RO_AUTO_SNR_VARIABLE,
)


def auto_opt_result_to_grouped_payloads(
    result: AutoOptResult,
) -> dict[str, LabberPayload]:
    return RO_AUTO_GROUPED_AXES_SPEC.payloads_from_result(result)


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
            data.variable(RO_AUTO_READOUT_FREQ_VARIABLE).z,
            data.variable(RO_AUTO_READOUT_GAIN_VARIABLE).z,
            data.variable(RO_AUTO_READOUT_LENGTH_VARIABLE).z,
        ]
    ).astype(np.float64)
    signals = data.variable(RO_AUTO_SNR_VARIABLE).z.astype(np.float64)
    _validate_auto_opt_arrays(params, signals)
    return AutoOptResult(
        params=params,
        signals=signals,
    )


_RO_AUTO_ITERATION_AXIS = (
    VariableAxisSpec.generated_arange("Iteration", "a.u.", dtype=np.int64),
)
# Keep schema 1.0: Direct readout failed at Run's first frequency write, so no
# valid Direct artifacts exist. Existing Pulse artifacts remain readable. A major
# bump would reject those valid files at the native reader's version check.
RO_AUTO_GROUPED_AXES_SPEC = GroupedAxesSpec(
    variables=(
        VariableSpec(
            variable=RO_AUTO_READOUT_FREQ_VARIABLE,
            axes=_RO_AUTO_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="params",
                label="Readout Frequency",
                unit="Hz",
                scale=1e6,
                dtype=np.float64,
                index=0,
                index_axis=1,
            ),
        ),
        VariableSpec(
            variable=RO_AUTO_READOUT_GAIN_VARIABLE,
            axes=_RO_AUTO_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="params",
                label="Readout Gain",
                unit="a.u.",
                dtype=np.float64,
                index=1,
                index_axis=1,
            ),
        ),
        VariableSpec(
            variable=RO_AUTO_READOUT_LENGTH_VARIABLE,
            axes=_RO_AUTO_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="params",
                label="Readout Length",
                unit="s",
                scale=1e-6,
                dtype=np.float64,
                index=2,
                index_axis=1,
            ),
        ),
        VariableSpec(
            variable=RO_AUTO_SNR_VARIABLE,
            axes=_RO_AUTO_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="signals",
                label="SNR",
                unit="a.u.",
                dtype=np.float64,
            ),
        ),
    ),
    result_type=AutoOptResult,
    cfg_type=AutoOptCfg,
    cfg_schema_version="3.0",
    tag="twotone/ge/ro_optimize/auto",
    result_builder=_build_auto_opt_result,
    result_validator=_validate_auto_opt_result,
)


class AutoOptExp(Experiment[AutoOptResult, AutoOptCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[AutoOptResult, AutoOptCfg]] = (
        RO_AUTO_GROUPED_AXES_SPEC
    )

    def run(
        self,
        cfg: AutoOptCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> AutoOptResult:
        """Search readout frequency, gain and RF pulse duration for high SNR.

        At each candidate, apply optional reset, omit or apply qub_pulse, then
        read out; configure qub_pulse for the intended excitation. The ADC
        integration window stays configured while RF waveform length varies.
        Feed moment-based SNR, optionally reduced by skew_penalty for distorted
        IQ distributions, back to the optimizer. Sweep min/max values define
        continuous bounds, not a Cartesian grid; num_points is the budget.
        Return params shaped (Npoint, 3) in MHz, normalized gain and us, plus
        SNR shaped (Npoint,). Unvisited rows remain NaN. SNR is not assignment
        fidelity. acquire_kwargs is forwarded to acquisition.
        """
        # Prepare the bounded search and preserve one parameter row per candidate.
        run_cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        num_points = run_cfg.num_points
        optimizer = ReadoutOptimizer(
            run_cfg.freq,
            run_cfg.gain,
            run_cfg.length,
            num_points,
        )
        params = np.full((num_points, 3), np.nan, dtype=np.float64)
        point_indices = np.arange(num_points, dtype=np.float64)

        def configure_scatter(ax: Axes) -> None:
            ax.lines[0].set_linestyle("None")
            ax.lines[0].set_marker("o")

        # Show search progress and the sampled score along each readout coordinate.
        figure, initial_ax = context.plots.subplots("measurement", figsize=(8, 5))
        initial_ax.remove()
        figure.suptitle("Readout Auto Optimization")
        grid = figure.add_gridspec(3, 2, width_ratios=[1.5, 1])
        viewers = [
            live_line(
                context.plots,
                "measurement",
                xlabel,
                "SNR (a.u.)",
                configure_axes=configure_scatter,
                axes=figure.add_subplot(cell),
            )
            for xlabel, cell in (
                ("Iteration", grid[:, 0]),
                ("Frequency (MHz)", grid[0, 1]),
                ("Readout Gain (a.u.)", grid[1, 1]),
                ("Readout Length (us)", grid[2, 1]),
            )
        ]
        current_index = 0

        def plot_fn(data: NDArray[np.float64]) -> None:
            idx = current_index
            snrs = np.abs(data)
            cur_freq, cur_gain, cur_len = params[idx, :]
            title = (
                f"Iteration {idx}, Frequency: {1e-3 * cur_freq:.4g} (GHz), "
                f"Gain: {cur_gain:.2g} (a.u.), Length: {cur_len:.2g} (us)"
            )
            for viewer, xs, viewer_title in zip(
                viewers,
                (point_indices, params[:, 0], params[:, 1], params[:, 2]),
                (title, None, None, None),
                strict=True,
            ):
                viewer.update(xs, snrs, title=viewer_title, refresh=False)
            context.plots.refresh("measurement")

        signals_buffer = SignalBuffer(
            (num_points,),
            dtype=np.float64,
            on_update=plot_fn,
        )

        # Feed back the previous score before asking for the next candidate.
        with Schedule(run_cfg, signals_buffer, scope=context.scope) as sched:
            for idx, (_, step) in enumerate(sched.scan("Iteration", range(num_points))):
                current_index = idx

                last_snr = None
                if idx > 0:
                    last_snr = np.abs(signals_buffer.array[idx - 1])
                cur_params = optimizer.next_params(idx, last_snr)

                if cur_params is None:
                    sched.set_stop()
                    break

                params[idx, :] = cur_params
                modules = step.cfg
                modules.readout.set_param("freq", cur_params[0])
                modules.readout.set_param("gain", cur_params[1])
                modules.readout.set_param("length", cur_params[2])
                tracker = MomentTracker()

                # Compare pulse-off/on preparations at this candidate's readout settings.
                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", cfg=modules.reset),
                        Branch("ge", [], Pulse("qub_pulse", cfg=modules.qub_pulse)),
                        Readout("readout", cfg=modules.readout),
                    )
                    .declare_sweep("ge", 2)
                    .build_and_acquire(
                        raw2signal_fn=lambda _raw, tracker=tracker: snr_as_signal(
                            [tracker],
                            ge_axis=1,
                            skew_penalty=sched.cfg.skew_penalty,
                        ),
                        trackers=[tracker],
                        **(acquire_kwargs or {}),
                        rounds=cfg.rounds,
                    )
                )

        # Keep the sampled path and scores, including unvisited NaN rows.
        signals = signals_buffer.array

        return AutoOptResult(params, signals)

    def analyze(
        self,
        source: RunRecord[AutoOptCfg, AutoOptResult],
        options: None,
        *,
        plots: Plots,
    ) -> AutoOptAnalysis:
        """Return the measured candidate with the largest non-NaN SNR.

        Report best_freq in MHz, normalized best_gain and RF best_length in
        us, and publish fit. This is the best sampled point, not a fitted
        global optimum. An all-NaN score array raises ValueError.
        source.cfg may be None.
        """
        del options
        result = source.result

        params, signals = result.params, result.signals
        snrs = np.abs(signals)

        max_id = np.nanargmax(snrs)
        max_snr = float(snrs[max_id])
        best_params = params[max_id, :]

        figsize = (8, 5)
        fig, ax = plots.subplots("fit", figsize=figsize)
        ax.remove()
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

        return AutoOptAnalysis(
            float(best_params[0]), float(best_params[1]), float(best_params[2])
        )
