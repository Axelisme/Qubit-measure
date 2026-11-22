"""Sweep readout gain using paired IQ moments to compare separation and QND."""

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.readout_metrics import readout_metrics
from zcu_tools.cfg_model import ConfigBase, FieldMeta
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
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Pulse,
    PulseReadout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class PowerResult:
    """gains: quantized readout gain (Ngain,) in normalized a.u. within [-1, 1].

    sums: float64 (Ngain, 8, 6, 6), without round normalization.
        Branch = prepare_pi*4 + r1_on*2 + middle_pi. Feature axes are
        IQ(r1), IQ(r2), each ordered (1, I, Q, I², IQ, Q²).
        sums[..., 0, 0] is the shot count; unmeasured sums are NaN.
    """

    gains: NDArray[np.float64]
    sums: NDArray[np.float64]


@pydantic_dataclass(frozen=True)
class PowerAnalyzeOptions:
    """gain: optional human-selected gain in a.u.; None proposes no writeback.

    Construction rejects nonfinite gains and values outside [-1, 1].
    """

    gain: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] | None = None


@dataclass(frozen=True)
class PowerQndAnalysis:
    """Human-selected gain in a.u., or None for no writeback.

    Gain-wise metrics and the Pareto frontier stay in the analysis figures.
    No automatic gain selection is performed.
    """

    gain: float | None


class PowerCfg(ConfigBase):
    """Paired-readout modules, gain sweep and wait in us; counts add over rounds."""

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Choose the qubit reset before preparation, or leave blank to skip reset.",
        ),
    ] = None
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit π pulse",
            description="Select the calibrated π pulse for preparation and for the pulse between the two reads.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout pulse and ADC window",
            description="Select the resonator pulse and ADC timing shared by the first and second reads.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Readout gain",
            unit="a.u.",
            description="Enter the gain sweep in a.u. for R1 on and R2.",
        ),
    ]
    wait: Annotated[
        float,
        FieldMeta(
            label="Wait after R1",
            unit="µs",
            decimals=3,
            description="Set the additional wait after the first readout pulse and its post-delay before the middle pulse.",
        ),
    ] = Field(default=0.0, ge=0.0)
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the number of paired-read shots per branch and gain in each round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Set how many times the full gain scan is acquired.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Qubit relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait in µs after the second read so the qubit relaxes.",
        ),
    ]


def restore_result(data: GroupedLoadData[PowerCfg]) -> PowerResult:
    """Restore paired IQ sums and the gain axis from the QND native record."""
    variable = data.variable("sums")
    return PowerResult(
        gains=variable.axes[3].astype(np.float64),
        sums=variable.z.astype(np.float64),
    )


def _r1_contrast(sums: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return |mean IQ(π) − mean IQ(I)| for R1 on, middle I, across gains."""
    counts = sums[:, :, 0, 0]
    counts = np.where(counts == 0, np.nan, counts)
    means = sums[:, :, 1:3, 0] / counts[..., None]
    return np.linalg.norm(means[:, 6] - means[:, 2], axis=-1)


class PowerExp(Experiment[PowerResult, PowerCfg]):
    """Measure eight preparation/R1/middle-pi branches at each readout gain."""

    Options: ClassVar[type[PowerAnalyzeOptions]] = PowerAnalyzeOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[PowerResult, PowerCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec(
                        label="R2 features", unit="", values=tuple(range(6))
                    ),
                    VariableAxisSpec(
                        label="R1 features", unit="", values=tuple(range(6))
                    ),
                    VariableAxisSpec(label="Branch", unit="", values=tuple(range(8))),
                    VariableAxisSpec(label="Gain", unit="a.u.", field_name="gains"),
                ),
                z=VariableZSpec(
                    field_name="sums", label="Shot sums", unit="", dtype=np.float64
                ),
            ),
        ),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        cfg_schema_version="1.0",
        tag="twotone/ro_optimize/power_qnd",
        result_builder=restore_result,
    )

    def run(
        self,
        cfg: PowerCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> PowerResult:
        """Acquire paired-readout sums (Ngain, 8, 6, 6) versus gain.

        Each branch resets optionally, prepares I/pi, emits R1 on/zero-gain,
        waits cfg.wait us, applies middle I/pi, then emits R2. R1 off retains
        the generator slot and ADC trigger. R2 always uses the swept gain.
        Wait begins at the R1 waveform end plus its configured post-delay,
        not the ADC window end. Readout timing must keep ADC windows disjoint.
        Identity plays the same qubit pulse at zero gain, preserving its native
        slot. All eight branches share reset, preparation, R1, wait, middle and
        R2 positions, including the same R1-to-R2 trigger interval.
        Live shows the R1-on, middle-I preparation mean-IQ difference magnitude.
        cfg is not mutated. acquire_kwargs is forwarded to summed acquisition.
        """
        # 1. Quantize readout gains for paired-read separation and QND characterization.
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": context.soccfg, "gen_ch": cfg.readout.gen_ch}
        )
        # 2. Monitor first-read preparation contrast across readout gain.
        fig, ax = context.plots.subplots(num="measurement")
        fig.suptitle("Paired-read gain sweep · Resonator · First-read contrast")
        ax.set(xlabel="Readout gain (a.u.)", ylabel="R1 mean IQ difference (a.u.)")
        viewer = live_line(ax)
        ax.lines[0].set_marker("o")

        @context.plots.on_plot_thread
        def update(sums: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Refresh readout-gain separation from the paired IQ moment sums.

            sums is float64 with shape (gain, 8, 6, 6); zero counts display as NaN.
            Return the changed artists without modifying the input array.
            """
            contrast = _r1_contrast(sums)
            count = int(np.nansum(sums[0, 2, 0, 0]))
            status = ax.set_title(
                f"Accumulated shots per branch and gain: {count} count"
            )
            return (*viewer.set(gains, contrast), status)

        fig.show()
        buffer = SignalBuffer((len(gains), 8, 6, 6), dtype=np.float64, on_update=update)
        # 3. Isolate readout backaction with eight preparation, first-read and middle-pi branches.
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", cfg.gain)
            builder.add_loop("branch", 8)
            r1_on = PulseReadout("r1", cfg.readout, gain=gain)
            r1_off = PulseReadout("r1", cfg.readout, gain=0.0)
            r2 = PulseReadout("r2", cfg.readout, gain=gain)
            qub_pulse = Pulse("qub_pulse", cfg.qub_pulse)
            identity = Pulse("qub_pulse", cfg.qub_pulse, gain=0.0)
            reset = Reset("reset", cfg.reset) if cfg.reset is not None else None
            with builder.branch(by="branch", align_timeline=True) as cases:
                for index in range(8):
                    with cases.case(index):
                        if reset is not None:
                            builder.emit(reset)
                        builder.emit(qub_pulse if index & 4 else identity)
                        builder.emit(r1_on if index & 2 else r1_off)
                        builder.timeline_inc(cfg.wait)
                        builder.emit(qub_pulse if index & 1 else identity)
                        builder.emit(r2)
            prog = builder.build()
            reducer = shot_sums(prog, [IQ("r1"), IQ("r2")])
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                reducer=reducer,
                options=RunOptions(converter=np.copy),
                **(acquire_kwargs or {}),
            )
        return PowerResult(gains, buffer.array)

    def analyze(
        self,
        source: RunRecord[PowerCfg, PowerResult],
        options: PowerAnalyzeOptions,
        *,
        plots: Plots,
    ) -> PowerQndAnalysis:
        """Compute gain-wise QND metrics and plot curves plus the Pareto frontier.

        source.cfg may be None. Each gain uses readout_metrics without smoothing
        or clipping. Only finite f_sep/qnd_on pairs enter the maximizing frontier.
        options.gain records a human choice, never an inferred optimum.
        PowerAnalyzeOptions validates finite gains within [-1, 1] at construction.
        """
        # 1. Derive paired-read metrics and identify nondominated separation/QND settings.
        result = source.result
        metrics = [readout_metrics(sums) for sums in result.sums]
        arrays = {
            name: np.array(
                [getattr(point, name) for point in metrics], dtype=np.float64
            )
            for name in (
                "snr",
                "f_sep",
                "thermal_population",
                "retention_on",
                "retention_off",
                "qnd_on",
                "qnd_off",
                "flip_up_on",
                "flip_up_off",
                "flip_down_on",
                "flip_down_off",
                "pi_miss_on",
                "pi_miss_off",
                "retention_from_covariance",
            )
        }
        f_sep, qnd = arrays["f_sep"], arrays["qnd_on"]
        valid = np.isfinite(f_sep) & np.isfinite(qnd)
        frontier = []
        for index in np.flatnonzero(valid):
            dominates = valid & (f_sep >= f_sep[index]) & (qnd >= qnd[index])
            dominates &= (f_sep > f_sep[index]) | (qnd > qnd[index])
            if not np.any(dominates):
                frontier.append(index)
        indices = np.array(frontier, dtype=np.int64)
        indices = indices[np.argsort(f_sep[indices], kind="stable")]

        # 2. Compare the live first-read contrast with the manual selection.
        fig, ax = plots.subplots(num="contrast")
        fig.suptitle("Paired-read gain sweep · Resonator · First-read contrast")
        ax.plot(result.gains, _r1_contrast(result.sums), "o", color="C0")
        ax.set(xlabel="Readout gain (a.u.)", ylabel="R1 mean IQ difference (a.u.)")
        if options.gain is not None:
            ax.axvline(options.gain, color="C3", ls="--")
            ax.set_title(f"Selected gain = {options.gain:.4g} a.u.")

        # 3. Inspect calibration and state disturbance by gain.
        _draw_gain_diagnostics(plots, result.gains, arrays)

        # 4. Expose the separation/QND trade-off with a Pareto frontier and gain lookup.
        # Keep dense gain labels in a lookup table, not on top of the points.
        gain_groups = np.array_split(np.flatnonzero(valid), 3)
        table_height = max(2.0, 0.26 * len(gain_groups[0]) + 0.8)
        fig, ax = plots.subplots(num="pareto", figsize=(13, 5 + table_height))
        fig.suptitle("Paired-read gain sweep · Resonator · Separation versus QND")
        grid = fig.add_gridspec(2, 3, height_ratios=(5, table_height))
        ax.set_subplotspec(grid[0, :])
        points = ax.scatter(
            f_sep[valid], qnd[valid], c=result.gains[valid], label="Measured settings"
        )
        ax.figure.colorbar(points, ax=ax, label="Gain (a.u.)")
        ax.plot(f_sep[indices], qnd[indices], "r.-", label="Pareto frontier")
        ax.set(xlabel="Gaussian separation Fsep", ylabel="Estimated QND Q (R1 on)")
        ax.legend(loc="best")
        for column, group in enumerate(gain_groups):
            table_ax = fig.add_subplot(grid[1, column])
            table_ax.axis("off")
            if len(group) == 0:
                continue
            table_ax.set_title("Gain lookup (* = Pareto frontier)", fontsize=10)
            table = table_ax.table(
                cellText=[
                    [
                        f"{result.gains[index]:.4g}",
                        f"{f_sep[index]:.6g}",
                        f"{qnd[index]:.6g}",
                        "*" if index in frontier else "",
                    ]
                    for index in group
                ],
                colLabels=["Gain (a.u.)", "Fsep", "QND Q", ""],
                colWidths=[0.3, 0.3, 0.3, 0.1],
                cellLoc="center",
                loc="upper center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.3)
        fig.tight_layout()
        return PowerQndAnalysis(gain=options.gain)


def _draw_gain_diagnostics(
    plots: Plots, gains: NDArray[np.float64], arrays: dict[str, NDArray[np.float64]]
) -> None:
    """Plot R1-on calibration, on/off disturbance and covariance retention vs gain."""
    fig, _ = plots.subplots(3, 1, num="separation", sharex=True, figsize=(8, 7))
    fig.suptitle("Paired-read gain sweep · Resonator · R1-on calibration")
    for ax, name, ylabel in zip(
        fig.axes,
        ("snr", "f_sep", "thermal_population"),
        (
            "g/e separation SNR",
            "Gaussian\nseparation Fsep",
            "Initial P(e)\n(estimated)",
        ),
        strict=True,
    ):
        ax.plot(gains, arrays[name], "o", color="C0")
        ax.set_ylabel(ylabel)
    fig.axes[-1].set_xlabel("Readout gain (a.u.)")
    fig.tight_layout()

    fig, _ = plots.subplots(3, 2, num="disturbance", sharex=True, figsize=(10, 8))
    fig.suptitle("Paired-read gain sweep · Resonator · State disturbance")
    for ax, name, ylabel in zip(
        fig.axes,
        (
            "retention",
            "qnd",
            "flip_up",
            "flip_down",
            "pi_miss",
            "retention_from_covariance",
        ),
        (
            "Estimated retention S",
            "Estimated QND Q",
            "P(g → e)",
            "P(e → g)",
            "π miss fraction",
            "Covariance retention S",
        ),
        strict=True,
    ):
        if name == "retention_from_covariance":
            ax.plot(gains, arrays[name], "o", color="C0", label="R1 on")
        else:
            ax.plot(gains, arrays[name + "_on"], "o", color="C0", label="R1 on")
            ax.plot(gains, arrays[name + "_off"], "s", color="C1", label="R1 off")
        ax.set_ylabel(ylabel)
    for ax in fig.axes[-2:]:
        ax.set_xlabel("Readout gain (a.u.)")
    fig.axes[0].legend(loc="best")
    fig.tight_layout()
