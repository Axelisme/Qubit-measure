"""Two-line CZ chevrons conditioned on the measured preparation."""

from collections.abc import Callable
from dataclasses import dataclass
from textwrap import fill
from typing import Annotated, ClassVar, Self

import numpy as np
from matplotlib import colormaps
from matplotlib.artist import Artist
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.text import Text
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
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
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    PulseCfg,
    SweepCfg,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
)
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    MuxReadout,
    MuxReadoutCfg,
    Reset,
    ResetCfg,
)

from zcu_lab.v2.support.paired_counts import (
    POINT_ALIAS_DESCRIPTION,
    joint_probabilities,
    validate_qubit_tones,
)
from zcu_lab.v2.support.paired_drive import SweptQubitDriveCfg
from zcu_lab.v2.support.rabi_duration import (
    RabiRate,
    ReturnData,
    cosine,
    fit_duration_rates,
)


class HeraldedChevronCfg(ConfigBase):
    """Configure matched dual-line sweeps, preparation and joint readout.

    Drive gains and relative phase are fixed; both lines use the same scanned
    carrier and positive full-envelope duration. Preparation and contrast use
    the Xπ pulses on the corresponding drive generators. Tone indices are
    positions in the full mux readout, in Qubit1/Qubit2 order. The selected
    tones require GE Post radii. Reset may prepare the ensemble but may not
    mask shots. Invalid routes, classifiers and durations raise ValueError.
    """

    reset: Annotated[
        ResetCfg | None,
        FieldMeta(
            label="Reset",
            description="Optional ordinary reset before nominal preparation; no herald mask.",
        ),
    ] = None
    x180_qubit1: Annotated[
        PulseCfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Calibrated Qubit1 Xπ for nominal preparation and terminal contrast.",
        ),
    ]
    x180_qubit2: Annotated[
        PulseCfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Calibrated Qubit2 Xπ for nominal preparation and terminal contrast.",
        ),
    ]
    drive: Annotated[
        SweptQubitDriveCfg,
        FieldMeta(
            label="Dual-line drive",
            description="Set the two charge-line drives with fixed signed gains and relative phase.",
        ),
    ]
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Mux readout",
            description="Use the same calibrated mux readout for the post-preparation herald and the terminal read. "
            + POINT_ALIAS_DESCRIPTION,
        ),
    ]
    qubit1_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit1 tone index",
            description="Zero-based Qubit1 position in the full mux readout.",
        ),
    ]
    qubit2_tone: Annotated[
        int,
        FieldMeta(
            label="Qubit2 tone index",
            description="Zero-based Qubit2 position in the full mux readout.",
        ),
    ]
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Common CZ drive frequency",
            description="Sweep the shared carrier of both CZ drives around the measured resonances.",
        ),
    ]
    duration: Annotated[
        SweepCfg,
        FieldMeta(
            unit="µs",
            label="Common waveform envelope duration",
            description="Sweep the full envelope length of both CZ drives, not the flat-top plateau alone.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Attempted shots per nominal preparation and scan point in one round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of complete acquisitions accumulated at each point.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="µs",
            description="Delay before the first repetition of each acquisition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation delay",
            unit="µs",
            decimals=3,
            description="Delay after each repetition before the next preparation.",
        ),
    ]

    @model_validator(mode="after")
    def validate_scan(self) -> Self:
        """Reject invalid tones, mismatched generators and nonpositive durations."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        if (
            self.drive.qubit1.ch != self.x180_qubit1.ch
            or self.drive.qubit2.ch != self.x180_qubit2.ch
        ):
            raise ValueError("Each drive must use its qubit's calibrated Xπ generator")
        if np.any(sweep2array(self.duration) <= 0):
            raise ValueError("Drive durations must be positive")
        return self


@dataclass(frozen=True)
class HeraldedChevronResult:
    """Common carrier MHz, full envelope µs and float64 joint counts.

    frequencies and durations are the requested scan coordinates. sums has
    shape (2, 2, frequency, duration, 3, 3, 3, 3): nominal Qubit1/Qubit2
    preparation (I/Xπ), frequency, duration, herald Qubit1/Qubit2, then terminal
    Qubit1/Qubit2. Each class axis is g/e/other. Unmeasured rows are all NaN;
    zeros are measured counts. Counts are summed over completed rounds.
    """

    frequencies: NDArray[np.float64]
    durations: NDArray[np.float64]
    sums: NDArray[np.float64]


def restore_result(data: GroupedLoadData[HeraldedChevronCfg]) -> HeraldedChevronResult:
    """Restore the requested scan coordinates and joint counts from Native sums.

    Missing sums or invalid persisted axes propagate the load error.
    """
    sums = data.variable("sums")
    return HeraldedChevronResult(sums.axes[5], sums.axes[4], sums.z.astype(np.float64))


class HeraldedChevronExp(Experiment[HeraldedChevronResult, HeraldedChevronCfg]):
    """Scan the CZ drive and measure herald-conditioned terminal return."""

    # ADR-0074: persist counts of the two explicitly tagged classifications.
    AXES_SPEC: ClassVar[GroupedAxesSpec[HeraldedChevronResult, HeraldedChevronCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="terminal_qubit1", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_qubit2", unit=""
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_qubit1", unit=""
                        ),
                        VariableAxisSpec(
                            label="Drive duration",
                            unit="s",
                            field_name="durations",
                            scale=1e-6,
                        ),
                        VariableAxisSpec(
                            label="Common drive frequency",
                            unit="Hz",
                            field_name="frequencies",
                            scale=1e6,
                        ),
                        VariableAxisSpec(
                            label="Prepared Qubit2",
                            unit="",
                            values=(0, 1),
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            label="Prepared Qubit1",
                            unit="",
                            values=(0, 1),
                            dtype=np.int64,
                        ),
                    ),
                    z=VariableZSpec(
                        field_name="sums",
                        label="Herald × terminal class counts",
                        unit="",
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=HeraldedChevronResult,
            cfg_type=HeraldedChevronCfg,
            cfg_schema_version="1.0",
            tag="heralded/chevron",
            result_builder=restore_result,
        )
    )

    def run(
        self, config: HeraldedChevronCfg, *, context: RunContext
    ) -> HeraldedChevronResult:
        """Scan the same swept drive after four equally attempted preparations.

        The shot program applies ordinary reset if configured, nominal I/Xπ,
        mux herald, simultaneous drive, simultaneous Xπ contrast and terminal
        mux readout. Constant/flat-top drives sweep duration in hardware;
        other length-settable envelopes rebuild on the host. Return accumulated
        counts from completed rounds, with unmeasured rows NaN. Compilation and
        acquisition failures propagate.
        """
        frequencies = sweep2array(config.frequency)
        durations = sweep2array(config.duration)
        # Compare the four measured herald branches as completed rounds arrive.
        fig, panels = _chevron_figure(
            context.plots, "measurement", frequencies, durations
        )
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Artist, ...]:
            _draw_returns(panels, _return_maps(_pool_preparations(sums)))
            return tuple(artist for panel in panels for artist in panel)

        if all(
            isinstance(pulse.waveform, (ConstWaveformCfg, FlatTopWaveformCfg))
            for pulse in (config.drive.qubit1, config.drive.qubit2)
        ):
            return self._run_fpga_duration(
                config, context, frequencies, durations, update_view
            )
        return self._run_host_duration(
            config, context, frequencies, durations, update_view
        )

    def _run_fpga_duration(
        self,
        cfg: HeraldedChevronCfg,
        context: RunContext,
        frequencies: NDArray[np.float64],
        durations: NDArray[np.float64],
        update_view: Callable[[NDArray[np.float64]], None],
    ) -> HeraldedChevronResult:
        """Acquire all nominal preparations and both sweeps in one program."""
        shape = (2, 2, len(frequencies), len(durations), 3, 3, 3, 3)
        buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
        with Schedule(buffer, scope=context.scope) as sched:
            prog = _build_shot_program(cfg, context, duration=None)
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=shot_sums(prog, _class_axes(cfg)),
                rounds=cfg.rounds,
                options=RunOptions(),
            )
        buffer.trigger_update(flush=True)
        return HeraldedChevronResult(frequencies, durations, buffer.array)

    def _run_host_duration(
        self,
        cfg: HeraldedChevronCfg,
        context: RunContext,
        frequencies: NDArray[np.float64],
        durations: NDArray[np.float64],
        update_view: Callable[[NDArray[np.float64]], None],
    ) -> HeraldedChevronResult:
        """Rebuild the sampled envelopes per length; frequency stays FPGA swept."""
        # Schedule.scan writes duration into the leading axis of the buffer.
        shape = (len(durations), 2, 2, len(frequencies), 3, 3, 3, 3)
        buffer = SignalBuffer(
            shape,
            dtype=np.float64,
            on_update=lambda sums: update_view(np.moveaxis(sums, 0, 3)),
        )
        classes = _class_axes(cfg)
        with Schedule(buffer, scope=context.scope) as sched:
            for duration, step in sched.scan("duration", durations.tolist()):
                prog = _build_shot_program(cfg, context, duration=duration)
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    reducer=shot_sums(prog, classes),
                    rounds=cfg.rounds,
                    options=RunOptions(),
                )
        buffer.trigger_update(flush=True)
        return HeraldedChevronResult(
            frequencies, durations, np.moveaxis(buffer.array, 0, 3)
        )

    def analyze(
        self,
        source: RunRecord[HeraldedChevronCfg, HeraldedChevronResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot conditional return from saved joint counts, with no writeback.

        Pool the nominal preparations and fit duration traces only for the
        target (herald ee, terminal gg). The lowest finite target return
        selects a slice; unresolved rates retain their reasons. Missing saved
        cfg, mismatched axes or invalid count rows raise ValueError.
        """
        del options
        cfg, result = source.cfg, source.result
        if cfg is None:
            raise ValueError("Heralded chevron analysis requires saved cfg")
        _validate_result(cfg, result)

        pooled = _pool_preparations(result.sums)
        maps = _return_maps(pooled)
        fig, panels = _chevron_figure(
            plots, "chevron", result.frequencies, result.durations
        )
        _draw_returns(panels, maps)

        # Select a duration slice from the measured target only.
        target = maps[-1]
        row = len(result.frequencies) // 2
        if np.isfinite(target).any():
            row, col = np.unravel_index(np.nanargmin(target), target.shape)
            fig.axes[3].plot(
                result.durations[col], result.frequencies[row], "w+", markersize=10
            )
            fig.suptitle(
                "Heralded CZ chevrons · Qubit1 / Qubit2\n"
                "Gray: no measured probability · + selects the target duration slice"
            )

        # Preserve all nine terminal outcomes for the target's fit weights and SEM.
        counts = pooled[..., 1, 1, :, :]
        totals = counts.sum(axis=(-2, -1))
        data = ReturnData(
            joint_probabilities(counts).reshape(*target.shape, 9),
            np.isfinite(totals) & (totals > 0),
            totals,
        )
        rates = fit_duration_rates(result.durations, data)
        _plot_duration_rates(plots, result.frequencies, rates)
        indices = sorted({0, int(row), len(result.frequencies) - 1})
        _plot_duration_traces(plots, result, counts, data, rates, indices)


def _class_axes(cfg: HeraldedChevronCfg) -> tuple[Classify, ...]:
    """Order the last four Result axes by read, then by qubit position."""
    return (
        Classify("herald", cfg.qubit1_tone),
        Classify("herald", cfg.qubit2_tone),
        Classify("terminal", cfg.qubit1_tone),
        Classify("terminal", cfg.qubit2_tone),
    )


def _build_shot_program(
    cfg: HeraldedChevronCfg, context: RunContext, *, duration: float | None
) -> TimelineProgramV2:
    """Compile the shot with one host duration, or a hardware duration sweep."""
    builder = TimelineBuilder(
        context.soccfg,
        reps=cfg.reps,
        initial_delay=cfg.initial_delay,
        relax_delay=cfg.relax_delay,
    )
    builder.add_loop("prepared_qubit1", 2)
    builder.add_loop("prepared_qubit2", 2)
    frequency = builder.add_sweep("frequency", cfg.frequency)
    length = (
        builder.add_sweep("duration", cfg.duration) if duration is None else duration
    )
    prepare_qubit1 = Pulse("prepare_qubit1", cfg.x180_qubit1)
    prepare_qubit2 = Pulse("prepare_qubit2", cfg.x180_qubit2)
    readout = MuxReadout("readout", cfg.readout)

    if cfg.reset is not None:
        builder.emit(Reset("reset", cfg.reset, terminal=readout))
    # Each loop applies I at index 0 and Xπ at index 1 on its own line.
    builder.fork("default", into=["prepare_qubit1_time", "prepare_qubit2_time"])
    with (
        builder.branch(
            by="prepared_qubit1", align_timeline=True, timeline="prepare_qubit1_time"
        ) as cases,
        cases.case(1),
    ):
        builder.emit(prepare_qubit1)
    with (
        builder.branch(
            by="prepared_qubit2", align_timeline=True, timeline="prepare_qubit2_time"
        ) as cases,
        cases.case(1),
    ):
        builder.emit(prepare_qubit2)
    builder.join("prepare_qubit1_time", "prepare_qubit2_time", into="default")

    # The readout's post-delay precedes the drive. There is no herald feedback.
    builder.emit(readout, tag="herald")
    builder.fork("default", into=["drive_qubit1_time", "drive_qubit2_time"])
    builder.emit(
        Pulse("drive_qubit1", cfg.drive.qubit1, freq=frequency, length=length),
        timeline="drive_qubit1_time",
    )
    builder.emit(
        Pulse("drive_qubit2", cfg.drive.qubit2, freq=frequency, length=length),
        timeline="drive_qubit2_time",
    )
    builder.join("drive_qubit1_time", "drive_qubit2_time", into="default")

    builder.fork("default", into=["contrast_qubit1_time", "contrast_qubit2_time"])
    builder.emit(
        Pulse("contrast_qubit1", cfg.x180_qubit1), timeline="contrast_qubit1_time"
    )
    builder.emit(
        Pulse("contrast_qubit2", cfg.x180_qubit2), timeline="contrast_qubit2_time"
    )
    builder.join("contrast_qubit1_time", "contrast_qubit2_time", into="default")
    builder.emit(readout, tag="terminal")
    return builder.build()


_DURATION_LABEL = "Common waveform envelope duration (µs)"
_FREQUENCY_LABEL = "Common CZ drive frequency (MHz)"
# (herald Qubit1, Qubit2), then terminal hit after the final double Xπ.
_RETURN_BINS = (
    ((0, 0), (1, 1)),  # gg → terminal ee
    ((0, 1), (1, 0)),  # ge → terminal eg
    ((1, 0), (0, 1)),  # eg → terminal ge
    ((1, 1), (0, 0)),  # ee → terminal gg (target)
)


def _validate_result(cfg: HeraldedChevronCfg, result: HeraldedChevronResult) -> None:
    """Reject inconsistent scan axes and partly missing or invalid class rows."""
    frequencies, durations = sweep2array(cfg.frequency), sweep2array(cfg.duration)
    tolerance = 2 * np.finfo(float).eps  # Allow Native MHz/µs scale round-trip.
    if (
        result.frequencies.shape != frequencies.shape
        or result.durations.shape != durations.shape
        or not np.allclose(result.frequencies, frequencies, rtol=tolerance, atol=0)
        or not np.allclose(result.durations, durations, rtol=tolerance, atol=0)
        or result.sums.shape != (2, 2, len(frequencies), len(durations), 3, 3, 3, 3)
    ):
        raise ValueError("Heralded chevron scan and count axes must match saved cfg")
    classes = (-4, -3, -2, -1)
    finite = np.all(np.isfinite(result.sums) & (result.sums >= 0), axis=classes)
    missing = np.all(np.isnan(result.sums), axis=classes)
    if not np.all(finite | missing):
        raise ValueError(
            "Heralded chevron contains invalid or partially missing count rows"
        )


def _pool_preparations(sums: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum nominal preparations without turning an unmeasured cell into zero shots."""
    measured = np.isfinite(sums[..., 0, 0, 0, 0]).any(axis=(0, 1))
    pooled = np.nansum(sums, axis=(0, 1))
    pooled[~measured] = np.nan
    return pooled


def _return_maps(pooled: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    """Use all nine terminal outcomes in each measured herald's own denominator."""
    return tuple(
        joint_probabilities(pooled[..., h1, h2, :, :])[..., t1, t2]
        for (h1, h2), (t1, t2) in _RETURN_BINS
    )


def _chevron_figure(
    plots: Plots,
    name: str,
    frequencies: NDArray[np.float64],
    durations: NDArray[np.float64],
) -> tuple[Figure, tuple[tuple[QuadMesh, Text], ...]]:
    """Build the same four return heatmaps and fixed probability scale for Live and Analyze."""
    fig, _ = plots.subplots(
        num=name,
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(12, 9),
        layout="constrained",
    )
    fig.suptitle(
        "Heralded CZ chevrons · Qubit1 / Qubit2\nGray: no measured probability"
    )
    fig.supxlabel(_DURATION_LABEL)
    fig.supylabel(_FREQUENCY_LABEL)
    cmap = colormaps["viridis"].with_extremes(bad="0.92")
    empty = np.full((len(frequencies), len(durations)), np.nan)
    panels = []
    for ax, ((h1, h2), (t1, t2)) in zip(fig.axes, _RETURN_BINS, strict=True):
        role = "Target" if (h1, h2) == (1, 1) else "Spectator"
        ax.set_title(
            f"{role} · Herald {'ge'[h1]}{'ge'[h2]} → |{h1}{h2}⟩ → terminal {'ge'[t1]}{'ge'[t2]}"
        )
        image = ax.pcolormesh(
            durations, frequencies, empty, shading="nearest", vmin=0, vmax=1, cmap=cmap
        )
        ax.grid(False)
        missing = ax.text(
            0.5,
            0.5,
            "No retained shots",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        panels.append((image, missing))
    fig.colorbar(panels[0][0], ax=fig.axes, label="Conditional return probability")
    return fig, tuple(panels)


def _draw_returns(
    panels: tuple[tuple[QuadMesh, Text], ...], maps: tuple[NDArray[np.float64], ...]
) -> None:
    """Refresh the maps, leaving zero-denominator or unmeasured cells masked."""
    for (image, missing), values in zip(panels, maps, strict=True):
        image.set_array(values)
        missing.set_visible(not np.isfinite(values).any())


_TARGET_TITLE = "Target · Herald ee → |11⟩ → terminal gg"


def _plot_duration_rates(
    plots: Plots,
    frequencies: NDArray[np.float64],
    rates: tuple[RabiRate, ...],
) -> None:
    """Show target rate estimates and finite fits below the duration-span limit."""
    fig, ax = plots.subplots(num="rabi_rate", figsize=(7, 5.1), layout="constrained")
    values = [rate.rate if rate.rate is not None else np.nan for rate in rates]
    errors = [
        rate.uncertainty if rate.uncertainty is not None else np.nan for rate in rates
    ]
    ax.errorbar(
        frequencies,
        values,
        yerr=errors,
        fmt=".-",
        color="C0",
        label="Selected rates ± fit SE",
    )
    rejected = [
        (freq, abs(rate.parameters[2]), rate.minimum_rate_mhz)
        for freq, rate in zip(frequencies, rates, strict=True)
        if rate.parameters is not None and rate.minimum_rate_mhz is not None
    ]
    if rejected:
        x, y, limits = zip(*rejected, strict=True)
        ax.plot(x, y, "x", color="C3", label="Unresolved fitted rates")
        floor = limits[0]  # All frequencies share the saved duration span.
        ax.axhline(
            floor,
            color="0.45",
            ls="--",
            label=f"Inverse duration span: {floor:.4g} MHz",
        )
    missing = sum(rate.rate is None and rate.parameters is None for rate in rates)
    fig.suptitle(f"{_TARGET_TITLE}\nDuration-fit rates")
    ax.set(
        xlabel=_FREQUENCY_LABEL,
        ylabel="Rabi rate (MHz)",
        title=f"{missing} scan slices without a usable fit" if missing else "",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)


def _plot_duration_traces(
    plots: Plots,
    result: HeraldedChevronResult,
    counts: NDArray[np.float64],
    data: ReturnData,
    rates: tuple[RabiRate, ...],
    indices: list[int],
) -> None:
    """Show selected target slices, measured shot SEM, and cosine diagnostics."""
    fig, _ = plots.subplots(
        num="rabi_traces",
        nrows=len(indices),
        sharex=True,
        figsize=(7, 8),
        layout="constrained",
    )
    fig.suptitle(f"{_TARGET_TITLE}\nDuration slices")
    durations = result.durations
    xs = np.linspace(durations[0], durations[-1], 401)
    for ax, i in zip(fig.axes, indices, strict=True):
        rate = rates[i]
        covariance = class_probability_covariance(counts[i].reshape(-1, 9))
        ax.errorbar(
            durations,
            data.populations[i, :, 0],
            yerr=np.sqrt(covariance[:, 0, 0]),
            fmt="." if rate.parameters is not None else ".-",
            color="C0",
            label="Measured ± shot SEM",
        )
        if rate.parameters is not None:
            ax.plot(xs, cosine(xs, *rate.parameters), "k-", label="Cosine fit")
        title = f"{result.frequencies[i]:g} MHz · "
        title += (
            f"Rabi rate = {rate.rate:.4g} ± {rate.uncertainty:.2g} MHz"
            if rate.rate is not None
            else "\n" + fill(rate.reason or "Rate unavailable", width=54)
        )
        quality = rate.quality
        if (
            quality is not None
            and quality.r2 is not None
            and quality.normalized_residual_rms is not None
        ):
            title += f"\nR² = {quality.r2:.3g} · Normalized RMS = {quality.normalized_residual_rms:.3g}"
        ax.set_title(title)
    entries = {
        label: handle
        for ax in fig.axes
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True)
    }
    fig.legend(entries.values(), entries.keys(), loc="outside lower center", ncols=2)
    fig.axes[-1].set_xlabel(_DURATION_LABEL)
    fig.supylabel("P(terminal gg | herald ee)")
