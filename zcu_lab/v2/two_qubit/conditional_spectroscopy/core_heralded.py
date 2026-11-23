"""Two-line conditional spectra grouped by measured post-preparation heralds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
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
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.device import set_flux

from .._drive import SweptQubitDriveCfg
from .._sums import joint_probabilities, validate_qubit_tones
from ._common import excitation_sem
from .core import summarize_spectra


class HeraldedSpectroscopyCfg(ConfigBase):
    """Ordinary reset → nominal Xπ preparation → mux herald → probe → contrast → mux.

    Xπ preparation precedes the herald. Both reads use readout with the same
    GE Post circle classifiers and readout.post_delay. All 81 class counts
    accumulate over completed rounds. Reset accepts only ordinary pulse-based
    modules, not a herald mask.
    """

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select an ordinary reset before nominal preparation, or leave blank.",
        ),
    ] = None
    x180_qubit1: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit1 Xπ",
            description="Enter the Rabi-calibrated Qubit1 flip for nominal preparation and readout contrast.",
        ),
    ]
    x180_qubit2: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit2 Xπ",
            description="Enter the Rabi-calibrated Qubit2 flip for nominal preparation and readout contrast.",
        ),
    ]
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Set the shared herald and terminal readout with GE Post radii and the wait after the herald pulse.",
        ),
    ]
    drive: Annotated[
        SweptQubitDriveCfg,
        FieldMeta(
            label="Two-qubit spectroscopy probe",
            description="Enter the two weak pulses applied together on the respective qubit drive lines at the shared RF sweep frequency.",
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
    contrast: Annotated[
        bool,
        FieldMeta(
            label="Double Xπ contrast",
            description="Enable an Xπ on each qubit after the probe, immediately before readout.",
        ),
    ] = True
    dc_device: Annotated[
        DeviceName | None,
        FieldMeta(
            label="Coupler DC source",
            description="Select the connected coupler-bias source by name; leave blank to keep all sources unchanged.",
        ),
    ] = None
    dc_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="DC source unit",
            description="Select amperes, volts or the selected source's native unit for the bias sweep.",
        ),
    ] = "native"
    dc: Annotated[
        SweepCfg,
        FieldMeta(
            label="Coupler DC bias sweep",
            description="Enter the sweep in DC source unit (A, V or native); leave the source blank to ignore this sweep.",
        ),
    ] = Field(default_factory=lambda: SweepCfg(start=0, stop=0, step=0, expts=1))
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Common probe RF frequency",
            description="Enter the RF sweep (MHz); each point replaces the frequency of both probe pulses.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter attempted shots per preparation and frequency per round.",
        ),
    ] = Field(gt=0)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter repeated frequency scans at each DC bias point.",
        ),
    ] = Field(default=1, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial wait",
            unit="µs",
            description="Enter the wait before the first shot of each round.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait after each shot before the next preparation.",
        ),
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        """Reject missing calibrated tones or probes on the wrong Xπ generators."""
        validate_qubit_tones(self.readout, self.qubit1_tone, self.qubit2_tone)
        if (
            self.drive.qubit1.ch != self.x180_qubit1.ch
            or self.drive.qubit2.ch != self.x180_qubit2.ch
        ):
            raise ValueError("Each probe must use its qubit's calibrated Xπ generator")
        return self


@dataclass(frozen=True)
class HeraldedSpectroscopyResult:
    """DC coordinates, probe MHz and raw (DC, 2, 2, frequency, 3, 3, 3, 3) counts.

    dc_values is in the cfg's DC sweep unit, or [0] when DC is inactive.
    frequencies is the requested common RF in MHz. sums is float64, with
    Qubit1/Qubit2 nominal g/e followed by RF, herald Qubit1/Qubit2 and terminal
    Qubit1/Qubit2. All four class axes are g/e/other. Unmeasured points are
    all NaN; completed rounds survive cancellation.
    """

    dc_values: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    sums: NDArray[np.float64]


def restore_result(
    data: GroupedLoadData[HeraldedSpectroscopyCfg],
) -> HeraldedSpectroscopyResult:
    """Restore the scan axes and 81-class count field, as declared in AXES_SPEC.

    Missing variables or invalid persisted axes propagate the load error.
    """
    sums = data.variable("sums")
    return HeraldedSpectroscopyResult(
        dc_values=sums.axes[7], frequencies=sums.axes[4], sums=sums.z.astype(np.float64)
    )


class HeraldedSpectroscopyExp(
    Experiment[HeraldedSpectroscopyResult, HeraldedSpectroscopyCfg]
):
    """Acquire all herald × terminal counts, then pool by measured herald class."""

    # ADR-0074: classify during acquisition and persist the joint class counts.
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldedSpectroscopyResult, HeraldedSpectroscopyCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="terminal_qubit2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_qubit1", unit=""),
                    VariableAxisSpec.generated_arange(label="herald_qubit2", unit=""),
                    VariableAxisSpec.generated_arange(label="herald_qubit1", unit=""),
                    VariableAxisSpec(
                        label="Probe frequency",
                        unit="Hz",
                        field_name="frequencies",
                        scale=1e6,
                    ),
                    VariableAxisSpec(
                        label="Prepared Qubit2", unit="", values=(0, 1), dtype=np.int64
                    ),
                    VariableAxisSpec(
                        label="Prepared Qubit1", unit="", values=(0, 1), dtype=np.int64
                    ),
                    VariableAxisSpec(
                        label="DC source value (inactive if dc_device=null)",
                        unit="a.u.",
                        field_name="dc_values",
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
        result_type=HeraldedSpectroscopyResult,
        cfg_type=HeraldedSpectroscopyCfg,
        cfg_schema_version="3.1",
        tag="two_qubit/conditional_spectroscopy_heralded",
        result_builder=restore_result,
    )

    def run(
        self, config: HeraldedSpectroscopyCfg, *, context: RunContext
    ) -> HeraldedSpectroscopyResult:
        """Scan DC on the host and preparation × RF in hardware using the Run context.

        Set only the configured DC source, if any. Return unnormalized joint
        counts; unmeasured points remain NaN after Stop. Acquisition failures
        propagate. Live spectra already pool by herald class.
        """
        # 1. Set up the sweep, live plots and count buffer.
        dc, frequencies = _scan_axes(config)
        update = _live_spectra(context.plots, config, dc, frequencies)
        buffer = SignalBuffer(
            (len(dc), 2, 2, len(frequencies), 3, 3, 3, 3),
            dtype=np.float64,
            on_update=update,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            # 2. Set the host bias and declare the preparation × RF loops.
            for dc_value, step in sched.scan("dc", dc.tolist()):
                if config.dc_device is not None:
                    set_flux(
                        context.devices[config.dc_device],
                        dc_value,
                        unit=config.dc_unit,
                        stop=context.scope.stop,
                    )
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=config.reps,
                    initial_delay=config.initial_delay,
                    relax_delay=config.relax_delay,
                )
                builder.add_loop("prepared_qubit1", 2)  # 0 = g, 1 = e (outer)
                builder.add_loop("prepared_qubit2", 2)  # 0 = g, 1 = e (inner)
                frequency = builder.add_sweep("frequency", config.frequency)
                x180_qubit1 = Pulse("x180_qubit1", config.x180_qubit1)
                x180_qubit2 = Pulse("x180_qubit2", config.x180_qubit2)
                readout = MuxReadout("readout", config.readout)
                # 3. Reset and apply the selected nominal Xπ preparation.
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset, terminal=readout))
                builder.fork("default", into=["qubit1", "qubit2"])
                with (
                    builder.branch(
                        by="prepared_qubit1", align_timeline=True, timeline="qubit1"
                    ) as cases,
                    cases.case(1),
                ):
                    builder.emit(x180_qubit1)
                with (
                    builder.branch(
                        by="prepared_qubit2", align_timeline=True, timeline="qubit2"
                    ) as cases,
                    cases.case(1),
                ):
                    builder.emit(x180_qubit2)
                builder.join("qubit1", "qubit2", into="default")

                # 4. Herald and wait the readout post-delay before the simultaneous probe.
                builder.emit(readout, tag="herald")
                builder.fork("default", into=["probe_qubit1_time", "probe_qubit2_time"])
                builder.emit(
                    Pulse("drive.qubit1", config.drive.qubit1, freq=frequency),
                    timeline="probe_qubit1_time",
                )
                builder.emit(
                    Pulse("drive.qubit2", config.drive.qubit2, freq=frequency),
                    timeline="probe_qubit2_time",
                )
                builder.join("probe_qubit1_time", "probe_qubit2_time", into="default")

                # 5. Apply optional sequential contrast and read the terminal state.
                if config.contrast:
                    builder.emit(x180_qubit1)
                    builder.emit(x180_qubit2)
                builder.emit(readout, tag="terminal")
                prog = builder.build()

                # 6. Count each measured herald and terminal class combination.
                reducer = shot_sums(
                    prog,
                    [
                        Classify("herald", config.qubit1_tone),
                        Classify("herald", config.qubit2_tone),
                        Classify("terminal", config.qubit1_tone),
                        Classify("terminal", config.qubit2_tone),
                    ],
                )
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    reducer=reducer,
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        return HeraldedSpectroscopyResult(dc, frequencies, buffer.array)

    def analyze(
        self,
        source: RunRecord[HeraldedSpectroscopyCfg, HeraldedSpectroscopyResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot spectra, coverage and preparation diagnostics from saved run counts.

        Pool nominal preparations by measured herald class before normalizing.
        No analysis options, fit or writeback. Missing cfg, inconsistent scan axes
        or invalid/partially missing count rows raise ValueError.
        """
        # 1. Check the saved scan and count layout before interpreting it.
        del options
        cfg, data = source.cfg, source.result
        if cfg is None:
            raise ValueError("Heralded spectroscopy analysis requires saved cfg")
        dc, frequencies = _scan_axes(cfg)
        _validate_counts(data, dc, frequencies)

        # 2. Pool preparations by measured herald before computing terminal P(e).
        pooled = _pool_by_herald(data.sums)
        spectra = summarize_spectra(pooled)
        sem = excitation_sem(pooled)

        # 3. Plot spectra, retained shots and preparation diagnostics per bias.
        for index, bias in enumerate(dc):
            title = _title(cfg, float(bias))
            _plot_spectra(
                plots,
                index,
                f"{title} · {_contrast(cfg)}",
                data.frequencies,
                spectra.excitation[index],
                sem[index],
            )
            _plot_coverage(
                plots, index, title, data.frequencies, spectra.acquired_shots[index]
            )
            _plot_herald_matrix(plots, index, title, data.sums[index])

        # 4. Show continuous RF × DC maps for a multi-bias scan.
        if cfg.dc_device is not None and len(dc) > 1:
            _plot_bias_maps(plots, cfg, data, spectra.excitation)


_STATES = ("g", "e", "other")
_HERALDS = ("gg", "ge", "eg", "ee")
_QUBITS = ("Qubit1", "Qubit2")
_RF_LABEL = "Common probe RF frequency (MHz)"


def _scan_axes(
    cfg: HeraldedSpectroscopyCfg,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the configured DC coordinates (or one inactive point) and RF MHz."""
    dc = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc)
    return dc, sweep2array(cfg.frequency)


def _validate_counts(
    data: HeraldedSpectroscopyResult,
    dc: NDArray[np.float64],
    frequencies: NDArray[np.float64],
) -> None:
    """Reject mismatched scan axes, invalid counts or partially missing count rows."""
    if (
        not np.array_equal(data.dc_values, dc)
        or data.frequencies.shape != frequencies.shape
        or not np.allclose(
            data.frequencies, frequencies, rtol=8 * np.finfo(float).eps, atol=0
        )
        or data.sums.shape != (len(dc), 2, 2, len(frequencies), 3, 3, 3, 3)
    ):
        raise ValueError(
            "Heralded spectroscopy requires saved axes and sums (DC,2,2,frequency,3,3,3,3)"
        )
    finite = np.all(np.isfinite(data.sums) & (data.sums >= 0), axis=(-4, -3, -2, -1))
    missing = np.all(np.isnan(data.sums), axis=(-4, -3, -2, -1))
    if not np.all(finite | missing):
        raise ValueError(
            "Heralded spectroscopy contains invalid or partially missing counts"
        )


def _pool_by_herald(sums: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum preparation counts per g/e herald; leave entirely missing RF bins NaN.

    Move RF after the two herald axes, so each branch ends in (RF, t1, t2).
    """
    counts = sums[..., :2, :2, :, :]
    measured = np.isfinite(sums[..., 0, 0, 0, 0]).any(axis=(1, 2))
    pooled = np.nansum(counts, axis=(1, 2))
    pooled[~measured] = np.nan
    return np.moveaxis(pooled, 1, 3)


def _dc_display_unit(cfg: HeraldedSpectroscopyCfg) -> str:
    """Spell out the device-native unit in figure labels."""
    return "native unit" if cfg.dc_unit == "native" else cfg.dc_unit


def _title(cfg: HeraldedSpectroscopyCfg, bias: float) -> str:
    """Name the experiment and active bias without showing an inactive coordinate."""
    title = "Heralded conditional spectroscopy"
    if cfg.dc_device is not None:
        title += f" · {cfg.dc_device} bias {bias:.7g} {_dc_display_unit(cfg)}"
    return title


def _contrast(cfg: HeraldedSpectroscopyCfg) -> str:
    """Name the contrast pulse choice in figure titles."""
    return "Double Xπ contrast" if cfg.contrast else "No contrast pulses"


def _branch_figure(
    plots: Plots,
    name: str,
    frequencies: NDArray[np.float64] | None = None,
    *,
    ylabel: str,
    sharey: bool = False,
) -> Figure:
    """Create a four-herald figure, keeping the planned RF range when supplied."""
    fig, _ = plots.subplots(
        num=name, nrows=2, ncols=2, sharey=sharey, figsize=(12, 9), layout="constrained"
    )
    for ax, branch in zip(fig.axes, _HERALDS, strict=True):
        ax.set(title=f"Herald: {branch}", xlabel=_RF_LABEL, ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    if frequencies is not None:
        low, high = float(frequencies.min()), float(frequencies.max())
        margin = 0.05 * (high - low) if high > low else 1.0
        for ax in fig.axes:
            ax.set_xlim(low - margin, high + margin)
    return fig


def _live_spectra(
    plots: Plots,
    cfg: HeraldedSpectroscopyCfg,
    dc: NDArray[np.float64],
    frequencies: NDArray[np.float64],
) -> Callable[[NDArray[np.float64]], None]:
    """Show live spectra and return the plot-thread updater for counts and progress."""
    views = []
    for index, bias in enumerate(dc):
        fig = _branch_figure(
            plots, f"spectra_{index}", frequencies, ylabel="Terminal P(e)"
        )
        panels = [live_errorbar(ax, num_lines=2) for ax in fig.axes]
        title = f"{_title(cfg, float(bias))} · {_contrast(cfg)}"
        fig.suptitle(
            f"{title}\n0 acquired shots · 0/{4 * len(frequencies)} preparation-frequency points sampled"
        )
        for ax in fig.axes:
            ax.legend(ax.get_lines(), _QUBITS, loc="upper left", bbox_to_anchor=(1, 1))
        views.append((fig, panels, title))
        fig.show()

    @plots.on_plot_thread
    def update(sums: NDArray[np.float64]) -> tuple[Figure, ...]:
        pooled = _pool_by_herald(sums)
        result, sem = summarize_spectra(pooled), excitation_sem(pooled)
        for index, (fig, panels, title) in enumerate(views):
            for panel, (q1, q2) in zip(panels, np.ndindex(2, 2), strict=True):
                panel.set(
                    frequencies,
                    result.excitation[index, q1, q2].T,
                    sem[index, q1, q2].T,
                )
            measured = np.count_nonzero(np.isfinite(sums[index, ..., 0, 0, 0, 0]))
            shots = int(np.nansum(sums[index]))
            fig.suptitle(
                f"{title}\n{shots:,} acquired shots · "
                f"{measured}/{4 * len(frequencies)} preparation-frequency points sampled"
            )
        return tuple(fig for fig, _, _ in views)

    return update


def _plot_spectra(
    plots: Plots,
    index: int,
    title: str,
    frequencies: NDArray[np.float64],
    excitation: NDArray[np.float64],
    sem: NDArray[np.float64],
) -> None:
    """Draw terminal P(e) and shot SEM for both qubits in every herald panel."""
    fig = _branch_figure(plots, f"spectra_{index}", frequencies, ylabel="Terminal P(e)")
    fig.suptitle(title)
    for ax, (q1, q2) in zip(fig.axes, np.ndindex(2, 2), strict=True):
        for values, errors, label in zip(
            excitation[q1, q2].T, sem[q1, q2].T, _QUBITS, strict=True
        ):
            ax.errorbar(frequencies, values, yerr=errors, fmt=".-", label=label)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))


def _plot_coverage(
    plots: Plots,
    index: int,
    title: str,
    frequencies: NDArray[np.float64],
    shots: NDArray[np.float64],
) -> None:
    """Draw each herald's retained shot counts without filling unmeasured RF bins."""
    fig = _branch_figure(
        plots,
        f"coverage_{index}",
        frequencies,
        ylabel="Retained shots (count)",
        sharey=True,
    )
    fig.suptitle(f"{title} · Retained shots by herald")
    fig.axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax, (q1, q2) in zip(fig.axes, np.ndindex(2, 2), strict=True):
        ax.plot(frequencies, shots[q1, q2], ".-", color="0.3")


def _plot_herald_matrix(
    plots: Plots, index: int, title: str, sums: NDArray[np.float64]
) -> None:
    """Plot nine pooled herald fractions per preparation, marking undefined rows."""
    rates = joint_probabilities(np.nansum(sums, axis=(2, 5, 6))).reshape(4, 9)
    fig, _ = plots.subplots(
        num=f"herald_{index}",
        nrows=4,
        sharex=True,
        figsize=(12, 9),
        layout="constrained",
    )
    fig.suptitle(f"{title} · Herald class fractions")
    fig.supylabel("Fraction of acquired shots by nominal preparation (Qubit1, Qubit2)")
    positions = range(rates.shape[1])
    for ax, preparation, row in zip(fig.axes, _HERALDS, rates, strict=True):
        ax.set(ylabel=preparation, ylim=(0, 1.16), xlim=(-0.5, 8.5))
        ax.set_yticks((0, 0.5, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        if np.all(np.isnan(row)):
            ax.text(
                0.5, 0.5, "No shots", ha="center", va="center", transform=ax.transAxes
            )
            continue
        ax.bar(positions, row)
        for position, fraction in zip(positions, row, strict=True):
            ax.annotate(
                "0%" if fraction == 0 else f"{fraction:.1%}",
                (position, fraction),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
    fig.axes[-1].set_xticks(
        positions, [f"{q1}, {q2}" for q1 in _STATES for q2 in _STATES]
    )
    fig.axes[-1].set_xlabel("Herald class (Qubit1, Qubit2)")


def _plot_bias_maps(
    plots: Plots,
    cfg: HeraldedSpectroscopyCfg,
    data: HeraldedSpectroscopyResult,
    excitation: NDArray[np.float64],
) -> None:
    """Draw one RF × DC heatmap figure per qubit, sharing the scale across its branches."""
    for target, role in enumerate(_QUBITS):
        fig = _branch_figure(
            plots,
            f"map_{role.lower()}",
            ylabel=f"Coupler DC bias · {cfg.dc_device} ({_dc_display_unit(cfg)})",
        )
        fig.suptitle(
            f"Heralded conditional spectroscopy · {role} terminal P(e) map\n{_contrast(cfg)}"
        )
        values = np.ma.masked_invalid(excitation[..., target])
        norm = Normalize()
        norm.autoscale(values)
        for ax, (q1, q2) in zip(fig.axes, np.ndindex(2, 2), strict=True):
            ax.pcolormesh(
                data.frequencies,
                data.dc_values,
                values[:, q1, q2, :],
                shading="nearest",
                norm=norm,
                cmap="viridis",
            )
            ax.grid(False)
        fig.colorbar(
            fig.axes[0].collections[0], ax=fig.axes, label=f"{role} terminal P(e)"
        )
