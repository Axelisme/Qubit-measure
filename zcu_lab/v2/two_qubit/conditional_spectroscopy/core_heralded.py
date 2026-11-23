"""Two-line conditional spectra grouped by measured post-preparation heralds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from numpy.typing import NDArray
from zcu_tools.cfg_model import FieldMeta
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
from zcu_tools.program.v2 import Classify, Reset, shot_sums
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.device import set_flux

from .._sums import joint_probabilities
from ._common import SpectroscopyScanCfg, build_spectroscopy_program, excitation_sem
from .core import summarize_spectra


class HeraldedSpectroscopyCfg(SpectroscopyScanCfg[Reset.Cfg]):
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
    readout: Annotated[
        MuxReadout.Cfg,
        FieldMeta(
            label="Mux readout",
            description="Set the shared herald and terminal readout with GE Post radii and the wait after the herald pulse.",
        ),
    ]


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
        cfg_schema_version="1.0",
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
        dc, frequencies = _scan_axes(config)
        update = _live_spectra(context.plots, config, dc, frequencies)
        buffer = SignalBuffer(
            (len(dc), 2, 2, len(frequencies), 3, 3, 3, 3),
            dtype=np.float64,
            on_update=update,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            for dc_value, step in sched.scan("dc", dc.tolist()):
                if config.dc_device is not None:
                    set_flux(
                        context.devices[config.dc_device],
                        dc_value,
                        unit=config.dc_unit,
                        stop=context.scope.stop,
                    )
                prog = build_spectroscopy_program(
                    config, soccfg=context.soccfg, herald=True
                )
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
        del options
        cfg, data = source.cfg, source.result
        if cfg is None:
            raise ValueError("Heralded spectroscopy analysis requires saved cfg")
        dc, frequencies = _scan_axes(cfg)
        _validate_counts(data, dc, frequencies)
        pooled = _pool_by_herald(data.sums)
        spectra = summarize_spectra(pooled)
        sem = excitation_sem(pooled)
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
        if cfg.dc_device is not None and len(dc) > 1:
            _plot_bias_maps(plots, cfg, data, spectra.excitation)


_STATES = ("g", "e", "other")
_HERALDS = ("gg", "ge", "eg", "ee")
_QUBITS = ("Qubit1", "Qubit2")
_RF_LABEL = "Common probe RF frequency (MHz)"


def _scan_axes(
    cfg: HeraldedSpectroscopyCfg,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    dc = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc)
    return dc, sweep2array(cfg.frequency)


def _validate_counts(
    data: HeraldedSpectroscopyResult,
    dc: NDArray[np.float64],
    frequencies: NDArray[np.float64],
) -> None:
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
    # (DC, preparation1, preparation2, RF, h1, h2, t1, t2) → (DC, h1, h2, RF, t1, t2).
    # A missing preparation contributes no shots; an entirely unmeasured RF bin stays NaN.
    counts = sums[..., :2, :2, :, :]
    measured = np.isfinite(sums[..., 0, 0, 0, 0]).any(axis=(1, 2))
    pooled = np.nansum(counts, axis=(1, 2))
    pooled[~measured] = np.nan
    return np.moveaxis(pooled, 1, 3)


def _title(cfg: HeraldedSpectroscopyCfg, bias: float) -> str:
    title = "Heralded conditional spectroscopy"
    if cfg.dc_device is not None:
        unit = "native unit" if cfg.dc_unit == "native" else cfg.dc_unit
        title += f" · {cfg.dc_device} bias {bias:.7g} {unit}"
    return title


def _contrast(cfg: HeraldedSpectroscopyCfg) -> str:
    return "Double Xπ contrast" if cfg.contrast else "No contrast pulses"


def _branch_figure(
    plots: Plots,
    name: str,
    frequencies: NDArray[np.float64] | None = None,
    *,
    ylabel: str,
    sharey: bool = False,
) -> Figure:
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
    views = []
    for index, bias in enumerate(dc):
        fig = _branch_figure(
            plots, f"spectra_{index}", frequencies, ylabel="Terminal P(e)"
        )
        panels = [live_errorbar(ax, num_lines=2) for ax in fig.axes]
        title = f"{_title(cfg, float(bias))} · {_contrast(cfg)}"
        fig.suptitle(
            f"{title}\n0/{4 * len(frequencies)} preparation-frequency points sampled"
        )
        fig.legend(
            fig.axes[0].get_lines(), _QUBITS, loc="outside lower center", ncols=2
        )
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
            fig.suptitle(
                f"{title}\n{measured}/{4 * len(frequencies)} preparation-frequency points sampled"
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
    fig = _branch_figure(plots, f"spectra_{index}", frequencies, ylabel="Terminal P(e)")
    fig.suptitle(title)
    for ax, (q1, q2) in zip(fig.axes, np.ndindex(2, 2), strict=True):
        for values, errors, label in zip(
            excitation[q1, q2].T, sem[q1, q2].T, _QUBITS, strict=True
        ):
            ax.errorbar(frequencies, values, yerr=errors, fmt=".-", label=label)
    fig.legend(
        *fig.axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2
    )


def _plot_coverage(
    plots: Plots,
    index: int,
    title: str,
    frequencies: NDArray[np.float64],
    shots: NDArray[np.float64],
) -> None:
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
    # Pool shot counts over terminal class and RF before normalizing each preparation row.
    rates = joint_probabilities(np.nansum(sums, axis=(2, 5, 6))).reshape(4, 9)
    fig, ax = plots.subplots(
        num=f"herald_{index}", figsize=(12, 9), layout="constrained"
    )
    fig.suptitle(f"{title} · Herald class fractions")
    mesh = ax.imshow(
        np.ma.masked_invalid(rates), vmin=0, vmax=1, cmap="Blues", aspect="auto"
    )
    ax.set_xticks(range(9), [f"{q1}, {q2}" for q1 in _STATES for q2 in _STATES])
    ax.set_yticks(range(4), _HERALDS)
    ax.set(
        xlabel="Herald class (Qubit1, Qubit2)",
        ylabel="Nominal preparation (Qubit1, Qubit2)",
    )
    ax.grid(False)
    ax.set_facecolor("0.9")
    for row, col in np.ndindex(rates.shape):
        probability = rates[row, col]
        if np.isfinite(probability):
            ax.text(
                col,
                row,
                f"{probability:.1%}",
                ha="center",
                va="center",
                color="white" if probability > 0.5 else "black",
            )
    fig.colorbar(
        mesh, ax=ax, label="Fraction of acquired shots", format=PercentFormatter(xmax=1)
    )


def _plot_bias_maps(
    plots: Plots,
    cfg: HeraldedSpectroscopyCfg,
    data: HeraldedSpectroscopyResult,
    excitation: NDArray[np.float64],
) -> None:
    unit = "native unit" if cfg.dc_unit == "native" else cfg.dc_unit
    for target, role in enumerate(_QUBITS):
        fig = _branch_figure(
            plots,
            f"map_{role.lower()}",
            ylabel=f"Coupler DC bias · {cfg.dc_device} ({unit})",
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
