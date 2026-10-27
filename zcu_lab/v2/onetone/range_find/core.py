"""Hardware-swept absolute-frequency survey and multi-resonator dip analysis."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from matplotlib.figure import Figure
from zcu_tools.analysis.dips import DipFinderOptions, find_spectrum_dips
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseReadout, sweep2param

from zcu_lab.v2.onetone._shared import (
    FreqResult as RangeFindResult,
)
from zcu_lab.v2.onetone._shared import (
    LinearFreqCfg,
    linear_freqs,
    spectrum_buffer,
)


class RangeFindCfg(LinearFreqCfg):
    """Pulse-readout acquisition for a wide absolute-frequency survey.

    Fields match onetone/freq linear. Dip names belong to Analyze options,
    not to acquisition or the saved cfg.
    """


DEFAULT_MIN_SEPARATION_MHZ = 1.0


@dataclass
class RangeFindOptions:
    """Choose frequency-ordered names and minimum spacing for coarse dip detection.

    resonators is a unique list of nonblank names, or empty to discover all dips.
    Names map to selected dips in ascending MHz order, not to a readout label.
    min_separation is finite positive MHz, default 1 MHz. smoothing_sigma is
    nonnegative Gaussian sigma in samples, default zero (no smoothing).
    prominence_factor is the nonnegative noise multiplier, default five.
    max_fwhm is finite positive MHz, default 25. Selection ranks depth / FWHM.
    Validation occurs on Analyze, allowing initial GUI values to be edited.
    """

    resonators: list[str] = field(default_factory=list)
    min_separation: float = DEFAULT_MIN_SEPARATION_MHZ
    smoothing_sigma: float = 0.0
    prominence_factor: float = 5.0
    max_fwhm: float = 25.0

    def validate(self) -> None:
        """Reject invalid names or numerical controls with ValueError."""
        self.to_dip_options()

    def to_dip_options(self) -> DipFinderOptions:
        """Validate names and return pure numerical controls for this selection.

        The name count limits selected dips; empty names discover all candidates.
        Invalid names or controls raise ValueError.
        """
        _validate_resonators(self.resonators)
        return DipFinderOptions(
            smoothing_sigma=self.smoothing_sigma,
            prominence_factor=self.prominence_factor,
            max_fwhm=self.max_fwhm,
            min_separation=self.min_separation,
            max_dips=len(self.resonators) if self.resonators else None,
        )


def _validate_resonators(resonators: object) -> None:
    # GUI editors and notebook callers can supply values outside annotations.
    if not isinstance(resonators, list) or any(
        not isinstance(name, str) or not name.strip() for name in resonators
    ):
        raise ValueError("resonators must be a list of nonblank strings")
    if len(set(resonators)) != len(resonators):
        raise ValueError("resonators must contain unique names")


@dataclass(frozen=True)
class ResonatorDip:
    """One dip: freq is absolute MHz; depth is magnitude prominence in a.u.

    name is None during discovery, or comes from Options in frequency order.
    Names never select a readout channel. depth uses the background-subtracted,
    optionally Gaussian-smoothed magnitude, not a fit. fwhm is half-prominence
    width in MHz, at least one local mean frequency step. score is depth / fwhm
    in a.u./MHz; neither width nor score is a fitted resonator calibration.
    """

    name: str | None
    freq: float
    depth: float
    fwhm: float
    score: float


class RangeFindExp(Experiment[RangeFindResult, RangeFindCfg]):
    """Acquire absolute MHz spectra and find optionally named dips."""

    Options = RangeFindOptions

    AXES_SPEC: ClassVar[AxesSpec[RangeFindResult, RangeFindCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=RangeFindResult,
        cfg_type=RangeFindCfg,
        cfg_schema_version="3.0",
        tag="onetone/range_find",
    )

    def run(self, config: RangeFindCfg, *, context: RunContext) -> RangeFindResult:
        """Reset then pulse/read out each FPGA-swept frequency.

        config supplies the same acquisition fields as onetone/freq linear.
        context supplies SoC, cancellation and live plots. Return absolute MHz
        and complex IQ; failures propagate and cancellation preserves partial IQ.
        """
        cfg = deepcopy(config)
        freqs = linear_freqs(cfg, context=context)
        signals = spectrum_buffer(freqs, plots=context.plots)
        with Schedule(cfg, signals, scope=context.scope) as sched:
            cfg = sched.cfg
            freq_sweep = cfg.freq
            cfg.readout.set_param("freq", sweep2param("freq", freq_sweep))
            _ = (
                sched.prog_builder(
                    context.soc,
                    context.soccfg,
                    reps=sched.cfg.reps,
                    relax_delay=sched.cfg.relax_delay,
                )
                .add_reset("reset", cfg.reset)
                .add(PulseReadout("readout", cfg.readout))
                .declare_sweep("freq", freq_sweep)
                .build_and_acquire(rounds=cfg.rounds)
            )
        return RangeFindResult(freqs=freqs, signals=signals.array)

    def analyze(
        self,
        source: RunRecord[RangeFindCfg, RangeFindResult],
        options: RangeFindOptions,
        *,
        plots: Plots,
    ) -> tuple[ResonatorDip, ...]:
        """Remove linear magnitude background and select prominent interior dips.

        source supplies matching (P,) absolute MHz and complex IQ arrays, P >= 11;
        cfg may be None. Partial/nonfinite scans and duplicate frequencies raise
        ValueError. Descending scans are sorted. No current Form or capture is read.

        Background joins median magnitude/frequency in each outer 10% of points
        (at least two per edge). The shared numerical dip finder defaults to no
        smoothing, prominence_factor=5 and max_fwhm=25 MHz. It rejects wider dips,
        ranks depth / FWHM and enforces min_separation in actual MHz. Selected
        dips receive the given names in frequency order. Empty names select all
        separated candidates without assigning names. Edge minima are excluded.

        plots receives one measured-magnitude curve with red stars at the dips,
        annotated by name or discovery index, MHz frequency, depth, FWHM and score.
        Return all discovered dips, or all requested named dips. Insufficient
        named dips raise ValueError naming the available count;
        no partial success or proposals are produced here.
        """
        dip_options = options.to_dip_options()
        freqs = np.asarray(source.result.freqs, dtype=np.float64)
        signals = np.asarray(source.result.signals, dtype=np.complex128)
        if freqs.ndim != 1 or signals.shape != freqs.shape or freqs.size < 11:
            raise ValueError(
                "Analysis needs matching (P,) frequency/IQ arrays with P >= 11"
            )
        if not np.all(np.isfinite(freqs)) or not np.all(np.isfinite(signals)):
            raise ValueError(
                "Analysis needs a complete finite scan; partial/NaN points remain"
            )
        order = np.argsort(freqs)
        freqs, magnitude = freqs[order], np.abs(signals[order])
        if np.any(np.diff(freqs) <= 0):
            raise ValueError("Analysis needs distinct absolute frequencies")

        selected = find_spectrum_dips(freqs, magnitude, dip_options)
        if len(selected) < len(options.resonators):
            raise ValueError(
                f"Found {len(selected)} significant separated dips; need {len(options.resonators)} "
                f"(min_separation={options.min_separation:g} MHz, max_fwhm={options.max_fwhm:g} MHz, "
                f"prominence_factor={options.prominence_factor:g}). "
                "Check scan range, resolution, power and requested names."
            )
        dips = tuple(
            ResonatorDip(
                options.resonators[position] if options.resonators else None,
                dip.freq,
                dip.depth,
                dip.fwhm,
                dip.score,
            )
            for position, dip in enumerate(sorted(selected, key=lambda dip: dip.freq))
        )

        plots.subplots("fit", nrows=1, figsize=(9, 4.5))

        def draw(figure: Figure) -> None:
            (ax,) = figure.axes
            ax.plot(freqs, magnitude, label="Measured magnitude")
            for position, dip in enumerate(dips, start=1):
                label = dip.name if dip.name is not None else f"Dip {position}"
                measured = magnitude[np.searchsorted(freqs, dip.freq)]
                ax.plot(
                    dip.freq,
                    measured,
                    marker="*",
                    color="red",
                    markersize=14,
                    linestyle="none",
                    label=(
                        f"{label}: {dip.freq:.6g} MHz, depth {dip.depth:.4g}, "
                        f"FWHM {dip.fwhm:.4g} MHz, score {dip.score:.4g} a.u./MHz"
                    ),
                )
                ax.annotate(
                    label,
                    (dip.freq, measured),
                    xytext=(8, 8),
                    textcoords="offset points",
                )
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Magnitude (a.u.)")
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize="small")
            figure.tight_layout()

        plots.draw("fit", draw)
        return dips
