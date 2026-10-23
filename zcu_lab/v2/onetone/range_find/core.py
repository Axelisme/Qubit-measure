"""Hardware-swept absolute-frequency survey and multi-resonator dip analysis."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from zcu_tools.experiment import MHZ_TO_HZ, AxesSpec, Axis, PersistableExperiment, ZSpec
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
    min_separation is finite positive MHz. The default 1 MHz separates typical
    MHz readout linewidths; increase it for broader resonators. Validation occurs
    on Analyze, allowing an empty initial GUI list to be edited before use.
    """

    resonators: list[str] = field(default_factory=list)
    min_separation: float = DEFAULT_MIN_SEPARATION_MHZ

    def validate(self) -> None:
        """Reject blank/duplicate/non-string names and nonpositive/nonfinite MHz."""
        _validate_options(self.resonators, self.min_separation)


def _validate_options(resonators: object, min_separation: object) -> None:
    # GUI editors and notebook callers can supply values outside annotations.
    if not isinstance(resonators, list) or any(
        not isinstance(name, str) or not name.strip() for name in resonators
    ):
        raise ValueError("resonators must be a list of nonblank strings")
    if len(set(resonators)) != len(resonators):
        raise ValueError("resonators must contain unique names")
    if (
        isinstance(min_separation, bool)
        or not isinstance(min_separation, (int, float))
        or not math.isfinite(min_separation)
        or min_separation <= 0
    ):
        raise ValueError("min_separation must be finite and > 0 MHz")


@dataclass(frozen=True)
class ResonatorDip:
    """One dip: freq is absolute MHz; depth is magnitude prominence in a.u.

    name is None during discovery, or comes from Options in frequency order.
    Names never select a readout channel. depth uses the background-subtracted,
    Gaussian-smoothed magnitude, not a fit.
    """

    name: str | None
    freq: float
    depth: float


class RangeFindExp(PersistableExperiment[RangeFindResult, RangeFindCfg]):
    """Acquire absolute MHz spectra and find optionally named dips."""

    Options = RangeFindOptions

    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=RangeFindResult,
        cfg_type=RangeFindCfg,
        cfg_schema_version="1.0",
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
            modules = cfg.modules
            freq_sweep = cfg.sweep.freq
            modules.readout.set_param("freq", sweep2param("freq", freq_sweep))
            _ = (
                sched.prog_builder(context.soc, context.soccfg)
                .add_reset("reset", modules.reset)
                .add(PulseReadout("readout", modules.readout))
                .declare_sweep("freq", freq_sweep)
                .build_and_acquire()
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
        (at least two per edge). Gaussian smoothing uses sigma=1 sample. Prominence
        must exceed five adjacent-difference noise estimates (MAD/0.67449/sqrt(2)),
        with a 1e-6 magnitude-range floor. Greedy descending prominence enforces
        min_separation in actual MHz; frequency-sorted dips receive the given names.
        Empty names select all separated candidates without assigning names.
        Edge minima are excluded. depth is prominence in a.u., not linewidth.

        plots receives one measured-magnitude curve with red stars at the dips,
        annotated by name or discovery index, MHz frequency and depth.
        Return all discovered dips, or all requested named dips. Insufficient
        named dips raise ValueError naming the available count;
        no partial success or proposals are produced here.
        """
        options.validate()
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

        # Edge medians leave the interior dips out of the linear background estimate.
        edge = max(2, freqs.size // 10)
        x0, x1 = float(np.median(freqs[:edge])), float(np.median(freqs[-edge:]))
        y0, y1 = float(np.median(magnitude[:edge])), float(np.median(magnitude[-edge:]))
        background = y0 + (y1 - y0) * (freqs - x0) / (x1 - x0)
        corrected = magnitude - background
        difference = np.diff(corrected)
        noise = float(np.median(np.abs(difference - np.median(difference)))) / (
            0.67449 * np.sqrt(2)
        )
        threshold = max(5 * noise, 1e-6 * float(np.ptp(magnitude)), np.finfo(float).eps)
        smoothed = gaussian_filter1d(corrected, sigma=1.0)
        candidates, properties = find_peaks(-smoothed, prominence=threshold)
        prominences = properties["prominences"]

        # Enforce actual MHz separation rather than distance in samples.
        selected: list[int] = []
        for candidate in np.argsort(-prominences, kind="stable"):
            index = int(candidates[candidate])
            if all(
                abs(freqs[index] - freqs[other]) >= options.min_separation
                for other in selected
            ):
                selected.append(index)
                if options.resonators and len(selected) == len(options.resonators):
                    break
        if len(selected) < len(options.resonators):
            raise ValueError(
                f"Found {len(selected)} significant separated dips; need {len(options.resonators)} "
                f"(min_separation={options.min_separation:g} MHz, prominence >= {threshold:.6g} a.u.). "
                "Check scan range, resolution, power and requested names."
            )
        depth_by_index = {
            int(index): float(depth)
            for index, depth in zip(candidates, prominences, strict=True)
        }
        dips = tuple(
            ResonatorDip(
                options.resonators[position] if options.resonators else None,
                float(freqs[index]),
                depth_by_index[index],
            )
            for position, index in enumerate(sorted(selected))
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
                    label=f"{label}: {dip.freq:.6g} MHz, depth {dip.depth:.4g}",
                )
                ax.annotate(
                    label,
                    (dip.freq, measured),
                    xytext=(8, 8),
                    textcoords="offset points",
                )
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Magnitude (a.u.)")
            ax.legend()
            figure.tight_layout()

        plots.draw("fit", draw)
        return dips
