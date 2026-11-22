"""Hardware-swept absolute-frequency survey and multi-resonator dip analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.dips import (
    DipFinderOptions,
    estimate_prominence_factor,
    find_spectrum_dips,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseReadout, Reset, SweepCfg, TimelineBuilder

from zcu_lab.v2.onetone._shared import (
    FreqResult as RangeFindResult,
)
from zcu_lab.v2.onetone._shared import (
    linear_freqs,
)
from zcu_lab.v2.support import live_line

DEFAULT_MIN_SEPARATION_MHZ = 1.0


@pydantic_dataclass
class RangeFindOptions:
    """Choose frequency-ordered names and minimum spacing for coarse dip detection.

    resonators requires at least one unique nonblank name; its length is the
    expected dip count. Use temporary names for unidentified dips.
    Names map to selected dips in ascending MHz order, not to a readout label.
    min_separation is finite positive MHz, default 1 MHz. smoothing_sigma is
    nonnegative Gaussian sigma in samples, default zero (no smoothing).
    prominence_factor=None estimates the noise multiplier for the name count;
    a finite nonnegative number uses that multiplier unchanged.
    max_fwhm is finite positive MHz, default 25. Selection ranks depth / FWHM.
    Validation occurs on construction, before analysis or picker setup.
    """

    resonators: Annotated[list[str], Field(strict=True)]
    min_separation: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)] = (
        DEFAULT_MIN_SEPARATION_MHZ
    )
    smoothing_sigma: Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)] = (
        0.0
    )
    prominence_factor: Annotated[
        float | None, Field(ge=0, strict=True, allow_inf_nan=False)
    ] = None
    max_fwhm: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)] = 25.0

    @field_validator("resonators")
    @classmethod
    def unique_nonblank_names(cls, names: list[str]) -> list[str]:
        """Keep names in input order, rejecting blank or duplicate labels."""
        if not names:
            raise ValueError("resonators must contain at least one name")
        if any(not name.strip() for name in names):
            raise ValueError("resonators must contain nonblank strings")
        if len(set(names)) != len(names):
            raise ValueError("resonators must contain unique names")
        return names


@dataclass(frozen=True)
class ResonatorDip:
    """One dip: freq is absolute MHz; depth is magnitude prominence in a.u.

    name comes from Options in frequency order.
    Names never select a readout channel. depth uses the background-subtracted,
    optionally Gaussian-smoothed magnitude, not a fit. fwhm is half-prominence
    width in MHz, at least one local mean frequency step. score is depth / fwhm
    in a.u./MHz; neither width nor score is a fitted resonator calibration.
    """

    name: str
    freq: float
    depth: float
    fwhm: float
    score: float


@dataclass(frozen=True)
class RangeFindDetection:
    """Frequency-ordered named dips and the actual prominence noise multiplier."""

    dips: tuple[ResonatorDip, ...]
    prominence_factor: float


class RangeFindCfg(ConfigBase):
    """Pulse-readout acquisition for a wide absolute-frequency survey."""

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Set the pulse and ADC window used together at each survey frequency.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Readout RF frequency",
            unit="MHz",
            description="Enter absolute readout RF endpoints in MHz covering the expected notches.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per frequency in each averaging round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter how many times to repeat and average the entire survey.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the relaxation wait in µs after each shot.",
        ),
    ]


class RangeFindExp(Experiment[RangeFindResult, RangeFindCfg]):
    """Acquire absolute MHz spectra and find the requested named dips."""

    Options = RangeFindOptions

    AXES_SPEC: ClassVar[AxesSpec[RangeFindResult, RangeFindCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
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
        # 1. Resolve the wide-band resonator survey on the readout frequency grid.
        cfg = deepcopy(config)
        freqs = linear_freqs(cfg.freq, cfg.readout, context=context)
        # 2. Preview magnitude dips while preserving complex IQ for dip detection.
        fig, ax = context.plots.subplots(num="measurement", layout="constrained")
        fig.suptitle("One-tone survey · Resonators · IQ magnitude")
        ax.set(
            xlabel="Readout RF frequency (MHz)", ylabel="Magnitude of mean IQ (a.u.)"
        )
        progress = ax.set_title(f"Averaging round 0 / {cfg.rounds}", fontsize=11)
        line = live_line(ax)
        signals = SignalBuffer((len(freqs),))
        rounds_done = 0

        @context.plots.on_plot_thread
        def update(completed: int) -> Iterable[Artist]:
            """Plot the cumulative mean and nominal shot count at a completed round."""
            progress.set_text(
                f"Averaging round {completed} / {cfg.rounds} · "
                f"{completed * cfg.reps:,} nominal shots/point"
            )
            return (*line.set(freqs, np.abs(signals.array)), progress)

        def show_round() -> bool:
            """Observe completed rounds without asking the acquisition to stop."""
            nonlocal rounds_done
            rounds_done += 1
            update(rounds_done)
            return False

        fig.show()
        # 3. Reset and read out each board-swept survey frequency.
        with Schedule(signals, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                relax_delay=cfg.relax_delay,
            )
            freq = builder.add_sweep("freq", cfg.freq)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(PulseReadout("readout", cfg.readout, freq=freq))
            sched.run_acquire(
                builder.build(),
                context.soc,
                rounds=cfg.rounds,
                options=RunOptions(stop_condition=show_round),
            )

        return RangeFindResult(freqs=freqs, signals=signals.array)

    def analyze(
        self,
        source: RunRecord[RangeFindCfg, RangeFindResult],
        options: RangeFindOptions,
        *,
        plots: Plots,
    ) -> RangeFindDetection:
        """Remove linear magnitude background and select prominent interior dips.

        source supplies matching (P,) absolute MHz and complex IQ arrays, P >= 11;
        cfg may be None. Partial/nonfinite scans and duplicate frequencies raise
        ValueError. Descending scans are sorted. No current Form or capture is read.

        Background joins median magnitude/frequency in each outer 10% of points
        (at least two per edge). The shared numerical dip finder defaults to no
        smoothing and max_fwhm=25 MHz. Automatic prominence_factor uses the name
        count; a number uses that factor unchanged. It rejects wider dips,
        ranks depth / FWHM and enforces min_separation in actual MHz. Selected
        dips receive the given names in frequency order. Edge minima are excluded.

        plots receives one measured-magnitude curve with selected dip stars
        annotated by name and measured MHz frequency.
        Return requested named dips and the actual factor. Insufficient
        named dips raise ValueError naming the available count;
        no partial success or proposals are produced here.
        """
        # 1. Require a complete, distinct frequency grid before ranking resonator dips.
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

        # 2. Resolve the requested dip count and its prominence threshold.
        dip_options = DipFinderOptions(
            smoothing_sigma=options.smoothing_sigma,
            prominence_factor=0.0
            if options.prominence_factor is None
            else options.prominence_factor,
            max_fwhm=options.max_fwhm,
            min_separation=options.min_separation,
            max_dips=len(options.resonators),
        )
        if options.prominence_factor is None:
            dip_options = replace(
                dip_options,
                prominence_factor=estimate_prominence_factor(
                    freqs, magnitude, dip_options, expected_dips=len(options.resonators)
                ),
            )
        selected = find_spectrum_dips(freqs, magnitude, dip_options)
        if len(selected) < len(options.resonators):
            raise ValueError(
                f"Found {len(selected)} significant separated dips; need {len(options.resonators)} "
                f"(min_separation={options.min_separation:g} MHz, max_fwhm={options.max_fwhm:g} MHz, "
                f"prominence_factor={dip_options.prominence_factor:g}). "
                "Check scan range, resolution, power and requested names."
            )
        dips = tuple(
            ResonatorDip(
                options.resonators[position],
                dip.freq,
                dip.depth,
                dip.fwhm,
                dip.score,
            )
            for position, dip in enumerate(sorted(selected, key=lambda dip: dip.freq))
        )

        self.plot_dips(
            freqs,
            magnitude,
            [(dip.name, dip.freq) for dip in dips],
            candidates=(),
            prominence_factor=dip_options.prominence_factor,
            plots=plots,
        )
        return RangeFindDetection(dips, dip_options.prominence_factor)

    @staticmethod
    def plot_dips(
        freqs: NDArray[np.float64],
        magnitude: NDArray[np.float64],
        selected: Sequence[tuple[str | None, float]],
        *,
        candidates: Sequence[float],
        prominence_factor: float,
        plots: Plots,
    ) -> None:
        """Plot sorted MHz samples, candidate frequencies and accepted (name, MHz) pairs.

        None denotes an unassigned selection. Use the magnitude of the original
        complex means, not the detection algorithm's smoothed data. The noise
        factor is the actual dimensionless multiplier used to find candidates.
        """
        fig, ax = plots.subplots(num="fit", layout="constrained")
        fig.suptitle("One-tone survey · Resonators · IQ magnitude")
        ax.set(
            xlabel="Readout RF frequency (MHz)",
            ylabel="Magnitude of mean IQ (a.u.)",
            title=f"Prominence noise factor: {prominence_factor:.4g}",
        )
        ax.plot(freqs, magnitude, ".-", color="C0", label="Measured samples")
        if candidates:
            ax.plot(
                candidates,
                np.interp(candidates, freqs, magnitude),
                "o",
                color="0.6",
                label="Automatic candidates",
            )
        frequencies = [freq for _, freq in selected]
        heights = np.interp(frequencies, freqs, magnitude)
        ax.plot(
            frequencies,
            heights,
            "*",
            color="crimson",
            markersize=11,
            label="Selected dips",
        )
        for row, ((name, freq), y) in enumerate(zip(selected, heights, strict=True), 1):
            ax.annotate(
                f"{name or 'Unassigned'}: {freq:.6f} MHz",
                (freq, y),
                xytext=(freq, 0.1 + 0.5 * row / (len(selected) + 1)),
                textcoords=("data", "axes fraction"),
                ha="right" if freq > (freqs[0] + freqs[-1]) / 2 else "left",
                fontsize=10,
                arrowprops={"arrowstyle": "-", "color": "0.35"},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3, fontsize=10)
        fig.show()
