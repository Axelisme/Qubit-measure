"""Hardware-swept absolute-frequency survey and multi-resonator dip analysis."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, ClassVar

from pydantic import Field, field_validator

from pydantic.dataclasses import dataclass as pydantic_dataclass

import numpy as np
from zcu_tools.analysis.dips import DipFinderOptions, find_spectrum_dips
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseReadout, Reset, SweepCfg, TimelineBuilder

from zcu_lab.v2.onetone._shared import (
    FreqResult as RangeFindResult,
)
from zcu_lab.v2.onetone._shared import (
    linear_freqs,
    spectrum_buffer,
)


class RangeFindCfg(ConfigBase):
    """Pulse-readout acquisition for a wide absolute-frequency survey."""

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="填入 resonator readout 設定。先用 lookback 取得 TOF，再以 onetone 取得 kappa，確認 trigger offset 與 waveform duration，單位 us。",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="填入 absolute MHz 起訖值。range_find 不會從已知中心自動設定掃描窗。",
        ),
    ]
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


DEFAULT_MIN_SEPARATION_MHZ = 1.0


@pydantic_dataclass
class RangeFindOptions:
    """Choose frequency-ordered names and minimum spacing for coarse dip detection.

    resonators is a unique list of nonblank names, or empty to discover all dips.
    Names map to selected dips in ascending MHz order, not to a readout label.
    min_separation is finite positive MHz, default 1 MHz. smoothing_sigma is
    nonnegative Gaussian sigma in samples, default zero (no smoothing).
    prominence_factor is the nonnegative noise multiplier, default five.
    max_fwhm is finite positive MHz, default 25. Selection ranks depth / FWHM.
    Validation occurs on construction, before analysis or picker setup.
    """

    resonators: Annotated[list[str], Field(strict=True)] = field(default_factory=list)
    min_separation: Annotated[
        float, Field(gt=0, strict=True, allow_inf_nan=False)
    ] = DEFAULT_MIN_SEPARATION_MHZ
    smoothing_sigma: Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)] = 0.0
    prominence_factor: Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)] = 5.0
    max_fwhm: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)] = 25.0

    @field_validator("resonators")
    @classmethod
    def unique_nonblank_names(cls, names: list[str]) -> list[str]:
        """Keep names in input order, rejecting blank or duplicate labels."""
        if any(not name.strip() for name in names):
            raise ValueError("resonators must contain nonblank strings")
        if len(set(names)) != len(names):
            raise ValueError("resonators must contain unique names")
        return names

    def to_dip_options(self) -> DipFinderOptions:
        """Validate names and return pure numerical controls for this selection.

        The name count limits selected dips; empty names discover all candidates.
        Invalid names or controls raise ValueError.
        """
        return DipFinderOptions(
            smoothing_sigma=self.smoothing_sigma,
            prominence_factor=self.prominence_factor,
            max_fwhm=self.max_fwhm,
            min_separation=self.min_separation,
            max_dips=len(self.resonators) if self.resonators else None,
        )


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
        cfg = deepcopy(config)
        freqs = linear_freqs(cfg.freq, cfg.readout, context=context)
        signals = spectrum_buffer(freqs, plots=context.plots)
        with Schedule(signals, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                relax_delay=cfg.relax_delay,
            )
            freq = builder.add_sweep("freq", cfg.freq)
            reset = Reset("reset", cfg.reset)
            if reset is not None:
                builder.emit(reset)
            builder.emit(PulseReadout("readout", cfg.readout, freq=freq))
            sched.run_acquire(builder.build(), context.soc, rounds=cfg.rounds)

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

        fig, ax = plots.subplots(nrows=1, num="fit", figsize=(9, 4.5))

        @plots.on_plot_thread
        def draw(_):
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
            fig.tight_layout()
            return (fig,)

        draw(None)
        return dips
