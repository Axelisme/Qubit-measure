from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import fit_qubit_freq
from zcu_tools.analysis.fitting.base import lorfunc
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    PulseReset,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line

_TITLE = "Single-tone reset resonance · qubit · IQ projection"
_XLABEL = "Reset drive RF frequency (MHz)"
_YLABEL = "Mean IQ on principal axis (a.u.)"


@dataclass(frozen=True)
class FreqResult:
    """Reset frequencies in MHz and mean complex IQ (frequency,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class FreqAnalysis:
    freq: float
    fwhm: float


def reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    return rotate2real(signals).real


class FreqCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a preparation reset before each shot, or leave blank to skip it.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Preparation pulse",
            description="Select the pulse after preparation reset, such as calibrated Xπ, or leave blank to skip it.",
        ),
    ] = None
    tested_reset: Annotated[
        PulseReset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Fill in the fixed-gain sideband pulse applied between preparation and readout.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Fill in the calibrated resonator readout for the end of each shot.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Reset drive RF frequency",
            unit="MHz",
            description="Enter the RF frequency range of the tested sideband drive (MHz).",
        ),
    ]

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per sweep point and sequence branch in each acquisition round (count).",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds to average (count).",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the delay after each shot to let the qubit relax (µs).",
        ),
    ]


class FreqExp(Experiment[FreqResult, FreqCfg]):
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/single_tone/freq",
    )

    def run(self, cfg: FreqCfg, *, context: RunContext) -> FreqResult:
        """Measure the single-tone reset resonance versus reset frequency.

        Apply optional reset and initialization, the frequency-swept tested
        reset pulse, then read out. Return frequencies in MHz and mean
        complex IQ shaped (Nfreq,). The projected response supports a
        resonance fit; it is not a calibrated reset fidelity.
        """
        # 1. Quantize reset frequency on the tested pulse's generator.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        reset_cfg = cfg.tested_reset
        freqs = sweep2array(
            cfg.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse_cfg.ch},
        )

        # 2. Retain complex IQ and display its projection along the response axis.
        plots = context.plots
        fig, ax = plots.subplots(num="measurement")
        ax.set(xlabel=_XLABEL, ylabel=_YLABEL, title=_TITLE)
        viewer = live_line(ax)
        ax.lines[0].set(marker=".", markersize=5)

        @plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            """Refresh the measurement artists from the current acquired data."""
            return viewer.set(freqs, reset_signal2real(data))

        fig.show()
        signals_buffer = SignalBuffer((len(freqs),), on_update=update)

        # 3. Resolve the prepared state's reset response versus sideband frequency.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            freq = builder.add_sweep("freq", cfg.freq)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(PulseReset("tested_reset", cfg.tested_reset, freq=freq))
            builder.emit(Readout("readout", cfg.readout))
            sched.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return FreqResult(freqs, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: None,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        """Fit a Lorentzian to the projected reset-frequency response.

        Omit unmeasured NaN points, return the resonance center and FWHM
        in MHz, and publish fit. source.cfg may be None.
        """
        # 1. Estimate the reset resonance and linewidth from measured frequency points.
        del options
        result = source.result

        freqs, signals = result.freqs, result.signals

        # Discard NaNs left by a possible early abort.
        val_mask = ~np.isnan(signals)
        freqs = freqs[val_mask]
        signals = signals[val_mask]

        real_signals = reset_signal2real(signals)

        freq, freq_err, fwhm, fwhm_err, _, (parameters, _) = fit_qubit_freq(
            freqs, real_signals, type="lor"
        )

        # 2. Assess the Lorentzian reset resonance and its frequency uncertainty.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        dense = np.linspace(freqs.min(), freqs.max(), 200)
        local = np.linspace(
            max(freqs.min(), freq - 2 * fwhm), min(freqs.max(), freq + 2 * fwhm), 200
        )
        dense = np.unique(np.concatenate((dense, local)))
        ax.plot(freqs, real_signals, ".", label="Measured", markersize=5)
        ax.plot(dense, lorfunc(dense, *parameters), "-", label="Lorentzian fit")
        ax.axvline(
            freq,
            color="C3",
            ls="--",
            label=f"Reset RF = {freq:.3f} ± {freq_err:.2g} MHz\nFWHM = {fwhm:.3g} ± {fwhm_err:.2g} MHz",
        )
        ax.set(xlabel=_XLABEL, ylabel=_YLABEL, title=_TITLE)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)

        fig.tight_layout()

        return FreqAnalysis(freq=float(freq), fwhm=float(fwhm))
