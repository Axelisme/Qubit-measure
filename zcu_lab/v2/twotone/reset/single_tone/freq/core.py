from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import fit_qubit_freq
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
    sweep2param,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class FreqResult:
    """Reset frequencies in MHz and mean complex IQ (frequency,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class FreqCfg(ConfigBase):
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
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[PulseReset.Cfg, FieldMeta(label="Tested reset")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]


@dataclass(frozen=True)
class FreqAnalysis:
    freq: float
    fwhm: float


class FreqExp(Experiment[FreqResult, FreqCfg]):
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),),
        z=AxesSpec.Z("signals", "Signal", "a.u.", dtype=np.complex128),
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
        # Quantize reset frequency on the tested pulse's generator.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        reset_cfg = cfg.tested_reset
        freqs = sweep2array(
            cfg.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse_cfg.ch},
        )

        # Retain complex IQ and display its projection along the response axis.
        viewer = live_line(context.plots, "measurement", "Frequency (MHz)", "Amplitude")
        signals_buffer = SignalBuffer(
            (len(freqs),),
            on_update=lambda data: viewer.update(freqs, reset_signal2real(data)),
        )

        # Sweep the tested reset frequency after the optional preparation.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            sched.cfg.tested_reset.set_param(
                "freq", sweep2param("freq", sched.cfg.freq)
            )
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", sched.cfg.reset),
                    Pulse("init_pulse", sched.cfg.init_pulse),
                    PulseReset("tested_reset", sched.cfg.tested_reset),
                    Readout("readout", sched.cfg.readout),
                )
                .declare_sweep("freq", sched.cfg.freq)
                .build_and_acquire(rounds=cfg.rounds)
            )

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
        del options
        result = source.result

        freqs, signals = result.freqs, result.signals

        # Discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        freqs = freqs[val_mask]
        signals = signals[val_mask]

        real_signals = reset_signal2real(signals)

        freq, freq_err, fwhm, _, y_fit, _ = fit_qubit_freq(
            freqs, real_signals, type="lor"
        )

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.plot(freqs, real_signals, label="signal", marker="o", markersize=3)
        ax.plot(freqs, y_fit, label=f"fit, FWHM = {fwhm:.1g} MHz")
        label = f"f_reset = {freq:.5g} ± {freq_err:.1g} MHz"
        ax.axvline(freq, color="r", ls="--", label=label)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)
        ax.set_title("Sideband Reset Frequency Sweep")

        fig.tight_layout()

        return FreqAnalysis(freq=float(freq), fwhm=float(fwhm))
