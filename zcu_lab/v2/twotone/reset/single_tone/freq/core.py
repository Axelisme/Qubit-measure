from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import fit_qubit_freq
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReset,
    PulseResetCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class FreqResult:
    """Reset frequencies in MHz and mean complex IQ (frequency,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class FreqModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[PulseCfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[PulseResetCfg, FieldMeta(label="Tested reset")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class FreqSweepCfg(ConfigBase):
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg


@dataclass(frozen=True)
class FreqAnalysis:
    freq: float
    fwhm: float


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        cfg_schema_version="2.0",
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
        modules = cfg.modules

        reset_cfg = modules.tested_reset
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": reset_cfg.pulse_cfg.ch},
        )

        # Retain complex IQ and display its projection along the response axis.
        viewer = context.plots.liveplot_1d(
            "measurement", "Frequency (MHz)", "Amplitude"
        )
        signals_buffer = SignalBuffer(
            (len(freqs),),
            on_update=lambda data: viewer.update(freqs, reset_signal2real(data)),
        )

        # Sweep the tested reset frequency after the optional preparation.
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            modules.tested_reset.set_param(
                "freq", sweep2param("freq", sched.cfg.sweep.freq)
            )
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    PulseReset("tested_reset", modules.tested_reset),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("freq", sched.cfg.sweep.freq)
                .build_and_acquire()
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
