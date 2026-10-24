from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, PersistableExperiment, ProgramExpCfg, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    Pulse,
    PulseReadout,
    Readout,
    Reset,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import SmoothingOptions, SmoothMethod, smooth_signal1d


@dataclass(frozen=True)
class FreqResult:
    """Readout frequencies in MHz and moment-SNR scores (frequency,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]


class FreqModuleCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Qubit pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]


class FreqSweepCfg(ConfigBase):
    freq: Annotated[SweepCfg, FieldMeta(label="Readout freq", unit="MHz")]


class FreqCfg(ProgramExpCfg):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg
    skew_penalty: Annotated[float, FieldMeta(label="Skew penalty", decimals=3)] = Field(
        default=0.0, ge=0.0
    )


@dataclass(frozen=True)
class FreqAnalyzeOptions:
    smooth: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    wavelet: str = "sym4"
    wavelet_level: int = 0


@dataclass(frozen=True)
class FreqAnalysis:
    best_freq: float


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    # freq stores Hz on disk -> scale=1e6 (disk = memory * 1e6)
    AXES_SPEC = AxesSpec(
        axes=(AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        cfg_schema_version="2.0",
        tag="twotone/ge/ro_optimize/freq",
    )

    def run(
        self,
        cfg: FreqCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqResult:
        """Measure readout-frequency dependence of ground/excited IQ separation.

        At each frequency, apply the optional reset, omit or apply qub_pulse,
        then read out. Configure qub_pulse for the intended excitation.
        Return MHz frequencies and moment-based SNR shaped (Nfreq,);
        skew_penalty optionally reduces scores for distorted IQ distributions.
        This score is not assignment fidelity. acquire_kwargs is forwarded
        to acquisition.
        """
        # Quantize the readout frequency grid on the pulse generator.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        modules = cfg.modules

        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.gen_ch},
        )

        # Show the separation score rather than either preparation's mean IQ.
        viewer = context.plots.liveplot_1d(
            "measurement",
            "Frequency (MHz)",
            "SNR",
        )
        signals_buffer = SignalBuffer(
            (len(freqs),),
            dtype=np.float64,
            on_update=lambda data: viewer.update(freqs, np.abs(data)),
        )

        # Compare pulse-off/on branches at each hardware-swept readout frequency.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg.modules
            freq_sweep = sched.cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.readout.set_param("freq", freq_param)
            tracker = MomentTracker()

            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", cfg=modules.reset),
                    Branch("ge", [], Pulse("qub_pulse", cfg=modules.qub_pulse)),
                    Readout("readout", cfg=modules.readout),
                )
                .declare_sweep("ge", 2)
                .declare_sweep("freq", freq_sweep)
                .build_and_acquire(
                    raw2signal_fn=lambda _raw: snr_as_signal(
                        [tracker],
                        ge_axis=1,
                        skew_penalty=sched.cfg.skew_penalty,
                    ),
                    trackers=[tracker],
                    **(acquire_kwargs or {}),
                )
            )

        # Return the moment-derived score; raw shots and IQ moments are not retained.
        signals = signals_buffer.array

        return FreqResult(freqs, signals)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        """Select the frequency with maximum smoothed SNR and publish fit.

        options controls smoothing; unmeasured NaN scores become zero before
        smoothing. Return best_freq in MHz. source.cfg may be None.
        """
        result = source.result
        freqs, signals = result.freqs, result.signals

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = smooth_signal1d(
            snrs,
            options=SmoothingOptions(
                method=options.smooth_method,
                sigma=options.smooth,
                wavelet=options.wavelet,
                wavelet_level=options.wavelet_level,
            ),
            axis=0,
        )

        max_id = np.argmax(snrs)
        max_freq = float(freqs[max_id])
        max_snr = float(snrs[max_id])

        _, ax = plots.subplots("fit", figsize=config.figsize)

        ax.plot(freqs, snrs)
        ax.axvline(max_freq, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return FreqAnalysis(max_freq)
