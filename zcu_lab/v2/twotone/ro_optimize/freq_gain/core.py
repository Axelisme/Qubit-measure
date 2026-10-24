from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
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
from zcu_tools.utils.process import SmoothingOptions, SmoothMethod, smooth_signal_nd

from zcu_lab.v2.support import live_heatmap


@dataclass(frozen=True)
class FreqGainResult:
    """MHz frequencies, normalized gains and moment-SNR scores (frequency, gain)."""

    freqs: NDArray[np.float64]
    gains: NDArray[np.float64]
    signals: NDArray[np.float64]


class FreqGainCfg(ConfigBase):
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
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Qubit pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    freq: Annotated[SweepCfg, FieldMeta(label="Readout freq", unit="MHz")]
    gain: Annotated[SweepCfg, FieldMeta(label="Readout gain", unit="a.u.")]
    skew_penalty: Annotated[float, FieldMeta(label="Skew penalty", decimals=3)] = Field(
        default=0.0, ge=0.0
    )


@dataclass(frozen=True)
class FreqGainAnalyzeOptions:
    smooth: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    wavelet: str = "sym4"
    wavelet_level: int = 0


@dataclass(frozen=True)
class FreqGainAnalysis:
    best_freq: float
    best_gain: float


class FreqGainExp(Experiment[FreqGainResult, FreqGainCfg]):
    Options: ClassVar[type[FreqGainAnalyzeOptions]] = FreqGainAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[FreqGainResult, FreqGainCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("gains", "Gain", "a.u."),
            AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u.", dtype=np.float64),
        result_type=FreqGainResult,
        cfg_type=FreqGainCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/ro_optimize/freq_gain",
    )

    def run(
        self,
        cfg: FreqGainCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqGainResult:
        """Measure readout SNR over a frequency-by-gain grid.

        At each point, apply the optional reset, omit or apply qub_pulse,
        then read out. Configure qub_pulse for the intended excitation.
        Return frequencies in MHz, normalized gains and moment-based SNR
        shaped (Nfreq, Ngain), optionally reduced by skew_penalty for distorted
        IQ distributions. This score is not assignment fidelity.
        acquire_kwargs is forwarded to acquisition.
        """
        # Prepare both readout axes on the pulse generator's hardware grid.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)

        freqs = sweep2array(
            cfg.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.readout.gen_ch},
        )
        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.readout.gen_ch},
        )

        # Retain one separation score per frequency/gain pair.
        viewer = live_heatmap(
            context.plots,
            "measurement",
            "Frequency (MHz)",
            "Gain (a.u.)",
        )
        signals_buffer = SignalBuffer(
            (len(freqs), len(gains)),
            dtype=np.float64,
            on_update=lambda data: viewer.update(freqs, gains, np.abs(data)),
        )

        # Compare pulse-off/on branches while sweeping both readout knobs in hardware.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            freq_sweep = sched.cfg.freq
            freq_param = sweep2param("freq", freq_sweep)
            sched.cfg.readout.set_param("freq", freq_param)

            gain_sweep = sched.cfg.gain
            gain_param = sweep2param("gain", gain_sweep)
            sched.cfg.readout.set_param("gain", gain_param)
            tracker = MomentTracker()

            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", cfg=sched.cfg.reset),
                    Branch("ge", [], Pulse("qub_pulse", cfg=sched.cfg.qub_pulse)),
                    Readout("readout", cfg=sched.cfg.readout),
                )
                .declare_sweep("ge", 2)
                .declare_sweep("freq", freq_sweep)
                .declare_sweep("gain", gain_sweep)
                .build_and_acquire(
                    raw2signal_fn=lambda _raw: snr_as_signal(
                        [tracker],
                        ge_axis=1,
                        skew_penalty=sched.cfg.skew_penalty,
                    ),
                    trackers=[tracker],
                    **(acquire_kwargs or {}),
                    rounds=cfg.rounds,
                )
            )

        # Return the moment-derived score; raw shots and IQ moments are not retained.
        signals = signals_buffer.array

        return FreqGainResult(freqs, gains, signals)

    def analyze(
        self,
        source: RunRecord[FreqGainCfg, FreqGainResult],
        options: FreqGainAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqGainAnalysis:
        """Select the joint maximum of the smoothed frequency/gain SNR map.

        options controls smoothing; unmeasured NaN scores become zero before
        smoothing. Return best_freq in MHz and normalized best_gain, and
        publish fit. source.cfg may be None.
        """
        result = source.result
        freqs, gains, signals = result.freqs, result.gains, result.signals

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = smooth_signal_nd(
            snrs,
            options=SmoothingOptions(
                method=options.smooth_method,
                sigma=options.smooth,
                wavelet=options.wavelet,
                wavelet_level=options.wavelet_level,
            ),
            axes=(0, 1),
        )

        max_freq_id, max_gain_id = np.unravel_index(np.argmax(snrs), snrs.shape)
        max_freq = float(freqs[max_freq_id])
        max_gain = float(gains[max_gain_id])
        max_snr = float(snrs[max_freq_id, max_gain_id])

        _, ax = plots.subplots("fit", figsize=config.figsize)

        ax.imshow(
            snrs.T,
            extent=(freqs[0], freqs[-1], gains[0], gains[-1]),
            aspect="auto",
            origin="lower",
            interpolation="none",
        )
        ax.scatter(max_freq, max_gain, color="r", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Gain (a.u.)")
        ax.legend()

        return FreqGainAnalysis(max_freq, max_gain)
