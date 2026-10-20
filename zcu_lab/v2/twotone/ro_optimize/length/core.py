from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    US_TO_S,
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
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)
from zcu_tools.utils.process import SmoothingOptions, SmoothMethod, smooth_signal1d


@dataclass(frozen=True)
class LengthResult:
    """ADC integration windows in us and moment-SNR scores (length,)."""

    lengths: NDArray[np.float64]
    signals: NDArray[np.float64]


class LengthModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    qub_pulse: Annotated[PulseCfg, FieldMeta(label="Qubit pulse")]
    readout: Annotated[PulseReadoutCfg, FieldMeta(label="Readout")]


class LengthSweepCfg(ConfigBase):
    length: Annotated[SweepCfg, FieldMeta(label="Readout length", unit="us")]


class LengthCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LengthModuleCfg
    sweep: LengthSweepCfg
    skew_penalty: Annotated[float, FieldMeta(label="Skew penalty", decimals=3)] = Field(
        default=0.0, ge=0.0
    )


@dataclass(frozen=True)
class LengthAnalyzeOptions:
    t0: float | None = None
    smooth: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    wavelet: str = "sym4"
    wavelet_level: int = 0


@dataclass(frozen=True)
class LengthAnalysis:
    best_length: float


class LengthExp(PersistableExperiment[LengthResult, LengthCfg]):
    Options: ClassVar[type[LengthAnalyzeOptions]] = LengthAnalyzeOptions

    # lengths stored in seconds on disk -> scale=US_TO_S (disk = memory * 1e-6)
    AXES_SPEC = AxesSpec(
        axes=(Axis("lengths", "Readout Length", "s", scale=US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.float64),
        result_type=LengthResult,
        cfg_type=LengthCfg,
        cfg_schema_version="2.0",
        tag="twotone/ge/ro_optimize/length",
    )

    def run(
        self,
        cfg: LengthCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> LengthResult:
        """Measure readout SNR versus the ADC integration window.

        Host-scan the window in us. At each point, apply optional reset,
        omit or apply qub_pulse, then read out. Configure qub_pulse for the
        intended excitation. Keep the RF pulse at the largest window plus
        0.11 us throughout the scan. Return lengths and moment-based SNR
        shaped (Nlength,), optionally reduced by skew_penalty for distorted
        IQ distributions. SNR is not assignment fidelity.
        acquire_kwargs is forwarded to acquisition.
        """
        # Quantize integration windows on the ADC clock, not the generator clock.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        modules = cfg.modules

        readout_cfg = modules.readout
        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "ro_ch": readout_cfg.ro_ch},
        )

        # Display the score for each integration window, not averaged state IQ.
        viewer = context.plots.liveplot_1d(
            "measurement",
            "Readout Length (us)",
            "SNR",
        )
        signals_buffer = SignalBuffer(
            (len(lengths),),
            dtype=np.float64,
            on_update=lambda data: viewer.update(lengths, np.abs(data)),
        )

        # Rebuild per window; hold RF duration fixed while comparing pulse-off/on.
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for _, step in sched.scan("length", lengths.tolist()):
                modules = step.cfg.modules
                modules.readout.set_param("ro_length", step.value)
                modules.readout.set_param("length", lengths.max() + 0.11)
                tracker = MomentTracker()

                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", cfg=modules.reset),
                        Branch("ge", [], Pulse("qub_pulse", cfg=modules.qub_pulse)),
                        PulseReadout("readout", cfg=modules.readout),
                    )
                    .declare_sweep("ge", 2)
                    .build_and_acquire(
                        raw2signal_fn=lambda _raw, tracker=tracker, skew_penalty=step.cfg.skew_penalty: (
                            snr_as_signal(
                                [tracker],
                                ge_axis=1,
                                skew_penalty=skew_penalty,
                            )
                        ),
                        trackers=[tracker],
                        **(acquire_kwargs or {}),
                    )
                )

        # Return the moment-derived score; raw shots and IQ moments are not retained.
        signals = signals_buffer.array

        return LengthResult(lengths, signals)

    def analyze(
        self,
        source: RunRecord[LengthCfg, LengthResult],
        options: LengthAnalyzeOptions,
        *,
        plots: Plots,
    ) -> LengthAnalysis:
        """Choose the integration window from the smoothed SNR curve.

        options controls smoothing and t0 (us). None or zero selects pure SNR;
        positive t0 maximizes SNR/sqrt(length + t0), favoring shorter windows
        more strongly for smaller t0. Negative t0 raises ValueError.
        NaN scores become zero before smoothing. Return best_length in us
        and publish fit. source.cfg may be None.
        """
        result = source.result
        if options.t0 is not None and options.t0 < 0.0:
            raise ValueError("t0 duration normalization must be non-negative")

        lengths, signals = result.lengths, result.signals

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

        if options.t0 is None or options.t0 == 0.0:
            max_id = np.argmax(snrs)
        else:
            max_id = np.argmax(snrs / np.sqrt(lengths + options.t0))

        max_length = float(lengths[max_id])
        max_snr = float(snrs[max_id])

        _, ax = plots.subplots("fit", figsize=config.figsize)

        ax.plot(lengths, snrs)
        ax.axvline(max_length, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Readout Length (us)")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return LengthAnalysis(max_length)
