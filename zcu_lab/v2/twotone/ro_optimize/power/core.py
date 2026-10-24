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
from zcu_tools.plotting import live_line
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
class PowerResult:
    """Normalized readout gains and moment-SNR scores (gain,)."""

    gains: NDArray[np.float64]
    signals: NDArray[np.float64]


class PowerCfg(ConfigBase):
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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset", group="Modules")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Qubit pulse", group="Modules")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout", group="Modules")]
    gain: Annotated[
        SweepCfg, FieldMeta(label="Readout gain", unit="a.u.", group="Sweep")
    ]
    skew_penalty: Annotated[float, FieldMeta(label="Skew penalty", decimals=3)] = Field(
        default=0.0, ge=0.0
    )


@dataclass(frozen=True)
class PowerAnalyzeOptions:
    penalty_ratio: float = 0.0
    smooth: float = 1.0
    smooth_method: SmoothMethod = "wavelet"
    wavelet: str = "sym4"
    wavelet_level: int = 0


@dataclass(frozen=True)
class PowerAnalysis:
    best_gain: float


class PowerExp(Experiment[PowerResult, PowerCfg]):
    Options: ClassVar[type[PowerAnalyzeOptions]] = PowerAnalyzeOptions

    # gains stored as-is on disk -> scale=IDENTITY (default)
    AXES_SPEC: ClassVar[AxesSpec[PowerResult, PowerCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("gains", "Probe Power", "a.u."),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/ro_optimize/gain",
    )

    def run(
        self,
        cfg: PowerCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> PowerResult:
        """Measure readout-gain dependence of ground/excited IQ separation.

        At each gain, apply the optional reset, omit or apply qub_pulse,
        then read out. Configure qub_pulse for the intended excitation.
        Return normalized gains and moment-based SNR shaped (Ngain,),
        optionally reduced by skew_penalty for distorted IQ distributions.
        This score is not assignment fidelity. acquire_kwargs is forwarded
        to acquisition.
        """
        # Quantize normalized readout gain on the pulse generator.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        modules = cfg

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.readout.gen_ch},
        )

        # Display ground/excited separation rather than either mean IQ trace.
        viewer = live_line(
            context.plots,
            "measurement",
            "Readout Power",
            "SNR",
        )
        signals_buffer = SignalBuffer(
            (len(gains),),
            dtype=np.float64,
            on_update=lambda data: viewer.update(gains, np.abs(data)),
        )

        # Compare pulse-off/on branches at each hardware-swept readout gain.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg
            gain_sweep = sched.cfg.gain
            gain_param = sweep2param("gain", gain_sweep)
            modules.readout.set_param("gain", gain_param)
            tracker = MomentTracker()

            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", cfg=modules.reset),
                    Branch("ge", [], Pulse("qub_pulse", cfg=modules.qub_pulse)),
                    Readout("readout", cfg=modules.readout),
                )
                .declare_sweep("ge", 2)
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

        return PowerResult(gains, signals)

    def analyze(
        self,
        source: RunRecord[PowerCfg, PowerResult],
        options: PowerAnalyzeOptions,
        *,
        plots: Plots,
    ) -> PowerAnalysis:
        """Choose readout gain using smoothed SNR and an optional gain penalty.

        options controls smoothing and penalty_ratio. Maximize
        SNR * exp(-gain * penalty_ratio); zero penalty selects pure SNR.
        NaN scores become zero before smoothing. Return normalized best_gain
        and publish the unpenalized SNR curve as fit. source.cfg may be None.
        """
        result = source.result
        powers, snrs = result.gains, result.signals
        snrs = np.abs(snrs)

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
        penaltized_snrs = snrs * np.exp(-powers * options.penalty_ratio)

        max_id = np.argmax(penaltized_snrs)
        max_power = float(powers[max_id])
        max_snr = float(snrs[max_id])

        _, ax = plots.subplots("fit", figsize=config.figsize)

        ax.plot(powers, snrs)
        ax.axvline(max_power, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Readout Power")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return PowerAnalysis(max_power)
