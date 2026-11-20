from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.analysis.fitting import FitQuality, HangerModel, TransmissionModel
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    PulseReadout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)

from zcu_lab.v2.onetone._shared import (
    FreqResult as FreqResult,
)
from zcu_lab.v2.onetone._shared import (
    linear_freqs,
)
from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.resonator import (
    ResonanceOptions as FreqAnalyzeOptions,
)
from zcu_lab.v2.support.resonator import (
    fit_resonance,
)

SamplingMode = Literal["linear", "homophasal"]


@dataclass(frozen=True)
class FreqAnalysis:
    """Resonator center and width in MHz, model parameters and native fit diagnostics."""

    freq: float
    fwhm: float
    params: dict[str, Any]
    fit_quality: FitQuality


def freq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return IQ magnitude without changing the complex input."""
    return np.abs(signals)


def homophasal_freqs_from_sweep(
    sweep: SweepCfg,
    params: HomophasalSamplingCfg,
) -> NDArray[np.float64]:
    """Return MHz points equally spaced in resonator phase over the sweep bounds."""

    def theta_from_freq(freq: float) -> float:
        """Return the fitted resonator phase in radians at freq MHz."""
        return params.theta0 + 2.0 * np.arctan(
            2.0 * params.q_l * (1.0 - freq / params.r_f)
        )

    theta_start = theta_from_freq(sweep.start)
    theta_stop = theta_from_freq(sweep.stop)
    thetas = np.linspace(theta_start, theta_stop, sweep.expts, endpoint=True)
    freqs = params.r_f * (
        1.0 - np.tan((thetas - params.theta0) / 2.0) / (2.0 * params.q_l)
    )
    return np.asarray(freqs, dtype=np.float64)


def _ensure_distinct_homophasal_freqs(freqs: NDArray[np.float64]) -> None:
    if len(freqs) <= 1:
        return
    diffs = np.diff(freqs)
    if np.any(np.isclose(diffs, 0.0, rtol=1e-9, atol=1e-12)):
        raise ValueError(
            "homophasal frequency points collapse after hardware quantization. "
            "Increase the sweep span or reduce expts so adjacent points remain "
            "distinct on the hardware grid."
        )


def homophasal_sweep2array(
    sweep: SweepCfg,
    params: HomophasalSamplingCfg,
    round_info: dict[str, Any],
) -> NDArray[np.float64]:
    """Return hardware-rounded equal-phase MHz points; reject collapsed neighbors."""
    freqs = homophasal_freqs_from_sweep(sweep, params)
    rounded = sweep2array(freqs, "freq", round_info, allow_array=True)
    _ensure_distinct_homophasal_freqs(rounded)
    return rounded


class HomophasalSamplingCfg(ConfigBase):
    r_f: Annotated[float, FieldMeta(label="Resonance frequency", unit="MHz")]
    rf_w: Annotated[float, FieldMeta(label="Linewidth", unit="MHz")]
    theta0: Annotated[float, FieldMeta(label="Phase offset", unit="rad")]

    @model_validator(mode="after")
    def _validate_positive_fit_params(self) -> HomophasalSamplingCfg:
        if self.r_f <= 0.0:
            raise ValueError(f"r_f must be positive, got {self.r_f}")
        if self.rf_w <= 0.0:
            raise ValueError(f"rf_w must be positive, got {self.rf_w}")
        return self

    @property
    def q_l(self) -> float:
        return self.r_f / self.rf_w


class FreqCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Resonator readout 設定與 timing（us）。",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="Absolute MHz 掃描起訖值。",
        ),
    ]
    sampling_mode: Annotated[SamplingMode, FieldMeta(label="Sampling mode")] = "linear"
    homophasal: Annotated[
        HomophasalSamplingCfg | None, FieldMeta(label="Homophasal")
    ] = None

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ]
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
    ]

    @model_validator(mode="after")
    def _validate_sampling_cfg(self) -> FreqCfg:
        if self.sampling_mode == "homophasal":
            if self.homophasal is None:
                raise ValueError(
                    "sampling_mode='homophasal' requires homophasal fit parameters"
                )
        elif self.homophasal is not None:
            raise ValueError(
                "homophasal fit parameters require sampling_mode='homophasal'"
            )
        return self


class FreqExp(Experiment[FreqResult, FreqCfg]):
    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    # freq stores Hz on disk -> scale=1e6 (disk = memory * 1e6)
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        cfg_schema_version="3.0",
        tag="onetone/freq",
    )

    def run(self, config: FreqCfg, *, context: RunContext) -> FreqResult:
        """Measure the resonator response with linear or equal-phase frequency spacing.

        Reset, then pulse and read out at each hardware-swept frequency. Return
        hardware-rounded frequencies in MHz and mean complex IQ at each point;
        fitting this response gives the resonance frequency and linewidth.
        Homophasal sampling requires positive fit parameters and rejects points
        that collapse onto the same hardware frequency.
        """
        # 1. Resolve linear or equal-phase samples on the readout hardware grid.
        cfg = deepcopy(config)
        soccfg = context.soccfg
        if cfg.sampling_mode == "linear":
            freqs = linear_freqs(cfg.freq, cfg.readout, context=context)
        else:
            params = cfg.homophasal
            if params is None:
                raise ValueError("homophasal sampling requires fit parameters")
            freqs = homophasal_sweep2array(
                cfg.freq,
                params,
                {
                    "soccfg": soccfg,
                    "gen_ch": cfg.readout.gen_ch,
                    "ro_ch": cfg.readout.ro_ch,
                },
            )

        # 2. Preview resonance magnitude while preserving complex IQ for fitting.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Frequency (MHz)", ylabel="Amplitude")
        line = live_line(ax)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Draw acquired complex IQ on the measurement axes; return changed artists.

            Read the scan-shaped data without changing it. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            return line.set(freqs, np.abs(data))

        fig.show()
        signals = SignalBuffer((len(freqs),), on_update=update)
        # 3. Reset and measure each resonator frequency with the chosen spacing.
        with Schedule(signals, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                relax_delay=cfg.relax_delay,
            )
            freq = (
                builder.add_sweep("freq", cfg.freq)
                if cfg.sampling_mode == "linear"
                else builder.add_table_sweep("freq", freqs.tolist())
            )
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(PulseReadout("readout", cfg.readout, freq=freq))
            sched.run_acquire(builder.build(), context.soc, rounds=cfg.rounds)
        return FreqResult(freqs=freqs, signals=signals.array)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        # 1. Fit the complex resonance with the requested background model.
        fig, _ = plots.subplots(num="fit", figsize=(9, 8))
        answer = fit_resonance(source.result.freqs, source.result.signals, options)
        # 2. Compare the fitted hanger or transmission model on interior samples.
        freqs, signals = source.result.freqs[1:-1], source.result.signals[1:-1]
        params = answer.params
        fig.clear()
        if "Qc" in params:
            HangerModel.visualize_fit(
                freqs,
                signals,
                params,
                figure=fig,
                fit_bg_amp_slope=options.fit_bg_amp_slope,
                fit_bg_phase_curvature=options.fit_bg_phase_curvature,
            )
        else:
            TransmissionModel.visualize_fit(
                freqs,
                signals,
                params,
                figure=fig,
                fit_bg_amp_slope=options.fit_bg_amp_slope,
                fit_bg_phase_curvature=options.fit_bg_phase_curvature,
            )
        return FreqAnalysis(
            freq=answer.freq,
            fwhm=answer.fwhm,
            params=dict(answer.params),
            fit_quality=answer.fit_quality,
        )
