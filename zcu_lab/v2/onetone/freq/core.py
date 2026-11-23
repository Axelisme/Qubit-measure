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
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
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
            description="Set the pulse and ADC window used together at each frequency.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Readout RF frequency",
            unit="MHz",
            description="Enter the absolute readout RF sweep endpoints in MHz, not mixer offsets.",
        ),
    ]
    sampling_mode: Annotated[SamplingMode, FieldMeta(label="Sampling mode")] = "linear"
    homophasal: Annotated[
        HomophasalSamplingCfg | None, FieldMeta(label="Homophasal")
    ] = None

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
            description="Enter how many times to repeat and average the entire sweep.",
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
        fig, ax = context.plots.subplots(num="measurement", layout="constrained")
        fig.suptitle("One-tone spectrum · Resonator · IQ magnitude")
        ax.set(
            xlabel="Readout RF frequency (MHz)", ylabel="Magnitude of mean IQ (a.u.)"
        )
        line = live_line(ax)
        signals = SignalBuffer((len(freqs),))

        @context.plots.on_plot_thread
        def update(_: None) -> Iterable[Artist]:
            """Plot the cumulative mean IQ magnitude."""
            return line.set(freqs, freq_signal2real(signals.array))

        def show_round() -> bool:
            """Observe completed rounds without asking the acquisition to stop."""
            update(None)
            return False

        fig.show()
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
            sched.run_acquire(
                builder.build(),
                context.soc,
                rounds=cfg.rounds,
                options=RunOptions(stop_condition=show_round),
            )
        return FreqResult(freqs=freqs, signals=signals.array)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        """Fit interior complex IQ samples and plot the estimate and raw IQ checks."""
        freqs, signals = source.result.freqs, source.result.signals
        answer = fit_resonance(freqs, signals, options)
        dense = np.linspace(freqs[1], freqs[-2], max(201, len(freqs)))
        params = answer.params
        if "Qc" in params:
            predicted = HangerModel.calc_signals(dense, **params)
            model_label = "Hanger fit"
        else:
            predicted = TransmissionModel.calc_signals(dense, **params)
            model_label = "Transmission fit"

        fig, ax = plots.subplots(num="fit", layout="constrained")
        fig.suptitle("One-tone spectrum · Resonator · IQ magnitude")
        ax.plot(
            freqs, freq_signal2real(signals), ".", color="C0", label="Measured samples"
        )
        ax.plot(dense, freq_signal2real(predicted), color="C1", label=model_label)
        ax.axvline(answer.freq, color="0.35", linestyle="--", label="Fitted center")
        ax.set(
            xlabel="Readout RF frequency (MHz)", ylabel="Magnitude of mean IQ (a.u.)"
        )
        quality = answer.fit_quality
        r2 = "unavailable" if quality.r2 is None else f"{quality.r2:.4f}"
        rms = quality.normalized_residual_rms
        residual = "unavailable" if rms is None else f"{rms:.3g}"
        ax.set_title(
            f"Resonator frequency = {answer.freq:.6f} MHz\n"
            f"Linewidth (FWHM) = {answer.fwhm:.4g} MHz · Fit errors unavailable\n"
            f"R² = {r2} · Normalized residual RMS = {residual}",
            loc="left",
            fontsize=11,
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3, fontsize=10)

        self._plot_iq_diagnostics(freqs, signals, dense, predicted, plots=plots)
        return FreqAnalysis(answer.freq, answer.fwhm, dict(params), quality)

    @staticmethod
    def _plot_iq_diagnostics(
        freqs: NDArray[np.float64],
        signals: NDArray[np.complex128],
        dense: NDArray[np.float64],
        predicted: NDArray[np.complex128],
        *,
        plots: Plots,
    ) -> None:
        """Show fit agreement beyond magnitude in raw phase and the IQ plane."""
        fig, _ = plots.subplots(1, 2, num="iq-diagnostics", layout="constrained")
        fig.suptitle("One-tone spectrum · Resonator · Complex fit check")
        phase_ax, iq_ax = fig.axes
        phase = np.unwrap(np.angle(signals))
        fitted_phase = np.unwrap(np.angle(predicted))
        # Align phase branches at the first fitted frequency, not a skipped edge.
        fitted_phase += 2 * np.pi * np.round((phase[1] - fitted_phase[0]) / (2 * np.pi))
        phase_ax.plot(freqs, phase, ".", color="C0", label="Measured samples")
        phase_ax.plot(dense, fitted_phase, color="C1", label="Complex fit")
        phase_ax.set(
            xlabel="Readout RF frequency (MHz)", ylabel="Unwrapped IQ phase (rad)"
        )
        iq_ax.plot(signals.real, signals.imag, ".", color="C0")
        iq_ax.plot(predicted.real, predicted.imag, color="C1")
        iq_ax.set(xlabel="Mean I (a.u.)", ylabel="Mean Q (a.u.)", aspect="equal")
        phase_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=10)
