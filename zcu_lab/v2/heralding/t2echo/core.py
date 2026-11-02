"""Hahn Echo with host-side ground-herald post-selection."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import (
    FitQuality,
    fit_decay,
    fit_decay_fringe,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    config,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Classify,
    Pulse,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
from zcu_tools.program.v2.modules import PulseReadout
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoAnalyzeOptions


@dataclass(frozen=True)
class HeraldingT2EchoResult:
    """times is the sweep grid (us); sums has shape (points, classes, 6).

    Herald states follow the saved classifier's state_names order and length.
    Components are count/ΣI/ΣQ/ΣI²/ΣIQ/ΣQ².
    Unmeasured sums are NaN. Conditional means at zero count are NaN.
    true_activate_detune is run-only MHz metadata, None after native load.
    """

    times: NDArray[np.float64]
    sums: NDArray[np.float64]
    true_activate_detune: float | None = None


@dataclass(frozen=True)
class HeraldingT2EchoAnalysis:
    """Herald-g conditional Echo fit and fitting-owner diagnostics.

    t2e is the Echo decay time in us; t2e_err is its standard error in us.
    detune is the observed fringe frequency in MHz, including the applied
    phase ramp; detune_err is its standard error in MHz. Both are zero for
    decay-only fits. These four numeric fields are never None.
    fit_quality maps "fit" to the fitter's dimensionless residual and
    relative-error metrics, fixed-phase and decay-bound diagnostics, or is
    None when diagnostics were not supplied.
    """

    t2e: float
    t2e_err: float
    detune: float
    detune_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


class HeraldingT2EchoCfg(ConfigBase):
    """Echo pulses, total delay in us and final-pulse phase-ramp MHz.

    Readout must contain a calibrated single-qubit classifier. The acquisition
    owner validates repetitions, rounds and classifier before taking shots.
    """

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi2_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi/2 Pulse")]
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    detune: float = 0.0
    reps: Annotated[int, FieldMeta(label="Reps")] = 1000
    rounds: Annotated[int, FieldMeta(label="Rounds")] = 10
    earlystop_snr: Annotated[float | None, FieldMeta(label="Early-stop SNR")] = Field(
        default=None, gt=0
    )
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="us", decimals=3)
    ] = 1.0


def restore_result(data: GroupedLoadData[HeraldingT2EchoCfg]) -> HeraldingT2EchoResult:
    """Restore per-class shot sums; old files fail naming the missing sums variable."""
    sums = data.variable("sums")
    return HeraldingT2EchoResult(
        times=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingT2EchoExp(Experiment[HeraldingT2EchoResult, HeraldingT2EchoCfg]):
    """Herald ground, run Hahn Echo, then average only accepted signal shots."""

    Options: ClassVar[type[T2EchoAnalyzeOptions]] = T2EchoAnalyzeOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[HeraldingT2EchoResult, HeraldingT2EchoCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec(
                            label="signal_features", unit="", values=tuple(range(6))
                        ),
                        VariableAxisSpec.generated_arange(
                            label="herald_states", unit=""
                        ),
                        VariableAxisSpec(
                            label="Time", unit="s", field_name="times", scale=1e-6
                        ),
                    ),
                    z=VariableZSpec(
                        field_name="sums", label="Shot sums", unit="", dtype=np.float64
                    ),
                ),
            ),
            result_type=HeraldingT2EchoResult,
            cfg_type=HeraldingT2EchoCfg,
            cfg_schema_version="2.0",
            tag="heralding/t2echo",
            result_builder=restore_result,
        )
    )

    def run(
        self, cfg: HeraldingT2EchoCfg, *, context: RunContext
    ) -> HeraldingT2EchoResult:
        """Acquire (Nt, classes, 6) shot sums versus total delay in us.

        Optional reset precedes herald. Echo is pi/2, half delay, pi, half delay,
        pi/2, signal readout. detune in MHz ramps only the final pi/2 phase.
        Readouts share the classifier; herald states follow its state_names.
        Components are count/ΣI/ΣQ/ΣI²/ΣIQ/ΣQ². Rounds add sums. Live and analyze
        select herald-g conditional IQ; zero-count means are NaN.
        Unmeasured sums remain NaN. Half-delay rounding and actual detune follow
        ordinary Echo. cfg is copied; heralding itself adds no active reset.
        """
        if cfg.readout.classifier is None:
            raise ValueError("Heralding requires a readout classifier")
        num_states = len(cfg.readout.classifier.state_names)
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        lengths = sweep2array(cfg.delay, "time", {"soccfg": soccfg, "scaler": 0.5})
        true_detune = actual_detune(
            cfg.delay,
            detune=cfg.detune,
            gen_ch=cfg.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=0.5,
        )
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(
            xlabel="Time (us)",
            ylabel="Amplitude",
            title=f"Heralded T2 Echo (detune={true_detune:.3f}MHz)",
        )
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(lengths, rotate2real(split_sums(data)[1][..., 0]).real)

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), num_states, 6),
            dtype=np.float64,
            on_update=update,
        )

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_sweep("delay", cfg.delay)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("pi2_pulse", cfg.pi2_pulse))
            builder.timeline_inc(0.5 * delay)
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.timeline_inc(0.5 * delay)
            builder.emit(
                Pulse(
                    "pi2_pulse",
                    cfg.pi2_pulse,
                    phase=cfg.pi2_pulse.phase + 360 * cfg.detune * delay,
                )
            )
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [Classify("herald"), IQ("signal")]),
                options=RunOptions(
                    converter=np.copy,
                    stop_condition=snr_checker(
                        buffer.at(),
                        cfg.earlystop_snr,
                        lambda data: rotate2real(split_sums(data)[1][..., 0]).real,
                    ),
                ),
            )

        return HeraldingT2EchoResult(lengths, buffer.array, true_detune)

    def analyze(
        self,
        source: RunRecord[HeraldingT2EchoCfg, HeraldingT2EchoResult],
        options: T2EchoAnalyzeOptions,
        *,
        plots: Plots,
    ) -> HeraldingT2EchoAnalysis:
        """Fit herald-g conditional Echo IQ after discarding nonfinite samples.

        Drop the first remaining sample, then rotate IQ onto its real axis.
        options.fit_method selects decay or fringe; options.fit_phase frees
        the fringe phase, otherwise fixes it at zero degrees. Zero source
        detune forces decay. Use cfg.detune when present, otherwise the run's
        true_activate_detune; a requested fringe without finite detune raises
        ValueError. Unknown fit_method also raises ValueError.
        Draw data and fit, and return lifetime/error in us, observed fringe
        frequency/error in MHz (both zero for decay), and the fitter's quality.
        Pass fractions are not fit weights; insufficient data raises in the
        fitting tool.
        """
        result = source.result
        fit_method = options.fit_method
        fit_phase = options.fit_phase
        if fit_method not in ("fringe", "decay"):
            raise ValueError(f"Unknown fit_method: {fit_method}")
        if fit_method == "fringe":
            detune_source = (
                source.cfg.detune
                if source.cfg is not None
                else result.true_activate_detune
            )
            if detune_source is None or not np.isfinite(detune_source):
                raise ValueError(
                    "Echo fringe fit requires source detune: cfg or saved finite actual_detune"
                )
            if detune_source == 0.0:
                fit_method = "decay"

        signals = split_sums(result.sums)[1][..., 0]
        finite = np.isfinite(result.times) & np.isfinite(signals)
        xs, signals = result.times[finite], signals[finite]

        xs = xs[1:]
        signals = signals[1:]

        # Project IQ onto its signal axis before choosing the envelope model.
        real_signals = rotate2real(signals).real

        if fit_method == "fringe":
            fixedparams = None if fit_phase else [None, None, None, 0.0, None]
            fit = fit_decay_fringe(xs, real_signals, fixedparams=fixedparams)
            t2e, t2eerr, detune, detune_err, y_fit = (
                fit.decay_time,
                fit.decay_time_err,
                fit.freq,
                fit.freq_err,
                fit.curve,
            )
        elif fit_method == "decay":
            fit = fit_decay(xs, real_signals)
            t2e, t2eerr, y_fit = (
                fit.decay_time,
                fit.decay_time_err,
                fit.curve,
            )
            detune = 0.0
            detune_err = 0.0
        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")

        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        ax.plot(xs, real_signals, label="data", ls="-", marker="o", markersize=5)
        ax.plot(xs, y_fit, label="fit", c="orange", zorder=1)

        t2e_str = f"{t2e:.2f}us ± {t2eerr:.2f}us"
        if fit_method == "fringe":
            detune_str = f"{detune:.2f}MHz ± {detune_err * 1e3:.2f}kHz"
            title = r"$T_{2echo}$ fringe = " + f"{t2e_str}, detune = {detune_str}"

        elif fit_method == "decay":
            title = r"$T_{2echo}$ decay = " + f"{t2e_str}"

        else:
            raise ValueError(f"Unknown fit_method: {fit_method}")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Delay Time (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True)

        fig.tight_layout()

        return HeraldingT2EchoAnalysis(
            t2e=float(t2e),
            t2e_err=float(t2eerr),
            detune=float(detune),
            detune_err=float(detune_err),
            fit_quality={"fit": fit.quality},
        )
