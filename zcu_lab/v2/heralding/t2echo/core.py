"""Hahn Echo with host-side ground-herald post-selection."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
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

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoAnalysis,
    T2EchoAnalyzeOptions,
    T2EchoExp,
    T2EchoResult,
    t2echo_signal2real,
)


@dataclass(frozen=True)
class HeraldingT2EchoResult:
    """times is the sweep grid (us); sums has shape (points, 3, 3).

    Herald states are g/e/other; components are count/sum I/sum Q.
    Unmeasured sums are NaN. Conditional means at zero count are NaN.
    true_activate_detune is run-only MHz metadata, None after native load.
    """

    times: NDArray[np.float64]
    sums: NDArray[np.float64]
    true_activate_detune: float | None = None


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
    acquire_retry: Annotated[int, FieldMeta(label="Acquire retry")] = Field(
        default=0, ge=0
    )
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
                        VariableAxisSpec.generated_arange("components", ""),
                        VariableAxisSpec.generated_arange("herald_states", ""),
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
        """Acquire (Nt, 3, 3) shot sums versus total delay in us.

        Optional reset precedes herald. Echo is pi/2, half delay, pi, half delay,
        pi/2, signal readout. detune in MHz ramps only the final pi/2 phase.
        Readouts share the GE Post circle classifier; classes are g/e/other.
        Components are count/sum I/sum Q. Rounds add sums. Live and analyze
        select herald-g conditional IQ; zero-count means are NaN.
        Unmeasured sums remain NaN. Half-delay rounding and actual detune follow
        ordinary Echo. cfg is copied; heralding itself adds no active reset.
        """
        if cfg.readout.classifier is None or cfg.readout.classifier.mode != "circle":
            raise ValueError(
                "Heralding requires a circle classifier with g/e/other; run GE Post"
            )
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
            return viewer.set(lengths, t2echo_signal2real(split_sums(data)[1][..., 0]))

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), 3, 3),
            dtype=np.float64,
            on_update=update,
        )

        with Schedule( buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            delay = builder.add_sweep("delay", cfg.delay)
            reset = Reset("reset", cfg.reset)
            if reset is not None:
                builder.emit(reset)
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
                    retry=cfg.acquire_retry,
                    stop_condition=snr_checker(
                        buffer.at(),
                        cfg.earlystop_snr,
                        lambda data: t2echo_signal2real(split_sums(data)[1][..., 0]),
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
    ) -> T2EchoAnalysis:
        """Filter nonfinite time/IQ, then use the ordinary Echo analysis owner.

        The owner drops the first remaining sample and selects decay/fringe
        using options and source detune. Saved cfg preserves detune after load.
        Fractions are diagnostics, not fit weights. Insufficient data fails in
        the analysis owner; no alternate fit is attempted here.
        """
        result = source.result
        signals = split_sums(result.sums)[1][..., 0]
        finite = np.isfinite(result.times) & np.isfinite(signals)
        fit_result = T2EchoResult(
            times=result.times[finite],
            signals=signals[finite],
            true_activate_detune=(
                source.cfg.detune
                if source.cfg is not None
                else result.true_activate_detune
            ),
        )
        return T2EchoExp().analyze(
            RunRecord(cfg=None, result=fit_result), options, plots=plots
        )
