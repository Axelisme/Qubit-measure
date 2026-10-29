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
from zcu_tools.experiment.v2.runtime.program_builder import AcquireOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    Pulse,
    Reset,
    SweepCfg,
    sweep2param,
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
    """Total delays in us, selected mean IQ and accepted/attempted shot fraction.

    All arrays share the delay grid. Zero-pass IQ is NaN. Actual phase-ramp
    frequency is in MHz and is run-only metadata, None after native load.
    """

    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    pass_fraction: NDArray[np.float64]
    true_activate_detune: float | None = None


class HeraldingT2EchoCfg(ConfigBase):
    """Echo pulses, total delay in us and final-pulse phase-ramp MHz.

    Readout must contain a calibrated single-qubit classifier. The acquisition
    owner validates repetitions, rounds and classifier before taking shots.
    """

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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi2_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi/2 Pulse")]
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    detune: float = 0.0


def restore_result(data: GroupedLoadData[HeraldingT2EchoCfg]) -> HeraldingT2EchoResult:
    """Restore selected IQ and fractions on the same native total-delay grid."""
    signals = data.variable("signals")
    fractions = data.variable("pass_fraction")
    if not np.array_equal(signals.axes[0], fractions.axes[0]):
        raise ValueError("Signal and pass fraction must share the same time grid")
    return HeraldingT2EchoResult(
        times=signals.axes[0].astype(np.float64),
        signals=signals.z.astype(np.complex128),
        pass_fraction=fractions.z.astype(np.float64),
    )


class HeraldingT2EchoExp(Experiment[HeraldingT2EchoResult, HeraldingT2EchoCfg]):
    """Herald ground, run Hahn Echo, then average only accepted signal shots."""

    Options: ClassVar[type[T2EchoAnalyzeOptions]] = T2EchoAnalyzeOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[HeraldingT2EchoResult, HeraldingT2EchoCfg]] = (
        GroupedAxesSpec(
            variables=tuple(
                VariableSpec(
                    variable=name,
                    axes=(
                        VariableAxisSpec("Time", "s", field_name="times", scale=1e-6),
                    ),
                    z=VariableZSpec(
                        field_name=name, label=label, unit="a.u.", dtype=dtype
                    ),
                )
                for name, label, dtype in (
                    ("signals", "Signal", np.complex128),
                    ("pass_fraction", "Herald pass fraction", np.float64),
                )
            ),
            result_type=HeraldingT2EchoResult,
            cfg_type=HeraldingT2EchoCfg,
            tag="heralding/t2echo",
            result_builder=restore_result,
        )
    )

    def run(
        self, cfg: HeraldingT2EchoCfg, *, context: RunContext
    ) -> HeraldingT2EchoResult:
        """Acquire selected IQ and pass fraction versus total delay in us.

        Optional reset precedes herald. Echo is pi/2 → half delay → pi → half
        delay → pi/2 → signal readout. detune (MHz) ramps only the final pi/2
        phase. Both readouts use cfg.readout and its classifier, with host
        selection, not active reset. Across rounds the acquisition owner weights
        by accepted counts. Zero-pass IQ stays NaN; unmeasured points stay NaN.
        Half-delay rounding and actual detune follow ordinary Echo. cfg is copied.
        """
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        lengths = sweep2array(cfg.length, "time", {"soccfg": soccfg, "scaler": 0.5})
        true_detune = actual_detune(
            cfg.length,
            detune=cfg.detune,
            gen_ch=cfg.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=0.5,
        )
        viewer = live_line(
            context.plots,
            "measurement",
            "Time (us)",
            "Amplitude",
            title=f"Heralded T2 Echo (detune={true_detune:.3f}MHz)",
        )
        buffer = SignalBuffer(
            (len(lengths),),
            on_update=lambda data: viewer.update(lengths, t2echo_signal2real(data)),
        )
        fractions = np.full(len(lengths), np.nan, dtype=np.float64)
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            length_param = sweep2param("length", cfg.length)
            herald_readout = PulseReadout("herald", cfg.readout)
            signal_readout = PulseReadout("signal", cfg.readout)

            # Keep the emitted instance for the public herald declaration.
            prog = ModularProgramV2(
                soccfg,
                modules=[
                    Reset("reset", cfg.reset),
                    herald_readout,
                    Pulse("pi2_pulse1", cfg.pi2_pulse),
                    Delay("t2e_delay1", delay=0.5 * length_param),
                    Pulse("pi_pulse", cfg.pi_pulse),
                    Delay("t2e_delay2", delay=0.5 * length_param),
                    Pulse(
                        "pi2_pulse2",
                        cfg.pi2_pulse.with_updates(
                            phase=cfg.pi2_pulse.phase + 360 * cfg.detune * length_param
                        ),
                    ),
                    signal_readout,
                ],
                sweep=[("length", cfg.length)],
                reps=cfg.reps,
                relax_delay=cfg.relax_delay,
            )
            herald = herald_readout.herald_spec(prog, occurrence=0)
            builder = sched.prog_builder(context.soc, soccfg)
            try:
                builder.run_program(
                    prog,
                    rounds=cfg.rounds,
                    herald=[herald],
                    options=AcquireOptions(
                        retry=cfg.acquire_retry,
                        stop_condition=snr_checker(
                            buffer.at(), cfg.earlystop_snr, t2echo_signal2real
                        ),
                    ),
                )
            finally:
                # Cancellation retains the last completed-round summary, if any.
                passed = prog.get_herald_pass_fraction()
                if passed is not None:
                    fractions[:] = passed[herald.ro_ch]

        return HeraldingT2EchoResult(lengths, buffer.array, fractions, true_detune)

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
        finite = np.isfinite(result.times) & np.isfinite(result.signals)
        fit_result = T2EchoResult(
            times=result.times[finite],
            signals=result.signals[finite],
            true_activate_detune=(
                source.cfg.detune
                if source.cfg is not None
                else result.true_activate_detune
            ),
        )
        return T2EchoExp().analyze(
            RunRecord(cfg=None, result=fit_result), options, plots=plots
        )
