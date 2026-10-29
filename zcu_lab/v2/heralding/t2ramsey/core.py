"""Single-qubit Ramsey fringes with ground-herald post-selection."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import FieldMeta
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
from zcu_tools.program.herald import herald_ratio, heralded
from zcu_tools.program.v2 import Delay, ModularProgramV2, Pulse, Reset, sweep2param
from zcu_tools.program.v2.modules import PulseReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyAnalysis,
    T2RamseyAnalyzeOptions,
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyResult,
    t2ramsey_signal2real,
)


@dataclass(frozen=True)
class HeraldingT2RamseyResult:
    """Delay in us, selected mean IQ and accepted fraction on one shared grid.

    true_activate_detune is the realized phase-ramp frequency in MHz, not the
    fitted fringe frequency. Like ordinary Ramsey, native load leaves it None.
    """

    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    pass_fraction: NDArray[np.float64]
    true_activate_detune: float | None = None


class HeraldingT2RamseyCfg(T2RamseyCfg):
    """Ordinary Ramsey controls restricted to one pulse readout with classifier."""

    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]


def restore_result(
    data: GroupedLoadData[HeraldingT2RamseyCfg],
) -> HeraldingT2RamseyResult:
    """Restore selected IQ and fractions on the same saved delay grid, in us."""
    signals = data.variable("signals")
    fractions = data.variable("pass_fraction")
    if not np.array_equal(signals.axes[0], fractions.axes[0]):
        raise ValueError("Signal and pass fraction must share the same time grid")
    return HeraldingT2RamseyResult(
        times=signals.axes[0].astype(np.float64),
        signals=signals.z.astype(np.complex128),
        pass_fraction=fractions.z.astype(np.float64),
    )


class HeraldingT2RamseyExp(Experiment[HeraldingT2RamseyResult, HeraldingT2RamseyCfg]):
    """Herald ground, apply pi/2, wait, apply phase-shifted pi/2, then read IQ."""

    Options: ClassVar[type[T2RamseyAnalyzeOptions]] = T2RamseyAnalyzeOptions
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingT2RamseyResult, HeraldingT2RamseyCfg]
    ] = GroupedAxesSpec(
        variables=tuple(
            VariableSpec(
                variable=name,
                axes=(VariableAxisSpec("Time", "s", field_name="times", scale=1e-6),),
                z=VariableZSpec(field_name=name, label=label, unit="a.u.", dtype=dtype),
            )
            for name, label, dtype in (
                ("signals", "Signal", np.complex128),
                ("pass_fraction", "Herald pass fraction", np.float64),
            )
        ),
        result_type=HeraldingT2RamseyResult,
        cfg_type=HeraldingT2RamseyCfg,
        cfg_schema_version="3.0",
        tag="heralding/t2ramsey",
        result_builder=restore_result,
    )

    def run(
        self, cfg: HeraldingT2RamseyCfg, *, context: RunContext
    ) -> HeraldingT2RamseyResult:
        """Acquire selected mean IQ and pass fraction versus free-evolution delay.

        cfg uses us and MHz. Optional reset precedes herald; post-selection
        itself is not active reset. Herald and signal use two PulseReadout
        instances sharing cfg.readout, including its classifier. Missing
        classifier or circle radius fails before acquire.
        Rounds are weighted by accepted counts by the acquisition owner.
        Zero-pass IQ is NaN with fraction zero; unmeasured points stay NaN.
        detune ramps the final pulse phase, not the drive frequency.
        cfg is copied before sweep quantization.
        """
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        lengths = sweep2array(cfg.length, "time", {"soccfg": soccfg})
        true_detune = actual_detune(
            cfg.length,
            detune=cfg.detune,
            gen_ch=cfg.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=1.0,
        )
        viewer = live_line(
            context.plots,
            "measurement",
            "Time (us)",
            "Real Signal",
            title=f"Heralded T2 Ramsey - detune {cfg.detune:.2f} MHz",
        )
        buffer = SignalBuffer(
            (len(lengths),),
            on_update=lambda data: viewer.update(lengths, t2ramsey_signal2real(data)),
        )
        fractions = np.full(len(lengths), np.nan, dtype=np.float64)

        def processed_to_signal(
            raw: list[NDArray[np.float64]],
        ) -> NDArray[np.complex128]:
            """Store pass fractions and return conditional IQ on this sweep's grid."""
            iq, fraction = herald_ratio(raw[0])
            fractions[:] = fraction
            return iq[0, ..., 0] + 1j * iq[0, ..., 1]

        with Schedule(cfg, buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            length_param = sweep2param("length", cfg.length)
            herald_readout = PulseReadout("herald", cfg.readout)
            signal_readout = PulseReadout("signal", cfg.readout)

            # Borrow the herald instance so occurrence addresses its emitted read.
            prog = ModularProgramV2(
                soccfg,
                modules=[
                    Reset("reset", cfg.reset),
                    herald_readout,
                    Pulse("pi2_pulse1", cfg.pi2_pulse),
                    Delay("t2_delay", delay=length_param),
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
            builder.run_program_processed(
                prog,
                rounds=cfg.rounds,
                processors={herald.ro_ch: heralded(herald, np.copy)},
                processed2signal_fn=processed_to_signal,
                options=AcquireOptions(
                    retry=cfg.acquire_retry,
                    stop_condition=snr_checker(
                        buffer.at(), cfg.earlystop_snr, t2ramsey_signal2real
                    ),
                ),
            )

        return HeraldingT2RamseyResult(
            lengths, buffer.array, fractions, true_activate_detune=true_detune
        )

    def analyze(
        self,
        source: RunRecord[HeraldingT2RamseyCfg, HeraldingT2RamseyResult],
        options: T2RamseyAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T2RamseyAnalysis:
        """Discard nonfinite delay/IQ before the ordinary Ramsey projection/fit.

        Options select decay/fringe and free/fixed phase exactly as in ordinary
        Ramsey. Pass fraction is diagnostic, not a fit weight. The existing
        owner reports insufficient data or unusable fits. Saved cfg is unused.
        """
        result = source.result
        finite = np.isfinite(result.times) & np.isfinite(result.signals)
        fit_source = RunRecord(
            cfg=None,
            result=T2RamseyResult(result.times[finite], result.signals[finite]),
        )
        return T2RamseyExp().analyze(fit_source, options, plots=plots)
