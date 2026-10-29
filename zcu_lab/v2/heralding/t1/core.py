"""Single-qubit T1 with host-side ground-herald post-selection."""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
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
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Delay,
    DelayAuto,
    LoadValue,
    ModularProgramV2,
    Pulse,
    SweepCfg,
    sweep2param,
)
from zcu_tools.program.v2.modules import PulseReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.time_domain.t1.core import (
    T1Analysis,
    T1AnalyzeOptions,
    T1Exp,
    T1Result,
)


@dataclass(frozen=True)
class HeraldingT1Result:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    pass_fraction: NDArray[np.float64]


class HeraldingT1Cfg(ConfigBase):
    reps: Annotated[int, FieldMeta(label="Reps")] = 1000
    rounds: Annotated[int, FieldMeta(label="Rounds")] = 10
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="us", decimals=3)
    ] = 1.0
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg | list[float], FieldMeta(label="Delay", unit="us")]
    uniform: Annotated[bool, FieldMeta(label="Uniform (linear) sweep")] = True


def restore_result(data: GroupedLoadData[HeraldingT1Cfg]) -> HeraldingT1Result:
    """Restore both variables on the same saved time grid, in us."""
    signals = data.variable("signals")
    fractions = data.variable("pass_fraction")
    if not np.array_equal(signals.axes[0], fractions.axes[0]):
        raise ValueError("Signal and pass fraction must share the same time grid")
    return HeraldingT1Result(
        times=signals.axes[0].astype(np.float64),
        signals=signals.z.astype(np.complex128),
        pass_fraction=fractions.z.astype(np.float64),
    )


class HeraldingT1Exp(Experiment[HeraldingT1Result, HeraldingT1Cfg]):
    """Herald ground, apply pi, wait, then average only accepted signal shots."""

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions
    AXES_SPEC: ClassVar[GroupedAxesSpec[HeraldingT1Result, HeraldingT1Cfg]] = (
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
            result_type=HeraldingT1Result,
            cfg_type=HeraldingT1Cfg,
            tag="heralding/t1",
            result_builder=restore_result,
        )
    )

    def run(self, cfg: HeraldingT1Cfg, *, context: RunContext) -> HeraldingT1Result:
        """Acquire post-selected mean IQ and accepted-shot fraction versus delay.

        Times use us. Both readouts use the same pulse/ADC settings and classifier
        from cfg.readout. Missing calibration fails before acquisition. The host
        selects shots, not FPGA feedback; no active reset is performed. Rounds
        are weighted by accepted counts by the acquisition owner. Zero-pass
        points retain NaN IQ and fraction zero; unmeasured points remain NaN.
        Uniform mode requires SweepCfg. Nonuniform delays use the shared T1
        hardware-grid owner and reject colliding points. cfg is not mutated.
        """
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        delay_table = None
        if cfg.uniform:
            if not isinstance(cfg.length, SweepCfg):
                raise ValueError("uniform mode requires SweepCfg")
            lengths = sweep2array(cfg.length, "time", {"soccfg": soccfg})
        else:
            delay_table = materialize_nonuniform_t1_delays(cfg.length, soccfg=soccfg)
            lengths = delay_table.times_us

        viewer = live_line(
            context.plots, "measurement", "Time (us)", "IQ magnitude (a.u.)"
        )
        buffer = SignalBuffer(
            (len(lengths),),
            on_update=lambda data: viewer.update(lengths, np.abs(data)),
        )
        fractions = np.full(len(lengths), np.nan, dtype=np.float64)
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            herald_readout = PulseReadout("herald", cfg.readout)
            signal_readout = PulseReadout("signal", cfg.readout)

            # Load the requested delay, herald, excite, wait, then read signal IQ.
            if delay_table is None:
                length_sweep = cfg.length
                if not isinstance(length_sweep, SweepCfg):
                    raise ValueError("uniform mode requires SweepCfg")
                prefix = []
                delay = Delay("t1_delay", sweep2param("length", length_sweep))
                sweep: list[tuple[str, SweepCfg | int]] = [("length", length_sweep)]
            else:
                prefix = [
                    LoadValue(
                        "load_t1_delay",
                        values=list(delay_table.cycles),
                        idx_reg="length_idx",
                        val_reg="t1_delay_cycle",
                        auto_compress=False,
                    )
                ]
                delay = DelayAuto("t1_delay", t="t1_delay_cycle")
                sweep = [("length_idx", len(delay_table.cycles))]
            # Borrow these instances so herald_spec addresses the emitted readout.
            prog = ModularProgramV2(
                soccfg,
                modules=[
                    *prefix,
                    herald_readout,
                    Pulse("pi_pulse", cfg.pi_pulse),
                    delay,
                    signal_readout,
                ],
                sweep=sweep,
                reps=cfg.reps,
                relax_delay=cfg.relax_delay,
            )
            herald = herald_readout.herald_spec(prog, occurrence=0)
            builder = sched.prog_builder(context.soc, soccfg)
            try:
                builder.run_program(prog, rounds=cfg.rounds, herald=[herald])
            finally:
                # Cancellation keeps the last completed-round summary, if any.
                passed = prog.get_herald_pass_fraction()
                if passed is not None:
                    fractions[:] = passed[herald.ro_ch]

        return HeraldingT1Result(lengths, buffer.array, fractions)

    def analyze(
        self,
        source: RunRecord[HeraldingT1Cfg, HeraldingT1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> T1Analysis:
        """Fit accepted-shot IQ with the existing single/dual-exponential T1 fit.

        Skip leading points on the original grid, then discard nonfinite times
        and IQ before PCA rotation or fitting. Pass fraction is a diagnostic,
        not an additional fit weight. Insufficient finite data fails in the T1
        analysis owner. Saved cfg is not needed for analysis.
        """
        if options.skip < 0:
            raise ValueError("Skip leading points must be nonnegative")
        times = source.result.times[options.skip :]
        signals = source.result.signals[options.skip :]
        finite = np.isfinite(times) & np.isfinite(signals)
        fit_source = RunRecord(
            cfg=None, result=T1Result(times=times[finite], signals=signals[finite])
        )
        return T1Exp().analyze(fit_source, replace(options, skip=0), plots=plots)
