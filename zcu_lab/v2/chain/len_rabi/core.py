"""Length Rabi with a K-cycle readout chain at every pulse length.

One cycle is one rep: the thermalisation wait K (relax_delay), then eight
blocks of (Rabi pulse of length x, read, read). The host reduces each cycle to
65 numbers and keeps only running sums. Analysis fits all 65 numbers jointly
with a Rabi curve, starting from the usual twotone fit of the first read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from pydantic import Field, model_validator
from zcu_tools.analysis.readout_chain import CycleCheck, CycleSums, fit_cycle
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment, GroupedAxesSpec, VariableAxisSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    PulseReadout,
    SweepCfg,
    TimelineBuilder,
    chain_cycle_sums,
)

from zcu_lab.v2.chain._cycle import (
    READS_PER_CYCLE,
    cycle_sums_variables,
    emit_cycle,
    first_read_iq,
    live_cycle_curves,
    plot_cycle_fit,
    rabi_model,
    require_first_round_covariance,
    restore_cycle_sums,
)
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiAnalyzeOptions,
    LenRabiExp,
    LenRabiResult,
)


class ChainLenRabiCfg(ConfigBase):
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Rabi Pulse",
            description="Drive pulse D(x) played before each pair of reads; the sweep sets its length. Use a const or flat_top waveform.",
        ),
    ]
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Pulse readout played twice after each Rabi pulse.",
        ),
    ]
    length: Annotated[SweepCfg, FieldMeta(label="Duration", unit="us")]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            description="Cycles per length in each round; require points × (reps − 1) >= 65 for the first live curve.",
        ),
    ] = Field(default=100, ge=2)
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            description="Software rounds. The live plot updates after each round.",
        ),
    ] = 100
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Thermalisation wait K before every cycle; set at least 5 T1 so each cycle starts in thermal equilibrium.",
        ),
    ] = 1.0

    @model_validator(mode="after")
    def _check_first_round_covariance(self) -> Self:
        require_first_round_covariance(self.length.expts, self.reps)
        return self


@dataclass(frozen=True)
class ChainLenRabiAnalysis:
    """Rabi fit of all 65 numbers, its checks and its starting value.

    pi_len and pi_len_err are the π length 1 / (2 f) in us and its error from
    the Wald error of f. f and f_err are the Rabi frequency in MHz (cycles per
    us) and its Wald error. reliability is R = 1 / (2π × length span × f_err),
    a diagnostic without a threshold. within_point, sharing and curve are the
    three lack-of-fit checks of fit_cycle (sharing has no p). J and df are the
    joint weighted residual and its degrees of freedom. first_read_pi_len is
    the π length in us of the twotone/rabi/len_rabi fit of the first read,
    which gave the starting value of f.
    """

    pi_len: float
    pi_len_err: float
    f: float
    f_err: float
    reliability: float | None
    within_point: CycleCheck
    sharing: CycleCheck
    curve: CycleCheck
    J: float
    df: int
    first_read_pi_len: float


# Lengths are us in memory and seconds on disk.
LENGTH_AXIS = VariableAxisSpec(label="Length", unit="s", field_name="axis", scale=1e-6)


class ChainLenRabiExp(Experiment[CycleSums, ChainLenRabiCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[CycleSums, ChainLenRabiCfg]] = GroupedAxesSpec(
        variables=cycle_sums_variables(LENGTH_AXIS),
        result_type=CycleSums,
        cfg_type=ChainLenRabiCfg,
        tag="chain/len_rabi",
        cfg_schema_version="1.0",
        result_builder=restore_cycle_sums,
        labber_layout="independent",
    )

    def run(self, cfg: ChainLenRabiCfg, *, context: RunContext) -> CycleSums:
        """Sweep the Rabi pulse length with one K-cycle per rep.

        Each cycle waits relax_delay (K), then plays eight blocks of the
        length-swept Rabi pulse and two reads. The FPGA sweeps the length, so
        the pulse must use a const or flat_top waveform; other waveforms raise
        ValueError before acquisition. One round makes cfg.reps cycles at every
        length. After each round the live plot shows the all-number and
        first-read curves in sigma. Return the running sums of all completed
        rounds as CycleSums along the length axis in us. A stop keeps completed
        rounds; a stop before the first round finishes raises RuntimeError.
        """
        if cfg.qub_pulse.waveform.style not in ("const", "flat_top"):
            raise ValueError(
                "Chain length Rabi sweeps the pulse length on the FPGA; "
                "select a const or flat_top waveform."
            )

        # 1. Length axis and the live plot.
        lengths = sweep2array(
            cfg.length, "time", {"soccfg": context.soccfg, "gen_ch": cfg.qub_pulse.ch}
        )
        update = live_cycle_curves(context.plots, lengths, xlabel="Pulse length (us)")
        augmented_sums = SignalBuffer(
            (len(lengths), 66, 66), dtype=np.float64, on_update=update
        )

        # 2. One cycle per rep: K, then (Rabi pulse, read, read) eight times.
        with Schedule(augmented_sums, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length = builder.add_sweep("length", cfg.length)
            rabi_pulse = Pulse("qub_pulse", cfg.qub_pulse, length=length)
            readout = PulseReadout("readout", cfg.readout)
            emit_cycle(builder, readout, lambda: builder.emit(rabi_pulse))

            prog = builder.build()
            prog.set_reads_per_shot(READS_PER_CYCLE)
            sched.run_acquire_summed(
                prog,
                context.soc,
                processor=chain_cycle_sums(prog, "readout"),
                rounds=cfg.rounds,
            )

        # 3. Summed acquisition writes whole rounds, so NaN means no round finished.
        if np.isnan(augmented_sums.array).any():
            raise RuntimeError(
                "Stopped before the first round finished; no sums to keep"
            )
        return CycleSums.from_augmented(lengths, augmented_sums.array)

    def analyze(
        self,
        source: RunRecord[ChainLenRabiCfg, CycleSums],
        options: None,
        *,
        plots: Plots,
    ) -> ChainLenRabiAnalysis:
        """Fit all 65 numbers with a Rabi curve, starting from the first read.

        First fit the first read's mean IQ with the twotone/rabi/len_rabi fit
        (decaying cosine), which also publishes its "fit" figure; its π length
        x_π gives the start f = 1 / (2 x_π). Then fit_cycle fits the saved
        sums with rabi_model and publishes "populations": per-length first-read
        populations with error bars and the fitted curves. Return the π length
        and f with Wald errors, R and the three checks. The source cfg may be
        None. Fitting failures of either fit propagate.
        """
        del options  # This analysis has no configurable options.
        sums = source.result
        lengths = sums.axis

        # 1. Starting value: the twotone length Rabi fit of the first read.
        first_read = LenRabiResult(lengths=lengths, signals=first_read_iq(sums))
        textbook = LenRabiExp().analyze(
            RunRecord(cfg=None, result=first_read),
            LenRabiAnalyzeOptions(),
            plots=plots,
        )

        # 2. Joint fit of all 65 numbers with the Rabi curve.
        model = rabi_model()
        fit = fit_cycle(sums, model, {"f": 1 / (2 * textbook.pi_len)})

        # 3. The π length is half a period of sin²(π f x).
        f, f_err = fit.parameters["f"], fit.errors["f"]
        pi_len = 1 / (2 * f)
        pi_len_err = f_err / (2 * f**2)
        plot_cycle_fit(
            plots,
            model,
            fit,
            lengths,
            xlabel="Pulse length (us)",
            result=f"pi length = {pi_len:.4g} ± {pi_len_err:.2g} us, R = {fit.reliability:.2g}",
        )

        return ChainLenRabiAnalysis(
            pi_len=pi_len,
            pi_len_err=pi_len_err,
            f=f,
            f_err=f_err,
            reliability=fit.reliability,
            within_point=fit.within_point,
            sharing=fit.sharing,
            curve=fit.curve,
            J=fit.J,
            df=fit.df,
            first_read_pi_len=textbook.pi_len,
        )
