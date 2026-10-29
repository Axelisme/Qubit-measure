"""Single-qubit amplitude Rabi with ground-herald post-selection."""

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
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import GainSweep, ModularProgramV2, Pulse, Reset, sweep2param
from zcu_tools.program.v2.modules import PulseReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.rabi.amp_rabi.core import (
    AmpRabiAnalysis,
    AmpRabiAnalyzeOptions,
    AmpRabiExp,
    AmpRabiResult,
)


@dataclass(frozen=True)
class HeraldingAmpRabiResult:
    """Gain, selected mean IQ and accepted-shot fraction, each shaped (Ngain,)."""

    amps: NDArray[np.float64]
    signals: NDArray[np.complex128]
    pass_fraction: NDArray[np.float64]


class HeraldingAmpRabiCfg(ConfigBase):
    """Fixed-duration drive gain sweep with one calibrated pulse readout."""

    reps: Annotated[int, FieldMeta(label="Reps")] = 1000
    rounds: Annotated[int, FieldMeta(label="Rounds")] = 10
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="us", decimals=3)
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Rabi Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Gain", unit="a.u.")]


def restore_result(
    data: GroupedLoadData[HeraldingAmpRabiCfg],
) -> HeraldingAmpRabiResult:
    """Restore selected IQ and pass fraction on the same saved gain grid."""
    signals = data.variable("signals")
    fractions = data.variable("pass_fraction")
    if not np.array_equal(signals.axes[0], fractions.axes[0]):
        raise ValueError("Signal and pass fraction must share the same gain grid")
    return HeraldingAmpRabiResult(
        amps=signals.axes[0].astype(np.float64),
        signals=signals.z.astype(np.complex128),
        pass_fraction=fractions.z.astype(np.float64),
    )


class HeraldingAmpRabiExp(Experiment[HeraldingAmpRabiResult, HeraldingAmpRabiCfg]):
    """Herald ground, apply the gain-swept drive, then read accepted signal shots."""

    Options: ClassVar[type[AmpRabiAnalyzeOptions]] = AmpRabiAnalyzeOptions
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingAmpRabiResult, HeraldingAmpRabiCfg]
    ] = GroupedAxesSpec(
        variables=tuple(
            VariableSpec(
                variable=name,
                axes=(VariableAxisSpec("Gain", "", field_name="amps"),),
                z=VariableZSpec(field_name=name, label=label, unit="a.u.", dtype=dtype),
            )
            for name, label, dtype in (
                ("signals", "Signal", np.complex128),
                ("pass_fraction", "Herald pass fraction", np.float64),
            )
        ),
        result_type=HeraldingAmpRabiResult,
        cfg_type=HeraldingAmpRabiCfg,
        tag="heralding/amp_rabi",
        result_builder=restore_result,
    )

    def run(
        self, cfg: HeraldingAmpRabiCfg, *, context: RunContext
    ) -> HeraldingAmpRabiResult:
        """Acquire selected mean IQ and pass fraction versus hardware-quantized gain.

        After optional reset, herald ground, apply the optional init pulse and
        fixed-duration gain-swept drive, then read signal IQ. The herald and signal
        readouts share cfg.readout, including its classifier. Missing classifier or
        circle radius fails before acquisition. The acquisition owner weights
        rounds by accepted counts. Zero-pass points have NaN IQ and fraction
        zero; unmeasured points remain NaN. No active reset is added. cfg is not
        mutated; pulse duration is in us and gain is in native amplitude units.
        """
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch}
        )
        viewer = live_line(
            context.plots, "measurement", "Pulse gain", "IQ magnitude (a.u.)"
        )
        buffer = SignalBuffer(
            (len(gains),), on_update=lambda data: viewer.update(gains, np.abs(data))
        )
        fractions = np.full(len(gains), np.nan, dtype=np.float64)
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            cfg.qub_pulse.set_param("gain", sweep2param("gain", cfg.gain))
            herald_readout = PulseReadout("herald", cfg.readout)
            signal_readout = PulseReadout("signal", cfg.readout)
            # Borrow the emitted herald instance so occurrence=0 selects its readout.
            prog = ModularProgramV2(
                soccfg,
                modules=[
                    Reset("reset", cfg.reset),
                    herald_readout,
                    Pulse("init_pulse", cfg.init_pulse),
                    Pulse("qubit_pulse", cfg.qub_pulse),
                    signal_readout,
                ],
                sweep=[("gain", cfg.gain)],
                reps=cfg.reps,
                relax_delay=cfg.relax_delay,
            )
            herald = herald_readout.herald_spec(prog, occurrence=0)
            builder = sched.prog_builder(context.soc, soccfg)
            try:
                builder.run_program(prog, rounds=cfg.rounds, herald=[herald])
            finally:
                # Cancellation retains the last completed-round summary, if any.
                passed = prog.get_herald_pass_fraction()
                if passed is not None:
                    fractions[:] = passed[herald.ro_ch]
        return HeraldingAmpRabiResult(gains, buffer.array, fractions)

    def analyze(
        self,
        source: RunRecord[HeraldingAmpRabiCfg, HeraldingAmpRabiResult],
        options: AmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AmpRabiAnalysis:
        """Fit finite selected IQ using the ordinary amplitude-Rabi analysis owner.

        Skip leading points on the original gain grid, then remove nonfinite
        gain/IQ before projection and fitting. Pass fraction is diagnostic, not
        a fit weight. Insufficient data fails in the existing analysis owner.
        Saved cfg is not required. Negative skip raises ValueError.
        """
        if options.skip < 0:
            raise ValueError("Skip leading points must be nonnegative")
        gains = source.result.amps[options.skip :]
        signals = source.result.signals[options.skip :]
        finite = np.isfinite(gains) & np.isfinite(signals)
        fit_source = RunRecord(
            cfg=None, result=AmpRabiResult(amps=gains[finite], signals=signals[finite])
        )
        return AmpRabiExp().analyze(fit_source, replace(options, skip=0), plots=plots)
