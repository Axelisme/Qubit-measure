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
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Classify,
    GainSweep,
    Pulse,
    Reset,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
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
    """amps is the sweep grid (native gain); sums has shape (points, 3, 3).

    Herald states are g/e/other; components are count/sum I/sum Q.
    Unmeasured sums are NaN. Conditional means at zero count are NaN.
    """

    amps: NDArray[np.float64]
    sums: NDArray[np.float64]


class HeraldingAmpRabiCfg(ConfigBase):
    """Fixed-duration drive gain sweep with one calibrated pulse readout."""

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Rabi Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Gain", unit="a.u.")]
    reps: Annotated[int, FieldMeta(label="Reps")] = 1000
    rounds: Annotated[int, FieldMeta(label="Rounds")] = 10
    relax_delay: Annotated[
        float, FieldMeta(label="Relax delay", unit="us", decimals=3)
    ] = 1.0


def restore_result(
    data: GroupedLoadData[HeraldingAmpRabiCfg],
) -> HeraldingAmpRabiResult:
    """Restore per-class shot sums; old files fail naming the missing sums variable."""
    sums = data.variable("sums")
    return HeraldingAmpRabiResult(
        amps=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingAmpRabiExp(Experiment[HeraldingAmpRabiResult, HeraldingAmpRabiCfg]):
    """Herald ground, sweep drive gain, and store per-class count/IQ sums.

    Store all herald classes, g/e/other. Live plots and fits use only the
    herald-g conditional mean IQ.
    """

    Options: ClassVar[type[AmpRabiAnalyzeOptions]] = AmpRabiAnalyzeOptions
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingAmpRabiResult, HeraldingAmpRabiCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange(label="components", unit=""),
                    VariableAxisSpec.generated_arange(label="herald_states", unit=""),
                    VariableAxisSpec(label="Gain", unit="", field_name="amps"),
                ),
                z=VariableZSpec(
                    field_name="sums", label="Shot sums", unit="", dtype=np.float64
                ),
            ),
        ),
        result_type=HeraldingAmpRabiResult,
        cfg_type=HeraldingAmpRabiCfg,
        tag="heralding/amp_rabi",
        result_builder=restore_result,
    )

    def run(
        self, cfg: HeraldingAmpRabiCfg, *, context: RunContext
    ) -> HeraldingAmpRabiResult:
        """Acquire (Ngain, 3, 3) shot sums versus hardware-quantized gain.

        After optional reset, herald, apply optional init and fixed-duration
        gain-swept drive, then read signal. Both readouts share cfg.readout and
        its GE Post circle classifier. Classes are g/e/other; components are
        count/sum I/sum Q. Rounds add sums. Live and analyze use herald-g
        conditional IQ; zero-count means are NaN. Unmeasured sums remain NaN.
        cfg is not mutated. Pulse duration uses us, gain uses native units.
        Heralding itself adds no active reset.
        """
        if cfg.readout.classifier is None or cfg.readout.classifier.mode != "circle":
            raise ValueError(
                "Heralding requires a circle classifier with g/e/other; run GE Post"
            )
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch}
        )
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Pulse gain", ylabel="IQ magnitude (a.u.)")
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(data):
            return viewer.set(gains, np.abs(split_sums(data)[1][..., 0]))

        fig.show()
        buffer = SignalBuffer(
            (len(gains), 3, 3),
            dtype=np.float64,
            on_update=update,
        )

        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", cfg.gain)
            reset = Reset("reset", cfg.reset)
            if reset is not None:
                builder.emit(reset)
            builder.emit(PulseReadout("herald", cfg.readout))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, gain=gain))
            builder.emit(PulseReadout("signal", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [Classify("herald"), IQ("signal")]),
                options=RunOptions(
                    converter=np.copy,
                ),
            )

        return HeraldingAmpRabiResult(gains, buffer.array)

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
        gains = source.result.amps[options.skip :]
        signals = split_sums(source.result.sums)[1][..., 0][options.skip :]
        finite = np.isfinite(gains) & np.isfinite(signals)
        fit_source = RunRecord(
            cfg=None, result=AmpRabiResult(amps=gains[finite], signals=signals[finite])
        )
        return AmpRabiExp().analyze(fit_source, replace(options, skip=0), plots=plots)
