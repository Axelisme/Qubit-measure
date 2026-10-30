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
from zcu_tools.program.v2 import (
    IQ,
    Classify,
    Delay,
    ModularProgramV2,
    Pulse,
    Reset,
    shot_sums,
    split_sums,
    sweep2param,
)
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
    """times is the sweep grid (us); sums has shape (points, 3, 3).

    Herald states are g/e/other; components are count/sum I/sum Q.
    Unmeasured sums are NaN. Conditional means at zero count are NaN.
    true_activate_detune is run-only MHz metadata, None after native load.
    """

    times: NDArray[np.float64]
    sums: NDArray[np.float64]
    true_activate_detune: float | None = None


class HeraldingT2RamseyCfg(T2RamseyCfg):
    """Ordinary Ramsey controls restricted to one pulse readout with classifier."""

    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]


def restore_result(
    data: GroupedLoadData[HeraldingT2RamseyCfg],
) -> HeraldingT2RamseyResult:
    """Restore per-class shot sums; old files fail naming the missing sums variable."""
    sums = data.variable("sums")
    return HeraldingT2RamseyResult(
        times=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingT2RamseyExp(Experiment[HeraldingT2RamseyResult, HeraldingT2RamseyCfg]):
    """Herald ground, apply pi/2, wait, apply phase-shifted pi/2, then read IQ."""

    Options: ClassVar[type[T2RamseyAnalyzeOptions]] = T2RamseyAnalyzeOptions
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingT2RamseyResult, HeraldingT2RamseyCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("components", ""),
                    VariableAxisSpec.generated_arange("herald_states", ""),
                    VariableAxisSpec("Time", "s", field_name="times", scale=1e-6),
                ),
                z=VariableZSpec(
                    field_name="sums", label="Shot sums", unit="", dtype=np.float64
                ),
            ),
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
        """Acquire (Nt, 3, 3) shot sums versus free-evolution delay in us.

        Optional reset precedes herald, pi/2, delay, phase-shifted pi/2, signal.
        Readouts share cfg.readout and its GE Post circle classifier.
        Classes are g/e/other; components are count/sum I/sum Q.
        Rounds add sums. Live and analyze select herald-g conditional IQ.
        Zero-count means are NaN; unmeasured sums remain NaN.
        detune in MHz ramps the final pulse phase, not the drive frequency.
        cfg is copied; heralding itself adds no active reset.
        """
        if cfg.readout.classifier is None or cfg.readout.classifier.mode != "circle":
            raise ValueError(
                "Heralding requires a circle classifier with g/e/other; run GE Post"
            )
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
            (len(lengths), 3, 3),
            dtype=np.float64,
            on_update=lambda data: viewer.update(
                lengths, t2ramsey_signal2real(split_sums(data)[1][..., 0])
            ),
        )

        with Schedule(cfg, buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            length_param = sweep2param("length", cfg.length)
            herald_readout = PulseReadout("herald", cfg.readout)
            signal_readout = PulseReadout("signal", cfg.readout)

            # Unique readout names resolve the herald and signal shot-sum axes.
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
            builder = sched.prog_builder(context.soc, soccfg)
            builder.run_program_summed(
                prog,
                rounds=cfg.rounds,
                processor=shot_sums(prog, [Classify("herald"), IQ("signal")]),
                sums2signal_fn=np.copy,
                options=AcquireOptions(
                    retry=cfg.acquire_retry,
                    stop_condition=snr_checker(
                        buffer.at(),
                        cfg.earlystop_snr,
                        lambda data: t2ramsey_signal2real(split_sums(data)[1][..., 0]),
                    ),
                ),
            )

        return HeraldingT2RamseyResult(
            lengths, buffer.array, true_activate_detune=true_detune
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
        signals = split_sums(result.sums)[1][..., 0]
        finite = np.isfinite(result.times) & np.isfinite(signals)
        fit_source = RunRecord(
            cfg=None,
            result=T2RamseyResult(result.times[finite], signals[finite]),
        )
        return T2RamseyExp().analyze(fit_source, options, plots=plots)
