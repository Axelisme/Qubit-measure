"""Single-qubit length Rabi with ground-herald post-selection."""

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
    ModularProgramV2,
    Pulse,
    Reset,
    shot_sums,
    split_sums,
    sweep2param,
)
from zcu_tools.program.v2.modules import PulseReadout

from zcu_lab.v2.support import live_line
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiAnalysis,
    LenRabiAnalyzeOptions,
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
    rabi_signal2real,
)


@dataclass(frozen=True)
class HeraldingLenRabiResult:
    """lengths is the sweep grid (us); sums has shape (points, 3, 3).

    Herald states are g/e/other; components are count/sum I/sum Q.
    Unmeasured sums are NaN. Conditional means at zero count are NaN.
    """

    lengths: NDArray[np.float64]
    sums: NDArray[np.float64]


class HeraldingLenRabiCfg(LenRabiCfg):
    """Length Rabi settings restricted to a calibrated single pulse readout."""

    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]


def restore_result(
    data: GroupedLoadData[HeraldingLenRabiCfg],
) -> HeraldingLenRabiResult:
    """Restore per-class shot sums; old files fail naming the missing sums variable."""
    sums = data.variable("sums")
    return HeraldingLenRabiResult(
        lengths=sums.axes[2].astype(np.float64),
        sums=sums.z.astype(np.float64),
    )


class HeraldingLenRabiExp(Experiment[HeraldingLenRabiResult, HeraldingLenRabiCfg]):
    """Herald ground, sweep drive length, and read only accepted signal shots."""

    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingLenRabiResult, HeraldingLenRabiCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("components", ""),
                    VariableAxisSpec.generated_arange("herald_states", ""),
                    VariableAxisSpec("Length", "s", field_name="lengths", scale=1e-6),
                ),
                z=VariableZSpec(
                    field_name="sums", label="Shot sums", unit="", dtype=np.float64
                ),
            ),
        ),
        result_type=HeraldingLenRabiResult,
        cfg_type=HeraldingLenRabiCfg,
        tag="heralding/len_rabi",
        result_builder=restore_result,
    )

    def run(
        self, cfg: HeraldingLenRabiCfg, *, context: RunContext
    ) -> HeraldingLenRabiResult:
        """Acquire (Nl, 3, 3) shot sums versus drive length in us.

        Sequence: optional reset, herald, optional init, drive, signal readout.
        Readouts share cfg.readout with its GE Post circle classifier.
        Classes are g/e/other; components are count/sum I/sum Q.
        Const/flat-top use a hardware sweep. Other editable waveforms use a
        round-major host scan with cached programs; arb is rejected.
        Both hardware rounds and host segments add their sums directly.
        Live and analyze use herald-g conditional IQ; zero-count means are NaN.
        Unmeasured sums remain NaN. Early-stop SNR follows each full curve.
        cfg is not mutated; heralding itself adds no active reset.
        """
        if cfg.qub_pulse.waveform.style == "arb":
            raise ValueError(
                "Len Rabi cannot sweep an arb waveform's asset-derived length. "
                "Select a waveform with an editable length instead."
            )
        if cfg.readout.classifier is None or cfg.readout.classifier.mode != "circle":
            raise ValueError(
                "Heralding requires a circle classifier with g/e/other; run GE Post"
            )
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        lengths = sweep2array(
            cfg.length, "time", {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch}
        )
        hardware_sweep = cfg.qub_pulse.waveform.style in ["const", "flat_top"]
        if not hardware_sweep:
            lengths = np.unique(lengths)

        viewer = live_line(context.plots, "measurement", "Length (us)", "Signal")
        buffer = SignalBuffer(
            (len(lengths), 3, 3),
            dtype=np.float64,
            on_update=lambda data: viewer.update(
                lengths, rabi_signal2real(split_sums(data)[1][..., 0])
            ),
        )
        check_snr = snr_checker(
            buffer.at(),
            cfg.earlystop_snr,
            lambda data: rabi_signal2real(split_sums(data)[1][..., 0]),
        )
        acquisition_buffer = (
            buffer
            if hardware_sweep
            else SignalBuffer((cfg.rounds, len(lengths), 3, 3), dtype=np.float64)
        )
        with Schedule(cfg, acquisition_buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            if hardware_sweep:
                # Reset -> herald -> optional init -> swept drive -> signal.
                cfg.qub_pulse.set_param("length", sweep2param("length", cfg.length))
                herald_readout = PulseReadout("herald", cfg.readout)
                prog = ModularProgramV2(
                    soccfg,
                    modules=[
                        Reset("reset", cfg.reset),
                        herald_readout,
                        Pulse("init_pulse", cfg.init_pulse),
                        Pulse("qubit_pulse", cfg.qub_pulse),
                        PulseReadout("signal", cfg.readout),
                    ],
                    sweep=[("length", cfg.length)],
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                )
                sched.prog_builder(soc, soccfg).run_program_summed(
                    prog,
                    rounds=cfg.rounds,
                    processor=shot_sums(prog, [Classify("herald"), IQ("signal")]),
                    sums2signal_fn=np.copy,
                    options=AcquireOptions(
                        retry=cfg.acquire_retry, stop_condition=check_snr
                    ),
                )
            else:
                # Each host round scans all lengths before checking curve SNR.
                programs: dict[float, ModularProgramV2] = {}
                for _, rep in sched.repeat("round", cfg.rounds):
                    for index, step in rep.scan("length", list(range(len(lengths)))):
                        length = float(lengths[index])
                        if length not in programs:
                            step.cfg.qub_pulse.set_param("length", length)
                            herald_readout = PulseReadout("herald", step.cfg.readout)
                            prog = ModularProgramV2(
                                soccfg,
                                modules=[
                                    Reset("reset", step.cfg.reset),
                                    herald_readout,
                                    Pulse("init_pulse", step.cfg.init_pulse),
                                    Pulse("qubit_pulse", step.cfg.qub_pulse),
                                    PulseReadout("signal", step.cfg.readout),
                                ],
                                reps=step.cfg.reps,
                                relax_delay=step.cfg.relax_delay,
                            )
                            programs[length] = prog
                        prog = programs[length]

                        try:
                            step.prog_builder(soc, soccfg).run_program_summed(
                                prog,
                                rounds=1,
                                processor=shot_sums(
                                    prog, [Classify("herald"), IQ("signal")]
                                ),
                                sums2signal_fn=np.copy,
                                options=AcquireOptions(retry=cfg.acquire_retry),
                            )
                        finally:
                            segments = acquisition_buffer.array
                            completed = np.any(np.isfinite(segments), axis=0)
                            combined = np.nansum(segments, axis=0)
                            buffer.set(np.where(completed, combined, np.nan))
                    if check_snr():
                        break
                viewer.update(
                    lengths, rabi_signal2real(split_sums(buffer.array)[1][..., 0])
                )
        return HeraldingLenRabiResult(lengths, buffer.array)

    def analyze(
        self,
        source: RunRecord[HeraldingLenRabiCfg, HeraldingLenRabiResult],
        options: LenRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> LenRabiAnalysis:
        """Fit finite selected IQ through the ordinary length-Rabi analysis owner.

        Remove nonfinite lengths and complex IQ before PCA projection. Pass
        fraction is diagnostic, not an additional fit weight. decay and fit_phase
        retain the ordinary fit semantics; insufficient data fails in that owner.
        """
        result = source.result
        signals = split_sums(result.sums)[1][..., 0]
        finite = np.isfinite(result.lengths) & np.isfinite(signals)
        return LenRabiExp().analyze(
            RunRecord(
                cfg=None,
                result=LenRabiResult(result.lengths[finite], signals[finite]),
            ),
            options,
            plots=plots,
        )
