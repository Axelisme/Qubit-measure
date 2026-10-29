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
from zcu_tools.program.herald import HeraldSpec
from zcu_tools.program.v2 import ModularProgramV2, Pulse, Reset, sweep2param
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
    """Same-grid lengths (us), selected mean IQ and accepted/attempted fractions."""

    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    pass_fraction: NDArray[np.float64]


class HeraldingLenRabiCfg(LenRabiCfg):
    """Length Rabi settings restricted to a calibrated single pulse readout."""

    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]


def restore_result(
    data: GroupedLoadData[HeraldingLenRabiCfg],
) -> HeraldingLenRabiResult:
    """Restore selected IQ and acceptance on the same saved length grid in us."""
    signals = data.variable("signals")
    fractions = data.variable("pass_fraction")
    if not np.array_equal(signals.axes[0], fractions.axes[0]):
        raise ValueError("Signal and pass fraction must share the same length grid")
    return HeraldingLenRabiResult(
        lengths=signals.axes[0].astype(np.float64),
        signals=signals.z.astype(np.complex128),
        pass_fraction=fractions.z.astype(np.float64),
    )


def _combine_host_rounds(
    signals: NDArray[np.complex128], fractions: NDArray[np.float64], reps: int
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Combine completed one-round acquisitions without reclassifying any shots."""
    completed = np.isfinite(fractions)
    counts = np.where(completed, fractions * reps, 0.0)
    if not np.all(np.isclose(counts, np.rint(counts), rtol=0, atol=1e-6)):
        raise ValueError("Herald fraction does not recover an integer shot count")
    counts = np.rint(counts)
    accepted = np.sum(counts, axis=0)
    attempted = np.sum(completed, axis=0) * reps
    # A zero-pass mean is NaN; multiplying it by zero would poison the sum.
    weighted = np.zeros_like(signals)
    np.multiply(signals, counts, out=weighted, where=counts > 0)
    means = np.full(signals.shape[1], np.nan, dtype=np.complex128)
    np.divide(np.sum(weighted, axis=0), accepted, out=means, where=accepted > 0)
    combined = np.full(signals.shape[1], np.nan, dtype=np.float64)
    np.divide(accepted, attempted, out=combined, where=attempted > 0)
    return means, combined


class HeraldingLenRabiExp(Experiment[HeraldingLenRabiResult, HeraldingLenRabiCfg]):
    """Herald ground, sweep drive length, and read only accepted signal shots."""

    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions
    AXES_SPEC: ClassVar[
        GroupedAxesSpec[HeraldingLenRabiResult, HeraldingLenRabiCfg]
    ] = GroupedAxesSpec(
        variables=tuple(
            VariableSpec(
                variable=name,
                axes=(
                    VariableAxisSpec("Length", "s", field_name="lengths", scale=1e-6),
                ),
                z=VariableZSpec(field_name=name, label=label, unit="a.u.", dtype=dtype),
            )
            for name, label, dtype in (
                ("signals", "Signal", np.complex128),
                ("pass_fraction", "Herald pass fraction", np.float64),
            )
        ),
        result_type=HeraldingLenRabiResult,
        cfg_type=HeraldingLenRabiCfg,
        tag="heralding/len_rabi",
        result_builder=restore_result,
    )

    def run(
        self, cfg: HeraldingLenRabiCfg, *, context: RunContext
    ) -> HeraldingLenRabiResult:
        """Acquire accepted mean IQ and pass fraction versus drive length in us.

        Sequence: optional reset, ground herald, optional initialization, drive,
        terminal readout. Reuse one pulse-readout instance for both emissions.
        Const/flat-top use a hardware sweep; other editable waveforms use a
        round-major host scan with a cached program per length. Arb is rejected.
        Acquisition owns classification and within-call count weighting. Host
        rounds combine selected means using the public acceptance fractions.
        Early-stop SNR is checked after each full curve, as in ordinary Len Rabi.
        Zero-pass points retain NaN IQ; unmeasured points retain NaN fractions.
        Missing classifiers fail before acquisition. cfg is not mutated.
        """
        if cfg.qub_pulse.waveform.style == "arb":
            raise ValueError(
                "Len Rabi cannot sweep an arb waveform's asset-derived length. "
                "Select a waveform with an editable length instead."
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
            (len(lengths),),
            on_update=lambda data: viewer.update(lengths, rabi_signal2real(data)),
        )
        fractions = np.full(len(lengths), np.nan, dtype=np.float64)
        check_snr = snr_checker(buffer.at(), cfg.earlystop_snr, rabi_signal2real)
        acquisition_buffer = (
            buffer if hardware_sweep else SignalBuffer((cfg.rounds, len(lengths)))
        )
        with Schedule(cfg, acquisition_buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            if hardware_sweep:
                # Reset -> herald -> optional init -> swept drive -> signal.
                cfg.qub_pulse.set_param("length", sweep2param("length", cfg.length))
                readout = PulseReadout("readout", cfg.readout)
                prog = ModularProgramV2(
                    soccfg,
                    modules=[
                        Reset("reset", cfg.reset),
                        readout,
                        Pulse("init_pulse", cfg.init_pulse),
                        Pulse("qubit_pulse", cfg.qub_pulse),
                        readout,
                    ],
                    sweep=[("length", cfg.length)],
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                )
                herald = readout.herald_spec(prog, occurrence=0)
                try:
                    sched.prog_builder(soc, soccfg).run_program(
                        prog,
                        rounds=cfg.rounds,
                        herald=[herald],
                        options=AcquireOptions(
                            retry=cfg.acquire_retry, stop_condition=check_snr
                        ),
                    )
                finally:
                    passed = prog.get_herald_pass_fraction()
                    if passed is not None:
                        fractions[:] = passed[herald.ro_ch]
            else:
                # Each host round scans all lengths before checking curve SNR.
                round_fractions = np.full((cfg.rounds, len(lengths)), np.nan)
                programs: dict[float, tuple[ModularProgramV2, HeraldSpec]] = {}
                for round_index, rep in sched.repeat("round", cfg.rounds):
                    for index, step in rep.scan("length", list(range(len(lengths)))):
                        length = float(lengths[index])
                        if length not in programs:
                            step.cfg.qub_pulse.set_param("length", length)
                            readout = PulseReadout("readout", step.cfg.readout)
                            prog = ModularProgramV2(
                                soccfg,
                                modules=[
                                    Reset("reset", step.cfg.reset),
                                    readout,
                                    Pulse("init_pulse", step.cfg.init_pulse),
                                    Pulse("qubit_pulse", step.cfg.qub_pulse),
                                    readout,
                                ],
                                reps=step.cfg.reps,
                                relax_delay=step.cfg.relax_delay,
                            )
                            programs[length] = (
                                prog,
                                readout.herald_spec(prog, occurrence=0),
                            )
                        prog, herald = programs[length]
                        try:
                            step.prog_builder(soc, soccfg).run_program(
                                prog,
                                rounds=1,
                                herald=[herald],
                                options=AcquireOptions(retry=cfg.acquire_retry),
                            )
                        finally:
                            passed = prog.get_herald_pass_fraction()
                            if passed is not None:
                                round_fractions[round_index, index] = passed[
                                    herald.ro_ch
                                ]
                                means, fractions = _combine_host_rounds(
                                    acquisition_buffer.array, round_fractions, cfg.reps
                                )
                                buffer.set(means)
                    if check_snr():
                        break
                viewer.update(lengths, rabi_signal2real(buffer.array))
        return HeraldingLenRabiResult(lengths, buffer.array, fractions)

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
        finite = np.isfinite(result.lengths) & np.isfinite(result.signals)
        return LenRabiExp().analyze(
            RunRecord(
                cfg=None,
                result=LenRabiResult(result.lengths[finite], result.signals[finite]),
            ),
            options,
            plots=plots,
        )
