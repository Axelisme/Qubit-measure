"""Measure mux Hahn Echo coherence with simultaneous pulses and shared waits."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.program.v2.modules import Join, MuxReadout

from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2._support.mux_coherence import (
    MuxCoherenceResult,
    MuxCoherenceToneCfg,
    validate_coherence_result,
    validate_coherence_scan,
)
from zcu_lab.v2._support.mux_coherence_analysis import (
    MuxCoherenceFit,
    fit_coherence_trace,
    prepare_coherence_trace,
    project_coherence_trace,
)
from zcu_lab.v2._support.mux_coherence_persistence import coherence_spec
from zcu_lab.v2._support.mux_liveplot import draw_mux_magnitudes
from zcu_lab.v2._support.mux_sweep_cfg import MuxDriveModuleCfg, participant_indices
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoAnalyzeOptions


class MuxEchoToneCfg(MuxCoherenceToneCfg):
    """One named qubit's x90 drive, MHz detune and x180 pi_pulse.

    Both pulses use the same generator, independent of the other qubits.
    Their frequency, gain, phase and waveform may differ. Nonparticipating
    tones still need valid pulses; name identifies the qubit in saved results.
    """

    pi_pulse: PulseCfg


class MuxT2EchoCfg(ProgramV2Cfg, ExpCfgModel):
    """Mux Hahn Echo pulses, participating qubits and a shared total-delay scan.

    modules holds optional reset and the complete mux readout. tones contains
    one uniquely named MuxEchoToneCfg per readout position. participants is a
    nonempty list of distinct names in acquisition order: Q3/Q1 gives result
    rows for original indices 2/0. All drives must use independent generators
    distinct from readout, including tones that do not participate.
    delay is a finite nonnegative total wait in us, split into two native half
    waits. Each tone's detune ramps its final x90 phase. reps/rounds are positive;
    initial_delay/relax_delay are finite nonnegative us shared by all tones.
    """

    modules: MuxDriveModuleCfg
    tones: Annotated[list[MuxEchoToneCfg], FieldMeta(element_label="name")]
    participants: list[str]
    delay: SweepCfg

    @model_validator(mode="after")
    def validate_scan(self) -> MuxT2EchoCfg:
        """Reject missing active pulses, swept values, shared channels or bad delays."""
        validate_coherence_scan(
            self, self.modules, self.tones, self.delay, self.participants
        )
        validate_coherence_scan(
            self,
            self.modules,
            [
                MuxCoherenceToneCfg(name=t.name, drive=t.pi_pulse, detune=t.detune)
                for t in self.tones
            ],
            self.delay,
            self.participants,
        )
        for index, tone in enumerate(self.tones):
            if tone.drive.ch != tone.pi_pulse.ch:
                raise ValueError(f"tones[{index}].pi_pulse.ch must equal drive.ch")
        return self


def _averaged_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_iq(prog, acquired, read_index=0)
            # Readout returns (delay, tone, I/Q); results keep tone rows.
            return (iq[..., 0] + 1j * iq[..., 1]).T
    raise ValueError("Coherence program is missing its named MuxReadout")


class MuxT2EchoExp(PersistableExperiment[MuxCoherenceResult, MuxT2EchoCfg]):
    """Measure and fit each participating qubit's Hahn Echo decay or fringes."""

    AXES_SPEC = coherence_spec(MuxT2EchoCfg, tag="mux/time_domain/t2echo")
    Options: ClassVar[type[T2EchoAnalyzeOptions]] = T2EchoAnalyzeOptions

    def run(self, config: MuxT2EchoCfg, *, context: RunContext) -> MuxCoherenceResult:
        """Measure x90 → half wait → x180 → half wait → x90 → mux readout.

        config supplies full tones, ordered participants and a total delay sweep
        in us. Each pulse stage starts together; the next wait or readout starts
        after its longest pulse. The final x90 phase ramps with each tone's MHz
        detune. Return requested total delays (P,), averaged complex IQ (K,P),
        original tone indices, actual detunes and source frequencies (K,).
        context supplies hardware and live plots. Cancellation keeps acquired
        averages and NaN gaps; call cancel_signal.raise_if_error before success.
        """
        # 1. Keep participant order while selecting from the complete tone list.
        cfg = MuxT2EchoCfg.model_validate(config.model_dump())
        indices = participant_indices(
            [tone.name for tone in cfg.tones], cfg.participants
        )
        tones = [cfg.tones[index] for index in indices]
        times = sweep2array(cfg.delay, "none")
        # Both waits use the native half-delay grid, not a rounded total delay.
        _ = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg, "scaler": 0.5})
        detunes = np.asarray(
            [
                actual_detune(
                    cfg.delay,
                    detune=tone.detune,
                    gen_ch=tone.drive.ch,
                    soccfg=context.soccfg,
                    time_scaler=0.5,
                )
                for tone in tones
            ],
            dtype=np.float64,
        )
        frequencies = np.asarray(
            [float(tone.drive.freq) for tone in tones], dtype=np.float64
        )
        readout = cfg.modules.readout.with_updates(
            tones=[cfg.modules.readout.tones[index] for index in indices]
        )

        # 2. Show each selected tone on the common requested total-delay axis.
        viewers = [
            context.plots.liveplot_1d(
                f"measurement_{index}", "Requested delay (us)", "Magnitude (a.u.)"
            )
            for index in indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(times, np.abs(data[row]))
            draw_mux_magnitudes(
                context.plots,
                np.broadcast_to(times, data.shape),
                data,
                [
                    f"{index}: {tone.name}"
                    for index, tone in zip(indices, tones, strict=True)
                ],
                "Requested delay (us)",
            )

        signals = SignalBuffer((len(indices), times.size), on_update=update)
        with Schedule(cfg, signals, stop=context.cancel_signal) as sched:
            # 3. Join each pulse stage so both shared waits start after all drives.
            length = sweep2param("length", cfg.delay)
            builder = (
                sched.prog_builder(context.soc, context.soccfg)
                .add_reset("reset", cfg.modules.reset)
                .add(
                    Join(
                        *[
                            Pulse(f"pi2_first_{index}", tone.drive)
                            for index, tone in zip(indices, tones, strict=True)
                        ]
                    ),
                    Delay("delay_first", 0.5 * length),
                    Join(
                        *[
                            Pulse(f"pi_{index}", tone.pi_pulse)
                            for index, tone in zip(indices, tones, strict=True)
                        ]
                    ),
                    Delay("delay_second", 0.5 * length),
                    Join(
                        *[
                            Pulse(
                                f"pi2_final_{index}",
                                tone.drive.with_updates(
                                    phase=tone.drive.phase + 360 * tone.detune * length
                                ),
                            )
                            for index, tone in zip(indices, tones, strict=True)
                        ]
                    ),
                    MuxReadout("readout", readout),
                )
                .declare_sweep("length", cfg.delay)
            )

            # 4. Average completed rounds into tone-major IQ, preserving NaN gaps.
            prog = builder.build()
            _ = builder.run_program(prog, raw2signal_fn=partial(_averaged_signal, prog))
        signals.trigger_update(flush=True)
        return MuxCoherenceResult(
            np.asarray(indices, dtype=np.int64),
            times,
            signals.array.copy(),
            detunes,
            frequencies,
        )

    def analyze(
        self,
        source: RunRecord[MuxT2EchoCfg, MuxCoherenceResult],
        options: T2EchoAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxCoherenceFit | MuxFitFailure, ...]:
        """Fit each source tone's Echo decay or fringes against requested delay.

        source is a Run/Load record, with optional cfg. options.fit_method selects
        decay or fringe; fit_phase frees the fringe phase (otherwise zero).
        Omit the first point and missing IQ. Return per-tone T2E in us, fringe
        rate in MHz, uncertainties and fit quality, or a fit failure reason.
        Publish fit_i plots; invalid sources or fit methods raise.
        """
        # 1. Omit the first requested point before removing missing IQ.
        if options.fit_method not in ("fringe", "decay"):
            raise ValueError(f"Unknown fit_method: {options.fit_method}")
        validate_coherence_result(source.result)
        fit_fringe = options.fit_method == "fringe"
        outcomes: list[MuxCoherenceFit | MuxFitFailure] = []
        for row, raw_index in enumerate(source.result.tone_indices):
            trace = prepare_coherence_trace(
                int(raw_index), source.result.times, source.result.signals[row], skip=1
            )
            fig, ax = plots.subplots(f"fit_{trace.index}", figsize=(8, 5))
            ax.set(xlabel="Requested delay (us)", ylabel="Signal real (a.u.)")
            fig.suptitle(f"echo tone {trace.index}")
            try:
                real = project_coherence_trace(
                    trace, minimum_points=6 if fit_fringe else 4
                )

                # 2. Echo fringes start at zero phase unless phase is fitted freely.
                phase = None if options.fit_phase else 0.0
                outcome, fitted = fit_coherence_trace(
                    trace, real, fit_fringe=fit_fringe, phase=phase
                )
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                ax.plot(
                    trace.times, np.abs(trace.signal), "o", label="acquired magnitude"
                )
                ax.text(
                    0.02, 0.98, str(exc), transform=ax.transAxes, va="top", wrap=True
                )
                outcomes.append(MuxFitFailure(trace.index, str(exc)))
            else:
                ax.plot(trace.times, real, "o", label="measured")
                ax.plot(trace.times, fitted, label="fit")
                ax.set_title(
                    f"T2 = {outcome.t2:.4g} ± {outcome.t2_err:.2g} us; "
                    f"fringe = {outcome.detune:.4g} MHz"
                )
                outcomes.append(outcome)

            # 3. Keep each tone's fit or failure visible without stopping other fits.
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
        return tuple(outcomes)
