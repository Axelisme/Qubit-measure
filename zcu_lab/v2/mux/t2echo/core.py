"""Measure mux Hahn Echo coherence with simultaneous pulses and shared waits.

Reset, join the x90 pulses, wait half the requested total delay, join the x180
pulses, wait the other half, join the final x90 pulses, then read out together.
Shorter pulses wait for the longest branch at each stage; native pre/post
delays stay unchanged. The final x90 phase includes each tone's detune.

Return original tone indices (K,), requested total delays (P,) in us and mean
complex IQ (K,P), plus actual detunes and source frequencies in MHz. Missing
rounds leave IQ NaN. Analyze each tone with decay or fringe fits, omitting the
original first point. Zero source detune forces decay. Report T2E/error in us,
fringe rate/error in MHz, fit quality or a per-tone numerical failure.

Timing, detune and persistence details: zcu_lab/v2/mux/t2echo/guide.md.
RunContext, stop and plot ownership: lib/zcu_tools/experiment/v2/runtime/README.md.
RunRecord and save/load contracts: lib/zcu_tools/experiment/README.md.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.runtime.program_builder import AcquireOptions
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Delay,
    Pulse,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.program.v2.modules import (
    Join,
    MuxReadout,
    MuxReadoutCfg,
)

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.acquire import mux_snr_checker
from zcu_lab.v2.support.lifetime_models import (
    LifetimeCurve,
    compare_lifetime_models,
    draw_lifetime_comparison,
)
from zcu_lab.v2.support.mux import MuxFitFailure, averaged_mux_signal
from zcu_lab.v2.support.mux_coherence import (
    MuxCoherenceResult,
    MuxCoherenceToneCfg,
    validate_coherence_result,
    validate_coherence_scan,
)
from zcu_lab.v2.support.mux_coherence_analysis import (
    MuxCoherenceFit,
    fit_coherence_trace,
    prepare_coherence_trace,
    project_coherence_trace,
)
from zcu_lab.v2.support.mux_coherence_persistence import coherence_spec
from zcu_lab.v2.support.mux_sweep_cfg import require_tone_value
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoAnalyzeOptions


class MuxEchoToneCfg(MuxCoherenceToneCfg):
    """One named qubit's x90 drive, MHz detune and x180 pi_pulse.

    Both pulses use the same generator, independent of the other qubits.
    Their frequency, gain, phase and waveform may differ. Nonparticipating
    tones may leave pulses None; name identifies the qubit in saved results.
    """

    pi_pulse: Pulse.Cfg | None = None

    def require_pi_pulse(self) -> Pulse.Cfg:
        """Return the active x180 pulse, or raise ValueError naming the tone."""
        if self.pi_pulse is None:
            raise ValueError(f"{self.name}: active tone requires pi_pulse")
        return self.pi_pulse


class MuxT2EchoCfg(ConfigBase):
    """Mux Hahn Echo pulses, participating qubits and a shared total-delay scan.

    modules holds optional reset and the complete mux readout. tones contains
    one uniquely named MuxEchoToneCfg per readout position. participants is a
    nonempty list of distinct original indices: [2, 0] gives result
    rows for original indices 2/0. All drives must use independent generators
    distinct from readout for selected tones only.
    delay is a finite nonnegative total wait in us, split into two native half
    waits. Each tone's detune ramps its final x90 phase. reps/rounds are positive;
    initial_delay/relax_delay are finite nonnegative us shared by all tones.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    acquire_retry: Annotated[
        int,
        FieldMeta(
            label="Acquire retry",
            description="Retries after a transient acquisition failure.",
        ),
    ] = Field(default=0, ge=0)
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Stop when every active tone reaches this trace SNR; None disables.",
        ),
    ] = Field(default=None, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    tones: Annotated[list[MuxEchoToneCfg], FieldMeta(element_label="name")]
    participants: list[StrictInt]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]

    @model_validator(mode="after")
    def validate_scan(self) -> MuxT2EchoCfg:
        """Reject missing active pulses, swept values, shared channels or bad delays."""
        validate_coherence_scan(self, self.tones, self.delay, self.participants)
        for index in self.participants:
            require_tone_value(self.tones[index].pi_pulse, ("tones", index, "pi_pulse"))
        validate_coherence_scan(
            self,
            [
                MuxCoherenceToneCfg(name=t.name, drive=t.pi_pulse, detune=t.detune)
                for t in self.tones
            ],
            self.delay,
            self.participants,
        )
        for index in self.participants:
            tone = self.tones[index]
            if tone.require_drive().ch != tone.require_pi_pulse().ch:
                raise ValueError(f"tones[{index}].pi_pulse.ch must equal drive.ch")
        return self


class MuxT2EchoExp(Experiment[MuxCoherenceResult, MuxT2EchoCfg]):
    """Measure and fit each participating qubit's Hahn Echo decay or fringes."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxCoherenceResult, MuxT2EchoCfg]] = (
        coherence_spec(MuxT2EchoCfg, tag="mux/time_domain/t2echo")
    )
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
        indices = list(cfg.participants)
        tones = [cfg.tones[index] for index in indices]
        times = sweep2array(cfg.delay, "none")
        # Both waits use the native half-delay grid, not a rounded total delay.
        _ = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg, "scaler": 0.5})
        detunes = np.asarray(
            [
                actual_detune(
                    cfg.delay,
                    detune=tone.detune,
                    gen_ch=tone.require_drive().ch,
                    soccfg=context.soccfg,
                    time_scaler=0.5,
                )
                for tone in tones
            ],
            dtype=np.float64,
        )
        frequencies = np.asarray(
            [float(tone.require_drive().freq) for tone in tones], dtype=np.float64
        )
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[index] for index in indices]
        )

        # 2. Show each selected tone on the common requested total-delay axis.
        viewers = [
            live_line(
                context.plots,
                f"measurement_{index}",
                "Requested delay (us)",
                "Magnitude (a.u.)",
                title=f"{index}: {cfg.tones[index].name}",
            )
            for index in indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(times, np.abs(data[row]))

        signals = SignalBuffer((len(indices), times.size), on_update=update)
        with Schedule(cfg, signals, scope=context.scope) as sched:
            # 3. Join each pulse stage so both shared waits start after all drives.
            length = sweep2param("length", cfg.delay)
            builder = (
                sched.prog_builder(
                    context.soc,
                    context.soccfg,
                    reps=sched.cfg.reps,
                    initial_delay=sched.cfg.initial_delay,
                    relax_delay=sched.cfg.relax_delay,
                )
                .add_reset("reset", cfg.reset)
                .add(
                    Join(
                        *[
                            Pulse(f"pi2_first_{index}", tone.require_drive())
                            for index, tone in zip(indices, tones, strict=True)
                        ]
                    ),
                    Delay("delay_first", 0.5 * length),
                    Join(
                        *[
                            Pulse(f"pi_{index}", tone.require_pi_pulse())
                            for index, tone in zip(indices, tones, strict=True)
                        ]
                    ),
                    Delay("delay_second", 0.5 * length),
                    Join(
                        *[
                            Pulse(
                                f"pi2_final_{index}",
                                tone.require_drive().with_updates(
                                    phase=tone.require_drive().phase
                                    + 360 * tone.detune * length
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
            _ = builder.run_program(
                prog,
                raw2signal_fn=lambda raw: averaged_mux_signal(prog, raw).T,
                rounds=cfg.rounds,
                options=AcquireOptions(
                    retry=cfg.acquire_retry,
                    stop_condition=mux_snr_checker(
                        signals, cfg.earlystop_snr, tone_axis=0
                    ),
                ),
            )
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
        max_model_window_factor: float = 2.0,
    ) -> tuple[MuxCoherenceFit | MuxFitFailure, ...]:
        """Fit each source tone's Echo decay or fringes against requested delay.

        source is a Run/Load record, with optional cfg. options.fit_method selects
        decay or fringe; zero source detune forces decay. Gating uses cfg detune
        when present, otherwise finite saved actual_detunes. Unknown detune fails
        a requested fringe fit with a missing-source reason, not a guessed model.
        Explicit decay needs no detune. fit_phase frees fringe phase (otherwise zero).
        Omit the first point and missing IQ. Return per-tone T2E in us, fringe
        rate in MHz, uncertainties and fit quality, or a fit failure reason.
        comparison records a second Gaussian-envelope fit on the same samples,
        using the same fixed/free phase. max_model_window_factor (default 2)
        marks implausible diagnostic lifetimes relative to max fitted delay;
        it does not change the primary scalar or quality. Draw both available
        curves in fit_i. Secondary numerical failure stays in its record.
        Invalid sources or fit methods raise.
        """
        # 1. Omit the first requested point before removing missing IQ.
        if options.fit_method not in ("fringe", "decay"):
            raise ValueError(f"Unknown fit_method: {options.fit_method}")
        validate_coherence_result(source.result)
        outcomes: list[MuxCoherenceFit | MuxFitFailure] = []
        for row, raw_index in enumerate(source.result.tone_indices):
            trace = prepare_coherence_trace(
                int(raw_index), source.result.times, source.result.signals[row], skip=1
            )
            fig, ax = plots.subplots(f"fit_{trace.index}", figsize=(8, 5))
            ax.set(xlabel="Requested delay (us)", ylabel="Signal real (a.u.)")
            fig.suptitle(f"echo tone {trace.index}")
            try:
                # Source detune, not a fitted oscillation, determines fringe eligibility.
                fit_fringe = options.fit_method == "fringe"
                if fit_fringe:
                    detune = (
                        source.cfg.tones[trace.index].detune
                        if source.cfg is not None
                        else float(source.result.actual_detunes[row])
                    )
                    if not np.isfinite(detune):
                        raise ValueError(
                            "Echo fringe fit requires source detune: cfg or saved finite actual_detune"
                        )
                    fit_fringe = detune != 0.0
                real = project_coherence_trace(
                    trace, minimum_points=6 if fit_fringe else 4
                )

                # 2. Echo fringes start at zero phase unless phase is fitted freely.
                phase = None if options.fit_phase else 0.0
                outcome, fitted = fit_coherence_trace(
                    trace,
                    real,
                    fit_fringe=fit_fringe,
                    phase=phase,
                    expected_frequency_mhz=None,
                )
            except (
                AnalysisFailed,
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
                comparison = compare_lifetime_models(
                    trace.times,
                    real,
                    LifetimeCurve(
                        "exponential_fringe" if fit_fringe else "exponential",
                        (outcome.t2,),
                        (outcome.t2_err,),
                        fitted,
                        outcome.quality,
                        (5 if phase is None else 4) if fit_fringe else 3,
                        fixed_phase=phase if fit_fringe else None,
                    ),
                    max_window_factor=max_model_window_factor,
                )
                outcome = replace(outcome, comparison=comparison)
                draw_lifetime_comparison(ax, trace.times, comparison)
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
