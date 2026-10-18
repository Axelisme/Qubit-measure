"""Mux Echo core with stage-wide joins and two common native half-delay waits."""

from __future__ import annotations

from typing import ClassVar

from pydantic import model_validator
from zcu_tools.experiment import PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ProgramV2Cfg, PulseCfg, SweepCfg

from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2._support.mux_coherence import (
    MuxCoherenceResult,
    MuxCoherenceToneCfg,
    acquire_coherence,
    validate_coherence_scan,
)
from zcu_lab.v2._support.mux_coherence_analysis import (
    MuxCoherenceFit,
    analyze_coherence,
)
from zcu_lab.v2._support.mux_coherence_persistence import coherence_spec
from zcu_lab.v2._support.mux_sweep_cfg import MuxDriveModuleCfg, participant_indices
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoAnalyzeOptions


class MuxEchoToneCfg(MuxCoherenceToneCfg):
    """One named full tone's x90 drive, MHz detune and required Echo x180.

    pi_pulse must use drive's independent generator. Both pulses are required
    and validated for nonparticipants too. Their fixed frequency, gain, phase
    and waveform may otherwise differ; inherited name is the capture identity.
    """

    pi_pulse: PulseCfg


class MuxT2EchoCfg(ProgramV2Cfg, ExpCfgModel):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxEchoToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of those names; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Every drive is required/validated, including nonparticipants, on independent
    generators distinct from readout. delay is the common finite nonnegative total requested-us SweepCfg, split
    into two native half waits. Each MHz detune ramps the final x90 phase.
    Inherited reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    modules: MuxDriveModuleCfg
    tones: list[MuxEchoToneCfg]
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


class MuxT2EchoExp(PersistableExperiment[MuxCoherenceResult, MuxT2EchoCfg]):
    """Stateless Hahn Echo with common delay axis and explicit native sources."""

    AXES_SPEC = coherence_spec(MuxT2EchoCfg, tag="mux/time_domain/t2echo")
    Options: ClassVar[type[T2EchoAnalyzeOptions]] = T2EchoAnalyzeOptions

    def run(self, config: MuxT2EchoCfg, *, context: RunContext) -> MuxCoherenceResult:
        """Acquire joined x90 → half delay → joined x180 → half delay → joined x90.

        Mux readout follows the longest final stage. Each delay begins after the
        longest prior stage, not independently per branch. Copy/validate config;
        return requested total times (P,) us, average complex IQ (K,P), original
        indices, actual MHz detunes and source x90 frequencies (K,). context owns
        hardware, fresh StopSignal and Plots; caller finishes/releases plots and
        cleans devices. Cancellation retains completed-round averages, unwritten
        slots stay NaN. Acquisition errors preserve their cause on cancel_signal:
        call raise_if_error before success. Validation/build/native conversion
        failures propagate. No save, GUI, binding or proposal side effects occur.
        """
        cfg = MuxT2EchoCfg.model_validate(config.model_dump())
        return acquire_coherence(
            cfg,
            cfg.modules,
            [
                (i, cfg.tones[i])
                for i in participant_indices(
                    [tone.name for tone in cfg.tones], cfg.participants
                )
            ],
            cfg.delay,
            context=context,
            pi_pulses=[tone.pi_pulse for tone in cfg.tones],
        )

    def analyze(
        self,
        source: RunRecord[MuxT2EchoCfg, MuxCoherenceResult],
        options: T2EchoAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxCoherenceFit | MuxFitFailure, ...]:
        """Fit explicit requested-delay source with existing Echo options/policy.

        cfg=None is supported. Omit the original first point, then filter missing
        IQ. fit_method selects fringe or decay (default); fixed fringe phase is
        zero unless fit_phase=True. Decay ignores fit_phase. Return per-index T2E
        (us), fringe rate (MHz), errors/quality or numeric failure reason. Invalid
        sources/options raise. Publish fit_i into caller-owned plots; caller
        finishes/releases them. Never read current forms, files or hardware.
        """
        if options.fit_method not in ("fringe", "decay"):
            raise ValueError(f"Unknown fit_method: {options.fit_method}")
        return analyze_coherence(
            source.result,
            mode="echo",
            fit_fringe=options.fit_method == "fringe",
            fit_phase=options.fit_phase,
            plots=plots,
        )
