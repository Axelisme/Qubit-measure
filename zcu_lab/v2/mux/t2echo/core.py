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
from zcu_lab.v2._support.mux_sweep_cfg import MuxDriveModuleCfg
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoAnalyzeOptions


class MuxEchoToneCfg(MuxCoherenceToneCfg):
    """One full tone's x90 drive, MHz detune and optional Echo x180 pi_pulse.

    pi_pulse is required only for active tones and must use the same independent
    generator as that tone's drive. Both pulses have fixed finite parameters;
    frequency/gain/phase/waveform may otherwise differ. Inactive pulses may be None.
    """

    pi_pulse: PulseCfg | None


class MuxT2EchoCfg(ProgramV2Cfg, ExpCfgModel):
    """Full ordered tones and one requested total-delay sweep in us.

    tones matches the full mux readout; active elements need x90 drive/x180
    pi_pulse on the same generator, independent across tones and from readout.
    Each detune is MHz and ramps final x90 phase with total requested delay.
    delay is finite nonnegative us, split into two common native half waits.
    modules supplies optional reset and full mux readout. Shared reps/rounds are
    positive; inherited initial_delay/relax_delay are finite nonnegative us.
    No per-tone acquisition controls, bindings or GUI state are stored.
    """

    modules: MuxDriveModuleCfg
    tones: list[MuxEchoToneCfg]
    delay: SweepCfg

    @model_validator(mode="after")
    def validate_scan(self) -> MuxT2EchoCfg:
        """Reject missing active pulses, swept values, shared channels or bad delays."""
        validate_coherence_scan(self, self.modules, self.tones, self.delay)
        for index, tone in enumerate(self.tones):
            if tone.active and tone.pi_pulse is None:
                raise ValueError(f"tones[{index}].pi_pulse is required")
        validate_coherence_scan(
            self,
            self.modules,
            [
                MuxCoherenceToneCfg(active=t.active, drive=t.pi_pulse, detune=t.detune)
                for t in self.tones
            ],
            self.delay,
        )
        for index, tone in enumerate(self.tones):
            if (
                tone.active
                and tone.drive is not None
                and tone.pi_pulse is not None
                and tone.drive.ch != tone.pi_pulse.ch
            ):
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
            cfg.tones,
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
