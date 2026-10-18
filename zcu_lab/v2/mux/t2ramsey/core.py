"""Mux Ramsey core: common requested delay, independent x90 pulses and detunes."""

from __future__ import annotations

from typing import ClassVar

from pydantic import model_validator
from zcu_tools.experiment import PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ProgramV2Cfg, SweepCfg

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
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyAnalyzeOptions


class MuxT2RamseyCfg(ProgramV2Cfg, ExpCfgModel):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxCoherenceToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of those names; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Every drive is required/validated, including nonparticipants, on independent
    generators distinct from readout. delay is one common finite nonnegative requested-us SweepCfg. Each tone
    has an independent MHz detune controlling the final x90 phase ramp.
    Inherited reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    modules: MuxDriveModuleCfg
    tones: list[MuxCoherenceToneCfg]
    participants: list[str]
    delay: SweepCfg

    @model_validator(mode="after")
    def validate_scan(self) -> MuxT2RamseyCfg:
        """Reject invalid full tones, fixed drives, channels and delay bounds."""
        validate_coherence_scan(
            self, self.modules, self.tones, self.delay, self.participants
        )
        return self


class MuxT2RamseyExp(PersistableExperiment[MuxCoherenceResult, MuxT2RamseyCfg]):
    """Stateless Ramsey run, per-original-index analysis and native source storage."""

    AXES_SPEC = coherence_spec(MuxT2RamseyCfg, tag="mux/time_domain/t2ramsey")
    Options: ClassVar[type[T2RamseyAnalyzeOptions]] = T2RamseyAnalyzeOptions

    def run(self, config: MuxT2RamseyCfg, *, context: RunContext) -> MuxCoherenceResult:
        """Acquire x90 Join → common delay → final x90 Join → mux readout.

        Copy/validate config without mutation. Delay (us) begins after the longest
        first stage. Return shared requested times (P,), average complex IQ (K,P),
        original indices and actual MHz detunes/source frequencies (K,). context
        owns hardware, fresh StopSignal and Plots. Caller finishes/releases plots
        and cleans up devices. Cancellation retains completed-round means, with
        NaN for no data; acquisition errors retain their cause on cancel_signal,
        so call raise_if_error() before reporting success. Validation/build/native
        conversion failures propagate. No saving, GUI or proposal work occurs.
        """
        cfg = MuxT2RamseyCfg.model_validate(config.model_dump())
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
        )

    def analyze(
        self,
        source: RunRecord[MuxT2RamseyCfg, MuxCoherenceResult],
        options: T2RamseyAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxCoherenceFit | MuxFitFailure, ...]:
        """Fit explicit requested-delay source with existing Ramsey options/policy.

        cfg=None is supported. fit_fringe selects damped cosine (default) versus
        decay; fit_phase frees fringe phase, otherwise choose 0/180 from the
        nearest-zero signal. Return per-index T2* (us), fringe rate (MHz), errors
        and quality or numeric failure reason. Invalid sources raise. Publish
        fit_i into caller-owned plots; caller finishes/releases them. No live
        cfg, hardware, files, GUI state or writeback destinations are consulted.
        """
        return analyze_coherence(
            source.result,
            mode="ramsey",
            fit_fringe=options.fit_fringe,
            fit_phase=options.fit_phase,
            plots=plots,
        )
