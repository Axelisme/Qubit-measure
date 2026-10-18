"""Simultaneous T1 acquisition on one requested delay axis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import US_TO_S, AxesSpec, Axis, PersistableExperiment, ZSpec
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
    PulseCfg,
    SweepCfg,
)
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    Join,
    MuxReadout,
    Pulse,
)

from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2._support.mux_sweep_cfg import MuxDriveModuleCfg, participant_indices
from zcu_lab.v2.twotone.time_domain.t1.core import T1AnalyzeOptions

from .analysis import MuxT1Fit, analyze_t1


class MuxT1ToneCfg(ConfigBase):
    """One named original tone's required fixed pi pulse.

    name is the nonempty capture-generated qubit name, unique in the complete
    list. drive is required and validated even for nonparticipants: fixed finite
    MHz frequency, normalized gain, degree phase and nonnegative us delays.
    Native waveform schemas own shape/clock conversion; length cannot be swept.
    """

    name: str
    drive: PulseCfg

    @model_validator(mode="after")
    def validate_drive(self) -> Self:
        """Reject swept or nonfinite pulses, including nonparticipants."""
        for name in ("freq", "gain", "phase", "pre_delay", "post_delay"):
            value = getattr(self.drive, name)
            if isinstance(value, QickParam) or not math.isfinite(value):
                raise ValueError(f"drive.{name} must be fixed and finite")
        if self.drive.pre_delay < 0 or self.drive.post_delay < 0:
            raise ValueError("drive delays must be nonnegative")
        if isinstance(
            self.drive.waveform, (ConstWaveformCfg, FlatTopWaveformCfg)
        ) and isinstance(self.drive.waveform.length, QickParam):
            raise ValueError("drive.waveform.length must be fixed")
        return self


class MuxT1Cfg(ProgramV2Cfg, ExpCfgModel):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxT1ToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of those names; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Every drive is required/validated, including nonparticipants, on independent
    generators distinct from readout. delays is one common SweepCfg or strictly increasing explicit list of finite
    nonnegative requested us. Fits retain requested times, not compiled delays.
    Inherited reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    modules: MuxDriveModuleCfg
    tones: list[MuxT1ToneCfg]
    participants: list[str]
    delays: SweepCfg | list[float]

    @model_validator(mode="after")
    def validate_scan(self) -> Self:
        """Reject invalid counts, channels and requested delay coordinates."""
        if self.reps <= 0 or self.rounds <= 0:
            raise ValueError("reps and rounds must be positive")
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.initial_delay, self.relax_delay)
        ):
            raise ValueError(
                "initial_delay and relax_delay must be finite nonnegative us"
            )
        if len(self.tones) != len(self.modules.readout.tones):
            raise ValueError("tones must match the full readout tone count")
        participant_indices([tone.name for tone in self.tones], self.participants)
        channels: set[int] = set()
        for index, tone in enumerate(self.tones):
            drive = tone.drive
            if drive.ch == self.modules.readout.gen_ch or drive.ch in channels:
                raise ValueError(
                    f"tones[{index}].drive.ch requires an independent generator"
                )
            channels.add(drive.ch)
        times = sweep2array(self.delays, "none", allow_array=True)
        if (
            times.ndim != 1
            or not times.size
            or not np.all(np.isfinite(times))
            or np.any(times < 0)
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("delays must be nonempty finite nonnegative increasing us")
        return self


@dataclass(frozen=True)
class MuxT1Result:
    """Averaged complex IQ on a common requested delay axis.

    times is float64 (P,) requested us, not compiled actual delays. tone_indices
    is int64 (K,) of active original full-tone indices in ascending order.
    signals is complex128 (K,P), I+jQ in ADC-length-normalized a.u. Unwritten
    entries remain NaN; completed-round means survive cancellation. The caller
    must check the RunContext stop signal before declaring success.
    """

    times: NDArray[np.float64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


def _averaged_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # Builder copies modules; collection must use the initialized program owner.
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_iq(prog, acquired, read_index=0)
            return iq[:, 0] + 1j * iq[:, 1]
    raise ValueError("T1 program is missing its named MuxReadout")


class MuxT1Exp(PersistableExperiment[MuxT1Result, MuxT1Cfg]):
    """Stateless mux T1 core; no GUI registration or writeback side effects.

    Native save/load retain requested times, original indices, complex IQ and
    full cfg from explicit RunRecord sources. No current form or library is read.
    """

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("times", "Requested delay", "s", US_TO_S),
            Axis("tone_indices", "Tone", "", dtype=np.int64),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=MuxT1Result,
        cfg_type=MuxT1Cfg,
        cfg_schema_version="1.0",
        tag="mux/t1",
    )

    def run(self, config: MuxT1Cfg, *, context: RunContext) -> MuxT1Result:
        """Acquire common-start pi pulses, then common Delay and mux readout.

        Copy/validate config without mutation. Lower active indices once, keeping
        the full readout in cfg. Each requested time (us) compiles once; Join
        waits for the longest pulse branch, including native pulse delays.
        Return requested times (P,), original indices (K,) and average IQ (K,P).
        context supplies compatible soc/soccfg and owns devices, a fresh stop
        signal and Plots. Publish measurement_i by original index; caller finishes
        and releases plots and hardware resources. Cancellation preserves complete
        points/current completed rounds, leaving unwritten IQ NaN. Acquisition
        failures retain their cause on context.cancel_signal; call raise_if_error()
        before reporting success. Admission/build errors propagate. No I/O saving.
        """
        cfg = MuxT1Cfg.model_validate(config.model_dump())
        indices = participant_indices(
            [tone.name for tone in cfg.tones], cfg.participants
        )
        times = sweep2array(cfg.delays, "none", allow_array=True)
        readout = cfg.modules.readout.with_updates(
            tones=[cfg.modules.readout.tones[i] for i in indices]
        )
        viewers = [
            context.plots.liveplot_1d(
                f"measurement_{i}", "Requested delay (us)", "Magnitude (a.u.)"
            )
            for i in indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(times, np.abs(data[:, row]))

        signals = SignalBuffer((len(times), len(indices)), on_update=update)
        with Schedule(cfg, signals, stop=context.cancel_signal) as sched:
            for delay, step in sched.scan("delay", times.tolist()):
                pulses = []
                for index in indices:
                    drive = cfg.tones[index].drive
                    pulses.append(Pulse(f"drive_{index}", drive))
                builder = (
                    step.prog_builder(context.soc, context.soccfg)
                    .add_reset("reset", cfg.modules.reset)
                    .add(
                        Join(*pulses),
                        Delay("t1_delay", float(delay)),
                        MuxReadout("readout", readout),
                    )
                )
                prog = builder.build()
                _ = builder.run_program(
                    prog, raw2signal_fn=partial(_averaged_signal, prog)
                )
        signals.trigger_update(flush=True)
        return MuxT1Result(
            times, np.asarray(indices, dtype=np.int64), signals.array.T.copy()
        )

    def analyze(
        self,
        source: RunRecord[MuxT1Cfg, MuxT1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxT1Fit | MuxFitFailure, ...]:
        """Fit explicit source using requested us, existing T1 options and quality.

        cfg=None is allowed. Shape/option errors raise. Per-tone numerical failures
        retain indexed reasons, not zero calibrations. plots owns fit_i figures;
        caller finishes/releases them. No hardware, file or current cfg is read.
        """
        return analyze_t1(source.result, options, plots=plots)
