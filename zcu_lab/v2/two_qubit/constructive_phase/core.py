"""Thermal B11 relative-phase × duration experiment."""

from functools import partial
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._drive import DualLineCfg, dual_line_drive
from .._rabi_analysis import joint_return, plot_return_data
from .._rabi_scan import DurationScanCfg, RabiScanResult
from .._readout import collect_joint_shots
from .analysis import ConstructivePhaseAnalysis, analyze_constructive_phase


class ConstructivePhaseCfg(DurationScanCfg):
    """Relative F2 minus F1 phase scan in degrees at fixed common frequency.

    Each point replaces F2.phase with F1.phase + relative_phase. Signed gains,
    their ratio, waveform styles and delays remain unchanged. No parasitic
    cancellation, amplitude-ratio search or heralding is performed.
    """

    relative_phase: Annotated[
        SweepCfg,
        FieldMeta(
            unit="deg",
            label="Relative phase",
            description="填入 F2 相對 F1 的 phase 起訖值，單位 deg。先以 chevron 選定 frequency 與 duration，再確認相位掃描範圍。",
        ),
    ]

    @model_validator(mode="after")
    def validate_phase(self) -> Self:
        """Require finite requested relative phases."""
        if not np.all(np.isfinite(sweep2array(self.relative_phase))):
            raise ValueError("Relative phases must be finite")
        return self


class ConstructivePhaseExp(Experiment[RabiScanResult, ConstructivePhaseCfg]):
    """Stateless relative-phase acquisition and native persistence."""

    AXES_SPEC: ClassVar[AxesSpec[RabiScanResult, ConstructivePhaseCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("tone_indices", "Tone", "", dtype=np.int64),
            AxesSpec.Axis("shot_indices", "Shot", "", dtype=np.int64),
            AxesSpec.Axis("durations", "Drive duration", "s", 1e-6),
            AxesSpec.Axis("coordinates", "F2 relative to F1 phase", "deg"),
        ),
        z=AxesSpec.Z("signals", "Raw IQ", "a.u."),
        result_type=RabiScanResult,
        cfg_type=ConstructivePhaseCfg,
        cfg_schema_version="3.0",
        tag="two_qubit/constructive_phase",
    )

    def run(
        self, config: ConstructivePhaseCfg, *, context: RunContext
    ) -> RabiScanResult:
        """Acquire B11 → dual drive → double Xπ → mux for phase/duration/round.

        Preserve all raw shots and full tones, including partial cancellation.
        Caller must check context.scope.stop.raise_if_error(). No input mutation.
        """
        phases = sweep2array(config.relative_phase)
        drives = [
            DualLineCfg(
                f1=config.drive.f1.model_copy(deep=True),
                f2=config.drive.f2.with_updates(
                    phase=float(config.drive.f1.phase) + phase
                ),
            )
            for phase in phases
        ]
        if not np.all(np.isfinite(phases)):
            raise ValueError("Each finite coordinate requires one explicit drive pair")
        cfg = config.model_copy(deep=True, update={"rounds": 1})
        durations = sweep2array(cfg.duration)
        ntones = len(cfg.readout.tones)
        viewer = live_line(
            context.plots,
            "measurement",
            "Drive duration (us)",
            "Completed shots (all coordinates)",
            num_lines=1,
        )

        def update_view(raw: NDArray[np.complex128]) -> None:
            viewer.update(
                durations, np.all(np.isfinite(raw), axis=-1).sum(axis=(0, 2, 3))
            )

        buffer = SignalBuffer(
            (len(drives), len(durations), config.rounds, cfg.reps, ntones),
            on_update=update_view,
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for index, step in sched.scan("coordinate", list(range(len(drives)))):
                for duration, duration_step in step.scan(
                    "duration", durations.tolist()
                ):
                    drive = drives[index].model_copy(deep=True)
                    for pulse in (drive.f1, drive.f2):
                        pulse.set_param("length", duration)
                    builder = duration_step.prog_builder(
                        context.soc,
                        context.soccfg,
                        reps=duration_step.cfg.reps,
                        initial_delay=duration_step.cfg.initial_delay,
                        relax_delay=duration_step.cfg.relax_delay,
                    )
                    builder.add_pulse("prepare_f1", cfg.x180_f1)
                    builder.add_pulse("prepare_f2", cfg.x180_f2)
                    builder.add(dual_line_drive(drive, frequency=float(drive.f1.freq)))
                    builder.add_pulse("contrast_f1", cfg.x180_f1)
                    builder.add_pulse("contrast_f2", cfg.x180_f2)
                    builder.add(MuxReadout("readout", cfg.readout))
                    prog = builder.build()
                    for _, round_step in duration_step.repeat("round", config.rounds):
                        _ = round_step.prog_builder(
                            context.soc,
                            context.soccfg,
                            reps=round_step.cfg.reps,
                            initial_delay=round_step.cfg.initial_delay,
                            relax_delay=round_step.cfg.relax_delay,
                        ).run_program(
                            prog,
                            raw2signal_fn=partial(collect_joint_shots, prog),
                            rounds=round_step.cfg.rounds,
                        )
        buffer.trigger_update(flush=True)
        return RabiScanResult(
            phases,
            durations,
            np.arange(config.rounds * cfg.reps, dtype=np.int64),
            np.arange(ntones, dtype=np.int64),
            buffer.array.reshape(
                len(drives), len(durations), config.rounds * cfg.reps, ntones
            ),
        )

    def analyze(
        self,
        source: RunRecord[ConstructivePhaseCfg, RabiScanResult],
        options: None,
        *,
        plots: Plots,
    ) -> ConstructivePhaseAnalysis:
        """Classify complete points and plot return using saved cfg only.

        Fit complete duration traces with the undamped cosine, excluding rates
        below 1/saved duration span. Fit rate²=A+B cosφ+C sinφ and return the
        maximum-rate relative phase, or a reason for unresolved/degenerate fits.
        No contrast/sigma, frequency/sigma or residual quality gate is applied.
        Missing source cfg, invalid raw or malformed axes raise ValueError.
        Incomplete points keep raw but are not classified or replaced by zero.
        """
        del options
        if source.cfg is None:
            raise ValueError(
                "Rabi analysis requires saved cfg; current Form/library is not a substitute"
            )
        answer = joint_return(
            source.cfg, source.result, sweep2array(source.cfg.relative_phase)
        )
        plot_return_data(
            source.result,
            answer,
            coordinate_label="F2 relative to F1 phase (deg)",
            map_name="interference",
            plots=plots,
        )
        return analyze_constructive_phase(source.result, answer, plots=plots)
