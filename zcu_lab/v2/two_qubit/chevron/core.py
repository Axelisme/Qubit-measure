"""Thermal B11 common-frequency × duration Rabi chevron."""

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
from .._rabi_analysis import (
    RabiAnalysis,
    fit_duration_rates,
    joint_return,
    plot_duration_rates,
    plot_return_data,
)
from .._rabi_scan import DurationScanCfg, RabiScanResult
from .._readout import collect_joint_shots


class ChevronCfg(DurationScanCfg):
    """Common MHz frequency scan with fixed signed gains and relative phase.

    Both frequencies change together; drive waveforms and timing remain source
    values except for the explicit common duration axis. No automatic gate or
    frequency selection is inferred from a single population minimum.
    """

    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            unit="MHz",
            label="Frequency",
            description="填入 common drive frequency 起訖值，單位 MHz。先用 conditional spectroscopy 選定掃描範圍。",
        ),
    ]

    @model_validator(mode="after")
    def validate_frequency(self) -> Self:
        """Require finite common frequency coordinates."""
        if not np.all(np.isfinite(sweep2array(self.frequency))):
            raise ValueError("Common frequencies must be finite")
        return self


class ChevronExp(Experiment[RabiScanResult, ChevronCfg]):
    """Stateless common-frequency acquisition and native persistence."""

    AXES_SPEC: ClassVar[AxesSpec[RabiScanResult, ChevronCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("tone_indices", "Tone", "", dtype=np.int64),
            AxesSpec.Axis("shot_indices", "Shot", "", dtype=np.int64),
            AxesSpec.Axis("durations", "Drive duration", "s", 1e-6),
            AxesSpec.Axis("coordinates", "Common drive frequency", "Hz", 1e6),
        ),
        z=AxesSpec.Z("signals", "Raw IQ", "a.u."),
        result_type=RabiScanResult,
        cfg_type=ChevronCfg,
        cfg_schema_version="3.0",
        tag="two_qubit/chevron",
    )

    def run(self, config: ChevronCfg, *, context: RunContext) -> RabiScanResult:
        """Acquire B11 → dual drive → double Xπ → mux for frequency/duration/round.

        Preserve all raw shots and full tones, including partial cancellation.
        Caller must check context.scope.stop.raise_if_error(). No input mutation.
        """
        frequencies = sweep2array(config.frequency)
        drives = [
            DualLineCfg(
                f1=config.drive.f1.with_updates(freq=frequency),
                f2=config.drive.f2.with_updates(freq=frequency),
            )
            for frequency in frequencies
        ]
        if len(drives) != len(frequencies) or not np.all(np.isfinite(frequencies)):
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
                    m = cfg
                    builder.add_pulse("prepare_f1", m.x180_f1)
                    builder.add_pulse("prepare_f2", m.x180_f2)
                    builder.add(dual_line_drive(drive, frequency=float(drive.f1.freq)))
                    builder.add_pulse("contrast_f1", m.x180_f1)
                    builder.add_pulse("contrast_f2", m.x180_f2)
                    builder.add(MuxReadout("readout", m.readout))
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
            frequencies,
            durations,
            np.arange(config.rounds * cfg.reps, dtype=np.int64),
            np.arange(ntones, dtype=np.int64),
            buffer.array.reshape(
                len(drives), len(durations), config.rounds * cfg.reps, ntones
            ),
        )

    def analyze(
        self,
        source: RunRecord[ChevronCfg, RabiScanResult],
        options: None,
        *,
        plots: Plots,
    ) -> RabiAnalysis:
        """Classify complete points and plot return using saved cfg only.

        Fit complete duration traces with the undamped cosine, excluding rates
        below 1/saved duration span. Return diagnostic rates and fit_quality,
        never an automatic frequency/duration selection or pulse proposal.
        Missing source cfg, invalid raw or malformed axes raise ValueError.
        Incomplete points keep raw but are not classified or replaced by zero.
        """
        del options
        if source.cfg is None:
            raise ValueError(
                "Rabi analysis requires saved cfg; current Form/library is not a substitute"
            )
        answer = joint_return(
            source.cfg, source.result, sweep2array(source.cfg.frequency)
        )
        plot_return_data(
            source.result,
            answer,
            coordinate_label="Common drive frequency (MHz)",
            map_name="chevron",
            plots=plots,
        )
        rates = fit_duration_rates(source.result, answer)
        plot_duration_rates(
            source.result,
            rates,
            coordinate_label="Common drive frequency (MHz)",
            plots=plots,
        )
        return RabiAnalysis(answer, rates)
