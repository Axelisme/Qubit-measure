"""Mux Ramsey core: common requested delay, independent x90 pulses and detunes."""

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
    SweepCfg,
    sweep2param,
)
from zcu_tools.program.v2.modules import Join, MuxReadout

from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2._support.mux_coherence import (
    MuxCoherenceResult,
    MuxCoherenceToneCfg,
    validate_coherence_scan,
)
from zcu_lab.v2._support.mux_coherence_analysis import (
    MuxCoherenceFit,
    analyze_coherence,
)
from zcu_lab.v2._support.mux_coherence_persistence import coherence_spec
from zcu_lab.v2._support.mux_liveplot import draw_mux_magnitudes
from zcu_lab.v2._support.mux_sweep_cfg import MuxDriveModuleCfg, participant_indices
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyAnalyzeOptions


class MuxT2RamseyCfg(ProgramV2Cfg, ExpCfgModel):
    """Named x90 drives and a shared delay sweep for mux Ramsey fringes.

    modules holds the optional reset and full mux readout. tones follows the
    original readout order; every tone has a unique nonempty name, a required
    x90 drive on an independent generator, and a MHz detune for its final phase
    ramp. All drives must be valid, including nonparticipants, and cannot share
    the readout generator. participants is a nonempty list of unique tone names
    in result-row order (Q3/Q1 selects original positions 2/0).
    delay is the common finite, nonnegative requested-us sweep. Inherited reps
    and rounds are positive; initial_delay and relax_delay are finite,
    nonnegative us before acquisition and between repetitions respectively.
    """

    modules: MuxDriveModuleCfg
    tones: Annotated[list[MuxCoherenceToneCfg], FieldMeta(element_label="name")]
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
    """Measure mux Ramsey fringes and fit each participating qubit's T2*."""

    AXES_SPEC = coherence_spec(MuxT2RamseyCfg, tag="mux/time_domain/t2ramsey")
    Options: ClassVar[type[T2RamseyAnalyzeOptions]] = T2RamseyAnalyzeOptions

    def run(self, config: MuxT2RamseyCfg, *, context: RunContext) -> MuxCoherenceResult:
        """Measure x90 → shared delay → phase-ramped x90 → mux readout.

        config supplies named x90 drives, MHz detunes and a requested-us delay
        sweep; it is copied before use. Each pulse stage starts together, and
        the delay begins after the longest first pulse. Return requested times
        (P,), average complex IQ (K,P), original tone indices, actual MHz detunes
        and source frequencies in participant order. context supplies hardware
        and plots; check its cancel_signal.raise_if_error() before reporting
        success. Cancellation keeps completed-round means and NaN for no data.
        """
        # 1. Keep participant order while selecting drives and mux readout tones.
        cfg = MuxT2RamseyCfg.model_validate(config.model_dump())
        indices = participant_indices(
            [tone.name for tone in cfg.tones], cfg.participants
        )
        tones = [cfg.tones[index] for index in indices]
        drives = [tone.drive for tone in tones]
        times = sweep2array(cfg.delay, "none")
        _ = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg, "scaler": 1.0})
        detunes = np.asarray(
            [
                actual_detune(
                    cfg.delay,
                    detune=tone.detune,
                    gen_ch=drive.ch,
                    soccfg=context.soccfg,
                    time_scaler=1.0,
                )
                for tone, drive in zip(tones, drives, strict=True)
            ],
            dtype=np.float64,
        )
        frequencies = np.asarray(
            [float(drive.freq) for drive in drives], dtype=np.float64
        )
        readout = cfg.modules.readout.with_updates(
            tones=[cfg.modules.readout.tones[index] for index in indices]
        )

        # 2. Plot requested delay for each original tone, including partial data.
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
            # 3. Join each x90 stage so the common wait follows its longest pulse.
            length = sweep2param("length", cfg.delay)
            builder = (
                sched.prog_builder(context.soc, context.soccfg)
                .add_reset("reset", cfg.modules.reset)
                .add(
                    Join(
                        *[
                            Pulse(f"pi2_first_{index}", drive)
                            for index, drive in zip(indices, drives, strict=True)
                        ]
                    ),
                    Delay("delay_first", 1.0 * length),
                    Join(
                        *[
                            Pulse(
                                f"pi2_final_{index}",
                                drive.with_updates(
                                    phase=drive.phase + 360 * tone.detune * length
                                ),
                            )
                            for index, tone, drive in zip(
                                indices, tones, drives, strict=True
                            )
                        ]
                    ),
                    MuxReadout("readout", readout),
                )
                .declare_sweep("length", cfg.delay)
            )

            # 4. Average rounds into tone-major IQ without filling missing points.
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
        source: RunRecord[MuxT2RamseyCfg, MuxCoherenceResult],
        options: T2RamseyAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxCoherenceFit | MuxFitFailure, ...]:
        """Fit each qubit's Ramsey decay and fringe rate from source.

        source contains requested delays and measured IQ, with or without cfg.
        options.fit_fringe selects a damped cosine instead of a decay;
        fit_phase frees its phase instead of choosing 0/180 degrees from the
        nearest-zero signal. Return T2* (us), fringe rate (MHz), uncertainties
        and fit quality, or a failure reason, for each original tone index.
        Draw fit_i in plots; invalid sources raise.
        """
        return analyze_coherence(
            source.result,
            mode="ramsey",
            fit_fringe=options.fit_fringe,
            fit_phase=options.fit_phase,
            plots=plots,
        )


def _averaged_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    """Convert normalized mux IQ from (point, tone, I/Q) to (tone, point)."""
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_iq(prog, acquired, read_index=0)
            return (iq[..., 0] + 1j * iq[..., 1]).T
    raise ValueError("Coherence program is missing its named MuxReadout")
