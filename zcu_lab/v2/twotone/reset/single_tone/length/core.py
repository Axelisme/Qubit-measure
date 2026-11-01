from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    PulseReset,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)

from zcu_lab.v2.support.reset import reset_scan_buffer, reset_signal2real


@dataclass(frozen=True)
class LengthResult:
    """Reset pulse lengths in us and mean complex IQ (length,)."""

    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


class LengthCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init pulse")] = None
    tested_reset: Annotated[PulseReset.Cfg, FieldMeta(label="Tested reset")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Length", unit="us")]

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


class LengthExp(Experiment[LengthResult, LengthCfg]):
    # length stores us in-memory, seconds on disk -> scale=1e-6 (1e-6)
    AXES_SPEC: ClassVar[AxesSpec[LengthResult, LengthCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("lengths", label="Length", unit="s", scale=1e-6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=LengthResult,
        cfg_type=LengthCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/single_tone/length",
    )

    def run(self, cfg: LengthCfg, *, context: RunContext) -> LengthResult:
        """Measure the single-tone reset response versus pulse duration.

        Apply optional reset and initialization, the length-swept tested
        reset pulse, then read out. Return reset lengths in us and mean
        complex IQ shaped (Nlength,). The curve shows duration dependence,
        not a calibrated reset fidelity; analysis only plots the response.
        """
        # Quantize the tested reset duration on its generator channel.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        length_sweep = cfg.length

        pulse_cfg = cfg.tested_reset.pulse_cfg

        lengths = sweep2array(
            length_sweep, "time", dict(soccfg=soccfg, gen_ch=pulse_cfg.ch)
        )

        # Retain complex IQ while displaying the duration-dependent projection.
        signals_buffer = reset_scan_buffer(context.plots, lengths, xlabel="Length (us)")

        # Sweep only the tested reset length after the optional preparation.
        with Schedule( signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length = builder.add_sweep("length", cfg.length)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(PulseReset("tested_reset", cfg.tested_reset, length=length))
            builder.emit(Readout("readout", cfg.readout))
            sched.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return LengthResult(lengths, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[LengthCfg, LengthResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot the projected reset IQ response versus pulse duration in us.

        Omit unmeasured NaN points and publish fit without fitting or returning
        an optimal length. source.cfg may be None.
        """
        del options
        result = source.result

        lens, signals = result.lengths, result.signals

        # Discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        lens = lens[val_mask]
        signals = signals[val_mask]

        real_signals = reset_signal2real(signals)

        fig, ax = plots.subplots("fit", figsize=config.figsize)
        ax.plot(lens, real_signals, marker=".")
        ax.set_xlabel("ProbeTime (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()
