from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import (
    GainSweep,
    Pulse,
    PulseReset,
    Readout,
    Reset,
    sweep2param,
)

from zcu_lab.v2.support.reset import reset_scan_buffer


@dataclass(frozen=True)
class GainResult:
    """Flux reset gains in a.u. and mean complex IQ, each shaped (gain,)."""

    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]


class GainCfg(ConfigBase):
    """Prepare the qubit, sweep a const flux reset's gain, and read mean IQ."""

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
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(label="Reset", description="Optional reset before initialization."),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Init pulse", description="Prepare e with pi; None skips preparation."
        ),
    ] = None
    tested_reset: Annotated[
        PulseReset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Zero-frequency const flux pulse at fixed length; gain is swept.",
        ),
    ]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Gain", unit="a.u.")]


class GainExp(Experiment[GainResult, GainCfg]):
    AXES_SPEC: ClassVar[AxesSpec[GainResult, GainCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("gains", "Flux reset gain", "a.u."),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=GainResult,
        cfg_type=GainCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/flux/gain",
    )

    def run(self, cfg: GainCfg, *, context: RunContext) -> GainResult:
        """Measure mean IQ versus flux reset gain at fixed pulse duration.

        Apply optional reset, initialization (normally pi), the tested const
        flux reset, then readout. Keep pre/post delays and duration unchanged.
        Return hardware-rounded gains in a.u. and mean complex IQ shaped (N,).
        Reject non-const or nonzero-frequency tested pulses before acquisition.
        The live projection is not a calibrated residual population.
        """
        cfg = deepcopy(cfg)
        pulse_cfg = cfg.tested_reset.pulse_cfg
        if pulse_cfg.waveform.style != "const" or pulse_cfg.freq != 0.0:
            raise ValueError("Flux reset requires a const pulse at zero frequency.")

        soc, soccfg = context.soc, context.soccfg
        gains = sweep2array(
            cfg.gain, "gain", {"soccfg": soccfg, "gen_ch": pulse_cfg.ch}
        )
        signals_buffer = reset_scan_buffer(
            context.plots, gains, xlabel="Flux reset gain (a.u.)"
        )

        # Sweep only the tested flux gain after optional preparation.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            sched.cfg.tested_reset.set_param(
                "gain", sweep2param("gain", sched.cfg.gain)
            )
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", sched.cfg.reset),
                    Pulse("init_pulse", sched.cfg.init_pulse),
                    PulseReset("tested_reset", sched.cfg.tested_reset),
                    Readout("readout", sched.cfg.readout),
                )
                .declare_sweep("gain", sched.cfg.gain)
                .build_and_acquire(rounds=cfg.rounds)
            )

        return GainResult(gains=gains, signals=signals_buffer.array)
