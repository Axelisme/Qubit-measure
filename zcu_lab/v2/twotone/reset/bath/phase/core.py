from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.base import cosfunc, fitcos
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting import live_line
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import BathReset, Pulse, Readout, Reset, SweepCfg, sweep2param
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class PhaseResult:
    """Absolute pi/2 phases in degrees and mean complex IQ (phase,)."""

    phases: NDArray[np.float64]
    signals: NDArray[np.complex128]


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


@dataclass(frozen=True)
class PhaseAnalysis:
    max_phase: float
    min_phase: float


class PhaseCfg(ConfigBase):
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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset", group="Modules")] = None
    init_pulse: Annotated[
        Pulse.Cfg | None, FieldMeta(label="Init pulse", group="Modules")
    ] = None
    tested_reset: Annotated[
        BathReset.Cfg, FieldMeta(label="Tested reset", group="Modules")
    ]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout", group="Modules")]
    phase: Annotated[SweepCfg, FieldMeta(label="Phase", unit="deg", group="Sweep")]


class PhaseExp(Experiment[PhaseResult, PhaseCfg]):
    AXES_SPEC: ClassVar[AxesSpec[PhaseResult, PhaseCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("phases", "Phase", "deg"),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=PhaseResult,
        cfg_type=PhaseCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/bath/phase",
    )

    def run(
        self,
        config: PhaseCfg,
        *,
        context: RunContext,
    ) -> PhaseResult:
        """Measure the bath-reset response to the final pi/2 phase.

        Apply optional reset and initialization, start the cavity and qubit
        tones together, apply the phase-swept pi/2 pulse after the cavity
        tone, then read out.
        Return phases in degrees and mean complex IQ shaped (Nphase,).
        The resulting fringe selects opposite preparation directions through
        cosine analysis; it does not measure reset fidelity.
        """
        # Quantize the final pi/2 phase on its own generator channel.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg

        phases = sweep2array(
            cfg.phase,
            "phase",
            {
                "soccfg": soccfg,
                "gen_ch": modules.tested_reset.pi2_cfg.ch,
            },
        )

        # Display the phase fringe without discarding the complex IQ response.
        viewer = live_line(context.plots, "measurement", "Phase (deg)", "Signal (a.u.)")
        signals_buffer = SignalBuffer(
            (len(phases),),
            on_update=lambda data: viewer.update(phases, bathreset_signal2real(data)),
        )

        # Sweep only the final rotation; keep the simultaneous bath tones fixed.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg
            modules.tested_reset.set_param(
                "pi2_phase", sweep2param("phase", sched.cfg.phase)
            )
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    BathReset("tested_reset", modules.tested_reset),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("phase", sched.cfg.phase)
                .build_and_acquire(rounds=cfg.rounds)
            )

        return PhaseResult(phases, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[PhaseCfg, PhaseResult],
        options: None,
        *,
        plots: Plots,
    ) -> PhaseAnalysis:
        """Fit a 360-degree cosine to the projected IQ phase fringe.

        Return max_phase and min_phase in degrees, each wrapped to the nearest
        zero-centered turn, and publish fit. These are extrema of the IQ
        projection, not calibrated populations; source.cfg may be None.
        """
        del options
        result = source.result

        phases, signals = result.phases, result.signals

        real_signals = bathreset_signal2real(signals)

        pOpt, _ = fitcos(phases, real_signals, fixedparams=[None, None, 1 / 360, None])
        y_fit = cosfunc(phases, *pOpt)

        init_phase = float(pOpt[3])

        max_phase = -init_phase
        min_phase = 180 - init_phase

        while abs(max_phase) > abs(max_phase - 360):
            max_phase -= 360
        while abs(max_phase) > abs(max_phase + 360):
            max_phase += 360
        while abs(min_phase) > abs(min_phase - 360):
            min_phase -= 360
        while abs(min_phase) > abs(min_phase + 360):
            min_phase += 360

        fig, ax = plots.subplots("fit")

        ax.plot(phases, real_signals, ".-", label="data")
        ax.plot(phases, y_fit, "-", label="fit")
        ax.axvline(
            max_phase, color="C1", linestyle="--", label=f"max: {max_phase:.2f} deg"
        )
        ax.axvline(
            min_phase, color="C2", linestyle="--", label=f"min: {min_phase:.2f} deg"
        )
        ax.set_xlabel("Phase (deg)")
        ax.set_ylabel("Signal (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return PhaseAnalysis(max_phase=max_phase, min_phase=min_phase)
