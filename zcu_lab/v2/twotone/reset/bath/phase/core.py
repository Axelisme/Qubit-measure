from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.base import cosfunc, fitcos
from zcu_tools.analysis.shot_uncertainty import iq_shot_sem
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    BathReset,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
from zcu_tools.utils.process import find_rotate_angle

from zcu_lab.v2.support import live_errorbar


@dataclass(frozen=True)
class PhaseResult:
    """Absolute pi/2 phases in degrees and mean complex IQ (phase,)."""

    phases: NDArray[np.float64]
    signals: NDArray[np.complex128]
    # Per point: [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]; absent in mean-only files.
    iq_moments: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class PhaseAnalysis:
    max_phase: float
    min_phase: float


def _projection(signals: NDArray[np.complex128]) -> tuple[float, float]:
    """Return the common IQ projection direction, or I for insufficient data."""
    try:
        angle = find_rotate_angle(signals)
    except ValueError:
        return (1.0, 0.0)
    return (float(np.cos(angle)), float(np.sin(angle)))


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project the mean IQ along the sweep's principal axis, or I if unavailable."""
    a, b = _projection(signals)
    return a * signals.real + b * signals.imag


class PhaseCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a preparation reset before each shot, or leave blank to skip it.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Preparation pulse",
            description="Select the pulse after preparation reset, such as calibrated Xπ, or leave blank to skip it.",
        ),
    ] = None
    tested_reset: Annotated[
        BathReset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Fill in the fixed cavity and qubit tones and the π/2 pulse played after the cavity tone.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Fill in the calibrated resonator readout for the end of each shot.",
        ),
    ]
    phase: Annotated[
        SweepCfg,
        FieldMeta(
            label="Tomography π/2 phase",
            unit="deg",
            description="Enter the absolute phase range of the final π/2 pulse (deg).",
        ),
    ]

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per sweep point and sequence branch in each acquisition round (count).",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds to average (count).",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the delay after each shot to let the qubit relax (µs).",
        ),
    ]


_SCAN_AXIS = VariableAxisSpec(field_name="phases", label="Phase", unit="deg")
_MOMENT_COLUMNS = (0, 1, 2, 3, 4, 5)
_XLABEL = "Tomography π/2 phase (deg)"
_YLABEL = "Mean IQ on principal axis (a.u.)"
_TITLE = "Bath reset phase · qubit · IQ projection"


def _restore(data: GroupedLoadData[PhaseCfg]) -> PhaseResult:
    """Restore the historical mean and, if present, moments on the same scan grid."""
    mean = data.variable("data")
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return PhaseResult(mean.axes[0], mean.z)
    columns, scan = moments.axes
    if not np.array_equal(columns, _MOMENT_COLUMNS) or not np.array_equal(
        scan, mean.axes[0]
    ):
        raise ValueError(
            "iq_moments axes do not match the stored scan and moment columns"
        )
    return PhaseResult(mean.axes[0], mean.z, moments.z)


class PhaseExp(Experiment[PhaseResult, PhaseCfg]):
    # ADR-0069: retain the historical mean variable and single-log Labber export.
    AXES_SPEC: ClassVar[GroupedAxesSpec[PhaseResult, PhaseCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "data",
                (_SCAN_AXIS,),
                VariableZSpec(
                    "signals", label="Signal", unit="a.u.", dtype=np.complex128
                ),
            ),
            VariableSpec(
                "iq_moments",
                (
                    VariableAxisSpec(label="Moment", unit="", values=_MOMENT_COLUMNS),
                    _SCAN_AXIS,
                ),
                VariableZSpec("iq_moments", label="IQ moments", unit=""),
                required=False,
            ),
        ),
        result_type=PhaseResult,
        cfg_type=PhaseCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/bath/phase",
        result_builder=_restore,
        labber_variables=("data",),
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
        # 1. Quantize the final pi/2 phase on its own generator channel.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg

        phases = sweep2array(
            cfg.phase,
            "phase",
            {
                "soccfg": soccfg,
                "gen_ch": cfg.tested_reset.pi2_cfg.ch,
            },
        )

        # 2. Display the phase fringe without discarding the complex IQ response.
        plots = context.plots
        fig, ax = plots.subplots(num="measurement")
        ax.set(xlabel=_XLABEL, ylabel=_YLABEL, title=_TITLE)
        viewer = live_errorbar(ax)

        @plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Update the matching point means and shot SEM from completed rounds."""
            signals = split_sums(moments)[1]
            direction = _projection(signals)
            contrast = direction[0] * signals.real + direction[1] * signals.imag
            sem = iq_shot_sem(moments, direction=direction)
            shots = int(np.nan_to_num(moments[..., 0]).max())
            title = ax.set_title(f"{_TITLE}\nRetained shots per point: {shots} count")
            return (*viewer.set(phases, contrast, sem), title)

        fig.show()
        buffer = SignalBuffer(
            (len(phases), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
        )

        # 3. Measure the reset phase fringe with fixed simultaneous bath tones.
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            phase = builder.add_sweep("phase", cfg.phase)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(BathReset("tested_reset", cfg.tested_reset, pi2_phase=phase))
            builder.emit(Readout("readout", cfg.readout))
            program = builder.build()
            reducer = shot_sums(program, (IQ("readout", offset_corrected=True),))
            sched.run_acquire_summed(program, soc, reducer=reducer, rounds=cfg.rounds)

        moments = buffer.array
        return PhaseResult(phases, split_sums(moments)[1], moments)

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
        # 1. Find maximum- and minimum-contrast phases from a fixed-period tomography fringe.
        del options
        result = source.result

        phases, signals = result.phases, result.signals

        real_signals = bathreset_signal2real(signals)

        pOpt, covariance = fitcos(
            phases, real_signals, fixedparams=[None, None, 1 / 360, None]
        )
        full_phase = np.linspace(phases.min(), phases.max(), 200)
        y_fit = cosfunc(full_phase, *pOpt)
        phase_error = np.sqrt(covariance[3, 3])

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

        # 2. Assess the phase-fringe fit and its equivalent extrema nearest zero degrees.
        fig, ax = plots.subplots(num="fit")

        if result.iq_moments is None:
            ax.plot(phases, real_signals, ".-", markersize=5, label="Measured")
        else:
            sem = iq_shot_sem(result.iq_moments, direction=_projection(signals))
            ax.errorbar(
                phases, real_signals, yerr=sem, fmt=".-", markersize=5, label="Measured"
            )
        ax.plot(full_phase, y_fit, "-", label="360° cosine fit")
        # Draw each equivalent phase in the turn nearest the observed scan.
        center = (float(full_phase[0]) + float(full_phase[-1])) / 2
        ax.axvline(
            max_phase + 360 * round((center - max_phase) / 360),
            color="C2",
            linestyle="--",
            label=f"Maximum = {max_phase:.2f} ± {phase_error:.2g} deg",
        )
        ax.axvline(
            min_phase + 360 * round((center - min_phase) / 360),
            color="C3",
            linestyle="--",
            label=f"Minimum = {min_phase:.2f} ± {phase_error:.2g} deg",
        )
        ax.set(xlabel=_XLABEL, ylabel=_YLABEL, title=_TITLE)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)

        fig.tight_layout()

        return PhaseAnalysis(max_phase=max_phase, min_phase=min_phase)
