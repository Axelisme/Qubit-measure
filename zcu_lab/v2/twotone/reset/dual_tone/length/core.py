from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
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
    config,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    TwoPulseReset,
    shot_sums,
    split_sums,
)

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.twotone.reset._projection import principal_iq_direction


@dataclass(frozen=True)
class LengthResult:
    """Pulse1 lengths in us and mean complex IQ (length,)."""

    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    # Per point: [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]; absent in mean-only files.
    iq_moments: NDArray[np.float64] | None = None


def reset_length_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project the mean IQ along the sweep's principal axis, or I if unavailable."""
    a, b = principal_iq_direction(signals)
    return a * signals.real + b * signals.imag


class LengthCfg(ConfigBase):
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
        TwoPulseReset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Fill in the two simultaneous reset pulses with calibrated carriers, gains and duration difference.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Fill in the calibrated resonator readout for the end of each shot.",
        ),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="Tone 1 pulse duration",
            unit="µs",
            description="Enter the tone 1 duration range (µs); tone 2 keeps its configured duration difference.",
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


_SCAN_AXIS = VariableAxisSpec(
    field_name="lengths", label="Length", unit="s", scale=1e-6
)
_MOMENT_COLUMNS = (0, 1, 2, 3, 4, 5)
_XLABEL = "Tone 1 pulse duration (µs)"
_YLABEL = "Mean IQ on principal axis (a.u.)"
_TITLE = "Dual-tone reset · qubit · IQ projection"


def _restore(data: GroupedLoadData[LengthCfg]) -> LengthResult:
    """Restore the historical mean and, if present, moments on the same scan grid."""
    mean = data.variable("data")
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return LengthResult(mean.axes[0], mean.z)
    columns, scan = moments.axes
    if not np.array_equal(columns, _MOMENT_COLUMNS) or not np.array_equal(
        scan, mean.axes[0]
    ):
        raise ValueError(
            "iq_moments axes do not match the stored scan and moment columns"
        )
    return LengthResult(mean.axes[0], mean.z, moments.z)


class LengthExp(Experiment[LengthResult, LengthCfg]):
    # Length stores seconds on disk; Result holds us -> scale=1e-6
    # ADR-0069: retain the historical mean variable and single-log Labber export.
    AXES_SPEC: ClassVar[GroupedAxesSpec[LengthResult, LengthCfg]] = GroupedAxesSpec(
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
        result_type=LengthResult,
        cfg_type=LengthCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/dual_tone/length",
        result_builder=_restore,
        labber_variables=("data",),
    )

    def run(self, cfg: LengthCfg, *, context: RunContext) -> LengthResult:
        """Measure the dual-tone reset response versus the first pulse length.

        Apply optional reset and initialization, start both tested reset
        pulses together, then read out after the longer pulse finishes.
        Sweep pulse1 length while preserving pulse2-minus-pulse1 length.
        Return pulse1 lengths in us and mean complex IQ shaped (Nlength,).
        Warn if channel time grids differ; raise ValueError if the derived
        pulse2 length is negative. The IQ response is not reset fidelity.
        """
        # 1. Check both channel grids and keep the configured pulse-duration difference.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        length_sweep = cfg.length

        reset_cfg = cfg.tested_reset
        pulse1_cfg = reset_cfg.pulse1_cfg
        pulse2_cfg = reset_cfg.pulse2_cfg
        length_diff = pulse2_cfg.waveform.length - pulse1_cfg.waveform.length

        pulse1_lengths = sweep2array(
            length_sweep, "time", dict(soccfg=soccfg, gen_ch=pulse1_cfg.ch)
        )
        pulse2_lengths = sweep2array(
            length_sweep, "time", dict(soccfg=soccfg, gen_ch=pulse2_cfg.ch)
        )
        if not np.allclose(pulse1_lengths, pulse2_lengths, atol=1e-2):
            warnings.warn(
                "Sweep lengths for pulse1 and pulse2 are different. This may lead to unexpected results.",
                stacklevel=2,
            )
        if np.any(pulse2_lengths + length_diff < 0):
            raise ValueError(
                "Find negative length in pulse2 while sweeping pulse1 length. Please check the sweep configuration."
            )
        lengths = pulse1_lengths  # Use pulse1 lengths as the x-axis values

        # 2. Retain complex IQ while displaying its common-axis projection.
        plots = context.plots
        fig, ax = plots.subplots(num="measurement")
        ax.set(xlabel=_XLABEL, ylabel=_YLABEL, title=_TITLE)
        viewer = live_errorbar(ax)

        @plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Update the matching point means and shot SEM from completed rounds."""
            signals = split_sums(moments)[1]
            direction = principal_iq_direction(signals)
            contrast = direction[0] * signals.real + direction[1] * signals.imag
            sem = iq_shot_sem(moments, direction=direction)
            return viewer.set(lengths, contrast, sem)

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
        )

        # 3. Measure duration-dependent reset response while preserving the tone-length offset.
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length = builder.add_sweep("length", cfg.length)
            tested_reset_cfg = deepcopy(cfg.tested_reset)
            tested_reset_cfg.pulse1_cfg.set_param("length", length)
            tested_reset_cfg.pulse2_cfg.set_param("length", length + length_diff)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.delay_by_timeline()
            builder.emit(TwoPulseReset("tested_reset", tested_reset_cfg))
            builder.delay_by_timeline()
            builder.emit(Readout("readout", cfg.readout))
            program = builder.build()
            reducer = shot_sums(program, (IQ("readout", offset_corrected=True),))
            sched.run_acquire_summed(program, soc, reducer=reducer, rounds=cfg.rounds)

        moments = buffer.array
        return LengthResult(lengths, split_sums(moments)[1], moments)

    def analyze(
        self,
        source: RunRecord[LengthCfg, LengthResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot the projected IQ response versus pulse1 length in us.

        Omit unmeasured NaN points and publish fit without fitting or returning
        an optimal duration. source.cfg may be None.
        """
        # 1. Recover the measured duration dependence without unacquired length points.
        del options
        result = source.result

        lens, signals = result.lengths, result.signals

        # Discard NaNs left by a possible early abort.
        val_mask = ~np.isnan(signals)
        lens = lens[val_mask]
        signals = signals[val_mask]

        real_signals = reset_length_signal2real(signals)

        # 2. Inspect reset contrast versus the common duration scan.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        if result.iq_moments is None:
            ax.plot(lens, real_signals, ".-", markersize=5)
        else:
            sem = iq_shot_sem(
                result.iq_moments[val_mask], direction=principal_iq_direction(signals)
            )
            ax.errorbar(lens, real_signals, yerr=sem, fmt=".-", markersize=5)
        ax.set(xlabel=_XLABEL, ylabel=_YLABEL, title=_TITLE)
        fig.tight_layout()
