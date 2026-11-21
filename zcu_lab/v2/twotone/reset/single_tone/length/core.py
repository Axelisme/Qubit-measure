from __future__ import annotations

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
    PulseReset,
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
class LengthResult:
    """Reset pulse lengths in us and mean complex IQ (length,)."""

    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    # Per point: [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]; absent in mean-only files.
    iq_moments: NDArray[np.float64] | None = None


def _projection(signals: NDArray[np.complex128]) -> tuple[float, float]:
    """Return the common IQ projection direction, or I for insufficient data."""
    try:
        angle = find_rotate_angle(signals)
    except ValueError:
        return (1.0, 0.0)
    return (float(np.cos(angle)), float(np.sin(angle)))


def reset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project the mean IQ along the sweep's principal axis, or I if unavailable."""
    a, b = _projection(signals)
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
        PulseReset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Fill in the sideband pulse at the calibrated carrier and a fixed gain.",
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
            label="Reset pulse duration",
            unit="µs",
            description="Enter the duration range of the tested sideband pulse (µs).",
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
_XLABEL = "Reset pulse duration (µs)"
_YLABEL = "Mean IQ on principal axis (a.u.)"
_TITLE = "Single-tone reset · qubit · IQ projection"


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
                VariableZSpec("signals", label="Signal", unit="a.u."),
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
        tag="twotone/reset/single_tone/length",
        result_builder=_restore,
        labber_variables=("data",),
    )

    def run(self, cfg: LengthCfg, *, context: RunContext) -> LengthResult:
        """Measure the single-tone reset response versus pulse duration.

        Apply optional reset and initialization, the length-swept tested
        reset pulse, then read out. Return reset lengths in us and mean
        complex IQ shaped (Nlength,). The curve shows duration dependence,
        not a calibrated reset fidelity; analysis only plots the response.
        """
        # 1. Quantize the tested reset duration on its generator channel.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        length_sweep = cfg.length

        pulse_cfg = cfg.tested_reset.pulse_cfg

        lengths = sweep2array(
            length_sweep, "time", dict(soccfg=soccfg, gen_ch=pulse_cfg.ch)
        )

        # 2. Retain complex IQ while displaying the duration-dependent projection.
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
            return (*viewer.set(lengths, contrast, sem), title)

        fig.show()
        buffer = SignalBuffer(
            (len(lengths), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
        )

        # 3. Sweep only the tested reset length after the optional preparation.
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length = builder.add_sweep("length", cfg.length)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(PulseReset("tested_reset", cfg.tested_reset, length=length))
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
        """Plot the projected reset IQ response versus pulse duration in us.

        Omit unmeasured NaN points and publish fit without fitting or returning
        an optimal length. source.cfg may be None.
        """
        # 1. Recover the projected reset-duration response from acquired points only.
        del options
        result = source.result

        lens, signals = result.lengths, result.signals

        # Discard NaNs left by a possible early abort.
        val_mask = ~np.isnan(signals)
        lens = lens[val_mask]
        signals = signals[val_mask]

        real_signals = reset_signal2real(signals)

        # 2. Inspect the sideband-reset response as its duration changes.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        if result.iq_moments is None:
            ax.plot(lens, real_signals, ".-", markersize=5)
        else:
            sem = iq_shot_sem(
                result.iq_moments[val_mask], direction=_projection(signals)
            )
            ax.errorbar(lens, real_signals, yerr=sem, fmt=".-", markersize=5)
        ax.set(xlabel=_XLABEL, ylabel=_YLABEL, title=_TITLE)
        fig.tight_layout()
