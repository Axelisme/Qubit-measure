from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.shot_uncertainty import iq_shot_sem
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    RunRecord,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    LoadValue,
    Pulse,
    Readout,
    Reset,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
from zcu_tools.utils.process import find_rotate_angle

from zcu_lab.v2.support import live_errorbar


@dataclass(frozen=True)
class ZigZagResult:
    times: NDArray[np.int64]
    signals: NDArray[np.complex128]
    iq_moments: NDArray[np.float64] | None = None
    """Retained shot sums [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²] per point; None for mean-only files."""


class ZigZagCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    X90_pulse: Annotated[Pulse.Cfg, FieldMeta(label="X90 Pulse")]
    X180_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="X180 Pulse")] = None
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    n_times: Annotated[
        int,
        FieldMeta(
            label="Max repetitions",
            unit="count",
            description="Repetition counts run from zero to this value.",
        ),
    ] = Field(ge=0)
    repeat_on: Annotated[
        Literal["X90_pulse", "X180_pulse"],
        FieldMeta(
            label="Repeat pulse",
            description="X90 repeats in pairs; each count adds one pi rotation.",
        ),
    ] = "X180_pulse"

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]


def _direction(signals: NDArray[np.complex128]) -> tuple[float, float]:
    """Project along the principal IQ axis, or use I if fewer than two points."""
    try:
        angle = find_rotate_angle(signals)
    except ValueError:
        return (1.0, 0.0)
    return (float(np.cos(angle)), float(np.sin(angle)))


_TRACE_AXIS = VariableAxisSpec(
    label="Times",
    unit="a.u.",
    field_name="times",
    dtype=np.int64,
)
_MOMENT_COLUMNS = tuple(range(6))


def _build_result(data: GroupedLoadData[ZigZagCfg]) -> ZigZagResult:
    """Restore the measured trace and any shot moments on the same native axis."""
    trace = data.variable("data")
    (coordinates,) = trace.axes
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return ZigZagResult(coordinates, trace.z)
    columns, moment_coordinates = moments.axes
    if not np.array_equal(coordinates, moment_coordinates) or not np.array_equal(
        columns, _MOMENT_COLUMNS
    ):
        raise ValueError(
            "/vars/iq_moments: expected columns 0–5 on the /vars/data grid"
        )
    return ZigZagResult(coordinates, trace.z, iq_moments=moments.z)


class ZigZagExp(Experiment[ZigZagResult, ZigZagCfg]):
    # times is an int counter (a.u.) -> Axis dtype int64, scale IDENTITY
    # ADR-0069: preserve the original trace; shot moments are optional on load.
    AXES_SPEC: ClassVar[GroupedAxesSpec[ZigZagResult, ZigZagCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "data",
                (_TRACE_AXIS,),
                VariableZSpec(
                    "signals", label="Signal", unit="a.u.", dtype=np.complex128
                ),
            ),
            VariableSpec(
                "iq_moments",
                (
                    VariableAxisSpec(label="Moment", unit="", values=_MOMENT_COLUMNS),
                    _TRACE_AXIS,
                ),
                VariableZSpec("iq_moments", label="IQ moments", unit=""),
                required=False,
            ),
        ),
        result_builder=_build_result,
        labber_variables=("data",),
        result_type=ZigZagResult,
        cfg_type=ZigZagCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/zigzag",
    )

    def run(self, cfg: ZigZagCfg, *, context: RunContext) -> ZigZagResult:
        """Measure accumulated rotation errors versus the repeated gate count.

        Reset, apply X90, repeat the selected gate, then read out mean complex IQ.
        Return integer counts from zero through n_times and the corresponding IQ;
        each count adds two X90 pulses or one X180 pulse.
        The selected repeat pulse must exist.
        Cancelled acquisition preserves completed-round averages or NaN if empty.
        """
        # 1. Prepare the count axis and live view without changing the caller's cfg.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        times = np.arange(0, cfg.n_times + 1)

        # 2. Monitor repeated-gate contrast on a common IQ projection axis.
        fig, ax = context.plots.subplots(num="measurement")
        gate = "X90 pairs" if cfg.repeat_on == "X90_pulse" else "X180"
        experiment = f"ZigZag · {gate}"
        ax.set(
            title=experiment,
            xlabel="Gate repetitions (count)",
            ylabel="IQ projection (a.u.)",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        viewer = live_errorbar(ax)

        @context.plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Update the projected IQ and shot SEM."""
            _, mean = split_sums(moments)
            i, q = _direction(mean)
            sem = iq_shot_sem(moments, direction=(i, q))
            return viewer.set(
                times.astype(np.float64), mean.real * i + mean.imag * q, sem
            )

        fig.show()
        signals_buffer = SignalBuffer(
            (len(times), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
        )
        # 3. Amplify rotation errors by repeating the selected gate after the initial X90.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            repeat_pulse = getattr(cfg, cfg.repeat_on)
            if repeat_pulse is None:
                raise ValueError(f"Repeat on pulse {cfg.repeat_on} not found")
            counts = [
                n * (2 if cfg.repeat_on == "X90_pulse" else 1)
                for n in range(cfg.n_times + 1)
            ]
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("times", len(times))
            builder.emit(
                LoadValue(
                    "load_repeat_count",
                    values=counts,
                    idx_reg="times",
                    val_reg="repeat_count",
                )
            )
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("X90_pulse", cfg.X90_pulse))
            # Legacy repeats flush the reference at entry and after every gate.
            builder.delay_by_timeline()
            with builder.repeat(
                "zigzag_loop", "repeat_count", range_hint=(0, max(counts))
            ):
                builder.emit(Pulse(cfg.repeat_on, repeat_pulse))
                builder.delay_by_timeline()
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                soc,
                rounds=cfg.rounds,
                reducer=shot_sums(prog, (IQ("readout", offset_corrected=True),)),
            )

        return ZigZagResult(
            times=times,
            signals=split_sums(signals_buffer.array)[1],
            iq_moments=signals_buffer.array,
        )

    def analyze(
        self, source: RunRecord[ZigZagCfg, ZigZagResult], options: None, *, plots: Plots
    ) -> None:
        """Redraw the source trace against gate repetition count without a fit.

        Pass None for options. The measurement figure shows projected mean IQ
        and, when retained moments are available, shot SEM along the same axis.
        Missing points stay missing. A source without cfg omits the pulse label.
        """
        # 1. Project the stored IQ and any retained shot moments on the Live axis.
        del options
        result = source.result
        i, q = _direction(result.signals)
        projected = result.signals.real * i + result.signals.imag * q
        title = "ZigZag"
        if source.cfg is not None:
            gate = "X90 pairs" if source.cfg.repeat_on == "X90_pulse" else "X180"
            title = f"{title} · {gate}"
        if result.iq_moments is None:
            sem = np.full(projected.shape, np.nan)
        else:
            sem = iq_shot_sem(result.iq_moments, direction=(i, q))

        # 2. Redraw the terminal view without presenting it or estimating a calibration.
        _, ax = plots.subplots(num="measurement")
        ax.set(
            title=title,
            xlabel="Gate repetitions (count)",
            ylabel="IQ projection (a.u.)",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        live_errorbar(ax).set(result.times.astype(np.float64), projected, sem)
