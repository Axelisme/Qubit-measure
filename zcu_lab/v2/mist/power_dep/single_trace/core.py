from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, TimelineBuilder

from zcu_lab.v2.support import live_line

_TITLE = "MIST · Resonator probe · IQ displacement"
_GAIN_LABEL = "Probe gain (a.u.)"
_IQ_LABEL = "Mean IQ displacement (a.u.)"


@dataclass(frozen=True)
class PowerDepResult:
    """Hardware probe gains and mean complex readout IQ at each gain."""

    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]


@pydantic_dataclass(frozen=True)
class PowerDepAnalyzeOptions:
    g0: complex | None = None
    e0: complex | None = None
    ac_coeff: float | None = None


def _reference_iq(signals: NDArray[np.complex128]) -> complex:
    points = max(len(signals) // 20, 1)
    return complex(signals[:points].mean())


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return mean IQ distances from the complex mean of the first 5% of scan points (at least one)."""
    # ADR-0069: distance preserves probe-induced shifts orthogonal to a fixed IQ axis.
    return np.abs(signals - _reference_iq(signals))


class PowerDepCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a reset before preparation, or leave off to skip it.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Preparation pulse",
            description="Select a qubit pulse played before the resonator probe, or leave off to skip preparation.",
        ),
    ] = None
    probe_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Resonator probe",
            description="Enter the resonator drive played before readout; Probe gain sweep sets its gain.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the calibrated readout that measures complex IQ after the probe.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Probe gain sweep",
            unit="a.u.",
            description="Enter the start and stop gains (a.u.) and the number of scan points.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the number of shots per gain point per round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter how many times to acquire the full gain sweep and average its IQ.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relaxation wait",
            unit="µs",
            decimals=3,
            description="Enter the wait (µs) after each shot's readout before the next shot.",
        ),
    ]


class PowerDepExp(Experiment[PowerDepResult, PowerDepCfg]):
    Options: ClassVar[type[PowerDepAnalyzeOptions]] = PowerDepAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[PowerDepResult, PowerDepCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("gains", label="Drive Power", unit="a.u."),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=PowerDepResult,
        cfg_type=PowerDepCfg,
        cfg_schema_version="3.0",
        tag="mist/gain",
    )

    def run(
        self,
        config: PowerDepCfg,
        *,
        context: RunContext,
    ) -> PowerDepResult:
        """Measure a MIST response trace versus probe gain.

        Reset, apply the optional init pulse, apply the gain-swept probe,
        then read out. The board sweeps gain while the other pulse settings
        stay fixed. Return hardware gain coordinates and mean complex IQ
        per gain. IQ displacement reveals the probe-induced response; it
        is not a calibrated state population. ``config`` supplies the
        probe/readout modules and gain range.
        """
        # 1. Prepare gain coordinates for the probe's generator channel.
        cfg = config
        soc, soccfg = context.soc, context.soccfg

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.probe_pulse.ch},
        )

        fig, ax = context.plots.subplots(num="measurement")
        fig.suptitle(_TITLE)
        ax.set(xlabel=_GAIN_LABEL, ylabel=_IQ_LABEL)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        progress = ax.set_title(f"Averaging round 0 / {cfg.rounds}")
        viewer = live_line(ax)
        signals_buffer = SignalBuffer((len(gains),))
        rounds_done = 0

        @context.plots.on_plot_thread
        def update(completed: int) -> Iterable[Artist]:
            """Draw the cumulative mean IQ displacement and completed-round progress."""
            progress.set_text(
                f"Averaging round {completed} / {cfg.rounds} · "
                f"{completed * cfg.reps:,} nominal shots/point"
            )
            return (
                *viewer.set(gains, mist_signal2real(signals_buffer.array)),
                progress,
            )

        def show_round() -> bool:
            """Observe a completed round without asking acquisition to stop."""
            nonlocal rounds_done
            rounds_done += 1
            update(rounds_done)
            return False

        fig.show()

        # 2. Bind the board gain sweep without changing the other probe settings.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            # 3. Reset and initialize before each gain-swept probe and readout.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", cfg.gain)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("probe_pulse", cfg.probe_pulse, gain=gain))
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(
                prog,
                soc,
                rounds=cfg.rounds,
                options=RunOptions(stop_condition=show_round),
            )

        # 4. Preserve complex IQ so analysis can choose its own reference.
        return PowerDepResult(gains=gains, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[PowerDepCfg, PowerDepResult],
        options: PowerDepAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot the IQ displacement from ``options.g0`` using ``source`` data.

        If g0 is None, use the complex mean of the first 5% of scan points
        (at least one), just as Live does. ``ac_coeff`` maps gain squared to
        photon number when supplied; ``e0`` only sets the vertical display range.
        Return no fit or calibrated population estimate.
        """
        # 1. Measure IQ displacement from the explicit or default scan reference.
        result = source.result
        g0, e0, ac_coeff = options.g0, options.e0, options.ac_coeff
        gains, signals = result.gains, result.signals
        reference = _reference_iq(signals) if g0 is None else g0
        displacement = np.abs(signals - reference)

        # 2. Use photon number only when a gain-squared calibration is supplied.
        if ac_coeff is None:
            xs = gains
            xlabel = _GAIN_LABEL
        else:
            xs = ac_coeff * gains**2
            xlabel = "Mean photon number"

        # 3. Connect the measured points; the norm of a mean difference is
        # not a shot average, so do not put shot-SEM bars on this observable.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        fig.suptitle(_TITLE)
        ax.set(xlabel=xlabel, ylabel=_IQ_LABEL)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.plot(xs, displacement, ".-")
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(reference - e0))
