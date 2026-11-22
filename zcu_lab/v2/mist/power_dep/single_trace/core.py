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
    shot_sums,
    split_sums,
)

from zcu_lab.v2.support import live_line

_TITLE = "MIST gain sweep · Resonator probe"
_GAIN_LABEL = "Probe gain (a.u.)"
_SHIFT_LABEL = "Magnitude of mean IQ shift (a.u.)"
_MOMENT_COLUMNS = (0, 1, 2, 3, 4, 5)
# Keep the original native axis and signal labels for prior mean-only files.
_GAIN_AXIS = VariableAxisSpec(field_name="gains", label="Drive Power", unit="a.u.")


@dataclass(frozen=True)
class PowerDepResult:
    """Hardware probe gains and offset-corrected mean complex readout IQ.

    iq_moments has one row per gain: [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²] in the
    same offset-corrected frame. None denotes a mean-only result.
    """

    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]
    iq_moments: NDArray[np.float64] | None = None


@pydantic_dataclass(frozen=True)
class PowerDepAnalyzeOptions:
    g0: complex | None = None
    e0: complex | None = None
    ac_coeff: float | None = None


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return the magnitude of mean IQ displacement from the first gain point."""
    return np.abs(signals - signals[0])


class PowerDepCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select the reset before each shot, or leave off to skip reset.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Preparation pulse",
            description="Select the qubit pulse after reset, or leave off to skip preparation.",
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
            description="Enter the start, stop (a.u.) and point count for the resonator probe gain.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the number of shots per gain point in each round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of complete gain sweeps to average.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the wait after each shot (µs) before the next shot begins.",
        ),
    ]


def _build_result(data: GroupedLoadData[PowerDepCfg]) -> PowerDepResult:
    """Restore mean IQ, with optional moments on matching gain coordinates."""
    signal = data.variable("data")
    gains = signal.axes[0]
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return PowerDepResult(gains, signal.z)
    columns, moment_gains = moments.axes
    if not np.array_equal(gains, moment_gains):
        raise ValueError("/vars/iq_moments: gain axis must equal /vars/data gain axis")
    if not np.array_equal(columns, _MOMENT_COLUMNS):
        raise ValueError(
            "/vars/iq_moments: moment axis must contain columns 0 through 5"
        )
    return PowerDepResult(gains, signal.z, moments.z)


class PowerDepExp(Experiment[PowerDepResult, PowerDepCfg]):
    Options: ClassVar[type[PowerDepAnalyzeOptions]] = PowerDepAnalyzeOptions

    # ADR-0069: new shot moments are optional natively and not a Labber log.
    AXES_SPEC: ClassVar[GroupedAxesSpec[PowerDepResult, PowerDepCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "data",
                (_GAIN_AXIS,),
                VariableZSpec(
                    "signals", label="Signal", unit="a.u.", dtype=np.complex128
                ),
            ),
            VariableSpec(
                "iq_moments",
                (
                    VariableAxisSpec(label="Moment", unit="", values=_MOMENT_COLUMNS),
                    _GAIN_AXIS,
                ),
                VariableZSpec("iq_moments", label="IQ moments", unit=""),
                required=False,
            ),
        ),
        result_type=PowerDepResult,
        cfg_type=PowerDepCfg,
        cfg_schema_version="3.0",
        tag="mist/gain",
        result_builder=_build_result,
        labber_variables=("data",),
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
        stay fixed. Return hardware gains, offset-corrected mean IQ and its
        shot moments per gain. ``config`` supplies the probe, gain range and
        one unmasked integrated read per shot on a single ADC. Unsupported
        readouts raise ValueError before acquisition.
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
        ax.set(
            xlabel=_GAIN_LABEL,
            ylabel=_SHIFT_LABEL,
            title=f"{_TITLE}\n0 shots per gain point",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        viewer = live_line(ax)

        @context.plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> Iterable[Artist]:
            """Update the IQ displacement and accumulated shot count."""
            counts, signals = split_sums(moments)
            shots = np.nanmax(counts, initial=0)
            title = ax.set_title(f"{_TITLE}\n{shots:,.0f} shots per gain point")
            return (*viewer.set(gains, mist_signal2real(signals)), title)

        fig.show()
        moments_buffer = SignalBuffer(
            (len(gains), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
        )

        # 2. Bind the board gain sweep without changing the other probe settings.
        with Schedule(moments_buffer, scope=context.scope) as sched:
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
            reducer = shot_sums(prog, (IQ("readout", offset_corrected=True),))
            sched.run_acquire_summed(prog, soc, reducer=reducer, rounds=cfg.rounds)

        # 4. Preserve complex IQ so analysis can choose its own reference.
        moments = moments_buffer.array
        return PowerDepResult(gains, split_sums(moments)[1], moments)

    def analyze(
        self,
        source: RunRecord[PowerDepCfg, PowerDepResult],
        options: PowerDepAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot the IQ displacement from ``options.g0`` using ``source`` data.

        If g0 is None, use the first gain point as in the live view.
        ``ac_coeff`` maps gain squared to photon number; ``e0`` only sets the
        vertical display range. Return no fit or calibrated population estimate.
        """
        # 1. Measure IQ displacement from g0, defaulting to the first gain sample.
        result = source.result
        g0, e0, ac_coeff = options.g0, options.e0, options.ac_coeff

        gains, signals = result.gains, result.signals

        if g0 is None:
            g0 = signals[0]

        delta = signals - g0
        amp_diff = np.abs(delta)

        # 2. Use photon number only when a gain-squared calibration is supplied.
        if ac_coeff is None:
            xs = gains
            xlabel = _GAIN_LABEL
        else:
            xs = ac_coeff * gains**2
            xlabel = r"Mean photon number $\bar n$"

        # 3. Show MIST contrast with optional e0-based vertical scaling.
        _, ax = plots.subplots(num="fit", figsize=config.figsize)
        ax.set(xlabel=xlabel, ylabel=_SHIFT_LABEL, title=_TITLE)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.plot(xs, amp_diff, ".", markersize=5)
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(g0 - e0))
