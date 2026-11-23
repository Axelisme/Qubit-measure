from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.shot_uncertainty import IQCovarianceError, iq_mean_covariance
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
from zcu_tools.experiment.v2.runtime.run_acquire import RunOptions
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

from zcu_lab.v2.support import live_errorbar

logger = logging.getLogger(__name__)

_TITLE = "MIST · Resonator probe · IQ displacement"
_GAIN_LABEL = "Probe gain (a.u.)"
_IQ_LABEL = "Mean IQ displacement (a.u.)"
_GAIN_AXIS = VariableAxisSpec(field_name="gains", label="Drive Power", unit="a.u.")
_MOMENT_COLUMNS = np.arange(6, dtype=np.float64)


@dataclass(frozen=True)
class PowerDepResult:
    """Hardware probe gains and their mean complex readout IQ, both in a.u.

    ``iq_moments`` is a float64 array with one row per gain, in order
    [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]. The sums use the same offset-corrected
    shots as the mean IQ. None denotes a mean-only result.
    """

    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]
    iq_moments: NDArray[np.float64] | None = None


@pydantic_dataclass(frozen=True)
class PowerDepAnalyzeOptions:
    g0: complex | None = None
    e0: complex | None = None
    ac_coeff: float | None = None


def _reference_size(signals: NDArray[np.complex128]) -> int:
    return max(len(signals) // 20, 1)


def _reference_iq(signals: NDArray[np.complex128]) -> complex:
    return complex(signals[: _reference_size(signals)].mean())


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return mean IQ distances from the complex mean of the first 5% of scan points (at least one)."""
    # ADR-0069: distance preserves probe-induced shifts orthogonal to a fixed IQ axis.
    return np.abs(signals - _reference_iq(signals))


def _point_covariances(moments: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate the scan in one batch, isolating only numerical failures."""
    try:
        return iq_mean_covariance(moments)
    except IQCovarianceError:
        # The batch validated the input contract. Retry row by row to preserve
        # other estimates and report the numerical reason at the failing gain.
        covariance = np.full((len(moments), 2, 2), np.nan)
        for index, point in enumerate(moments):
            try:
                covariance[index] = iq_mean_covariance(point)
            except IQCovarianceError as error:
                logger.warning(
                    "Shot covariance unavailable for propagated SEM at gain index %d: %s",
                    index,
                    error,
                )
        return covariance


def _distance_sem(
    delta: NDArray[np.complex128],
    distance: NDArray[np.float64],
    moments: NDArray[np.float64],
    *,
    reference_size: int,
) -> NDArray[np.float64]:
    """Propagate shot covariance to each distance; zero reference_size fixes g0."""
    if moments.shape != (*distance.shape, len(_MOMENT_COLUMNS)):
        raise ValueError("IQ moments must have shape (number of gains, 6)")
    covariance = _point_covariances(moments)
    if reference_size:
        # For the scan-start mean, Cov(z_i, r) = Σ_i/m for its own points.
        reference_cov = (covariance[:reference_size] / reference_size**2).sum(axis=0)
        covariance[:reference_size] *= 1 - 2 / reference_size
        covariance += reference_cov
        if (
            reference_size == 1
            and len(moments)
            and moments[0, 0] >= 2
            and np.isfinite(moments[0]).all()
        ):
            # The self-difference is identically zero even if its shot
            # covariance failed estimation. Other gains still need that covariance.
            covariance[0] = 0

    defined = np.isfinite(distance) & np.isfinite(covariance).all(axis=(1, 2))
    nonzero = defined & (distance > 0)
    direction = np.column_stack((delta.real[nonzero], delta.imag[nonzero]))
    direction /= distance[nonzero, None]
    variance = np.full(distance.shape, np.nan)
    variance[nonzero] = np.einsum(
        "ni,nij,nj->n", direction, covariance[nonzero], direction
    )
    zero = defined & (distance == 0)
    if zero.any():
        # The radial direction is undefined at zero displacement.
        logger.info(
            "Using maximum directional shot variance at %d zero IQ distances",
            np.count_nonzero(zero),
        )
        variance[zero] = np.linalg.eigvalsh(covariance[zero])[:, -1]
    failed = defined & ~np.isfinite(variance)
    if failed.any():
        logger.warning(
            "Omitting shot SEM: non-finite projected variance at zero-based gain indices %s",
            np.flatnonzero(failed).tolist(),
        )
        variance[failed] = np.nan
    return np.sqrt(np.maximum(variance, 0))


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


def _build_result(data: GroupedLoadData[PowerDepCfg]) -> PowerDepResult:
    """Restore mean IQ and check that any retained moments share its gain axis."""
    signal = data.variable("data")
    (gains,) = signal.axes
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return PowerDepResult(gains, signal.z)
    columns, moment_gains = moments.axes
    if not np.array_equal(gains, moment_gains):
        raise ValueError(
            "/vars/iq_moments: Drive Power axis must equal /vars/data axis"
        )
    if not np.array_equal(columns, _MOMENT_COLUMNS):
        raise ValueError(
            "/vars/iq_moments: Moment axis must contain columns 0 through 5"
        )
    return PowerDepResult(gains, signal.z, moments.z)


class PowerDepExp(Experiment[PowerDepResult, PowerDepCfg]):
    Options: ClassVar[type[PowerDepAnalyzeOptions]] = PowerDepAnalyzeOptions

    # ADR-0069: save optional shot moments in Native; Labber keeps the mean IQ only.
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
        stay fixed. Return hardware gain coordinates, mean complex IQ and
        six shot sums per gain. Live shows IQ displacement and propagated
        shot SEM. The readout must supply one integrated ADC read per shot
        without masking. ``config`` supplies the modules and gain range.
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
        viewer = live_errorbar(ax)
        moments_buffer = SignalBuffer(
            (len(gains), len(_MOMENT_COLUMNS)), dtype=np.float64
        )

        @context.plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> Iterable[Artist]:
            """Draw cumulative IQ displacement and its shot SEM."""
            _, signals = split_sums(moments)
            delta = signals - _reference_iq(signals)
            distance = np.abs(delta)
            sem = _distance_sem(
                delta, distance, moments, reference_size=_reference_size(signals)
            )
            return viewer.set(gains, distance, sem)

        def show_round() -> bool:
            """Observe a completed round without asking acquisition to stop."""
            update(moments_buffer.array)
            return False

        fig.show()

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
            sched.run_acquire_summed(
                prog,
                soc,
                reducer=reducer,
                rounds=cfg.rounds,
                options=RunOptions(stop_condition=show_round),
            )

        # 4. Preserve complex IQ and same-shot moments for reanalysis.
        moments = moments_buffer.array
        _, signals = split_sums(moments)
        return PowerDepResult(gains=gains, signals=signals, iq_moments=moments)

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
        Show propagated shot SEM wherever the retained moments define one.
        Log unavailable covariance and omit only dependent bars. Invalid moment
        shape, counts or infinite moments raise ValueError. Return no fit or
        calibrated population estimate.
        """
        # 1. Measure IQ displacement from the explicit or default scan reference.
        result = source.result
        g0, e0, ac_coeff = options.g0, options.e0, options.ac_coeff
        gains, signals = result.gains, result.signals
        reference = _reference_iq(signals) if g0 is None else g0
        delta = signals - reference
        displacement = np.abs(delta)

        # 2. Use photon number only when a gain-squared calibration is supplied.
        if ac_coeff is None:
            xs = gains
            xlabel = _GAIN_LABEL
        else:
            xs = ac_coeff * gains**2
            xlabel = "Mean photon number"

        # 3. Connect the measured points and show uncertainty when moments are available.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        fig.suptitle(_TITLE)
        ax.set(xlabel=xlabel, ylabel=_IQ_LABEL)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.plot(xs, displacement, ".-")
        if result.iq_moments is not None:
            sem = _distance_sem(
                delta,
                displacement,
                result.iq_moments,
                reference_size=_reference_size(signals) if g0 is None else 0,
            )
            if np.isfinite(sem).any():
                ax.errorbar(
                    xs,
                    displacement,
                    yerr=sem,
                    fmt="none",
                    ecolor="C0",
                    label="Propagated shot SEM",
                )
                ax.legend()
        if e0 is not None:
            ax.set_ylim(0, 1.1 * np.abs(reference - e0))
