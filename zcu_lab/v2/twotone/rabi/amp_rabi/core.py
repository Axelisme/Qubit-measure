from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.fitting import FitQuality, cosfunc, fit_rabi
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
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
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
from zcu_tools.utils.process import find_rotate_angle

from zcu_lab.v2.support import live_errorbar

_PULSE_GAIN_LABEL = "Pulse gain (a.u.)"
_IQ_PROJECTION_LABEL = "IQ projection (a.u.)"


@dataclass(frozen=True)
class AmpRabiResult:
    """One-dimensional drive gains and offset-corrected mean complex IQ.

    iq_moments, when available, has shape (Ngain, 6) with columns
    [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²] of retained shots in the same offset-corrected
    frame. None denotes a mean-only result.
    """

    amps: NDArray[np.float64]
    signals: NDArray[np.complex128]
    iq_moments: NDArray[np.float64] | None = None


@validated_dataclass(frozen=True)
class AmpRabiAnalyzeOptions:
    skip: Annotated[int, FieldMeta(label="Skip points"), Field(ge=0)] = 0


@dataclass(frozen=True)
class AmpRabiAnalysis:
    pi_amp: float
    pi_amp_err: float
    pi2_amp: float
    pi2_amp_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


def _rabi_rotation(signals: NDArray[np.complex128]) -> complex | None:
    """Return the rotation aligning mean complex IQ with the real axis, or None."""
    try:
        angle = find_rotate_angle(signals)
    except ValueError:
        return None
    return complex(np.exp(-1j * angle))


def _project_real(
    signals: NDArray[np.complex128], rotation: complex | None
) -> NDArray[np.float64]:
    """Return the real component after rotation, or the original I if absent."""
    return signals.real if rotation is None else (signals * rotation).real


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project mean complex IQ onto its principal axis, or take I if undefined."""
    return _project_real(signals, _rabi_rotation(signals))


def _mean_iq(moments: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Decode the mean complex IQ per gain from cumulative six-column shot sums."""
    return split_sums(moments)[1]


def _contrast_with_sem(
    signals: NDArray[np.complex128], moments: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return projected mean IQ and SEM from the matching six-column shot sums.

    Both use the same direction (I when the rotation is undefined). Points
    with fewer than two shots or missing sums have NaN SEM. Invalid moments
    raise ValueError. Neither array is changed.
    """
    rotation = _rabi_rotation(signals)
    direction = (1.0, 0.0) if rotation is None else (rotation.real, -rotation.imag)
    return _project_real(signals, rotation), iq_shot_sem(moments, direction=direction)


class AmpRabiCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Init Pulse",
            description="Fixed preparation pulse before each gain-swept Rabi pulse; None skips preparation.",
        ),
    ] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Rabi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
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
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Stop after a completed round reaches this SNR; None disables.",
        ),
    ] = Field(default=None, gt=0)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]


_GAIN_AXIS = VariableAxisSpec(field_name="amps", label="Gain", unit="")
_MOMENT_COLUMNS = (0, 1, 2, 3, 4, 5)


def _build_result(data: GroupedLoadData[AmpRabiCfg]) -> AmpRabiResult:
    """Restore mean IQ with optional moments from the loaded native variables.

    Reuse the loaded arrays. Present moments must share the mean's gain
    coordinates and carry the declared column order, or raise ValueError.
    """
    signal = data.variable("data")
    gains = signal.axes[0]
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return AmpRabiResult(gains, signal.z)
    columns, moment_gains = moments.axes
    if not np.array_equal(gains, moment_gains):
        raise ValueError("/vars/iq_moments: Gain axis must equal /vars/data Gain axis")
    if not np.array_equal(columns, _MOMENT_COLUMNS):
        raise ValueError(
            "/vars/iq_moments: Moment axis must contain columns 0 through 5"
        )
    return AmpRabiResult(gains, signal.z, moments.z)


class AmpRabiExp(Experiment[AmpRabiResult, AmpRabiCfg]):
    Options: ClassVar[type[AmpRabiAnalyzeOptions]] = AmpRabiAnalyzeOptions

    AXES_SPEC: ClassVar[GroupedAxesSpec[AmpRabiResult, AmpRabiCfg]] = GroupedAxesSpec(
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
        result_type=AmpRabiResult,
        cfg_type=AmpRabiCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/rabi_gain",
        result_builder=_build_result,
        labber_variables=("data",),
    )

    def run(
        self,
        cfg: AmpRabiCfg,
        *,
        context: RunContext,
    ) -> AmpRabiResult:
        """Apply the configured reset and init pulse, sweep qubit drive gain at fixed duration, and read out.

        The configured readout must supply one integrated ADC read per shot without masking.
        Show projected IQ and shot SEM and, if configured, check early-stop SNR after completed rounds.
        Return gains, offset-corrected mean IQ and the moments of those same shots.
        """
        # 1. Quantize drive gains for rotation-amplitude calibration.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
        )

        # 2. Monitor Rabi contrast as the rotation amplitude changes.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel=_PULSE_GAIN_LABEL, ylabel=_IQ_PROJECTION_LABEL)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        viewer = live_errorbar(ax)

        @context.plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Update the live trace, shot SEM and retained counts; return changed artists."""
            contrast, sem = _contrast_with_sem(_mean_iq(moments), moments)
            counts = moments[:, 0]
            low, high = counts.min(), counts.max()
            retained = f"{low:.0f}" if low == high else f"{low:.0f}–{high:.0f}"
            title = ax.set_title(f"Retained shots / point: {retained}")
            return (*viewer.set(gains, contrast, sem), title)

        fig.show()
        moments_buffer = SignalBuffer(
            (len(gains), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
        )

        # 3. Keep pulse duration fixed while the FPGA sweeps its rotation amplitude.
        with Schedule(moments_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain_sweep = cfg.gain
            gain_param = builder.add_sweep("gain", gain_sweep)

            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, gain=gain_param))
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            reducer = shot_sums(prog, (IQ("readout", offset_corrected=True),))
            sched.run_acquire_summed(
                prog,
                soc,
                reducer=reducer,
                rounds=cfg.rounds,
                options=RunOptions(
                    stop_condition=snr_checker(
                        moments_buffer.at(),
                        cfg.earlystop_snr,
                        lambda moments: rabi_signal2real(_mean_iq(moments)),
                    ),
                ),
            )

        moments = moments_buffer.array
        return AmpRabiResult(amps=gains, signals=_mean_iq(moments), iq_moments=moments)

    def analyze(
        self,
        source: RunRecord[AmpRabiCfg, AmpRabiResult],
        options: AmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AmpRabiAnalysis:
        """Fit a nondecaying Rabi oscillation and return pi/pi2 gains and errors.

        Skip the first ``options.skip`` points and project IQ onto a real
        contrast axis. The point nearest zero gain fixes the phase to 0 or
        180 degrees. If shot moments are present, plot the SEM along the same
        axis; invalid statistics raise ValueError. The cosine fit is unchanged.
        Publish ``fit`` and report fit quality; these are pulse amplitude
        calibrations, not gate fidelities. Source cfg may be None.
        """
        # 1. Calibrate pi and pi/2 gains from unskipped contrast using the near-zero-gain phase.
        result = source.result

        gains, signals = result.amps, result.signals
        gains = gains[options.skip :]
        signals = signals[options.skip :]

        if result.iq_moments is None:
            real_signals, sem = rabi_signal2real(signals), None
        else:
            real_signals, sem = _contrast_with_sem(
                signals, result.iq_moments[options.skip :]
            )

        zero_signal = real_signals[np.argmin(np.abs(gains))]
        if zero_signal > 0.5 * (np.max(real_signals) + np.min(real_signals)):
            init_phase = 0.0
        else:
            init_phase = 180.0

        fit = fit_rabi(gains, real_signals, decay=False, init_phase=init_phase)
        pi_amp, pi_amp_err = fit.pi_x, fit.pi_x_err
        pi2_amp, pi2_amp_err = fit.pi2_x, fit.pi2_x_err

        # 2. Assess the oscillation fit and uncertainty of the calibrated rotation gains.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        if sem is None:
            ax.plot(gains, real_signals, ".-", label="Measured", markersize=5)
        else:
            ax.errorbar(
                gains, real_signals, yerr=sem, label="Measured", fmt=".-", markersize=5
            )
        fit_gains = np.linspace(gains.min(), gains.max(), 200)
        r2 = fit.quality.r2
        fit_label = "Fit (R² unavailable)" if r2 is None else f"Fit (R² = {r2:.3f})"
        ax.plot(fit_gains, cosfunc(fit_gains, *fit.params), "-", label=fit_label)
        ax.axvline(
            pi_amp, ls="--", c="red", label=f"π = {pi_amp:.3g} ± {pi_amp_err:.2g} a.u."
        )
        ax.axvspan(pi_amp - pi_amp_err, pi_amp + pi_amp_err, color="red", alpha=0.2)
        ax.axvline(
            pi2_amp,
            ls="--",
            c="red",
            label=f"π/2 = {pi2_amp:.3g} ± {pi2_amp_err:.2g} a.u.",
        )
        ax.axvspan(pi2_amp - pi2_amp_err, pi2_amp + pi2_amp_err, color="red", alpha=0.2)
        ax.set(xlabel=_PULSE_GAIN_LABEL, ylabel=_IQ_PROJECTION_LABEL)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2)

        fig.tight_layout()

        # Include the gain uncertainties in the GUI summary as well as the figure.
        return AmpRabiAnalysis(
            pi_amp, pi_amp_err, pi2_amp, pi2_amp_err, fit_quality={"fit": fit.quality}
        )
