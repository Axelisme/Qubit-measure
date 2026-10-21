from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import field_serializer
from zcu_tools.analysis.fitting.singleshot_rabi import (
    RabiJointFitResult,
    fit_rabi_joint,
)
from zcu_tools.analysis.singleshot import classify_rabi_iq, classify_result
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.plotting.singleshot_rabi import RabiJointPlotOptions, plot_rabi_joint
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)

from zcu_lab.v2._support.singleshot.util import raw_shots_to_signal


@dataclass(frozen=True)
class AmpRabiResult:
    gains: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class AmpRabiFit:
    pi_gain: float
    pi_gain_error: float
    pi2_gain: float
    pi2_gain_error: float
    frequency: float
    amplitude: float
    joint_fit: RabiJointFitResult


class AmpRabiSweepCfg(ConfigBase):
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]


class AmpRabiModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    qub_pulse: Annotated[PulseCfg, FieldMeta(label="Qub Pulse")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class AmpRabiCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AmpRabiModuleCfg
    sweep: AmpRabiSweepCfg
    g_center: Annotated[complex, FieldMeta(label="Ground center")]
    e_center: Annotated[complex, FieldMeta(label="Excited center")]
    radius: Annotated[float, FieldMeta(label="Classification radius")]

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        """Keep experiment comments JSON-safe with a lossless complex literal."""
        return str(value)


def _flatten_round_shots(raw: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """(round, gain, shot) -> (gain, round * shot), preserving every IQ sample."""
    return raw.transpose(1, 0, 2).reshape(raw.shape[1], -1)


def _gain_calibration(fit: RabiJointFitResult) -> AmpRabiFit:
    if not fit.backend.valid:
        return AmpRabiFit(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, fit)
    pi_gain = np.pi / fit.omega
    omega_index = fit.backend.parameter_names.index("log_omega")
    # d(pi / omega) / d(log omega) = -pi / omega.
    variance = fit.backend.covariance[omega_index, omega_index]
    pi_error = pi_gain * np.sqrt(max(float(variance), 0.0))
    return AmpRabiFit(
        pi_gain,
        float(pi_error),
        pi_gain / 2,
        float(pi_error / 2),
        fit.omega / (2 * np.pi),
        abs(fit.initial_populations[1] - fit.p_inf),
        fit,
    )


@dataclass(frozen=True)
class AmpRabiAnalyzeOptions:
    initial_state: Literal["ground", "excited"] = "ground"
    max_calls: int | None = None


class AmpRabiExp(PersistableExperiment[AmpRabiResult, AmpRabiCfg]):
    Options: ClassVar[type[AmpRabiAnalyzeOptions]] = AmpRabiAnalyzeOptions

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("shot_indices", "Shot Index", "None", dtype=np.int64),
            Axis("gains", "Gain", "a.u.", dtype=np.float64),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=AmpRabiResult,
        cfg_type=AmpRabiCfg,
        cfg_schema_version="2.0",
        tag="singleshot/amp_rabi",
    )

    def run(
        self,
        cfg: AmpRabiCfg,
        *,
        context: RunContext,
    ) -> AmpRabiResult:
        """Measure Rabi oscillations versus drive gain while retaining raw IQ.

        Apply optional reset, the gain-swept qubit pulse, then readout.
        Each host round acquires cfg.reps shots at every hardware-swept gain.
        Return gains (a.u.) and complex IQ shaped (Ngain, reps * rounds);
        rounds are concatenated, never averaged into one IQ point.

        cfg.g_center/e_center must be distinct finite centers and cfg.radius
        positive and finite for the live G/E/Other classification. These live
        fractions are not the saved data. The later raw-IQ joint fit estimates
        the oscillation period, pi/pi2 gains and readout calibration.
        Acquisition requires one readout channel and raises on an unexpected
        raw-shot buffer shape.
        """

        # Validate live classification before acquiring any shots.
        soc, soccfg = context.soc, context.soccfg
        g_center, e_center, radius = cfg.g_center, cfg.e_center, cfg.radius
        classify_result(np.empty(0, dtype=np.complex128), g_center, e_center, radius)
        cfg = deepcopy(cfg)
        rounds, cfg.rounds = cfg.rounds, 1
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.modules.qub_pulse.ch},
        )

        # Preview populations from complete shot columns without losing raw IQ.
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(-0.02, 1.02)
            for line, kwargs in zip(
                ax.lines,
                [
                    {"label": label, "color": color}
                    for label, color in (
                        ("Ground", "blue"),
                        ("Excited", "red"),
                        ("Other", "green"),
                    )
                ],
                strict=True,
            ):
                line.set(**kwargs)
            ax.legend()

        viewer = context.plots.liveplot_1d(
            "measurement",
            "Pulse gain (a.u.)",
            "Classified population",
            num_lines=3,
            configure_axes=configure_axes,
        )

        def update_view(raw: NDArray[np.complex128]) -> None:
            shots = _flatten_round_shots(raw)
            complete = np.all(np.isfinite(shots), axis=0)
            populations = np.full((gains.size, 2), np.nan)
            if np.any(complete):
                populations = classify_rabi_iq(
                    shots[:, complete], g_center, e_center, radius
                )
            viewer.update(
                gains, np.column_stack((populations, 1 - populations.sum(axis=1))).T
            )

        buffer = SignalBuffer(
            (rounds, gains.size, cfg.reps),
            dtype=np.complex128,
            on_update=update_view,
        )
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            modules.qub_pulse.set_param(
                "gain", sweep2param("gain", sched.cfg.sweep.gain)
            )

            # The board sweeps gain within each reset/drive/readout acquisition.
            program = (
                sched.prog_builder(soc, soccfg)
                .add_reset("reset", modules.reset)
                .add_pulse("qubit_pulse", modules.qub_pulse)
                .add_readout("readout", modules.readout)
                .declare_sweep("gain", sched.cfg.sweep.gain)
                .build()
            )

            # Store each completed round separately so every shot survives.
            for _, step in sched.repeat("round", rounds):
                try:
                    program.acquire(soc, progress=False, cancel_flag=step.stop)
                except StoppedPartialAcquireError:
                    step.set_stop()
                    break
                raw = raw_shots_to_signal(program)
                if raw.shape != (cfg.reps, gains.size):
                    raise ValueError(
                        f"Amp Rabi raw IQ shape mismatch: expected {(cfg.reps, gains.size)}, got {raw.shape}"
                    )
                buffer[step].set(raw.T)

        # Join round/shot axes only; unacquired rounds remain NaN.
        update_view(buffer.array)
        return AmpRabiResult(
            gains,
            np.arange(rounds * cfg.reps, dtype=np.int64),
            _flatten_round_shots(buffer.array),
        )

    def analyze(
        self,
        source: RunRecord[AmpRabiCfg, AmpRabiResult],
        options: AmpRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AmpRabiFit:
        result = source.result
        if result.signals.shape != (
            result.gains.size,
            result.shot_indices.size,
        ) or not np.iscomplexobj(result.signals):
            raise ValueError("Amp Rabi requires raw IQ with shape (gain, shot)")
        # Cancelled rounds occupy entirely missing shot columns. Do not turn
        # partially missing sweeps into unequal per-gain sample counts.
        acquired = ~np.all(np.isnan(result.signals), axis=0)
        order = np.argsort(result.gains)
        gains, signals = result.gains[order], result.signals[order][:, acquired]
        joint = fit_rabi_joint(
            gains,
            signals,
            decay=False,
            fit_phase=False,
            initial_state=options.initial_state,
            max_calls=options.max_calls,
        )
        readout = source.cfg.modules.readout if source.cfg is not None else None
        ro_length = readout.ro_length if readout is not None else None
        ro_length_us = float(ro_length) if isinstance(ro_length, Real) else None
        plot_rabi_joint(
            gains,
            signals,
            joint,
            RabiJointPlotOptions(
                ro_length_us=ro_length_us,
                sweep="gain",
                figsize=config.figsize,
            ),
            plots=plots,
        )
        return _gain_calibration(joint)
