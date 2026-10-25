from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils.single_shot.ge import (
    GEFitOptions,
    GENumericalError,
    singleshot_ge_analysis,
)
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import Pulse, PulseReadout, Reset

from zcu_lab.v2.support.singleshot.plot import draw_ge_scatter
from zcu_lab.v2.support.singleshot.util import ge_signals_by_state, raw_shots_to_signal

from .analysis import GEConfusionResult, calc_confusion_matrix, plot_confusion_matrix


def _default_prepared_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class GE_Result:
    """Raw probe-off/on rows; prepared_states=[0, 1] records that acquisition order."""

    signals: NDArray[np.complex128]
    shot_indices: NDArray[np.int64]
    prepared_states: NDArray[np.int64]


@dataclass(frozen=True)
class GEAnalyzeOptions:
    initial_state: Literal["ground", "excited"] = "ground"
    backend: Literal["pca", "center"] = "pca"
    logscale: bool = False
    align_t1: bool = True
    length_ratio: float | None = None
    angle: float | None = None


@dataclass(frozen=True)
class GEAnalysis:
    """Prepared-label assignment and model-based GE calibration estimates.

    initial_state maps probe off/on to predominantly g/e preparation rows, not
    pure states. fidelity is the empirical equal-weight prepared-label assignment
    score at the contrast-maximizing histogram edge, not latent pure-state readout
    fidelity. theta rotates IQ by exp(1j*theta), in radians; threshold is the
    selected edge in rotated ADC units. ge_s and g_center/e_center are fitted
    width and original-frame complex centers in normalized ADC units.
    init_pops is the fitted (2,2) g/e population matrix for the preparation rows.
    fit_quality describes those histogram fits, or None when unavailable.
    Poor histogram residuals make populations and the population-corrected post
    confusion matrix unreliable even when empirical assignment is high.
    """

    initial_state: Literal["ground", "excited"]
    fidelity: float
    theta: float
    threshold: float
    ge_s: float
    g_center: complex
    e_center: complex
    init_pops: NDArray[np.float64]
    fit_quality: Mapping[str, FitQuality] | None = None

    def validate_calibration(self) -> None:
        populations = np.asarray(self.init_pops, dtype=np.float64)
        scalars = [
            self.fidelity,
            self.theta,
            self.threshold,
            self.ge_s,
            self.g_center,
            self.e_center,
        ]
        if (
            not np.isfinite(scalars).all()
            or not 0.0 <= self.fidelity <= 1.0
            or self.ge_s <= 0
            or np.isclose(self.g_center, self.e_center)
            or populations.shape != (2, 2)
            or not np.isfinite(populations).all()
            or np.any(populations < 0)
            or np.any(populations.sum(axis=1) > 1 + 1e-12)
        ):
            raise ValueError(
                "Invalid GE calibration: check centers, width and populations"
            )


@dataclass(frozen=True)
class GEPostAnalyzeOptions:
    radius: float | None = None
    consider_other: bool = False


@dataclass(frozen=True)
class GEPostAnalysis:
    confusion: GEConfusionResult


class GE_Cfg(ConfigBase):
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="us", description="Delay before the first shot."
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Delay after each shot.",
        ),
    ] = 1.0
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    probe_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    shots: Annotated[int, FieldMeta(label="Shots")]


class GE_Exp(Experiment[GE_Result, GE_Cfg]):
    Options: ClassVar[type[GEAnalyzeOptions]] = GEAnalyzeOptions
    PostOptions: ClassVar[type[GEPostAnalyzeOptions]] = GEPostAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[GE_Result, GE_Cfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("shot_indices", "Shot Index", "None", dtype=np.int64),
            AxesSpec.Axis("prepared_states", "Prepared State", "None", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=GE_Result,
        cfg_type=GE_Cfg,
        cfg_schema_version="3.0",
        tag="singleshot/ge",
    )

    def run(self, config: GE_Cfg, *, context: RunContext) -> GE_Result:
        """Collect probe-off/on IQ clouds for ground/excited readout calibration.

        Each shot applies optional reset and init, omits or applies the probe
        pulse, then reads out. Return complex IQ shaped (2, cfg.shots), shot
        indices, and prepared_states=[0, 1] denoting probe off/on, not guaranteed
        pure ground/excited states. Primary analysis maps these rows to physical
        G/E using its initial_state option; post-analysis derives the radius
        and confusion matrix from that fit.

        Use one round and cfg.shots reps on a copied cfg. Raw-shot conversion
        requires exactly one readout channel.
        """

        # Configure raw-shot acquisition rather than averaging the IQ clouds.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(config)

        signals_buffer = SignalBuffer(
            (2, cfg.shots),
            on_update=lambda data: draw_ge_scatter(context.plots, "measurement", data),
        )
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for with_probe, step in sched.scan("w/o probe pulse", [False, True]):
                probe_cfg = step.cfg.probe_pulse if with_probe else None

                # Compare identical preparations with the probe omitted/applied.
                program = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=cfg.shots,
                        initial_delay=cfg.initial_delay,
                        relax_delay=cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", step.cfg.reset),
                        Pulse("init_pulse", step.cfg.init_pulse),
                        Pulse("probe_pulse", probe_cfg),
                        PulseReadout("readout", step.cfg.readout),
                    )
                    .build()
                )

                # Store each cloud independently, preserving every complex shot.
                try:
                    program.acquire(soc, progress=True, cancel_flag=step.stop)
                except StoppedPartialAcquireError:
                    step.set_stop()
                    break
                signals_buffer[step].set(raw_shots_to_signal(program))
        signals_buffer.trigger_update(flush=True)
        signals = signals_buffer.array

        # Keep acquisition order; physical G/E mapping belongs to analysis.
        return GE_Result(
            signals=signals,
            shot_indices=np.arange(signals.shape[1], dtype=np.int64),
            prepared_states=_default_prepared_states(),
        )

    def analyze(
        self,
        source: RunRecord[GE_Cfg, GE_Result],
        options: GEAnalyzeOptions,
        *,
        plots: Plots,
        figure_name: str = "fit",
    ) -> GEAnalysis:
        """Fit readout separation and preparation populations from raw IQ clouds.

        source contains probe-off/on IQ shaped (2, N), in normalized ADC units.
        options.initial_state maps the rows to physical G/E. Return fitted
        centers, width, model populations and empirical prepared-label assignment
        fidelity, which includes preparation mixing and is not pure-state readout
        fidelity. Poor histogram fit quality makes model populations and post
        population correction unreliable. Publish raw g/e IQ scatter to plots
        as figure_name, including on numerical failure. Histogram fitting stays
        internal. cfg=None is supported.

        Rotation/classification, histogram fitting, fitted population/center
        validation and fidelity calculations wrap ValueError, RuntimeError,
        ArithmeticError and numpy.linalg.LinAlgError in GENumericalError with
        the original cause. Plotting, invalid backend/initial_state and final
        calibration validation errors propagate unchanged.
        """
        # Acquisition order is always probe off/on; fit in physical g/e order.
        signals = ge_signals_by_state(source.result.signals, options.initial_state)
        if options.backend not in ("pca", "center"):
            raise ValueError(f"Unknown backend: {options.backend}")
        if figure_name in plots:
            raise ValueError(f"Figure already exists: {figure_name}")
        draw_ge_scatter(
            plots, figure_name, signals, labels=("g preparation", "e preparation")
        )
        fidelity, init_pops, fit = singleshot_ge_analysis(
            signals,
            angle=options.angle,
            backend=options.backend,
            options=GEFitOptions(
                logscale=options.logscale,
                align_t1=options.align_t1,
                length_ratio=options.length_ratio,
            ),
            plots=None,
        )
        analysis = GEAnalysis(
            initial_state=options.initial_state,
            fidelity=fidelity,
            theta=fit["theta"],
            threshold=fit["threshold"],
            ge_s=fit["s"],
            g_center=fit["g_center"],
            e_center=fit["e_center"],
            init_pops=init_pops,
            fit_quality=fit["fit_quality"],
        )
        analysis.validate_calibration()
        return analysis

    def post_analyze(
        self,
        source: RunRecord[GE_Cfg, GE_Result],
        primary: GEAnalysis,
        options: GEPostAnalyzeOptions,
        *,
        plots: Plots,
        figure_name: str = "post",
    ) -> GEPostAnalysis:
        """Derive classification radius and confusion matrix without refitting IQ.

        source contains probe-off/on IQ shaped (2, N), in normalized ADC units.
        primary supplies physical G/E mapping and fitted populations/centers.
        options sets the radius policy, in the same IQ units. Return the
        classification diagnostic and publish it as figure_name in plots.
        Classification uses nearest-center regions and radius, not the primary
        histogram threshold. Solving the fitted preparation matrix removes its
        inferred mixing; poor primary histogram fits make this correction
        unreliable. It is not the empirical prepared-label assignment score.
        cfg=None is supported.

        Radius optimization, classification and confusion-matrix solving wrap
        ValueError, RuntimeError, ArithmeticError and numpy.linalg.LinAlgError
        in GENumericalError with the original cause. Primary validation, row
        mapping and plotting errors, including name conflicts, propagate unchanged.
        """
        primary.validate_calibration()
        g_signals, e_signals = ge_signals_by_state(
            source.result.signals, primary.initial_state
        )
        signals = (g_signals, e_signals)
        try:
            confusion = calc_confusion_matrix(signals, primary, options)
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            raise GENumericalError(str(exc)) from exc
        plot_confusion_matrix(confusion, signals, primary, plots, figure_name)
        return GEPostAnalysis(confusion=confusion)
