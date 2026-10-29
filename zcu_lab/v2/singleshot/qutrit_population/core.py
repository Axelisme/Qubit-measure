"""Three preparation-label IQ references through legacy Module/Schedule acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.singleshot import ReferenceIQAnalysis, analyze_reference_iq
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import Pulse, PulseReadout

from zcu_lab.v2.support.singleshot.util import raw_shots_to_signal


@dataclass(frozen=True)
class QutritPopulationResult:
    """signals: (3,shots) complex mean-sample ADC IQ, not classified populations.

    prepared_states: [0,1,2] names g/no pulse, e/pi01, f/pi03 then pi32 rows.
    shot_indices: zero-based acquired shot axis. Unfinished rows remain NaN.
    These labels describe requested preparations, not guaranteed pure states.
    """

    signals: NDArray[np.complex128]
    shot_indices: NDArray[np.int64]
    prepared_states: NDArray[np.int64]


class QutritPopulationCfg(ConfigBase):
    """Three calibrated pulses, raw-shot readout and timing in us.

    pi01 prepares e; pi03 followed by pi32 prepares f. readout collects one ADC.
    shots counts raw shots per preparation, not averages; >=6 permits splitting.
    initial_delay precedes acquisition; relax_delay follows each repetition.
    """

    pi01: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi 01",
            description="Calibrated g to e pulse; frequency in MHz, duration in us.",
        ),
    ]
    pi03: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi 03",
            description="Calibrated g to level 3 pulse, applied first for f.",
        ),
    ]
    pi32: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi 32",
            description="Calibrated level 3 to f pulse, applied after Pi 03. Set its generator explicitly.",
        ),
    ]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    shots: Annotated[int, FieldMeta(label="Shots per preparation")] = Field(
        default=4096, ge=6
    )
    relax_delay: Annotated[float, FieldMeta(label="Relax delay", unit="us")] = Field(
        default=1.0, ge=0
    )
    initial_delay: Annotated[float, FieldMeta(label="Initial delay", unit="us")] = (
        Field(default=1.0, ge=0)
    )


class QutritPopulationExp(Experiment[QutritPopulationResult, QutritPopulationCfg]):
    """Acquire reference IQ and report preparation-coordinate readout diagnostics."""

    AXES_SPEC: ClassVar[AxesSpec[QutritPopulationResult, QutritPopulationCfg]] = (
        AxesSpec(
            axes=(
                AxesSpec.Axis("shot_indices", "Shot Index", "None", dtype=np.int64),
                AxesSpec.Axis(
                    "prepared_states", "Prepared State", "None", dtype=np.int64
                ),
            ),
            z=AxesSpec.Z("signals", "Signal", "a.u.", dtype=np.complex128),
            result_type=QutritPopulationResult,
            cfg_type=QutritPopulationCfg,
            cfg_schema_version="3.0",
            tag="singleshot/qutrit_population",
        )
    )

    def run(
        self, cfg: QutritPopulationCfg, *, context: RunContext
    ) -> QutritPopulationResult:
        """Acquire g/no pulse, e/pi01, f/pi03 then pi32, followed by readout.

        Use legacy Modules and one acquire round with shots repetitions per row.
        Return complex ADC IQ (3,shots) with explicit preparation/shot axes.
        Cancellation leaves unfinished rows NaN; analysis rejects partial data.
        No reset or preparation pulse is emitted for g. The operator must supply
        measured pulse calibrations and a relaxation interval suitable for g.
        """
        buffer = SignalBuffer((3, cfg.shots))
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for label, step in sched.scan("preparation", [0, 1, 2]):
                builder = step.prog_builder(
                    context.soc,
                    context.soccfg,
                    reps=cfg.shots,
                    initial_delay=cfg.initial_delay,
                    relax_delay=cfg.relax_delay,
                )
                if label == 1:
                    builder.add(Pulse("pi01", step.cfg.pi01))
                elif label == 2:
                    builder.add(
                        Pulse("pi03", step.cfg.pi03), Pulse("pi32", step.cfg.pi32)
                    )
                program = builder.add(PulseReadout("readout", step.cfg.readout)).build()
                try:
                    program.acquire(context.soc, progress=True, cancel_flag=step.stop)
                except StoppedPartialAcquireError:
                    step.set_stop()
                    break
                buffer[step].set(raw_shots_to_signal(program))
        result = QutritPopulationResult(
            signals=buffer.array,
            shot_indices=np.arange(cfg.shots, dtype=np.int64),
            prepared_states=np.arange(3, dtype=np.int64),
        )
        plots = context.plots
        plots.subplots("measurement")

        def scatter(figure):
            ax = figure.axes[0]
            for label, row in zip(
                ("g: no pulse", "e: pi01", "f: pi03 then pi32"),
                result.signals,
                strict=True,
            ):
                ax.scatter(row.real, row.imag, s=2, alpha=0.3, label=label)
            ax.set(
                xlabel="I (mean-sample ADC)",
                ylabel="Q (mean-sample ADC)",
                title="Reference preparations (not pure states)",
            )
            ax.legend()

        plots.draw("measurement", scatter)
        return result

    def analyze(
        self,
        source: RunRecord[QutritPopulationCfg, QutritPopulationResult],
        options: None,
        *,
        plots: Plots,
    ) -> ReferenceIQAnalysis:
        """Split acquired IQ into training/evaluation halves and publish diagnostics.

        Only result.signals and prepared_states enter numerical analysis. cfg may
        be None. Output includes raw/corrected evaluation fractions, training
        confusion, label errors, separation and matrix quality. Correction keeps
        negative/above-one estimates and refers to preparation coordinates, not
        pure-state readout fidelity. Invalid/partial reference IQ fails explicitly.
        """
        del options
        result = analyze_reference_iq(
            source.result.signals, source.result.prepared_states
        )
        plots.subplots("fit", nrows=2, ncols=2, figsize=(10, 7), layout="constrained")

        def diagnostics(figure):
            for ax, matrix, title in zip(
                figure.axes[:3],
                (
                    result.raw_populations,
                    result.corrected_populations,
                    result.confusion_matrix,
                ),
                (
                    "Raw evaluation fractions",
                    "Corrected (unclipped)",
                    "Training confusion",
                ),
                strict=True,
            ):
                ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
                ax.set(
                    xticks=range(3),
                    yticks=range(3),
                    xticklabels=("g", "e", "f"),
                    yticklabels=("g", "e", "f"),
                    xlabel="Classified label"
                    if title != "Corrected (unclipped)"
                    else "Preparation coordinate",
                    ylabel="Requested preparation",
                    title=title,
                )
                for (i, j), value in np.ndenumerate(matrix):
                    ax.text(
                        j,
                        i,
                        f"{value:.5f}",
                        ha="center",
                        va="center",
                        color="white" if value > 0.5 else "black",
                    )
            ax = figure.axes[3]
            ax.axis("off")
            ax.text(
                0,
                0.95,
                f"Training / evaluation shots per label: {result.training_shots} / {result.evaluation_shots}\n\nMinimum pooled Mahalanobis: {result.minimum_separation:.5f}\nConfusion condition (2-norm): {result.condition_number:.5f}\nMinimum singular value: {result.minimum_singular_value:.5f}\n\nMacro preparation-label error: {result.macro_error:.3%}\nPer-label errors (g, e, f):\n"
                + ", ".join(f"{error:.3%}" for error in result.label_errors)
                + "\n\nIncludes preparation contamination.\nNot pure-state readout fidelity.\nCorrection retains negative and >1 values.",
                va="top",
                fontsize=10,
            )

        plots.draw("fit", diagnostics)
        return result
