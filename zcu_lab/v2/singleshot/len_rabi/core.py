from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.fitting.singleshot_rabi import (
    RabiJointFitResult,
    fit_rabi_joint,
)
from zcu_tools.analysis.singleshot import classify_rabi_iq
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, TimelineBuilder

from zcu_lab.v2.singleshot.plot_rabi import RabiJointPlotOptions, plot_rabi_joint
from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.singleshot.util import (
    rabi_preview_classifier,
    raw_shots_to_signal,
)


@dataclass(frozen=True)
class LenRabiResult:
    lengths: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class LenRabiCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Qub Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Length", unit="us")]
    shots: Annotated[int, FieldMeta(label="Shots")]
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


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class LenRabiAnalyzeOptions:
    decay: bool = True
    fit_phase: bool = False
    initial_state: Literal["ground", "excited"] = "ground"
    max_calls: int | None = None


class LenRabiExp(Experiment[LenRabiResult, LenRabiCfg]):
    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[LenRabiResult, LenRabiCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                "shot_indices", label="Shot Index", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis(
                "lengths", label="Length", unit="s", scale=1e-6, dtype=np.float64
            ),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u.", dtype=np.complex128),
        result_type=LenRabiResult,
        cfg_type=LenRabiCfg,
        cfg_schema_version="3.0",
        tag="singleshot/len_rabi",
    )

    def run(
        self,
        cfg: LenRabiCfg,
        *,
        context: RunContext,
    ) -> LenRabiResult:
        """Measure Rabi oscillations versus pulse length with raw single-shot IQ.

        At each host-side length, apply optional reset, the qubit pulse, then
        readout for cfg.shots repetitions in one round. Return rounded lengths
        in us and complex IQ shaped (Nlength, shots), not averaged populations.
        Later analysis jointly fits Rabi dynamics and readout distributions.

        Const, flat_top and sampled waveforms (cosine, gauss, DRAG) use the
        same host scan, compiling one program per length. Native arb waveforms
        raise ValueError before scanning because their length is asset-derived.
        cfg.readout supplies an optional circle and radius for live preview.
        Without a classifier, raw acquisition and joint fitting still work.
        Acquisition requires exactly one readout channel and raises on an
        unexpected shot-buffer shape.
        """

        # Set up one raw-shot cloud per length without changing the caller cfg.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        classifier = rabi_preview_classifier(cfg.readout)

        if cfg.qub_pulse.waveform.style == "arb":
            raise ValueError("Arb waveform length is asset-derived and cannot be swept")
        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
        )
        expected_shape = (len(lengths), cfg.shots)

        # Preview classified populations while keeping raw IQ as the result.
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        viewer = None
        if classifier is not None:
            figure, ax = context.plots.subplots(num="measurement")
            ax.set(xlabel="Length (us)", ylabel="Signal")
            viewer = live_line(ax, num_lines=3)
            configure_axes(ax)
            figure.show()

        @context.plots.on_plot_thread
        def update_view(raw_iq: NDArray[np.complex128]):
            if viewer is None or classifier is None or classifier.radius is None:
                return ()
            acquired_rows = np.all(np.isfinite(raw_iq), axis=1)
            populations = classify_rabi_iq(
                raw_iq, classifier.g_center, classifier.e_center, classifier.radius
            )
            populations[~acquired_rows] = np.nan
            other = 1.0 - populations.sum(axis=1)
            return viewer.set(lengths, np.column_stack((populations, other)).T)

        buffer = SignalBuffer(
            expected_shape,
            dtype=np.complex128,
            on_update=update_view,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            for length, step in sched.scan("length", lengths.tolist()):
                # Acquire one length at a time to bound the board raw-buffer size.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.shots,
                    initial_delay=cfg.initial_delay,
                    relax_delay=cfg.relax_delay,
                )
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(Pulse("qub_pulse", cfg.qub_pulse, length=length))
                builder.emit(Readout("readout", cfg.readout))
                program = builder.build()
                try:
                    program.acquire(soc, progress=True, cancel_flag=step.stop)
                except StoppedPartialAcquireError:
                    step.set_stop()
                    break

                # Write a full shot cloud only after this point finishes.
                raw_iq = raw_shots_to_signal(program)
                expected_point_shape = (cfg.shots,)
                if raw_iq.shape != expected_point_shape:
                    raise ValueError(
                        "Len Rabi raw IQ shape mismatch: "
                        f"expected {expected_point_shape}, got {raw_iq.shape}"
                    )
                buffer[step].set(raw_iq)

        # Completed rows survive cancellation; unacquired lengths remain NaN.
        signals = buffer.array
        update_view(signals)

        return LenRabiResult(
            lengths=lengths,
            shot_indices=np.arange(cfg.shots, dtype=np.int64),
            signals=signals,
        )

    def analyze(
        self,
        source: RunRecord[LenRabiCfg, LenRabiResult],
        options: LenRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> RabiJointFitResult:
        result = source.result

        fit = fit_rabi_joint(
            result.lengths,
            result.signals,
            decay=options.decay,
            fit_phase=options.fit_phase,
            max_calls=options.max_calls,
            initial_state=options.initial_state,
        )
        readout = source.cfg.readout if source.cfg is not None else None
        ro_length = readout.ro_length if readout is not None else None
        ro_length_us = float(ro_length) if isinstance(ro_length, Real) else None
        plot_rabi_joint(
            result.lengths,
            result.signals,
            fit,
            RabiJointPlotOptions(
                ro_length_us=ro_length_us,
                sweep="length",
                figsize=config.figsize,
            ),
            plots=plots,
        )
        return fit
