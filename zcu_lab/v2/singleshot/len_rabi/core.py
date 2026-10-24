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
from zcu_tools.analysis.singleshot import classify_rabi_iq
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg

from zcu_lab.v2._support import live_line
from zcu_lab.v2._support.singleshot.util import raw_shots_to_signal
from zcu_lab.v2.singleshot.plot_rabi import RabiJointPlotOptions, plot_rabi_joint


@dataclass(frozen=True)
class LenRabiResult:
    lengths: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class LenRabiCfg(ConfigBase):
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
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Qub Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[SweepCfg, FieldMeta(label="Length", unit="us")]
    shots: Annotated[int, FieldMeta(label="Shots")]
    g_center: Annotated[complex, FieldMeta(label="Ground center")]
    e_center: Annotated[complex, FieldMeta(label="Excited center")]
    radius: Annotated[float, FieldMeta(label="Classification radius")]

    @field_serializer("g_center", "e_center")
    def serialize_center(self, value: complex) -> str:
        """Keep experiment comments JSON-safe with a lossless complex literal."""
        return str(value)


@dataclass(frozen=True)
class LenRabiAnalyzeOptions:
    decay: bool = True
    fit_phase: bool = False
    initial_state: Literal["ground", "excited"] = "ground"
    max_calls: int | None = None


class LenRabiExp(Experiment[LenRabiResult, LenRabiCfg]):
    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[LenRabiResult, LenRabiCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("shot_indices", "Shot Index", "None", dtype=np.int64),
            AxesSpec.Axis("lengths", "Length", "s", scale=1e-6, dtype=np.float64),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u.", dtype=np.complex128),
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

        Only const and flat_top waveforms are supported. cfg supplies distinct
        finite G/E centers and a positive finite radius for the live preview.
        Acquisition requires exactly one readout channel and raises on an
        unexpected shot-buffer shape.
        """

        # Set up one raw-shot cloud per length without changing the caller cfg.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)

        modules = cfg
        assert modules.qub_pulse.waveform.style in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )
        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
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

        viewer = live_line(
            context.plots,
            "measurement",
            "Length (us)",
            "Signal",
            num_lines=3,
            configure_axes=configure_axes,
        )

        def update_view(raw_iq: NDArray[np.complex128]) -> None:
            acquired_rows = np.all(np.isfinite(raw_iq), axis=1)
            populations = classify_rabi_iq(
                raw_iq, cfg.g_center, cfg.e_center, cfg.radius
            )
            populations[~acquired_rows] = np.nan
            other = 1.0 - populations.sum(axis=1)
            viewer.update(lengths, np.column_stack((populations, other)).T)

        buffer = SignalBuffer(
            expected_shape,
            dtype=np.complex128,
            on_update=update_view,
        )
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for length, step in sched.scan("length", lengths.tolist()):
                modules = step.cfg
                modules.qub_pulse.set_param("length", length)

                # Acquire one length at a time to bound the board raw-buffer size.
                program = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=cfg.shots,
                        initial_delay=cfg.initial_delay,
                        relax_delay=cfg.relax_delay,
                    )
                    .add_reset("reset", modules.reset)
                    .add_pulse("qubit_pulse", modules.qub_pulse)
                    .add_readout("readout", modules.readout)
                    .build()
                )
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
