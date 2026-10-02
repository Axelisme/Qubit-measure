from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import field_serializer

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
)

from .rabi_analysis import classify_rabi_iq, plot_rabi_joint
from .rabi_fit import RabiJointFitResult, fit_rabi_joint
from .util import raw_shots_to_signal


@dataclass(frozen=True)
class LenRabiResult:
    lengths: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class LenRabiSweepCfg(ConfigBase):
    length: SweepCfg


class LenRabiModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class LenRabiCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LenRabiModuleCfg
    sweep: LenRabiSweepCfg
    shots: int
    g_center: complex
    e_center: complex
    radius: float

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


class LenRabiExp(PersistableExperiment[LenRabiResult, LenRabiCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "shot_indices",
                "Shot Index",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis("lengths", "Length", "s", scale=US_TO_S, dtype=np.float64),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=LenRabiResult,
        cfg_type=LenRabiCfg,
        tag="singleshot/len_rabi",
    )

    def run(
        self,
        cfg: LenRabiCfg,
        *,
        context: RunContext,
    ) -> LenRabiResult:
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        if cfg.rounds != 1:
            warnings.warn(
                "rounds will be overwritten to 1 for singleshot measurement",
                stacklevel=2,
            )
            cfg.rounds = 1
        if cfg.reps != cfg.shots:
            warnings.warn(
                "reps will be overwritten by singleshot measurement shots", stacklevel=2
            )
            cfg.reps = cfg.shots

        modules = cfg.modules
        assert modules.qub_pulse.waveform.style in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )
        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )
        expected_shape = (len(lengths), cfg.shots)

        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(0.0, 1.0)
            for line, label in zip(
                ax.lines, ("Ground", "Excited", "Other"), strict=True
            ):
                line.set_label(label)
            ax.legend()

        viewer = context.plots.liveplot_1d(
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
        with Schedule(cfg, buffer, stop=context.cancel_signal) as sched:
            for length, step in sched.scan("length", lengths.tolist()):
                modules = step.cfg.modules
                modules.qub_pulse.set_param("length", length)
                program = (
                    step.prog_builder(soc, soccfg)
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

                raw_iq = raw_shots_to_signal(program)
                expected_point_shape = (cfg.shots,)
                if raw_iq.shape != expected_point_shape:
                    raise ValueError(
                        "Len Rabi raw IQ shape mismatch: "
                        f"expected {expected_point_shape}, got {raw_iq.shape}"
                    )
                buffer[step].set(raw_iq)
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
        readout = source.cfg.modules.readout if source.cfg is not None else None
        plot_rabi_joint(
            result.lengths, result.signals, fit, readout, sweep="length", plots=plots
        )
        return fit
