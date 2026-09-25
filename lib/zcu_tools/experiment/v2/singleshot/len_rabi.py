from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.base import StoppedPartialAcquireError
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
    cfg_snapshot: LenRabiCfg | None = None


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

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: LenRabiCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> LenRabiResult:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        if cfg.rounds != 1:
            warnings.warn("rounds will be overwritten to 1 for singleshot measurement")
            cfg.rounds = 1
        if cfg.reps != cfg.shots:
            warnings.warn("reps will be overwritten by singleshot measurement shots")
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

        with LivePlot1D(
            "Length (us)",
            "Signal",
            segment_kwargs=dict(
                num_lines=3,
                line_kwargs=[
                    dict(label="Ground"),
                    dict(label="Excited"),
                    dict(label="Other"),
                ],
            ),
        ) as viewer:
            viewer.get_ax().set_ylim(0.0, 1.0)

            def update_view(raw_iq: NDArray[np.complex128]) -> None:
                acquired_rows = np.all(np.isfinite(raw_iq), axis=1)
                populations = classify_rabi_iq(raw_iq, g_center, e_center, radius)
                populations[~acquired_rows] = np.nan
                other = 1.0 - populations.sum(axis=1)
                viewer.update(lengths, np.column_stack((populations, other)).T)

            buffer = SignalBuffer(
                expected_shape,
                dtype=np.complex128,
                on_update=update_view,
            )
            with Schedule(cfg, buffer) as sched:
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
            cfg_snapshot=cfg,
        )

    @retrieve_result
    def analyze(
        self,
        result: LenRabiResult | None = None,
        *,
        decay: bool = True,
        fit_phase: bool = False,
        initial_state: Literal["ground", "excited"] = "ground",
        max_calls: int | None = None,
    ) -> tuple[RabiJointFitResult, Figure]:
        assert result is not None, "no result found"

        fit = fit_rabi_joint(
            result.lengths,
            result.signals,
            decay=decay,
            fit_phase=fit_phase,
            max_calls=max_calls,
            initial_state=initial_state,
        )
        readout = (
            result.cfg_snapshot.modules.readout
            if result.cfg_snapshot is not None
            else None
        )
        return fit, plot_rabi_joint(
            result.lengths, result.signals, fit, readout, sweep="length"
        )
