from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from qick.asm_v2 import QickSweep1D

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
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    BathReset,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)
from zcu_tools.program.v2.modules import BathResetCfg


def _default_phase_values() -> NDArray[np.float64]:
    return np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)


@dataclass(frozen=True)
class LengthResult:
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    phases: NDArray[np.float64] = dataclass_field(default_factory=_default_phase_values)
    cfg_snapshot: LengthCfg | None = None


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.asarray(
        np.abs(signals[..., 2] - signals[..., 0]) ** 2
        + np.abs(signals[..., 3] - signals[..., 1]) ** 2,
        dtype=np.float64,
    )  # (lengths, )


class LengthModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    tested_reset: BathResetCfg
    readout: ReadoutCfg


class LengthSweepCfg(ConfigBase):
    length: SweepCfg


class LengthCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LengthModuleCfg
    sweep: LengthSweepCfg


class LengthExp(PersistableExperiment[LengthResult, LengthCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("phases", "Pi2 Phase", "deg", scale=IDENTITY, dtype=np.float64),
            Axis("lengths", "Length", "s", scale=US_TO_S, dtype=np.float64),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=LengthResult,
        cfg_type=LengthCfg,
        tag="twotone/reset/bath/length",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: LengthCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> LengthResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        rounds = cfg.rounds  # round loop is driven by MeasureSession.repeat below

        length_sweep = cfg.sweep.length
        tested_reset = modules.tested_reset
        qub_lengths = sweep2array(
            length_sweep,
            "time",
            {"soccfg": soccfg, "gen_ch": tested_reset.qubit_tone_cfg.ch},
            allow_array=True,
        )
        lengths = qub_lengths  # use qubit tone length as x-axis

        orig_res_length = tested_reset.cavity_tone_cfg.waveform.length
        orig_qub_length = tested_reset.qubit_tone_cfg.waveform.length
        length_diff = orig_res_length - orig_qub_length

        prog_cache: dict[float, ModularProgramV2] = {}

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, LengthCfg],
            update_hook: Callable | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            cfg.rounds = 1  # round loop is driven by the scan's .repeat("rounds", ...)
            modules = cfg.modules

            tested_reset = modules.tested_reset

            phase_param = QickSweep1D("phase", 0.0, 270.0)
            tested_reset.set_param("pi2_phase", phase_param)

            length = float(tested_reset.qubit_tone_cfg.waveform.length)
            if length not in prog_cache:
                prog_cache[length] = ModularProgramV2(
                    soccfg,
                    cfg,
                    modules=[
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        BathReset("tested_reset", tested_reset),
                        Readout("readout", modules.readout),
                    ],
                    sweep=[
                        ("phase", 4),  # (0, 90, 180, 270)
                    ],
                )

            return prog_cache[length].acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        def average_signals(
            signals: NDArray[np.complex128],
        ) -> NDArray[np.complex128]:
            _signals = np.asarray(signals)  # shape: (rounds, lengths, 4)
            mean_signals = np.full_like(_signals[0], fill_value=np.nan)
            mask = np.any(np.isfinite(_signals), axis=0)
            mean_signals[mask] = np.nanmean(_signals[:, mask], axis=0)
            return mean_signals  # (lengths, 4)

        with LivePlot1D("Length (us)", "Signal (a.u.)") as viewer:
            with MeasureSession(cfg) as run:
                buffer = run.buffer(
                    (rounds, len(lengths), 4),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        lengths, bathreset_signal2real(average_signals(data))
                    ),
                )
                for rep in run.repeat("rounds", rounds):
                    for step in rep.scan("length", lengths.tolist()):
                        modules = step.cfg.modules
                        modules.tested_reset.set_param("qub_length", step.value)
                        modules.tested_reset.set_param(
                            "res_length", step.value + length_diff
                        )
                        buffer[rep.index, step].measure(measure_fn, pbar_n=1)
                signals = average_signals(buffer.array)

        return LengthResult(
            lengths=lengths,
            signals=signals,
            phases=_default_phase_values(),
            cfg_snapshot=orig_cfg,
        )

    @retrieve_result
    def analyze(self, result: LengthResult | None = None) -> Figure:
        assert result is not None, "no result found"

        lens, signals = result.lengths, result.signals

        real_signals = bathreset_signal2real(signals)

        fig, ax = plt.subplots()

        ax.plot(lens, real_signals, marker=".")
        ax.set_xlabel("Length (us)")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title("Bath Reset Length Measurement")
        ax.grid(True)

        return fig
