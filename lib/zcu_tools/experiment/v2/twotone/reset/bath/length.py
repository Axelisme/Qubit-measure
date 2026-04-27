from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from qick.asm_v2 import QickSweep1D
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
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
from zcu_tools.utils.datasaver import load_data, save_data

# (lens, signals)
LengthResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return (
        np.abs(signals[..., 2] - signals[..., 0]) ** 2
        + np.abs(signals[..., 3] - signals[..., 1]) ** 2
    )  # (lengths, )


class LengthModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
    tested_reset: BathResetCfg
    readout: ReadoutCfg


class LengthSweepCfg(ConfigBase):
    length: SweepCfg


class LengthCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LengthModuleCfg
    sweep: LengthSweepCfg


class LengthExp(AbsExperiment[LengthResult, LengthCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: LengthCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> LengthResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        rounds = cfg.rounds
        cfg.rounds = 1  # implemented round loop by scan

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
            update_hook: Optional[Callable],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
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
                **(acquire_kwargs or {}),
            )

        def update_length(i: int, ctx: TaskState, length: float) -> None:
            modules = ctx.cfg.modules
            modules.tested_reset.set_param("qub_length", length)
            modules.tested_reset.set_param("res_length", length + length_diff)

        def average_signals(
            signals: list[list[NDArray[np.complex128]]],
        ) -> NDArray[np.complex128]:
            _signals = np.array(signals)  # shape: (rounds, lengths, 4)
            mean_signals = np.full_like(_signals[0], fill_value=np.nan)
            mask = np.any(np.isfinite(_signals), axis=0)
            mean_signals[mask] = np.nanmean(_signals[:, mask], axis=0)
            return mean_signals  # (lengths, 4)

        with LivePlot1D("Length (us)", "Signal (a.u.)") as viewer:
            signals = run_task(
                task=Task(measure_fn=measure_fn, result_shape=(4,), pbar_n=1)
                .scan("length", lengths.tolist(), before_each=update_length)
                .repeat("rounds", rounds),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, bathreset_signal2real(average_signals(ctx.root_data))
                ),
            )
            signals = average_signals(signals)

        cfg.rounds = rounds

        # Cache results
        self.last_cfg = deepcopy(cfg)
        self.last_result = (lengths, signals)

        return lengths, signals

    def analyze(self, result: Optional[LengthResult] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        real_signals = bathreset_signal2real(signals)

        fig, ax = plt.subplots()

        ax.plot(lens, real_signals, marker=".")
        ax.set_xlabel("Length (us)")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title("Bath Reset Length Measurement")
        ax.grid(True)

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[LengthResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/bath/length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            y_info={"name": "Pi2 Phase", "unit": "deg", "values": [0, 90, 180, 270]},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> LengthResult:
        signals, lens, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert lens is not None
        assert len(lens.shape) == 1 and len(signals.shape) == 2
        assert signals.shape == (len(lens), 2)

        lens = lens * 1e6  # s -> us

        lens = lens.astype(np.float64)
        signals = signals.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                self.last_cfg = LengthCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (lens, signals)

        return lens, signals
