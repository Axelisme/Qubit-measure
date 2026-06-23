from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import safe_labber_filepath
from zcu_tools.utils.labber_io import load_labber_data, save_labber_data
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class RabiCheckResult:
    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: RabiCheckCfg | None = None


def reset_rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class RabiCheckModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    rabi_pulse: PulseCfg
    tested_reset: ResetCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class RabiCheckSweepCfg(ConfigBase):
    gain: SweepCfg


class RabiCheckCfg(ProgramV2Cfg, ExpCfgModel):
    modules: RabiCheckModuleCfg
    sweep: RabiCheckSweepCfg


class RabiCheckExp(AbsExperiment[RabiCheckResult, RabiCheckCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: RabiCheckCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> RabiCheckResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.rabi_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, RabiCheckCfg],
            update_hook: Callable | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            # Attach gain sweep to initialization pulse
            gain_param = sweep2param("gain", cfg.sweep.gain)
            modules.rabi_pulse.set_param("gain", gain_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                sweep=[
                    ("reset_sel", 3),
                    ("gain", cfg.sweep.gain),
                ],
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("rabi_pulse", modules.rabi_pulse),
                    Branch(
                        "reset_sel",
                        [],
                        Reset("tested_reset_1", modules.tested_reset),
                        [
                            Reset("tested_reset_2", modules.tested_reset),
                            Pulse("pi_pulse", modules.pi_pulse),
                        ],
                    ),
                    Readout("readout", modules.readout),
                ],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot1D(
            "Pulse gain", "Amplitude", segment_kwargs=dict(num_lines=3)
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(3, len(gains)),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    gains, reset_rabi_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_result = RabiCheckResult(gains, signals, cfg_snapshot=cfg)

        return self.last_result

    def analyze(self, result: RabiCheckResult | None = None) -> Figure:
        """Analyze reset rabi check results. (No specific analysis implemented)"""
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, signals = result.gains, result.signals
        real_signals = reset_rabi_signal2real(signals)

        wo_signals, w_signals, wp_signals = real_signals

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(gains, wo_signals, label="Without Reset", marker=".")
        ax.plot(gains, w_signals, label="With Reset", marker=".")
        ax.plot(gains, wp_signals, label="  + Pi Pulse", marker=".")
        ax.legend()
        ax.grid(True)

        return fig

    def save(
        self,
        filepath: str,
        result: RabiCheckResult | None = None,
        comment: str | None = None,
        tag: str = "twotone/reset/rabi_check",
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, signals = result.gains, result.signals

        if result.cfg_snapshot is None:
            raise ValueError("Cannot save result without configuration snapshot")
        cfg = result.cfg_snapshot
        comment = make_comment(cfg, comment)

        save_labber_data(
            safe_labber_filepath(filepath),
            z=("Signal", "a.u.", signals),  # (Ny=3, Nx=gains) native (inner last)
            axes=[
                ("Amplitude", "a.u.", gains),  # inner axis (x)
                ("Reset", "None", np.array([0, 1, 2])),  # outer axis (y), discrete
            ],
            comment=comment,
            tags=tag,
        )

    def load(self, filepath: str) -> RabiCheckResult:
        data = load_labber_data(filepath)
        signals = np.asarray(data.z)  # native (Ny=3, Nx)
        gains = np.asarray(data.axes[0].values)  # axes[0] = Amplitude
        y_values = np.asarray(data.axes[1].values)  # axes[1] = Reset [0, 1, 2]
        comment = data.comment

        assert gains.ndim == 1 and y_values.ndim == 1
        assert signals.shape == (len(y_values), len(gains))

        gains = gains.astype(np.float64)
        signals = signals.astype(np.complex128)

        cfg_snapshot = None
        if comment:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                cfg_snapshot = RabiCheckCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = RabiCheckResult(gains, signals, cfg_snapshot=cfg_snapshot)

        return self.last_result
