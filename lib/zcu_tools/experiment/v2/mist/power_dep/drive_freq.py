from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

# (freqs, gains, signals)
DriveFreqResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def drivefreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = signals - np.mean(signals[:, :avg_len])

    return np.abs(mist_signals)


class DriveFreqModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class DriveFreqSweepCfg(ConfigBase):
    freq: SweepCfg
    gain: SweepCfg


class DriveFreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: DriveFreqModuleCfg
    sweep: DriveFreqSweepCfg


class DriveFreqExp(AbsExperiment[DriveFreqResult, DriveFreqCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: DriveFreqCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> DriveFreqResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        freq_sweep = cfg.sweep.freq
        gain_sweep = cfg.sweep.gain

        probe_pulse = modules.probe_pulse
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": probe_pulse.ch}
        )
        gains = sweep2array(
            gain_sweep, "gain", {"soccfg": soccfg, "gen_ch": probe_pulse.ch}
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, DriveFreqCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            gain_sweep = cfg.sweep.gain
            freq_param = sweep2param("freq", freq_sweep)
            gain_param = sweep2param("gain", gain_sweep)
            modules.probe_pulse.set_param("freq", freq_param)
            modules.probe_pulse.set_param("gain", gain_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Pulse("probe_pulse", modules.probe_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("freq", freq_sweep), ("gain", gain_sweep)],
            ).acquire(
                soc, progress=False, round_hook=update_hook, **(acquire_kwargs or {})
            )

        with LivePlot2D("Pulse frequency (MHz)", "Pulse gain (a.u.)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs), len(gains)),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, gains, drivefreq_signal2real(ctx.root_data)
                ),
            )

        # record the last result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (freqs, gains, signals)

        return freqs, gains, signals

    def analyze(self, result: Optional[DriveFreqResult] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, gains, signals = result

        real_signals = drivefreq_signal2real(signals)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(ax, Axes)

        ax.imshow(
            real_signals.T,
            extent=(freqs[0], freqs[-1], gains[0], gains[-1]),
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax.set_xlabel("Pulse frequency (MHz)", fontsize=14)
        ax.set_ylabel("Pulse gain (a.u.)", fontsize=14)

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[DriveFreqResult] = None,
        comment: Optional[str] = None,
        tag: str = "mist/",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, gains, signals = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Pulse frequency", "unit": "Hz", "values": 1e6 * freqs},
            y_info={"name": "Pulse gain", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> DriveFreqResult:
        signals, freqs, gains, comment = load_data(filepath, return_comment=True, **kwargs)
        assert gains is not None
        assert len(freqs.shape) == 1 and len(gains.shape) == 1
        assert signals.shape == (len(gains), len(freqs))

        freqs = freqs.astype(np.float64)
        gains = gains.astype(np.float64)
        signals = signals.astype(np.complex128)

        if comment is not None:

            cfg, _, _ = parse_comment(comment)

            if cfg is not None:

                self.last_cfg = DriveFreqCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (freqs, gains, signals)

        return freqs, gains, signals
