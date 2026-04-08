from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
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


class DriveFreqModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    init_pulse: NotRequired[PulseCfg]
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class DriveFreqCfg(ModularProgramCfg, TaskCfg):
    modules: DriveFreqModuleCfg
    sweep: dict[str, SweepCfg]


class DriveFreqExp(AbsExperiment[DriveFreqResult, DriveFreqCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> DriveFreqResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")
        _cfg = check_type(deepcopy(cfg), DriveFreqCfg)
        modules = _cfg["modules"]

        freq_sweep = _cfg["sweep"]["freq"]
        gain_sweep = _cfg["sweep"]["gain"]

        probe_pulse = modules["probe_pulse"]
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": probe_pulse["ch"]}
        )
        gains = sweep2array(
            gain_sweep, "gain", {"soccfg": soccfg, "gen_ch": probe_pulse["ch"]}
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: DriveFreqCfg = cast(DriveFreqCfg, ctx.cfg)
            modules = cfg["modules"]

            freq_sweep = cfg["sweep"]["freq"]
            gain_sweep = cfg["sweep"]["gain"]
            freq_param = sweep2param("freq", freq_sweep)
            gain_param = sweep2param("gain", gain_sweep)
            Pulse.set_param(modules["probe_pulse"], "freq", freq_param)
            Pulse.set_param(modules["probe_pulse"], "gain", gain_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("init_pulse", modules.get("init_pulse")),
                    Pulse("probe_pulse", modules["probe_pulse"]),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("freq", freq_sweep), ("gain", gain_sweep)],
            ).acquire(soc, progress=False, callback=update_hook)

        with LivePlot2D("Pulse frequency (MHz)", "Pulse gain (a.u.)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs), len(gains)),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, gains, drivefreq_signal2real(ctx.root_data)
                ),
            )

        # record the last result
        self.last_cfg = _cfg
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
        signals, freqs, gains, cfg = load_data(filepath, return_cfg=True, **kwargs)
        assert gains is not None
        assert len(freqs.shape) == 1 and len(gains.shape) == 1
        assert signals.shape == (len(gains), len(freqs))

        freqs = freqs.astype(np.float64)
        gains = gains.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = cast(DriveFreqCfg, cfg)
        self.last_result = (freqs, gains, signals)

        return freqs, gains, signals
