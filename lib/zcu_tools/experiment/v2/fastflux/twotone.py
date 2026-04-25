from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Join,
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
from zcu_tools.utils.process import rotate2real

# (gains, freqs, signals2D)
TwoToneResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def twotone_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class TwoToneModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    flux_pulse: PulseCfg
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class TwoToneSweepCfg(ConfigBase):
    gain: SweepCfg
    freq: SweepCfg


class TwotoneCfg(ProgramV2Cfg, ExpCfgModel):
    modules: TwoToneModuleCfg
    sweep: TwoToneSweepCfg


class TwoToneExp(AbsExperiment[TwoToneResult, TwotoneCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: TwotoneCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> TwoToneResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        # uniform in square space
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.flux_pulse.ch},
        )
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, TwotoneCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            gain_sweep = cfg.sweep.gain
            freq_sweep = cfg.sweep.freq

            gain_param = sweep2param("gain", gain_sweep)
            freq_param = sweep2param("freq", freq_sweep)
            modules.flux_pulse.set_param("gain", gain_param)
            modules.qub_pulse.set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Join(
                        Pulse("flux_pulse", modules.flux_pulse),
                        Pulse("qub_pulse", modules.qub_pulse),
                    ),
                    Readout("readout", modules.readout),
                ],
                sweep=[
                    ("gain", gain_sweep),
                    ("freq", freq_sweep),
                ],
            ).acquire(
                soc, progress=False, round_hook=update_hook, **(acquire_kwargs or {})
            )

        with LivePlot2D("Flux Pulse Gain (a.u.)", "Frequency (MHz)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(gains), len(freqs)),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    gains, freqs, twotone_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = deepcopy(cfg)
        self.last_result = (gains, freqs, signals)

        return gains, freqs, signals

    def analyze(self, result: Optional[TwoToneResult] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, freqs, signals2D = result

        real_signals = twotone_signal2real(signals2D)

        fig, ax = plt.subplots()

        ax.imshow(
            real_signals.T,
            extent=[gains[0], gains[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax.set_xlabel("Flux Pulse Gain (a.u.)")
        ax.set_ylabel("Frequency (MHz)")

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[TwoToneResult] = None,
        comment: Optional[str] = None,
        tag: str = "fastflux/twotone",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, freqs, signals2D = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Flux Pulse Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> TwoToneResult:
        signals2D, gains, freqs, comment = load_data(filepath, return_comment=True, **kwargs)
        assert freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert signals2D.shape == (len(gains), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz

        gains = gains.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.last_cfg = TwotoneCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (gains, freqs, signals2D)

        return gains, freqs, signals2D
