from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, TypeAlias, cast

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
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

# (flux_gains, mist_gains, signals2D)
MistResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class MistModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
    flux_pulse: PulseCfg
    mist_pulse: PulseCfg
    readout: ReadoutCfg


class MistSweepCfg(ConfigBase):
    flux_gain: SweepCfg
    mist_gain: SweepCfg


class MistCfg(ProgramV2Cfg, ExpCfgModel):
    modules: MistModuleCfg
    sweep: MistSweepCfg


class MistExp(AbsExperiment[MistResult, MistCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: MistCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> MistResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        flux_gain_sweep = cfg.sweep.flux_gain
        mist_gain_sweep = cfg.sweep.mist_gain

        flux_gains = sweep2array(
            flux_gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.flux_pulse.ch},
        )
        mist_gains = sweep2array(
            mist_gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.mist_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, MistCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: MistCfg = cast(MistCfg, ctx.cfg)
            modules = cfg.modules

            flux_gain_sweep = cfg.sweep.flux_gain
            mist_gain_sweep = cfg.sweep.mist_gain

            flux_gain_param = sweep2param("flux_gain", flux_gain_sweep)
            mist_gain_param = sweep2param("mist_gain", mist_gain_sweep)
            modules.flux_pulse.set_param("gain", flux_gain_param)
            modules.mist_pulse.set_param("gain", mist_gain_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Join(
                        Pulse("flux_pulse", modules.flux_pulse),
                        Pulse("mist_pulse", modules.mist_pulse),
                    ),
                    Readout("readout", modules.readout),
                ],
                sweep=[
                    ("flux_gain", flux_gain_sweep),
                    ("mist_gain", mist_gain_sweep),
                ],
            ).acquire(
                soc, progress=False, round_hook=update_hook, **(acquire_kwargs or {})
            )

        with LivePlot2D("Flux Pulse Gain (a.u.)", "Mist Pulse Gain (a.u.)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(flux_gains), len(mist_gains)),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    flux_gains, mist_gains, mist_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = deepcopy(cfg)
        self.last_result = (flux_gains, mist_gains, signals)

        return flux_gains, mist_gains, signals

    def analyze(
        self, result: Optional[MistResult] = None, ac_coeff: Optional[float] = None
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        flux_gains, mist_gains, signals2D = result

        real_signals = mist_signal2real(signals2D)

        fig, ax = plt.subplots(figsize=config.figsize)

        if ac_coeff is not None:
            mist_photons = ac_coeff * mist_gains**2
            ylabel = "Photon Number (a.u.)"
        else:
            mist_photons = mist_gains**2
            ylabel = "Mist Pulse Gain^2 (a.u.)"

        im = NonUniformImage(ax, interpolation="nearest", cmap="RdBu_r")
        im.set_data(flux_gains, mist_photons, real_signals.T)
        im.set_extent(
            (flux_gains[0], flux_gains[-1], mist_photons[0], mist_photons[-1])
        )
        ax.add_artist(im)
        ax.set_xlabel("Flux Pulse Gain (a.u.)")
        ax.set_ylabel(ylabel)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[MistResult] = None,
        comment: Optional[str] = None,
        tag: str = "fastflux/mist",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        flux_gains, mist_gains, signals2D = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Flux Pulse Gain", "unit": "a.u.", "values": flux_gains},
            y_info={"name": "Mist Pulse Gain", "unit": "a.u.", "values": mist_gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> MistResult:
        signals2D, flux_gains, mist_gains, comment = load_data(filepath, return_comment=True, **kwargs)
        assert mist_gains is not None
        assert len(flux_gains.shape) == 1 and len(mist_gains.shape) == 1
        assert signals2D.shape == (len(flux_gains), len(mist_gains))

        flux_gains = flux_gains.astype(np.float64)
        mist_gains = mist_gains.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.last_cfg = MistCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (flux_gains, mist_gains, signals2D)

        return flux_gains, mist_gains, signals2D
