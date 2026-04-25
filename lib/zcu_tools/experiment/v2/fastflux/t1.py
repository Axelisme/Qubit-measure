from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typing_extensions import Any, Callable, Optional, TypeAlias, cast

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program.v2 import (
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
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.process import rotate2real

# (gains, lengths, signals2D)
T1Result: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class T1ModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    flux_pulse: PulseCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1SweepCfg(ConfigBase):
    gain: SweepCfg
    length: SweepCfg


class T1Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1ModuleCfg
    sweep: T1SweepCfg


class T1Exp(AbsExperiment[T1Result, T1Cfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: T1Cfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> T1Result:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gain_sweep = cfg.sweep.gain
        length_sweep = cfg.sweep.length

        # uniform in square space
        lf_ch = modules.flux_pulse.ch
        gains = sweep2array(gain_sweep, "gain", {"soccfg": soccfg, "gen_ch": lf_ch})
        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg, "gen_ch": lf_ch})

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, T1Cfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: T1Cfg = cast(T1Cfg, ctx.cfg)
            modules = cfg.modules

            gain_sweep = cfg.sweep.gain
            length_sweep = cfg.sweep.length

            gain_param = sweep2param("gain", gain_sweep)
            length_param = sweep2param("length", length_sweep)
            modules.flux_pulse.set_param("gain", gain_param)
            modules.flux_pulse.set_param("length", length_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("pi_pulse", modules.pi_pulse),
                    Pulse("flux_pulse", modules.flux_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[
                    ("gain", gain_sweep),
                    ("length", length_sweep),
                ],
            ).acquire(
                soc, progress=False, round_hook=update_hook, **(acquire_kwargs or {})
            )

        with LivePlot2D("Flux Pulse Gain (a.u.)", "Time (us)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(gains), len(lengths)),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    gains, lengths, t1_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = deepcopy(cfg)
        self.last_result = (gains, lengths, signals)

        return gains, lengths, signals

    def analyze(self, result: Optional[T1Result] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, lengths, signals2D = result

        real_signals = t1_signal2real(signals2D)

        def is_good_fit(fit_signal, real_signal) -> bool:
            real_ptp = np.ptp(real_signal)
            fit_ptp = np.ptp(fit_signal)
            residual = real_signal - fit_signal
            smooth_residual = gaussian_filter1d(residual, sigma=1)
            return fit_ptp > 0.5 * real_ptp and fit_ptp > np.mean(
                np.abs(smooth_residual)
            )

        prev_pOpt = None
        list_pOpt = []
        for real_signal in real_signals:
            *_, fit_signal, (pOpt, _) = fit_decay(
                lengths,
                real_signal,
                fit_params=prev_pOpt,
            )
            if is_good_fit(fit_signal, real_signal):
                prev_pOpt = pOpt
            else:
                prev_pOpt = None
            list_pOpt.append(prev_pOpt)
        mean_y0 = np.median([pOpt[0] for pOpt in list_pOpt if pOpt is not None]).item()

        t1s = np.full_like(gains, np.nan)
        t1errs = np.zeros_like(gains)
        for i, real_signal in enumerate(real_signals):
            t1, t1err, fit_signal, *_ = fit_decay(
                lengths,
                real_signals[i, :],
                fit_params=list_pOpt[i],
                fixedparams=(mean_y0, None, None),
            )
            if is_good_fit(fit_signal, real_signal):
                t1s[i] = t1
                t1errs[i] = t1err

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.imshow(
            real_signals.T,
            extent=[gains[0], gains[-1], lengths[0], lengths[-1]],
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        # ax1.set_xlabel("Flux Pulse Gain (a.u.)")
        ax.set_ylabel("Time (us)")

        ax.errorbar(gains, t1s, yerr=t1errs, fmt=".", label="T1", color="black")
        ax.set_xlabel("Flux Pulse Gain (a.u.)")
        ax.legend()

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[T1Result] = None,
        comment: Optional[str] = None,
        tag: str = "fastflux/t1",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, lengths, signals2D = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Flux Pulse Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Time", "unit": "s", "values": lengths * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> T1Result:
        signals2D, gains, lengths, comment = load_data(filepath, return_comment=True, **kwargs)
        assert lengths is not None
        assert len(gains.shape) == 1 and len(lengths.shape) == 1
        assert signals2D.shape == (len(gains), len(lengths))

        lengths = lengths * 1e-6  # s -> us

        gains = gains.astype(np.float64)
        lengths = lengths.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.last_cfg = T1Cfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (gains, lengths, signals2D)

        return gains, lengths, signals2D
