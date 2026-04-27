from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Optional, TypeAlias

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    Pulse,
    SweepCfg,
    TwoToneCfg,
    TwoToneProgram,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real

# (amps, signals)
AmpRabiResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AmpRabiSweepCfg(ConfigBase):
    gain: SweepCfg


class AmpRabiCfg(TwoToneCfg, ExpCfgModel):
    sweep: AmpRabiSweepCfg


class AmpRabiExp(AbsExperiment[AmpRabiResult, AmpRabiCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: AmpRabiCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> AmpRabiResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        gain_param = sweep2param("gain", cfg.sweep.gain)
        modules.qub_pulse.set_param("gain", gain_param)

        with LivePlot1D("Pulse gain", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    pbar_n=cfg.rounds,
                    measure_fn=lambda ctx, update_hook: TwoToneProgram(
                        soccfg,
                        ctx.cfg,
                        sweep=[("gain", ctx.cfg.sweep.gain)],
                    ).acquire(
                        soc,
                        progress=False,
                        round_hook=update_hook,
                        **(acquire_kwargs or {}),
                    ),
                    result_shape=(len(gains),),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    gains, rabi_signal2real(ctx.root_data)
                ),
            )

        self.last_cfg = deepcopy(cfg)
        self.last_result = (gains, signals)

        return gains, signals

    def analyze(
        self, result: Optional[AmpRabiResult] = None, skip: int = 0
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, signals = result
        gains = gains[skip:]
        signals = signals[skip:]

        real_signals = rabi_signal2real(signals)

        if real_signals[0] > 0.5 * (np.max(real_signals) + np.min(real_signals)):
            init_phase = 0.0
        else:
            init_phase = 180

        pi_amp, pi_amp_err, pi2_amp, pi2_amp_err, _, _, y_fit, _ = fit_rabi(
            gains, real_signals, decay=False, init_phase=init_phase
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(gains, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(gains, y_fit, label="fit")
        ax.axvline(
            pi_amp, ls="--", c="red", label=f"pi = {pi_amp:.3g} ± {pi_amp_err:.2g}"
        )
        ax.axvspan(pi_amp - pi_amp_err, pi_amp + pi_amp_err, color="red", alpha=0.2)
        ax.axvline(
            pi2_amp,
            ls="--",
            c="red",
            label=f"pi/2 = {pi2_amp:.3g} ± {pi2_amp_err:.2g}",
        )
        ax.axvspan(pi2_amp - pi2_amp_err, pi2_amp + pi2_amp_err, color="red", alpha=0.2)
        ax.set_xlabel("Pulse gain (a.u.)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        return pi_amp, pi2_amp, fig

    def save(
        self,
        filepath: str,
        result: Optional[AmpRabiResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/rabi_gain",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, signals = result
        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Gain", "unit": "", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> AmpRabiResult:
        signals, gains, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert gains is not None
        assert len(gains.shape) == 1 and len(signals.shape) == 1
        assert gains.shape == signals.shape

        gains = gains.astype(np.float64)
        signals = signals.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                self.last_cfg = AmpRabiCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (gains, signals)

        return gains, signals
