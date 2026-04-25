from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Literal, Optional, TypeAlias

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import SweepCfg, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import minus_background

FreqResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def qubfreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals))


class FreqSweepCfg(ConfigBase):
    freq: SweepCfg


class FreqCfg(TwoToneCfg, ExpCfgModel):
    sweep: FreqSweepCfg


class FreqExp(AbsExperiment[FreqResult, FreqCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> FreqResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        # predicted sweep points before FPGA coercion
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        # bind sweep parameter as *QickParam* so it is executed by FPGA
        freq_param = sweep2param("freq", cfg.sweep.freq)
        modules.qub_pulse.set_param("freq", freq_param)

        with LivePlot1D("Frequency (MHz)", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: TwoToneProgram(
                        soccfg, ctx.cfg, sweep=[("freq", ctx.cfg.sweep.freq)]
                    ).acquire(
                        soc,
                        progress=False,
                        round_hook=update_hook,
                        **(acquire_kwargs or {}),
                    ),
                    result_shape=(len(freqs),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, qubfreq_signal2real(ctx.root_data)
                ),
            )

        # cache
        self.last_cfg = deepcopy(cfg)
        self.last_result = (freqs, signals)

        return freqs, signals

    def analyze(
        self,
        result: Optional[FreqResult] = None,
        *,
        model_type: Literal["lor", "sinc"] = "lor",
        plot_fit: bool = True,
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, signals = result

        # discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        freqs = freqs[val_mask]
        signals = signals[val_mask]

        real_signals = qubfreq_signal2real(signals)

        freq, freq_err, fwhm, _, y_fit, _ = fit_qubit_freq(
            freqs, real_signals, model_type
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(freqs, real_signals, label="signal", marker="o", markersize=3)
        if plot_fit:
            ax.plot(freqs, y_fit, label=f"fit, FWHM={fwhm:.1g} MHz")
            label = f"f_q = {freq:.5g} ± {freq_err:.1g} MHz"
            ax.axvline(freq, color="r", ls="--", label=label)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return freq, fwhm, fig

    def save(
        self,
        filepath: str,
        result: Optional[FreqResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        cfg = self.last_cfg
        assert cfg is not None

        freqs, signals = result
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "MHz", "values": freqs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        signals, freqs, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert freqs is not None
        assert len(freqs.shape) == 1 and len(signals.shape) == 1
        assert freqs.shape == signals.shape

        freqs = freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.last_cfg = FreqCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (freqs, signals)

        return freqs, signals
