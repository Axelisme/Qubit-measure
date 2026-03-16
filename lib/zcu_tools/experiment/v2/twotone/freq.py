from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Dict, Literal, Optional, Tuple, TypeAlias

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import minus_background, rotate2real

FreqResult: TypeAlias = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def qubfreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals))


class FreqCfg(TwoToneCfg, TaskCfg):
    sweep: Dict[str, SweepCfg]


class FreqExp(AbsExperiment[FreqResult, FreqCfg]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> FreqResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        _cfg = check_type(deepcopy(cfg), FreqCfg)

        # predicted sweep points before FPGA coercion
        fpts = sweep2array(_cfg["sweep"]["freq"])  # MHz

        # bind sweep parameter as *QickParam* so it is executed by FPGA
        modules = _cfg["modules"]
        modules["qub_pulse"]["freq"] = sweep2param("freq", _cfg["sweep"]["freq"])

        with LivePlotter1D("Frequency (MHz)", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: TwoToneProgram(
                        soccfg, ctx.cfg
                    ).acquire(soc, progress=False, callback=update_hook),
                    result_shape=(len(fpts),),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    fpts, qubfreq_signal2real(ctx.root_data)
                ),
            )

        # cache
        self.last_cfg = _cfg
        self.last_result = (fpts, signals)

        return fpts, signals

    def analyze(
        self,
        result: Optional[FreqResult] = None,
        *,
        model_type: Literal["lor", "sinc"] = "lor",
        plot_fit: bool = True,
    ) -> Tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        # discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        fpts = fpts[val_mask]
        signals = signals[val_mask]

        y = rotate2real(signals).real

        freq, freq_err, kappa, _, y_fit, _ = fit_qubit_freq(fpts, y, model_type)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(fpts, y, label="signal", marker="o", markersize=3)
        if plot_fit:
            ax.plot(fpts, y_fit, label=f"fit, kappa={kappa:.1g} MHz")
            label = f"f_q = {freq:.5g} ± {freq_err:.1g} MHz"
            ax.axvline(freq, color="r", ls="--", label=label)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return freq, kappa, fig

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

        fpts, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "MHz", "values": fpts},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        signals, fpts, _ = load_data(filepath, **kwargs)
        assert fpts is not None
        assert len(fpts.shape) == 1 and len(signals.shape) == 1
        assert fpts.shape == signals.shape

        fpts = fpts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (fpts, signals)

        return fpts, signals
