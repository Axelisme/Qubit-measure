from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import rotate2real

from ..runner import HardTask, Runner


def qubfreq_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


FreqResultType = Tuple[np.ndarray, np.ndarray]


class FreqExperiment(AbsExperiment[FreqResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> FreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # canonicalise sweep section to single-axis form «{'freq': ...}»
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

        # predicted sweep points before FPGA coercion
        fpts = sweep2array(cfg["sweep"]["freq"])  # MHz

        # bind sweep parameter as *QickParam* so it is executed by FPGA
        cfg["qub_pulse"]["freq"] = sweep2param("freq", cfg["sweep"]["freq"])

        with LivePlotter1D(
            "Frequency (MHz)", "Amplitude", disable=not progress
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        TwoToneProgram(soccfg, ctx.cfg).acquire(
                            soc, progress=False, callback=update_hook
                        )
                    ),
                    result_shape=(len(fpts),),
                ),
                update_hook=lambda ctx: viewer.update(
                    fpts,
                    qubfreq_signal2real(np.asarray(ctx.get_data())),  # type: ignore
                ),
            ).run(cfg)
            signals = np.asarray(signals)  # type: ignore

        # cache
        self.last_cfg = cfg
        self.last_result = (fpts, signals)

        return fpts, signals

    def analyze(
        self,
        result: Optional[FreqResultType] = None,
        *,
        model_type: Literal["lor", "sinc"] = "lor",
        plot_fit: bool = True,
    ) -> Tuple[float, float, plt.Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        # discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)  # type: ignore
        fpts = fpts[val_mask]
        signals = signals[val_mask]

        y = rotate2real(signals).real  # type: ignore

        freq, freq_err, kappa, _, y_fit, _ = fit_qubit_freq(fpts, y, model_type)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, plt.Figure)

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
        result: Optional[FreqResultType] = None,
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
