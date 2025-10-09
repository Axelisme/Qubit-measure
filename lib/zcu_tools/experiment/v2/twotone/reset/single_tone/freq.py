from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    set_reset_cfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_resonence_freq
from zcu_tools.utils.process import rotate2real

from ....runner import HardTask, Runner

# (fpts, signals)
SingleToneResetFreqResultType = Tuple[np.ndarray, np.ndarray]


def reset_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class FreqExperiment(AbsExperiment[SingleToneResetFreqResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> SingleToneResetFreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Canonicalise sweep section to single-axis form
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        # Check that reset pulse is single pulse type
        if cfg["tested_reset"]["type"] != "pulse":
            raise ValueError("This experiment only supports single pulse reset")

        set_reset_cfg(
            cfg["tested_reset"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        with LivePlotter1D(
            "Frequency (MHz)", "Amplitude", disable=not progress
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                make_reset("reset", ctx.cfg.get("reset")),
                                Pulse("init_pulse", ctx.cfg.get("init_pulse")),
                                make_reset("tested_reset", ctx.cfg["tested_reset"]),
                                make_readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(fpts),),
                ),
                update_hook=lambda ctx: viewer.update(
                    fpts, reset_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (fpts, signals)

        return fpts, signals

    def analyze(
        self,
        result: Optional[SingleToneResetFreqResultType] = None,
        *,
        type: Literal["lor", "sinc"] = "lor",
        plot: bool = True,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        # Discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)
        fpts = fpts[val_mask]
        signals = signals[val_mask]

        real_signals = reset_signal2real(signals)

        freq, freq_err, kappa, _, y_fit, _ = fit_resonence_freq(
            fpts, real_signals, type=type
        )

        if plot:
            plt.figure(figsize=config.figsize)
            plt.tight_layout()
            plt.plot(fpts, real_signals, label="signal", marker="o", markersize=3)
            plt.plot(fpts, y_fit, label=f"fit, κ = {kappa:.1g} MHz")
            label = f"f_reset = {freq:.5g} ± {freq_err:.1g} MHz"
            plt.axvline(freq, color="r", ls="--", label=label)
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Signal Real (a.u.)")
            plt.title("Reset frequency optimization")
            plt.legend()
            plt.grid(True)
            plt.show()

        return freq, kappa

    def save(
        self,
        filepath: str,
        result: Optional[SingleToneResetFreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/reset/single_tone/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
