from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v1 import TwoToneProgram
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real

from ...template import sweep1D_soft_template


def rabi_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


LenRabiResultType = Tuple[np.ndarray, np.ndarray]


class LenRabiExperiment(AbsExperiment[LenRabiResultType]):
    """Rabi oscillation by varying pulse *length*."""

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> LenRabiResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"]["length"]

        lens = sweep2array(len_sweep, allow_array=True)

        del cfg["sweep"]

        def updateCfg(cfg, _, length) -> None:
            cfg["dac"]["qub_pulse"]["length"] = length

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            prog = TwoToneProgram(soccfg, cfg)
            (avgi, avgq), _ = prog.acquire(soc, progress=False, callback=cb)
            return avgi[0] + 1j * avgq[0]

        signals = sweep1D_soft_template(
            cfg,
            measure_fn,
            LivePlotter1D("Length (us)", "Signal", disable=not progress),
            xs=lens,
            updateCfg=updateCfg,
            signal2real=rabi_signal2real,
            progress=progress,
        )

        self.last_cfg = cfg
        self.last_result = (lens, signals)

        return lens, signals

    def analyze(
        self,
        result: Optional[LenRabiResultType] = None,
        *,
        decay: bool = True,
        plot: bool = True,
        max_contrast: bool = True,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        if max_contrast:
            real_signals = rotate2real(signals).real
        else:
            real_signals = np.abs(signals)

        pi_len, pi2_len, y_fit, _ = fit_rabi(lens, real_signals, decay=decay)

        if plot:
            plt.figure(figsize=config.figsize)
            plt.tight_layout()
            plt.plot(lens, real_signals, label="meas", ls="-", marker="o", markersize=3)
            plt.plot(lens, y_fit, label="fit")
            plt.axvline(pi_len, ls="--", c="red", label=f"pi = {pi_len:.3g}")
            plt.axvline(pi2_len, ls="--", c="red", label=f"pi/2 = {pi2_len:.3g}")
            plt.xlabel("Pulse length (us)")
            plt.ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
            plt.legend(loc=4)
            plt.show()

        return pi_len, pi2_len

    def save(
        self,
        filepath: str,
        result: Optional[LenRabiResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/rabi_length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
