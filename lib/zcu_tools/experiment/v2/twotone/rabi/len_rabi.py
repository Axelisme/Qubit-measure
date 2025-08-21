from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real

from ...template import sweep1D_soft_template, sweep_hard_template


def rabi_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


LenRabiResultType = Tuple[np.ndarray, np.ndarray]  # (lens, signals)


class LenRabiExperiment(AbsExperiment[LenRabiResultType]):
    """Rabi oscillation by varying pulse *length*."""

    def _run_for_flat(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> LenRabiResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        qub_pulse = cfg["qub_pulse"]

        assert qub_pulse["style"] in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"]["length"]

        lens = sweep2array(len_sweep)  # predicted

        qub_pulse["length"] = sweep2param("length", len_sweep)

        prog = TwoToneProgram(soccfg, cfg)

        signals = sweep_hard_template(
            cfg,
            lambda _, cb: prog.acquire(soc, progress=progress, callback=cb)[0][0].dot(
                [1, 1j]
            ),
            LivePlotter1D("Length (us)", "Signal", disable=not progress),
            ticks=(lens,),
            signal2real=rabi_signal2real,
        )

        real_lens = prog.get_pulse_param("qubit_pulse", "length", as_array=True)
        assert isinstance(real_lens, np.ndarray)
        real_lens += lens[0] - real_lens[0]  # correct absolute offset

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (real_lens, signals)

        return real_lens, signals

    def _run_for_arb(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> LenRabiResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"]["length"]
        del cfg["sweep"]

        lens = sweep2array(len_sweep)  # predicted

        def updateCfg(cfg: Dict[str, Any], _: int, length: Any) -> None:
            qub_pulse = cfg["qub_pulse"]

            qub_pulse["length"] = length
            if qub_pulse["style"] == "gauss":
                # TODO: better way to derive sigma?
                qub_pulse["sigma"] = length / 5.0

        # initialize pulse length
        updateCfg(cfg, 0, lens[0])

        def measure_fn(cfg: Dict[str, Any], callback) -> np.ndarray:
            return (
                TwoToneProgram(soccfg, cfg)
                .acquire(soc, progress=False, callback=callback)[0][0]
                .dot([1, 1j])
            )

        signals = sweep1D_soft_template(
            cfg,
            measure_fn,
            LivePlotter1D("Length (us)", "Signal", disable=not progress),
            xs=lens,
            updateCfg=updateCfg,
            signal2real=rabi_signal2real,
            progress=progress,
        )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (lens, signals)

        return lens, signals

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> LenRabiResultType:
        qub_pulse = cfg["qub_pulse"]

        if qub_pulse["style"] in ["const", "flat_top"]:
            # use hard sweep for flat top pulse
            return self._run_for_flat(soc, soccfg, cfg, progress=progress)
        else:
            # use soft sweep for arb pulse
            return self._run_for_arb(soc, soccfg, cfg, progress=progress)

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
