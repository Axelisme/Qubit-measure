from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, Runner, SoftTask
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import Pulse, TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real


def rabi_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


LenRabiResultType = Tuple[np.ndarray, np.ndarray]  # (lens, signals)


class LenRabiExperiment(AbsExperiment[LenRabiResultType]):
    def _run_for_flat(self, soc, soccfg, cfg: Dict[str, Any]) -> LenRabiResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        assert cfg["qub_pulse"]["waveform"]["style"] in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        lens = sweep2array(cfg["sweep"]["length"])  # predicted

        Pulse.set_param(
            cfg["qub_pulse"], "length", sweep2param("length", cfg["sweep"]["length"])
        )

        with LivePlotter1D("Length (us)", "Signal") as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        TwoToneProgram(soccfg, ctx.cfg).acquire(
                            soc, progress=False, callback=update_hook
                        )
                    ),
                    result_shape=(len(lens),),
                ),
                update_hook=lambda ctx: viewer.update(
                    lens, rabi_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (lens, signals)

        return lens, signals

    def _run_for_arb(self, soc, soccfg, cfg: Dict[str, Any]) -> LenRabiResultType:
        cfg = deepcopy(cfg)  # avoid in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"].pop("length")

        lens = sweep2array(len_sweep)  # predicted

        with LivePlotter1D("Length (us)", "Signal") as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="length",
                    sweep_values=lens,
                    update_cfg_fn=lambda _, ctx, length: Pulse.set_param(
                        ctx.cfg["qub_pulse"], "length", length
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            TwoToneProgram(soccfg, ctx.cfg).acquire(
                                soc, progress=False, callback=update_hook
                            )
                        ),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    lens, rabi_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (lens, signals)

        return lens, signals

    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> LenRabiResultType:
        qub_waveform = cfg["qub_pulse"]["waveform"]

        if qub_waveform["style"] in ["const", "flat_top"]:
            # use hard sweep for flat top pulse
            return self._run_for_flat(soc, soccfg, cfg)
        else:
            # use soft sweep for arb pulse
            return self._run_for_arb(soc, soccfg, cfg)

    def analyze(
        self, result: Optional[LenRabiResultType] = None, *, decay: bool = True
    ) -> Tuple[float, float, float, plt.Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        real_signals = rabi_signal2real(signals)

        nan_mask = np.isnan(real_signals)
        if np.all(nan_mask):
            raise ValueError("All data are NaN!")

        lens = lens[~nan_mask]
        real_signals = real_signals[~nan_mask]

        pi_len, pi2_len, freq, y_fit, _ = fit_rabi(
            lens, real_signals, decay=decay, init_phase=None
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, plt.Figure)

        ax.plot(lens, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(lens, y_fit, label="fit")
        ax.axvline(pi_len, ls="--", c="red", label=f"pi = {pi_len:.3g} μs")
        ax.axvline(pi2_len, ls="--", c="red", label=f"pi/2 = {pi2_len:.3g} μs")
        ax.set_xlabel("Pulse length (μs)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.set_title(f"Rabi Oscillation (f={freq:.3f} MHz)")
        ax.legend(loc=4)

        fig.tight_layout()

        return pi_len, pi2_len, freq, fig

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
