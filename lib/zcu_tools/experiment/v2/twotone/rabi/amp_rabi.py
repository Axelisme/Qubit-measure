from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import TwoToneProgram, TwoToneProgramCfg, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real

# (amps, signals)
AmpRabiResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AmpRabiTaskConfig(TaskConfig, TwoToneProgramCfg): ...


class AmpRabiExp(AbsExperiment[AmpRabiResultType, AmpRabiTaskConfig]):
    def run(self, soc, soccfg, cfg: AmpRabiTaskConfig) -> AmpRabiResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "gain")

        amps = sweep2array(cfg["sweep"]["gain"])  # predicted

        cfg["qub_pulse"]["gain"] = sweep2param("gain", cfg["sweep"]["gain"])

        with LivePlotter1D("Pulse gain", "Amplitude") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        TwoToneProgram(soccfg, ctx.cfg).acquire(
                            soc, progress=False, callback=update_hook
                        )
                    ),
                    result_shape=(len(amps),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(amps, rabi_signal2real(ctx.data)),
            )

        self.last_cfg = cfg
        self.last_result = (amps, signals)

        return amps, signals

    def analyze(
        self, result: Optional[AmpRabiResultType] = None
    ) -> Tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result
        real_signals = rabi_signal2real(signals)

        if real_signals[0] > 0.5 * (np.max(real_signals) + np.min(real_signals)):
            init_phase = 0.0
        else:
            init_phase = 180

        pi_amp, pi2_amp, _, y_fit, _ = fit_rabi(
            pdrs, real_signals, decay=False, init_phase=init_phase, min_length=0.0
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(pdrs, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(pdrs, y_fit, label="fit")
        ax.axvline(pi_amp, ls="--", c="red", label=f"pi = {pi_amp:.3g}")
        ax.axvline(pi2_amp, ls="--", c="red", label=f"pi/2 = {pi2_amp:.3g}")
        ax.set_xlabel("Pulse gain (a.u.)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        return pi_amp, pi2_amp, fig

    def save(
        self,
        filepath: str,
        result: Optional[AmpRabiResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/rabi_gain",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Gain", "unit": "", "values": pdrs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> AmpRabiResultType:
        signals, pdrs, _ = load_data(filepath, **kwargs)
        assert pdrs is not None
        assert len(pdrs.shape) == 1 and len(signals.shape) == 1
        assert pdrs.shape == signals.shape

        pdrs = pdrs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (pdrs, signals)

        return pdrs, signals
