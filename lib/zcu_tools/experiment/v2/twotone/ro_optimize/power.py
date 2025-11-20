from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data

from .base import snr_as_signal

PowerResultType = Tuple[np.ndarray, np.ndarray]  # (powers, snrs)


class OptimizePowerTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class OptimizePowerExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: OptimizePowerTaskConfig) -> PowerResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "power")
        cfg["sweep"] = {"ge": make_ge_sweep(), "power": cfg["sweep"]["power"]}

        gains = sweep2array(cfg["sweep"]["power"])  # predicted power points

        Pulse.set_param(
            cfg["qub_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )
        Readout.set_param(
            cfg["readout"], "gain", sweep2param("power", cfg["sweep"]["power"])
        )

        with LivePlotter1D("Readout Power", "SNR") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        (
                            prog := ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset(
                                        "reset",
                                        ctx.cfg.get("reset", {"type": "none"}),
                                    ),
                                    Pulse("qub_pulse", ctx.cfg["qub_pulse"]),
                                    Readout("readout", ctx.cfg["readout"]),
                                ],
                            )
                        )
                        and (
                            prog.acquire(
                                soc,
                                progress=False,
                                callback=update_hook,
                                record_stderr=True,
                            ),
                            prog.get_stderr(),
                        )
                    ),
                    raw2signal_fn=lambda raw: snr_as_signal(raw, axis=0),
                    result_shape=(len(gains),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(gains, np.abs(ctx.data)),
            )

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (gains, signals)

        return gains, signals

    def analyze(self, result: Optional[PowerResultType] = None) -> float:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        powers, signals = result

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, 1)

        max_id = np.argmax(snrs)
        max_power = float(powers[max_id])
        max_snr = float(snrs[max_id])

        plt.figure(figsize=config.figsize)
        plt.plot(powers, snrs)
        plt.axvline(max_power, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        plt.xlabel("Readout Power")
        plt.ylabel("SNR (a.u.)")
        plt.legend()
        plt.grid(True)
        plt.show()

        return max_power

    def save(
        self,
        filepath: str,
        result: Optional[PowerResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/gain",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Probe Power", "unit": "a.u.", "values": pdrs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
