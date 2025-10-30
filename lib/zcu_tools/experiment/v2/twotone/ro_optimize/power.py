from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import TwoToneProgram, set_readout_cfg, sweep2param
from zcu_tools.utils.datasaver import save_data

from ...runner import HardTask, Runner
from .base import snr_as_signal

PowerResultType = Tuple[np.ndarray, np.ndarray]  # (powers, snrs)


class OptimizePowerExperiment(AbsExperiment[PowerResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> PowerResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "power")

        gains = sweep2array(cfg["sweep"]["power"])  # predicted power points

        # prepend ge sweep as outer loop
        cfg["sweep"] = {
            "ge": {"start": 0, "stop": cfg["qub_pulse"]["gain"], "expts": 2},
            "power": cfg["sweep"]["power"],
        }

        cfg["qub_pulse"]["gain"] = sweep2param("ge", cfg["sweep"]["ge"])
        set_readout_cfg(
            cfg["readout"], "gain", sweep2param("power", cfg["sweep"]["power"])
        )

        def measure_fn(ctx, update_hook):
            prog = TwoToneProgram(soccfg, ctx.cfg)
            avg_d = prog.acquire(
                soc, progress=False, callback=update_hook, record_stderr=True
            )
            std_d = prog.get_stderr()
            return avg_d, std_d

        with LivePlotter1D("Readout Power", "SNR", disable=not progress) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, axis=0),
                    result_shape=(len(gains),),
                ),
                update_hook=lambda ctx: viewer.update(
                    gains, np.abs(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (gains, signals)

        return gains, signals

    def analyze(
        self, result: Optional[PowerResultType] = None, *, plot: bool = True
    ) -> float:
        if result is None:
            result = self.last_result

        powers, signals = result

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, 1)

        max_id = np.argmax(snrs)
        max_power = float(powers[max_id])
        max_snr = float(snrs[max_id])

        if plot:
            plt.figure(figsize=config.figsize)
            plt.plot(powers, snrs)
            plt.axvline(max_power, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
            plt.xlabel("Readout Power")
            plt.ylabel("SNR (a.u.)")
            plt.legend()
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
