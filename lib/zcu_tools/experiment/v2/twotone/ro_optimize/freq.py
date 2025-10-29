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

FreqResultType = Tuple[np.ndarray, np.ndarray]  # (fpts, snrs)


class OptimizeFreqExperiment(AbsExperiment[FreqResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        # prepend ge sweep as outer loop
        cfg["sweep"] = {
            "ge": {"start": 0, "stop": cfg["qub_pulse"]["gain"], "expts": 2},
            "freq": cfg["sweep"]["freq"],
        }

        cfg["qub_pulse"]["gain"] = sweep2param("ge", cfg["sweep"]["ge"])
        set_readout_cfg(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        with LivePlotter1D("Frequency (MHz)", "SNR", disable=not progress) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        TwoToneProgram(soccfg, ctx.cfg).acquire(
                            soc,
                            progress=False,
                            callback=update_hook,
                            record_stderr=True,
                        )
                    ),
                    raw2signal_fn=lambda raw: snr_as_signal(raw, axis=0),
                    result_shape=(len(fpts),),
                ),
                update_hook=lambda ctx: viewer.update(
                    fpts, np.abs(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (fpts, signals)

        return fpts, signals  # fpts

    def analyze(
        self,
        result: Optional[FreqResultType] = None,
        *,
        plot: bool = True,
        smooth: float = 1.0,
    ) -> float:
        if result is None:
            result = self.last_result

        fpts, signals = result

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, smooth)

        max_id = np.argmax(snrs)
        max_fpt = float(fpts[max_id])
        max_snr = float(snrs[max_id])

        if plot:
            plt.figure(figsize=config.figsize)
            plt.plot(fpts, snrs)
            plt.axvline(max_fpt, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("SNR (a.u.)")
            plt.legend()
            plt.show()

        return max_fpt

    def save(
        self,
        filepath: str,
        result: Optional[FreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result

        fpts, singals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": singals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
