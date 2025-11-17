from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, Runner, TaskContext
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import ModularProgramV2, Pulse, Readout, Reset, sweep2param
from zcu_tools.utils.datasaver import save_data

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
            "ge": make_ge_sweep(),
            "freq": cfg["sweep"]["freq"],
        }

        Pulse.set_param(
            cfg["qub_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )
        Readout.set_param(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        def measure_fn(
            ctx: TaskContext, update_hook: Callable[[int, Any], None]
        ) -> Tuple[np.ndarray, np.ndarray]:
            prog = ModularProgramV2(
                soccfg,
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse("qub_pulse", ctx.cfg["qub_pulse"]),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            )
            avg_d = prog.acquire(
                soc, progress=False, callback=update_hook, record_stderr=True
            )
            std_d = prog.get_stderr()
            return avg_d, std_d

        with LivePlotter1D("Frequency (MHz)", "SNR", disable=not progress) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=measure_fn,
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
        self, result: Optional[FreqResultType] = None, *, smooth: float = 1.0
    ) -> float:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, smooth)

        max_id = np.argmax(snrs)
        max_fpt = float(fpts[max_id])
        max_snr = float(snrs[max_id])

        plt.figure(figsize=config.figsize)
        plt.plot(fpts, snrs)
        plt.axvline(max_fpt, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("SNR (a.u.)")
        plt.legend()
        plt.grid(True)
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
        assert result is not None, "no result found"

        fpts, singals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": singals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
