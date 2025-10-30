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

from ...runner import HardTask, Runner, SoftTask
from .base import snr_as_signal

LengthResultType = Tuple[np.ndarray, np.ndarray]  # (lengths, snrs)


class OptimizeLengthExperiment(AbsExperiment[LengthResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> LengthResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        length_sweep = cfg["sweep"]["length"]

        # replace length sweep with ge sweep, and use soft loop for length
        cfg["sweep"] = {
            "ge": {"start": 0, "stop": cfg["qub_pulse"]["gain"], "expts": 2}
        }

        # set with / without pi gain for qubit pulse
        cfg["qub_pulse"]["gain"] = sweep2param("ge", cfg["sweep"]["ge"])

        lengths = sweep2array(length_sweep)  # predicted readout lengths

        # set initial readout length and adjust pulse length
        set_readout_cfg(cfg["readout"], "ro_length", lengths[0])
        set_readout_cfg(cfg["readout"], "length", lengths.max() + 0.1)

        def measure_fn(ctx, update_hook):
            prog = TwoToneProgram(soccfg, ctx.cfg)
            avg_d = prog.acquire(
                soc, progress=False, callback=update_hook, record_stderr=True
            )
            std_d = prog.get_stderr()
            return avg_d, std_d

        with LivePlotter1D(
            "Readout Length (us)", "SNR", disable=not progress
        ) as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="length",
                    sweep_values=lengths,
                    update_cfg_fn=lambda _, ctx, length: set_readout_cfg(
                        ctx.cfg["readout"], "ro_length", length
                    ),
                    sub_task=HardTask(
                        measure_fn=measure_fn,
                        raw2signal_fn=lambda raw: snr_as_signal(raw, axis=-1),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    lengths, np.abs(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (lengths, signals)

        return lengths, signals

    def analyze(
        self,
        result: Optional[LengthResultType] = None,
        *,
        plot: bool = True,
        t0: Optional[float] = None,
    ) -> float:
        if result is None:
            result = self.last_result

        lengths, signals = result

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, 1)

        if t0 is None:
            max_id = np.argmax(snrs)
        else:
            max_id = np.argmax(snrs / np.sqrt(lengths + t0))

        max_length = float(lengths[max_id])
        max_snr = float(snrs[max_id])

        if plot:
            plt.figure(figsize=config.figsize)
            plt.plot(lengths, snrs)
            plt.axvline(
                max_length, color="r", ls="--", label=f"max SNR = {max_snr:.2f}"
            )
            plt.xlabel("Readout Length (us)")
            plt.ylabel("SNR (a.u.)")
            plt.legend()
            plt.show()

        return max_length

    def save(
        self,
        filepath: str,
        result: Optional[LengthResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result

        lengths, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Readout Length", "unit": "s", "values": lengths * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
