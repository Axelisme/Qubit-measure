from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typing_extensions import NotRequired
from matplotlib.figure import Figure

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.experiment.v2.utils import snr_as_signal
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

FreqResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


class OptimizeFreqTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class OptimizeFreqExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: OptimizeFreqTaskConfig) -> FreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        cfg["sweep"] = {"ge": make_ge_sweep(), "freq": cfg["sweep"]["freq"]}

        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        Pulse.set_param(
            cfg["qub_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )
        Readout.set_param(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        with LivePlotter1D("Frequency (MHz)", "SNR") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        (
                            prog := ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset(
                                        "reset", ctx.cfg.get("reset", {"type": "none"})
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
                                record_statistic=True,
                            ),
                            prog.get_covariance(),
                            prog.get_median(),
                        )
                    ),
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    result_shape=(len(fpts),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(fpts, np.abs(ctx.data)),
            )

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (fpts, signals)

        return fpts, signals  # fpts

    def analyze(
        self, result: Optional[FreqResultType] = None, *, smooth: float = 1.0
    ) -> Tuple[float, Figure]:
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

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(fpts, snrs)
        ax.axvline(max_fpt, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return max_fpt, fig

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
