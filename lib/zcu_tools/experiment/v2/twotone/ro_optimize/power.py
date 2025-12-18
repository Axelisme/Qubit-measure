from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typing_extensions import NotRequired

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
from zcu_tools.utils.datasaver import load_data, save_data

PowerResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


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
                                record_statistic=True,
                            ),
                            prog.get_covariance(),
                            prog.get_median(),
                        )
                    ),
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    result_shape=(len(gains),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(gains, np.abs(ctx.data)),
            )

        # record the last cfg and result
        self.last_cfg = cfg
        self.last_result = (gains, signals)

        return gains, signals

    def analyze(
        self, result: Optional[PowerResultType] = None, penalty_ratio: float = 0.0
    ) -> Tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        powers, signals = result

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, 1)
        penaltized_snrs = snrs * np.exp(-powers * penalty_ratio)

        max_id = np.argmax(penaltized_snrs)
        max_power = float(powers[max_id])
        max_snr = float(snrs[max_id])

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(powers, snrs)
        ax.axvline(max_power, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Readout Power")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return max_power, fig

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

    def load(self, filepath: str, **kwargs) -> PowerResultType:
        signals, pdrs, _ = load_data(filepath, **kwargs)
        assert pdrs is not None
        assert len(pdrs.shape) == 1 and len(signals.shape) == 1
        assert pdrs.shape == signals.shape

        pdrs = pdrs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (pdrs, signals)

        return pdrs, signals
