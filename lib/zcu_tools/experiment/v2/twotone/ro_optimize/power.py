from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typeguard import check_type
from typing_extensions import NotRequired, TypedDict

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskCfg, run_task
from zcu_tools.experiment.v2.tracker import PCATracker
from zcu_tools.experiment.v2.utils import snr_as_signal
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
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

PowerResult = Tuple[NDArray[np.float64], NDArray[np.float64]]


class PowerModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class PowerCfg(ModularProgramCfg, TaskCfg):
    modules: PowerModuleCfg
    sweep: Dict[str, SweepCfg]


class PowerExp(AbsExperiment[PowerResult, PowerCfg]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> PowerResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "power")
        _cfg = check_type(deepcopy(cfg), PowerCfg)

        _cfg["sweep"] = {"ge": make_ge_sweep(), "power": _cfg["sweep"]["power"]}

        gains = sweep2array(_cfg["sweep"]["power"])  # predicted power points

        modules = _cfg["modules"]
        Pulse.set_param(
            modules["qub_pulse"], "on/off", sweep2param("ge", _cfg["sweep"]["ge"])
        )
        Readout.set_param(
            modules["readout"], "gain", sweep2param("power", _cfg["sweep"]["power"])
        )

        with LivePlotter1D("Readout Power", "SNR") as viewer:

            def measure_fn(ctx, update_hook):
                modules = ctx.cfg["modules"]
                prog = ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        Pulse("qub_pulse", modules["qub_pulse"]),
                        Readout("readout", modules["readout"]),
                    ],
                )
                tracker = PCATracker()
                avg_d = prog.acquire(
                    soc,
                    progress=False,
                    callback=lambda i, avg_d: update_hook(
                        i, (avg_d, [tracker.covariance], [tracker.rough_median])
                    ),
                    statistic_trackers=[tracker],
                )
                return avg_d, [tracker.covariance], [tracker.rough_median]

            signals = run_task(
                task=HardTask(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    result_shape=(len(gains),),
                    dtype=np.float64,
                ),
                init_cfg=_cfg,
                update_hook=lambda ctx: viewer.update(gains, np.abs(ctx.data)),
            )

        # record the last cfg and result
        self.last_cfg = _cfg
        self.last_result = (gains, signals)

        return gains, signals

    def analyze(
        self, result: Optional[PowerResult] = None, penalty_ratio: float = 0.0
    ) -> Tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        powers, snrs = result
        snrs = np.abs(snrs)

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
        result: Optional[PowerResult] = None,
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

    def load(self, filepath: str, **kwargs) -> PowerResult:
        signals, pdrs, _ = load_data(filepath, **kwargs)
        assert pdrs is not None
        assert len(pdrs.shape) == 1 and len(signals.shape) == 1
        assert pdrs.shape == signals.shape

        pdrs = pdrs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (pdrs, signals)

        return pdrs, signals
