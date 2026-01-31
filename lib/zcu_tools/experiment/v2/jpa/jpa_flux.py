from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typing_extensions import NotRequired

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import (
    format_sweep1D,
    make_ge_sweep,
    set_flux_in_dev_cfg,
    sweep2array,
)
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.experiment.v2.tracker import PCATracker
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

JPAFluxResultType = Tuple[NDArray[np.float64], NDArray[np.float64]]


class JPAFluxTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg

    dev: Mapping[str, DeviceInfo]


class JPAFluxExp(AbsExperiment[JPAFluxResultType, JPAFluxTaskConfig]):
    def run(self, soc, soccfg, cfg: JPAFluxTaskConfig) -> JPAFluxResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "jpa_flux")

        jpa_flxs = sweep2array(cfg["sweep"]["jpa_flux"], allow_array=True)

        cfg["sweep"] = {"ge": make_ge_sweep()}
        Pulse.set_param(
            cfg["pi_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        with LivePlotter1D("JPA Flux value (a.u.)", "Signal Difference") as viewer:

            def measure_fn(ctx, update_hook):
                prog = ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset(
                            "reset",
                            ctx.cfg.get("reset", {"type": "none"}),
                        ),
                        Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                        Readout("readout", ctx.cfg["readout"]),
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
                task=SoftTask(
                    sweep_name="JPA Flux value",
                    sweep_values=jpa_flxs.tolist(),
                    update_cfg_fn=lambda i, ctx, flx: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flx, label="jpa_flux_dev"
                    ),
                    sub_task=HardTask(
                        measure_fn=measure_fn,
                        raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(jpa_flxs, np.abs(ctx.data)),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (jpa_flxs, signals)

        return jpa_flxs, signals

    def analyze(
        self, result: Optional[JPAFluxResultType] = None
    ) -> Tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_flxs, signals = result
        signals = gaussian_filter1d(signals, sigma=1)
        snrs = np.abs(signals)

        max_idx = np.argmax(snrs)
        best_jpa_flux = jpa_flxs[max_idx]

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.plot(jpa_flxs, snrs, label="signal difference")
        ax.axvline(
            best_jpa_flux,
            color="r",
            ls="--",
            label=f"best JPA flux = {best_jpa_flux:.2g} a.u.",
        )
        ax.set_xlabel("JPA Flux value (a.u.)")
        ax.set_ylabel("Signal Difference (a.u.)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return float(best_jpa_flux), fig

    def save(
        self,
        filepath: str,
        result: Optional[JPAFluxResultType] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/flux",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_flxs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "JPA Flux value", "unit": "a.u.", "values": jpa_flxs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> JPAFluxResultType:
        signals, jpa_flxs, _ = load_data(filepath, **kwargs)
        assert jpa_flxs is not None
        assert len(jpa_flxs.shape) == 1 and len(signals.shape) == 1
        assert jpa_flxs.shape == signals.shape

        jpa_flxs = jpa_flxs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (jpa_flxs, signals)

        return jpa_flxs, signals
