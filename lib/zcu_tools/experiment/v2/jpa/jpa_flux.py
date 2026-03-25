from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typeguard import check_type
from typing_extensions import Any, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, set_flux_in_dev_cfg
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.tracker import PCATracker
from zcu_tools.experiment.v2.utils import make_ge_sweep, snr_as_signal, sweep2array
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

JPAFluxResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class JPAFluxModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class JPAFluxCfg(ModularProgramCfg, TaskCfg):
    modules: JPAFluxModuleCfg
    sweep: dict[str, SweepCfg]


class JPAFluxExp(AbsExperiment[JPAFluxResult, JPAFluxCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> JPAFluxResult:
        _cfg = check_type(deepcopy(cfg), JPAFluxCfg)

        _cfg["sweep"] = format_sweep1D(_cfg["sweep"], "jpa_flux")

        jpa_flxs = sweep2array(_cfg["sweep"]["jpa_flux"], allow_array=True)

        modules = _cfg["modules"]
        _cfg["sweep"] = {"ge": make_ge_sweep()}
        Pulse.set_param(
            modules["pi_pulse"], "on/off", sweep2param("ge", _cfg["sweep"]["ge"])
        )

        with LivePlotter1D("JPA Flux value (a.u.)", "Signal Difference") as viewer:

            def measure_fn(ctx, update_hook):
                modules = ctx.cfg["modules"]
                prog = ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        Pulse("pi_pulse", modules["pi_pulse"]),
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
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    dtype=np.float64,
                ).scan(
                    "JPA Flux value",
                    jpa_flxs.tolist(),
                    before_each=lambda i, ctx, flx: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flx, label="jpa_flux_dev"
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(jpa_flxs, np.abs(ctx.root_data)),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (jpa_flxs, signals)

        return jpa_flxs, signals

    def analyze(self, result: Optional[JPAFluxResult] = None) -> tuple[float, Figure]:
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
        result: Optional[JPAFluxResult] = None,
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

    def load(self, filepath: str, **kwargs) -> JPAFluxResult:
        signals, jpa_flxs, _ = load_data(filepath, **kwargs)
        assert jpa_flxs is not None
        assert len(jpa_flxs.shape) == 1 and len(signals.shape) == 1
        assert jpa_flxs.shape == signals.shape

        jpa_flxs = jpa_flxs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (jpa_flxs, signals)

        return jpa_flxs, signals
