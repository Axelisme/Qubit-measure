from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import (
    format_sweep1D,
    set_flux_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.tracker import MomentTracker
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data

FluxResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class FluxModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class FluxCfg(ModularProgramCfg, TaskCfg):
    modules: FluxModuleCfg
    sweep: dict[str, SweepCfg]


class FluxExp(AbsExperiment[FluxResult, FluxCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> FluxResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "jpa_flux")
        _cfg = check_type(deepcopy(cfg), FluxCfg)

        jpa_fluxs = sweep2array(_cfg["sweep"]["jpa_flux"], allow_array=True)

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any],
            update_hook: Optional[Callable[[int, list[MomentTracker]], None]],
        ) -> list[MomentTracker]:
            cfg: FluxCfg = cast(FluxCfg, ctx.cfg)
            setup_devices(cfg, progress=False)
            modules = cfg["modules"]

            assert update_hook is not None

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Branch("ge", [], Pulse("pi_pulse", modules["pi_pulse"])),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("ge", 2)],
            )
            tracker = MomentTracker()
            prog.acquire(
                soc,
                progress=False,
                callback=lambda i, avg_d: update_hook(i, [tracker]),
                statistic_trackers=[tracker],
            )
            return [tracker]

        with LivePlot1D("JPA Flux value (a.u.)", "Signal Difference") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    dtype=np.float64,
                    pbar_n=_cfg["rounds"],
                ).scan(
                    "JPA Flux value",
                    jpa_fluxs.tolist(),
                    before_each=lambda i, ctx, flux: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flux, label="jpa_flux_dev"
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(jpa_fluxs, np.abs(ctx.root_data)),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (jpa_fluxs, signals)

        return jpa_fluxs, signals

    def analyze(self, result: Optional[FluxResult] = None) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_fluxs, signals = result
        signals = gaussian_filter1d(signals, sigma=1)
        snrs = np.abs(signals)

        max_idx = np.argmax(snrs)
        best_jpa_flux = jpa_fluxs[max_idx]

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.plot(jpa_fluxs, snrs, label="signal difference")
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
        result: Optional[FluxResult] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/flux",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_fluxs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "JPA Flux value", "unit": "a.u.", "values": jpa_fluxs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FluxResult:
        signals, jpa_fluxs, _ = load_data(filepath, **kwargs)
        assert jpa_fluxs is not None
        assert len(jpa_fluxs.shape) == 1 and len(signals.shape) == 1
        assert jpa_fluxs.shape == signals.shape

        jpa_fluxs = jpa_fluxs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (jpa_fluxs, signals)

        return jpa_fluxs, signals
