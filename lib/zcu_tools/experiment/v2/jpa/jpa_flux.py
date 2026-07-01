from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import Field
from scipy.ndimage import gaussian_filter1d

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, setup_devices
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)


@dataclass(frozen=True)
class FluxResult:
    fluxes: NDArray[np.float64]
    signals: NDArray[np.float64]
    cfg_snapshot: FluxCfg | None = None


class FluxModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class FluxSweepCfg(ConfigBase):
    jpa_flux: SweepCfg


class FluxCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FluxModuleCfg
    sweep: FluxSweepCfg
    skew_penalty: float = Field(default=0.0, ge=0.0)


class FluxExp(PersistableExperiment[FluxResult, FluxCfg]):
    # jpa_flux stored as-is (a.u.) on disk -> scale=IDENTITY; signals are float64
    AXES_SPEC = AxesSpec(
        axes=(Axis("fluxes", "JPA Flux value", "a.u.", scale=IDENTITY),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.float64),
        result_type=FluxResult,
        cfg_type=FluxCfg,
        tag="jpa/flux",
    )

    @record_result
    def run(self, soc, soccfg, cfg: FluxCfg) -> FluxResult:
        cfg = deepcopy(cfg)
        jpa_fluxs = sweep2array(cfg.sweep.jpa_flux, allow_array=True)

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, FluxCfg],
            update_hook: Callable[[int, list[MomentTracker]], None] | None,
        ) -> list[MomentTracker]:
            cfg = ctx.cfg
            setup_devices(cfg, progress=False)
            modules = cfg.modules

            assert update_hook is not None

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                    Readout("readout", modules.readout),
                ],
                sweep=[("ge", 2)],
            )
            tracker = MomentTracker()
            prog.acquire(
                soc,
                progress=False,
                round_hook=lambda i, _avg_d: update_hook(i, [tracker]),
                trackers=[tracker],
                stop_checkers=[ctx.is_stop],
            )
            return [tracker]

        with LivePlot1D("JPA Flux value (a.u.)", "Signal Difference") as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(jpa_fluxs),),
                    dtype=np.float64,
                    on_update=lambda data: viewer.update(jpa_fluxs, np.abs(data)),
                )
                for step in run.scan("JPA Flux value", jpa_fluxs.tolist()):
                    if step.cfg.dev is not None:
                        set_flux_in_dev_cfg(
                            step.cfg.dev,
                            step.value,
                            label="jpa_flux_dev",
                        )
                    signals_buffer[step].measure(
                        measure_fn,
                        raw2signal_fn=lambda raw: snr_as_signal(
                            raw,
                            ge_axis=0,
                            skew_penalty=run.cfg.skew_penalty,
                        ),
                        pbar_n=step.cfg.rounds,
                    )
                signals = signals_buffer.array

        return FluxResult(fluxes=jpa_fluxs, signals=signals, cfg_snapshot=cfg)

    @retrieve_result
    def analyze(self, result: FluxResult | None = None) -> tuple[float, Figure]:
        assert result is not None, "no result found"

        jpa_fluxs = result.fluxes
        signals = result.signals
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
