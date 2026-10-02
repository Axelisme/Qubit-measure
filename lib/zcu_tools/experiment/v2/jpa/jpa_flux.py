from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
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
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
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


@dataclass(frozen=True)
class FluxAnalysis:
    best_flux: float


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

    def run(self, cfg: FluxCfg, *, context: RunContext) -> FluxResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        jpa_fluxs = sweep2array(cfg.sweep.jpa_flux, allow_array=True)

        viewer = context.plots.liveplot_1d(
            "measurement", "JPA Flux value (a.u.)", "Signal Difference"
        )
        signals_buffer = SignalBuffer(
            (len(jpa_fluxs),),
            dtype=np.float64,
            on_update=lambda data: viewer.update(jpa_fluxs, np.abs(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for jpa_flux, step in sched.scan("JPA Flux value", jpa_fluxs.tolist()):
                if step.cfg.dev is not None:
                    set_flux_in_dev_cfg(
                        step.cfg.dev,
                        jpa_flux,
                        label="jpa_flux_dev",
                    )
                setup_devices(
                    step.cfg,
                    context.devices,
                    progress=False,
                    cancel_signal=context.cancel_signal.event,
                )
                modules = step.cfg.modules
                tracker = MomentTracker()
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("ge", 2)
                    .build_and_acquire(
                        raw2signal_fn=lambda raw, tracker=tracker, skew_penalty=step.cfg.skew_penalty: (
                            snr_as_signal(
                                [tracker],
                                ge_axis=1,
                                skew_penalty=skew_penalty,
                            )
                        ),
                        trackers=[tracker],
                    )
                )
            signals = signals_buffer.array

        return FluxResult(fluxes=jpa_fluxs, signals=signals)

    def analyze(
        self, source: RunRecord[FluxCfg, FluxResult], options: None, *, plots: Plots
    ) -> FluxAnalysis:
        del options
        result = source.result

        jpa_fluxs = result.fluxes
        signals = result.signals
        signals = gaussian_filter1d(signals, sigma=1)
        snrs = np.abs(signals)

        max_idx = np.argmax(snrs)
        best_jpa_flux = jpa_fluxs[max_idx]

        fig, ax = plots.subplots("fit", figsize=config.figsize)
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

        return FluxAnalysis(best_flux=float(best_jpa_flux))
