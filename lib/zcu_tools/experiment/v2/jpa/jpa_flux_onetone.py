from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import (
    set_flux_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)


@dataclass(frozen=True)
class OneToneFluxResult:
    fluxes: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: OneToneFluxCfg | None = None


class OneToneFluxModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    readout: PulseReadoutCfg


class OneToneFluxSweepCfg(ConfigBase):
    jpa_flux: SweepCfg
    freq: SweepCfg


class OneToneFluxCfg(ProgramV2Cfg, ExpCfgModel):
    modules: OneToneFluxModuleCfg
    # Field(...) makes dev required in this subclass, overriding the Optional
    # default from ExpCfgModel — intentional Pydantic pattern (type: ignore[override]).
    dev: Mapping[str, DeviceInfo] = Field(...)  # type: ignore[override]
    sweep: OneToneFluxSweepCfg


class OneToneFluxExp(PersistableExperiment[OneToneFluxResult, OneToneFluxCfg]):
    # inner axis (fastest-varying) = freqs (MHz on disk); outer = jpa flux (a.u.)
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("fluxes", "JPA Flux value", "a.u."),
            Axis("freqs", "Readout frequency", "Hz", scale=MHZ_TO_HZ),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=OneToneFluxResult,
        cfg_type=OneToneFluxCfg,
        tag="jpa/flux_onetone",
    )

    @record_result
    def run(self, soc, soccfg, cfg: OneToneFluxCfg) -> OneToneFluxResult:
        cfg = deepcopy(cfg)
        modules = cfg.modules
        jpa_flux_sweep = cfg.sweep.jpa_flux

        jpa_fluxs = sweep2array(jpa_flux_sweep, allow_array=True)
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.pulse_cfg.ch},
            allow_array=True,
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, OneToneFluxCfg],
            update_hook: Callable[[int, list[NDArray[np.float64]]], None] | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            setup_devices(cfg, progress=False)
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.readout.set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    PulseReadout("readout", modules.readout),
                ],
                sweep=[("freq", freq_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
            )

        with LivePlot2DwithLine(
            "JPA Flux value (a.u.)",
            "Readout frequency (MHz)",
            line_axis=1,
            num_lines=5,
        ) as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(jpa_fluxs), len(freqs)),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        jpa_fluxs, freqs, np.abs(data)
                    ),
                )
                for step in run.scan("JPA Flux value", jpa_fluxs.tolist()):
                    set_flux_in_dev_cfg(
                        step.cfg.dev,
                        step.value,
                        label="jpa_flux_dev",
                    )
                    signals_buffer[step].measure(measure_fn, pbar_n=step.cfg.rounds)
                signals = signals_buffer.array

        return OneToneFluxResult(
            fluxes=jpa_fluxs, freqs=freqs, signals=signals, cfg_snapshot=cfg
        )

    @retrieve_result
    def analyze(self, result: OneToneFluxResult | None = None) -> None:
        assert result is not None, "no result found"
        raise NotImplementedError("analysis not implemented yet")
