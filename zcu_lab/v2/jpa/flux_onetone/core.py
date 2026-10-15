from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils import set_flux
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import (
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


class OneToneFluxModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[PulseReadoutCfg, FieldMeta(label="Readout")]


class OneToneFluxSweepCfg(ConfigBase):
    jpa_flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="JPA flux device value",
            description="Flux bounds in flux_unit coordinates.",
        ),
    ]
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]


class OneToneFluxCfg(ProgramV2Cfg, ExpCfgModel):
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux unit",
            description="Working unit for the selected flux driver; no conversion.",
        ),
    ]
    modules: OneToneFluxModuleCfg
    jpa_flux_dev: Annotated[
        str,
        FieldMeta(
            label="JPA flux device",
            description="Connected flux device driven by the survey.",
        ),
    ]
    sweep: OneToneFluxSweepCfg


class OneToneFluxExp(PersistableExperiment[OneToneFluxResult, OneToneFluxCfg]):
    # Axes are declared inner-first (ADR-0063): inner (fastest-varying) axis is
    # freqs (stored as Hz on disk, MHz in memory); outer axis is jpa flux (a.u.).
    # signals.shape == (len(fluxes), len(freqs)); non-square sweeps round-trip.
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Readout frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("fluxes", "JPA Flux value", "a.u."),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=OneToneFluxResult,
        cfg_type=OneToneFluxCfg,
        tag="jpa/flux_onetone",
        cfg_schema_version="3.0",
    )

    def run(self, cfg: OneToneFluxCfg, *, context: RunContext) -> OneToneFluxResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules
        jpa_flux_sweep = cfg.sweep.jpa_flux

        jpa_fluxs = sweep2array(jpa_flux_sweep, allow_array=True)
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.gen_ch},
            allow_array=True,
        )

        viewer = context.plots.liveplot_2d_with_line(
            "measurement",
            "JPA Flux value (a.u.)",
            "Readout frequency (MHz)",
            line_axis=1,
            num_lines=5,
        )
        signals_buffer = SignalBuffer(
            (len(jpa_fluxs), len(freqs)),
            on_update=lambda data: viewer.update(jpa_fluxs, freqs, np.abs(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for jpa_flux, step in sched.scan("JPA Flux value", jpa_fluxs.tolist()):
                set_flux(
                    context.devices[step.cfg.jpa_flux_dev],
                    jpa_flux,
                    unit=step.cfg.flux_unit,
                    stop=context.cancel_signal,
                    progress=False,
                )
                modules = step.cfg.modules
                modules.readout.set_param(
                    "freq", sweep2param("freq", step.cfg.sweep.freq)
                )
                _ = (
                    step.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        PulseReadout("readout", modules.readout),
                    )
                    .declare_sweep("freq", step.cfg.sweep.freq)
                    .build_and_acquire()
                )
        signals = signals_buffer.array

        return OneToneFluxResult(fluxes=jpa_fluxs, freqs=freqs, signals=signals)
