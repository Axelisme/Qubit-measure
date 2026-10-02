from __future__ import annotations

from collections.abc import Mapping
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
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils import (
    set_flux_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)


@dataclass(frozen=True)
class FluxDepResult:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FluxDepModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    readout: PulseReadoutCfg


class FluxDepSweepCfg(ConfigBase):
    freq: SweepCfg
    flux: SweepCfg


class FluxDepCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FluxDepModuleCfg
    # Field(...) makes dev required in this subclass, overriding the Optional
    # default from ExpCfgModel — intentional Pydantic pattern (type: ignore[override]).
    dev: Mapping[str, DeviceInfo] = Field(...)  # type: ignore[override]
    sweep: FluxDepSweepCfg


class FluxDepExp(PersistableExperiment[FluxDepResult, FluxDepCfg]):
    # inner axis (fastest-varying) = freqs (MHz in memory, Hz on disk);
    # outer axis = flux device values (a.u.).
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("values", "Flux device value", "a.u."),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FluxDepResult,
        cfg_type=FluxDepCfg,
        tag="onetone/flux_dep",
    )

    def run(
        self,
        config: FluxDepCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FluxDepResult:
        """Run one sweep using this operation's instruments and named plots."""
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(config)
        modules = cfg.modules
        freq_sweep = cfg.sweep.freq
        flux_sweep = cfg.sweep.flux

        dev_values = sweep2array(flux_sweep, allow_array=True)
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": modules.readout.pulse_cfg.ch,
                "ro_ch": modules.readout.ro_cfg.ro_ch,
            },
        )

        set_flux_in_dev_cfg(cfg.dev, dev_values[0])
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )

        viewer = context.plots.liveplot_2d_with_line(
            "measurement",
            "Flux device value",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=10,
            uniform=False,
        )
        signals_buffer = SignalBuffer(
            (len(dev_values), len(freqs)),
            on_update=lambda data: viewer.update(
                dev_values,
                freqs,
                fluxdep_signal2real(data),
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for _, step in sched.scan("flux", dev_values.tolist()):
                cfg = step.cfg
                set_flux_in_dev_cfg(cfg.dev, step.value)
                setup_devices(
                    cfg,
                    context.devices,
                    progress=False,
                    cancel_signal=context.cancel_signal,
                )
                modules = cfg.modules

                freq_sweep = cfg.sweep.freq
                modules.readout.set_param("freq", sweep2param("freq", freq_sweep))

                _ = (
                    step.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add(PulseReadout("readout", modules.readout))
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )

        signals_buffer.trigger_update(flush=True)
        sched.stop.raise_if_error()
        return FluxDepResult(
            values=dev_values,
            freqs=freqs,
            signals=signals_buffer.array,
        )
