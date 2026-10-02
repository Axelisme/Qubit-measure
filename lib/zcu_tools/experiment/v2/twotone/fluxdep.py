from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import minus_background


@dataclass(frozen=True)
class FreqFluxResult:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def freqflux_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class FreqFluxSweepCfg(ConfigBase):
    flux: SweepCfg
    freq: SweepCfg


class FreqFluxModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class FreqFluxCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqFluxModuleCfg
    dev: Mapping[str, DeviceInfo]  # type: ignore[dict-item]
    sweep: FreqFluxSweepCfg
    fail_retry: int = 0


class FreqFluxExp(PersistableExperiment[FreqFluxResult, FreqFluxCfg]):
    # inner freqs stores MHz on disk -> scale=MHZ_TO_HZ; outer values raw -> IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("values", "Flux device value", "a.u.", scale=IDENTITY),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqFluxResult,
        cfg_type=FreqFluxCfg,
        tag="twotone/flux_dep/freq",
    )

    def run(
        self,
        cfg: FreqFluxCfg,
        *,
        context: RunContext,
    ) -> FreqFluxResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules

        value_sweep = cfg.sweep.flux
        freq_sweep = cfg.sweep.freq

        dev_values = sweep2array(value_sweep, allow_array=True)
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch}
        )

        viewer = context.plots.liveplot_2d_with_line(
            "measurement",
            "Flux device value",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=2,
        )
        signals_buffer = SignalBuffer(
            (len(dev_values), len(freqs)),
            on_update=lambda data: viewer.update(
                dev_values, freqs, freqflux_signal2real(data)
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

                # Frequency is swept by FPGA (hard sweep)
                freq_sweep = cfg.sweep.freq
                modules.qub_pulse.set_param("freq", sweep2param("freq", freq_sweep))

                _ = (
                    step.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add_pulse("init_pulse", modules.init_pulse)
                    .add_pulse("qubit_pulse", modules.qub_pulse)
                    .add_readout("readout", modules.readout)
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire(retry=cfg.fail_retry)
                )

        return FreqFluxResult(
            values=dev_values,
            freqs=freqs,
            signals=signals_buffer.array,
        )
