from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)


@dataclass(frozen=True)
class SA_FreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class SA_FreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    readout: PulseReadoutCfg


class SA_FreqSweepCfg(ConfigBase):
    freq: SweepCfg


class SA_FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: SA_FreqModuleCfg
    sweep: SA_FreqSweepCfg


def safreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class SA_FreqExp(PersistableExperiment[SA_FreqResult, SA_FreqCfg]):
    # freqs stored as Hz on disk -> scale=MHZ_TO_HZ; signals are complex.
    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=SA_FreqResult,
        cfg_type=SA_FreqCfg,
        tag="onetone/sa_freq",
    )

    def run(self, config: SA_FreqCfg, *, context: RunContext) -> SA_FreqResult:
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        modules = cfg.modules

        # Predicted frequency points (before mapping to ADC domain)
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": modules.readout.pulse_cfg.ch,
                "ro_ch": modules.readout.ro_cfg.ro_ch,
            },
        )

        viewer = context.plots.liveplot_1d(
            "measurement", "SA Frequency (MHz)", "Amplitude"
        )
        signals_buffer = SignalBuffer(
            (len(freqs),),
            on_update=lambda data: viewer.update(freqs, safreq_signal2real(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            cfg = sched.cfg
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            modules.readout.set_param("ro_freq", sweep2param("ro_freq", freq_sweep))

            _ = (
                sched.prog_builder(soc, soccfg)
                .add_reset("reset", modules.reset)
                .add(PulseReadout("readout", modules.readout))
                .declare_sweep("ro_freq", freq_sweep)
                .build_and_acquire()
            )

        return SA_FreqResult(freqs=freqs, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[SA_FreqCfg, SA_FreqResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        del options
        freqs = source.result.freqs
        signals = source.result.signals

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        amps = safreq_signal2real(signals)

        ax.plot(freqs, amps, label="signal", marker="o", markersize=3)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
