from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
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
    cfg_snapshot: SA_FreqCfg | None = None


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

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: SA_FreqCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> SA_FreqResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
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

        with LivePlot1D("SA Frequency (MHz)", "Amplitude") as viewer:
            signals_buffer = SignalBuffer(
                (len(freqs),),
                on_update=lambda data: viewer.update(freqs, safreq_signal2real(data)),
            )
            with Schedule(cfg, signals_buffer) as sched:
                cfg = sched.cfg
                modules = cfg.modules

                freq_sweep = cfg.sweep.freq
                modules.readout.set_param("ro_freq", sweep2param("ro_freq", freq_sweep))

                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add(PulseReadout("readout", modules.readout))
                    .declare_sweep("ro_freq", freq_sweep)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )

            return SA_FreqResult(
                freqs=freqs, signals=signals_buffer.array, cfg_snapshot=orig_cfg
            )

    @retrieve_result
    def analyze(self, result: SA_FreqResult | None = None) -> Figure:
        assert result is not None, "no result found"

        freqs = result.freqs
        signals = result.signals

        fig, ax = plt.subplots(figsize=config.figsize)

        amps = safreq_signal2real(signals)

        ax.plot(freqs, amps, label="signal", marker="o", markersize=3)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return fig
