from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import Field

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
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    Branch,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadoutCfg,
    Readout,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import SmoothMethod, smooth_signal1d


@dataclass(frozen=True)
class FreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.float64]
    cfg_snapshot: FreqCfg | None = None


class FreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    qub_pulse: PulseCfg
    readout: PulseReadoutCfg


class FreqSweepCfg(ConfigBase):
    freq: SweepCfg


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg
    skew_penalty: float = Field(default=0.0, ge=0.0)


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    # freq stores Hz on disk -> scale=MHZ_TO_HZ (disk = memory * 1e6)
    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="twotone/ge/ro_optimize/freq",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqResult:
        original_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.pulse_cfg.ch},
        )

        with LivePlot1D("Frequency (MHz)", "SNR") as viewer:
            signals_buffer = SignalBuffer(
                (len(freqs),),
                dtype=np.float64,
                on_update=lambda data: viewer.update(freqs, np.abs(data)),
            )
            with Schedule(cfg, signals_buffer) as sched:
                modules = sched.cfg.modules
                freq_sweep = sched.cfg.sweep.freq
                freq_param = sweep2param("freq", freq_sweep)
                modules.readout.set_param("freq", freq_param)
                tracker = MomentTracker()

                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", cfg=modules.reset),
                        Branch("ge", [], Pulse("qub_pulse", cfg=modules.qub_pulse)),
                        Readout("readout", cfg=modules.readout),
                    )
                    .declare_sweep("ge", 2)
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire(
                        raw2signal_fn=lambda _raw: snr_as_signal(
                            [tracker],
                            ge_axis=1,
                            skew_penalty=sched.cfg.skew_penalty,
                        ),
                        trackers=[tracker],
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array

        return FreqResult(freqs, signals, cfg_snapshot=original_cfg)

    @retrieve_result
    def analyze(
        self,
        result: FreqResult | None = None,
        *,
        smooth: float = 1.0,
        smooth_method: SmoothMethod = "wavelet",
        wavelet: str = "sym4",
        wavelet_level: int = 0,
    ) -> tuple[float, Figure]:
        assert result is not None, "no result found"

        freqs, signals = result.freqs, result.signals

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = smooth_signal1d(
            snrs,
            method=smooth_method,
            sigma=smooth,
            axis=0,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
        )

        max_id = np.argmax(snrs)
        max_freq = float(freqs[max_id])
        max_snr = float(snrs[max_id])

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(freqs, snrs)
        ax.axvline(max_freq, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return max_freq, fig
