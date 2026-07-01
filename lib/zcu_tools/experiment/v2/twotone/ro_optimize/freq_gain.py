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
    IDENTITY,
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
from zcu_tools.liveplot import LivePlot2D
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
from zcu_tools.utils.process import SmoothMethod, smooth_signal_nd


@dataclass(frozen=True)
class FreqGainResult:
    freqs: NDArray[np.float64]
    gains: NDArray[np.float64]
    signals: NDArray[np.float64]
    cfg_snapshot: FreqGainCfg | None = None


class FreqGainModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    qub_pulse: PulseCfg
    readout: PulseReadoutCfg


class FreqGainSweepCfg(ConfigBase):
    freq: SweepCfg
    gain: SweepCfg


class FreqGainCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqGainModuleCfg
    sweep: FreqGainSweepCfg
    skew_penalty: float = Field(default=0.0, ge=0.0)


class FreqGainExp(PersistableExperiment[FreqGainResult, FreqGainCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("gains", "Gain", "a.u.", scale=IDENTITY),
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.float64),
        result_type=FreqGainResult,
        cfg_type=FreqGainCfg,
        tag="twotone/ge/ro_optimize/freq",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqGainCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqGainResult:
        original_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.pulse_cfg.ch},
        )
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.readout.pulse_cfg.ch},
        )

        with LivePlot2D("Frequency (MHz)", "Gain (a.u.)") as viewer:
            signals_buffer = SignalBuffer(
                (len(freqs), len(gains)),
                dtype=np.float64,
                on_update=lambda data: viewer.update(freqs, gains, np.abs(data)),
            )
            with Schedule(cfg, signals_buffer) as sched:
                modules = sched.cfg.modules
                freq_sweep = sched.cfg.sweep.freq
                freq_param = sweep2param("freq", freq_sweep)
                modules.readout.set_param("freq", freq_param)

                gain_sweep = sched.cfg.sweep.gain
                gain_param = sweep2param("gain", gain_sweep)
                modules.readout.set_param("gain", gain_param)
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
                    .declare_sweep("gain", gain_sweep)
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

        return FreqGainResult(freqs, gains, signals, cfg_snapshot=original_cfg)

    @retrieve_result
    def analyze(
        self,
        result: FreqGainResult | None = None,
        *,
        smooth: float = 1.0,
        smooth_method: SmoothMethod = "wavelet",
        wavelet: str = "sym4",
        wavelet_level: int = 0,
    ) -> tuple[float, float, Figure]:
        assert result is not None, "no result found"

        freqs, gains, signals = result.freqs, result.gains, result.signals

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = smooth_signal_nd(
            snrs,
            method=smooth_method,
            sigma=smooth,
            axes=(0, 1),
            wavelet=wavelet,
            wavelet_level=wavelet_level,
        )

        max_freq_id, max_gain_id = np.unravel_index(np.argmax(snrs), snrs.shape)
        max_freq = float(freqs[max_freq_id])
        max_gain = float(gains[max_gain_id])
        max_snr = float(snrs[max_freq_id, max_gain_id])

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.imshow(
            snrs.T,
            extent=(freqs[0], freqs[-1], gains[0], gains[-1]),
            aspect="auto",
            origin="lower",
            interpolation="none",
        )
        ax.scatter(max_freq, max_gain, color="r", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Gain (a.u.)")
        ax.legend()

        return max_freq, max_gain, fig
