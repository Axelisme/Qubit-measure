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
class PowerResult:
    gains: NDArray[np.float64]
    signals: NDArray[np.float64]
    cfg_snapshot: PowerCfg | None = None


class PowerModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    qub_pulse: PulseCfg
    readout: PulseReadoutCfg


class PowerSweepCfg(ConfigBase):
    gain: SweepCfg


class PowerCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerModuleCfg
    sweep: PowerSweepCfg
    skew_penalty: float = Field(default=0.0, ge=0.0)


class PowerExp(PersistableExperiment[PowerResult, PowerCfg]):
    # gains stored as-is on disk -> scale=IDENTITY (default)
    AXES_SPEC = AxesSpec(
        axes=(Axis("gains", "Probe Power", "a.u."),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        tag="twotone/ge/ro_optimize/gain",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: PowerCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> PowerResult:
        original_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.readout.pulse_cfg.ch},
        )

        with LivePlot1D("Readout Power", "SNR") as viewer:
            signals_buffer = SignalBuffer(
                (len(gains),),
                dtype=np.float64,
                on_update=lambda data: viewer.update(gains, np.abs(data)),
            )
            with Schedule(cfg, signals_buffer) as sched:
                modules = sched.cfg.modules
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

        return PowerResult(gains, signals, cfg_snapshot=original_cfg)

    @retrieve_result
    def analyze(
        self,
        result: PowerResult | None = None,
        penalty_ratio: float = 0.0,
        *,
        smooth: float = 1.0,
        smooth_method: SmoothMethod = "wavelet",
        wavelet: str = "sym4",
        wavelet_level: int = 0,
    ) -> tuple[float, Figure]:
        assert result is not None, "no result found"

        powers, snrs = result.gains, result.signals
        snrs = np.abs(snrs)

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
        penaltized_snrs = snrs * np.exp(-powers * penalty_ratio)

        max_id = np.argmax(penaltized_snrs)
        max_power = float(powers[max_id])
        max_snr = float(snrs[max_id])

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(powers, snrs)
        ax.axvline(max_power, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Readout Power")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return max_power, fig
