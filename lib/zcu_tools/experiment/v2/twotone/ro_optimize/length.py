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
    US_TO_S,
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
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
)
from zcu_tools.utils.process import SmoothMethod, smooth_signal1d


@dataclass(frozen=True)
class LengthResult:
    lengths: NDArray[np.float64]
    signals: NDArray[np.float64]
    cfg_snapshot: LengthCfg | None = None


class LengthModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    qub_pulse: PulseCfg
    readout: PulseReadoutCfg


class LengthSweepCfg(ConfigBase):
    length: SweepCfg


class LengthCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LengthModuleCfg
    sweep: LengthSweepCfg
    skew_penalty: float = Field(default=0.0, ge=0.0)


class LengthExp(PersistableExperiment[LengthResult, LengthCfg]):
    # lengths stored in seconds on disk -> scale=US_TO_S (disk = memory * 1e-6)
    AXES_SPEC = AxesSpec(
        axes=(Axis("lengths", "Readout Length", "s", scale=US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.float64),
        result_type=LengthResult,
        cfg_type=LengthCfg,
        tag="twotone/ge/ro_optimize/length",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: LengthCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> LengthResult:
        original_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        readout_cfg = modules.readout
        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "ro_ch": readout_cfg.ro_cfg.ro_ch},
        )

        with LivePlot1D("Readout Length (us)", "SNR") as viewer:
            signals_buffer = SignalBuffer(
                (len(lengths),),
                dtype=np.float64,
                on_update=lambda data: viewer.update(lengths, np.abs(data)),
            )
            with Schedule(cfg, signals_buffer) as sched:
                for _, step in sched.scan("length", lengths.tolist()):
                    modules = step.cfg.modules
                    modules.readout.set_param("ro_length", step.value)
                    modules.readout.set_param("length", lengths.max() + 0.11)
                    tracker = MomentTracker()

                    _ = (
                        step.prog_builder(soc, soccfg)
                        .add(
                            Reset("reset", cfg=modules.reset),
                            Branch("ge", [], Pulse("qub_pulse", cfg=modules.qub_pulse)),
                            PulseReadout("readout", cfg=modules.readout),
                        )
                        .declare_sweep("ge", 2)
                        .build_and_acquire(
                            raw2signal_fn=lambda _raw: snr_as_signal(
                                [tracker],
                                ge_axis=1,
                                skew_penalty=step.cfg.skew_penalty,
                            ),
                            trackers=[tracker],
                            **(acquire_kwargs or {}),
                        )
                    )
                signals = signals_buffer.array

        return LengthResult(lengths, signals, cfg_snapshot=original_cfg)

    @retrieve_result
    def analyze(
        self,
        result: LengthResult | None = None,
        *,
        t0: float | None = None,
        smooth: float = 1.0,
        smooth_method: SmoothMethod = "wavelet",
        wavelet: str = "sym4",
        wavelet_level: int = 0,
    ) -> tuple[float, Figure]:
        assert result is not None, "no result found"

        lengths, signals = result.lengths, result.signals

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

        if t0 is None:
            max_id = np.argmax(snrs)
        else:
            max_id = np.argmax(snrs / np.sqrt(lengths + t0))

        max_length = float(lengths[max_id])
        max_snr = float(snrs[max_id])

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(lengths, snrs)
        ax.axvline(max_length, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Readout Length (us)")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return max_length, fig
