from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias

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
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
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


RawResult: TypeAlias = list[MomentTracker]


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

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, FreqCfg],
            update_hook: Callable[[int, RawResult], None] | None,
        ) -> RawResult:
            cfg = ctx.cfg
            modules = cfg.modules

            assert update_hook is not None, "update_hook is required for measure_fn"

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.readout.set_param("freq", freq_param)

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Branch("ge", [], Pulse("qub_pulse", modules.qub_pulse)),
                    Readout("readout", modules.readout),
                ],
                sweep=[("ge", 2), ("freq", freq_sweep)],
            )
            tracker = MomentTracker()
            prog.acquire(
                soc,
                progress=False,
                round_hook=lambda i, avg_d: update_hook(i, [tracker]),
                trackers=[tracker],
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )
            return [tracker]

        with LivePlot1D("Frequency (MHz)", "SNR") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=1),
                    result_shape=(len(freqs),),
                    dtype=np.float64,
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(freqs, np.abs(ctx.root_data)),
            )

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
