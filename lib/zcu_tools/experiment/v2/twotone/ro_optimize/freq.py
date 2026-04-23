from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter1d
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

FreqResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class FreqModuleCfg(BaseModel):
    reset: Optional[ResetCfg] = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class FreqSweepCfg(BaseModel):
    freq: SweepCfg


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg


RawResult: TypeAlias = list[MomentTracker]


class FreqExp(AbsExperiment[FreqResult, FreqCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> FreqResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, FreqCfg],
            update_hook: Optional[Callable[[int, RawResult], None]],
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
                sweep=[
                    ("ge", 2),
                    ("freq", freq_sweep),
                ],
            )
            tracker = MomentTracker()
            prog.acquire(
                soc,
                progress=False,
                round_hook=lambda i, avg_d: update_hook(i, [tracker]),
                trackers=[tracker],
                **(acquire_kwargs or {}),
            )
            return [tracker]

        with LivePlot1D("Frequency (MHz)", "SNR") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    result_shape=(len(freqs),),
                    dtype=np.float64,
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(freqs, np.abs(ctx.root_data)),
            )

        # record the last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (freqs, signals)

        return freqs, signals  # freqs

    def analyze(
        self, result: Optional[FreqResult] = None, *, smooth: float = 1.0
    ) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, signals = result

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, smooth)

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

    def save(
        self,
        filepath: str,
        result: Optional[FreqResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, singals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": singals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        signals, freqs, _, cfg = load_data(filepath, return_cfg=True, **kwargs)
        assert freqs is not None
        assert len(freqs.shape) == 1 and len(signals.shape) == 1
        assert freqs.shape == signals.shape

        freqs = freqs * 1e-6  # Hz -> MHz

        freqs = freqs.astype(np.float64)
        signals = signals.astype(np.float64)

        self.last_cfg = FreqCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (freqs, signals)

        return freqs, signals
