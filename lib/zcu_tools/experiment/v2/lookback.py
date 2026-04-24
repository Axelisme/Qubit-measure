from __future__ import annotations

import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter1d
from typing_extensions import Callable, Optional, TypeAlias

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    DirectReadoutCfg,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadoutCfg,
    Readout,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data

LookbackResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


class LookbackModuleCfg(BaseModel):
    reset: Optional[ResetCfg] = None
    init_pulse: Optional[PulseCfg] = None
    readout: PulseReadoutCfg


class LookbackCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LookbackModuleCfg


def lookback_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class LookbackExp(AbsExperiment[LookbackResult, LookbackCfg]):
    def run(self, soc, soccfg, cfg: LookbackCfg) -> LookbackResult:
        setup_devices(cfg, progress=True)

        if cfg.reps != 1:
            warnings.warn("reps is not 1 in config, this will be ignored.")
            cfg.reps = 1

        modules = cfg.modules

        prog = ModularProgramV2(
            soccfg,
            cfg,
            [
                Reset("reset", cfg=modules.reset),
                Pulse("init_pulse", cfg=modules.init_pulse, tag="init_pulse"),
                Readout("readout", cfg=modules.readout),
            ],
        )
        Ts = prog.get_time_axis(ro_index=0) + cfg.modules.readout.ro_cfg.trig_offset
        assert isinstance(Ts, np.ndarray)

        def measure_fn(ctx: TaskState, update_hook: Optional[Callable]):
            return prog.acquire_decimated(soc, progress=False, round_hook=update_hook)

        with LivePlot1D("Time (us)", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw[0].dot([1, 1j]),
                    result_shape=(len(Ts),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    Ts, lookback_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (Ts, signals)

        return Ts, signals

    def analyze(
        self,
        result: Optional[LookbackResult] = None,
        *,
        ratio: float = 0.3,
        smooth: Optional[float] = None,
        ro_cfg: Optional[DirectReadoutCfg] = None,
        plot_fit: bool = True,
    ) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result

        if smooth is not None:
            signals = gaussian_filter1d(signals, smooth)
        y = np.abs(signals)

        # start from max point, find largest idx where y is smaller than ratio * max_y
        max_idx = np.argmax(y)
        candidate_mask = y[:max_idx] < ratio * y[max_idx]
        if not np.any(candidate_mask):
            offset = Ts[0]
        else:
            offset = Ts[np.nonzero(candidate_mask)[0][-1]]

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(Ts, signals.real, label="I value")
        ax.plot(Ts, signals.imag, label="Q value")
        ax.plot(Ts, y, label="mag")
        if plot_fit:
            ax.axvline(offset, color="r", linestyle="--", label="predict_offset")
        if ro_cfg is not None:
            trig_offset = ro_cfg.trig_offset
            ro_length = ro_cfg.ro_length
            ax.axvline(trig_offset, color="g", linestyle="--", label="ro start")
            ax.axvline(
                trig_offset + ro_length, color="g", linestyle="--", label="ro end"
            )
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("a.u.")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()

        return offset, fig

    def save(
        self,
        filepath: str,
        result: Optional[LookbackResult] = None,
        comment: Optional[str] = None,
        tag: str = "lookback",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> LookbackResult:
        signals, Ts, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert Ts is not None
        assert len(Ts.shape) == 1 and len(signals.shape) == 1
        assert Ts.shape == signals.shape

        Ts = Ts * 1e6  # s -> us

        Ts = Ts.astype(np.float64)
        signals = signals.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.last_cfg = LookbackCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (Ts, signals)

        return Ts, signals
