from __future__ import annotations

import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

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
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadoutCfg,
    Readout,
    Reset,
    ResetCfg,
)


@dataclass(frozen=True)
class LookbackResult:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: LookbackCfg | None = None


class LookbackModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    readout: PulseReadoutCfg


class LookbackCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LookbackModuleCfg


def lookback_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class LookbackExp(PersistableExperiment[LookbackResult, LookbackCfg]):
    # times stored in seconds on disk -> scale=US_TO_S (mem us)
    AXES_SPEC = AxesSpec(
        axes=(Axis("times", "Time", "s", US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=LookbackResult,
        cfg_type=LookbackCfg,
        tag="lookback",
    )

    @record_result
    def run(self, soc, soccfg, cfg: LookbackCfg) -> LookbackResult:
        orig_cfg = deepcopy(cfg)

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

        def measure_fn(ctx: TaskState, update_hook: Callable | None):
            return prog.acquire_decimated(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
            )

        with LivePlot1D("Time (us)", "Amplitude") as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(Ts),),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        Ts, lookback_signal2real(data)
                    ),
                )
                signals_buffer.measure(
                    measure_fn,
                    raw2signal_fn=lambda raw: raw[0].dot([1, 1j]),
                    pbar_n=run.cfg.rounds,
                )
                return LookbackResult(
                    times=Ts,
                    signals=signals_buffer.array,
                    cfg_snapshot=orig_cfg,
                )

    @retrieve_result
    def analyze(
        self,
        result: LookbackResult | None = None,
        *,
        ratio: float = 0.3,
        smooth: float | None = None,
        plot_fit: bool = True,
    ) -> tuple[float, Figure]:
        assert result is not None, "no result found"

        Ts = result.times
        signals = result.signals
        cfg = result.cfg_snapshot
        ro_cfg = cfg.modules.readout.ro_cfg if cfg is not None else None

        if smooth is not None:
            signals = gaussian_filter1d(signals, smooth)
        y = np.abs(signals)

        # start from max point, find largest idx where y is smaller than ratio * max_y
        max_idx = np.argmax(y)
        candidate_mask = y[:max_idx] < ratio * y[max_idx]
        if not np.any(candidate_mask):
            offset = float(Ts[0])
        else:
            offset = float(Ts[np.nonzero(candidate_mask)[0][-1]])

        fig, ax = plt.subplots(figsize=config.figsize)

        # np.real/np.imag are used instead of .real/.imag because numpy 2.4
        # stubs narrow the overloaded descriptor in a way that confuses pyright
        # when the array dtype was widened by gaussian_filter1d.
        ax.plot(Ts, np.real(signals), label="I value")
        ax.plot(Ts, np.imag(signals), label="Q value")
        ax.plot(Ts, y, label="mag")
        if plot_fit:
            ax.axvline(offset, color="r", linestyle="--", label="predict_offset")
        if ro_cfg is not None:
            trig_offset = float(ro_cfg.trig_offset)
            ro_length = float(ro_cfg.ro_length)
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
