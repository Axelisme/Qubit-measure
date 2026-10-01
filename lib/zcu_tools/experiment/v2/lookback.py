from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
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
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import QickContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
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


@dataclass(frozen=True)
class LookbackAnalyzeOptions:
    ratio: float = 0.3
    smooth: float | None = None
    plot_fit: bool = True


@dataclass(frozen=True)
class LookbackAnalysis:
    predict_offset: float


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

    def run(self, config: LookbackCfg, *, context: QickContext) -> LookbackResult:
        run_cfg = deepcopy(config)
        if run_cfg.reps != 1:
            warnings.warn("reps is not 1 in config, this will be ignored.")
            run_cfg.reps = 1

        setup_devices(run_cfg, progress=True)
        viewer = context.plots.liveplot_1d("measurement", "Time (us)", "Amplitude")
        with Schedule(run_cfg) as sched:
            modules = sched.cfg.modules
            builder = sched.prog_builder(context.soc, context.soccfg).add(
                Reset("reset", cfg=modules.reset),
                Pulse("init_pulse", cfg=modules.init_pulse),
                Readout("readout", cfg=modules.readout),
            )
            program = builder.build()
            times = (
                program.get_time_axis(ro_index=0)
                + sched.cfg.modules.readout.ro_cfg.trig_offset
            )
            assert isinstance(times, np.ndarray)

            signals_buffer = SignalBuffer(
                (len(times),),
                on_update=lambda data: viewer.update(times, lookback_signal2real(data)),
            )
            sched.register_buffer(signals_buffer)
            _ = builder.run_program_decimated(program)
            return LookbackResult(times=times, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[LookbackCfg, LookbackResult],
        options: LookbackAnalyzeOptions,
        *,
        plots: Plots,
    ) -> LookbackAnalysis:
        result = source.result
        Ts = result.times
        signals = result.signals
        cfg = source.cfg
        ro_cfg = cfg.modules.readout.ro_cfg if cfg is not None else None

        if options.smooth is not None:
            signals = gaussian_filter1d(signals, options.smooth)
        y = np.abs(signals)

        # start from max point, find largest idx where y is smaller than ratio * max_y
        max_idx = np.argmax(y)
        candidate_mask = y[:max_idx] < options.ratio * y[max_idx]
        if not np.any(candidate_mask):
            offset = float(Ts[0])
        else:
            offset = float(Ts[np.nonzero(candidate_mask)[0][-1]])

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        # np.real/np.imag are used instead of .real/.imag because numpy 2.4
        # stubs narrow the overloaded descriptor in a way that confuses pyright
        # when the array dtype was widened by gaussian_filter1d.
        ax.plot(Ts, np.real(signals), label="I value")
        ax.plot(Ts, np.imag(signals), label="Q value")
        ax.plot(Ts, y, label="mag")
        if options.plot_fit:
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

        return LookbackAnalysis(predict_offset=offset)
