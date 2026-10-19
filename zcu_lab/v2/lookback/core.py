from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
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


@dataclass(frozen=True)
class LookbackAnalyzeOptions:
    ratio: float = 0.3
    smooth: float | None = None
    plot_fit: bool = True


@dataclass(frozen=True)
class LookbackAnalysis:
    predict_offset: float


class LookbackModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[PulseCfg | None, FieldMeta(label="Init pulse")] = None
    readout: Annotated[PulseReadoutCfg, FieldMeta(label="Readout")]


class LookbackCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LookbackModuleCfg


def lookback_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class LookbackExp(PersistableExperiment[LookbackResult, LookbackCfg]):
    Options: ClassVar[type[LookbackAnalyzeOptions]] = LookbackAnalyzeOptions

    # times stored in seconds on disk -> scale=US_TO_S (mem us)
    AXES_SPEC = AxesSpec(
        axes=(Axis("times", "Time", "s", US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=LookbackResult,
        cfg_type=LookbackCfg,
        cfg_schema_version="2.0",
        tag="lookback",
    )

    def run(self, config: LookbackCfg, *, context: RunContext) -> LookbackResult:
        """Measure the readout arrival and envelope as a decimated ADC trace.

        Apply optional reset, optional initialization pulse, then readout.
        Return sample times in us, including trig_offset, and complex ADC
        samples rather than integrated IQ. The acquisition copy forces reps=1
        (warning otherwise) without changing config; rounds follow that copy.
        """
        # Decimated acquisition uses one repetition per round, not an IQ-shot loop.
        run_cfg = deepcopy(config)
        if run_cfg.reps != 1:
            warnings.warn(
                "reps is not 1 in config, this will be ignored.", stacklevel=1
            )
            run_cfg.reps = 1

        viewer = context.plots.liveplot_1d("measurement", "Time (us)", "Amplitude")
        # Compile reset→optional preparation→readout before acquisition so an
        # interrupted trace still has its fixed sample-time axis and buffer.
        modules = deepcopy(run_cfg.modules)
        program = ModularProgramV2(
            context.soccfg,
            deepcopy(run_cfg),
            modules=[
                Reset("reset", cfg=modules.reset),
                Pulse("init_pulse", cfg=modules.init_pulse),
                Readout("readout", cfg=modules.readout),
            ],
        )
        times = program.get_time_axis(ro_index=0) + run_cfg.modules.readout.trig_offset
        assert isinstance(times, np.ndarray)

        # Acquire the time-resolved trace for arrival-time analysis.
        signals_buffer = SignalBuffer(
            (len(times),),
            on_update=lambda data: viewer.update(times, lookback_signal2real(data)),
        )
        with Schedule(run_cfg, signals_buffer, stop=context.cancel_signal) as sched:
            _ = sched.prog_builder(context.soc, context.soccfg).run_program_decimated(
                program
            )
        return LookbackResult(times=times, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[LookbackCfg, LookbackResult],
        options: LookbackAnalyzeOptions,
        *,
        plots: Plots,
    ) -> LookbackAnalysis:
        """Estimate arrival time in us from the rising readout magnitude.

        Optionally smooth complex samples with a Gaussian width in samples.
        Find the last pre-peak sample below ratio times the peak magnitude;
        fall back to the first sample time if no such sample exists. Return
        that time as predict_offset, not a fitted resonator response. Saved cfg
        is optional and only supplies the readout-window markers on the plot.
        """
        result = source.result
        Ts = result.times
        signals = result.signals
        cfg = source.cfg
        readout_cfg = cfg.modules.readout if cfg is not None else None

        if options.smooth is not None:
            signals = gaussian_filter1d(signals, options.smooth)
        y = np.abs(signals)

        # Work backward from the peak to locate the last below-threshold sample.
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
        if readout_cfg is not None:
            trig_offset = float(readout_cfg.trig_offset)
            ro_length = float(readout_cfg.ro_length)
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
