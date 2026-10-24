from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, Pulse, PulseReadout, Readout, Reset

from zcu_lab.v2.support import live_line


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


class LookbackCfg(ConfigBase):
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="us", description="Delay before the first shot."
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Delay after each shot.",
        ),
    ] = 1.0
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init pulse")] = None
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]


def lookback_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class LookbackExp(Experiment[LookbackResult, LookbackCfg]):
    Options: ClassVar[type[LookbackAnalyzeOptions]] = LookbackAnalyzeOptions

    # times stored in seconds on disk -> scale=1e-6 (mem us)
    AXES_SPEC: ClassVar[AxesSpec[LookbackResult, LookbackCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("times", "Time", "s", 1e-6),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=LookbackResult,
        cfg_type=LookbackCfg,
        cfg_schema_version="3.0",
        tag="lookback",
    )

    def run(self, config: LookbackCfg, *, context: RunContext) -> LookbackResult:
        """Measure the readout arrival and envelope as a decimated ADC trace.

        Apply optional reset, optional initialization pulse, then readout.
        Return sample times in us, including trig_offset, and complex ADC
        samples rather than integrated IQ. The program uses reps=1 and config's
        software rounds without changing config.
        """
        # Decimated acquisition uses one repetition per round, not an IQ-shot loop.
        run_cfg = deepcopy(config)

        viewer = live_line(context.plots, "measurement", "Time (us)", "Amplitude")
        # Compile reset→optional preparation→readout before acquisition so an
        # interrupted trace still has its fixed sample-time axis and buffer.
        modules = deepcopy(run_cfg)
        program = ModularProgramV2(
            context.soccfg,
            reps=1,
            initial_delay=run_cfg.initial_delay,
            relax_delay=run_cfg.relax_delay,
            modules=[
                Reset("reset", cfg=modules.reset),
                Pulse("init_pulse", cfg=modules.init_pulse),
                Readout("readout", cfg=modules.readout),
            ],
        )
        times = program.get_time_axis(ro_index=0) + run_cfg.readout.trig_offset
        assert isinstance(times, np.ndarray)

        # Acquire the time-resolved trace for arrival-time analysis.
        signals_buffer = SignalBuffer(
            (len(times),),
            on_update=lambda data: viewer.update(times, lookback_signal2real(data)),
        )
        with Schedule(run_cfg, signals_buffer, scope=context.scope) as sched:
            _ = sched.prog_builder(
                context.soc,
                context.soccfg,
                reps=1,
                initial_delay=sched.cfg.initial_delay,
                relax_delay=sched.cfg.relax_delay,
            ).run_program_decimated(program, rounds=run_cfg.rounds)
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
        readout_cfg = cfg.readout if cfg is not None else None

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
