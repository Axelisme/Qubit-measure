from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from scipy.ndimage import gaussian_filter1d
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, PulseReadout, Readout, Reset, TimelineBuilder

from zcu_lab.v2.support import live_line

_TIME_LABEL = "Time from readout start (µs)"
_MAGNITUDE_LABEL = "|Mean IQ| (a.u.)"


@dataclass(frozen=True)
class LookbackResult:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]


@pydantic_dataclass(frozen=True)
class LookbackAnalyzeOptions:
    ratio: float = 0.3
    smooth: float | None = None
    plot_fit: bool = True


@dataclass(frozen=True)
class LookbackAnalysis:
    predict_offset: float


def lookback_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return the real-valued measurement view of complex IQ without changing the input."""
    return np.abs(signals)


class LookbackCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init pulse")] = None
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Select the pulse and ADC window used to record the readout arrival.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Averaging rounds",
            unit="count",
            description="Enter the number of one-shot ADC traces to average.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Inter-shot wait",
            unit="µs",
            decimals=3,
            description="Enter the wait after each shot (µs) before repeating the readout.",
        ),
    ]
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="µs", description="Delay before the first shot."
        ),
    ] = 1.0


class LookbackExp(Experiment[LookbackResult, LookbackCfg]):
    Options: ClassVar[type[LookbackAnalyzeOptions]] = LookbackAnalyzeOptions
    # A decimated trace uses one repetition in each averaging round.
    REPS: ClassVar[int] = 1

    # times stored in seconds on disk -> scale=1e-6 (mem us)
    AXES_SPEC: ClassVar[AxesSpec[LookbackResult, LookbackCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("times", label="Time", unit="s", scale=1e-6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
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
        # 1. Decimated acquisition uses one repetition per round, not an IQ-shot loop.
        run_cfg = deepcopy(config)

        title = f"Lookback · ADC {run_cfg.readout.ro_ch} · Readout magnitude"
        fig, ax = context.plots.subplots(num="measurement", layout="tight")
        ax.set(
            title=f"{title}\nWaiting for first shot",
            xlabel=_TIME_LABEL,
            ylabel=_MAGNITUDE_LABEL,
        )
        line = live_line(ax)
        fig.show()
        # 2. Fix the reset/preparation/readout sample axis before a possible stop.
        builder = TimelineBuilder(
            context.soccfg,
            reps=self.REPS,
            initial_delay=run_cfg.initial_delay,
            relax_delay=run_cfg.relax_delay,
        )
        if run_cfg.reset is not None:
            builder.emit(Reset("reset", run_cfg.reset))
        builder.emit(Pulse("init_pulse", run_cfg.init_pulse))
        builder.emit(Readout("readout", run_cfg.readout))
        program = builder.build()
        times = program.get_time_axis(ro_index=0) + run_cfg.readout.trig_offset
        assert isinstance(times, np.ndarray)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Draw acquired complex IQ on the measurement axes; return changed artists.

            Read the scan-shaped data without changing it. Update the live
            artists over the prepared coordinates; plot errors propagate.
            """
            title_artist = ax.set_title(
                f"{title}\nRunning average · full window per shot"
            )
            return (*line.set(times, lookback_signal2real(data)), title_artist)

        @context.plots.on_plot_thread
        def show_acquired_average(_: None) -> Iterable[Artist]:
            return (ax.set_title(f"{title}\nAcquired average"),)

        # 3. Acquire the time-resolved trace for arrival-time analysis.
        signals_buffer = SignalBuffer(
            (len(times),),
            on_update=update,
        )
        with Schedule(signals_buffer, scope=context.scope) as sched:
            sched.run_acquire_decimated(program, context.soc, rounds=run_cfg.rounds)

        if np.any(np.isfinite(signals_buffer.array)):
            show_acquired_average(None)
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
        that time as predict_offset, not a fitted resonator response. Display
        the trace and the estimated arrival; plot_fit controls decision markers.
        Saved cfg is optional and identifies the ADC in the titles when present.
        """
        # 1. Optionally smooth complex ADC samples before locating readout arrival.
        result = source.result
        Ts = result.times
        signals = result.signals
        filtered = (
            signals
            if options.smooth is None
            else gaussian_filter1d(signals, options.smooth)
        )
        magnitude = np.abs(filtered)

        # 2. Work backward from the peak to locate the last below-threshold sample.
        max_idx = np.argmax(magnitude)
        threshold = options.ratio * magnitude[max_idx]
        candidates = np.flatnonzero(magnitude[:max_idx] < threshold)
        offset_idx = int(candidates[-1]) if candidates.size else 0
        offset = float(Ts[offset_idx])

        adc = f"ADC {source.cfg.readout.ro_ch}" if source.cfg is not None else "ADC"
        _, ax = plots.subplots(num="fit", figsize=config.figsize, layout="tight")
        ax.set(
            title=f"Lookback · {adc} · Readout magnitude",
            xlabel=_TIME_LABEL,
            ylabel=_MAGNITUDE_LABEL,
        )
        ax.plot(
            Ts,
            lookback_signal2real(signals),
            "o-",
            color="C0",
            ms=2,
            label="Acquired average",
        )
        if options.smooth is not None:
            ax.plot(
                Ts, magnitude, "--", color="C1", label="Gaussian-smoothed magnitude"
            )
        if options.plot_fit:
            ax.axhline(
                threshold, color="0.35", linestyle=":", label="Arrival threshold"
            )
            label = (
                "Estimated readout arrival"
                if candidates.size
                else "Window start fallback"
            )
            ax.plot(
                offset,
                magnitude[offset_idx],
                "D",
                color="C3",
                zorder=4,
                label=f"{label} = {offset:.4f} µs",
            )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)

        # The signed components separate phase changes from changes in magnitude.
        _, iq_ax = plots.subplots(
            num="quadratures", figsize=config.figsize, layout="tight"
        )
        iq_ax.set(
            title=f"Lookback · {adc} · Readout quadratures",
            xlabel=_TIME_LABEL,
            ylabel="Mean IQ (a.u.)",
        )
        iq_ax.plot(Ts, np.real(signals), color="C0", label="I (in-phase)")
        iq_ax.plot(Ts, np.imag(signals), color="C1", label="Q (quadrature)")
        iq_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)

        return LookbackAnalysis(predict_offset=offset)
