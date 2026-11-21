"""T1 relaxation versus background-tone gain and duration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from zcu_tools.analysis.fitting import (
    fit_decay,
)
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)

from zcu_lab.v2.support import live_heatmap, live_line

from .core import t1_signal2real


@dataclass(frozen=True)
class ScanT1WithToneResult:
    """Test-tone gains in a.u., durations in us and mean IQ shaped (gain, time)."""

    values: NDArray[np.float64]
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class ScanT1WithToneAnalysis:
    """Accepted test-tone gains in a.u. and fitted T1 values/errors in us."""

    gains: NDArray[np.float64]
    t1s: NDArray[np.float64]
    t1errs: NDArray[np.float64]


class ScanT1WithToneCfg(ConfigBase):
    reset: Reset.Cfg | None = None
    pi_pulse: Pulse.Cfg
    test_pulse: Pulse.Cfg
    readout: Readout.Cfg
    gain: Annotated[SweepCfg, FieldMeta(label="Gain", unit="a.u.")]
    length: Annotated[SweepCfg, FieldMeta(label="Length", unit="µs")]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]


class ScanT1WithToneExp(Experiment[ScanT1WithToneResult, ScanT1WithToneCfg]):
    # inner = times (result_shape, s on disk -> US_TO_S); outer = values (gain, a.u.)
    AXES_SPEC: ClassVar[AxesSpec[ScanT1WithToneResult, ScanT1WithToneCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("times", label="Time", unit="s", scale=1e-6),
            AxesSpec.Axis("values", label="Readout Gain", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=ScanT1WithToneResult,
        cfg_type=ScanT1WithToneCfg,
        cfg_schema_version="4.0",
        tag="twotone/ge/t1_with_tone_sweep",
    )

    def run(
        self, cfg: ScanT1WithToneCfg, *, context: RunContext
    ) -> ScanT1WithToneResult:
        """Measure tone-driven relaxation versus test-tone gain and duration.

        cfg supplies reset, π, test tone and readout modules, a host gain
        SweepCfg and a hardware duration sweep in us. context supplies acquisition and
        live plots. At each gain apply reset → π → test tone → readout.
        Return quantized gains, durations and mean complex IQ with shape
        (gain, time), keeping NaN for unmeasured points. Invalid pulse or sweep
        settings propagate validation or compilation errors.
        """
        # 1. Prepare the host gain axis and the generator's tone-duration axis.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.test_pulse.ch},
        )
        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "gen_ch": cfg.test_pulse.ch},
        )

        # 2. Show the gain/time map and the five most recent gain rows.
        fig, _ = context.plots.subplots(1, 2, num="measurement")
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel="Test tone gain (a.u.)", ylabel="Time (µs)")
        line_ax.set(xlabel="Time (µs)", ylabel="Rotated IQ real (a.u.)")
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        heat_ax.xaxis.set_major_formatter(formatter)
        heat_ax.yaxis.set_major_formatter(formatter)
        heatmap = live_heatmap(heat_ax, uniform=True)
        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        line_ax.grid()
        history_rows = 5
        lines = live_line(line_ax, num_lines=history_rows)
        for artist in line_ax.lines[:-1]:
            artist.set(marker="None", alpha=0.3, color="red")
        line_ax.lines[-1].set(label="current line", marker=".", color="C0")
        line_ax.legend()

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]):
            """Update the gain map and recent time traces from IQ; return artists."""
            real = t1_signal2real(data)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((history_rows, len(lengths)), np.nan)
            for offset in range(min(history_rows, current + 1)):
                history[-offset - 1] = real[current - offset]
            return (*heatmap.set(gains, lengths, real), *lines.set(lengths, history))

        fig.show()
        signals_buffer = SignalBuffer(
            (len(gains), len(lengths)),
            on_update=update,
        )

        # 3. Scan gains on the host; sweep tone duration inside each program.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("gain", gains.tolist()):
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )

                length_sweep = cfg.length
                length_param = builder.add_sweep("length", length_sweep)

                if cfg.reset is not None:
                    builder.emit(cfg.reset.build("reset"))
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
                builder.emit(
                    Pulse(
                        "test_pulse",
                        cfg.test_pulse,
                        length=length_param,
                        gain=step.value,
                    )
                )
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()
                step.run_acquire(prog, soc, rounds=cfg.rounds)

        # 4. Preserve gain-major IQ averages and unfinished NaN slots.
        return ScanT1WithToneResult(
            values=gains,
            times=lengths,
            signals=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[ScanT1WithToneCfg, ScanT1WithToneResult],
        options: None,
        *,
        plots: Plots,
    ) -> ScanT1WithToneAnalysis:
        """Fit T1 for each smoothed gain row and publish the accepted decay times.

        Pass saved gain/time coordinates and complex IQ with None options;
        cfg is not needed. Return accepted gains in a.u. and T1/errors in us.
        Skip NaN rows and fits with relative error above 0.3; raise ValueError
        when none survive. Publish fit panels without changing the saved arrays.
        """
        # 1. Smooth the saved gain/time map before projecting its response axis.
        result = source.result
        del options

        gains, ts, signals = result.values, result.times, result.signals

        signals = np.asarray(gaussian_filter(signals, sigma=1), dtype=np.complex128)
        real_signals = t1_signal2real(signals)

        # 2. Retain each gain row only when its fitted decay is resolved.
        t1s = np.full(len(gains), np.nan, dtype=np.float64)
        t1errs = np.zeros_like(t1s)

        for i in range(len(gains)):
            real_signal = real_signals[i, :]

            # Leave incomplete rows out of the fit summary.
            if np.any(np.isnan(real_signal)):
                continue

            fit = fit_decay(ts, real_signal)
            t1 = fit.decay_time
            t1err = fit.decay_time_err

            if t1err > 0.3 * t1:
                continue

            t1s[i] = t1
            t1errs[i] = t1err

        if np.all(np.isnan(t1s)):
            raise ValueError("No valid Fitting T1 found. Please check the data.")

        valid_idxs = ~np.isnan(t1s)
        gains = gains[valid_idxs]
        t1s = t1s[valid_idxs]
        t1errs = t1errs[valid_idxs]

        # 3. Compare the response map with accepted lifetimes across gains.
        fig, _ = plots.subplots(num="fit", nrows=2, ncols=1, sharex=True)
        ax1, ax2 = fig.axes

        fig.suptitle("T1 while readout")

        ax1.set_ylabel("T1 (µs)")
        ax1.imshow(
            real_signals.T,
            aspect="auto",
            extent=(gains[0], gains[-1], t1s[0], t1s[-1]),
            origin="lower",
        )
        ax2.errorbar(
            gains, t1s, yerr=t1errs, label="Fitting T1", elinewidth=1, capsize=1
        )
        ax2.set_xlabel("Readout Gain (a.u.)")
        ax2.set_ylabel("T1 (µs)")
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(gains[0], gains[-1])
        ax2.grid()

        # 4. Return accepted coordinates and their fitted lifetimes.
        return ScanT1WithToneAnalysis(gains=gains, t1s=t1s, t1errs=t1errs)
