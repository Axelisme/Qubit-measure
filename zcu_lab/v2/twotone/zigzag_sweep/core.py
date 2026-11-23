from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as validated_dataclass
from scipy.ndimage import gaussian_filter1d
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    LoadValue,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_heatmap


@dataclass(frozen=True)
class ZigZagScanResult:
    """Mean complex IQ with shape (repetition count, sweep value).

    times counts pi rotations, with two X90 pulses per count when selected.
    values is the hardware-rounded gain in a.u. or frequency in MHz.
    """

    times: NDArray[np.int64]
    values: NDArray[np.float64]
    signals: NDArray[np.complex128]


@validated_dataclass(frozen=True)
class ZigZagScanAnalyzeOptions:
    find_range: tuple[float | None, float | None] = (None, None)


@dataclass(frozen=True)
class ZigZagScanAnalysis:
    min_value: float


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    return rotate2real(signals).real


class ZigZagScanCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    X90_pulse: Annotated[Pulse.Cfg, FieldMeta(label="X90 Pulse")]
    X180_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="X180 Pulse")] = None
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[
        SweepCfg | None,
        FieldMeta(
            label="Repeated gate gain",
            unit="a.u.",
            description="Sweep the signed gain of the repeated gate, not the initial X90.",
        ),
    ] = None
    freq: Annotated[
        SweepCfg | None,
        FieldMeta(
            label="Repeated gate RF frequency",
            unit="MHz",
            description="Sweep the RF carrier of the repeated gate, not the initial X90.",
        ),
    ] = None
    n_times: Annotated[
        int,
        FieldMeta(
            label="Max repetitions",
            unit="count",
            description="Repetition counts run from zero to this value.",
        ),
    ]
    repeat_on: Annotated[
        Literal["X90_pulse", "X180_pulse"],
        FieldMeta(
            label="Repeat pulse",
            description="X90 repeats in pairs; each count adds one pi rotation.",
        ),
    ] = "X180_pulse"

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


def _live_plot(
    plots: Plots,
    times: NDArray[np.int64],
    values: NDArray[np.float64],
    *,
    frequency: bool,
    paired_x90: bool,
):
    """Present count-by-setting IQ and return the plot-thread update callback."""
    fig, ax = plots.subplots(num="measurement")
    scan_label = (
        "Repeated gate RF frequency (MHz)" if frequency else "Repeated gate gain (a.u.)"
    )
    gate = "X90 pairs" if paired_x90 else "X180"
    ax.set(
        title=f"ZigZag sweep · {gate}",
        xlabel=scan_label,
        ylabel="Gate repetitions (count)",
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    heatmap = live_heatmap(ax, uniform=True)
    ax.grid(False)
    fig.colorbar(ax.images[0], ax=ax, label="IQ projection (a.u.)")

    @plots.on_plot_thread
    def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
        """Refresh the ZigZag parameter-sweep contrast map.

        data is complex IQ shaped (repeat count, swept value).
        Return the changed artists without modifying the input array.
        """
        real = zigzag_signal2real(data)
        return heatmap.set(values, times.astype(np.float64), real.T)

    fig.show()
    return update


class ZigZagScanExp(Experiment[ZigZagScanResult, ZigZagScanCfg]):
    Options: ClassVar[type[ZigZagScanAnalyzeOptions]] = ZigZagScanAnalyzeOptions

    SWEEP_MAP = {
        "gain": {"name": "Gain (a.u.)", "param_key": "gain"},
        "freq": {"name": "Frequency (MHz)", "param_key": "freq"},
    }

    # inner-first: signals are (times, values), so values is inner and times
    # (int64) outer; both a.u. -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[ZigZagScanResult, ZigZagScanCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("values", label="Sweep value", unit="a.u."),
            AxesSpec.Axis("times", label="Times", unit="a.u.", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=ZigZagScanResult,
        cfg_type=ZigZagScanCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/zigzag_scan",
    )

    def run(self, cfg: ZigZagScanCfg, *, context: RunContext) -> ZigZagScanResult:
        """Scan repeated-gate errors versus pulse gain or frequency.

        After optional reset and a fixed X90 preparation, repeat the selected
        pulse and read out. Each count adds one X180 or two X90 pulses, from
        zero through n_times. Only the repeated pulse uses the swept parameter;
        the preparation keeps its original settings.

        cfg must provide the selected repeat_on pulse and a gain or frequency
        sweep, or ValueError is raised. Frequency takes precedence if both
        sweeps are present. Return counts, rounded sweep values and mean IQ.
        analyze selects the value with least signal variation across counts.
        """
        # 1. Select the repeated pulse and round its gain or frequency sweep.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        repeat_on = cfg.repeat_on

        times = np.arange(0, cfg.n_times + 1)

        if cfg.freq is not None:
            x_key = "freq"
            x_sweep = cfg.freq
        elif cfg.gain is not None:
            x_key = "gain"
            x_sweep = cfg.gain
        else:
            raise ValueError("No sweep parameter found in cfg")

        repeat_pulse = getattr(cfg, repeat_on)
        if repeat_pulse is None:
            raise ValueError(f"Repeat on pulse {repeat_on} not found")

        if x_key == "freq":
            values = sweep2array(
                x_sweep,
                "freq",
                {
                    "soccfg": soccfg,
                    "gen_ch": repeat_pulse.ch,
                    "mixer_freq": repeat_pulse.mixer_freq,
                },
            )
        else:
            values = sweep2array(
                x_sweep, "gain", {"soccfg": soccfg, "gen_ch": repeat_pulse.ch}
            )

        update = _live_plot(
            context.plots,
            times,
            values,
            frequency=x_key == "freq",
            paired_x90=repeat_on == "X90_pulse",
        )
        signals_buffer = SignalBuffer(
            (len(times), len(values)),
            on_update=update,
        )
        # 3. Amplify rotation errors across repeat counts while sweeping only the repeated gate.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("times", len(times))
            x_param = builder.add_sweep(x_key, x_sweep)
            # Preparation keeps its original settings; only the repeat is swept.
            repeated_cfg = deepcopy(repeat_pulse)
            repeated_cfg.set_param(x_key, x_param)
            loop_n = [
                int(n) for n in (2 * times if repeat_on == "X90_pulse" else times)
            ]
            builder.emit(
                LoadValue(
                    "load_repeat_count",
                    values=loop_n,
                    idx_reg="times",
                    val_reg="repeat_count",
                )
            )
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("X90_pulse", cfg.X90_pulse))
            builder.delay_by_timeline()
            with builder.repeat(
                "zigzag_loop", "repeat_count", range_hint=(min(loop_n), max(loop_n))
            ):
                builder.emit(Pulse(repeat_on, repeated_cfg))
                builder.delay_by_timeline()
            builder.emit(Readout("readout", cfg.readout))
            sched.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        return ZigZagScanResult(
            times=times, values=values, signals=signals_buffer.array
        )

    def analyze(
        self,
        source: RunRecord[ZigZagScanCfg, ZigZagScanResult],
        options: ZigZagScanAnalyzeOptions,
        *,
        plots: Plots,
    ) -> ZigZagScanAnalysis:
        """Choose the swept value with least count-dependent IQ variation.

        Sum absolute changes between adjacent counts after projecting IQ onto
        its signal axis, then smooth that loss over sweep values. find_range
        optionally bounds the search in the same units as result.values.
        Return the minimum-loss value and plot the map and loss. This is an
        error-amplification tuning criterion, not a gate-fidelity estimate.
        Fewer than two valid counts or an all-NaN search range raises ValueError.
        """
        # 1. Recover the repeated-gate response and requested calibration bounds.
        result = source.result
        find_range = options.find_range

        times = result.times
        values = result.values
        signals = result.signals

        # 2. Compare every sweep value over the same acquired count range.
        real_signals = zigzag_signal2real(signals)  # (times , values)
        valid_cutoff = np.min(np.sum(~np.isnan(real_signals), axis=0))

        if valid_cutoff < 2:
            raise ValueError("Not enough valid data points for analysis")

        times = times[:valid_cutoff]
        real_signals = real_signals[:valid_cutoff]

        # 3. Minimize smoothed count-to-count variation within the requested bounds.
        val_diff = np.sum(np.abs(np.diff(real_signals, axis=0)), axis=0)
        loss = gaussian_filter1d(val_diff, sigma=1)

        if find_range[0] is not None:
            loss[values < find_range[0]] = np.nan
        if find_range[1] is not None:
            loss[values > find_range[1]] = np.nan
        min_value = values[np.nanargmin(loss)]

        # 4. Relate the selected gain or frequency to repeat-count contrast and variation loss.
        fig, _ = plots.subplots(
            num="fit", nrows=2, ncols=1, sharex=True, layout="constrained"
        )
        ax1, ax2 = fig.axes

        dx = (values[1] - values[0]) * 0.5
        dy = (times[1] - times[0]) * 0.5
        image = ax1.imshow(
            zigzag_signal2real(signals),
            aspect="auto",
            extent=(values[0] - dx, values[-1] + dx, times[0] - dy, times[-1] + dy),
            origin="lower",
            interpolation="nearest",
            cmap="RdBu_r",
        )
        ax1.grid(False)
        cfg = source.cfg
        if cfg is None:
            scan_label, unit = "Repeated gate setting", ""
            gate = ""
        else:
            unit = "MHz" if cfg.freq is not None else "a.u."
            scan_label = (
                "Repeated gate RF frequency (MHz)"
                if cfg.freq is not None
                else "Repeated gate gain (a.u.)"
            )
            gate = " · X90 pairs" if cfg.repeat_on == "X90_pulse" else " · X180"
        fig.colorbar(image, ax=[ax1, ax2], label="IQ projection (a.u.)")
        ax1.set(title=f"ZigZag sweep{gate}", ylabel="Gate repetitions (count)")
        ax2.plot(values, loss, ".-", label="Smoothed IQ variation")
        ax2.axvline(
            min_value,
            color="red",
            linestyle="--",
            label=f"Minimum = {min_value:.5g} {unit}".rstrip(),
        )
        ax2.legend()
        ax2.set(xlabel=scan_label, ylabel="IQ variation (a.u.)")
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        return ZigZagScanAnalysis(min_value=float(min_value))
