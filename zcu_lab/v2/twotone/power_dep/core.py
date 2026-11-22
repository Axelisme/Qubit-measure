from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.image import NonUniformImage
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.typing import NDArray
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
from zcu_tools.utils.process import minus_background

from zcu_lab.v2.support import live_line


def _contrast_map(
    axes: Axes,
    gains: NDArray[np.float64],
    freqs: NDArray[np.float64],
    contrast: NDArray[np.float64],
) -> NonUniformImage:
    # NonUniformImage requires the colormap before the first set_data call.
    image = NonUniformImage(axes, cmap="viridis", interpolation="nearest")
    axes.add_image(image)
    _set_contrast_map(image, gains, freqs, contrast)
    return image


def _set_contrast_map(
    image: NonUniformImage,
    gains: NDArray[np.float64],
    freqs: NDArray[np.float64],
    contrast: NDArray[np.float64],
) -> None:
    # Keep scan-order limits; nearest-sample rendering takes ascending centers.
    dg = 0.5 * (gains[-1] - gains[0]) / max(1, len(gains) - 1)
    df = 0.5 * (freqs[-1] - freqs[0]) / max(1, len(freqs) - 1)
    image.set_extent((gains[0] - dg, gains[-1] + dg, freqs[0] - df, freqs[-1] + df))
    gx, fy = np.argsort(gains), np.argsort(freqs)
    image.set_data(gains[gx], freqs[fy], contrast[np.ix_(gx, fy)].T)
    image.autoscale()


@dataclass(frozen=True)
class PowerResult:
    """Drive gains, frequencies in MHz, and mean IQ shaped (Ngain, Nfreq)."""

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class PowerCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Probe Pulse",
            description="Fill in the qubit drive waveform; the scans supply its RF frequency and gain.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Fill in the calibrated resonator readout after the probe.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Probe gain",
            unit="a.u.",
            description="Fill in the probe amplitude endpoints (a.u.) and the number of spectra.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Probe RF frequency",
            unit="MHz",
            description="Fill in the drive RF endpoints (MHz) and number of points per spectrum.",
        ),
    ]
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


class PowerExp(Experiment[PowerResult, PowerCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=1e6; outer gains -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[PowerResult, PowerCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("gains", label="Power", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=PowerResult,
        cfg_type=PowerCfg,
        cfg_schema_version="3.0",
        tag="twotone/power_dep",
    )

    def run(
        self,
        cfg: PowerCfg,
        *,
        context: RunContext,
    ) -> PowerResult:
        """Map the qubit spectrum as a function of drive gain.

        For each host-selected gain, reset, apply the optional init pulse,
        sweep the qubit probe frequency on the FPGA, then read out. Return
        mean complex IQ on the (gain, frequency) grid. Gain is the configured
        drive amplitude, not a calibrated power in dBm.
        """
        # 1. Prepare drive-gain rows and the probe's hardware frequency grid.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        gain_sweep = cfg.gain
        freq_sweep = cfg.freq

        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
            allow_array=True,
        )
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
        )

        # 2. Track power-dependent spectral contrast and recent frequency traces.
        fig, _ = context.plots.subplots(
            1, 2, num="measurement", figsize=(10, 4.8), layout="constrained"
        )
        fig.suptitle("Two-tone power dependence · qubit probe · IQ contrast")
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel="Probe gain (a.u.)", ylabel="Probe RF frequency (MHz)")
        contrast_label = "Background-subtracted IQ magnitude (a.u.)"
        line_ax.set(xlabel="Probe RF frequency (MHz)", ylabel=contrast_label)
        heat_ax.grid(False)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        heat_ax.xaxis.set_major_formatter(formatter)
        heat_ax.yaxis.set_major_formatter(formatter)
        image = _contrast_map(
            heat_ax, gains, freqs, np.full((len(gains), len(freqs)), np.nan)
        )
        fig.colorbar(image, ax=heat_ax, label=contrast_label)
        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        lines = live_line(line_ax, num_lines=2)
        line_ax.lines[0].set(
            label="Previous gain", marker="o", markersize=3, color="0.55"
        )
        line_ax.lines[1].set(label="Latest gain", marker="o", markersize=3, color="C0")
        line_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            """Refresh the drive-gain contrast map and two recent frequency traces.

            data is complex IQ shaped (gain, frequency); NaN marks unmeasured points.
            Return the changed artists without modifying the input array.
            """
            # Display each row's contrast relative to its own median background.
            real = np.abs(minus_background(data, axis=1))
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((2, len(freqs)), np.nan)
            for offset in range(min(2, current + 1)):
                history[-offset - 1] = real[current - offset]
            progress = heat_ax.set_title(
                f"Gain rows with data: {len(valid)}/{len(gains)}"
            )
            if current >= 0:
                line_ax.set_title(f"Latest gain: {gains[current]:.4g} a.u.")
            _set_contrast_map(image, gains, freqs, real)
            return (image, *lines.set(freqs, history), progress)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(gains), len(freqs)),
            on_update=update,
        )

        # 3. Acquire a complete spectrum at each fixed drive gain.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("gain", gains.tolist()):
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )

                freq_sweep = cfg.freq
                freq_param = builder.add_sweep("freq", freq_sweep)

                # Reset and initialize before each probe; read out its response.

                if cfg.reset is not None:
                    builder.emit(cfg.reset.build("reset"))
                builder.emit(Pulse("init_pulse", cfg.init_pulse))
                builder.emit(
                    Pulse("qub_pulse", cfg.qub_pulse, freq=freq_param, gain=step.value)
                )
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()
                step.run_acquire(prog, soc, rounds=cfg.rounds)

        return PowerResult(
            gains=gains,
            freqs=freqs,
            signals=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[PowerCfg, PowerResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Draw the gain–frequency map and two terminal trace slots from stored IQ.

        Pass None for options. Missing samples stay blank, and a missing cfg
        does not prevent plotting. The named measurement figure is added to
        plots without opening it or returning a fit.
        """
        del options
        result = source.result
        gains, freqs = result.gains, result.freqs
        contrast = np.abs(minus_background(result.signals, axis=1))
        valid = np.flatnonzero(~np.all(np.isnan(contrast), axis=1))
        current = int(valid[-1]) if valid.size else -1
        history = np.full((2, len(freqs)), np.nan)
        for offset in range(min(2, current + 1)):
            history[-offset - 1] = contrast[current - offset]

        fig, _ = plots.subplots(
            1, 2, num="measurement", figsize=(10, 4.8), layout="constrained"
        )
        fig.suptitle("Two-tone power dependence · qubit probe · IQ contrast")
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel="Probe gain (a.u.)", ylabel="Probe RF frequency (MHz)")
        contrast_label = "Background-subtracted IQ magnitude (a.u.)"
        line_ax.set(xlabel="Probe RF frequency (MHz)", ylabel=contrast_label)
        heat_ax.grid(False)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        heat_ax.xaxis.set_major_formatter(formatter)
        heat_ax.yaxis.set_major_formatter(formatter)
        image = _contrast_map(heat_ax, gains, freqs, contrast)
        fig.colorbar(image, ax=heat_ax, label=contrast_label)

        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        lines = live_line(line_ax, num_lines=2)
        line_ax.lines[0].set(
            label="Previous gain", marker="o", markersize=3, color="0.55"
        )
        line_ax.lines[1].set(label="Latest gain", marker="o", markersize=3, color="C0")
        line_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2)
        if current >= 0:
            line_ax.set_title(f"Latest gain: {gains[current]:.4g} a.u.")
        lines.set(freqs, history)
