"""AC Stark Ramsey fringes versus resonator drive gain."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_gain, sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_heatmap, live_line


@dataclass(frozen=True)
class AcStarkRamseyResult:
    """Stark gains, Ramsey delays in us, and mean IQ shaped (Ngain, Nlength)."""

    gains: NDArray[np.float64]
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


@validated_dataclass(frozen=True)
class AcStarkRamseyAnalyzeOptions:
    """FFT center in MHz (subtracted from the axis) and exclusive gain cutoff."""

    detune: Annotated[
        float,
        FieldMeta(
            label="FFT reference frequency",
            unit="MHz",
            description="Set the center of the FFT window in MHz; the displayed axis subtracts this value.",
        ),
    ] = 0.0
    cutoff: Annotated[
        float | None,
        FieldMeta(
            label="Stark gain cutoff",
            unit="a.u.",
            description="Exclude gains at or above this value; blank keeps every row.",
        ),
    ] = None


def acstark_ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    return rotate2real(signals).real


class AcStarkRamseyCfg(ConfigBase):
    reset: Reset.Cfg | None = None
    stark_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Stark resonator tone",
            description="Fill in the resonator drive pulse played alongside the Ramsey sequence.",
        ),
    ]
    pi2_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Ramsey π/2 pulse",
            description="Fill in the calibrated π/2 qubit drive for the two Ramsey rotations.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Fill in the resonator readout used after the tone and Ramsey sequence end.",
        ),
    ]
    wait_delay: Annotated[
        float,
        FieldMeta(
            label="Tone settling delay",
            unit="µs",
            description="Fill in the wait in µs from Stark tone start to the first π/2 pulse.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Stark tone gain",
            unit="a.u.",
            description="Fill in the gain endpoints and number of delay traces; the grid is uniform in gain squared.",
        ),
    ]
    delay: Annotated[
        SweepCfg,
        FieldMeta(
            label="Free evolution time",
            unit="µs",
            description="Fill in the time endpoints in µs and the number of delays between π/2 pulses.",
        ),
    ]
    detune: Annotated[
        float,
        FieldMeta(
            label="Phase-ramp detune",
            unit="MHz",
            description="Set the added phase slope in MHz, positive for advancing the second π/2 phase.",
        ),
    ] = 0.0

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


class AcStarkRamseyExp(Experiment[AcStarkRamseyResult, AcStarkRamseyCfg]):
    Options: ClassVar[type[AcStarkRamseyAnalyzeOptions]] = AcStarkRamseyAnalyzeOptions

    # gains stored in a.u. -> IDENTITY; lengths stored in s on disk -> US_TO_S
    AXES_SPEC: ClassVar[AxesSpec[AcStarkRamseyResult, AcStarkRamseyCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("lengths", label="Time", unit="s", scale=1e-6),
            AxesSpec.Axis("gains", label="Stark Pulse Gain", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=AcStarkRamseyResult,
        cfg_type=AcStarkRamseyCfg,
        cfg_schema_version="4.0",
        tag="twotone/ge/ac_stark_ramsey",
    )

    def run(
        self,
        cfg: AcStarkRamseyCfg,
        *,
        context: RunContext,
    ) -> AcStarkRamseyResult:
        """Measure Stark-induced Ramsey fringes versus tone gain and free delay.

        After reset, start the fixed configured Stark pulse alongside
        wait_delay -> pi/2 -> swept delay -> pi/2. Advance the final pulse's
        phase by 360 * detune * delay (MHz times us), then read out after both
        branches finish. Choose Stark duration and wait_delay for the desired
        overlap. Host gains are spaced uniformly in gain squared before
        rounding; the FPGA sweeps the Ramsey delay, not Stark pulse duration.
        Return mean complex IQ on the (gain, delay) grid with delays in us.
        """
        # 1. Prepare Ramsey delays and gain points spaced by nominal Stark power.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        gain_sweep = cfg.gain

        lengths = sweep2array(
            cfg.delay,
            "time",
            {"soccfg": soccfg, "gen_ch": cfg.stark_pulse.ch},
        )
        gains = np.sqrt(
            np.linspace(gain_sweep.start**2, gain_sweep.stop**2, gain_sweep.expts)
        )
        gains = round_zcu_gain(gains, soccfg, cfg.stark_pulse.ch)

        signals_buffer = self._live_spectra(context.plots, gains, lengths)

        # 3. At each Stark gain, sweep the Ramsey delay and its programmed phase.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("gain", gains.tolist()):
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                length = builder.add_sweep("delay", cfg.delay)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.fork("default", into=["stark", "ramsey"])
                builder.emit(
                    Pulse("stark_pulse", cfg.stark_pulse, gain=step.value),
                    timeline="stark",
                )
                with builder.timeline("ramsey"):
                    builder.timeline_inc(cfg.wait_delay)
                    pi2_pulse = Pulse("pi2_pulse", cfg.pi2_pulse)
                    builder.emit(pi2_pulse)
                    builder.timeline_inc(length)
                    builder.emit(
                        Pulse(
                            "pi2_pulse",
                            cfg.pi2_pulse,
                            phase=cfg.pi2_pulse.phase + 360 * cfg.detune * length,
                        )
                    )
                builder.join("stark", "ramsey", into="default")
                builder.emit(Readout("readout", cfg.readout))
                step.run_acquire(builder.build(), soc, rounds=cfg.rounds)

        # 4. Preserve the gain-delay map for Stark-shift spectroscopy.
        return AcStarkRamseyResult(gains, lengths, signals_buffer.array)

    def _live_spectra(
        self, plots: Plots, gains: NDArray[np.float64], lengths: NDArray[np.float64]
    ) -> SignalBuffer:
        """Attach the map and recent spectra to the run's live buffer."""
        # 2. Track Stark-dependent Ramsey oscillations and recent delay traces.
        fig, _ = plots.subplots(
            1, 2, num="measurement", figsize=(10, 4.8), layout="constrained"
        )
        fig.suptitle("AC Stark Ramsey · qubit · Ramsey fringes")
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel="Stark tone gain (a.u.)", ylabel="Free evolution time (µs)")
        contrast = "Signed IQ principal-axis projection (a.u.)"
        line_ax.set(xlabel="Free evolution time (µs)", ylabel=contrast)
        heat_ax.grid(False)
        heatmap = live_heatmap(heat_ax, uniform=False)
        (image,) = heatmap.set(
            gains, lengths, np.full((len(gains), len(lengths)), np.nan)
        )
        image.set_cmap("viridis")
        fig.colorbar(image, ax=heat_ax, label=contrast)
        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        lines = live_line(line_ax, num_lines=2)
        line_ax.lines[0].set(
            label="Previous gain", marker="o", markersize=3, color="0.55"
        )
        line_ax.lines[1].set(label="Latest gain", marker="o", markersize=3, color="C0")
        line_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2)

        @plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            """Refresh the Stark Ramsey map and two recent delay traces.

            data is complex IQ shaped (gain, delay); NaN marks unmeasured points.
            Return the changed artists without modifying the input array.
            """
            real = acstark_ramsey_signal2real(data)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((2, len(lengths)), np.nan)
            for offset in range(min(2, current + 1)):
                history[-offset - 1] = real[current - offset]
            progress = heat_ax.set_title(
                f"Gain rows with data: {len(valid)}/{len(gains)}"
            )
            if current >= 0:
                line_ax.set_title(f"Latest gain: {gains[current]:.4g} a.u.")
            return (
                *heatmap.set(gains, lengths, real),
                *lines.set(lengths, history),
                progress,
            )

        fig.show()
        return SignalBuffer(
            (len(gains), len(lengths)),
            on_update=update,
        )

    def analyze(
        self,
        source: RunRecord[AcStarkRamseyCfg, AcStarkRamseyResult],
        options: AcStarkRamseyAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot Ramsey contrast and its FFT versus Stark gain squared.

        ``cutoff`` excludes gains at or above that value. ``detune`` in MHz
        selects a +/-5 MHz FFT window and is subtracted from its frequency
        axis; set it to the acquisition detune when removing that offset.
        Publish ``fit`` with per-gain normalized FFT magnitudes, not a fitted
        Stark coefficient or coherence time. Source cfg may be None.
        """
        # 1. Resolve Stark-shift sidebands near the chosen Ramsey detuning below cutoff.
        result = source.result
        detune, cutoff = options.detune, options.cutoff

        gains, lens, signals = result.gains, result.lengths, result.signals

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            signals = signals[valid_indices, :]

        min_interval = 0.2  # MHz
        n_length = signals.shape[1]
        n_length = max(n_length, int(n_length / (np.ptp(lens) * min_interval)))

        real_signals = acstark_ramsey_signal2real(signals)
        fft_signals = np.abs(
            np.fft.fft(
                signals - np.mean(signals, axis=1, keepdims=True), n=n_length, axis=1
            )
        )
        fft_freqs = np.fft.fftfreq(n_length, d=(np.ptp(lens) / (signals.shape[1] - 1)))
        freq_mask = np.logical_and(fft_freqs >= detune - 5.0, fft_freqs <= detune + 5.0)
        fft_freqs = fft_freqs[freq_mask] - detune
        fft_signals = fft_signals[:, freq_mask]

        fft_signals /= np.max(fft_signals, axis=1, keepdims=True)

        # 2. Show the Run contrast first; the FFT indicates the fringe shift.
        fig, _ = plots.subplots(
            num="fit", nrows=2, ncols=1, figsize=(8, 7), layout="constrained"
        )
        fringes, spectrum = fig.axes
        fig.suptitle("AC Stark Ramsey · qubit · fringes and spectrum")
        fringes.set(
            xlabel="Stark tone gain (a.u.)",
            ylabel="Free evolution time (µs)",
        )
        fringes.grid(False)
        (time_image,) = live_heatmap(fringes, uniform=False).set(
            gains, lens, real_signals
        )
        time_image.set_cmap("viridis")
        fig.colorbar(
            time_image, ax=fringes, label="Signed IQ principal-axis projection (a.u.)"
        )

        order = np.argsort(fft_freqs)
        spectrum.set(
            xlabel="Squared Stark tone gain (a.u.²)",
            ylabel="FFT offset (MHz)",
            title=f"FFT relative to {detune:g} MHz",
        )
        spectrum.grid(False)
        (fft_image,) = live_heatmap(spectrum, uniform=False, clim=(0, 1)).set(
            gains**2, fft_freqs[order], fft_signals[:, order]
        )
        fft_image.set_cmap("viridis")
        fig.colorbar(
            fft_image,
            ax=spectrum,
            label="Row-normalized FFT magnitude (a.u.)",
            ticks=(0, 0.5, 1),
        )
