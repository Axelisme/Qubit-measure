"""AC Stark Ramsey fringes versus resonator drive gain."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray
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
    GainSweep,
    Join,
    Pulse,
    Readout,
    Reset,
    SoftDelay,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_heatmap_with_line


@dataclass(frozen=True)
class AcStarkRamseyResult:
    """Stark gains, Ramsey delays in us, and mean IQ shaped (Ngain, Nlength)."""

    gains: NDArray[np.float64]
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class AcStarkRamseyAnalyzeOptions:
    """FFT center in MHz (subtracted from the axis) and exclusive gain cutoff."""

    detune: Annotated[float, FieldMeta(label="FFT detune (MHz)")] = 0.0
    cutoff: Annotated[float | None, FieldMeta(label="Gain cutoff (a.u.)")] = None


def acstark_ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AcStarkRamseyCfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: Reset.Cfg | None = None
    pi2_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Ramsey pi/2 pulse")]
    stark_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Stark resonator tone")]
    readout: Readout.Cfg
    wait_delay: Annotated[float, FieldMeta(label="Tone settling delay", unit="us")]
    gain: Annotated[GainSweep, FieldMeta(label="Stark gain", unit="a.u.")]
    length: Annotated[SweepCfg, FieldMeta(label="Ramsey delay", unit="us")]
    detune: Annotated[float, FieldMeta(label="Phase-ramp detune", unit="MHz")] = 0.0


class AcStarkRamseyExp(Experiment[AcStarkRamseyResult, AcStarkRamseyCfg]):
    Options: ClassVar[type[AcStarkRamseyAnalyzeOptions]] = AcStarkRamseyAnalyzeOptions

    # gains stored in a.u. -> IDENTITY; lengths stored in s on disk -> US_TO_S
    AXES_SPEC: ClassVar[AxesSpec[AcStarkRamseyResult, AcStarkRamseyCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("lengths", "Time", "s", scale=1e-6),
            AxesSpec.Axis("gains", "Stark Pulse Gain", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=AcStarkRamseyResult,
        cfg_type=AcStarkRamseyCfg,
        cfg_schema_version="3.0",
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
        # Prepare Ramsey delays and gain points spaced by nominal Stark power.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        gain_sweep = cfg.gain

        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "gen_ch": cfg.stark_pulse.ch},
        )
        gains = np.sqrt(
            np.linspace(gain_sweep.start**2, gain_sweep.stop**2, gain_sweep.expts)
        )
        gains = round_zcu_gain(gains, soccfg, cfg.stark_pulse.ch)

        viewer = live_heatmap_with_line(
            context.plots,
            "measurement",
            "Stark Pulse Gain (a.u.)",
            "Time (us)",
            line_axis=1,
            num_lines=2,
            uniform=False,
        )
        signals_buffer = SignalBuffer(
            (len(gains), len(lengths)),
            on_update=lambda data: viewer.update(
                gains,
                lengths,
                acstark_ramsey_signal2real(data),
            ),
        )

        # At each Stark gain, sweep the Ramsey delay and its programmed phase.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("resonator gain", gains.tolist()):
                step_cfg = step.cfg

                step_cfg.stark_pulse.set_param("gain", step.value)
                length_sweep = step_cfg.length
                length_param = sweep2param("length", length_sweep)

                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", step_cfg.reset),
                        # Join waits for both the tone and the Ramsey branch.
                        Join(
                            Pulse("stark_pulse", step_cfg.stark_pulse),
                            [
                                SoftDelay("wait_delay", delay=step_cfg.wait_delay),
                                Pulse("pi2_pulse1", step_cfg.pi2_pulse),
                                SoftDelay("t2_delay", delay=length_param),
                                Pulse(
                                    name="pi2_pulse2",
                                    cfg=step_cfg.pi2_pulse.with_updates(
                                        phase=step_cfg.pi2_pulse.phase
                                        + 360 * cfg.detune * length_param
                                    ),
                                ),
                            ],
                        ),
                        Readout("readout", step_cfg.readout),
                    )
                    .declare_sweep("length", length_sweep)
                    .build_and_acquire(rounds=cfg.rounds)
                )

        return AcStarkRamseyResult(gains, lengths, signals_buffer.array)

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

        gains2 = gains**2

        fig, _ = plots.subplots("fit", nrows=2, ncols=1, sharex=True)
        ax1, ax2 = fig.axes

        im1 = NonUniformImage(ax1, cmap="viridis", interpolation="nearest")
        im1.set_data(gains2, lens, real_signals.T)
        im1.set_extent((gains2[0], gains2[-1], lens[0], lens[-1]))
        ax1.add_image(im1)
        ax1.set_ylim(lens[0], lens[-1])
        ax1.set_ylabel("Time (us)", fontsize=14)

        im2 = NonUniformImage(ax2, cmap="viridis", interpolation="nearest")
        im2.set_data(gains2, fft_freqs, fft_signals.T)
        im2.set_extent((gains2[0], gains2[-1], fft_freqs[0], fft_freqs[-1]))
        ax2.add_image(im2)
        ax2.set_xlim(gains2[0], gains2[-1])
        ax2.set_ylim(-5.0, 5.0)
        ax2.set_xlabel("Stark Pulse Gain² (a.u.²)", fontsize=14)
        ax2.set_ylabel("FFT Detune (MHz)", fontsize=14)

        fig.tight_layout()
