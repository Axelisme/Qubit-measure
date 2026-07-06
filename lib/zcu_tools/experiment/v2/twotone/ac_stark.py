from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    MHZ_TO_HZ,
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
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import (
    round_zcu_gain,
    snr_checker,
    sweep2array,
)
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.program.v2 import (
    Join,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SoftDelay,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import fitlor
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class AcStarkResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: AcStarkCfg | None = None


@dataclass(frozen=True)
class AcStarkRamseyResult:
    gains: NDArray[np.float64]
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: AcStarkRamseyCfg | None = None


def acstark_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    real_signals = rotate2real(signals).real

    valid_mask = np.any(~np.isnan(real_signals), axis=1)

    if not np.any(valid_mask):
        return real_signals

    valid_signals = real_signals[valid_mask, :]

    min_vals = np.nanmin(valid_signals, axis=1, keepdims=True)
    max_vals = np.nanmax(valid_signals, axis=1, keepdims=True)
    valid_signals = (valid_signals - min_vals) / (max_vals - min_vals)

    real_signals[valid_mask, :] = valid_signals

    return real_signals


def get_resonance_freq(
    xs: NDArray[np.float64], freqs: NDArray[np.float64], amps: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s_xs = []
    s_freqs = []

    prev_freq = np.nan
    for x, amp in zip(xs, amps):
        if np.any(np.isnan(amp)):
            continue

        param, _ = fitlor(freqs, amp)
        curr_freq = param[3]

        if abs(curr_freq - prev_freq) > 0.1 * (freqs[-1] - freqs[0]):
            continue

        prev_freq = curr_freq

        s_xs.append(x)
        s_freqs.append(curr_freq)

    return np.array(s_xs), np.array(s_freqs)


class AcStarkModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    stark_pulse1: PulseCfg
    stark_pulse2: PulseCfg
    readout: ReadoutCfg


class AcStarkSweepCfg(ConfigBase):
    gain: SweepCfg
    freq: SweepCfg


class AcStarkCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AcStarkModuleCfg
    sweep: AcStarkSweepCfg


class AcStarkExp(PersistableExperiment[AcStarkResult, AcStarkCfg]):
    # gains stored in a.u. -> IDENTITY; freqs stored in Hz on disk -> MHZ_TO_HZ
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("gains", "Stark Pulse Gain", "a.u."),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=AcStarkResult,
        cfg_type=AcStarkCfg,
        tag="twotone/ge/ac_stark",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: AcStarkCfg,
        *,
        earlystop_snr: float | None = None,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> AcStarkResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gain_sweep = cfg.sweep.gain
        freq_sweep = cfg.sweep.freq

        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.stark_pulse2.ch},
        )
        gains = np.sqrt(
            np.linspace(gain_sweep.start**2, gain_sweep.stop**2, gain_sweep.expts)
        )
        gains = round_zcu_gain(gains, soccfg, modules.stark_pulse1.ch)

        current_snr = 0.0

        def update_snr(snr: float) -> None:
            nonlocal current_snr
            current_snr = snr

        with LivePlot2DwithLine(
            "Stark Pulse Gain (a.u.)",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=2,
            uniform=False,
        ) as viewer:
            signals_buffer = SignalBuffer(
                (len(gains), len(freqs)),
                on_update=lambda data: viewer.update(
                    gains,
                    freqs,
                    acstark_signal2real(data),
                    title=f"snr = {current_snr:.1f}" if current_snr else None,
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
                for _, step in sched.scan("resonator gain", gains.tolist()):
                    step_cfg = step.cfg
                    modules = step_cfg.modules

                    modules.stark_pulse1.set_param("gain", step.value)
                    freq_sweep = step_cfg.sweep.freq
                    modules.stark_pulse2.set_param(
                        "freq", sweep2param("freq", freq_sweep)
                    )

                    _ = (
                        step.prog_builder(soc, soccfg)
                        .add(
                            Reset("reset", modules.reset),
                            Pulse(
                                "stark_pulse1", modules.stark_pulse1, block_mode=False
                            ),
                            Pulse("stark_pulse2", modules.stark_pulse2),
                            Readout("readout", modules.readout),
                        )
                        .declare_sweep("freq", freq_sweep)
                        .build_and_acquire(
                            stop_condition=snr_checker(
                                signals_buffer[step],
                                earlystop_snr,
                                lambda x: rotate2real(x).real,
                                after_check=update_snr,
                            ),
                            **(acquire_kwargs or {}),
                        )
                    )

        return AcStarkResult(gains, freqs, signals_buffer.array, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(
        self,
        result: AcStarkResult | None = None,
        *,
        chi: float,
        kappa: float,
        deg: int = 1,
        cutoff: float | None = None,
    ) -> tuple[float, Figure]:
        assert result is not None, "No result found"

        gains, freqs, signals = result.gains, result.freqs, result.signals

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            signals = signals[valid_indices, :]

        amps = acstark_signal2real(signals)
        s_gains, s_freqs = get_resonance_freq(gains, freqs, amps)

        gains2 = gains**2
        s_gains2 = s_gains**2

        # fitting max_freqs with ax2 + bx + c
        x2_fit = np.linspace(min(gains2), max(gains2), 100)
        if deg == 1:
            b, c = np.polyfit(s_gains2, s_freqs, 1)
            y_fit = b * x2_fit + c
        elif deg == 2:
            a, b, c = np.polyfit(s_gains2, s_freqs, 2)
            y_fit = a * x2_fit**2 + b * x2_fit + c
        else:
            raise ValueError(f"Degree {deg} is not supported.")

        # Calculate the Stark shift
        eta = kappa**2 / (kappa**2 + chi**2)
        ac_coeff = abs(b) / (2 * eta * chi)

        # plot the data and the fitted polynomial
        avg_n = ac_coeff * gains2

        fig, ax1 = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        # Use NonUniformImage for better visualization with gain^2 as x-axis
        im = NonUniformImage(ax1, cmap="viridis", interpolation="nearest")
        im.set_data(avg_n, freqs, amps.T)
        im.set_extent((avg_n[0], avg_n[-1], freqs[0], freqs[-1]))
        ax1.add_image(im)

        # Set proper limits for the plot
        ax1.set_xlim(avg_n[0], avg_n[-1])
        ax1.set_ylim(freqs[0], freqs[-1])

        # Plot the resonance frequencies and fitted curve
        ax1.plot(ac_coeff * s_gains2, s_freqs, ".", c="k")

        # Fit curve in terms of gain^2
        label = r"$\bar n$" + f" = {ac_coeff:.2g} " + r"$gain^2$"
        n_fit = ac_coeff * x2_fit
        ax1.plot(n_fit, y_fit, "-", label=label)

        # Create secondary x-axis for gain^2 (Readout Gain²)
        ax2 = ax1.twiny()

        # main x-axis: avg_n, secondary x-axis: gain^2
        # avg_n = ac_coeff * gains^2
        ax1.set_xticks(ax1.get_xticks())
        # ax1.set_xticklabels([f"{avg_n:.1f}" for avg_n in ax1.get_xticks()])
        ax1.set_xlabel(r"Average Photon Number ($\bar n$)", fontsize=14)

        # 上方次 x 軸顯示 gain
        avgn_ticks = ax1.get_xticks()
        gain_ticks = np.sqrt(avgn_ticks / ac_coeff)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(avgn_ticks)
        ax2.set_xticklabels([f"{gain:.2g}" for gain in gain_ticks])
        ax2.set_xlabel("Readout Gain (a.u.)", fontsize=14)

        ax1.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
        ax1.legend(fontsize="x-large")
        ax1.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return ac_coeff, fig


def acstark_ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AcStarkRamseyModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi2_pulse: PulseCfg
    stark_pulse: PulseCfg
    readout: ReadoutCfg


class AcStarkRamseySweepCfg(ConfigBase):
    gain: SweepCfg
    length: SweepCfg


class AcStarkRamseyCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AcStarkRamseyModuleCfg
    wait_delay: float
    sweep: AcStarkRamseySweepCfg


class AcStarkRamseyExp(PersistableExperiment[AcStarkRamseyResult, AcStarkRamseyCfg]):
    # gains stored in a.u. -> IDENTITY; lengths stored in s on disk -> US_TO_S
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("lengths", "Time", "s", scale=US_TO_S),
            Axis("gains", "Stark Pulse Gain", "a.u."),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=AcStarkRamseyResult,
        cfg_type=AcStarkRamseyCfg,
        tag="twotone/ge/ac_stark_ramsey",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: AcStarkRamseyCfg,
        *,
        detune: float = 0.0,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> AcStarkRamseyResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gain_sweep = cfg.sweep.gain

        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.stark_pulse.ch},
        )
        gains = np.sqrt(
            np.linspace(gain_sweep.start**2, gain_sweep.stop**2, gain_sweep.expts)
        )
        gains = round_zcu_gain(gains, soccfg, modules.stark_pulse.ch)

        with LivePlot2DwithLine(
            "Stark Pulse Gain (a.u.)",
            "Time (us)",
            line_axis=1,
            num_lines=2,
            uniform=False,
        ) as viewer:
            signals_buffer = SignalBuffer(
                (len(gains), len(lengths)),
                on_update=lambda data: viewer.update(
                    gains,
                    lengths,
                    acstark_ramsey_signal2real(data),
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
                for _, step in sched.scan("resonator gain", gains.tolist()):
                    step_cfg = step.cfg
                    modules = step_cfg.modules

                    modules.stark_pulse.set_param("gain", step.value)
                    length_sweep = step_cfg.sweep.length
                    length_param = sweep2param("length", length_sweep)

                    _ = (
                        step.prog_builder(soc, soccfg)
                        .add(
                            Reset("reset", modules.reset),
                            Join(
                                Pulse("stark_pulse", modules.stark_pulse),
                                [
                                    SoftDelay("wait_delay", delay=step_cfg.wait_delay),
                                    Pulse("pi2_pulse1", modules.pi2_pulse),
                                    SoftDelay("t2_delay", delay=length_param),
                                    Pulse(
                                        name="pi2_pulse2",
                                        cfg=modules.pi2_pulse.with_updates(
                                            phase=modules.pi2_pulse.phase
                                            + 360 * detune * length_param
                                        ),
                                    ),
                                ],
                            ),
                            Readout("readout", modules.readout),
                        )
                        .declare_sweep("length", length_sweep)
                        .build_and_acquire(
                            **(acquire_kwargs or {}),
                        )
                    )

        return AcStarkRamseyResult(
            gains, lengths, signals_buffer.array, cfg_snapshot=orig_cfg
        )

    @retrieve_result
    def analyze(
        self,
        result: AcStarkRamseyResult | None = None,
        *,
        detune: float = 0.0,
        cutoff: float | None = None,
    ) -> Figure:
        assert result is not None, "No result found"

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

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        assert isinstance(fig, Figure)

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

        return fig
