from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import (
    round_zcu_gain,
    sweep2array,
    wrap_earlystop_check,
)
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Join,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SoftDelay,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fitlor
from zcu_tools.utils.process import rotate2real

# (amps, freqs, signals2D)
AcStarkResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


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


class AcStarkModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    stark_pulse1: PulseCfg
    stark_pulse2: PulseCfg
    readout: ReadoutCfg


class AcStarkCfg(ModularProgramCfg, TaskCfg):
    modules: AcStarkModuleCfg
    sweep: dict[str, SweepCfg]


class AcStarkExp(AbsExperiment[AcStarkResult, AcStarkCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        earlystop_snr: Optional[float] = None,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> AcStarkResult:
        _cfg = check_type(deepcopy(cfg), AcStarkCfg)
        setup_devices(_cfg, progress=True)
        modules = _cfg["modules"]

        gain_sweep = _cfg["sweep"]["gain"]
        freq_sweep = _cfg["sweep"]["freq"]

        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules["stark_pulse2"].ch},
        )
        gains = np.sqrt(
            np.linspace(
                gain_sweep["start"] ** 2, gain_sweep["stop"] ** 2, gain_sweep["expts"]
            )
        )
        gains = round_zcu_gain(gains, soccfg, modules["stark_pulse1"].ch)

        freq_param = sweep2param("freq", freq_sweep)
        modules["stark_pulse2"].set_param("freq", freq_param)

        with LivePlot2DwithLine(
            "Stark Pulse Gain (a.u.)",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=2,
            uniform=False,
        ) as viewer:
            ax1d = viewer.get_ax("1d")

            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: (
                        (modules := ctx.cfg["modules"])
                        and (
                            prog := ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset("reset", modules.get("reset")),
                                    Pulse(
                                        "stark_pulse1",
                                        modules["stark_pulse1"],
                                        block_mode=False,
                                    ),
                                    Pulse("stark_pulse2", modules["stark_pulse2"]),
                                    Readout("readout", modules["readout"]),
                                ],
                                sweep=[("freq", ctx.cfg["sweep"]["freq"])],
                            )
                        ).acquire(
                            soc,
                            progress=False,
                            callback=wrap_earlystop_check(
                                prog,
                                update_hook,
                                earlystop_snr,
                                signal2real_fn=np.abs,
                                after_check=lambda snr: ax1d.set_title(
                                    f"snr = {snr:.1f}"
                                ),
                            ),
                            **(acquire_kwargs or {}),
                        )
                    ),
                    result_shape=(len(freqs),),
                    pbar_n=_cfg["rounds"],
                ).scan(
                    "resonator gain",
                    list(gains.tolist()),
                    before_each=lambda _, ctx, gain: ctx.cfg["modules"][
                        "stark_pulse1"
                    ].set_param("gain", gain),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    gains, freqs, acstark_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (gains, freqs, signals)

        return gains, freqs, signals

    def analyze(
        self,
        result: Optional[AcStarkResult] = None,
        *,
        chi: float,
        kappa: float,
        deg: int = 1,
        cutoff: Optional[float] = None,
    ) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, freqs, signals = result

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

    def save(
        self,
        filepath: str,
        result: Optional[AcStarkResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ac_stark",
        **kwargs,
    ) -> None:
        """Save AC Stark experiment data."""
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, freqs, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Stark Pulse Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> AcStarkResult:
        signals2D, gains, freqs, cfg = load_data(filepath, return_cfg=True, **kwargs)
        assert freqs is not None
        assert len(gains.shape) == 1 and len(freqs.shape) == 1
        assert signals2D.shape == (len(gains), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz

        gains = gains.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = cast(AcStarkCfg, cfg)
        self.last_result = (gains, freqs, signals2D)

        return gains, freqs, signals2D


def acstark_ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class AcStarkRamseyModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi2_pulse: PulseCfg
    stark_pulse: PulseCfg
    readout: ReadoutCfg


class AcStarkRamseyCfg(ModularProgramCfg, TaskCfg):
    modules: AcStarkRamseyModuleCfg
    wait_delay: float
    sweep: dict[str, SweepCfg]


class AcStarkRamseyExp(AbsExperiment[AcStarkResult, AcStarkRamseyCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        detune: float = 0.0,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> AcStarkResult:
        _cfg = check_type(deepcopy(cfg), AcStarkRamseyCfg)
        setup_devices(_cfg, progress=True)
        modules = _cfg["modules"]

        gain_sweep = _cfg["sweep"]["gain"]

        lengths = sweep2array(
            _cfg["sweep"]["length"],
            "time",
            {"soccfg": soccfg, "gen_ch": modules["stark_pulse"].ch},
        )
        gains = np.sqrt(
            np.linspace(
                gain_sweep["start"] ** 2, gain_sweep["stop"] ** 2, gain_sweep["expts"]
            )
        )
        gains = round_zcu_gain(gains, soccfg, modules["stark_pulse"].ch)

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: AcStarkRamseyCfg = cast(AcStarkRamseyCfg, ctx.cfg)
            modules = cfg["modules"]

            length_sweep = cfg["sweep"]["length"]
            length_param = sweep2param("length", length_sweep)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Join(
                        Pulse("stark_pulse", modules["stark_pulse"]),
                        [
                            SoftDelay("wait_delay", delay=cfg["wait_delay"]),
                            Pulse("pi2_pulse1", modules["pi2_pulse"]),
                            SoftDelay("t2_delay", delay=length_param),
                            Pulse(
                                name="pi2_pulse2",
                                cfg=modules["pi2_pulse"].with_updates(
                                    phase=modules["pi2_pulse"].phase
                                    + 360 * detune * length_param
                                ),
                            ),
                        ],
                    ),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("length", length_sweep)],
            ).acquire(
                soc,
                progress=False,
                callback=update_hook,
                **(acquire_kwargs or {}),
            )

        with LivePlot2DwithLine(
            "Stark Pulse Gain (a.u.)",
            "Time (us)",
            line_axis=1,
            num_lines=2,
            uniform=False,
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths),),
                    pbar_n=_cfg["rounds"],
                ).scan(
                    "resonator gain",
                    list[float](gains.tolist()),
                    before_each=lambda _, ctx, gain: ctx.cfg["modules"][
                        "stark_pulse"
                    ].set_param("gain", gain),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    gains,
                    lengths,
                    acstark_ramsey_signal2real(np.asarray(ctx.root_data)),
                ),
            )
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (gains, lengths, signals)

        return gains, lengths, signals

    def analyze(
        self,
        result: Optional[AcStarkResult] = None,
        *,
        detune: float = 0.0,
        cutoff: Optional[float] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, lens, signals = result

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

    def save(
        self,
        filepath: str,
        result: Optional[AcStarkResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ac_stark_ramsey",
        **kwargs,
    ) -> None:
        """Save AC Stark experiment data."""
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        gains, lens, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Stark Pulse Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Time", "unit": "s", "values": lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> AcStarkResult:
        signals2D, gains, lens, cfg = load_data(filepath, return_cfg=True, **kwargs)
        assert gains is not None and lens is not None
        assert len(gains.shape) == 1 and len(lens.shape) == 1
        assert signals2D.shape == (len(lens), len(gains))

        lens = lens * 1e6  # s -> us
        signals2D = signals2D.T  # transpose back

        gains = gains.astype(np.float64)
        lens = lens.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = cast(AcStarkRamseyCfg, cfg)
        self.last_result = (gains, lens, signals2D)

        return gains, lens, signals2D
