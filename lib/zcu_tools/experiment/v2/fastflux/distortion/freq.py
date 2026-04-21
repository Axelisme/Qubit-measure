from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
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
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
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

# (lengths, freqs, signals2D)
FreqResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def freq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


def get_resonance_freq(
    xs: NDArray[np.float64], freqs: NDArray[np.float64], amps: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s_xs = []
    s_freqs = []

    for x, amp in zip(xs, amps):
        if np.any(np.isnan(amp)):
            continue

        param, _ = fitlor(freqs, amp)
        curr_freq = param[3]

        s_xs.append(x)
        s_freqs.append(curr_freq)

    return np.array(s_xs), np.array(s_freqs)


class FreqModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    flux_pulse: PulseCfg
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class FreqCfg(ModularProgramCfg, TaskCfg):
    modules: FreqModuleCfg
    readout_t: float
    sweep: dict[str, SweepCfg]


class FreqExp(AbsExperiment[FreqResult, FreqCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> FreqResult:
        _cfg = check_type(deepcopy(cfg), FreqCfg)
        setup_devices(_cfg, progress=True)
        modules = _cfg["modules"]

        length_sweep = _cfg["sweep"]["length"]
        freq_sweep = _cfg["sweep"]["freq"]

        qub_pulse = modules["qub_pulse"]

        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": qub_pulse.ch}
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: FreqCfg = cast(FreqCfg, ctx.cfg)
            modules = cfg["modules"]

            length_sweep = cfg["sweep"]["length"]
            freq_sweep = cfg["sweep"]["freq"]

            length_params = sweep2param("length", length_sweep)
            freq_params = sweep2param("freq", freq_sweep)
            modules["qub_pulse"].set_param("freq", freq_params)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Join(
                        Pulse("flux_pulse", modules["flux_pulse"]),
                        [
                            SoftDelay("wait_time", delay=length_params),
                            Pulse("qub_pulse", modules["qub_pulse"]),
                        ],
                        SoftDelay("readout_t", cfg["readout_t"]),
                    ),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[
                    ("length", length_sweep),
                    ("freq", freq_sweep),
                ],
            ).acquire(
                soc, progress=False, round_hook=update_hook, **(acquire_kwargs or {})
            )

        with LivePlot2D("Time (us)", "Frequency (MHz)") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths), len(freqs)),
                    pbar_n=_cfg["rounds"],
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, freqs, freq_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (lengths, freqs, signals)

        return lengths, freqs, signals

    def analyze(
        self, cfg: Optional[FreqCfg] = None, result: Optional[FreqResult] = None
    ) -> Figure:
        if cfg is None:
            cfg = self.last_cfg
        assert cfg is not None, "No config found"
        _cfg = check_type(deepcopy(cfg), FreqCfg)
        modules = _cfg["modules"]

        flux_pulse = modules["flux_pulse"]
        qub_len = float(modules["qub_pulse"].waveform.length)

        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        lengths, freqs, signals = result

        # align to center of qubit pulse
        lengths = lengths + qub_len / 2

        amps = freq_signal2real(signals)
        s_lengths, s_freqs = get_resonance_freq(lengths, freqs, amps)

        sort_idxs = np.argsort(np.abs(s_freqs - s_freqs[0]))
        mean_background = np.mean(s_freqs[sort_idxs[: int(len(s_freqs) * 0.1)]])
        mean_topdetune = np.mean(s_freqs[sort_idxs[int(len(s_freqs) * 0.9) :]])

        start_t = float(flux_pulse.pre_delay)
        end_t = start_t + float(flux_pulse.waveform.length)

        ideal_lengths = np.linspace(lengths[0], lengths[-1], 1000)
        ideal_curve = np.full_like(ideal_lengths, mean_background)
        ideal_curve[(ideal_lengths >= start_t) & (ideal_lengths <= end_t)] = (
            mean_topdetune
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.imshow(
            amps.T,
            extent=(lengths[0], lengths[-1], freqs[0], freqs[-1]),
            aspect="auto",
            interpolation="none",
            cmap="RdBu_r",
            origin="lower",
        )

        # Plot the resonance frequencies and fitted curve
        ax.plot(s_lengths, s_freqs, ".", c="k")
        ax.plot(ideal_lengths, ideal_curve, "g-", label="Ideal")

        plot_kwargs = dict(color="gray", alpha=0.3)
        ax.axvspan(start_t - qub_len / 2, start_t + qub_len / 2, **plot_kwargs)
        ax.axvspan(end_t - qub_len / 2, end_t + qub_len / 2, **plot_kwargs)

        ax.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
        ax.set_xlabel("Time (us)", fontsize=14)
        ax.legend(fontsize="x-large")

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[FreqResult] = None,
        comment: Optional[str] = None,
        tag: str = "fastflux/distortion/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        lengths, freqs, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Wait Time", "unit": "s", "values": lengths * 1e-6},
            y_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        signals2D, lengths, freqs = load_data(filepath, **kwargs)
        assert freqs is not None and lengths is not None
        assert len(freqs.shape) == 1 and len(lengths.shape) == 1
        assert signals2D.shape == (len(freqs), len(lengths))

        lengths = lengths * 1e6  # s -> us
        freqs = freqs * 1e-6  # Hz -> MHz
        signals2D = signals2D.T  # transpose back

        freqs = freqs.astype(np.float64)
        lengths = lengths.astype(np.float64)
        signals2D = signals2D.T.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (lengths, freqs, signals2D)

        return lengths, freqs, signals2D
