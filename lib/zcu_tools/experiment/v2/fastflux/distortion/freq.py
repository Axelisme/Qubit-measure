from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Callable, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    NonBlocking,
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


class FreqModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    flux_pulse: PulseCfg
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class FreqCfg(ModularProgramCfg, TaskCfg):
    modules: FreqModuleCfg
    readout_t: float
    sweep: dict[str, SweepCfg]


class FreqExp(AbsExperiment):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> FreqResult:
        _cfg = check_type(deepcopy(cfg), FreqCfg)
        modules = _cfg["modules"]

        # force length be the outer loop
        _cfg["sweep"] = {
            "length": _cfg["sweep"]["length"],
            "freq": _cfg["sweep"]["freq"],
        }

        lengths = sweep2array(_cfg["sweep"]["length"], "time", {"soccfg": soccfg})
        freqs = sweep2array(
            _cfg["sweep"]["freq"],
            "freq",
            {"soccfg": soccfg, "gen_ch": modules["qub_pulse"]["ch"]},
        )

        length_params = sweep2param("length", _cfg["sweep"]["length"])
        freq_params = sweep2param("freq", _cfg["sweep"]["freq"])
        Pulse.set_param(modules["qub_pulse"], "freq", freq_params)

        with LivePlotter2D("Time (us)", "Frequency (MHz)") as viewer:

            def measure_fn(ctx: TaskState, update_hook: Optional[Callable]):
                modules = ctx.cfg["modules"]
                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        NonBlocking(
                            [
                                Pulse(
                                    "flux_pulse",
                                    modules["flux_pulse"],
                                    block_mode=False,
                                ),
                                SoftDelay("wait_time", delay=length_params),
                                Pulse("qub_pulse", modules["qub_pulse"]),
                            ]
                        ),
                        SoftDelay("readout_t", ctx.cfg["readout_t"]),
                        Readout("readout", modules["readout"]),
                    ],
                ).acquire(soc, progress=False, callback=update_hook)

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn, result_shape=(len(lengths), len(freqs))
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, freqs, freq_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg: Optional[dict] = deepcopy(cfg)
        self.last_result: Optional[FreqResult] = (lengths, freqs, signals)

        return lengths, freqs, signals

    def analyze(
        self,
        cfg: Optional[dict[str, Any]] = None,
        result: Optional[FreqResult] = None,
        timeFly: Optional[float] = None,
    ) -> tuple[float, Figure]:
        if cfg is None:
            cfg = self.last_cfg
        assert cfg is not None, "No config found"
        _cfg = check_type(deepcopy(cfg), FreqCfg)
        modules = _cfg["modules"]

        flux_pulse = modules["flux_pulse"]
        qub_len = float(modules["qub_pulse"]["waveform"]["length"])

        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        lengths, freqs, signals = result

        # align to center of qubit pulse
        lengths = lengths + qub_len / 2

        amps = freq_signal2real(signals)
        s_lengths, s_freqs = get_resonance_freq(lengths, freqs, amps)

        sort_idxs = np.argsort(np.abs(s_freqs))
        mean_background = np.mean(s_freqs[sort_idxs[: int(len(s_freqs) * 0.1)]])
        mean_topdetune = np.mean(s_freqs[sort_idxs[int(len(s_freqs) * 0.9) :]])
        if timeFly is None:
            detune_ratios = (s_freqs - mean_background) / (
                mean_topdetune - mean_background
            )
            top_idx = np.argmax(detune_ratios)
            candidate_mask = detune_ratios[:top_idx] < 0.5
            if not np.any(candidate_mask):
                offset = float(s_lengths[0])
            else:
                offset = s_lengths[np.nonzero(candidate_mask)[0][-1]]
            timeFly = offset - float(flux_pulse["pre_delay"])
        assert timeFly is not None

        start_t = float(flux_pulse["pre_delay"]) + timeFly
        end_t = start_t + float(flux_pulse["waveform"]["length"])

        ideal_lengths = np.linspace(s_lengths[0], s_lengths[-1], 1000)
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
        )

        # Plot the resonance frequencies and fitted curve
        ax.plot(s_lengths, s_freqs, ".", c="k")
        ax.plot(ideal_lengths, ideal_curve, "g-", label="Ideal")

        label = f"timeFly: {timeFly:.2f} us"
        plot_kwargs = dict(color="gray", alpha=0.3)
        ax.axvspan(
            start_t - qub_len / 2, start_t + qub_len / 2, label=label, **plot_kwargs
        )
        ax.axvspan(end_t - qub_len / 2, end_t + qub_len / 2, **plot_kwargs)

        ax.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
        ax.set_xlabel("Time (us)", fontsize=14)
        ax.legend(fontsize="x-large")

        fig.tight_layout()

        return timeFly, fig

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
