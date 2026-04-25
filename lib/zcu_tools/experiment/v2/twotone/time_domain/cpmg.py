from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typing_extensions import (
    Any,
    Callable,
    Optional,
    Sequence,
    TypeAlias,
    Union,
)

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array, wrap_earlystop_check
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.notebook.utils import make_sweep
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
    SoftDelay,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe
from zcu_tools.utils.process import rotate2real


def cpmg_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    real_signals = rotate2real(signals).real
    max_vals = np.max(real_signals, axis=1, keepdims=True)
    min_vals = np.min(real_signals, axis=1, keepdims=True)
    return (real_signals - min_vals) / np.clip(max_vals - min_vals, 1e-12, None)


# (times, lengths(time x length), signals)
CPMG_Result: TypeAlias = tuple[
    NDArray[np.int64], NDArray[np.float64], NDArray[np.complex128]
]


class CPMG_ModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    pi2_pulse: PulseCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class CPMG_SweepCfg(ConfigBase):
    times: Union[SweepCfg, Sequence[int]]
    length: SweepCfg


class CPMG_Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: CPMG_ModuleCfg
    sweep: CPMG_SweepCfg
    length_range: Union[list[tuple[float, float]], tuple[float, float]]
    length_expts: int


class CPMG_Exp(AbsExperiment[CPMG_Result, CPMG_Cfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: CPMG_Cfg,
        *,
        detune_ratio: float = 0.0,
        earlystop_snr: Optional[float] = None,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> CPMG_Result:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        pi2_pulse = modules.pi2_pulse
        pi_pulse = modules.pi_pulse

        time_sweep = cfg.sweep.times
        times = sweep2array(time_sweep, allow_array=True)
        if not np.allclose(times, np.round(times)):
            raise ValueError("Times must be integers")
        times = times.astype(np.int64)

        length_ranges = cfg.length_range
        if isinstance(length_ranges, tuple):
            ranges = np.zeros((len(times), 2), dtype=np.float64)
            ranges[:, 0] = length_ranges[0]
            ranges[:, 1] = length_ranges[1]
        else:
            ranges = np.array(length_ranges, dtype=np.float64)
        length_ranges = ranges

        lengths = np.array(  # (times x length)
            [
                sweep2array(
                    make_sweep(
                        start=length_ranges[i, 0],
                        stop=length_ranges[i, 1],
                        expts=cfg.length_expts,
                    ),
                    "time",
                    {"soccfg": soccfg, "scaler": 1 / (2 * t)},
                )
                for i, t in enumerate(times)
            ]
        )
        length_idxs = np.arange(cfg.length_expts, dtype=np.int64)

        if np.min(times) <= 0:
            raise ValueError("times should be larger than 0")

        min_interval = np.min(length_ranges[:, 0] / (2 * times))
        if min_interval < 0.5 * (  # the first and last delays are half of the interval
            pi2_pulse.waveform.length + pi_pulse.waveform.length
        ):
            raise ValueError(
                "The interval is too short to measure the CPMG signal",
                f"min_interval: {min_interval:.2g}, ",
                f"pi2_pulse_length: {pi2_pulse.waveform.length:.2g}, ",
                f"pi_pulse_length: {pi_pulse.waveform.length:.2g}",
            )

        with LivePlot2DwithLine(
            "Number of Pi", "Time idxs", line_axis=1, num_lines=2
        ) as viewer:
            ax1d = viewer.get_ax("1d")

            def measure_fn(
                ctx: TaskState[NDArray[np.complex128], Any, CPMG_Cfg],
                update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
            ) -> list[NDArray[np.float64]]:
                cfg = ctx.cfg
                modules = cfg.modules

                assert update_hook is not None

                time = ctx.env["time"]
                pi2_pulse = modules.pi2_pulse
                pi_pulse = modules.pi_pulse
                dpulse_len = pi_pulse.waveform.length - pi2_pulse.waveform.length

                length_sweep = cfg.sweep.length
                length_param = sweep2param("length", length_sweep)
                detune_param = (
                    360
                    * detune_ratio
                    * sweep2param(
                        "length",
                        make_sweep(start=0, step=1, expts=length_sweep["expts"]),
                    )
                )

                interval = length_param / (2 * time)

                return (
                    prog := ModularProgramV2(
                        soccfg,
                        cfg,
                        modules=[
                            Reset("reset", modules.reset),
                            Pulse("pi2_pulse1", pi2_pulse, block_mode=False),
                            Delay("first_delay", interval - 0.5 * dpulse_len),
                            Repeat("pi_loop", time - 1).add_content(
                                [
                                    Pulse("pi_pulse", pi_pulse, block_mode=False),
                                    SoftDelay("inner_delay", 2 * interval),
                                ]
                            ),
                            Pulse("last_pi_pulse", pi_pulse, block_mode=False),
                            Delay("last_delay", interval + 0.5 * dpulse_len),
                            Pulse(
                                name="pi2_pulse2",
                                cfg=pi2_pulse.with_updates(
                                    phase=pi2_pulse.phase + detune_param
                                ),
                            ),
                            Readout("readout", modules.readout),
                        ],
                        sweep=[("length", length_sweep)],
                    )
                ).acquire(
                    soc,
                    progress=False,
                    round_hook=wrap_earlystop_check(
                        prog,
                        update_hook,
                        earlystop_snr,
                        signal2real_fn=lambda x: rotate2real(x).real,
                        after_check=lambda snr: ax1d.set_title(f"snr = {snr:.1f}"),
                    ),
                    **(acquire_kwargs or {}),
                )

            def update_fn(
                i: int, ctx: TaskState[Any, Any, CPMG_Cfg], time: float
            ) -> None:
                ctx.env.update(time=int(time), length_idx=i)
                ctx.cfg.sweep.length = make_sweep(
                    start=length_ranges[i, 0],
                    stop=length_ranges[i, 1],
                    expts=ctx.cfg.length_expts,
                )

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(length_idxs),),
                    pbar_n=cfg.rounds,
                ).scan("times", times.tolist(), before_each=update_fn),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    times.astype(np.float64),
                    lengths[ctx.env["length_idx"]],
                    cpmg_signal2real(np.asarray(ctx.root_data)),
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (times, lengths, signals)

        return times, lengths, signals

    def analyze(
        self,
        result: Optional[CPMG_Result] = None,
        fit_fringe: bool = True,
        t2r: Optional[float] = None,
        t1: Optional[float] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, lengths, signals2D = result

        real_signals2D = rotate2real(signals2D).real
        norm_signals = cpmg_signal2real(signals2D)

        def is_good_fit(lens, fit_signal, real_signal, t2r, t2err) -> bool:
            fit_ptp = np.ptp(fit_signal)
            residual = real_signal - fit_signal
            smooth_residual = gaussian_filter1d(residual, sigma=1)
            mean_residual = np.mean(np.abs(smooth_residual))
            return (
                t2r < np.max(lens) and t2err < 0.5 * t2r and fit_ptp > 4 * mean_residual
            )

        t2s = np.full(len(times), np.nan, dtype=np.float64)
        t2errs = np.zeros_like(t2s)
        for i, (real_signal, lens) in enumerate(zip(real_signals2D, lengths)):
            if np.any(np.isnan(real_signal)):
                continue

            plens = lens - lens[0]
            if fit_fringe:
                t2, t2err, _, _, fit_signal, _ = fit_decay_fringe(plens, real_signal)
            else:
                t2, t2err, fit_signal, _ = fit_decay(plens, real_signal)

            if is_good_fit(plens, fit_signal, real_signal, t2, t2err):
                t2s[i] = t2
                t2errs[i] = t2err

        if np.all(np.isnan(t2s)):
            raise ValueError("No valid Fitting T2 found. Please check the data.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        assert isinstance(ax1, Axes)
        assert isinstance(ax2, Axes)

        X = np.broadcast_to(times[None, :], norm_signals.T.shape)
        Y = lengths.T
        ax1.pcolormesh(X, Y, norm_signals.T, shading="nearest")
        ax1.set_xlabel("Number of Pi")
        ax1.set_ylabel("Time (us)")

        ax2.errorbar(times, t2s, yerr=t2errs, label=r"$T_{CPMG}$", color="red", fmt=".")
        if t1 is not None:
            ax2.axhline(2 * t1, color="blue", linestyle="--", label=r"$2T_1$")
        if t2r is not None:
            ax2.axhline(t2r, color="green", linestyle="--", label=r"$T_{2r}$")
        ax2.set_xlim(0.3, None)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.set_xlabel("Number of Pi")

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[CPMG_Result] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/cpmg",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        _filepath = Path(filepath)

        times, lengths, signals2D = result
        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        np.savez_compressed(
            _filepath,
            times=times,
            lengths=lengths,
            signals2D=signals2D,
            comment=np.asarray(comment),
        )

        float_times = times.astype(np.float64)
        length_idxs = np.arange(lengths.shape[1])

        # lengths
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_length")),
            x_info={"name": "Number of Pi", "unit": "a.u.", "values": float_times},
            y_info={"name": "Time Index", "unit": "a.u.", "values": length_idxs},
            z_info={"name": "Length", "unit": "s", "values": 1e-6 * lengths.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

        # signals
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_signals")),
            x_info={"name": "Number of pi", "unit": "a.u.", "values": float_times},
            y_info={"name": "Time Index", "unit": "a.u.", "values": length_idxs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> CPMG_Result:
        data = np.load(filepath)
        times = data["times"]
        lengths = data["lengths"]
        signals2D = data["signals2D"]
        comment = None
        if "comment" in data:
            comment = str(data["comment"].tolist())

        times = times.astype(np.int64)
        lengths = lengths.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.last_cfg = CPMG_Cfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (times, lengths, signals2D)

        return times, lengths, signals2D
