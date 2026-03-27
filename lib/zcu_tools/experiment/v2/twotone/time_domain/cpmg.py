from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    Union,
)

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
    SoftDelay,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.process import rotate2real


def cpmg_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    real_signals = rotate2real(signals).real
    max_vals = np.max(real_signals, axis=1, keepdims=True)
    min_vals = np.min(real_signals, axis=1, keepdims=True)
    return (real_signals - min_vals) / (max_vals - min_vals)


# (times, lengths(time x length), signals)
CPMG_Result: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class CPMG_ModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi2_pulse: PulseCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class CPMG_SweepCfg(TypedDict, closed=True):
    times: Union[SweepCfg, NDArray, list]
    length: SweepCfg


class CPMG_Cfg(ModularProgramCfg, TaskCfg):
    modules: CPMG_ModuleCfg
    sweep: CPMG_SweepCfg


class CPMG_Exp(AbsExperiment[CPMG_Result, CPMG_Cfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> CPMG_Result:
        _cfg = check_type(deepcopy(cfg), CPMG_Cfg)
        modules = _cfg["modules"]

        pi2_pulse = modules["pi2_pulse"]
        pi_pulse = modules["pi_pulse"]

        length_sweep = _cfg["sweep"]["length"]
        time_sweep: SweepCfg = _cfg["sweep"].pop("times")  # type: ignore

        # TODO: convert times and length sweeps to arrays in different time
        times = sweep2array(time_sweep, allow_array=True)
        lengths = np.array(
            [
                sweep2array(
                    length_sweep, "time", {"soccfg": soccfg, "scaler": 1 / (2 * t)}
                )
                for t in times
            ]
        )
        length_idxs = np.arange(length_sweep["expts"])

        cpmg_spans = sweep2param("length", _cfg["sweep"]["length"])

        if np.min(times) <= 0:
            raise ValueError("times should be larger than 0")

        min_interval = np.min(lengths) / np.max(times)
        if min_interval < 0.5 * (  # the first and last delays are half of the interval
            pi2_pulse["waveform"]["length"] + pi_pulse["waveform"]["length"]
        ):
            raise ValueError(
                "The interval is too short to measure the CPMG signal",
                f"min_interval: {min_interval:.2g}, ",
                f"pi2_pulse_length: {pi2_pulse['waveform']['length']:.2g}, ",
                f"pi_pulse_length: {pi_pulse['waveform']['length']:.2g}",
            )

        with LivePlotter2DwithLine(
            "Number of Pi", "Time (us)", line_axis=1, num_lines=2, title="CPMG"
        ) as viewer:

            def measure_fn(ctx: TaskState, update_hook: Callable[[int, Any], None]):
                cfg = ctx.cfg
                modules = cfg["modules"]
                pi2_pulse = modules["pi2_pulse"]
                pi_pulse = modules["pi_pulse"]
                dpulse_len = (
                    pi_pulse["waveform"]["length"] - pi2_pulse["waveform"]["length"]
                )

                interval = cpmg_spans / (2 * ctx.env["time"])

                return ModularProgramV2(
                    soccfg,
                    cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        Pulse(
                            "pi2_pulse1",
                            pi2_pulse,
                            pulse_name="pi2_pulse",
                            block_mode=False,
                        ),
                        Delay("first_cpmg_delay", interval - 0.5 * dpulse_len),
                        Repeat(
                            name="cpmg_pi_loop",
                            n=ctx.env["time"] - 1,
                            sub_module=[
                                Pulse(
                                    "pi_pulse",
                                    pi_pulse,
                                    pulse_name="pi_pulse",
                                    block_mode=False,
                                ),
                                SoftDelay("inner_cpmg_delay", 2 * interval),
                            ],
                        ),
                        Pulse(
                            "last_pi_pulse",
                            pi_pulse,
                            pulse_name="pi_pulse",
                            block_mode=False,
                        ),
                        Delay("last_cpmg_delay", interval + 0.5 * dpulse_len),
                        Pulse("pi2_pulse2", pi2_pulse, pulse_name="pi2_pulse"),
                        Readout("readout", modules["readout"]),
                    ],
                ).acquire(soc, progress=False, callback=update_hook)

            def update_fn(i, ctx: TaskState, time):
                ctx.env.update(time=int(time))

            signals = run_task(
                task=Task(measure_fn=measure_fn, result_shape=(len(lengths),)).scan(
                    "times", times.tolist(), before_each=update_fn
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    times,
                    length_idxs.astype(np.float64),
                    cpmg_signal2real(np.asarray(ctx.root_data)),
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (times, lengths, signals)

        return times, lengths, signals

    def analyze(self, result: Optional[CPMG_Result] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, lengths, signals2D = result

        real_signals2D = rotate2real(signals2D).real
        norm_signals = cpmg_signal2real(signals2D)

        def is_good_fit(fit_signal, real_signal) -> bool:
            real_ptp = np.ptp(real_signal)
            fit_ptp = np.ptp(fit_signal)
            residual = real_signal - fit_signal
            smooth_residual = gaussian_filter1d(residual, sigma=1)
            return fit_ptp > 0.5 * real_ptp and fit_ptp > np.mean(
                np.abs(smooth_residual)
            )

        prev_pOpt = None
        t2s = np.full(len(times), np.nan, dtype=np.float64)
        t2errs = np.zeros_like(t2s)
        for i, (real_signal, lens) in enumerate(zip(real_signals2D, lengths)):
            if np.any(np.isnan(real_signal)):
                continue

            t2r, t2err, fit_signal, (pOpt, _) = fit_decay(
                lens, real_signal, fit_params=prev_pOpt
            )

            if is_good_fit(fit_signal, real_signal):
                prev_pOpt = pOpt

                t2s[i] = t2r
                t2errs[i] = t2err
            else:
                prev_pOpt = None

        if np.all(np.isnan(t2s)):
            raise ValueError("No valid Fitting T2 found. Please check the data.")

        fig, ax = plt.subplots()
        assert isinstance(ax, Axes)

        X = np.broadcast_to(times[None, :], norm_signals.T.shape)
        Y = lengths.T
        ax.pcolormesh(X, Y, norm_signals.T, shading="nearest")
        ax.errorbar(times, t2s, yerr=t2errs, label="Fitting T2", color="red")
        ax.legend()
        ax.set_ylabel("Time (us)")
        ax.set_xlabel("Number of Pi")

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

        np.savez_compressed(
            _filepath, times=times, lengths=lengths, signals2D=signals2D
        )

        length_idxs = np.arange(lengths.shape[1])

        # lengths
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_length")),
            x_info={"name": "Number of Pi", "unit": "a.u.", "values": times},
            y_info={"name": "Time Index", "unit": "a.u.", "values": length_idxs},
            z_info={"name": "Length", "unit": "s", "values": lengths.T * 1e-6},
            comment=comment,
            tag=tag,
            **kwargs,
        )

        # signals
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_signals")),
            x_info={"name": "Number of pi", "unit": "a.u.", "values": times},
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

        lengths = lengths * 1e6  # s -> us
        signals2D = signals2D.T  # transpose back

        times = times.astype(np.float64)
        lengths = lengths.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (times, lengths, signals2D)

        return times, lengths, signals2D
