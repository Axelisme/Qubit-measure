from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typing_extensions import NotRequired

import zcu_tools.utils.fitting as ft
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2D
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay, fit_dual_decay
from zcu_tools.utils.process import rotate2real

# (times, signals)
T1ResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # (times, signals)


class T1TaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1Experiment(AbsExperiment):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    def run(self, soc, soccfg, cfg: T1TaskConfig) -> T1ResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        ts = sweep2array(cfg["sweep"]["length"])

        with LivePlotter1D(
            "Time (us)", "Amplitude", segment_kwargs={"title": "T1 relaxation"}
        ) as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                                Delay(
                                    name="t1_delay",
                                    delay=sweep2param(
                                        "length", ctx.cfg["sweep"]["length"]
                                    ),
                                ),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(ts),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(ts, t1_signal2real(ctx.data)),
            )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, signals)

        return ts, signals

    def analyze(
        self, result: Optional[T1ResultType] = None, *, dual_exp: bool = False
    ) -> Tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        real_signals = rotate2real(signals).real

        if dual_exp:
            t1, t1err, t1b, t1berr, y_fit, (pOpt, _) = fit_dual_decay(xs, real_signals)
        else:
            t1, t1err, y_fit, (pOpt, _) = fit_decay(xs, real_signals)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(xs, y_fit, label="fit")
        t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
        if dual_exp:
            t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us"
            ax.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
            ax.set_title(f"T1 = {t1_str}, T1b = {t1b_str}", fontsize=15)
        else:
            ax.set_title(f"T1 = {t1_str}", fontsize=15)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return t1, t1err, fig

    def save(
        self,
        filepath: str,
        result: Optional[T1ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )


class T1WithToneTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: T1WithToneTaskConfig) -> T1ResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        Pulse.set_param(
            cfg["test_pulse"], "length", sweep2param("length", cfg["sweep"]["length"])
        )

        ts = sweep2array(cfg["sweep"]["length"])

        with LivePlotter1D(
            "Time (us)", "Amplitude", segment_kwargs={"title": "T1 relaxation"}
        ) as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", cfg.get("reset", {"type": "none"})),
                                Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                                Pulse(name="test_pulse", cfg=cfg["test_pulse"]),
                                Readout("readout", cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(ts),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(ts, t1_signal2real(ctx.data)),
            )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, signals)

        return ts, signals

    def analyze(
        self, result: Optional[T1ResultType] = None, *, dual_exp: bool = False
    ) -> Tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        real_signals = t1_signal2real(signals)

        if dual_exp:
            t1, t1err, t1b, t1berr, y_fit, (pOpt, _) = fit_dual_decay(xs, real_signals)
        else:
            t1, t1err, y_fit, (pOpt, _) = fit_decay(xs, real_signals)

        t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
        if dual_exp:
            t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us"

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(xs, y_fit, label="fit")
        if dual_exp:
            ax.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
            ax.set_title(f"T1 = {t1_str}, T1b = {t1b_str}", fontsize=15)
        else:
            ax.set_title(f"T1 = {t1_str}", fontsize=15)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return t1, t1err, fig

    def save(
        self,
        filepath: str,
        result: Optional[T1ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1_with_tone",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )


# (values, times, signals)
T1SweepResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def t1_sweep_tone_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    real_signals = np.full_like(signals, np.nan, dtype=np.float64)
    for i in range(signals.shape[0]):
        if np.all(np.isnan(signals[i, :])):
            continue
        real_signals[i, :] = rotate2real(signals[i, :]).real

        min_val = np.nanmin(real_signals[i, :])
        max_val = np.nanmax(real_signals[i, :])
        real_signals[i, :] = (real_signals[i, :] - min_val) / (
            max_val - min_val + 1e-12
        )

    return real_signals


class T1WithToneSweepTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweepExperiment(AbsExperiment):
    SWEEP_MAP = {
        "gain": {"name": "Gain (a.u.)", "param_key": "gain"},
        "freq": {"name": "Frequency (MHz)", "param_key": "freq"},
    }

    def run(self, soc, soccfg, cfg: T1WithToneSweepTaskConfig) -> T1SweepResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        assert isinstance(cfg["sweep"], dict)
        len_sweep = cfg["sweep"].pop("length")

        # extract sweep parameters
        x_key = list(cfg["sweep"].keys())[0]
        if x_key not in self.SWEEP_MAP:
            raise ValueError(f"Unsupported sweep key: {x_key}")
        x_info = self.SWEEP_MAP[x_key]

        cfg["sweep"] = {
            x_info["param_key"]: cfg["sweep"][x_key],
            "length": len_sweep,
        }

        Pulse.set_param(cfg["test_pulse"], "length", sweep2param("length", len_sweep))
        Pulse.set_param(
            cfg["test_pulse"],
            x_info["param_key"],
            sweep2param(x_info["param_key"], cfg["sweep"][x_key]),
        )

        values = sweep2array(cfg["sweep"][x_key])  # predicted
        ts = sweep2array(cfg["sweep"]["length"])  # predicted times

        with LivePlotter2D(x_info["name"], "Time (us)") as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", cfg.get("reset", {"type": "none"})),
                                Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                                Pulse(name="test_pulse", cfg=cfg["test_pulse"]),
                                Readout("readout", cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(values), len(ts)),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    values, ts, t1_sweep_tone_signal2real(ctx.data)
                ),
            )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (values, ts, signals)

        return values, ts, signals

    def analyze(
        self, result: Optional[T1SweepResultType] = None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, ts, signals = result

        signals: NDArray[np.complex128] = gaussian_filter(signals, sigma=1)  # type: ignore
        real_signals = t1_sweep_tone_signal2real(signals)

        t1s = np.full(len(values), np.nan, dtype=np.float64)
        t1errs = np.zeros_like(t1s)

        for i in range(len(values)):
            real_signal = real_signals[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signal)):
                continue

            t1, t1err, *_ = fit_decay(ts, real_signal)

            if t1err > 0.3 * t1:
                continue

            t1s[i] = t1
            t1errs[i] = t1err

        if np.all(np.isnan(t1s)):
            raise ValueError("No valid Fitting T1 found. Please check the data.")

        valid_idxs = ~np.isnan(t1s)
        valid_values = values[valid_idxs]
        t1s = t1s[valid_idxs]
        t1errs = t1errs[valid_idxs]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        assert isinstance(fig, Figure)
        assert isinstance(ax1, Axes)
        assert isinstance(ax2, Axes)

        fig.suptitle("T1 while readout")

        ax1.set_ylabel("T1 over sweep value")
        ax1.imshow(
            real_signals.T,
            aspect="auto",
            extent=(values[0], values[-1], t1s[0], t1s[-1]),
            origin="lower",
        )
        ax2.errorbar(
            valid_values, t1s, yerr=t1errs, label="Fitting T1", elinewidth=1, capsize=1
        )
        ax2.set_xlabel("Readout Gain (a.u.)")
        ax2.set_ylabel("T1 (us)")
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(values[0], values[-1])
        ax2.grid()

        return valid_values, t1s, t1errs, fig

    def save(
        self,
        filepath: str,
        result: Optional[T1SweepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1_with_tone_sweep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Sweep Value", "unit": "a.u.", "values": values},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
