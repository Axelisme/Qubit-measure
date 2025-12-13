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
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task, SoftTask
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2DwithLine
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
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_decay, fit_dual_decay
from zcu_tools.utils.process import rotate2real
from zcu_tools.utils.math import vdc_permutation

# (times, signals)
T1ResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # (times, signals)


class T1TaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg

    t1_delay: NotRequired[float]


class T1Experiment(AbsExperiment):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    def _run_non_uniform(self, soc, soccfg, cfg: T1TaskConfig) -> T1ResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"]["length"]
        del cfg["sweep"]

        if isinstance(len_sweep, dict):
            ts = (
                np.linspace(
                    len_sweep["start"] ** (1 / 3),
                    len_sweep["stop"] ** (1 / 3),
                    len_sweep["expts"],
                )
                ** 3
            )
        else:
            ts = np.asarray(len_sweep)
        ts = np.array([soccfg.cycles2us(soccfg.us2cycles(t)) for t in ts])

        def measure_fn(ctx, update_hook):
            rounds = ctx.cfg.pop("rounds", 1)
            ctx.cfg["rounds"] = 1

            acc_signals = np.zeros_like(ts, dtype=np.complex128)
            for ir in range(rounds):
                for i, t1_delay in enumerate(ts):
                    raw_i = ModularProgramV2(
                        soccfg,
                        ctx.cfg,
                        modules=[
                            Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                            Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                            Delay("t1_delay", delay=t1_delay),
                            Readout("readout", ctx.cfg["readout"]),
                        ],
                    ).acquire(soc, progress=False)

                    signal_i = raw_i[0][0].dot([1, 1j])

                    acc_signals[i] += signal_i

                update_hook(ir, acc_signals / (ir + 1))

            return acc_signals / rounds

        with LivePlotter1D(
            "Time (us)", "Amplitude", segment_kwargs={"title": "T1 relaxation"}
        ) as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw,
                    result_shape=(len(ts),),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(ts, t1_signal2real(ctx.data)),
            )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, signals)

        return ts, signals

    def _run_uniform(self, soc, soccfg, cfg: T1TaskConfig) -> T1ResultType:
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

    def run(self, soc, soccfg, cfg: T1TaskConfig, uniform: bool = True) -> T1ResultType:
        if uniform:
            return self._run_uniform(soc, soccfg, cfg)
        else:
            return self._run_non_uniform(soc, soccfg, cfg)

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

    def load(self, filepath: str, **kwargs) -> T1ResultType:
        signals, Ts, _ = load_data(filepath, **kwargs)
        assert Ts is not None
        assert len(Ts.shape) == 1 and len(signals.shape) == 1
        assert Ts.shape == signals.shape

        Ts = Ts * 1e6  # s -> us

        Ts = Ts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (Ts, signals)

        return Ts, signals


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

    def load(self, filepath: str, **kwargs) -> T1ResultType:
        signals, Ts, _ = load_data(filepath, **kwargs)
        assert Ts is not None
        assert len(Ts.shape) == 1 and len(signals.shape) == 1
        assert Ts.shape == signals.shape

        Ts = Ts * 1e6  # s -> us

        Ts = Ts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (Ts, signals)

        return Ts, signals


# (values, times, signals)
T1SweepResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def t1_sweep_tone_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class T1WithToneSweepTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweepExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: T1WithToneSweepTaskConfig) -> T1SweepResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        gain_sweep = cfg["sweep"]["gain"]
        cfg["sweep"] = {"length": cfg["sweep"]["length"]}

        Pulse.set_param(
            cfg["test_pulse"], "length", sweep2param("length", cfg["sweep"]["length"])
        )

        gains = sweep2array(gain_sweep, allow_array=True)  # predicted
        ts = sweep2array(cfg["sweep"]["length"])  # predicted times

        with LivePlotter2DwithLine(
            "gain", "Time (us)", line_axis=1, num_lines=5
        ) as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="gain",
                    sweep_values=gains.tolist(),
                    update_cfg_fn=lambda _, ctx, gain: Pulse.set_param(
                        ctx.cfg["test_pulse"], "gain", gain
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset(
                                        "reset", ctx.cfg.get("reset", {"type": "none"})
                                    ),
                                    Pulse(name="pi_pulse", cfg=ctx.cfg["pi_pulse"]),
                                    Pulse(name="test_pulse", cfg=ctx.cfg["test_pulse"]),
                                    Readout("readout", cfg=ctx.cfg["readout"]),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        ),
                        result_shape=(len(ts),),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    gains, ts, t1_sweep_tone_signal2real(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (gains, ts, signals)

        return gains, ts, signals

    def analyze(
        self, result: Optional[T1SweepResultType] = None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, ts, signals = result

        signals: NDArray[np.complex128] = gaussian_filter(signals, sigma=1)  # type: ignore
        real_signals = t1_sweep_tone_signal2real(signals)

        t1s = np.full(len(gains), np.nan, dtype=np.float64)
        t1errs = np.zeros_like(t1s)

        for i in range(len(gains)):
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
        gains = gains[valid_idxs]
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
            extent=(gains[0], gains[-1], t1s[0], t1s[-1]),
            origin="lower",
        )
        ax2.errorbar(
            gains, t1s, yerr=t1errs, label="Fitting T1", elinewidth=1, capsize=1
        )
        ax2.set_xlabel("Readout Gain (a.u.)")
        ax2.set_ylabel("T1 (us)")
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(gains[0], gains[-1])
        ax2.grid()

        return gains, t1s, t1errs, fig

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

        gains, Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Readout Gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> T1SweepResultType:
        signals, gains, Ts = load_data(filepath, **kwargs)
        assert gains is not None and Ts is not None
        assert len(gains.shape) == 1 and len(Ts.shape) == 1
        assert signals.shape == (len(Ts), len(gains))

        Ts = Ts * 1e6  # s -> us
        signals = signals.T  # transpose back

        gains = gains.astype(np.float64)
        Ts = Ts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (gains, Ts, signals)

        return gains, Ts, signals
