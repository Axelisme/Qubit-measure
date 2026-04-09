from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from typeguard import check_type
from typing_extensions import Any, NotRequired, Optional, TypeAlias, TypedDict, Union

import zcu_tools.utils.fitting as ft
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.utils import round_zcu_time, sweep2array
from zcu_tools.liveplot import LivePlot1D, LivePlot2DwithLine
from zcu_tools.program import SweepCfg
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

# (times, signals)
T1Result: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # (times, signals)


class T1ModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1Cfg(ModularProgramCfg, TaskCfg):
    modules: T1ModuleCfg
    sweep: dict[str, Union[SweepCfg, NDArray[np.float64]]]


class T1Exp(AbsExperiment[T1Result, T1Cfg]):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    def _run_non_uniform(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> T1Result:
        _cfg = check_type(deepcopy(cfg), T1Cfg)

        length_sweep: Union[SweepCfg, NDArray[np.float64]] = _cfg["sweep"]["length"]  # type: ignore

        if isinstance(length_sweep, dict):
            lengths = (
                np.linspace(
                    length_sweep["start"] ** (1 / 1.3),
                    length_sweep["stop"] ** (1 / 1.3),
                    length_sweep["expts"],
                )
                ** 1.3
            )
        else:
            lengths = np.asarray(length_sweep)
        lengths = round_zcu_time(lengths, soccfg)
        lengths = np.unique(lengths)

        def measure_fn(ctx, update_hook):
            rounds = ctx.cfg.pop("rounds", 1)
            ctx.cfg["rounds"] = 1
            modules = ctx.cfg["modules"]

            acc_signals = np.zeros_like(lengths, dtype=np.complex128)
            for ir in range(rounds):
                for i, t1_delay in enumerate(lengths):
                    raw_i = ModularProgramV2(
                        soccfg,
                        ctx.cfg,
                        modules=[
                            Reset("reset", modules.get("reset")),
                            Pulse("pi_pulse", modules["pi_pulse"]),
                            Delay("t1_delay", delay=t1_delay),
                            Readout("readout", modules["readout"]),
                        ],
                    ).acquire(soc, progress=False, **(acquire_kwargs or {}))

                    acc_signals[i] += raw_i[0][0].dot([1, 1j])

                update_hook(ir, acc_signals / (ir + 1))

            return acc_signals / rounds

        with LivePlot1D(
            "Time (us)", "Amplitude", segment_kwargs={"title": "T1 relaxation"}
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw,
                    result_shape=(len(lengths),),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, t1_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (lengths, signals)

        return lengths, signals

    def _run_uniform(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> T1Result:
        _cfg = check_type(deepcopy(cfg), T1Cfg)

        lengths = sweep2array(_cfg["sweep"]["length"], "time", {"soccfg": soccfg})

        with LivePlot1D(
            "Time (us)", "Amplitude", segment_kwargs={"title": "T1 relaxation"}
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: (
                        (modules := ctx.cfg["modules"])
                        and (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset("reset", modules.get("reset")),
                                    Pulse("pi_pulse", modules["pi_pulse"]),
                                    Delay(
                                        name="t1_delay",
                                        delay=sweep2param(
                                            "length", ctx.cfg["sweep"]["length"]
                                        ),
                                    ),
                                    Readout("readout", modules["readout"]),
                                ],
                                sweep=[("length", ctx.cfg["sweep"]["length"])],
                            ).acquire(
                                soc,
                                progress=False,
                                callback=update_hook,
                                **(acquire_kwargs or {}),
                            )
                        )
                    ),
                    result_shape=(len(lengths),),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, t1_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (lengths, signals)

        return lengths, signals

    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        uniform: bool = True,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> T1Result:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        if uniform:
            return self._run_uniform(soc, soccfg, cfg, acquire_kwargs=acquire_kwargs)
        else:
            return self._run_non_uniform(
                soc, soccfg, cfg, acquire_kwargs=acquire_kwargs
            )

    def analyze(
        self, result: Optional[T1Result] = None, *, dual_exp: bool = False
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        real_signals = rotate2real(signals).real

        if dual_exp:
            t1, t1err, t1b, t1berr, y_fit, (pOpt, _) = fit_dual_decay(xs, real_signals)
        else:
            t1, t1err, y_fit, (pOpt, _) = fit_decay(xs, real_signals)
            t1b = 0.0
            t1berr = 0.0

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(xs, real_signals, label="data", ls="-", marker="o", markersize=5)
        ax.plot(xs, y_fit, label="fit", c="orange", zorder=1)

        t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
        if dual_exp:
            t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us"
            ax.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
            title = f"$T_1$ = {t1_str}, " + r"$T_{1b}$ = " + f"{t1b_str}"
        else:
            title = f"$T_1$ = {t1_str}"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Delay Time (us)", fontsize=14)
        ax.set_ylabel("Signal (a.u.)", fontsize=14)
        ax.legend(loc="upper right")
        ax.grid(True)

        fig.tight_layout()

        return t1, t1err, fig

    def save(
        self,
        filepath: str,
        result: Optional[T1Result] = None,
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

    def load(self, filepath: str, **kwargs) -> T1Result:
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


class T1WithToneModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneCfg(ModularProgramCfg, TaskCfg):
    modules: T1WithToneModuleCfg
    sweep: dict[str, SweepCfg]


class T1WithToneExp(AbsExperiment[T1Result, T1WithToneCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> T1Result:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        _cfg = check_type(deepcopy(cfg), T1WithToneCfg)
        modules = _cfg["modules"]

        lengths = sweep2array(
            _cfg["sweep"]["length"],
            "time",
            {"soccfg": soccfg, "gen_ch": modules["test_pulse"]["ch"]},
        )

        length_param = sweep2param("length", _cfg["sweep"]["length"])
        Pulse.set_param(modules["test_pulse"], "length", length_param)

        with LivePlot1D(
            "Time (us)", "Amplitude", segment_kwargs={"title": "T1 relaxation"}
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: ModularProgramV2(
                        soccfg,
                        ctx.cfg,
                        modules=[
                            Reset("reset", modules.get("reset")),
                            Pulse(name="pi_pulse", cfg=modules["pi_pulse"]),
                            Pulse(name="test_pulse", cfg=modules["test_pulse"]),
                            Readout("readout", modules["readout"]),
                        ],
                        sweep=[("length", ctx.cfg["sweep"]["length"])],
                    ).acquire(
                        soc,
                        progress=False,
                        callback=update_hook,
                        **(acquire_kwargs or {}),
                    ),
                    result_shape=(len(lengths),),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, t1_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (lengths, signals)

        return lengths, signals

    def analyze(
        self, result: Optional[T1Result] = None, *, dual_exp: bool = False
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        real_signals = t1_signal2real(signals)

        if dual_exp:
            t1b, t1berr, t1, t1err, y_fit, (pOpt, _) = fit_dual_decay(xs, real_signals)
        else:
            t1, t1err, y_fit, (pOpt, _) = fit_decay(xs, real_signals)
            t1b = 0.0
            t1berr = 0.0

        t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
        if dual_exp:
            t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us"
        else:
            t1b_str = "N/A"

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
        result: Optional[T1Result] = None,
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

    def load(self, filepath: str, **kwargs) -> T1Result:
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
T1WithToneSweepResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def t1_sweep_tone_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class T1WithToneSweepCfg(ModularProgramCfg, TaskCfg):
    modules: T1WithToneModuleCfg
    sweep: dict[str, SweepCfg]


class T1WithToneSweepExp(AbsExperiment[T1WithToneSweepResult, T1WithToneSweepCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> T1WithToneSweepResult:
        _cfg = check_type(deepcopy(cfg), T1WithToneSweepCfg)
        modules = _cfg["modules"]

        gain_sweep: Union[SweepCfg, NDArray[np.float64]] = _cfg["sweep"]["gain"]  # type: ignore

        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules["test_pulse"]["ch"]},
            allow_array=True,
        )
        lengths = sweep2array(
            _cfg["sweep"]["length"],
            "time",
            {"soccfg": soccfg, "gen_ch": modules["test_pulse"]["ch"]},
        )

        length_param = sweep2param("length", _cfg["sweep"]["length"])
        Pulse.set_param(modules["test_pulse"], "length", length_param)

        with LivePlot2DwithLine(
            "gain", "Time (us)", line_axis=1, num_lines=5
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: ModularProgramV2(
                        soccfg,
                        ctx.cfg,
                        modules=[
                            Reset(
                                "reset",
                                ctx.cfg["modules"].get("reset"),
                            ),
                            Pulse(name="pi_pulse", cfg=ctx.cfg["modules"]["pi_pulse"]),
                            Pulse(
                                name="test_pulse",
                                cfg=ctx.cfg["modules"]["test_pulse"],
                            ),
                            Readout("readout", cfg=ctx.cfg["modules"]["readout"]),
                        ],
                        sweep=[("length", ctx.cfg["sweep"]["length"])],
                    ).acquire(
                        soc,
                        progress=False,
                        callback=update_hook,
                        **(acquire_kwargs or {}),
                    ),
                    result_shape=(len(lengths),),
                ).scan(
                    "gain",
                    gains.tolist(),
                    before_each=lambda _, ctx, gain: Pulse.set_param(
                        ctx.cfg["modules"]["test_pulse"], "gain", gain
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    gains, lengths, t1_sweep_tone_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (gains, lengths, signals)

        return gains, lengths, signals

    def analyze(
        self, result: Optional[T1WithToneSweepResult] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], Figure]:
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
        result: Optional[T1WithToneSweepResult] = None,
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

    def load(self, filepath: str, **kwargs) -> T1WithToneSweepResult:
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
