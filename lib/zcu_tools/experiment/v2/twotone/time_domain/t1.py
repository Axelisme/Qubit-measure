from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

import zcu_tools.utils.fitting as ft
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
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
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D, LivePlot2DwithLine
from zcu_tools.program.v2 import (
    Delay,
    DelayAuto,
    LoadValue,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_decay, fit_dual_decay
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class T1Result:
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: T1Cfg | T1WithToneCfg | None = None


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # (times, signals)


class T1ModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1SweepCfg(ConfigBase):
    length: SweepCfg | list[float]


class T1Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1ModuleCfg
    sweep: T1SweepCfg


class T1Exp(PersistableExperiment[T1Result, T1Cfg]):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    # times stored as seconds on disk -> scale=US_TO_S
    AXES_SPEC = AxesSpec(
        axes=(Axis("times", "Time", "s", US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=T1Result,
        cfg_type=T1Cfg,
        tag="twotone/ge/t1",
    )

    def _run_non_uniform(
        self,
        soc,
        soccfg,
        cfg: T1Cfg,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> T1Result:
        original_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)

        length_sweep = cfg.sweep.length

        if isinstance(length_sweep, SweepCfg):
            expected_t1 = 0.2 * length_sweep.stop
            y0 = np.exp(-length_sweep.start / expected_t1)
            yN = np.exp(-length_sweep.stop / expected_t1)
            y_seq = np.linspace(y0, yN, length_sweep.expts, endpoint=True)
            lengths = -expected_t1 * np.log(y_seq)
        else:
            lengths = np.asarray(length_sweep)
        length_cycles = np.asarray(
            [int(soccfg.us2cycles(t)) for t in lengths], dtype=np.int32
        )
        length_cycles = np.unique(length_cycles)
        lengths = np.asarray(
            [soccfg.cycles2us(int(cycle)) for cycle in length_cycles], dtype=np.float64
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, T1Cfg],
            update_hook: Callable | None,
        ):
            cfg = ctx.cfg
            modules = cfg.modules

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    LoadValue(
                        "load_t1_delay",
                        values=list(length_cycles),
                        idx_reg="length_idx",
                        val_reg="t1_delay_cycle",
                        auto_compress=False,
                    ),
                    Reset("reset", modules.reset),
                    Pulse("pi_pulse", modules.pi_pulse),
                    DelayAuto("t1_delay", t="t1_delay_cycle"),
                    Readout("readout", modules.readout),
                ],
                sweep=[("length_idx", len(length_cycles))],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot1D("Time (us)", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, t1_signal2real(ctx.root_data)
                ),
            )

        return T1Result(times=lengths, signals=signals, cfg_snapshot=original_cfg)

    def _run_uniform(
        self,
        soc,
        soccfg,
        cfg: T1Cfg,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> T1Result:
        original_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)

        lengths = sweep2array(cfg.sweep.length, "time", {"soccfg": soccfg})

        with LivePlot1D(
            "Time (us)", "Amplitude", segment_kwargs={"title": "T1 relaxation"}
        ) as viewer:

            def measure_fn(
                ctx: TaskState[NDArray[np.complex128], Any, T1Cfg],
                update_hook: Callable[[int, list[NDArray[np.float64]]], None] | None,
            ) -> list[NDArray[np.float64]]:
                modules = ctx.cfg.modules
                length_sweep = ctx.cfg.sweep.length
                assert isinstance(length_sweep, SweepCfg), (
                    "uniform mode requires SweepCfg"
                )
                length_param = sweep2param("length", length_sweep)
                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.reset),
                        Pulse("pi_pulse", modules.pi_pulse),
                        Delay("t1_delay", length_param),
                        Readout("readout", modules.readout),
                    ],
                    sweep=[("length", length_sweep)],
                ).acquire(
                    soc,
                    progress=False,
                    round_hook=update_hook,
                    stop_checkers=[ctx.is_stop],
                    **(acquire_kwargs or {}),
                )

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, t1_signal2real(ctx.root_data)
                ),
            )

        return T1Result(times=lengths, signals=signals, cfg_snapshot=original_cfg)

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: T1Cfg,
        *,
        uniform: bool = True,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> T1Result:
        if uniform:
            return self._run_uniform(soc, soccfg, cfg, acquire_kwargs=acquire_kwargs)
        else:
            return self._run_non_uniform(
                soc, soccfg, cfg, acquire_kwargs=acquire_kwargs
            )

    @retrieve_result
    def analyze(
        self,
        result: T1Result | None = None,
        *,
        dual_exp: bool = False,
        skip: int = 0,
    ) -> tuple[float, float, Figure]:
        assert result is not None, "no result found"

        xs, signals = result.times, result.signals

        xs = xs[skip:]
        signals = signals[skip:]

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


class T1WithToneModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneSweepAxisCfg(ConfigBase):
    length: SweepCfg


class T1WithToneCfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1WithToneModuleCfg
    sweep: T1WithToneSweepAxisCfg


class T1WithToneExp(PersistableExperiment[T1Result, T1WithToneCfg]):
    # times stored as seconds on disk -> scale=US_TO_S
    AXES_SPEC = AxesSpec(
        axes=(Axis("times", "Time", "s", US_TO_S),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=T1Result,
        cfg_type=T1WithToneCfg,
        tag="twotone/ge/t1_with_tone",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: T1WithToneCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> T1Result:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.test_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, T1WithToneCfg],
            update_hook: Callable[[int, list[NDArray[np.float64]]], None] | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            length_sweep = cfg.sweep.length
            length_param = sweep2param("length", length_sweep)
            modules.test_pulse.set_param("length", length_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse(name="pi_pulse", cfg=modules.pi_pulse),
                    Pulse(name="test_pulse", cfg=modules.test_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("length", length_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot1D(
            "Time (us)", "Amplitude", segment_kwargs={"title": "T1 relaxation"}
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, t1_signal2real(ctx.root_data)
                ),
            )

        return T1Result(times=lengths, signals=signals, cfg_snapshot=deepcopy(cfg))

    @retrieve_result
    def analyze(
        self, result: T1Result | None = None, *, dual_exp: bool = False
    ) -> tuple[float, float, Figure]:
        assert result is not None, "no result found"

        xs, signals = result.times, result.signals

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


@dataclass(frozen=True)
class ScanT1WithToneResult:
    values: NDArray[np.float64]
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: ScanT1WithToneCfg | None = None


def t1_with_tone_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class ScanT1WithToneSweepCfg(ConfigBase):
    gain: SweepCfg | list[float]
    length: SweepCfg


class ScanT1WithToneCfg(ProgramV2Cfg, ExpCfgModel):
    modules: T1WithToneModuleCfg
    sweep: ScanT1WithToneSweepCfg


class ScanT1WithToneExp(PersistableExperiment[ScanT1WithToneResult, ScanT1WithToneCfg]):
    # inner = times (result_shape, s on disk -> US_TO_S); outer = values (gain, a.u.)
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("times", "Time", "s", US_TO_S),
            Axis("values", "Readout Gain", "a.u."),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=ScanT1WithToneResult,
        cfg_type=ScanT1WithToneCfg,
        tag="twotone/ge/t1_with_tone_sweep",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: ScanT1WithToneCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> ScanT1WithToneResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.test_pulse.ch},
            allow_array=True,
        )
        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.test_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, ScanT1WithToneCfg],
            update_hook: Callable[[int, list[NDArray[np.float64]]], None] | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            length_sweep = cfg.sweep.length
            length_param = sweep2param("length", length_sweep)
            modules.test_pulse.set_param("length", length_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("pi_pulse", modules.pi_pulse),
                    Pulse("test_pulse", modules.test_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("length", length_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot2DwithLine(
            "gain", "Time (us)", line_axis=1, num_lines=5
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths),),
                    pbar_n=cfg.rounds,
                ).scan(
                    "gain",
                    gains.tolist(),
                    before_each=lambda _, ctx, gain: (
                        ctx.cfg.modules.test_pulse.set_param("gain", gain)
                    ),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    gains, lengths, t1_with_tone_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        return ScanT1WithToneResult(
            values=gains, times=lengths, signals=signals, cfg_snapshot=deepcopy(cfg)
        )

    @retrieve_result
    def analyze(
        self, result: ScanT1WithToneResult | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], Figure]:
        assert result is not None, "no result found"

        gains, ts, signals = result.values, result.times, result.signals

        signals: NDArray[np.complex128] = gaussian_filter(signals, sigma=1)  # type: ignore
        real_signals = t1_with_tone_signal2real(signals)

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
