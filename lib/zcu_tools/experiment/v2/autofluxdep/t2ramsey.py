from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Callable, NotRequired, Optional, TypedDict, cast

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState
from zcu_tools.experiment.v2.utils import sweep2array, wrap_earlystop_check
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.notebook.utils import make_comment, make_sweep
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
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_decay_fringe
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.process import rotate2real

from .executor import FluxDepInfoDict, MeasurementTask, T_RootResult


def t2ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    if np.any(np.isnan(signals)):
        return signals.real

    real_signals = rotate2real(signals).real
    max_val = np.max(real_signals)
    min_val = np.min(real_signals)
    init_val = real_signals[0]
    real_signals = (real_signals - min_val) / (max_val - min_val + 1e-12)
    if init_val < 0.5 * (max_val + min_val):
        real_signals = 1.0 - real_signals
    return real_signals


def t2ramsey_fluxdep_signal2real(
    signals: NDArray[np.complex128],
) -> NDArray[np.float64]:
    return np.array(list(map(t2ramsey_signal2real, signals)), dtype=np.float64)


class T2RamseyModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class T2RamseyCfgTemplate(ModularProgramCfg, TaskCfg):
    modules: T2RamseyModuleCfg
    sweep_range: tuple[float, float]


class T2RamseyCfg(ModularProgramCfg, TaskCfg):
    modules: T2RamseyModuleCfg
    dev: dict[str, DeviceInfo]
    sweep: dict[str, SweepCfg]
    activate_detune: float


class T2RamseyResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    length: NDArray[np.float64]
    t2r: NDArray[np.float64]
    t2r_err: NDArray[np.float64]
    t2r_detune: NDArray[np.float64]
    t2r_detune_err: NDArray[np.float64]
    success: NDArray[np.bool_]


class T2RamseyPlotDict(TypedDict, closed=True):
    t2r: LivePlot1D
    t2r_curve: LivePlot1D


class T2RamseyTask(MeasurementTask[T2RamseyResult, T_RootResult, T2RamseyPlotDict]):
    def __init__(
        self,
        num_expts: int,
        detune_ratio: float,
        cfg_maker: Callable[
            [TaskState[T2RamseyResult, T_RootResult], ModuleLibrary],
            Optional[dict[str, Any]],
        ],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.num_expts = num_expts
        self.detune_ratio = detune_ratio
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        def measure_ramsey_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: T2RamseyCfg = cast(T2RamseyCfg, ctx.cfg)
            modules = cfg["modules"]

            assert update_hook is not None

            detune = cfg["activate_detune"]

            length_sweep = cfg["sweep"]["length"]
            length_param = sweep2param("length", length_sweep)

            prog = ModularProgramV2(
                ctx.env["soccfg"],
                ctx.cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse(name="pi2_pulse1", cfg=modules["pi2_pulse"]),
                    Delay(name="t2r_delay", delay=length_param),
                    Pulse(
                        name="pi2_pulse2",
                        cfg=modules["pi2_pulse"].with_updates(
                            phase=modules["pi2_pulse"].phase
                            + 360 * detune * length_param
                        ),
                    ),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("length", length_sweep)],
            )
            return prog.acquire(
                ctx.env["soc"],
                progress=False,
                callback=wrap_earlystop_check(
                    prog,
                    update_hook,
                    self.earlystop_snr,
                    signal2real_fn=t2ramsey_signal2real,
                ),
            )

        self.lengths = np.linspace(0, 1, num_expts)
        self.task = Task[T_RootResult, list[NDArray[np.float64]]](
            measure_fn=measure_ramsey_fn,
            result_shape=(num_expts,),
        )

    def init(
        self, ctx: TaskState[T2RamseyResult, T_RootResult], dynamic_pbar=False
    ) -> None:
        self.init_cfg = deepcopy(ctx.cfg)
        self.task.init(ctx.child("raw_signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[T2RamseyResult, T_RootResult]) -> None:
        info: FluxDepInfoDict = ctx.env["info"]

        cfg_temp = self.cfg_maker(ctx, ctx.env["ml"])

        if cfg_temp is None:
            return  # skip this task

        cfg_temp = check_type(cfg_temp, T2RamseyCfgTemplate)

        len_sweep = make_sweep(*cfg_temp["sweep_range"], self.num_expts)
        self.lengths = sweep2array(len_sweep, "time", {"soccfg": ctx.env["soccfg"]})

        cfg_temp = dict(cfg_temp)
        del cfg_temp["sweep_range"]  # type: ignore
        deepupdate(
            cfg_temp,
            {"dev": ctx.cfg["dev"], "sweep": {"length": len_sweep}},
            behavior="force",
        )
        cfg_temp["activate_detune"] = self.detune_ratio / len_sweep["step"]
        cfg = check_type(cfg_temp, T2RamseyCfg)

        self.task.run(ctx.child("raw_signals", new_cfg=cfg))  # type: ignore

        raw_signals = ctx.value["raw_signals"]
        assert isinstance(raw_signals, np.ndarray)

        real_signals = t2ramsey_signal2real(raw_signals)

        t2r, t2r_err, t2r_detune, t2r_detune_err, fit_signals, _ = fit_decay_fringe(
            self.lengths, real_signals
        )
        t2r_detune = t2r_detune - cfg["activate_detune"]

        success = True
        mean_err = np.mean(np.abs(real_signals - fit_signals))
        if t2r > 2 * np.max(self.lengths) or mean_err > 0.1 * np.ptp(fit_signals):
            t2r = np.nan
            t2r_err = np.nan
            t2r_detune = np.nan
            t2r_detune_err = np.nan
            success = False

        if success:
            info["t2r"] = t2r
            info["t2r_detune"] = t2r_detune
            info["smooth_t2r"] = 0.5 * (info.last.get("smooth_t2r", t2r) + t2r)

        with MinIntervalFunc.force_execute():
            ctx.set_value(
                T2RamseyResult(
                    raw_signals=raw_signals,
                    length=self.lengths.copy(),
                    t2r=np.array(t2r),
                    t2r_err=np.array(t2r_err),
                    t2r_detune=np.array(t2r_detune),
                    t2r_detune_err=np.array(t2r_detune_err),
                    success=np.array(success),
                )
            )

    def get_default_result(self) -> T2RamseyResult:
        return T2RamseyResult(
            raw_signals=self.task.get_default_result(),
            length=self.lengths.copy(),
            t2r=np.array(np.nan),
            t2r_err=np.array(np.nan),
            t2r_detune=np.array(np.nan),
            t2r_detune_err=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def num_axes(self) -> dict[str, int]:
        return dict(t2r=1, t2r_curve=1)

    def make_plotter(self, name, axs) -> T2RamseyPlotDict:
        return T2RamseyPlotDict(
            t2r=LivePlot1D(
                "Flux device value",
                "T2Ramsey (us)",
                existed_axes=[axs["t2r"]],
                segment_kwargs=dict(
                    title=name + "(t2r)", line_kwargs=[dict(linestyle="None")]
                ),
            ),
            t2r_curve=LivePlot1D(
                "Signal",
                "Time (us)",
                existed_axes=[axs["t2r_curve"]],
                segment_kwargs=dict(title=name + "(t2r curve)"),
            ),
        )

    def update_plotter(self, plotters, ctx: TaskState, signals: T2RamseyResult) -> None:
        flux_values = ctx.env["flux_values"]
        info: FluxDepInfoDict = ctx.env["info"]

        real_signals = t2ramsey_fluxdep_signal2real(signals["raw_signals"])

        plotters["t2r"].update(flux_values, signals["t2r"], refresh=False)
        plotters["t2r_curve"].update(
            self.lengths, real_signals[info["flux_idx"]], refresh=False
        )

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        np.savez_compressed(filepath, flux_values=flux_values, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flux_values}

        # signals
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
            x_info=x_info,
            y_info={
                "name": "Time Index",
                "unit": "a.u.",
                "values": np.arange(self.num_expts),
            },
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": result["raw_signals"].T,
            },
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/signals",
        )

        # length
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_length")),
            x_info=x_info,
            y_info={
                "name": "Time Index",
                "unit": "a.u.",
                "values": np.arange(self.num_expts),
            },
            z_info={
                "name": "Time (us)",
                "unit": "s",
                "values": result["length"].T * 1e-6,
            },
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/length",
        )

        # t2r
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_t2r")),
            x_info=x_info,
            z_info={"name": "T2 Ramsey", "unit": "s", "values": result["t2r"] * 1e-6},
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/t2r",
        )

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        data = np.load(filepath)

        flux_values = data["flux_values"]
        t2r_err = data["t2r_err"]
        t2r_detune_err = data["t2r_detune_err"]
        success = data["success"]

        signals_stored, flux_sig, len_idxs = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_signals")), **kwargs
        )
        assert flux_sig is not None and len_idxs is not None
        assert np.array_equal(flux_values, flux_sig)
        assert signals_stored.shape == (len(len_idxs), len(flux_values))

        length_stored, flux_len, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_length")), **kwargs
        )
        assert flux_len is not None
        assert length_stored.shape == (len(flux_len), len(len_idxs))
        assert np.array_equal(flux_values, flux_len)

        t2r_stored, flux_t2r, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_t2r")), **kwargs
        )
        assert flux_t2r is not None
        assert t2r_stored.shape == (len(flux_t2r),)
        assert np.array_equal(flux_values, flux_t2r)

        length = length_stored.astype(np.float64) * 1e6
        raw_signals = signals_stored.T.astype(np.complex128)
        t2r = t2r_stored.astype(np.float64) * 1e6
        t2r_err = t2r_err.astype(np.float64)
        t2r_detune = data["t2r_detune"].astype(np.float64)
        t2r_detune_err = t2r_detune_err.astype(np.float64)
        success = success.astype(np.bool_)

        return {
            "raw_signals": raw_signals,
            "length": length,
            "t2r": t2r,
            "t2r_err": t2r_err,
            "t2r_detune": t2r_detune,
            "t2r_detune_err": t2r_detune_err,
            "success": success,
            "flux_values": flux_values,
            "lengths": length_stored[0],
        }
