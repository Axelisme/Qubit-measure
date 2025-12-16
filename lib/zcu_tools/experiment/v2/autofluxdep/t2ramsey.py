from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import (
    Callable,
    Dict,
    List,
    NotRequired,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, TaskContextView
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.library import ModuleLibrary
from zcu_tools.liveplot import LivePlotter1D
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

from .executor import FluxDepInfoDict, MeasurementTask, T_RootResultType


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


class T2RamseyCfgTemplate(ModularProgramCfg):
    reset: NotRequired[Union[ResetCfg, str]]
    pi2_pulse: Union[PulseCfg, str]
    readout: Union[ReadoutCfg, str]

    sweep_range: Tuple[float, float]


class T2RamseyCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi2_pulse: PulseCfg
    readout: ReadoutCfg
    activate_detune: float

    sweep: Dict[str, SweepCfg]


class T2RamseyResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    length: NDArray[np.float64]
    t2r: NDArray[np.float64]
    t2r_err: NDArray[np.float64]
    t2r_detune: NDArray[np.float64]
    t2r_detune_err: NDArray[np.float64]
    success: NDArray[np.bool_]


class PlotterDictType(TypedDict, closed=True):
    t2r: LivePlotter1D
    t2r_curve: LivePlotter1D


class T2RamseyMeasurementTask(
    MeasurementTask[T2RamseyResult, T_RootResultType, TaskConfig, PlotterDictType]
):
    def __init__(
        self,
        num_expts: int,
        detune_ratio: float,
        cfg_maker: Callable[
            [TaskContextView, ModuleLibrary], Optional[T2RamseyCfgTemplate]
        ],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.num_expts = num_expts
        self.detune_ratio = detune_ratio
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        def measure_ramsey_fn(ctx: TaskContextView, update_hook: Callable):
            len_sweep = ctx.cfg["sweep"]["length"]

            assert len_sweep["expts"] == self.num_expts

            t2r_params = sweep2param("length", len_sweep)

            prog = ModularProgramV2(
                ctx.env_dict["soccfg"],
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse(name="pi2_pulse1", cfg=ctx.cfg["pi2_pulse"]),
                    Delay(name="t2r_delay", delay=t2r_params),
                    Pulse(
                        name="pi2_pulse2",
                        cfg=Pulse.set_param(
                            PulseCfg(ctx.cfg["pi2_pulse"]),
                            "phase",
                            ctx.cfg["pi2_pulse"]["phase"]
                            + 360 * ctx.cfg["activate_detune"] * t2r_params,
                        ),
                    ),
                    Readout("readout", ctx.cfg["readout"]),
                ],
            )
            return prog.acquire(
                ctx.env_dict["soc"],
                progress=False,
                callback=wrap_earlystop_check(
                    prog,
                    update_hook,
                    self.earlystop_snr,
                    signal2real_fn=t2ramsey_signal2real,
                ),
            )

        self.lengths = np.linspace(0, 1, num_expts)
        self.task = HardTask[
            np.complex128, T_RootResultType, T2RamseyCfg, List[NDArray[np.float64]]
        ](
            measure_fn=measure_ramsey_fn,
            result_shape=(num_expts,),
        )

    def num_axes(self) -> Dict[str, int]:
        return dict(t2r=1, t2r_curve=1)

    def make_plotter(self, name, axs) -> PlotterDictType:
        return PlotterDictType(
            t2r=LivePlotter1D(
                "Flux device value",
                "T2Ramsey (us)",
                existed_axes=[axs["t2r"]],
                segment_kwargs=dict(
                    title=name + "(t2r)", line_kwargs=[dict(linestyle="None")]
                ),
            ),
            t2r_curve=LivePlotter1D(
                "Signal",
                "Time (us)",
                existed_axes=[axs["t2r_curve"]],
                segment_kwargs=dict(title=name + "(t2r curve)"),
            ),
        )

    def update_plotter(self, plotters, ctx, signals) -> None:
        flx_values = ctx.env_dict["flx_values"]
        info: FluxDepInfoDict = ctx.env_dict["info"]

        real_signals = t2ramsey_fluxdep_signal2real(signals["raw_signals"])

        plotters["t2r"].update(flx_values, signals["t2r"], refresh=False)
        plotters["t2r_curve"].update(
            self.lengths, real_signals[info["flx_idx"]], refresh=False
        )

    def save(self, filepath, flx_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        np.savez_compressed(filepath, flx_values=flx_values, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flx_values}

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

    def load(self, filepath: str, **kwargs) -> T2RamseyResult:
        data = np.load(filepath)

        flx_values = data["flx_values"]
        t2r_err = data["t2r_err"]
        t2r_detune_err = data["t2r_detune_err"]
        success = data["success"]

        signals_stored, flx_sig, len_idxs = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_signals")), **kwargs
        )
        assert flx_sig is not None and len_idxs is not None
        assert np.array_equal(flx_values, flx_sig)
        assert signals_stored.shape == (len(len_idxs), len(flx_values))

        length_stored, flx_len, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_length")), **kwargs
        )
        assert flx_len is not None
        assert length_stored.shape == (len(flx_len), len(len_idxs))
        assert np.array_equal(flx_values, flx_len)

        t2r_stored, flx_t2r, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_t2r")), **kwargs
        )
        assert flx_t2r is not None
        assert t2r_stored.shape == (len(flx_t2r),)
        assert np.array_equal(flx_values, flx_t2r)

        length = length_stored[0].astype(np.float64) * 1e6
        raw_signals = signals_stored.T.astype(np.complex128)
        t2r = t2r_stored.astype(np.float64) * 1e6
        t2r_err = t2r_err.astype(np.float64)
        t2r_detune = data["t2r_detune"].astype(np.float64)
        t2r_detune_err = t2r_detune_err.astype(np.float64)
        success = success.astype(np.bool_)

        return T2RamseyResult(
            raw_signals=raw_signals,
            length=length,
            t2r=t2r,
            t2r_err=t2r_err,
            t2r_detune=t2r_detune,
            t2r_detune_err=t2r_detune_err,
            success=success,
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.init_cfg = deepcopy(ctx.cfg)
        self.task.init(ctx(addr="raw_signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx) -> None:
        ml: ModuleLibrary = ctx.env_dict["ml"]
        info: FluxDepInfoDict = ctx.env_dict["info"]

        cfg_temp = self.cfg_maker(ctx, ml)

        if cfg_temp is None:
            return  # skip this task

        len_sweep = make_sweep(*cfg_temp["sweep_range"], self.num_expts)
        self.lengths = sweep2array(len_sweep)

        cfg_temp = dict(cfg_temp)
        deepupdate(
            cfg_temp, {"dev": ctx.cfg.get("dev", {}), "sweep": {"length": len_sweep}}
        )
        cfg_temp = ml.make_cfg(cfg_temp)
        cfg_temp["activate_detune"] = self.detune_ratio / len_sweep["step"]

        cfg = cast(T2RamseyCfg, cfg_temp)
        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))  # type: ignore

        raw_signals = ctx.get_data()["raw_signals"]
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
            ctx.set_data(
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
