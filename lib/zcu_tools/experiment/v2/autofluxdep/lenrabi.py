from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Callable, NotRequired, Optional, TypedDict, cast

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState
from zcu_tools.experiment.v2.utils import round_zcu_time, wrap_earlystop_check
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.notebook.utils import make_comment, make_sweep
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
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
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.process import rotate2real

from .executor import FluxDepInfoDict, MeasurementTask, T_RootResult


def lenrabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
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


def lenrabi_fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.array(list(map(lenrabi_signal2real, signals)), dtype=np.float64)


def auto_fit_lenrabi(
    lengths: NDArray[np.float64], real_signals: NDArray[np.float64]
) -> tuple[float, float, float, float, NDArray[np.float64]]:
    *decay_args, decay_signals, _ = fit_rabi(lengths, real_signals, decay=True)
    *normal_args, normal_signals, _ = fit_rabi(lengths, real_signals, decay=False)

    decay_loss = np.mean(np.abs(real_signals - decay_signals))
    normal_loss = np.mean(np.abs(real_signals - normal_signals))
    if decay_loss < normal_loss:
        fit_loss = float(decay_loss)
        fit_signals = decay_signals
        fit_params = decay_args
    else:
        fit_loss = float(normal_loss)
        fit_signals = normal_signals
        fit_params = normal_args
    pi_len, pi2_len, rabi_freq = fit_params

    return pi_len, pi2_len, rabi_freq, fit_loss, fit_signals


class LenRabiModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    rabi_pulse: PulseCfg
    readout: ReadoutCfg


class LenRabiCfgTemplate(ModularProgramCfg, TaskCfg):
    modules: LenRabiModuleCfg
    sweep_range: tuple[float, float]


class LenRabiCfg(ModularProgramCfg, TaskCfg):
    modules: LenRabiModuleCfg
    dev: dict[str, DeviceInfo]
    sweep: dict[str, SweepCfg]


class LenRabiResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    length: NDArray[np.float64]
    pi_length: NDArray[np.float64]
    pi2_length: NDArray[np.float64]
    rabi_freq: NDArray[np.float64]
    success: NDArray[np.bool_]


class LenRabiPlotterDict(TypedDict, closed=True):
    rabi_curve: LivePlotter2DwithLine


class LenRabiTask(MeasurementTask[LenRabiResult, T_RootResult, LenRabiPlotterDict]):
    def __init__(
        self,
        num_expts: int,
        cfg_maker: Callable[
            [TaskState[LenRabiResult, T_RootResult], ModuleLibrary],
            Optional[dict[str, Any]],
        ],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.num_expts = num_expts
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        def measure_fn(ctx: TaskState, update_hook: Callable):
            len_sweep = ctx.cfg["sweep"]["length"]
            modules = ctx.cfg["modules"]

            assert len_sweep["expts"] == self.num_expts

            len_params = sweep2param("length", len_sweep)
            Pulse.set_param(modules["rabi_pulse"], "length", len_params)
            prog = ModularProgramV2(
                ctx.env["soccfg"],
                ctx.cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Pulse("rabi_pulse", modules["rabi_pulse"]),
                    Readout("readout", modules["readout"]),
                ],
            )
            return prog.acquire(
                ctx.env["soc"],
                progress=False,
                callback=wrap_earlystop_check(
                    prog,
                    update_hook,
                    snr_threshold=self.earlystop_snr,
                    signal2real_fn=lenrabi_signal2real,
                ),
            )

        self.lengths = np.linspace(0, 1, num_expts)
        self.task = Task[T_RootResult, list[NDArray[np.float64]]](
            measure_fn, result_shape=(num_expts,)
        )

    def init(
        self, ctx: TaskState[LenRabiResult, T_RootResult], dynamic_pbar: bool = False
    ) -> None:
        self.init_cfg = deepcopy(ctx.cfg)
        self.task.init(ctx.child("raw_signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx: TaskState[LenRabiResult, T_RootResult]) -> None:
        cfg_temp = self.cfg_maker(ctx, ctx.env["ml"])
        if cfg_temp is None:
            return  # skip this task
        cfg_temp = check_type(cfg_temp, LenRabiCfgTemplate)

        len_sweep = make_sweep(*cfg_temp["sweep_range"], self.num_expts)

        cfg_temp = dict(cfg_temp)
        del cfg_temp["sweep_range"]  # type: ignore
        deepupdate(
            cfg_temp,
            {"dev": ctx.cfg["dev"], "sweep": {"length": len_sweep}},
            behavior="force",
        )
        cfg = check_type(cfg_temp, LenRabiCfg)

        rabi_pulse = cfg["modules"]["rabi_pulse"]
        self.task.run(ctx.child("raw_signals", new_cfg=cfg))

        real_signals = lenrabi_signal2real(ctx.value["raw_signals"])

        self.lengths = sweep2array(len_sweep)
        self.lengths = round_zcu_time(
            self.lengths, ctx.env["soccfg"], gen_ch=rabi_pulse["ch"]
        )

        pi_len, pi2_len, rabi_freq, mean_err, fit_signals = auto_fit_lenrabi(
            self.lengths, real_signals
        )

        success = True
        if (
            pi_len < 0.03
            or mean_err > 0.1 * np.ptp(fit_signals)
            or pi_len > 0.9 * np.max(self.lengths)
        ):
            pi_len, pi2_len, rabi_freq = np.nan, np.nan, np.nan
            success = False

        if success:
            info: FluxDepInfoDict = ctx.env["info"]

            cur_pi_product = pi_len * rabi_pulse["gain"]
            prev_pi_product = info.last.get("smooth_pi_product", cur_pi_product)
            num_step = max(1, info["flx_idx"] - info.last.get("lenrabi_success_idx", 0))
            weight = 0.7**num_step
            smooth_pi_product = (1 - weight) * cur_pi_product + weight * prev_pi_product

            info["pi_length"] = pi_len
            info["pi2_length"] = pi2_len
            info["pi_pulse"] = deepcopy(rabi_pulse)
            info["pi_pulse"]["waveform"]["length"] = pi_len
            info["pi2_pulse"] = deepcopy(rabi_pulse)
            info["pi2_pulse"]["waveform"]["length"] = pi2_len
            info["smooth_pi_product"] = smooth_pi_product
            info["lenrabi_success_idx"] = info["flx_idx"]

        with MinIntervalFunc.force_execute():
            ctx.set_value(
                LenRabiResult(
                    raw_signals=ctx.value["raw_signals"],
                    length=self.lengths.copy(),
                    pi_length=np.array(pi_len),
                    pi2_length=np.array(pi2_len),
                    rabi_freq=np.array(rabi_freq),
                    success=np.array(success),
                )
            )

    def get_default_result(self) -> LenRabiResult:
        return LenRabiResult(
            raw_signals=self.task.get_default_result(),
            length=self.lengths.copy(),
            pi_length=np.array(np.nan),
            pi2_length=np.array(np.nan),
            rabi_freq=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        self.task.cleanup()

    def num_axes(self) -> dict[str, int]:
        return dict(rabi_curve=2)

    def make_plotter(self, name, axs) -> LenRabiPlotterDict:
        self.pi_line = axs["rabi_curve"][1].axvline(np.nan, color="red", linestyle="--")
        return LenRabiPlotterDict(
            rabi_curve=LivePlotter2DwithLine(
                "Flux device value",
                "Signal",
                line_axis=1,
                num_lines=3,
                title=name + "(rabi_curve)",
                existed_axes=[axs["rabi_curve"]],
            ),
        )

    def update_plotter(self, plotters, ctx: TaskState, signals) -> None:
        flx_values = ctx.env["flx_values"]

        self.pi_line.set_xdata([ctx.env["info"].get("pi_length", np.nan)])
        plotters["rabi_curve"].update(
            flx_values,
            self.lengths,
            lenrabi_fluxdep_signal2real(signals["raw_signals"]),
            refresh=False,
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

        # pi length
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_pi_length")),
            x_info=x_info,
            z_info={
                "name": "Pi length",
                "unit": "s",
                "values": result["pi_length"] * 1e-6,
            },
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/pi_length",
        )

        # success
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_success")),
            x_info=x_info,
            z_info={"name": "Success", "unit": "bool", "values": result["success"]},
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/success",
        )

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        # main container (has pi2_length and rabi_freq)
        data = np.load(filepath)

        flx_values = data["flx_values"]
        pi2_length = data["pi2_length"]
        rabi_freq = data["rabi_freq"]
        success_main = data["success"]

        # signals
        signals_stored, flx_sig, len_idxs = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_signals")), **kwargs
        )
        assert flx_sig is not None and len_idxs is not None
        assert np.array_equal(flx_values, flx_sig)
        assert signals_stored.shape == (len(len_idxs), len(flx_sig))

        length_stored, flx_len, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_length")), **kwargs
        )
        assert flx_len is not None
        assert length_stored.shape == (len(flx_len), len(len_idxs))
        assert np.array_equal(flx_values, flx_len)

        # pi_length
        pi_length_data, flx_pi, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_pi_length")), **kwargs
        )
        assert flx_pi is not None
        assert pi_length_data.shape == (len(flx_pi), len(len_idxs))
        assert np.array_equal(flx_values, flx_pi)

        # success
        success_data, flx_succ, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_success")), **kwargs
        )
        assert flx_succ is not None
        assert success_data.shape == (len(flx_succ), len(len_idxs))
        assert np.array_equal(flx_values, flx_succ)

        assert (
            pi2_length.shape
            == rabi_freq.shape
            == success_main.shape
            == pi_length_data.shape
        )

        length = length_stored.astype(np.float64) * 1e6
        raw_signals = signals_stored.T.astype(np.complex128)
        pi_length = pi_length_data.astype(np.float64)
        pi2_length = pi2_length.astype(np.float64)
        rabi_freq = rabi_freq.astype(np.float64)
        success = success_data.astype(np.bool_) & success_main.astype(np.bool_)

        return {
            "raw_signals": raw_signals,
            "length": length,
            "pi_length": pi_length,
            "pi2_length": pi2_length,
            "rabi_freq": rabi_freq,
            "success": success,
            "flx_values": flx_values,
            "lengths": len_idxs,
        }
