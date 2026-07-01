from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from typing_extensions import (
    TypedDict,  # closed/extra_items (PEP 728) not in stdlib 3.13
)

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import ScheduleStep
from zcu_tools.experiment.v2.utils import snr_checker, sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.notebook.utils import make_sweep
from zcu_tools.program.v2 import (
    Delay,
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
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import (
    load_labber_data,
    reserve_labber_filepath,
    save_labber_data,
)
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.process import rotate2real

from .executor import FluxDepCfg, FluxDepInfoDict, MeasurementTask, T_RootResult


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
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


def t1_fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.array(list(map(t1_signal2real, signals)), dtype=np.float64)


class T1ModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class T1CfgTemplate(ProgramV2Cfg, ExpCfgModel):
    modules: T1ModuleCfg
    sweep_range: tuple[float, float]


class T1SweepCfg(ConfigBase):
    length: SweepCfg


class T1Cfg(ProgramV2Cfg, FluxDepCfg):
    modules: T1ModuleCfg
    sweep: T1SweepCfg


class T1Result(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    length: NDArray[np.float64]
    t1: NDArray[np.float64]
    t1_err: NDArray[np.float64]
    success: NDArray[np.bool_]


class T1PlotDict(TypedDict, closed=True):
    t1: LivePlot1D
    t1_curve: LivePlot1D


class T1Task(MeasurementTask[T1Result, T_RootResult, T1PlotDict]):
    def __init__(
        self,
        num_expts: int,
        cfg_maker: Callable[
            [ScheduleStep[FluxDepCfg, Any], ModuleLibrary],
            T1CfgTemplate | None,
        ],
        earlystop_snr: float | None = None,
    ) -> None:
        self.num_expts = num_expts
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr
        self.last_cfg = None

        self.lengths = np.linspace(0, 1, num_expts)

    def init(self, dynamic_pbar=False) -> None:
        pass

    def run(
        self,
        state: ScheduleStep[FluxDepCfg, Any],
    ) -> None:
        info: FluxDepInfoDict = state.env["info"]

        cfg_temp = self.cfg_maker(state, state.env["ml"])

        if cfg_temp is None:
            return  # skip this task

        len_sweep = make_sweep(*cfg_temp.sweep_range, self.num_expts)
        self.lengths = sweep2array(len_sweep, "time", {"soccfg": state.env["soccfg"]})

        cfg = cfg_temp.to_dict()
        del cfg["sweep_range"]
        deepupdate(
            cfg,
            {"dev": state.cfg.dev, "sweep": {"length": len_sweep}},
            behavior="force",
        )
        cfg = T1Cfg.model_validate(cfg)

        self.last_cfg = cfg

        raw_step = state.child("raw_signals", cfg=cfg)
        signals_buffer = raw_step.buffer(self.num_expts)
        cfg = raw_step.cfg
        modules = cfg.modules
        setup_devices(cfg, progress=False)

        length_sweep = cfg.sweep.length
        length_param = sweep2param("length", length_sweep)

        _ = (
            raw_step.prog_builder(state.env["soc"], state.env["soccfg"])
            .add(
                Reset("reset", modules.reset),
                Pulse("pi_pulse", modules.pi_pulse),
                Delay("t1_delay", delay=length_param),
                Readout("readout", modules.readout),
            )
            .declare_sweep("length", length_sweep)
            .build_and_acquire(
                stop_checkers=[
                    snr_checker(signals_buffer.at(), self.earlystop_snr, t1_signal2real)
                ],
            )
        )

        raw_signals = raw_step.array_data

        real_signals = t1_signal2real(raw_signals)

        t1, t1err, fit_signals, _ = fit_decay(self.lengths, real_signals)

        success = True
        mean_err = np.mean(np.abs(real_signals - fit_signals))
        if t1 > 2 * np.max(self.lengths) or mean_err > 0.1 * np.ptp(fit_signals):
            t1, t1err = np.nan, np.nan
            success = False

        if success:
            info["t1"] = t1
            info["smooth_t1"] = 0.5 * (info.last.get("smooth_t1", t1) + t1)

        with MinIntervalFunc.force_execute():
            state.set_data(
                T1Result(
                    raw_signals=raw_signals,
                    length=self.lengths.copy(),
                    t1=np.array(t1),
                    t1_err=np.array(t1err),
                    success=np.array(success),
                )
            )

    def get_default_result(self) -> T1Result:
        return T1Result(
            raw_signals=np.full((self.num_expts,), np.nan, dtype=np.complex128),
            length=self.lengths.copy(),
            t1=np.array(np.nan),
            t1_err=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        pass

    def num_axes(self) -> dict[str, int]:
        return dict(t1=1, t1_curve=1)

    def make_plotter(self, name, axs) -> T1PlotDict:
        return T1PlotDict(
            t1=LivePlot1D(
                "Flux device value",
                "T1 (us)",
                existed_axes=[axs["t1"]],
                segment_kwargs=dict(
                    title=name + "(t1)", line_kwargs=[dict(linestyle="None")]
                ),
            ),
            t1_curve=LivePlot1D(
                "Signal",
                "Time (us)",
                existed_axes=[axs["t1_curve"]],
                segment_kwargs=dict(title=name + "(t1 curve)"),
            ),
        )

    def update_plotter(
        self,
        plotters,
        ctx: ScheduleStep[Any, Any],
        signals: T1Result,
    ) -> None:
        flux_values = ctx.env["flux_values"]
        info: FluxDepInfoDict = ctx.env["info"]

        real_signals = t1_fluxdep_signal2real(signals["raw_signals"])

        plotters["t1"].update(flux_values, signals["t1"], refresh=False)
        plotters["t1_curve"].update(
            self.lengths, real_signals[info["flux_idx"]], refresh=False
        )

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        np.savez_compressed(filepath, flux_values=flux_values, **result)

        flux_axis = ("Flux value", "a.u.", flux_values)
        time_axis = ("Time Index", "a.u.", np.arange(self.num_expts))

        # signals: native z is (Nflux, Ntime) = (outer, inner), axes inner-first
        save_labber_data(
            reserve_labber_filepath(
                str(filepath.with_name(filepath.name + "_signals"))
            ),
            z=("Signal", "a.u.", result["raw_signals"]),
            axes=[time_axis, flux_axis],
            comment=comment,
            tags=prefix_tag + "/signals",
        )

        # length: native z is (Nflux, Ntime)
        save_labber_data(
            reserve_labber_filepath(str(filepath.with_name(filepath.name + "_length"))),
            z=("Time (us)", "s", result["length"] * 1e-6),
            axes=[time_axis, flux_axis],
            comment=comment,
            tags=prefix_tag + "/length",
        )

        # t1: 1D over flux only
        save_labber_data(
            reserve_labber_filepath(str(filepath.with_name(filepath.name + "_t1"))),
            z=("T1", "s", result["t1"] * 1e-6),
            axes=[flux_axis],
            comment=comment,
            tags=prefix_tag + "/t1",
        )

    @classmethod
    def load(cls, filepath: str) -> dict:
        _filepath = Path(filepath)

        data = np.load(filepath)

        flux_values = data["flux_values"]
        success = data["success"]

        # signals: native z is (Nflux, Ntime), axes inner-first [Time, Flux]
        signal_path = str(_filepath.with_name(_filepath.name + "_signals"))
        sig = load_labber_data(signal_path)
        signals_stored = sig.z
        len_idxs = sig.axes[0].values
        flux_sig = sig.axes[1].values
        comment = sig.comment
        assert np.array_equal(flux_values, flux_sig)
        assert signals_stored.shape == (len(flux_values), len(len_idxs))

        # length: native z is (Nflux, Ntime)
        length_path = str(_filepath.with_name(_filepath.name + "_length"))
        ln = load_labber_data(length_path)
        length_stored = ln.z
        flux_len = ln.axes[1].values
        assert length_stored.shape == (len(flux_len), len(len_idxs))
        assert np.array_equal(flux_values, flux_len)

        # t1: 1D over flux only
        t1_path = str(_filepath.with_name(_filepath.name + "_t1"))
        t = load_labber_data(t1_path)
        t1_stored = t.z
        flux_t1 = t.axes[0].values
        assert t1_stored.shape == (len(flux_t1),)
        assert np.array_equal(flux_values, flux_t1)

        length = length_stored[0].real.astype(np.float64) * 1e6  # back to us
        raw_signals = signals_stored.astype(np.complex128)
        t1 = t1_stored.real.astype(np.float64) * 1e6  # back to us
        t1_err = data["t1_err"].astype(np.float64)
        success = success.astype(np.bool_)
        last_cfg = None
        if comment:
            last_cfg, _, _ = parse_comment(comment)

        return {
            "raw_signals": raw_signals,
            "length": length,
            "t1": t1,
            "t1_err": t1_err,
            "success": success,
            "flux_values": flux_values,
            "lengths": length_stored[0].real,
            "last_cfg": last_cfg,
        }
