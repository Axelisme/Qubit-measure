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
from zcu_tools.utils.datasaver import load_labber_data, save_labber_data
from zcu_tools.utils.fitting import fit_decay_fringe
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.process import rotate2real

from .executor import FluxDepCfg, FluxDepInfoDict, MeasurementTask, T_RootResult


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


class T2RamseyModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class T2RamseyCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    modules: T2RamseyModuleCfg
    sweep_range: tuple[float, float]


class T2RamseySweepCfg(ConfigBase):
    length: SweepCfg


class T2RamseyCfg(ProgramV2Cfg, FluxDepCfg):
    modules: T2RamseyModuleCfg
    sweep: T2RamseySweepCfg
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
            [
                ScheduleStep[FluxDepCfg, Any],
                ModuleLibrary,
            ],
            T2RamseyCfgTemplate | None,
        ],
        earlystop_snr: float | None = None,
    ) -> None:
        self.num_expts = num_expts
        self.detune_ratio = detune_ratio
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
        cfg["activate_detune"] = self.detune_ratio / len_sweep.step
        cfg = T2RamseyCfg.model_validate(cfg)
        self.last_cfg = cfg

        raw_step = state.child("raw_signals", cfg=cfg)
        signals_buffer = raw_step.buffer(self.num_expts)
        cfg = raw_step.cfg
        modules = cfg.modules
        setup_devices(cfg, progress=False)

        detune = cfg.activate_detune
        length_sweep = cfg.sweep.length
        length_param = sweep2param("length", length_sweep)

        _ = (
            raw_step.prog_builder(state.env["soc"], state.env["soccfg"])
            .add(
                Reset("reset", modules.reset),
                Pulse("pi2_pulse1", modules.pi2_pulse),
                Delay("t2r_delay", delay=length_param),
                Pulse(
                    name="pi2_pulse2",
                    cfg=modules.pi2_pulse.with_updates(
                        phase=modules.pi2_pulse.phase + 360 * detune * length_param
                    ),
                ),
                Readout("readout", modules.readout),
            )
            .declare_sweep("length", length_sweep)
            .build_and_acquire(
                stop_checkers=[
                    snr_checker(
                        signals_buffer.at(),
                        self.earlystop_snr,
                        t2ramsey_signal2real,
                    )
                ],
            )
        )

        raw_signals = raw_step.array_data

        real_signals = t2ramsey_signal2real(raw_signals)

        t2r, t2r_err, t2r_detune, t2r_detune_err, fit_signals, _ = fit_decay_fringe(
            self.lengths, real_signals
        )
        t2r_detune = t2r_detune - cfg.activate_detune

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
            state.set_data(
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
            raw_signals=np.full((self.num_expts,), np.nan, dtype=np.complex128),
            length=self.lengths.copy(),
            t2r=np.array(np.nan),
            t2r_err=np.array(np.nan),
            t2r_detune=np.array(np.nan),
            t2r_detune_err=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        pass

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

    def update_plotter(
        self,
        plotters,
        ctx: ScheduleStep[Any, Any],
        signals: T2RamseyResult,
    ) -> None:
        flux_values = ctx.env["flux_values"]
        info: FluxDepInfoDict = ctx.env["info"]

        real_signals = t2ramsey_fluxdep_signal2real(signals["raw_signals"])

        plotters["t2r"].update(flux_values, signals["t2r"], refresh=False)
        plotters["t2r_curve"].update(
            self.lengths, real_signals[info["flux_idx"]], refresh=False
        )

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)
        cfg = self.last_cfg
        assert cfg is not None

        np.savez_compressed(filepath, flux_values=flux_values, **result)

        comment = make_comment(cfg, comment)

        flux_axis = ("Flux value", "a.u.", flux_values)
        time_axis = ("Time Index", "a.u.", np.arange(self.num_expts))

        # signals: native z is (Ny, Nx) = (num_expts, n_flux), inner axis = flux
        save_labber_data(
            str(filepath.with_name(filepath.name + "_signals")),
            z=("Signal", "a.u.", result["raw_signals"]),
            axes=[flux_axis, time_axis],
            comment=comment,
            tags=prefix_tag + "/signals",
        )

        # length: 1-D per-time vector broadcast to an explicit (Ny, Nx) grid
        length_grid = np.broadcast_to(
            (result["length"] * 1e-6)[:, None],
            (self.num_expts, len(flux_values)),
        )
        save_labber_data(
            str(filepath.with_name(filepath.name + "_length")),
            z=("Time (us)", "s", length_grid),
            axes=[flux_axis, time_axis],
            comment=comment,
            tags=prefix_tag + "/length",
        )

        # t2r: 1-D over flux
        save_labber_data(
            str(filepath.with_name(filepath.name + "_t2r")),
            z=("T2 Ramsey", "s", result["t2r"] * 1e-6),
            axes=[flux_axis],
            comment=comment,
            tags=prefix_tag + "/t2r",
        )

    @classmethod
    def load(cls, filepath: str) -> dict:
        _filepath = Path(filepath)

        data = np.load(filepath)

        flux_values = data["flux_values"]
        t2r_err = data["t2r_err"]
        t2r_detune_err = data["t2r_detune_err"]
        success = data["success"]

        # signals: native z is (Ny, Nx) = (num_expts, n_flux), inner axis = flux
        signal_path = str(_filepath.with_name(_filepath.name + "_signals"))
        d_sig = load_labber_data(signal_path)
        signals_stored = np.asarray(d_sig.z)
        flux_sig = np.asarray(d_sig.axes[0].values)
        len_idxs = np.asarray(d_sig.axes[1].values)
        comment = d_sig.comment

        assert np.array_equal(flux_values, flux_sig)
        assert signals_stored.shape == (len(len_idxs), len(flux_values))

        # length: native z is (Ny, Nx) = (num_expts, n_flux), inner axis = flux
        length_path = str(_filepath.with_name(_filepath.name + "_length"))
        d_len = load_labber_data(length_path)
        length_stored = np.asarray(d_len.z)
        flux_len = np.asarray(d_len.axes[0].values)
        assert length_stored.shape == (len(len_idxs), len(flux_len))
        assert np.array_equal(flux_values, flux_len)

        # t2r: 1-D over flux
        t2r_path = str(_filepath.with_name(_filepath.name + "_t2r"))
        d_t2r = load_labber_data(t2r_path)
        t2r_stored = np.asarray(d_t2r.z)
        flux_t2r = np.asarray(d_t2r.axes[0].values)
        assert t2r_stored.shape == (len(flux_t2r),)
        assert np.array_equal(flux_values, flux_t2r)

        length = length_stored.astype(np.float64) * 1e6
        raw_signals = signals_stored.astype(np.complex128)
        t2r = t2r_stored.astype(np.float64) * 1e6
        t2r_err = t2r_err.astype(np.float64)
        t2r_detune = data["t2r_detune"].astype(np.float64)
        t2r_detune_err = t2r_detune_err.astype(np.float64)
        success = success.astype(np.bool_)
        last_cfg = None
        if comment:
            last_cfg, _, _ = parse_comment(comment)

        return {
            "raw_signals": raw_signals,
            "length": length,
            "t2r": t2r,
            "t2r_err": t2r_err,
            "t2r_detune": t2r_detune,
            "t2r_detune_err": t2r_detune_err,
            "success": success,
            "flux_values": flux_values,
            "lengths": length_stored[:, 0],
            "last_cfg": last_cfg,
        }
