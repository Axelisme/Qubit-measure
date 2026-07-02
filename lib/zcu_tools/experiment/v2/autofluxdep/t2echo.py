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
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.process import rotate2real

from .env import FluxDepEnv, FluxDepInfoDict
from .executor import FluxDepCfg, MeasurementTask, T_RootResult


def t2echo_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
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


def t2echo_fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.array(list(map(t2echo_signal2real, signals)), dtype=np.float64)


class T2EchoModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    pi2_pulse: PulseCfg
    readout: ReadoutCfg


class T2EchoCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    modules: T2EchoModuleCfg
    sweep_range: tuple[float, float]


class T2EchoSweepCfg(ConfigBase):
    length: SweepCfg


class T2EchoCfg(ProgramV2Cfg, FluxDepCfg):
    modules: T2EchoModuleCfg
    sweep: T2EchoSweepCfg
    activate_detune: float


class T2EchoResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    length: NDArray[np.float64]
    t2e: NDArray[np.float64]
    t2e_err: NDArray[np.float64]
    success: NDArray[np.bool_]


class T2EchoPlotDict(TypedDict, closed=True):
    t2e: LivePlot1D
    t2e_curve: LivePlot1D


class T2EchoTask(MeasurementTask[T2EchoResult, T_RootResult, T2EchoPlotDict]):
    def __init__(
        self,
        num_expts: int,
        detune_ratio: float,
        cfg_maker: Callable[
            [
                ScheduleStep[FluxDepCfg, Any, FluxDepEnv],
                ModuleLibrary,
            ],
            T2EchoCfgTemplate | None,
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
        state: ScheduleStep[FluxDepCfg, Any, FluxDepEnv],
    ) -> None:
        info: FluxDepInfoDict = state.env.info

        cfg_temp = self.cfg_maker(state, state.env.ml)

        if cfg_temp is None:
            return  # skip this task

        len_sweep = make_sweep(*cfg_temp.sweep_range, self.num_expts)
        self.lengths = sweep2array(
            len_sweep, "time", {"soccfg": state.env.soccfg, "scaler": 0.5}
        )

        cfg = cfg_temp.to_dict()
        del cfg["sweep_range"]
        deepupdate(
            cfg,
            {"dev": state.cfg.dev, "sweep": {"length": len_sweep}},
            behavior="force",
        )
        cfg = T2EchoCfg.model_validate(cfg)
        cfg.activate_detune = self.detune_ratio / len_sweep.step
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
            raw_step.prog_builder(state.env.soc, state.env.soccfg)
            .add(
                Reset("reset", modules.reset),
                Pulse("pi2_pulse1", modules.pi2_pulse),
                Delay("t2e_delay1", delay=0.5 * length_param),
                Pulse("pi_pulse", modules.pi_pulse),
                Delay("t2e_delay2", delay=0.5 * length_param),
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
                        t2echo_signal2real,
                    )
                ],
            )
        )

        raw_signals = raw_step.array_data

        real_signals = t2echo_signal2real(raw_signals)

        if self.detune_ratio == 0.0:
            t2e, t2e_err, fit_signals, _ = fit_decay(self.lengths, real_signals)
        else:
            t2e, t2e_err, _, _, fit_signals, _ = fit_decay_fringe(
                self.lengths, real_signals
            )

        success = True
        mean_err = np.mean(np.abs(real_signals - fit_signals))
        if t2e > 2 * np.max(self.lengths) or mean_err > 0.1 * np.ptp(fit_signals):
            t2e, t2e_err = np.nan, np.nan
            success = False

        if success:
            info["t2e"] = t2e
            info["smooth_t2e"] = 0.5 * (info.last.get("smooth_t2e", t2e) + t2e)

        with MinIntervalFunc.force_execute():
            state.set_data(
                T2EchoResult(
                    raw_signals=raw_signals,
                    length=self.lengths.copy(),
                    t2e=np.array(t2e),
                    t2e_err=np.array(t2e_err),
                    success=np.array(success),
                )
            )

    def get_default_result(self) -> T2EchoResult:
        return T2EchoResult(
            raw_signals=np.full((self.num_expts,), np.nan, dtype=np.complex128),
            length=self.lengths.copy(),
            t2e=np.array(np.nan),
            t2e_err=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        pass

    def num_axes(self) -> dict[str, int]:
        return dict(t2e=1, t2e_curve=1)

    def make_plotter(self, name, axs) -> T2EchoPlotDict:
        return T2EchoPlotDict(
            t2e=LivePlot1D(
                "Flux device value",
                "T2 Echo (us)",
                existed_axes=[axs["t2e"]],
                segment_kwargs=dict(
                    title=name + "(t2e)", line_kwargs=[dict(linestyle="None")]
                ),
            ),
            t2e_curve=LivePlot1D(
                "Signal",
                "Time (us)",
                existed_axes=[axs["t2e_curve"]],
                segment_kwargs=dict(title=name + "(t2e curve)"),
            ),
        )

    def update_plotter(
        self,
        plotters,
        ctx: ScheduleStep[Any, Any, FluxDepEnv],
        signals: T2EchoResult,
    ) -> None:
        flux_values = ctx.env.flux_values
        info: FluxDepInfoDict = ctx.env.info

        real_signals = t2echo_fluxdep_signal2real(signals["raw_signals"])

        plotters["t2e"].update(flux_values, signals["t2e"], refresh=False)
        plotters["t2e_curve"].update(
            self.lengths, real_signals[info["flux_idx"]], refresh=False
        )

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)
        cfg = self.last_cfg
        assert cfg is not None

        np.savez_compressed(filepath, flux_values=flux_values, **result)

        flux_axis = ("Flux value", "a.u.", flux_values)
        time_axis = ("Time Index", "a.u.", np.arange(self.num_expts))
        comment = make_comment(cfg, comment)

        # signals
        save_labber_data(
            str(filepath.with_name(filepath.name + "_signals")),
            z=("Signal", "a.u.", result["raw_signals"].T),
            axes=[flux_axis, time_axis],
            comment=comment,
            tags=prefix_tag + "/signals",
        )

        # length
        save_labber_data(
            str(filepath.with_name(filepath.name + "_length")),
            z=("Time (us)", "s", result["length"].T * 1e-6),
            axes=[flux_axis, time_axis],
            comment=comment,
            tags=prefix_tag + "/length",
        )

        # t2e
        save_labber_data(
            str(filepath.with_name(filepath.name + "_t2e")),
            z=("T2 Echo", "s", result["t2e"] * 1e-6),
            axes=[flux_axis],
            comment=comment,
            tags=prefix_tag + "/t2e",
        )

    @classmethod
    def load(cls, filepath: str) -> dict:
        _filepath = Path(filepath)

        data = np.load(filepath)

        flux_values = data["flux_values"]
        t2e_err = data["t2e_err"]
        success = data["success"]

        signal_path = str(_filepath.with_name(_filepath.name + "_signals"))
        ld_sig = load_labber_data(signal_path)
        signals_stored = np.asarray(ld_sig.z)
        flux_sig = np.asarray(ld_sig.axes[0].values)
        len_idxs = np.asarray(ld_sig.axes[1].values)
        comment = ld_sig.comment
        # native load is (Ny, Nx) = (num_expts, n_flux), inner axis (flux) last
        assert np.array_equal(flux_values, flux_sig)
        assert signals_stored.shape == (len(len_idxs), len(flux_values))

        length_path = str(_filepath.with_name(_filepath.name + "_length"))
        ld_len = load_labber_data(length_path)
        length_stored = np.asarray(ld_len.z)
        flux_len = np.asarray(ld_len.axes[0].values)
        # native load is (num_expts, n_flux); old dict-API load was flux-major
        assert length_stored.shape == (len(len_idxs), len(flux_len))
        assert np.array_equal(flux_values, flux_len)

        t2e_path = str(_filepath.with_name(_filepath.name + "_t2e"))
        ld_t2e = load_labber_data(t2e_path)
        t2e_stored = np.asarray(ld_t2e.z)
        flux_t2e = np.asarray(ld_t2e.axes[0].values)
        assert t2e_stored.shape == (len(flux_t2e),)
        assert np.array_equal(flux_values, flux_t2e)

        # length trace at flux index 0: a num_expts-long vector
        length = length_stored[:, 0].astype(np.float64) * 1e6
        raw_signals = signals_stored.T.astype(np.complex128)
        t2e = t2e_stored.astype(np.float64) * 1e6
        t2e_err = t2e_err.astype(np.float64)
        success = success.astype(np.bool_)
        last_cfg = None
        if comment:
            last_cfg, _, _ = parse_comment(comment)

        return {
            "raw_signals": raw_signals,
            "length": length,
            "t2e": t2e,
            "t2e_err": t2e_err,
            "success": success,
            "flux_values": flux_values,
            "lengths": length_stored[:, 0],
            "last_cfg": last_cfg,
        }
