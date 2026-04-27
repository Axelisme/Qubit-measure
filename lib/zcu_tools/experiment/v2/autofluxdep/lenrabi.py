from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, TypedDict

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState
from zcu_tools.experiment.v2.utils import sweep2array, wrap_earlystop_check
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.notebook.utils import make_sweep
from zcu_tools.program.v2 import (
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
from zcu_tools.utils import deepupdate
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.func_tools import MinIntervalFunc
from zcu_tools.utils.process import rotate2real

from .executor import FluxDepCfg, FluxDepInfoDict, MeasurementTask


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
) -> tuple[float, float, float, float, float, float, float, NDArray[np.float64]]:
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
    pi_len, pi_len_err, pi2_len, pi2_len_err, rabi_freq, rabi_freq_err = fit_params

    return (
        pi_len,
        pi_len_err,
        pi2_len,
        pi2_len_err,
        rabi_freq,
        rabi_freq_err,
        fit_loss,
        fit_signals,
    )


class LenRabiModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    rabi_pulse: PulseCfg
    readout: ReadoutCfg


class LenRabiCfgTemplate(ProgramV2Cfg, ExpCfgModel):
    modules: LenRabiModuleCfg
    sweep_range: tuple[float, float]


class LenRabiSweepCfg(ConfigBase):
    length: SweepCfg


class LenRabiCfg(ProgramV2Cfg, FluxDepCfg):
    modules: LenRabiModuleCfg
    sweep: LenRabiSweepCfg


class LenRabiResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    length: NDArray[np.float64]
    pi_length: NDArray[np.float64]
    pi2_length: NDArray[np.float64]
    rabi_freq: NDArray[np.float64]
    success: NDArray[np.bool_]


class LenRabiPlotDict(TypedDict, closed=True):
    rabi_curve: LivePlot2DwithLine


class LenRabiTask(MeasurementTask[LenRabiResult, Any, LenRabiPlotDict]):
    def __init__(
        self,
        num_expts: int,
        cfg_maker: Callable[
            [TaskState[LenRabiResult, Any, FluxDepCfg], ModuleLibrary],
            Optional[LenRabiCfgTemplate],
        ],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.num_expts = num_expts
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr
        self.last_cfg = None

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, LenRabiCfg],
            update_hook: Optional[Callable],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules
            len_sweep = cfg.sweep.length

            setup_devices(cfg, progress=False)

            assert update_hook is not None

            len_params = sweep2param("length", len_sweep)
            modules.rabi_pulse.set_param("length", len_params)

            prog = ModularProgramV2(
                ctx.env["soccfg"],
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("rabi_pulse", modules.rabi_pulse),
                    Readout("readout", modules.readout),
                ],
                sweep=[("length", len_sweep)],
            )
            return prog.acquire(
                ctx.env["soc"],
                progress=False,
                round_hook=wrap_earlystop_check(
                    prog,
                    update_hook,
                    snr_threshold=self.earlystop_snr,
                    signal2real_fn=lenrabi_signal2real,
                ),
            )

        self.lengths = np.linspace(0, 1, num_expts)
        self.task = Task[Any, list[NDArray[np.float64]], LenRabiCfg](
            measure_fn, result_shape=(num_expts,)
        )

    def init(self, dynamic_pbar: bool = False) -> None:
        self.task.init(dynamic_pbar=dynamic_pbar)

    def run(self, state: TaskState[LenRabiResult, Any, FluxDepCfg]) -> None:
        cfg_temp = self.cfg_maker(state, state.env["ml"])
        if cfg_temp is None:
            return  # skip this task

        len_sweep = make_sweep(*cfg_temp.sweep_range, self.num_expts)

        cfg = cfg_temp.to_dict()
        del cfg["sweep_range"]
        deepupdate(
            cfg,
            {"dev": state.cfg.dev, "sweep": {"length": len_sweep}},
            behavior="force",
        )
        cfg = LenRabiCfg.model_validate(cfg)
        self.last_cfg = cfg

        rabi_pulse = cfg.modules.rabi_pulse
        self.task.set_pbar_n(cfg.rounds)
        self.task.run(
            state.child_with_cfg("raw_signals", cfg, child_type=NDArray[np.complex128])
        )

        real_signals = lenrabi_signal2real(state.value["raw_signals"])

        self.lengths = sweep2array(
            len_sweep, "time", {"soccfg": state.env["soccfg"], "gen_ch": rabi_pulse.ch}
        )

        (pi_len, _, pi2_len, _, rabi_freq, _, mean_err, fit_signals) = auto_fit_lenrabi(
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
            info: FluxDepInfoDict = state.env["info"]

            cur_pi_product = pi_len * rabi_pulse.gain
            prev_pi_product = info.last.get("smooth_pi_product", cur_pi_product)
            num_step = max(
                1, info["flux_idx"] - info.last.get("lenrabi_success_idx", 0)
            )
            weight = 0.7**num_step
            smooth_pi_product = (1 - weight) * cur_pi_product + weight * prev_pi_product

            info["pi_length"] = pi_len
            info["pi2_length"] = pi2_len
            info["pi_pulse"] = deepcopy(rabi_pulse)
            info["pi_pulse"]["waveform"]["length"] = pi_len
            if pi2_len > 0.03:  # skip if pi2_len is too short to be reliable
                info["pi2_pulse"] = deepcopy(rabi_pulse)
                info["pi2_pulse"]["waveform"]["length"] = pi2_len
            info["smooth_pi_product"] = smooth_pi_product
            info["lenrabi_success_idx"] = info["flux_idx"]

        with MinIntervalFunc.force_execute():
            state.set_value(
                LenRabiResult(
                    raw_signals=state.value["raw_signals"],
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

    def make_plotter(self, name, axs) -> LenRabiPlotDict:
        self.pi_line = axs["rabi_curve"][1].axvline(np.nan, color="red", linestyle="--")
        return LenRabiPlotDict(
            rabi_curve=LivePlot2DwithLine(
                "Flux device value",
                "Signal",
                line_axis=1,
                num_lines=3,
                title=name + "(rabi_curve)",
                existed_axes=[axs["rabi_curve"]],
            ),
        )

    def update_plotter(self, plotters, ctx: TaskState, signals) -> None:
        flux_values = ctx.env["flux_values"]

        self.pi_line.set_xdata([ctx.env["info"].get("pi_length", np.nan)])
        plotters["rabi_curve"].update(
            flux_values,
            self.lengths,
            lenrabi_fluxdep_signal2real(signals["raw_signals"]),
            refresh=False,
        )

    def save(self, filepath, flux_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)
        cfg = self.last_cfg
        assert cfg is not None

        np.savez_compressed(filepath, flux_values=flux_values, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flux_values}
        comment = make_comment(cfg, comment)

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
            comment=comment,
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
            comment=comment,
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
            comment=comment,
            tag=prefix_tag + "/pi_length",
        )

        # success
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_success")),
            x_info=x_info,
            z_info={"name": "Success", "unit": "bool", "values": result["success"]},
            comment=comment,
            tag=prefix_tag + "/success",
        )

    @classmethod
    def load(cls, filepath: str, **kwargs) -> dict:
        _filepath = Path(filepath)

        # main container (has pi2_length and rabi_freq)
        data = np.load(filepath)

        flux_values = data["flux_values"]
        pi2_length = data["pi2_length"]
        rabi_freq = data["rabi_freq"]
        success_main = data["success"]

        # signals
        signal_path = str(_filepath.with_name(_filepath.name + "_signals"))
        signals_stored, flux_sig, len_idxs, comment = load_data(
            signal_path, return_comment=True, **kwargs
        )
        assert flux_sig is not None and len_idxs is not None
        assert np.array_equal(flux_values, flux_sig)
        assert signals_stored.shape == (len(len_idxs), len(flux_sig))

        length_path = str(_filepath.with_name(_filepath.name + "_length"))
        length_stored, flux_len, _ = load_data(length_path, **kwargs)
        assert flux_len is not None
        assert length_stored.shape == (len(flux_len), len(len_idxs))
        assert np.array_equal(flux_values, flux_len)

        # pi_length
        pi_length_path = str(_filepath.with_name(_filepath.name + "_pi_length"))
        pi_length_data, flux_pi, _ = load_data(pi_length_path, **kwargs)
        assert flux_pi is not None
        assert pi_length_data.shape == (len(flux_pi), len(len_idxs))
        assert np.array_equal(flux_values, flux_pi)

        # success
        success_path = str(_filepath.with_name(_filepath.name + "_success"))
        success_data, flux_succ, _ = load_data(success_path, **kwargs)
        assert flux_succ is not None
        assert success_data.shape == (len(flux_succ), len(len_idxs))
        assert np.array_equal(flux_values, flux_succ)

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
        last_cfg = None
        if comment is not None:
            last_cfg, _, _ = parse_comment(comment)

        return {
            "raw_signals": raw_signals,
            "length": length,
            "pi_length": pi_length,
            "pi2_length": pi2_length,
            "rabi_freq": rabi_freq,
            "success": success,
            "flux_values": flux_values,
            "lengths": len_idxs,
            "last_cfg": last_cfg,
        }
