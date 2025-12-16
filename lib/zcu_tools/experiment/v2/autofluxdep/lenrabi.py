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
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.notebook.utils import make_comment
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

from .executor import FluxDepInfoDict, MeasurementTask, T_RootResultType


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
) -> Tuple[float, float, float, float, NDArray[np.float64]]:
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


class LenRabiCfgTemplate(TypedDict):
    reset: NotRequired[Union[ResetCfg, str]]
    rabi_pulse: Union[PulseCfg, str]
    readout: Union[ReadoutCfg, str]


class LenRabiCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    rabi_pulse: PulseCfg
    readout: ReadoutCfg

    sweep: Dict[str, SweepCfg]


class LenRabiResult(TypedDict, closed=True):
    raw_signals: NDArray[np.complex128]
    pi_length: NDArray[np.float64]
    pi2_length: NDArray[np.float64]
    rabi_freq: NDArray[np.float64]
    success: NDArray[np.bool_]


class PlotterDictType(TypedDict, closed=True):
    rabi_curve: LivePlotter2DwithLine


class LenRabiMeasurementTask(
    MeasurementTask[LenRabiResult, T_RootResultType, TaskConfig, PlotterDictType]
):
    def __init__(
        self,
        length_sweep: SweepCfg,
        ref_pi_product: float,
        cfg_maker: Callable[
            [
                TaskContextView[LenRabiResult, T_RootResultType, TaskConfig],
                ModuleLibrary,
            ],
            Optional[LenRabiCfgTemplate],
        ],
        earlystop_snr: Optional[float] = None,
    ) -> None:
        self.length_sweep = length_sweep
        self.ref_product = ref_pi_product
        self.cfg_maker = cfg_maker
        self.earlystop_snr = earlystop_snr

        def measure_fn(ctx: TaskContextView, update_hook: Callable):
            Pulse.set_param(
                ctx.cfg["rabi_pulse"],
                "length",
                sweep2param("length", self.length_sweep),
            )
            prog = ModularProgramV2(
                ctx.env_dict["soccfg"],
                ctx.cfg,
                modules=[
                    Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                    Pulse("rabi_pulse", ctx.cfg["rabi_pulse"]),
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
                    signal2real_fn=lenrabi_signal2real,
                ),
            )

        self.task = HardTask[
            np.complex128, T_RootResultType, LenRabiCfg, List[NDArray[np.float64]]
        ](measure_fn=measure_fn, result_shape=(self.length_sweep["expts"],))

    def num_axes(self) -> Dict[str, int]:
        return dict(rabi_curve=2)

    def make_plotter(self, name, axs) -> PlotterDictType:
        self.pi_line = axs["rabi_curve"][1].axvline(np.nan, color="red", linestyle="--")
        return PlotterDictType(
            rabi_curve=LivePlotter2DwithLine(
                "Flux device value",
                "Signal",
                line_axis=1,
                num_lines=3,
                title=name + "(rabi_curve)",
                existed_axes=[axs["rabi_curve"]],
            ),
        )

    def update_plotter(self, plotters, ctx, signals) -> None:
        flx_values = ctx.env_dict["flx_values"]

        self.pi_line.set_xdata([ctx.env_dict["info"].get("pi_length", np.nan)])
        plotters["rabi_curve"].update(
            flx_values,
            sweep2array(self.length_sweep),
            lenrabi_fluxdep_signal2real(signals["raw_signals"]),
            refresh=False,
        )

    def save(self, filepath, flx_values, result, comment, prefix_tag) -> None:
        filepath = Path(filepath)

        lengths = sweep2array(self.length_sweep)
        np.savez_compressed(filepath, flx_values=flx_values, lengths=lengths, **result)

        x_info = {"name": "Flux value", "unit": "a.u.", "values": flx_values}

        # signals
        save_data(
            filepath=str(filepath.with_name(filepath.name + "_signals")),
            x_info=x_info,
            y_info={"name": "Length", "unit": "s", "values": lengths * 1e-6},
            z_info={
                "name": "Signal",
                "unit": "a.u.",
                "values": result["raw_signals"].T,
            },
            comment=make_comment(self.init_cfg, comment),
            tag=prefix_tag + "/signals",
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

    def load(self, filepath: str, **kwargs) -> LenRabiResult:
        # main container (has pi2_length and rabi_freq)
        data = np.load(filepath)
        pi2_length = data["pi2_length"]
        rabi_freq = data["rabi_freq"]
        success_main = data["success"]

        # signals
        signals_stored, flx_values, lengths = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_signals")), **kwargs
        )
        assert flx_values is not None and lengths is not None
        assert signals_stored.shape == (len(lengths), len(flx_values))

        # pi_length
        pi_length_data, flx_pi, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_pi_length")), **kwargs
        )
        assert flx_pi is not None
        assert pi_length_data.shape == (len(flx_pi),)
        assert np.array_equal(flx_pi, flx_values)

        # success
        success_data, flx_succ, _ = load_data(
            str(Path(filepath).with_name(Path(filepath).name + "_success")), **kwargs
        )
        assert flx_succ is not None
        assert success_data.shape == (len(flx_succ),)
        assert np.array_equal(flx_succ, flx_values)

        assert (
            pi2_length.shape
            == rabi_freq.shape
            == success_main.shape
            == pi_length_data.shape
        )

        raw_signals = signals_stored.T.astype(np.complex128)
        pi_length = pi_length_data.astype(np.float64)
        pi2_length = pi2_length.astype(np.float64)
        rabi_freq = rabi_freq.astype(np.float64)
        success = success_data.astype(np.bool_) & success_main.astype(np.bool_)

        return LenRabiResult(
            raw_signals=raw_signals,
            pi_length=pi_length,
            pi2_length=pi2_length,
            rabi_freq=rabi_freq,
            success=success,
        )

    def init(self, ctx, dynamic_pbar=False) -> None:
        self.init_cfg = deepcopy(ctx.cfg)
        self.task.init(ctx(addr="raw_signals"), dynamic_pbar=dynamic_pbar)  # type: ignore

    def run(self, ctx) -> None:
        ml: ModuleLibrary = ctx.env_dict["ml"]

        cfg_temp = self.cfg_maker(ctx, ml)

        if cfg_temp is None:
            return  # skip this task

        cfg_temp = dict(cfg_temp)
        deepupdate(
            cfg_temp,
            {"dev": ctx.cfg.get("dev", {}), "sweep": {"length": self.length_sweep}},
        )
        cfg = cast(LenRabiCfg, ml.make_cfg(cfg_temp))

        rabi_pulse = cfg["rabi_pulse"]
        self.task.run(ctx(addr="raw_signals", new_cfg=cfg))  # type: ignore

        raw_signals = ctx.get_data()["raw_signals"]
        assert isinstance(raw_signals, np.ndarray)

        real_signals = lenrabi_signal2real(raw_signals)

        lengths = sweep2array(self.length_sweep)
        pi_len, pi2_len, rabi_freq, mean_err, fit_signals = auto_fit_lenrabi(
            lengths, real_signals
        )

        success = True
        if (
            pi2_len < 0.03
            or mean_err > 0.1 * np.ptp(fit_signals)
            or pi_len > 0.6 * np.max(lengths)
        ):
            pi_len, pi2_len, rabi_freq = np.nan, np.nan, np.nan
            success = False

        if success:
            info: FluxDepInfoDict = ctx.env_dict["info"]
            info["pi_product"] = pi_len * rabi_pulse["gain"]
            new_gain_factor = (
                info["m_ratio"] * info["pi_product"] / info.first["pi_product"]
            )

            info.update(
                pi_length=pi_len,
                pi2_length=pi2_len,
                gain_factor=np.sqrt(
                    info.last.get("gain_factor", 1.0) * new_gain_factor
                ),
            )
            info["pi_pulse"] = deepcopy(rabi_pulse)
            info["pi_pulse"]["waveform"]["length"] = pi_len
            info["pi2_pulse"] = deepcopy(rabi_pulse)
            info["pi2_pulse"]["waveform"]["length"] = pi2_len

        with MinIntervalFunc.force_execute():
            ctx.set_data(
                LenRabiResult(
                    raw_signals=raw_signals,
                    pi_length=np.array(pi_len),
                    pi2_length=np.array(pi2_len),
                    rabi_freq=np.array(rabi_freq),
                    success=np.array(success),
                )
            )

    def get_default_result(self) -> LenRabiResult:
        return LenRabiResult(
            raw_signals=self.task.get_default_result(),
            pi_length=np.array(np.nan),
            pi2_length=np.array(np.nan),
            rabi_freq=np.array(np.nan),
            success=np.array(False),
        )

    def cleanup(self) -> None:
        self.task.cleanup()
